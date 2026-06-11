"""MoE token routing helpers for eager reference and Ascend Triton paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only development and unit-test environments.
    triton = None
    tl = None


@dataclass
class MoeRoutingPlan:
    routed_states: torch.Tensor
    sorted_token_ids: torch.Tensor
    sorted_weights: torch.Tensor
    expert_counts: torch.Tensor
    group_list: torch.Tensor


def prepare_moe_routing_reference(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> MoeRoutingPlan:
    """Build a stable expert-major routing plan with ordinary PyTorch ops."""
    num_tokens, top_k = selected_experts.shape
    flat_experts = selected_experts.reshape(-1)
    flat_weights = routing_weights.reshape(-1)
    token_ids = torch.arange(
        num_tokens, device=hidden_states.device, dtype=torch.long
    ).repeat_interleave(top_k)
    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[order]
    sorted_token_ids = token_ids[order]
    sorted_weights = flat_weights[order]
    expert_counts = torch.bincount(
        sorted_experts, minlength=num_experts
    ).to(torch.int32)
    group_list = torch.cumsum(expert_counts, dim=0, dtype=torch.int64)
    return MoeRoutingPlan(
        routed_states=hidden_states[sorted_token_ids],
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        expert_counts=expert_counts,
        group_list=group_list,
    )


def finalize_moe_routing_reference(
    expert_output: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Apply routing weights and reduce expert assignments to source tokens."""
    output = torch.zeros(
        (num_tokens, expert_output.shape[-1]),
        device=expert_output.device,
        dtype=expert_output.dtype,
    )
    output.index_add_(
        0,
        sorted_token_ids,
        expert_output * sorted_weights.unsqueeze(-1),
    )
    return output


if triton is not None:
    @triton.jit
    def _moe_count_kernel(
        expert_ids_ptr,
        expert_counts_ptr,
        num_assignments,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_assignments
        expert_ids = tl.load(expert_ids_ptr + offsets, mask=mask, other=0)
        tl.atomic_add(expert_counts_ptr + expert_ids, 1, mask=mask)


    @triton.jit
    def _moe_count_and_gather_kernel(
        hidden_ptr,
        sorted_assignment_ids_ptr,
        routing_weights_ptr,
        routed_states_ptr,
        sorted_token_ids_ptr,
        sorted_weights_ptr,
        hidden_size: tl.constexpr,
        top_k: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        target = tl.program_id(0)
        assignment_id = tl.load(sorted_assignment_ids_ptr + target)
        token_id = assignment_id // top_k

        tl.store(sorted_token_ids_ptr + target, token_id)
        weight = tl.load(routing_weights_ptr + assignment_id)
        tl.store(sorted_weights_ptr + target, weight)

        hidden_offsets = tl.arange(0, BLOCK_H)
        hidden_mask = hidden_offsets < hidden_size
        values = tl.load(
            hidden_ptr + token_id * hidden_size + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        )
        tl.store(
            routed_states_ptr + target * hidden_size + hidden_offsets,
            values,
            mask=hidden_mask,
        )


    @triton.jit
    def _moe_weighted_scatter_kernel(
        expert_output_ptr,
        sorted_token_ids_ptr,
        sorted_weights_ptr,
        final_output_ptr,
        hidden_size: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        assignment_id = tl.program_id(0)
        token_id = tl.load(sorted_token_ids_ptr + assignment_id)
        weight = tl.load(sorted_weights_ptr + assignment_id).to(tl.float32)
        hidden_offsets = tl.arange(0, BLOCK_H)
        hidden_mask = hidden_offsets < hidden_size
        values = tl.load(
            expert_output_ptr + assignment_id * hidden_size + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        tl.atomic_add(
            final_output_ptr + token_id * hidden_size + hidden_offsets,
            values * weight,
            mask=hidden_mask,
        )
else:
    def _missing_triton(*args, **kwargs):
        raise RuntimeError(
            "Triton is required for the NPU MoE routing backend"
        )


    _moe_count_kernel = _missing_triton
    _moe_count_and_gather_kernel = _missing_triton
    _moe_weighted_scatter_kernel = _missing_triton


def prepare_moe_routing_npu(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> MoeRoutingPlan:
    """Count, expert-group and gather assignments without host synchronization."""
    if triton is None:
        raise RuntimeError(
            "Triton is required for the NPU MoE routing backend"
        )
    hidden_states = hidden_states.contiguous()
    expert_ids = selected_experts.contiguous().view(-1).to(torch.int32)
    flat_weights = routing_weights.contiguous().view(-1)
    num_assignments = expert_ids.numel()
    hidden_size = hidden_states.shape[-1]
    top_k = selected_experts.shape[-1]

    expert_counts = torch.zeros(
        num_experts, device=hidden_states.device, dtype=torch.int32
    )
    count_block = 256
    _moe_count_kernel[(triton.cdiv(num_assignments, count_block),)](
        expert_ids,
        expert_counts,
        num_assignments=num_assignments,
        BLOCK_SIZE=count_block,
    )
    group_list = torch.cumsum(expert_counts, dim=0, dtype=torch.int64)
    # Ascend Triton 3.2 cannot consume the old value returned by
    # tl.atomic_add. Keep sorting on the NPU with torch.argsort, then let
    # Triton fuse the ordered gather and metadata writes.
    sorted_assignment_ids = torch.argsort(expert_ids).to(torch.int32)
    routed_states = torch.empty(
        (num_assignments, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    sorted_token_ids = torch.empty(
        num_assignments, device=hidden_states.device, dtype=torch.int64
    )
    sorted_weights = torch.empty(
        num_assignments,
        device=hidden_states.device,
        dtype=routing_weights.dtype,
    )
    block_h = triton.next_power_of_2(hidden_size)
    _moe_count_and_gather_kernel[(num_assignments,)](
        hidden_states,
        sorted_assignment_ids,
        flat_weights,
        routed_states,
        sorted_token_ids,
        sorted_weights,
        hidden_size=hidden_size,
        top_k=top_k,
        BLOCK_H=block_h,
    )
    return MoeRoutingPlan(
        routed_states=routed_states,
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        expert_counts=expert_counts,
        group_list=group_list,
    )


def finalize_moe_routing_npu(
    expert_output: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Fuse routing-weight multiplication and token scatter/reduction."""
    if triton is None:
        raise RuntimeError(
            "Triton is required for the NPU MoE routing backend"
        )
    num_assignments, hidden_size = expert_output.shape
    accumulation = torch.zeros(
        (num_tokens, hidden_size),
        device=expert_output.device,
        dtype=torch.float32,
    )
    block_h = triton.next_power_of_2(hidden_size)
    _moe_weighted_scatter_kernel[(num_assignments,)](
        expert_output.contiguous(),
        sorted_token_ids,
        sorted_weights,
        accumulation,
        hidden_size=hidden_size,
        BLOCK_H=block_h,
    )
    return accumulation.to(expert_output.dtype)


def prepare_moe_routing(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> MoeRoutingPlan:
    if hidden_states.device.type == "npu":
        return prepare_moe_routing_npu(
            hidden_states,
            selected_experts,
            routing_weights,
            num_experts,
        )
    return prepare_moe_routing_reference(
        hidden_states,
        selected_experts,
        routing_weights,
        num_experts,
    )


def finalize_moe_routing(
    expert_output: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    if expert_output.device.type == "npu":
        return finalize_moe_routing_npu(
            expert_output,
            sorted_token_ids,
            sorted_weights,
            num_tokens,
        )
    return finalize_moe_routing_reference(
        expert_output,
        sorted_token_ids,
        sorted_weights,
        num_tokens,
    )
