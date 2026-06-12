"""Small-batch routed expert kernels.

Unlike the GMM path, this backend keeps assignments in token/top-k order and
therefore avoids expert sorting, gather and atomic scatter. It is intended for
decode batches where the routing overhead is larger than the benefit of
grouping one-token expert matrices.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def routed_expert_matmul_reference(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    expert_start: int = 0,
    local_num_experts: int | None = None,
) -> torch.Tensor:
    """Vectorized PyTorch reference for token-major routed experts."""
    num_tokens, top_k = selected_experts.shape
    hidden_size = hidden_states.shape[-1]
    if local_num_experts is None:
        local_num_experts = gate_up_weight.shape[0]
    flat_global_experts = selected_experts.reshape(-1).long()
    local_mask = (
        (flat_global_experts >= expert_start)
        & (flat_global_experts < expert_start + local_num_experts)
    )
    flat_experts = flat_global_experts[local_mask] - expert_start
    assignment_states = (
        hidden_states[:, None, :]
        .expand(num_tokens, top_k, hidden_size)
        .reshape(-1, hidden_size)
    )[local_mask]
    assignment_output = hidden_states.new_zeros(
        (num_tokens * top_k, hidden_size)
    )
    if flat_experts.numel() == 0:
        return assignment_output.reshape(
            num_tokens, top_k, hidden_size
        ).sum(dim=1)
    selected_gate_up = gate_up_weight.index_select(0, flat_experts)
    gate_up = torch.bmm(
        assignment_states.unsqueeze(1), selected_gate_up
    ).squeeze(1)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    selected_down = down_weight.index_select(0, flat_experts)
    expert_output = torch.bmm(
        activated.unsqueeze(1), selected_down
    ).squeeze(1)
    weighted = expert_output * routing_weights.reshape(-1, 1)[local_mask]
    assignment_output[local_mask] = weighted
    return assignment_output.reshape(
        num_tokens, top_k, hidden_size
    ).sum(dim=1)


if triton is not None:
    @triton.jit
    def _routed_swiglu_gemv_kernel(
        hidden_ptr,
        assignment_ids_ptr,
        expert_ids_ptr,
        gate_up_ptr,
        activated_ptr,
        hidden_size: tl.constexpr,
        intermediate_size: tl.constexpr,
        top_k: tl.constexpr,
        stride_expert: tl.constexpr,
        stride_hidden: tl.constexpr,
        expert_start: tl.constexpr,
        local_num_experts: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        local_assignment_id = tl.program_id(0)
        assignment_id = tl.load(
            assignment_ids_ptr + local_assignment_id
        )
        block_id = tl.program_id(1)
        token_id = assignment_id // top_k
        global_expert_id = tl.load(
            expert_ids_ptr + assignment_id
        ).to(tl.int64)
        expert_id = global_expert_id - expert_start
        valid_expert = (
            (expert_id >= 0) & (expert_id < local_num_experts)
        )
        safe_expert_id = tl.where(valid_expert, expert_id, 0)
        n_offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < intermediate_size
        gate_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(0, hidden_size, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < hidden_size
            values = tl.load(
                hidden_ptr + token_id * hidden_size + k_offsets,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            weight_base = (
                gate_up_ptr
                + safe_expert_id * stride_expert
                + k_offsets[:, None] * stride_hidden
            )
            gate_weight = tl.load(
                weight_base + n_offsets[None, :],
                mask=(
                    valid_expert
                    & k_mask[:, None]
                    & n_mask[None, :]
                ),
                other=0.0,
            ).to(tl.float32)
            up_weight = tl.load(
                weight_base
                + intermediate_size
                + n_offsets[None, :],
                mask=(
                    valid_expert
                    & k_mask[:, None]
                    & n_mask[None, :]
                ),
                other=0.0,
            ).to(tl.float32)
            gate_acc += tl.sum(values[:, None] * gate_weight, axis=0)
            up_acc += tl.sum(values[:, None] * up_weight, axis=0)

        activated = gate_acc * tl.sigmoid(gate_acc) * up_acc
        tl.store(
            activated_ptr
            + local_assignment_id * intermediate_size
            + n_offsets,
            activated,
            mask=n_mask,
        )


    @triton.jit
    def _routed_down_gemv_kernel(
        activated_ptr,
        assignment_ids_ptr,
        expert_ids_ptr,
        down_ptr,
        expert_output_ptr,
        hidden_size: tl.constexpr,
        intermediate_size: tl.constexpr,
        stride_expert: tl.constexpr,
        stride_intermediate: tl.constexpr,
        expert_start: tl.constexpr,
        local_num_experts: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        local_assignment_id = tl.program_id(0)
        assignment_id = tl.load(
            assignment_ids_ptr + local_assignment_id
        )
        block_id = tl.program_id(1)
        global_expert_id = tl.load(
            expert_ids_ptr + assignment_id
        ).to(tl.int64)
        expert_id = global_expert_id - expert_start
        valid_expert = (
            (expert_id >= 0) & (expert_id < local_num_experts)
        )
        safe_expert_id = tl.where(valid_expert, expert_id, 0)
        n_offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < hidden_size
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(0, intermediate_size, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < intermediate_size
            values = tl.load(
                activated_ptr
                + local_assignment_id * intermediate_size
                + k_offsets,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            weights = tl.load(
                down_ptr
                + safe_expert_id * stride_expert
                + k_offsets[:, None] * stride_intermediate
                + n_offsets[None, :],
                mask=(
                    valid_expert
                    & k_mask[:, None]
                    & n_mask[None, :]
                ),
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(values[:, None] * weights, axis=0)

        tl.store(
            expert_output_ptr
            + assignment_id * hidden_size
            + n_offsets,
            accumulator,
            mask=n_mask,
        )


    @triton.jit
    def _routed_topk_reduce_kernel(
        expert_output_ptr,
        routing_weights_ptr,
        output_ptr,
        hidden_size: tl.constexpr,
        top_k: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        block_id = tl.program_id(1)
        n_offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < hidden_size
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for slot in range(0, top_k):
            assignment_id = token_id * top_k + slot
            weight = tl.load(
                routing_weights_ptr + assignment_id
            ).to(tl.float32)
            values = tl.load(
                expert_output_ptr
                + assignment_id * hidden_size
                + n_offsets,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += values * weight
        tl.store(
            output_ptr + token_id * hidden_size + n_offsets,
            accumulator,
            mask=n_mask,
        )


def routed_expert_matmul_npu(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    expert_start: int = 0,
    local_num_experts: int | None = None,
    filter_local_experts: bool = False,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for routed_gemv on NPU")
    hidden_states = hidden_states.contiguous()
    expert_ids = selected_experts.contiguous().reshape(-1).to(torch.int32)
    routing_weights = routing_weights.contiguous()
    gate_up_weight = gate_up_weight.contiguous()
    down_weight = down_weight.contiguous()
    num_tokens, top_k = selected_experts.shape
    total_assignments = num_tokens * top_k
    hidden_size = hidden_states.shape[-1]
    intermediate_size = down_weight.shape[1]
    if local_num_experts is None:
        local_num_experts = gate_up_weight.shape[0]
    if filter_local_experts:
        local_mask = (
            (expert_ids >= expert_start)
            & (expert_ids < expert_start + local_num_experts)
        )
        assignment_ids = torch.nonzero(
            local_mask, as_tuple=False
        ).reshape(-1).to(torch.int32)
    else:
        assignment_ids = torch.arange(
            total_assignments,
            device=hidden_states.device,
            dtype=torch.int32,
        )
    num_assignments = assignment_ids.numel()
    if num_assignments == 0:
        return torch.zeros_like(hidden_states)
    activated = torch.empty(
        (num_assignments, intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    expert_output = torch.empty(
        (total_assignments, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    expert_output.zero_()
    output = torch.empty_like(hidden_states)
    block_k = 64
    block_n = 64

    _routed_swiglu_gemv_kernel[
        (num_assignments, triton.cdiv(intermediate_size, block_n))
    ](
        hidden_states,
        assignment_ids,
        expert_ids,
        gate_up_weight,
        activated,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        stride_expert=gate_up_weight.stride(0),
        stride_hidden=gate_up_weight.stride(1),
        expert_start=expert_start,
        local_num_experts=local_num_experts,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
    )
    _routed_down_gemv_kernel[
        (num_assignments, triton.cdiv(hidden_size, block_n))
    ](
        activated,
        assignment_ids,
        expert_ids,
        down_weight,
        expert_output,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        stride_expert=down_weight.stride(0),
        stride_intermediate=down_weight.stride(1),
        expert_start=expert_start,
        local_num_experts=local_num_experts,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
    )
    _routed_topk_reduce_kernel[
        (num_tokens, triton.cdiv(hidden_size, block_n))
    ](
        expert_output,
        routing_weights,
        output,
        hidden_size=hidden_size,
        top_k=top_k,
        BLOCK_N=block_n,
    )
    return output


def routed_expert_matmul(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    expert_start: int = 0,
    local_num_experts: int | None = None,
    filter_local_experts: bool = False,
) -> torch.Tensor:
    if hidden_states.device.type == "npu":
        return routed_expert_matmul_npu(
            hidden_states,
            selected_experts,
            routing_weights,
            gate_up_weight,
            down_weight,
            expert_start=expert_start,
            local_num_experts=local_num_experts,
            filter_local_experts=filter_local_experts,
        )
    return routed_expert_matmul_reference(
        hidden_states,
        selected_experts,
        routing_weights,
        gate_up_weight,
        down_weight,
        expert_start=expert_start,
        local_num_experts=local_num_experts,
    )
