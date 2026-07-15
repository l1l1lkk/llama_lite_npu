"""Independent, CPU-only PyTorch reference for Qwen3-style MoE inference.

This module intentionally does not import the production MoE router, expert
executor, routing helpers, or optimized kernels.  It accepts the runtime GMM
weight layout directly:

* router: ``[num_experts, hidden_size]``
* gate/up: ``[local_num_experts, hidden_size, 2 * intermediate_size]``
* down: ``[local_num_experts, intermediate_size, hidden_size]``

Expert arithmetic and token accumulation use FP32.  Router logits retain the
input dtype, while softmax is evaluated in FP32 and selected weights are cast
back to the logits dtype to characterize the current production contract.
The functions are inference-only and operate on flat ``[tokens, hidden]`` CPU
tensors.  Non-finite inputs are outside this reference contract.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _require_cpu_floating(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


@torch.no_grad()
def route_topk_reference(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    top_k: int,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logits, selected weights, and global expert IDs.

    ``hidden_states`` must be flat ``[tokens, hidden]`` and ``router_weight``
    must be ``[experts, hidden]``.  Valid ``top_k`` values are ``1..experts``.
    The selected expert order for tied scores is intentionally unspecified.
    """

    _require_cpu_floating("hidden_states", hidden_states)
    _require_cpu_floating("router_weight", router_weight)
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden]")
    if router_weight.ndim != 2:
        raise ValueError("router_weight must have shape [experts, hidden]")
    if hidden_states.shape[1] != router_weight.shape[1]:
        raise ValueError("hidden size does not match router weight")
    if hidden_states.dtype != router_weight.dtype:
        raise TypeError("hidden_states and router_weight must have the same dtype")

    num_experts = router_weight.shape[0]
    if not 1 <= top_k <= num_experts:
        raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")

    router_logits = F.linear(hidden_states, router_weight)
    router_probabilities = F.softmax(router_logits.float(), dim=-1)
    routing_weights, selected_experts = torch.topk(
        router_probabilities,
        top_k,
        dim=-1,
    )
    if norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1,
            keepdim=True,
        )
    return (
        router_logits,
        routing_weights.to(router_logits.dtype),
        selected_experts,
    )


@torch.no_grad()
def expert_forward_reference(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    expert_start: int = 0,
    local_num_experts: Optional[int] = None,
) -> torch.Tensor:
    """Evaluate local expert contributions with explicit token/slot loops.

    Expert IDs are global.  IDs outside ``[expert_start, expert_end)`` are
    ignored, which models one EP rank's local contribution without modeling a
    collective.  The returned tensor is FP32 with shape ``[tokens, hidden]``.
    """

    for name, tensor in (
        ("hidden_states", hidden_states),
        ("routing_weights", routing_weights),
        ("gate_up_weight", gate_up_weight),
        ("down_weight", down_weight),
    ):
        _require_cpu_floating(name, tensor)
    if selected_experts.device.type != "cpu":
        raise ValueError("selected_experts must be a CPU tensor")
    if selected_experts.dtype not in (torch.int32, torch.int64):
        raise TypeError("selected_experts must use int32 or int64")
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden]")
    if selected_experts.ndim != 2:
        raise ValueError("selected_experts must have shape [tokens, top_k]")
    if routing_weights.shape != selected_experts.shape:
        raise ValueError("routing_weights must match selected_experts shape")
    if selected_experts.shape[0] != hidden_states.shape[0]:
        raise ValueError("routing tensors must match the token count")
    if gate_up_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError("expert weights must be rank-3 runtime tensors")

    weight_experts, hidden_size, doubled_intermediate = gate_up_weight.shape
    if doubled_intermediate % 2:
        raise ValueError("gate_up output size must be even")
    intermediate_size = doubled_intermediate // 2
    if hidden_states.shape[1] != hidden_size:
        raise ValueError("hidden size does not match gate_up weight")
    if down_weight.shape != (weight_experts, intermediate_size, hidden_size):
        raise ValueError("down weight must have shape [experts, intermediate, hidden]")

    if local_num_experts is None:
        local_num_experts = weight_experts
    if local_num_experts != weight_experts:
        raise ValueError("local_num_experts must match the local weight count")
    if expert_start < 0:
        raise ValueError("expert_start must be non-negative")
    expert_end = expert_start + local_num_experts

    hidden_fp32 = hidden_states.float()
    routing_fp32 = routing_weights.float()
    gate_up_fp32 = gate_up_weight.float()
    down_fp32 = down_weight.float()
    output = torch.zeros(
        hidden_states.shape[0],
        hidden_size,
        dtype=torch.float32,
    )

    for token_id in range(hidden_states.shape[0]):
        for slot in range(selected_experts.shape[1]):
            global_expert_id = int(selected_experts[token_id, slot].item())
            if not expert_start <= global_expert_id < expert_end:
                continue
            local_expert_id = global_expert_id - expert_start
            gate_up = hidden_fp32[token_id] @ gate_up_fp32[local_expert_id]
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
            expert_output = activated @ down_fp32[local_expert_id]
            output[token_id] += (
                expert_output * routing_fp32[token_id, slot]
            )
    return output


@torch.no_grad()
def moe_forward_reference(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    top_k: int,
    norm_topk_prob: bool,
    expert_start: int = 0,
    local_num_experts: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run independent routing and local expert execution.

    Returns ``(output_fp32, router_logits, routing_weights, expert_ids)``.
    Communication and cross-rank reduction are deliberately out of scope.
    """

    router_logits, routing_weights, selected_experts = route_topk_reference(
        hidden_states,
        router_weight,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
    )
    output = expert_forward_reference(
        hidden_states,
        selected_experts,
        routing_weights,
        gate_up_weight,
        down_weight,
        expert_start=expert_start,
        local_num_experts=local_num_experts,
    )
    return output, router_logits, routing_weights, selected_experts
