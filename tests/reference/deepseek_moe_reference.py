"""Independent CPU reference for DeepSeek-V2/V3 MoE inference semantics.

The implementation is deliberately separate from production routing,
placement, expert execution, communication, and optimized kernels.  It uses
flat CPU tensors and FP32 arithmetic throughout.  The supported routed-expert
runtime layout is:

* hidden states: ``[tokens, hidden]``
* router weight: ``[experts, hidden]``
* routed gate/up: ``[local_experts, hidden, 2 * intermediate]``
* routed down: ``[local_experts, intermediate, hidden]``
* shared gate/up: ``[hidden, 2 * shared_intermediate]``
* shared down: ``[shared_intermediate, hidden]``

The gate half precedes the up half in every gate/up tensor.  Expert IDs are
global int64 IDs.  Non-finite inputs, DeepSeek-V4 routing, collectives, MLA,
quantization, and training/backward behavior are outside this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DeepSeekRoutingSpec:
    """Validated V2/V3 grouped-routing configuration."""

    num_experts: int
    experts_per_token: int
    num_groups: int = 1
    topk_groups: int = 1
    score_func: str = "softmax"
    topk_method: str = "group_limited_greedy"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0

    def __post_init__(self) -> None:
        integer_fields = (
            ("num_experts", self.num_experts),
            ("experts_per_token", self.experts_per_token),
            ("num_groups", self.num_groups),
            ("topk_groups", self.topk_groups),
        )
        for name, value in integer_fields:
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if self.num_experts % self.num_groups:
            raise ValueError("num_experts must be divisible by num_groups")
        if self.topk_groups > self.num_groups:
            raise ValueError("topk_groups must not exceed num_groups")

        experts_per_group = self.num_experts // self.num_groups
        selected_capacity = self.topk_groups * experts_per_group
        if self.experts_per_token > selected_capacity:
            raise ValueError(
                "experts_per_token exceeds the selected-group capacity"
            )

        if self.score_func not in ("softmax", "sigmoid"):
            raise ValueError(
                "score_func must be 'softmax' or 'sigmoid'"
            )
        if self.topk_method not in (
            "group_limited_greedy",
            "noaux_tc",
        ):
            raise ValueError(
                "topk_method must be 'group_limited_greedy' or 'noaux_tc'"
            )
        if (
            self.topk_method == "group_limited_greedy"
            and self.score_func != "softmax"
        ):
            raise ValueError(
                "group_limited_greedy requires score_func='softmax'"
            )
        if self.topk_method == "noaux_tc":
            if self.score_func != "sigmoid":
                raise ValueError("noaux_tc requires score_func='sigmoid'")
            if experts_per_group < 2:
                raise ValueError("noaux_tc requires at least two experts per group")

        if type(self.norm_topk_prob) is not bool:
            raise TypeError("norm_topk_prob must be bool")
        if (
            isinstance(self.routed_scaling_factor, bool)
            or not isinstance(self.routed_scaling_factor, (int, float))
            or not math.isfinite(float(self.routed_scaling_factor))
            or self.routed_scaling_factor <= 0
        ):
            raise ValueError("routed_scaling_factor must be finite and positive")

    @property
    def experts_per_group(self) -> int:
        return self.num_experts // self.num_groups


def _require_cpu_floating(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _require_hidden(name: str, tensor: torch.Tensor) -> None:
    _require_cpu_floating(name, tensor)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [tokens, hidden]")


@torch.no_grad()
def deepseek_route_reference(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    spec: DeepSeekRoutingSpec,
    *,
    correction_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FP32 logits, final combine weights, and global int64 IDs.

    Correction bias affects expert and group selection only.  Final combine
    weights are gathered from the original softmax or sigmoid scores.
    Tie ordering is intentionally unspecified.
    """

    if not isinstance(spec, DeepSeekRoutingSpec):
        raise TypeError("spec must be a DeepSeekRoutingSpec")
    _require_hidden("hidden_states", hidden_states)
    _require_cpu_floating("router_weight", router_weight)
    if router_weight.ndim != 2:
        raise ValueError("router_weight must have shape [experts, hidden]")
    if router_weight.shape != (spec.num_experts, hidden_states.shape[1]):
        raise ValueError(
            "router_weight must have shape [num_experts, hidden]"
        )

    if spec.topk_method == "noaux_tc":
        if correction_bias is None:
            raise ValueError("noaux_tc requires correction_bias")
    elif correction_bias is not None:
        raise ValueError(
            "group_limited_greedy does not accept correction_bias"
        )

    if correction_bias is not None:
        _require_cpu_floating("correction_bias", correction_bias)
        if correction_bias.shape != (spec.num_experts,):
            raise ValueError(
                "correction_bias must have shape [num_experts]"
            )

    hidden_fp32 = hidden_states.float()
    weight_fp32 = router_weight.float()
    router_logits = hidden_fp32 @ weight_fp32.t()
    if spec.score_func == "softmax":
        original_scores = F.softmax(router_logits, dim=-1)
    else:
        original_scores = torch.sigmoid(router_logits)

    selection_scores = original_scores
    if correction_bias is not None:
        selection_scores = selection_scores + correction_bias.float()
        if not torch.isfinite(selection_scores).all().item():
            raise ValueError("selection scores must remain finite")

    grouped_scores = selection_scores.reshape(
        hidden_states.shape[0],
        spec.num_groups,
        spec.experts_per_group,
    )
    if spec.topk_method == "group_limited_greedy":
        group_scores = grouped_scores.max(dim=-1).values
    else:
        group_scores = grouped_scores.topk(2, dim=-1).values.sum(dim=-1)

    selected_groups = group_scores.topk(spec.topk_groups, dim=-1).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = group_mask.unsqueeze(-1).expand_as(grouped_scores).reshape(
        hidden_states.shape[0],
        spec.num_experts,
    )
    masked_selection_scores = selection_scores.masked_fill(
        ~expert_mask,
        -torch.inf,
    )
    selected_experts = masked_selection_scores.topk(
        spec.experts_per_token,
        dim=-1,
    ).indices.to(torch.int64)

    routing_weights = original_scores.gather(1, selected_experts)
    if spec.norm_topk_prob:
        normalization_sum = routing_weights.sum(
            dim=-1,
            keepdim=True,
        )
        if (
            not torch.isfinite(normalization_sum).all().item()
            or not (normalization_sum > 0).all().item()
        ):
            raise ValueError(
                "routing weight normalization sum must be positive and finite"
            )
        routing_weights = routing_weights / normalization_sum
    routing_weights = routing_weights * float(spec.routed_scaling_factor)
    if not torch.isfinite(routing_weights).all().item():
        raise ValueError("routing weights must remain finite after scaling")
    return router_logits, routing_weights, selected_experts


@torch.no_grad()
def deepseek_routed_experts_reference(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    expert_start: int = 0,
) -> torch.Tensor:
    """Evaluate one contiguous expert slice with explicit token/slot loops.

    Global expert IDs outside this slice are ignored.  This models one local
    contribution for the project's replicated-token EP contract.  It does not
    model communication or all-to-all token dispatch.
    """

    _require_hidden("hidden_states", hidden_states)
    _require_cpu_floating("routing_weights", routing_weights)
    _require_cpu_floating("gate_up_weight", gate_up_weight)
    _require_cpu_floating("down_weight", down_weight)
    if selected_experts.device.type != "cpu":
        raise ValueError("selected_experts must be a CPU tensor")
    if selected_experts.dtype != torch.int64:
        raise TypeError("selected_experts must use int64 global IDs")
    if selected_experts.ndim != 2:
        raise ValueError("selected_experts must have shape [tokens, top_k]")
    if routing_weights.shape != selected_experts.shape:
        raise ValueError("routing_weights must match selected_experts shape")
    if selected_experts.shape[0] != hidden_states.shape[0]:
        raise ValueError("routing tensors must match the token count")
    if selected_experts.numel() and selected_experts.min().item() < 0:
        raise ValueError("selected_experts must contain non-negative IDs")
    if type(expert_start) is not int or expert_start < 0:
        raise ValueError("expert_start must be a non-negative integer")
    if gate_up_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError("routed expert weights must be rank-3 tensors")

    local_experts, hidden_size, doubled_intermediate = gate_up_weight.shape
    if local_experts <= 0:
        raise ValueError("gate_up_weight must contain at least one expert")
    if doubled_intermediate <= 0 or doubled_intermediate % 2:
        raise ValueError("gate_up_weight final dimension must be positive and even")
    intermediate_size = doubled_intermediate // 2
    if hidden_states.shape[1] != hidden_size:
        raise ValueError("hidden size does not match gate_up_weight")
    if down_weight.shape != (
        local_experts,
        intermediate_size,
        hidden_size,
    ):
        raise ValueError(
            "down_weight must have shape [local_experts, intermediate, hidden]"
        )

    hidden_fp32 = hidden_states.float()
    routing_fp32 = routing_weights.float()
    gate_up_fp32 = gate_up_weight.float()
    down_fp32 = down_weight.float()
    output = torch.zeros(
        hidden_states.shape[0],
        hidden_size,
        dtype=torch.float32,
    )
    expert_end = expert_start + local_experts

    for token_id in range(hidden_states.shape[0]):
        for slot_id in range(selected_experts.shape[1]):
            global_expert_id = int(
                selected_experts[token_id, slot_id].item()
            )
            if not expert_start <= global_expert_id < expert_end:
                continue
            local_expert_id = global_expert_id - expert_start
            gate_up = hidden_fp32[token_id] @ gate_up_fp32[local_expert_id]
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
            expert_output = activated @ down_fp32[local_expert_id]
            output[token_id] += (
                expert_output * routing_fp32[token_id, slot_id]
            )
    return output


@torch.no_grad()
def deepseek_shared_expert_reference(
    hidden_states: torch.Tensor,
    gate_up_weight: Optional[torch.Tensor] = None,
    down_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Evaluate one combined shared SwiGLU expert, or zero shared experts."""

    _require_hidden("hidden_states", hidden_states)
    if gate_up_weight is None and down_weight is None:
        return torch.zeros(
            hidden_states.shape,
            dtype=torch.float32,
        )
    if gate_up_weight is None or down_weight is None:
        raise ValueError("shared gate_up_weight and down_weight must be paired")

    _require_cpu_floating("shared gate_up_weight", gate_up_weight)
    _require_cpu_floating("shared down_weight", down_weight)
    if gate_up_weight.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("shared expert weights must be rank-2 tensors")
    hidden_size, doubled_intermediate = gate_up_weight.shape
    if doubled_intermediate <= 0 or doubled_intermediate % 2:
        raise ValueError(
            "shared gate_up_weight final dimension must be positive and even"
        )
    shared_intermediate = doubled_intermediate // 2
    if hidden_states.shape[1] != hidden_size:
        raise ValueError("hidden size does not match shared gate_up_weight")
    if down_weight.shape != (shared_intermediate, hidden_size):
        raise ValueError(
            "shared down_weight must have shape [shared_intermediate, hidden]"
        )

    gate_up = hidden_states.float() @ gate_up_weight.float()
    gate, up = gate_up.chunk(2, dim=-1)
    return (F.silu(gate) * up) @ down_weight.float()


@torch.no_grad()
def deepseek_moe_reference(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    routed_gate_up_weight: torch.Tensor,
    routed_down_weight: torch.Tensor,
    spec: DeepSeekRoutingSpec,
    *,
    correction_bias: Optional[torch.Tensor] = None,
    shared_gate_up_weight: Optional[torch.Tensor] = None,
    shared_down_weight: Optional[torch.Tensor] = None,
    expert_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run independent V2/V3 routing, routed experts, and shared expert."""

    router_logits, routing_weights, selected_experts = (
        deepseek_route_reference(
            hidden_states,
            router_weight,
            spec,
            correction_bias=correction_bias,
        )
    )
    routed_output = deepseek_routed_experts_reference(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_gate_up_weight,
        routed_down_weight,
        expert_start=expert_start,
    )
    shared_output = deepseek_shared_expert_reference(
        hidden_states,
        shared_gate_up_weight,
        shared_down_weight,
    )
    return (
        routed_output + shared_output,
        router_logits,
        routing_weights,
        selected_experts,
    )
