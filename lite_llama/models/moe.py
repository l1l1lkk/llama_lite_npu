"""Correctness-first MoE routing and expert execution."""

from __future__ import annotations

import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingResult(NamedTuple):
    """Tuple-compatible output of a token-to-expert router."""

    router_logits: torch.Tensor
    routing_weights: torch.Tensor
    selected_experts: torch.Tensor


@dataclass(frozen=True)
class GroupedTopKConfig:
    """Immutable DeepSeek-V2/V3 grouped-routing policy."""

    num_groups: int
    topk_groups: int
    score_func: str
    topk_method: str
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("num_groups", self.num_groups),
            ("topk_groups", self.topk_groups),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.topk_groups > self.num_groups:
            raise ValueError("topk_groups must not exceed num_groups")
        if self.score_func not in {"softmax", "sigmoid"}:
            raise ValueError("score_func must be 'softmax' or 'sigmoid'")
        if self.topk_method not in {"group_limited_greedy", "noaux_tc"}:
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
        if self.topk_method == "noaux_tc" and self.score_func != "sigmoid":
            raise ValueError("noaux_tc requires score_func='sigmoid'")
        if type(self.norm_topk_prob) is not bool:
            raise TypeError("norm_topk_prob must be bool")
        if (
            isinstance(self.routed_scaling_factor, bool)
            or not isinstance(self.routed_scaling_factor, (int, float))
            or not math.isfinite(float(self.routed_scaling_factor))
            or self.routed_scaling_factor <= 0
        ):
            raise ValueError("routed_scaling_factor must be finite and positive")


class ExpertPlacement(NamedTuple):
    """Immutable metadata describing one rank's routed-expert weights."""

    parallel_mode: str
    world_size: int
    rank: int
    local_num_experts: int
    expert_start: int
    expert_end: int
    local_intermediate_size: int

    @classmethod
    def from_config(
        cls,
        *,
        num_experts: int,
        intermediate_size: int,
        tp_config=None,
    ) -> "ExpertPlacement":
        world_size = getattr(tp_config, "world_size", 1)
        rank = getattr(tp_config, "rank", 0)
        parallel_mode = getattr(tp_config, "moe_parallel_mode", "tp")
        if parallel_mode not in {"tp", "ep"}:
            raise ValueError(
                "moe_parallel_mode must be 'tp' or 'ep', got "
                f"{parallel_mode!r}"
            )
        if parallel_mode == "tp" and intermediate_size % world_size != 0:
            raise ValueError(
                f"moe_intermediate_size={intermediate_size} must be divisible "
                f"by tensor parallel world_size={world_size}"
            )
        if parallel_mode == "ep" and num_experts % world_size != 0:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by "
                f"expert parallel world_size={world_size}"
            )
        if parallel_mode == "ep":
            local_num_experts = num_experts // world_size
            expert_start = rank * local_num_experts
            local_intermediate_size = intermediate_size
        else:
            local_num_experts = num_experts
            expert_start = 0
            local_intermediate_size = intermediate_size // world_size
        return cls(
            parallel_mode=parallel_mode,
            world_size=world_size,
            rank=rank,
            local_num_experts=local_num_experts,
            expert_start=expert_start,
            expert_end=expert_start + local_num_experts,
            local_intermediate_size=local_intermediate_size,
        )


class SharedExpertPlacement(NamedTuple):
    """Immutable ownership metadata for one combined shared expert."""

    parallel_mode: str
    world_size: int
    rank: int
    shared_intermediate_size: int
    local_intermediate_size: int
    intermediate_start: int
    intermediate_end: int
    reduce_output: bool

    @classmethod
    def from_config(
        cls,
        *,
        shared_intermediate_size: int,
        tp_config=None,
    ) -> "SharedExpertPlacement":
        world_size = getattr(tp_config, "world_size", 1)
        rank = getattr(tp_config, "rank", 0)
        parallel_mode = getattr(tp_config, "moe_parallel_mode", "tp")
        if parallel_mode not in {"tp", "ep"}:
            raise ValueError(
                "moe_parallel_mode must be 'tp' or 'ep', got "
                f"{parallel_mode!r}"
            )
        if type(world_size) is not int or world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("rank must be an integer in [0, world_size)")
        if (
            type(shared_intermediate_size) is not int
            or shared_intermediate_size <= 0
        ):
            raise ValueError(
                "shared_intermediate_size must be a positive integer"
            )
        if (
            parallel_mode == "tp"
            and shared_intermediate_size % world_size != 0
        ):
            raise ValueError(
                "shared_intermediate_size="
                f"{shared_intermediate_size} must be divisible by tensor "
                f"parallel world_size={world_size}"
            )

        if parallel_mode == "tp":
            local_intermediate_size = shared_intermediate_size // world_size
            intermediate_start = rank * local_intermediate_size
            reduce_output = world_size > 1
        else:
            local_intermediate_size = shared_intermediate_size
            intermediate_start = 0
            reduce_output = False
        return cls(
            parallel_mode=parallel_mode,
            world_size=world_size,
            rank=rank,
            shared_intermediate_size=shared_intermediate_size,
            local_intermediate_size=local_intermediate_size,
            intermediate_start=intermediate_start,
            intermediate_end=intermediate_start + local_intermediate_size,
            reduce_output=reduce_output,
        )


class SoftmaxTopKRouter(nn.Module):
    """Softmax top-k router shared by MoE model adapters."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        if not 0 < top_k <= num_experts:
            raise ValueError(
                f"top_k must be in [1, {num_experts}], got {top_k}"
            )
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=dtype)
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> RoutingResult:
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = F.linear(flat_states, self.weight)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        routing_weights, selected_experts = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        return RoutingResult(
            router_logits=router_logits,
            routing_weights=routing_weights.to(router_logits.dtype),
            selected_experts=selected_experts,
        )


class Qwen3MoeTopKRouter(SoftmaxTopKRouter):
    """Compatibility name for the Qwen3 softmax top-k router."""


class DeepSeekGroupedTopKRouter(nn.Module):
    """Grouped top-k router for the audited DeepSeek-V2/V3 policies."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        num_groups: int,
        topk_groups: int,
        score_func: str,
        topk_method: str,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if type(num_experts) is not int or num_experts <= 0:
            raise ValueError("num_experts must be a positive integer")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        grouped_config = GroupedTopKConfig(
            num_groups=num_groups,
            topk_groups=topk_groups,
            score_func=score_func,
            topk_method=topk_method,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
        )
        if num_experts % grouped_config.num_groups:
            raise ValueError("num_experts must be divisible by num_groups")
        experts_per_group = num_experts // grouped_config.num_groups
        selected_capacity = grouped_config.topk_groups * experts_per_group
        if top_k > selected_capacity:
            raise ValueError("top_k exceeds the selected-group capacity")
        if (
            grouped_config.topk_method == "noaux_tc"
            and experts_per_group < 2
        ):
            raise ValueError("noaux_tc requires at least two experts per group")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.grouped_config = grouped_config
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=dtype)
        )
        if grouped_config.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(num_experts, dtype=torch.float32)
            )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> RoutingResult:
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = F.linear(flat_states, self.weight)
        if self.grouped_config.score_func == "softmax":
            original_scores = F.softmax(router_logits.float(), dim=-1)
        else:
            original_scores = torch.sigmoid(router_logits.float())

        selection_scores = original_scores
        if self.grouped_config.topk_method == "noaux_tc":
            selection_scores = (
                selection_scores + self.e_score_correction_bias
            )

        experts_per_group = (
            self.num_experts // self.grouped_config.num_groups
        )
        grouped_scores = selection_scores.reshape(
            flat_states.shape[0],
            self.grouped_config.num_groups,
            experts_per_group,
        )
        if self.grouped_config.topk_method == "group_limited_greedy":
            group_scores = grouped_scores.max(dim=-1).values
        else:
            group_scores = grouped_scores.topk(2, dim=-1).values.sum(dim=-1)

        selected_groups = group_scores.topk(
            self.grouped_config.topk_groups,
            dim=-1,
        ).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, selected_groups, True)
        expert_mask = group_mask.unsqueeze(-1).expand_as(grouped_scores).reshape(
            flat_states.shape[0],
            self.num_experts,
        )
        masked_selection_scores = selection_scores.masked_fill(
            ~expert_mask,
            -torch.inf,
        )
        selected_experts = masked_selection_scores.topk(
            self.top_k,
            dim=-1,
        ).indices

        routing_weights = original_scores.gather(1, selected_experts)
        if self.grouped_config.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1,
                keepdim=True,
            )
        routing_weights = (
            routing_weights * self.grouped_config.routed_scaling_factor
        )
        return RoutingResult(
            router_logits=router_logits,
            routing_weights=routing_weights.to(router_logits.dtype),
            selected_experts=selected_experts,
        )


class RoutedExpertExecutor(nn.Module):
    """Routed experts with eager and Ascend GMM execution paths."""

    _warned_gmm_unavailable = False

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        intermediate_size: int,
        tp_config=None,
        layer_index: int | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        placement = ExpertPlacement.from_config(
            num_experts=num_experts,
            intermediate_size=intermediate_size,
            tp_config=tp_config,
        )
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.placement = placement
        self.parallel_mode = placement.parallel_mode
        self.local_num_experts = placement.local_num_experts
        self.expert_start = placement.expert_start
        self.expert_end = placement.expert_end
        self.local_intermediate_size = placement.local_intermediate_size
        self.tp_config = tp_config
        self.layer_index = layer_index
        self.backend = os.environ.get(
            "LITE_LLAMA_MOE_BACKEND", "auto"
        ).lower()
        if self.backend not in {"auto", "eager", "gmm", "routed_gemv"}:
            raise ValueError(
                "LITE_LLAMA_MOE_BACKEND must be one of "
                "auto/eager/gmm/routed_gemv, "
                f"got {self.backend!r}"
            )
        self.routed_gemv_max_assignments = int(
            os.environ.get("LITE_LLAMA_MOE_GEMV_MAX_ASSIGNMENTS", "64")
        )
        self.validate_gmm = os.environ.get(
            "LITE_LLAMA_MOE_VALIDATE", "0"
        ).lower() in {"1", "true", "yes", "on"}
        self.alignment_rtol = float(
            os.environ.get("LITE_LLAMA_MOE_ALIGNMENT_RTOL", "1e-2")
        )
        self.alignment_atol = float(
            os.environ.get("LITE_LLAMA_MOE_ALIGNMENT_ATOL", "1e-2")
        )

        # GMM-native layout: [expert, input, output]. Converted checkpoints
        # remain [expert, output, input] and are transposed by the loader.
        self.gate_up_weight = nn.Parameter(
            torch.empty(
                self.local_num_experts,
                hidden_size,
                2 * self.local_intermediate_size,
                dtype=dtype,
            )
        )
        self.down_weight = nn.Parameter(
            torch.empty(
                self.local_num_experts,
                self.local_intermediate_size,
                hidden_size,
                dtype=dtype,
            )
        )

    @staticmethod
    def _swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        # The production NPU path reuses the project's Triton SwiGLU kernel.
        # CPU remains a dependency-free reference path for unit tests.
        if gate.device.type == "npu":
            from ..kernels import swiglu_forward

            return swiglu_forward(gate, up)
        return F.silu(gate) * up

    def _forward_eager_local(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        # Only launch experts selected by at least one token. This implementation
        # intentionally prioritizes correctness; grouped matmul replaces this
        # host-visible dispatch in the next performance release.
        active_experts = torch.unique(selected_experts).tolist()
        for global_expert_id in active_experts:
            if not self.expert_start <= global_expert_id < self.expert_end:
                continue
            expert_id = global_expert_id - self.expert_start
            token_indices, topk_slots = torch.where(
                selected_experts == global_expert_id
            )
            current_states = hidden_states[token_indices]
            gate_up = F.linear(
                current_states, self.gate_up_weight[expert_id].transpose(0, 1)
            )
            gate, up = gate_up.chunk(2, dim=-1)
            current_states = self._swiglu(gate, up)
            current_states = F.linear(
                current_states, self.down_weight[expert_id].transpose(0, 1)
            )
            current_states = current_states * routing_weights[
                token_indices, topk_slots
            ].unsqueeze(-1)
            final_hidden_states.index_add_(
                0, token_indices, current_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states

    @staticmethod
    def _npu_gmm_available() -> bool:
        try:
            import torch_npu
        except ImportError:
            return False
        return hasattr(torch_npu, "npu_grouped_matmul")

    @staticmethod
    def _run_grouped_matmul(
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        kwargs = dict(
            x=[x],
            weight=[weight],
            bias=None,
            group_list=group_list,
            split_item=2,
        )
        try:
            result = torch_npu.npu_grouped_matmul(
                **kwargs,
                group_type=0,
                group_list_type=0,
            )
        except TypeError as exc:
            if "unexpected keyword" not in str(exc):
                raise
            result = torch_npu.npu_grouped_matmul(**kwargs)
        return result[0] if isinstance(result, (list, tuple)) else result

    def _forward_grouped_local(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if __package__:
            from ..kernels.moe_routing import (
                finalize_moe_routing,
                prepare_moe_routing,
                prepare_moe_routing_local,
            )
        else:  # Direct file loading used by dependency-light CPU tests.
            from qwen3_moe_routing import (
                finalize_moe_routing,
                prepare_moe_routing,
                prepare_moe_routing_local,
            )

        if self.parallel_mode == "ep":
            plan = prepare_moe_routing_local(
                hidden_states,
                selected_experts,
                routing_weights,
                self.expert_start,
                self.local_num_experts,
            )
            if plan.routed_states.shape[0] == 0:
                return torch.zeros_like(hidden_states)
        else:
            plan = prepare_moe_routing(
                hidden_states,
                selected_experts,
                routing_weights,
                self.num_experts,
            )
        gate_up = self._run_grouped_matmul(
            plan.routed_states,
            self.gate_up_weight,
            plan.group_list,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = self._swiglu(gate, up)
        expert_output = self._run_grouped_matmul(
            activated,
            self.down_weight,
            plan.group_list,
        )
        return finalize_moe_routing(
            expert_output,
            plan.sorted_token_ids,
            plan.sorted_weights,
            hidden_states.shape[0],
        )

    def _forward_routed_local(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if __package__:
            from ..kernels.moe_routed_gemv import routed_expert_matmul
        else:
            routed_expert_matmul = sys.modules[
                "qwen3_moe_routed_gemv"
            ].routed_expert_matmul
        return routed_expert_matmul(
            hidden_states,
            selected_experts,
            routing_weights,
            self.gate_up_weight,
            self.down_weight,
            expert_start=self.expert_start,
            local_num_experts=self.local_num_experts,
            filter_local_experts=self.parallel_mode == "ep",
        )

    def _should_use_routed_gemv(
        self,
        *,
        num_tokens: int,
        top_k: int,
        device_type: str,
    ) -> bool:
        return (
            device_type == "npu"
            and num_tokens * top_k <= self.routed_gemv_max_assignments
        )

    def _validate_local_outputs(
        self,
        reference: torch.Tensor,
        actual: torch.Tensor,
        *,
        rtol: float,
        atol: float,
    ) -> None:
        if torch.allclose(reference, actual, rtol=rtol, atol=atol):
            return
        difference = (reference.float() - actual.float()).abs()
        layer = (
            "unknown" if self.layer_index is None else str(self.layer_index)
        )
        raise RuntimeError(
            f"MoE GMM alignment failed at layer {layer}: "
            f"shape={tuple(reference.shape)}, "
            f"max_abs_diff={difference.max().item():.6g}, "
            f"mean_abs_diff={difference.mean().item():.6g}, "
            f"rtol={rtol}, atol={atol}"
        )

    def _use_grouped_backend(self, hidden_states: torch.Tensor) -> bool:
        if self.backend == "eager":
            return False
        if hidden_states.device.type != "npu":
            if self.backend == "gmm":
                raise RuntimeError(
                    "LITE_LLAMA_MOE_BACKEND=gmm requires an NPU tensor"
                )
            return False
        available = self._npu_gmm_available()
        if self.backend == "gmm" and not available:
            raise RuntimeError(
                "LITE_LLAMA_MOE_BACKEND=gmm was requested, but "
                "torch_npu.npu_grouped_matmul is unavailable"
            )
        if not available and not self._warned_gmm_unavailable:
            warnings.warn(
                "torch_npu.npu_grouped_matmul is unavailable; "
                "falling back to eager MoE experts",
                RuntimeWarning,
                stacklevel=2,
            )
            type(self)._warned_gmm_unavailable = True
        return available

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        use_routed = self.backend == "routed_gemv" or (
            self.backend == "auto"
            and self._should_use_routed_gemv(
                num_tokens=hidden_states.shape[0],
                top_k=selected_experts.shape[1],
                device_type=hidden_states.device.type,
            )
        )
        use_grouped = (
            not use_routed and self._use_grouped_backend(hidden_states)
        )
        if use_routed:
            final_hidden_states = self._forward_routed_local(
                hidden_states, selected_experts, routing_weights
            )
            if self.validate_gmm:
                reference = self._forward_eager_local(
                    hidden_states, selected_experts, routing_weights
                )
                self._validate_local_outputs(
                    reference,
                    final_hidden_states,
                    rtol=self.alignment_rtol,
                    atol=self.alignment_atol,
                )
        elif use_grouped:
            final_hidden_states = self._forward_grouped_local(
                hidden_states, selected_experts, routing_weights
            )
            if self.validate_gmm:
                reference = self._forward_eager_local(
                    hidden_states, selected_experts, routing_weights
                )
                self._validate_local_outputs(
                    reference,
                    final_hidden_states,
                    rtol=self.alignment_rtol,
                    atol=self.alignment_atol,
                )
        else:
            final_hidden_states = self._forward_eager_local(
                hidden_states, selected_experts, routing_weights
            )

        if getattr(self.tp_config, "enabled", False):
            from ..executor.tp_utils import tp_all_reduce

            final_hidden_states = tp_all_reduce(final_hidden_states)
        return final_hidden_states


def _shared_expert_all_reduce(output: torch.Tensor) -> torch.Tensor:
    """Reduce a TP shared-expert partial through the existing process group."""

    from ..executor.tp_utils import tp_all_reduce

    return tp_all_reduce(output)


class SharedExpertMLP(nn.Module):
    """One combined shared SwiGLU expert with explicit reduce ownership."""

    def __init__(
        self,
        hidden_size: int,
        shared_intermediate_size: int,
        tp_config=None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        placement = SharedExpertPlacement.from_config(
            shared_intermediate_size=shared_intermediate_size,
            tp_config=tp_config,
        )
        self.hidden_size = hidden_size
        self.shared_intermediate_size = shared_intermediate_size
        self.local_intermediate_size = placement.local_intermediate_size
        self.tp_config = tp_config
        self.placement = placement
        self.gate_up_weight = nn.Parameter(
            torch.empty(
                hidden_size,
                2 * self.local_intermediate_size,
                dtype=dtype,
            )
        )
        self.down_weight = nn.Parameter(
            torch.empty(
                self.local_intermediate_size,
                hidden_size,
                dtype=dtype,
            )
        )

    def _forward_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = hidden_states @ self.gate_up_weight
        gate, up = gate_up.chunk(2, dim=-1)
        activated = RoutedExpertExecutor._swiglu(gate, up)
        return activated @ self.down_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self._forward_local(hidden_states)
        if self.placement.reduce_output:
            output = _shared_expert_all_reduce(output)
        return output


class DeepSeekMoeBlock(nn.Module):
    """Minimal routed-plus-shared DeepSeek-V2/V3 MoE orchestration."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        shared_intermediate_size: int,
        num_groups: int,
        topk_groups: int,
        score_func: str,
        topk_method: str,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        tp_config=None,
        layer_index: int | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gate = DeepSeekGroupedTopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            num_groups=num_groups,
            topk_groups=topk_groups,
            score_func=score_func,
            topk_method=topk_method,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            dtype=dtype,
        )
        self.experts = RoutedExpertExecutor(
            hidden_size=hidden_size,
            num_experts=num_experts,
            intermediate_size=intermediate_size,
            tp_config=tp_config,
            layer_index=layer_index,
            dtype=dtype,
        )
        self.shared_experts = SharedExpertMLP(
            hidden_size=hidden_size,
            shared_intermediate_size=shared_intermediate_size,
            tp_config=tp_config,
            dtype=dtype,
        )
        self.last_router_logits: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        routing = self.gate(flat_states)
        self.last_router_logits = routing.router_logits
        routed_output = self.experts(
            flat_states,
            routing.selected_experts,
            routing.routing_weights,
        )
        shared_output = self.shared_experts(flat_states)
        return (routed_output + shared_output).reshape(original_shape)


class Qwen3MoeExperts(RoutedExpertExecutor):
    """Compatibility name for the Qwen3 routed-expert executor."""


class Qwen3SparseMoeBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        norm_topk_prob: bool = True,
        tp_config=None,
        layer_index: int | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gate = Qwen3MoeTopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
            dtype=dtype,
        )
        self.experts = Qwen3MoeExperts(
            hidden_size=hidden_size,
            num_experts=num_experts,
            intermediate_size=intermediate_size,
            tp_config=tp_config,
            layer_index=layer_index,
            dtype=dtype,
        )
        self.last_router_logits: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits, routing_weights, selected_experts = self.gate(flat_states)
        self.last_router_logits = router_logits
        output = self.experts(
            flat_states, selected_experts, routing_weights
        )
        return output.reshape(original_shape)
