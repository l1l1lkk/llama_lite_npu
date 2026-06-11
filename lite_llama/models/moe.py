"""Correctness-first Qwen3 MoE routing and expert execution."""

from __future__ import annotations

import os
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3MoeTopKRouter(nn.Module):
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return (
            router_logits,
            routing_weights.to(router_logits.dtype),
            selected_experts,
        )


class Qwen3MoeExperts(nn.Module):
    """Tensor-parallel experts with eager and Ascend GMM execution paths."""

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
        world_size = getattr(tp_config, "world_size", 1)
        if intermediate_size % world_size != 0:
            raise ValueError(
                f"moe_intermediate_size={intermediate_size} must be divisible "
                f"by tensor parallel world_size={world_size}"
            )
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.local_intermediate_size = intermediate_size // world_size
        self.tp_config = tp_config
        self.layer_index = layer_index
        self.backend = os.environ.get(
            "LITE_LLAMA_MOE_BACKEND", "auto"
        ).lower()
        if self.backend not in {"auto", "eager", "gmm"}:
            raise ValueError(
                "LITE_LLAMA_MOE_BACKEND must be one of auto/eager/gmm, "
                f"got {self.backend!r}"
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
                num_experts,
                hidden_size,
                2 * self.local_intermediate_size,
                dtype=dtype,
            )
        )
        self.down_weight = nn.Parameter(
            torch.empty(
                num_experts,
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
        for expert_id in active_experts:
            token_indices, topk_slots = torch.where(
                selected_experts == expert_id
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
            )
        else:  # Direct file loading used by dependency-light CPU tests.
            from qwen3_moe_routing import (
                finalize_moe_routing,
                prepare_moe_routing,
            )

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
        use_grouped = self._use_grouped_backend(hidden_states)
        if use_grouped:
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
