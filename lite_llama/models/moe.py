"""Correctness-first Qwen3 MoE routing and expert execution."""

from __future__ import annotations

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
    """Stacked expert weights with correctness-first dynamic dispatch."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        intermediate_size: int,
        tp_config=None,
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

        self.gate_up_weight = nn.Parameter(
            torch.empty(
                num_experts,
                2 * self.local_intermediate_size,
                hidden_size,
                dtype=dtype,
            )
        )
        self.down_weight = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                self.local_intermediate_size,
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

    def forward(
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
                current_states, self.gate_up_weight[expert_id]
            )
            gate, up = gate_up.chunk(2, dim=-1)
            current_states = self._swiglu(gate, up)
            current_states = F.linear(
                current_states, self.down_weight[expert_id]
            )
            current_states = current_states * routing_weights[
                token_indices, topk_slots
            ].unsqueeze(-1)
            final_hidden_states.index_add_(
                0, token_indices, current_states.to(final_hidden_states.dtype)
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
