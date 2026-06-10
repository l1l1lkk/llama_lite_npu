"""Pure helpers for converting Hugging Face Qwen3 MoE expert weights."""

from __future__ import annotations

import torch


def _required_weight(
    state: dict[str, torch.Tensor],
    key: str,
    *,
    layer: int,
    expert: int | None = None,
    consume: bool = False,
) -> torch.Tensor:
    try:
        return state.pop(key) if consume else state[key]
    except KeyError as exc:
        location = f"layer {layer}"
        if expert is not None:
            location += f" expert {expert}"
        raise KeyError(f"{location} is missing required weight {key}") from exc


def stack_qwen3_moe_weights(
    hf_state: dict[str, torch.Tensor],
    *,
    num_layers: int,
    num_experts: int,
    consume: bool = False,
) -> dict[str, torch.Tensor]:
    """Stack expert weights into the runtime's dense 3D layout."""
    converted: dict[str, torch.Tensor] = {}
    for layer_id in range(num_layers):
        mlp_prefix = f"model.layers.{layer_id}.mlp"
        converted[f"layers.{layer_id}.mlp.gate.weight"] = _required_weight(
            hf_state,
            f"{mlp_prefix}.gate.weight",
            layer=layer_id,
            consume=consume,
        )

        gate_up_experts = []
        down_experts = []
        for expert_id in range(num_experts):
            expert_prefix = f"{mlp_prefix}.experts.{expert_id}"
            gate = _required_weight(
                hf_state,
                f"{expert_prefix}.gate_proj.weight",
                layer=layer_id,
                expert=expert_id,
                consume=consume,
            )
            up = _required_weight(
                hf_state,
                f"{expert_prefix}.up_proj.weight",
                layer=layer_id,
                expert=expert_id,
                consume=consume,
            )
            down = _required_weight(
                hf_state,
                f"{expert_prefix}.down_proj.weight",
                layer=layer_id,
                expert=expert_id,
                consume=consume,
            )
            if gate.shape != up.shape:
                raise ValueError(
                    f"layer {layer_id} expert {expert_id} gate/up shapes differ: "
                    f"{tuple(gate.shape)} != {tuple(up.shape)}"
                )
            gate_up_experts.append(torch.cat((gate, up), dim=0))
            down_experts.append(down)

        converted[
            f"layers.{layer_id}.mlp.experts.gate_up_weight"
        ] = torch.stack(gate_up_experts, dim=0)
        converted[
            f"layers.{layer_id}.mlp.experts.down_weight"
        ] = torch.stack(down_experts, dim=0)

    return converted
