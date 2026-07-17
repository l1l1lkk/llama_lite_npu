"""DeepSeek-V2/V3 MoE component construction and CPU state preparation."""

from __future__ import annotations

from collections.abc import Mapping

import torch

if __package__:
    from .model_config import DeepSeekMoeConfig
    from .moe import DeepSeekMoeBlock
    from ..executor.tp_utils import (
        TPConfig,
        prepare_moe_down_for_ep,
        prepare_moe_down_for_gmm,
        prepare_moe_gate_up_for_ep,
        prepare_moe_gate_up_for_gmm,
        prepare_shared_expert_down,
        prepare_shared_expert_gate_up,
    )
else:  # Narrow aliases installed by dependency-light direct-file tests.
    from deepseek_moe_component_config import DeepSeekMoeConfig
    from deepseek_moe_component_moe import DeepSeekMoeBlock
    from deepseek_moe_component_tp_utils import (
        TPConfig,
        prepare_moe_down_for_ep,
        prepare_moe_down_for_gmm,
        prepare_moe_gate_up_for_ep,
        prepare_moe_gate_up_for_gmm,
        prepare_shared_expert_down,
        prepare_shared_expert_gate_up,
    )


def _require_component_config(config) -> DeepSeekMoeConfig:
    if not isinstance(config, DeepSeekMoeConfig):
        raise TypeError("config must be a DeepSeekMoeConfig")
    return config


def _require_moe_layer(config: DeepSeekMoeConfig, layer_index: int) -> None:
    if type(layer_index) is not int:
        raise TypeError("layer_index must be an integer")
    if not 0 <= layer_index < config.num_layers:
        raise ValueError(
            f"layer_index={layer_index} must be in [0, {config.num_layers})"
        )
    if not config.is_moe_layer(layer_index):
        raise ValueError(
            f"layer {layer_index} is not a scheduled DeepSeek MoE layer"
        )


def _validated_tp_config(config: DeepSeekMoeConfig, tp_config) -> TPConfig:
    tp = TPConfig() if tp_config is None else tp_config
    world_size = getattr(tp, "world_size", None)
    rank = getattr(tp, "rank", None)
    mode = getattr(tp, "moe_parallel_mode", None)
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("tp_config.world_size must be a positive integer")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("tp_config.rank must be an integer in [0, world_size)")
    config.validate_moe_parallel(world_size, mode)
    return tp


def _required_cpu_float_tensor(
    state: Mapping[str, torch.Tensor],
    key: str,
) -> torch.Tensor:
    if key not in state:
        raise KeyError(f"missing canonical DeepSeek MoE weight: {key}")
    tensor = state[key]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"canonical DeepSeek MoE weight {key} must be a Tensor")
    if tensor.device.type != "cpu" or not tensor.is_floating_point():
        raise ValueError(
            f"canonical DeepSeek MoE weight {key} must be a floating CPU tensor"
        )
    return tensor


def _require_shape(
    tensor: torch.Tensor,
    expected: tuple[int, ...],
    *,
    key: str,
) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"canonical DeepSeek MoE weight {key} must have shape "
            f"{expected}, got {tuple(tensor.shape)}"
        )


def build_deepseek_moe_block(
    config: DeepSeekMoeConfig,
    layer_index: int,
    tp_config=None,
    dtype: torch.dtype = torch.float16,
) -> DeepSeekMoeBlock:
    """Build one scheduled routed-plus-shared MoE block, never a full model."""

    config = _require_component_config(config)
    _require_moe_layer(config, layer_index)
    tp = _validated_tp_config(config, tp_config)
    if not isinstance(dtype, torch.dtype) or not torch.empty(
        (), dtype=dtype
    ).is_floating_point():
        raise TypeError("dtype must be a floating torch.dtype")
    return DeepSeekMoeBlock(
        hidden_size=config.hidden_size,
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
        intermediate_size=config.moe_intermediate_size,
        shared_intermediate_size=config.shared_intermediate_size,
        num_groups=config.num_groups,
        topk_groups=config.topk_groups,
        score_func=config.score_func,
        topk_method=config.topk_method,
        norm_topk_prob=config.norm_topk_prob,
        routed_scaling_factor=config.routed_scaling_factor,
        tp_config=tp,
        layer_index=layer_index,
        dtype=dtype,
    )


def prepare_deepseek_moe_layer_state(
    canonical_state: Mapping[str, torch.Tensor],
    config: DeepSeekMoeConfig,
    layer_index: int,
    tp_config=None,
) -> dict[str, torch.Tensor]:
    """Prepare one canonical CPU MoE layer for one TP/EP runtime rank."""

    if not isinstance(canonical_state, Mapping):
        raise TypeError("canonical_state must be a mapping")
    config = _require_component_config(config)
    _require_moe_layer(config, layer_index)
    tp = _validated_tp_config(config, tp_config)
    prefix = f"layers.{layer_index}.mlp."
    keys = {
        "gate": prefix + "gate.weight",
        "routed_gate_up": prefix + "experts.gate_up_weight",
        "routed_down": prefix + "experts.down_weight",
        "shared_gate_up": prefix + "shared_experts.gate_up_weight",
        "shared_down": prefix + "shared_experts.down_weight",
        "bias": prefix + "gate.e_score_correction_bias",
    }
    gate = _required_cpu_float_tensor(canonical_state, keys["gate"])
    routed_gate_up = _required_cpu_float_tensor(
        canonical_state, keys["routed_gate_up"]
    )
    routed_down = _required_cpu_float_tensor(
        canonical_state, keys["routed_down"]
    )
    shared_gate_up = _required_cpu_float_tensor(
        canonical_state, keys["shared_gate_up"]
    )
    shared_down = _required_cpu_float_tensor(
        canonical_state, keys["shared_down"]
    )

    _require_shape(
        gate,
        (config.num_experts, config.hidden_size),
        key=keys["gate"],
    )
    _require_shape(
        routed_gate_up,
        (
            config.num_experts,
            2 * config.moe_intermediate_size,
            config.hidden_size,
        ),
        key=keys["routed_gate_up"],
    )
    _require_shape(
        routed_down,
        (
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        ),
        key=keys["routed_down"],
    )
    _require_shape(
        shared_gate_up,
        (2 * config.shared_intermediate_size, config.hidden_size),
        key=keys["shared_gate_up"],
    )
    _require_shape(
        shared_down,
        (config.hidden_size, config.shared_intermediate_size),
        key=keys["shared_down"],
    )

    checkpoint_dtype = gate.dtype
    checkpoint_device = gate.device
    for key, tensor in (
        (keys["routed_gate_up"], routed_gate_up),
        (keys["routed_down"], routed_down),
        (keys["shared_gate_up"], shared_gate_up),
        (keys["shared_down"], shared_down),
    ):
        if tensor.dtype != checkpoint_dtype or tensor.device != checkpoint_device:
            raise ValueError(
                f"canonical DeepSeek MoE weight {key} must match "
                f"dtype={checkpoint_dtype} and device={checkpoint_device}"
            )

    runtime: dict[str, torch.Tensor] = {"gate.weight": gate}
    if config.uses_correction_bias:
        bias = _required_cpu_float_tensor(canonical_state, keys["bias"])
        _require_shape(bias, (config.num_experts,), key=keys["bias"])
        if bias.dtype != torch.float32 or bias.device != checkpoint_device:
            raise ValueError(
                f"canonical DeepSeek MoE weight {keys['bias']} must be "
                f"FP32 on device={checkpoint_device}"
            )
        runtime["gate.e_score_correction_bias"] = bias

    if tp.moe_parallel_mode == "tp":
        runtime["experts.gate_up_weight"] = prepare_moe_gate_up_for_gmm(
            routed_gate_up,
            config.moe_intermediate_size,
            tp,
        )
        runtime["experts.down_weight"] = prepare_moe_down_for_gmm(
            routed_down,
            tp,
        )
    else:
        runtime["experts.gate_up_weight"] = prepare_moe_gate_up_for_ep(
            routed_gate_up,
            tp,
        )
        runtime["experts.down_weight"] = prepare_moe_down_for_ep(
            routed_down,
            tp,
        )
    runtime["shared_experts.gate_up_weight"] = (
        prepare_shared_expert_gate_up(
            shared_gate_up,
            config.shared_intermediate_size,
            tp,
        )
    )
    runtime["shared_experts.down_weight"] = prepare_shared_expert_down(
        shared_down,
        tp,
    )
    return runtime


__all__ = [
    "build_deepseek_moe_block",
    "prepare_deepseek_moe_layer_state",
]
