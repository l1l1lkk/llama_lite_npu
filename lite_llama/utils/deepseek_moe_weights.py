"""Pure DeepSeek-V2/V3 MoE checkpoint canonicalization helpers."""

from __future__ import annotations

import torch


def _validated_layer_indices(moe_layer_indices) -> tuple[int, ...]:
    try:
        layer_indices = tuple(moe_layer_indices)
    except TypeError as exc:
        raise TypeError("moe_layer_indices must be an iterable of integers") from exc
    if not layer_indices:
        raise ValueError("moe_layer_indices must not be empty")
    seen = set()
    for layer_id in layer_indices:
        if type(layer_id) is not int or layer_id < 0:
            raise ValueError(
                "moe_layer_indices must contain non-negative integers"
            )
        if layer_id in seen:
            raise ValueError(
                f"moe_layer_indices contains duplicate layer {layer_id}"
            )
        seen.add(layer_id)
    return layer_indices


def _required_float_tensor(
    state: dict[str, torch.Tensor],
    key: str,
    *,
    layer: int,
    expert: int | None = None,
) -> torch.Tensor:
    location = f"layer {layer}"
    if expert is not None:
        location += f" expert {expert}"
    try:
        tensor = state[key]
    except KeyError as exc:
        raise KeyError(f"{location} is missing required weight {key}") from exc
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{location} weight {key} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{location} weight {key} must be floating point")
    return tensor


def _require_rank(
    tensor: torch.Tensor,
    rank: int,
    *,
    key: str,
    layer: int,
    expert: int | None = None,
) -> None:
    if tensor.ndim == rank:
        return
    location = f"layer {layer}"
    if expert is not None:
        location += f" expert {expert}"
    raise ValueError(
        f"{location} weight {key} must be rank {rank}, got shape "
        f"{tuple(tensor.shape)}"
    )


def _require_dtype_device(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    key: str,
    layer: int,
    expert: int | None = None,
) -> None:
    if tensor.dtype == dtype and tensor.device == device:
        return
    location = f"layer {layer}"
    if expert is not None:
        location += f" expert {expert}"
    raise ValueError(
        f"{location} weight {key} dtype/device mismatch: expected "
        f"{dtype} on {device}, got {tensor.dtype} on {tensor.device}"
    )


def stack_deepseek_moe_weights(
    hf_state: dict[str, torch.Tensor],
    *,
    moe_layer_indices,
    num_experts: int,
    use_correction_bias: bool,
    consume: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert explicit HF DeepSeek MoE layers to canonical checkpoint keys."""

    if not isinstance(hf_state, dict):
        raise TypeError("hf_state must be a dict")
    layer_indices = _validated_layer_indices(moe_layer_indices)
    if type(num_experts) is not int or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    if type(use_correction_bias) is not bool:
        raise TypeError("use_correction_bias must be bool")
    if type(consume) is not bool:
        raise TypeError("consume must be bool")

    converted: dict[str, torch.Tensor] = {}
    consumed_keys: list[str] = []
    checkpoint_dtype = None
    checkpoint_device = None

    for layer_id in layer_indices:
        mlp_prefix = f"model.layers.{layer_id}.mlp"
        output_prefix = f"layers.{layer_id}.mlp"
        router_key = f"{mlp_prefix}.gate.weight"
        router = _required_float_tensor(
            hf_state,
            router_key,
            layer=layer_id,
        )
        _require_rank(router, 2, key=router_key, layer=layer_id)
        if router.shape[0] != num_experts or router.shape[1] <= 0:
            raise ValueError(
                f"layer {layer_id} weight {router_key} must have shape "
                f"[{num_experts}, hidden], got {tuple(router.shape)}"
            )
        if checkpoint_dtype is None:
            checkpoint_dtype = router.dtype
            checkpoint_device = router.device
        _require_dtype_device(
            router,
            dtype=checkpoint_dtype,
            device=checkpoint_device,
            key=router_key,
            layer=layer_id,
        )
        hidden_size = router.shape[1]
        converted[f"{output_prefix}.gate.weight"] = router
        consumed_keys.append(router_key)

        if use_correction_bias:
            bias_key = f"{mlp_prefix}.gate.e_score_correction_bias"
            bias = _required_float_tensor(
                hf_state,
                bias_key,
                layer=layer_id,
            )
            _require_rank(bias, 1, key=bias_key, layer=layer_id)
            if bias.shape != (num_experts,):
                raise ValueError(
                    f"layer {layer_id} weight {bias_key} must have shape "
                    f"[{num_experts}], got {tuple(bias.shape)}"
                )
            if bias.device != checkpoint_device:
                raise ValueError(
                    f"layer {layer_id} weight {bias_key} device mismatch: "
                    f"expected {checkpoint_device}, got {bias.device}"
                )
            converted[
                f"{output_prefix}.gate.e_score_correction_bias"
            ] = bias.float()
            consumed_keys.append(bias_key)

        gate_up_experts = []
        down_experts = []
        intermediate_size = None
        for expert_id in range(num_experts):
            expert_prefix = f"{mlp_prefix}.experts.{expert_id}"
            gate_key = f"{expert_prefix}.gate_proj.weight"
            up_key = f"{expert_prefix}.up_proj.weight"
            down_key = f"{expert_prefix}.down_proj.weight"
            gate = _required_float_tensor(
                hf_state,
                gate_key,
                layer=layer_id,
                expert=expert_id,
            )
            up = _required_float_tensor(
                hf_state,
                up_key,
                layer=layer_id,
                expert=expert_id,
            )
            down = _required_float_tensor(
                hf_state,
                down_key,
                layer=layer_id,
                expert=expert_id,
            )
            for key, tensor in ((gate_key, gate), (up_key, up), (down_key, down)):
                _require_rank(
                    tensor,
                    2,
                    key=key,
                    layer=layer_id,
                    expert=expert_id,
                )
                _require_dtype_device(
                    tensor,
                    dtype=checkpoint_dtype,
                    device=checkpoint_device,
                    key=key,
                    layer=layer_id,
                    expert=expert_id,
                )
            if gate.shape != up.shape:
                raise ValueError(
                    f"layer {layer_id} expert {expert_id} gate/up shapes "
                    f"differ: {tuple(gate.shape)} != {tuple(up.shape)}"
                )
            if gate.shape[0] <= 0 or gate.shape[1] != hidden_size:
                raise ValueError(
                    f"layer {layer_id} expert {expert_id} weight {gate_key} "
                    f"must have shape [intermediate, {hidden_size}], got "
                    f"{tuple(gate.shape)}"
                )
            if intermediate_size is None:
                intermediate_size = gate.shape[0]
            if gate.shape[0] != intermediate_size:
                raise ValueError(
                    f"layer {layer_id} expert {expert_id} intermediate size "
                    f"mismatch: expected {intermediate_size}, got {gate.shape[0]}"
                )
            if down.shape != (hidden_size, intermediate_size):
                raise ValueError(
                    f"layer {layer_id} expert {expert_id} weight {down_key} "
                    f"must have shape [{hidden_size}, {intermediate_size}], got "
                    f"{tuple(down.shape)}"
                )
            gate_up_experts.append(torch.cat((gate, up), dim=0))
            down_experts.append(down)
            consumed_keys.extend((gate_key, up_key, down_key))

        converted[
            f"{output_prefix}.experts.gate_up_weight"
        ] = torch.stack(gate_up_experts, dim=0)
        converted[
            f"{output_prefix}.experts.down_weight"
        ] = torch.stack(down_experts, dim=0)

        shared_prefix = f"{mlp_prefix}.shared_experts"
        shared_gate_key = f"{shared_prefix}.gate_proj.weight"
        shared_up_key = f"{shared_prefix}.up_proj.weight"
        shared_down_key = f"{shared_prefix}.down_proj.weight"
        shared_gate = _required_float_tensor(
            hf_state,
            shared_gate_key,
            layer=layer_id,
        )
        shared_up = _required_float_tensor(
            hf_state,
            shared_up_key,
            layer=layer_id,
        )
        shared_down = _required_float_tensor(
            hf_state,
            shared_down_key,
            layer=layer_id,
        )
        for key, tensor in (
            (shared_gate_key, shared_gate),
            (shared_up_key, shared_up),
            (shared_down_key, shared_down),
        ):
            _require_rank(tensor, 2, key=key, layer=layer_id)
            _require_dtype_device(
                tensor,
                dtype=checkpoint_dtype,
                device=checkpoint_device,
                key=key,
                layer=layer_id,
            )
        if shared_gate.shape != shared_up.shape:
            raise ValueError(
                f"layer {layer_id} shared gate/up shapes differ: "
                f"{tuple(shared_gate.shape)} != {tuple(shared_up.shape)}"
            )
        if shared_gate.shape[0] <= 0 or shared_gate.shape[1] != hidden_size:
            raise ValueError(
                f"layer {layer_id} weight {shared_gate_key} must have shape "
                f"[shared_intermediate, {hidden_size}], got "
                f"{tuple(shared_gate.shape)}"
            )
        shared_intermediate_size = shared_gate.shape[0]
        if shared_down.shape != (hidden_size, shared_intermediate_size):
            raise ValueError(
                f"layer {layer_id} weight {shared_down_key} must have shape "
                f"[{hidden_size}, {shared_intermediate_size}], got "
                f"{tuple(shared_down.shape)}"
            )
        converted[
            f"{output_prefix}.shared_experts.gate_up_weight"
        ] = torch.cat((shared_gate, shared_up), dim=0)
        converted[
            f"{output_prefix}.shared_experts.down_weight"
        ] = shared_down
        consumed_keys.extend(
            (shared_gate_key, shared_up_key, shared_down_key)
        )

    if consume:
        for key in consumed_keys:
            del hf_state[key]
    return converted
