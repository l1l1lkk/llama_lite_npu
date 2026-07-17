"""Bounded safetensors reader for one scheduled DeepSeek-V2/V3 MoE layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
from safetensors import safe_open

if __package__:
    from ..models.model_config import DeepSeekMoeConfig
    from .deepseek_moe_weights import stack_deepseek_moe_weights
else:  # Narrow aliases installed by dependency-light direct-file tests.
    from deepseek_moe_checkpoint_config import DeepSeekMoeConfig
    from deepseek_moe_checkpoint_weights import stack_deepseek_moe_weights


_STANDARD_INDEX_NAME = "model.safetensors.index.json"


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _checkpoint_root(checkpoints_dir) -> Path:
    try:
        root = Path(checkpoints_dir)
    except TypeError as exc:
        raise TypeError("checkpoints_dir must be a path") from exc
    if not root.is_dir():
        raise FileNotFoundError(
            f"DeepSeek checkpoint directory does not exist: {root}"
        )
    return root.resolve()


def _validated_discovered_file(
    root: Path,
    discovered_path: Path,
    *,
    label: str,
) -> Path:
    if not discovered_path.exists():
        raise FileNotFoundError(f"missing {label}: {discovered_path.name}")
    resolved = discovered_path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"discovered {label} escapes checkpoint directory: "
            f"{discovered_path}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(
            f"discovered {label} is not a regular file: {discovered_path.name}"
        )
    return resolved


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON in {path.name}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid {label} in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} in {path.name} must be a JSON object")
    return value


def _load_checkpoint_config_from_root(root: Path) -> DeepSeekMoeConfig:
    config_path = _validated_discovered_file(
        root,
        root / "config.json",
        label="config.json",
    )
    config_data = _load_json_mapping(config_path, label="config")
    return DeepSeekMoeConfig.from_dict(config_data)


def load_deepseek_moe_checkpoint_config(
    checkpoints_dir,
) -> DeepSeekMoeConfig:
    """Load only ``config.json`` into the audited MoE component config."""

    root = _checkpoint_root(checkpoints_dir)
    return _load_checkpoint_config_from_root(root)


def _required_hf_layer_keys(
    config: DeepSeekMoeConfig,
    layer_index: int,
) -> tuple[str, ...]:
    prefix = f"model.layers.{layer_index}.mlp"
    keys = [f"{prefix}.gate.weight"]
    if config.uses_correction_bias:
        keys.append(f"{prefix}.gate.e_score_correction_bias")
    for expert_id in range(config.num_experts):
        expert = f"{prefix}.experts.{expert_id}"
        keys.extend(
            (
                f"{expert}.gate_proj.weight",
                f"{expert}.up_proj.weight",
                f"{expert}.down_proj.weight",
            )
        )
    shared = f"{prefix}.shared_experts"
    keys.extend(
        (
            f"{shared}.gate_proj.weight",
            f"{shared}.up_proj.weight",
            f"{shared}.down_proj.weight",
        )
    )
    return tuple(keys)


def _validated_shard_path(root: Path, value: Any, *, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"weight_map value for {key} must be a non-empty string")
    posix = PurePosixPath(value.replace("\\", "/"))
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
    ):
        raise ValueError(f"unsafe shard path for {key}: {value!r}")
    if posix.suffix != ".safetensors":
        raise ValueError(
            f"shard path for {key} must end with .safetensors: {value!r}"
        )
    candidate = root.joinpath(*posix.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"shard path escapes checkpoint directory: {value!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"mapped safetensors shard is missing: {value}")
    return candidate


def _single_file_mapping(
    root: Path,
    required_keys: tuple[str, ...],
) -> dict[str, Path]:
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(
            "checkpoint directory contains no safetensors file or index"
        )
    if len(shards) != 1:
        raise ValueError(
            "multiple safetensors files require model.safetensors.index.json"
        )
    shard = _validated_discovered_file(
        root,
        shards[0],
        label="single safetensors file",
    )
    return {key: shard for key in required_keys}


def _indexed_mapping(
    root: Path,
    index_path: Path,
    required_keys: tuple[str, ...],
) -> dict[str, Path]:
    index = _load_json_mapping(index_path, label="safetensors index")
    if "weight_map" not in index:
        raise ValueError("safetensors index is missing weight_map")
    weight_map = index["weight_map"]
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("safetensors index weight_map must be a non-empty object")

    validated: dict[str, Path] = {}
    for key, value in weight_map.items():
        if not isinstance(key, str) or not key:
            raise ValueError("weight_map keys must be non-empty strings")
        validated[key] = _validated_shard_path(root, value, key=key)

    missing = [key for key in required_keys if key not in validated]
    if missing:
        raise KeyError(
            "safetensors index does not map required DeepSeek MoE weight: "
            + missing[0]
        )
    return {key: validated[key] for key in required_keys}


def _layer_file_mapping(
    root: Path,
    required_keys: tuple[str, ...],
) -> dict[str, Path]:
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError("checkpoint directory contains multiple safetensors indexes")
    if not indexes:
        return _single_file_mapping(root, required_keys)
    if indexes[0].name != _STANDARD_INDEX_NAME:
        raise ValueError(
            f"safetensors index must be named {_STANDARD_INDEX_NAME}"
        )
    index_path = _validated_discovered_file(
        root,
        indexes[0],
        label="safetensors index",
    )
    return _indexed_mapping(root, index_path, required_keys)


def _read_required_tensors(
    file_mapping: Mapping[str, Path],
) -> dict[str, torch.Tensor]:
    by_shard: dict[Path, list[str]] = {}
    for key, shard in file_mapping.items():
        by_shard.setdefault(shard, []).append(key)

    tensors: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(
            str(shard),
            framework="pt",
            device="cpu",
        ) as reader:
            available = set(reader.keys())
            for key in keys:
                if key not in available:
                    raise KeyError(
                        f"safetensors shard {shard.name} is missing mapped "
                        f"weight {key}"
                    )
                tensors[key] = reader.get_tensor(key)
    return tensors


def load_deepseek_moe_canonical_layer(
    checkpoints_dir,
    config: DeepSeekMoeConfig,
    layer_index: int,
) -> dict[str, torch.Tensor]:
    """Load and canonicalize one scheduled MoE layer from safetensors."""

    root = _checkpoint_root(checkpoints_dir)
    if type(config) is not DeepSeekMoeConfig:
        raise TypeError(
            "config must be an exact DeepSeekMoeConfig; subclasses are not "
            "accepted"
        )
    if type(layer_index) is not int:
        raise TypeError("layer_index must be an integer")
    file_config = _load_checkpoint_config_from_root(root)
    mismatches = [
        item.name
        for item in fields(DeepSeekMoeConfig)
        if item.init
        if getattr(file_config, item.name) != getattr(config, item.name)
    ]
    if mismatches:
        raise ValueError(
            "provided DeepSeekMoeConfig does not match audited config.json "
            "semantics; differing fields: "
            + ", ".join(mismatches)
        )
    if not 0 <= layer_index < file_config.num_layers:
        raise ValueError(
            f"layer_index={layer_index} must be in [0, {file_config.num_layers})"
        )
    if not file_config.is_moe_layer(layer_index):
        raise ValueError(
            f"layer {layer_index} is not a scheduled DeepSeek MoE layer"
        )

    required_keys = _required_hf_layer_keys(file_config, layer_index)
    file_mapping = _layer_file_mapping(root, required_keys)
    source_state = _read_required_tensors(file_mapping)
    return stack_deepseek_moe_weights(
        source_state,
        moe_layer_indices=(layer_index,),
        num_experts=file_config.num_experts,
        use_correction_bias=file_config.uses_correction_bias,
        consume=False,
    )


__all__ = [
    "load_deepseek_moe_checkpoint_config",
    "load_deepseek_moe_canonical_layer",
]
