"""
Refactored model configuration dataclasses plus unit‑tests.

Usage
-----
pytest -q
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Type, TypeVar, Optional
import json
import math
import os

T = TypeVar("T", bound="BaseConfig")


# ----------------------------------------------------------------------------- #
#                               Utility helpers                                 #
# ----------------------------------------------------------------------------- #

def _apply_aliases(
    raw: Mapping[str, Any], aliases: Mapping[str, str]
) -> dict[str, Any]:
    """Return a shallow‑copied dict with alias keys renamed."""
    out: dict[str, Any] = dict(raw)
    for old, new in aliases.items():
        if old in out:
            out[new] = out.pop(old)
    return out


def _filter_fields(data: Mapping[str, Any], cls: Type) -> dict[str, Any]:
    """Drop keys that are not declared in *cls* dataclass fields."""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


# ----------------------------------------------------------------------------- #
#                            BaseConfig definition                              #
# ----------------------------------------------------------------------------- #
@dataclass
class BaseConfig:
    """Minimal base class providing *from_dict* utility."""

    # subclasses may override
    _ALIASES: Mapping[str, str] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_dict(cls: Type[T], data: Mapping[str, Any]) -> T:
        # apply alias mapping declared on the concrete subclass
        aliased = _apply_aliases(data, getattr(cls, "_ALIASES", {}))
        # only keep valid fields
        return cls(**_filter_fields(aliased, cls))  # type: ignore[arg-type]

    # pretty repr truncated for long sequences
    def __repr__(self) -> str:  # pragma: no cover
        cls = self.__class__.__name__
        parts = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_"))
        return f"{cls}({parts})"


# ----------------------------------------------------------------------------- #
#                               Model configs                                   #
# ----------------------------------------------------------------------------- #
@dataclass
class LlamaConfig(BaseConfig):
    # architecture‑specific
    architectures: list[str] = field(default_factory=lambda: ["LlamaForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    hidden_size: int = 2048
    intermediate_size: Optional[int] = None  # default handled in post‑init
    max_position_embeddings: Optional[int] = None
    mlp_bias: bool = False
    model_type: str = "llama"
    num_heads: int = 32
    num_layers: int = 32
    num_kv_heads: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-5
    rope_scaling: Optional[dict[str, Any]] = None
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    transformers_version: Optional[str] = None
    use_cache: bool = True
    vocab_size: int = 32064
    _name_or_path: Optional[str] = None
    max_batch_size: int = 64
    max_seq_len: int = 2048
    device: str = "cuda"

    # alias mapping used by `from_dict`
    _ALIASES = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "num_key_value_heads": "num_kv_heads",
        "max_length": "max_seq_len",
    }

    # ----------------------------- validation -------------------------------- #
    def __post_init__(self) -> None:
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads

        assert (
            self.hidden_size == self.head_dim * self.num_heads
        ), "hidden_size must equal num_heads × head_dim"


# ---------------------------------------------------------------------------- #
@dataclass
class Qwen2Config(BaseConfig):
    max_batch_size: int = 4
    max_seq_len: int = 2048
    architectures: Optional[list[str]] = None
    attention_dropout: float = 0.0
    bos_token_id: Optional[int] = 151643
    eos_token_id: Optional[int] = 151645
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    hidden_size: int = 1536
    intermediate_size: Optional[int] = None
    max_position_embeddings: int = 32768
    mlp_bias: bool = False
    model_type: str = "qwen2"
    num_heads: int = 12
    num_layers: int = 28
    num_kv_heads: Optional[int] = 2
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[dict[str, Any]] = None
    rope_theta: float = 1_000_000.0
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.43.1"
    use_cache: bool = True
    vocab_size: int = 151_936
    tie_word_embeddings: bool = False
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 21
    device: str = "cuda"
    head_dim: Optional[int] = None

    _ALIASES = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "num_key_value_heads": "num_kv_heads",
        "max_length": "max_seq_len",
    }

    def __post_init__(self) -> None:
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads


# ---------------------------------------------------------------------------- #
@dataclass
class Qwen3Config(BaseConfig):
    vocab_size: int = 151_936
    hidden_size: int = 4096
    intermediate_size: Optional[int] = None
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: Optional[int] = 32
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 32_768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict[str, Any]] = None
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    attention_dropout: float = 0.0
    device: str = "cuda"
    model_type: str = "qwen3"
    max_seq_len: int = 2048
    max_batch_size: int = 64
    page_size: int = 0

    _ALIASES = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "num_key_value_heads": "num_kv_heads",
        "max_length": "max_seq_len",
    }

    def __post_init__(self) -> None:
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads


# ---------------------------------------------------------------------------- #
@dataclass
class Qwen3MoeConfig(Qwen3Config):
    architectures: list[str] = field(
        default_factory=lambda: ["Qwen3MoeForCausalLM"]
    )
    model_type: str = "qwen3_moe"
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = field(default_factory=list)
    norm_topk_prob: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001

    def validate_tensor_parallel(self, world_size: int) -> None:
        if world_size < 1:
            raise ValueError("tensor parallel world_size must be positive")
        for name, value in (
            ("num_heads", self.num_heads),
            ("num_kv_heads", self.num_kv_heads),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("vocab_size", self.vocab_size),
        ):
            if value is None or value % world_size != 0:
                raise ValueError(
                    f"{name}={value} must be divisible by tensor parallel "
                    f"world_size={world_size}"
                )


# ---------------------------------------------------------------------------- #
@dataclass
class DeepSeekMoeConfig(BaseConfig):
    """Validated DeepSeek-V2/V3 MoE component configuration.

    This intentionally excludes attention, MLA, vocabulary, and full-model
    registration fields.  It only describes the audited routed-plus-shared
    MoE component.
    """

    architectures: list[str] = field(
        default_factory=lambda: ["DeepseekV2ForCausalLM"]
    )
    model_type: str = "deepseek_v2"
    hidden_size: int = 2048
    num_layers: int = 27
    intermediate_size: int = 10944
    num_experts: int = 64
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 1408
    num_shared_experts: int = 2
    score_func: str = "softmax"
    topk_method: str = "group_limited_greedy"
    num_groups: int = 1
    topk_groups: int = 1
    norm_topk_prob: bool = False
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    torch_dtype: str = "bfloat16"

    _ALIASES = {
        "num_hidden_layers": "num_layers",
        "n_routed_experts": "num_experts",
        "n_shared_experts": "num_shared_experts",
        "scoring_func": "score_func",
        "n_group": "num_groups",
        "topk_group": "topk_groups",
    }
    _UNSUPPORTED_ROUTING_FIELDS = frozenset(
        {
            "static_hash",
            "static_hash_routing",
            "hash_routing",
            "hash_router",
        }
    )
    _OFFICIAL_ARCHITECTURES = {
        "deepseek_v2": ["DeepseekV2ForCausalLM"],
        "deepseek_v3": ["DeepseekV3ForCausalLM"],
    }
    _REQUIRED_SOURCE_FIELDS = (
        "architectures",
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "topk_method",
        "norm_topk_prob",
        "routed_scaling_factor",
        "first_k_dense_replace",
        "moe_layer_freq",
        "torch_dtype",
    )

    @classmethod
    def _validate_source_identity(
        cls,
        model_type: Any,
        architectures: Any,
    ) -> None:
        if model_type == "deepseek_v4":
            raise ValueError("DeepSeek-V4 MoE routing is not supported")
        if model_type not in cls._OFFICIAL_ARCHITECTURES:
            raise ValueError(
                "model_type must be 'deepseek_v2' or 'deepseek_v3'"
            )
        expected = cls._OFFICIAL_ARCHITECTURES[model_type]
        if architectures != expected:
            raise ValueError(
                f"architectures for {model_type} must exactly equal {expected!r}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeepSeekMoeConfig":
        if not isinstance(data, Mapping):
            raise TypeError("DeepSeek MoE config data must be a mapping")
        for required_source_field in cls._REQUIRED_SOURCE_FIELDS:
            if required_source_field not in data:
                raise ValueError(
                    f"{required_source_field} is required for DeepSeek MoE "
                    "source validation"
                )
        for alias, canonical in cls._ALIASES.items():
            has_alias = alias in data
            has_canonical = canonical in data
            if not has_alias and not has_canonical:
                raise ValueError(
                    "DeepSeek MoE source config requires at least one of "
                    f"{alias} or {canonical}"
                )
            if (
                has_alias
                and has_canonical
                and data[alias] != data[canonical]
            ):
                raise ValueError(
                    f"conflicting DeepSeek MoE source fields {alias} and "
                    f"{canonical}"
                )
        cls._validate_source_identity(
            data["model_type"],
            data["architectures"],
        )
        unsupported = sorted(cls._UNSUPPORTED_ROUTING_FIELDS.intersection(data))
        if unsupported:
            raise ValueError(
                "unsupported DeepSeek-V4/hash routing config fields: "
                + ", ".join(unsupported)
            )
        model_type = data["model_type"]

        normalized = dict(data)
        if (
            model_type == "deepseek_v2"
            and normalized.get("topk_method") == "greedy"
        ):
            # The frozen V2-Lite HF config uses ``greedy`` for the audited
            # one-group policy.  Production names that policy explicitly.
            normalized["topk_method"] = "group_limited_greedy"
        return super().from_dict(normalized)

    def __post_init__(self) -> None:
        type(self)._validate_source_identity(
            self.model_type,
            self.architectures,
        )

        positive_integer_fields = (
            "hidden_size",
            "num_layers",
            "intermediate_size",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "num_shared_experts",
            "num_groups",
            "topk_groups",
            "moe_layer_freq",
        )
        for name in positive_integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.first_k_dense_replace) is not int
            or self.first_k_dense_replace < 0
            or self.first_k_dense_replace >= self.num_layers
        ):
            raise ValueError(
                "first_k_dense_replace must be an integer in [0, num_layers)"
            )
        if type(self.norm_topk_prob) is not bool:
            raise TypeError("norm_topk_prob must be bool")
        if not isinstance(self.torch_dtype, str) or not self.torch_dtype:
            raise TypeError("torch_dtype must be a non-empty string")
        if (
            isinstance(self.routed_scaling_factor, bool)
            or not isinstance(self.routed_scaling_factor, (int, float))
            or not math.isfinite(float(self.routed_scaling_factor))
            or self.routed_scaling_factor <= 0
        ):
            raise ValueError(
                "routed_scaling_factor must be finite and positive"
            )

        expected_policy = {
            "deepseek_v2": ("softmax", "group_limited_greedy"),
            "deepseek_v3": ("sigmoid", "noaux_tc"),
        }[self.model_type]
        if (self.score_func, self.topk_method) != expected_policy:
            raise ValueError(
                f"{self.model_type} requires score_func={expected_policy[0]!r} "
                f"and topk_method={expected_policy[1]!r}"
            )
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok must not exceed num_experts")
        if self.num_experts % self.num_groups:
            raise ValueError("num_experts must be divisible by num_groups")
        if self.topk_groups > self.num_groups:
            raise ValueError("topk_groups must not exceed num_groups")
        experts_per_group = self.num_experts // self.num_groups
        if self.num_experts_per_tok > self.topk_groups * experts_per_group:
            raise ValueError(
                "num_experts_per_tok exceeds selected-group capacity"
            )
        if self.topk_method == "noaux_tc" and experts_per_group < 2:
            raise ValueError("noaux_tc requires at least two experts per group")
        if not self.moe_layer_indices():
            raise ValueError("layer schedule must contain at least one MoE layer")

    @property
    def shared_intermediate_size(self) -> int:
        return self.num_shared_experts * self.moe_intermediate_size

    @property
    def uses_correction_bias(self) -> bool:
        return (
            self.score_func == "sigmoid" and self.topk_method == "noaux_tc"
        )

    def is_moe_layer(self, layer_index: int) -> bool:
        return (
            type(layer_index) is int
            and 0 <= layer_index < self.num_layers
            and layer_index >= self.first_k_dense_replace
            and layer_index % self.moe_layer_freq == 0
        )

    def moe_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            layer_index
            for layer_index in range(self.num_layers)
            if self.is_moe_layer(layer_index)
        )

    def validate_moe_parallel(self, world_size: int, mode: str) -> None:
        if type(world_size) is not int or world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        if mode not in {"tp", "ep"}:
            raise ValueError("mode must be 'tp' or 'ep'")
        if mode == "tp":
            for name, value in (
                ("moe_intermediate_size", self.moe_intermediate_size),
                ("shared_intermediate_size", self.shared_intermediate_size),
            ):
                if value % world_size:
                    raise ValueError(
                        f"{name}={value} must be divisible by tensor "
                        f"parallel world_size={world_size}"
                    )
        elif self.num_experts % world_size:
            raise ValueError(
                f"num_experts={self.num_experts} must be divisible by "
                f"expert parallel world_size={world_size}"
            )


# ---------------------------------------------------------------------------- #
@dataclass
class VisionConfig(BaseConfig):
    hidden_size: int = 768
    image_size: int = 224
    intermediate_size: int = 3072
    model_type: str = "clip_vision_model"
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    patch_size: int = 16
    projection_dim: int = 768
    vocab_size: int = 1000


# ---------------------------------------------------------------------------- #
@dataclass
class LlavaConfig(BaseConfig):
    architectures: list[str]
    ignore_index: int
    image_token_index: int
    model_type: str
    pad_token_id: int
    projector_hidden_act: str
    text_config: LlamaConfig
    tie_word_embeddings: bool
    torch_dtype: str
    vision_config: VisionConfig
    vision_feature_layer: int
    vision_feature_select_strategy: str
    vocab_size: int
    image_seq_length: int = 576
    max_batch_size: int = 64
    max_seq_len: int = 2048
    device: str = "cuda"

    # no aliases ‑ relies on static *from_dict* below.

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "LlavaConfig":
        text_cfg = LlamaConfig.from_dict(data.get("text_config", {}))
        vision_cfg = VisionConfig.from_dict(data.get("vision_config", {}))

        # retain only valid primitive fields *excluding* the nested configs
        kwargs = _filter_fields(data, LlavaConfig)
        kwargs.pop("text_config", None)
        kwargs.pop("vision_config", None)

        # supply defaults for mandatory fields if absent
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("torch_dtype", "float16")

        return LlavaConfig(text_config=text_cfg, vision_config=vision_cfg, **kwargs)

    @classmethod
    def from_json(cls, json_path: os.PathLike | str) -> "LlavaConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------- #
@dataclass
class Qwen3VLVisionConfig(BaseConfig):
    """Vision encoder configuration for Qwen3-VL."""

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)
    initializer_range: float = 0.02
    model_type: str = "qwen3_vl_vision"

    _ALIASES = {
        "num_hidden_layers": "depth",
    }


# ---------------------------------------------------------------------------- #
@dataclass
class Qwen3VLConfig(BaseConfig):
    """Top-level configuration for Qwen3-VL multimodal model."""

    architectures: list[str] = field(default_factory=lambda: ["Qwen3VLForConditionalGeneration"])
    model_type: str = "qwen3_vl"
    text_config: Qwen3Config = field(default_factory=Qwen3Config)
    vision_config: Qwen3VLVisionConfig = field(default_factory=Qwen3VLVisionConfig)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    vocab_size: int = 151936
    max_batch_size: int = 64
    max_seq_len: int = 2048
    device: str = "cuda"

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "Qwen3VLConfig":
        text_cfg = Qwen3Config.from_dict(data.get("text_config", {}))
        vision_cfg = Qwen3VLVisionConfig.from_dict(data.get("vision_config", {}))

        kwargs = _filter_fields(data, Qwen3VLConfig)
        kwargs.pop("text_config", None)
        kwargs.pop("vision_config", None)

        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("torch_dtype", "bfloat16")

        return Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg, **kwargs)

    @classmethod
    def from_json(cls, json_path: os.PathLike | str) -> "Qwen3VLConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ----------------------------------------------------------------------------- #
#                                    Tests                                      #
# ----------------------------------------------------------------------------- #

def _make_fake_json(tmp_path) -> str:
    path = tmp_path / "llava.json"
    sample = {
        "architectures": ["LlavaForConditionalGeneration"],
        "ignore_index": -1,
        "image_token_index": 32000,
        "model_type": "llava",
        "pad_token_id": 32001,
        "projector_hidden_act": "gelu",
        "text_config": {"hidden_size": 1024, "num_attention_heads": 8},
        "vision_config": {"hidden_size": 384},
        "vision_feature_layer": -2,
        "vision_feature_select_strategy": "default",
        "vocab_size": 32064,
    }
    path.write_text(json.dumps(sample))
    return str(path)


def test_llama_default():
    cfg = LlamaConfig()
    assert cfg.head_dim == cfg.hidden_size // cfg.num_heads
    assert cfg.intermediate_size == cfg.hidden_size * 4


def test_llama_from_alias():
    cfg = LlamaConfig.from_dict({"num_attention_heads": 16, "hidden_size": 1024})
    assert cfg.num_heads == 16
    assert cfg.head_dim == 64


def test_qwen2_sliding_window_disabled():
    cfg = Qwen2Config(use_sliding_window=False)
    assert cfg.sliding_window is None


def test_qwen3_valid_head_dim():
    cfg = Qwen3Config(head_dim=None, hidden_size=2048, num_heads=16)
    assert cfg.head_dim == 128


def test_llava_roundtrip(tmp_path):
    json_path = _make_fake_json(tmp_path)
    cfg = LlavaConfig.from_json(json_path)
    assert cfg.text_config.hidden_size == 1024
    assert cfg.vision_config.hidden_size == 384


def test_qwen3vl_vision_config_defaults():
    cfg = Qwen3VLVisionConfig()
    assert cfg.depth == 27
    assert cfg.hidden_size == 1152
    assert cfg.out_hidden_size == 3584
    assert cfg.spatial_merge_size == 2
    assert cfg.deepstack_visual_indexes == (8, 16, 24)


def test_qwen3vl_config_from_dict():
    data = {
        "model_type": "qwen3_vl",
        "text_config": {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
        },
        "vision_config": {
            "depth": 27,
            "hidden_size": 1152,
            "out_hidden_size": 3584,
        },
        "vocab_size": 151936,
    }
    cfg = Qwen3VLConfig.from_dict(data)
    assert cfg.text_config.hidden_size == 4096
    assert cfg.text_config.num_heads == 32
    assert cfg.vision_config.depth == 27
    assert cfg.vision_config.out_hidden_size == 3584
    assert cfg.image_token_id == 151655


def test_qwen3vl_config_defaults():
    cfg = Qwen3VLConfig()
    assert cfg.model_type == "qwen3_vl"
    assert cfg.image_token_id == 151655
    assert cfg.vision_start_token_id == 151652
    assert cfg.vision_end_token_id == 151653
    assert isinstance(cfg.text_config, Qwen3Config)
    assert isinstance(cfg.vision_config, Qwen3VLVisionConfig)
