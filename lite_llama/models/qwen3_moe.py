"""Qwen3 MoE model built on the existing Qwen3 execution path."""

from __future__ import annotations

from .model_config import Qwen3MoeConfig
from .moe import Qwen3SparseMoeBlock
from .qwen3 import FusedMLP, Qwen3DecoderLayer, Qwen3Model
from ..executor.tp_utils import TPConfig


class Qwen3MoeDecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_index: int,
        tp_config: TPConfig = None,
    ) -> None:
        super().__init__(config, tp_config)
        is_sparse = (
            layer_index not in config.mlp_only_layers
            and config.num_experts > 0
            and (layer_index + 1) % config.decoder_sparse_step == 0
        )
        if is_sparse:
            self.mlp = Qwen3SparseMoeBlock(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                intermediate_size=config.moe_intermediate_size,
                norm_topk_prob=config.norm_topk_prob,
                tp_config=tp_config,
            )
        else:
            self.mlp = FusedMLP(config, tp_config)


class Qwen3MoeModel(Qwen3Model):
    """Qwen3 attention/KV path with MoE decoder feed-forward blocks."""

    def __init__(
        self, config: Qwen3MoeConfig, tp_config: TPConfig = None
    ) -> None:
        config.validate_tensor_parallel(
            getattr(tp_config, "world_size", 1)
        )
        super().__init__(
            config,
            tp_config=tp_config,
            decoder_layer_factory=lambda layer_index: Qwen3MoeDecoderLayer(
                config, layer_index, tp_config
            ),
        )
