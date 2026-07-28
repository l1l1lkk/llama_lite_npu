import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .model_config import Qwen3Config
from .RotaryEmbedding import Qwen3RotaryEmbedding
from ..kernels import *
from ..executor.tp_utils import TPConfig, tp_all_reduce


class Attention(nn.Module):
    def __init__(self, num_q_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = num_q_heads * head_dim

    def context_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
        qk_scale=None,
    ) -> torch.Tensor:
        combined_kv = torch.cat([xk, xv], dim=-2)
        update_kv_buffer(
            combined_kv, atten_info.cur_select_index, atten_info.kv_buffer[layer_index]
        )
        if getattr(atten_info, "is_paged_chunk_prefill", False):
            output = paged_chunk_flash_attention(
                xq,
                atten_info.kv_buffer[layer_index][:, : self.num_kv_heads, :],
                atten_info.kv_buffer[layer_index][:, self.num_kv_heads :, :],
                qk_scale,
                atten_info.b_req_tokens_table,
                atten_info.b_req_idx,
                atten_info.b_start_loc,
                atten_info.chunk_context_len,
                atten_info.chunk_q_seq_len,
                atten_info.max_actual_q_seq_len,
            )
        else:
            output = flash_attention2_no_pad(
                xq, xk, xv, qk_scale,
                atten_info.b_start_loc, atten_info.b_seq_len, atten_info.max_actual_seq_len,
            )
        return output

    def token_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index: int,
        qk_scale=None,
    ) -> torch.Tensor:
        combined_kv = torch.cat([xk, xv], dim=-2)
        update_kv_buffer(
            combined_kv, atten_info.cur_select_index, atten_info.kv_buffer[layer_index]
        )
        output = flash_decoding(
            xq,
            atten_info.kv_buffer[layer_index][:, : self.num_kv_heads, :],
            atten_info.kv_buffer[layer_index][:, self.num_kv_heads :, :],
            qk_scale,
            atten_info.b_req_tokens_table,
            atten_info.b_seq_len,
            atten_info.max_actual_seq_len,
        )
        return output


class Qwen3Attention(nn.Module):
    """TP-aware Qwen3 attention with QK normalization and GQA."""

    def __init__(
        self, config: Qwen3Config, tp_config: TPConfig = None
    ) -> None:
        super().__init__()
        self.tp = tp_config or TPConfig()
        tp_w = self.tp.world_size

        self.config = config
        self.num_heads = config.num_heads // tp_w
        self.num_kv_heads = config.num_kv_heads // tp_w
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.rmsnorm_eps = config.rms_norm_eps

        self.q_proj_weight = nn.Parameter(
            torch.rand(self.num_heads * self.head_dim, self.hidden_size, dtype=torch.float16)
        )
        self.kv_proj_weight = nn.Parameter(
            torch.rand(2 * self.num_kv_heads * self.head_dim, self.hidden_size, dtype=torch.float16)
        )
        self.o_proj_weight = nn.Parameter(
            torch.rand(self.hidden_size, self.num_heads * self.head_dim, dtype=torch.float16)
        )
        self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.float16))
        self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.float16))

        self.attn = Attention(self.num_heads, self.num_kv_heads, self.head_dim)

    def _get_qkv(
        self,
        x: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(-1, self.hidden_size)

        xq = F.linear(x, self.q_proj_weight.data)
        xkv = F.linear(x, self.kv_proj_weight.data)
        xk, xv = torch.split(xkv, self.num_kv_heads * self.head_dim, dim=-1)

        xq = xq.view(batch_size * seq_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size * seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size * seq_len, self.num_kv_heads, self.head_dim)

        xq, _ = skip_rmsnorm(xq, None, self.q_norm_weight.data, self.rmsnorm_eps)
        xk, _ = skip_rmsnorm(xk, None, self.k_norm_weight.data, self.rmsnorm_eps)

        cos, sin = position_embeddings
        xq, xk = rope_emb_forward(xq, xk, cos, sin, batch_size, seq_len)
        return xq, xk, xv

    def forward(
        self,
        x: torch.Tensor,
        atten_info,
        layer_index: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        qk_scale=None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self._get_qkv(x, position_embeddings)

        if seq_len > 1:
            attn_output = self.attn.context_forward(
                xq, xk, xv, atten_info, layer_index, qk_scale,
            )
            attn_output = attn_output.view(
                batch_size, seq_len, self.num_heads * self.head_dim
            )
        else:
            attn_output = self.attn.token_forward(
                xq, xk, xv, atten_info, layer_index, qk_scale,
            )
            attn_output = attn_output.view(
                batch_size, seq_len, self.num_heads * self.head_dim
            )

        output = F.linear(attn_output, self.o_proj_weight.data)
        # TP: all-reduce partial results from row-sharded O projection
        output = tp_all_reduce(output)
        return output


class FusedMLP(nn.Module):
    """TP-aware SwiGLU FFN."""

    def __init__(self, config: Qwen3Config, tp_config: TPConfig = None):
        super().__init__()
        self.tp = tp_config or TPConfig()
        tp_w = self.tp.world_size

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size // tp_w

        self.gate_up_proj = nn.Linear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=False,
            dtype=torch.float16,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False, dtype=torch.float16
        )

    def forward(self, x):
        h = swiglu_packed_forward(self.gate_up_proj(x))
        out = self.down_proj(h)
        # TP: all-reduce partial results from row-sharded down projection
        out = tp_all_reduce(out)
        return out


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, tp_config: TPConfig = None):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.rmsnorm_eps = config.rms_norm_eps

        self.input_layernorm_weight = nn.Parameter(
            torch.ones(self.hidden_size, dtype=torch.float16)
        )
        self.post_attention_layernorm_weight = nn.Parameter(
            torch.ones(self.hidden_size, dtype=torch.float16)
        )

        self.self_attn = Qwen3Attention(config, tp_config)
        self.mlp = FusedMLP(config, tp_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        atten_info,
        layer_index: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        qk_scale=None,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states, residual = skip_rmsnorm(
            hidden_states, residual, self.input_layernorm_weight.data, self.rmsnorm_eps
        )
        hidden_states = self.self_attn(
            hidden_states, atten_info, layer_index, position_embeddings, qk_scale
        )
        hidden_states, residual = skip_rmsnorm(
            hidden_states, residual, self.post_attention_layernorm_weight.data, self.rmsnorm_eps,
        )
        hidden_states = self.mlp.forward(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        tp_config: TPConfig = None,
        decoder_layer_factory=None,
    ):
        super().__init__()
        self.tp = tp_config or TPConfig()
        tp_w = self.tp.world_size

        assert config.vocab_size != -1, "Vocab size must be set"
        self.rmsnorm_eps = config.rms_norm_eps

        hidden_size = config.hidden_size
        vocab_size = config.vocab_size // tp_w
        num_layers = config.num_layers
        head_dim = (
            config.head_dim if config.head_dim is not None
            else config.hidden_size // config.num_heads
        )

        self.qk_scale = 1.0 / (head_dim**0.5)

        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        # Embedding: replicated (each rank has full embedding table)
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size, dtype=torch.float16)
        self.norm_weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))

        # lm_head: column-sharded along vocab dim
        self.lm_head_weight = nn.Parameter(
            torch.rand(vocab_size, hidden_size, dtype=torch.float16)
        )

        if decoder_layer_factory is None:
            decoder_layer_factory = lambda _layer_index: Qwen3DecoderLayer(
                config, tp_config
            )
        self.layers = nn.ModuleList(
            [decoder_layer_factory(i) for i in range(num_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info,
        inputs_embeds: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len = input_ids.shape
        residual = None

        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.get_input_embeddings(input_ids)

        if seq_len > 1:
            qk_scale = self.qk_scale * 1.4426950408889634
        else:
            qk_scale = self.qk_scale

        position_embeddings = self.rotary_emb(h, position_ids)

        for i, layer in enumerate(self.layers):
            h, residual = layer(
                h, atten_info, i, position_embeddings, qk_scale, residual
            )
            if deepstack_visual_embeds is not None and i < len(deepstack_visual_embeds):
                vis_embeds = deepstack_visual_embeds[i].to(h.device, h.dtype)
                h = h.clone()
                h[visual_pos_masks, :] = h[visual_pos_masks, :] + vis_embeds

        h, _ = skip_rmsnorm(h, residual, self.norm_weight.data, self.rmsnorm_eps)

        # TP: keep the LM-head vocabulary shard local. Sampling performs only
        # the collectives required by the selected strategy.
        output = F.linear(h, self.lm_head_weight.data)
        return output

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
