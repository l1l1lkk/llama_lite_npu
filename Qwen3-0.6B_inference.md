# Qwen3-0.6B Inference Walkthrough (lite_llama)

This document explains how inference is implemented for Qwen3-0.6B in this repo, with per-layer behavior, tensor shapes, and parameter details. Exact numeric values (hidden_size, num_layers, num_heads, etc.) come from your local config.json; this doc uses symbols to avoid mismatches.

---

## 0. Get the real Qwen3-0.6B config

All model hyperparameters are read from checkpoints_dir/config.json. Print them first:

```python
# tools/print_qwen3_cfg.py
import json
from pathlib import Path

ckpt = Path("path/to/Qwen3-0.6B")
cfg = json.loads((ckpt / "config.json").read_text())
keys = [
    "model_type", "vocab_size", "hidden_size", "intermediate_size",
    "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "head_dim", "rms_norm_eps", "rope_theta", "max_position_embeddings",
    "max_length"
]
print({k: cfg.get(k) for k in keys})
```

Symbols used below (all from config.json):

- V = vocab_size
- H = hidden_size
- I = intermediate_size (if missing, code uses H * 4)
- L = num_layers
- N = num_heads
- N_kv = num_kv_heads (if missing, equals N)
- D = head_dim (if missing, H / N)
- S_max = max_seq_len
- eps = rms_norm_eps
- theta = rope_theta

---

## 1. Entry points and high-level flow

Key files:

- cli.py builds the generator and runs interactive inference.
- lite_llama/generate_stream.py handles tokenization + prefill + decode streaming.
- lite_llama/executor/model_executor.py builds the model and KV cache manager.
- lite_llama/models/qwen3.py defines the Qwen3 layers and forward pass.

Main call chain (simplified):

```
cli.py
  -> GenerateStreamText(...)
      -> ModelExecutor.build(...)
          -> _load_model_config(config.json)
          -> _load_model_weight(*.pth) -> model.half()
          -> KVCacheMemoryManager / ReqTokensManager / AttentionInfo
      -> generate_stream(...)
          -> prefill_alloc_kv_cache(...)
          -> loop decode (token by token)
```

---

## 2. Core runtime data structures

### 2.1 AttentionInfo (lite_llama/executor/executor_struct.py)

Carries per-request KV cache indexing and attention lengths into each layer:

- kv_buffer: list[Tensor] of length L
  Each layer has KV cache shaped [T_max, 2 * N_kv, D]
  T_max = max_num_tokens from KVCacheMemoryManager
- cur_select_index: indices to write for this step
  shape [B * S] (prefill) or [B] (decode)
- b_req_tokens_table: [max_request_num, S_max]
  maps (request, position) -> KV index
- b_start_loc: [B], starting index for each request in prefill
- b_seq_len: [B], current valid length per request
- max_actual_seq_len: max length inside the batch

### 2.2 KVCacheMemoryManager (lite_llama/executor/mem_manager.py)

- Pre-allocates KV cache:
  gpu_kv_buffer[layer] shape [T_max, 2 * N_kv, D]
  T_max = gpu_num_blocks * block_size (block_size default 1)
- alloc_kvcache_index(need_size) returns cur_select_index
- release_ref decrements ref counts

### 2.3 ReqTokensManager (lite_llama/executor/req_tokens_manager.py)

- b_req_tokens_table: [max_request_num, S_max]

---

## 3. Inference flow: prefill vs decode

### 3.1 Prefill (seq_len > 1)

Entry: GenerateStreamText.generate_stream(...)

Steps:

1) Build tokens: shape [B, T_total]
2) prefill_alloc_kv_cache(max_prompt_len, ...)
   - allocate KV indices for B x prompt_len
   - update b_start_loc / b_seq_len / b_req_tokens_table
3) Forward: model_executor.forward(input_ids, position_ids)
4) Sample from logits at the last position

### 3.2 Decode (seq_len == 1)

Each step generates one token:

1) decode_alloc_kv_cache(B): append KV index for each request
2) Feed only the latest token; read history from KV cache
3) Sample next token

---

## 4. Qwen3 model details (per-layer)

Implementation: lite_llama/models/qwen3.py

Overall structure:

```
Embedding -> [L x DecoderLayer] -> RMSNorm -> LM Head
```

### 4.1 Embedding

- embed_tokens.weight shape: [V, H]
- output h shape: [B, S, H]

### 4.2 DecoderLayer (repeated L times)

Qwen3DecoderLayer.forward() runs in this order:

#### (1) Skip-RMSNorm + Residual

Called as skip_rmsnorm(hidden_states, residual, weight):

```
if residual is None:
    residual = x
    y = RMSNorm(x)
else:
    residual = x + residual
    y = RMSNorm(residual)
```

Parameters:
- input_layernorm_weight: [H]
- eps = rms_norm_eps

Shapes: x [B, S, H] -> y [B, S, H]

#### (2) Self-Attention (Qwen3Attention)

Inside Qwen3Attention._get_qkv():

Q/K/V projections:

- q_proj_weight: [H, N * D]
- kv_proj_weight: [H, 2 * N_kv * D]
  - split into K: [H, N_kv * D]
  - V: [H, N_kv * D]

Reshape after projection:

- Q: [B*S, N, D]
- K: [B*S, N_kv, D]
- V: [B*S, N_kv, D]

Q/K extra RMSNorm:

```
Q = RMSNorm(Q, q_norm_weight, eps)
K = RMSNorm(K, k_norm_weight, eps)
```

- q_norm_weight: [D]
- k_norm_weight: [D]

RoPE (rope_emb_forward):

- cos/sin from Qwen3RotaryEmbedding
- cos/sin shape: [B, S, D]
- output Q/K remain [B*S, N or N_kv, D]

Attention compute:

- Prefill (seq_len > 1):
  flash_attention2_no_pad(Q, K, V, qk_scale, b_start_loc, b_seq_len, max_actual_seq_len)
- Decode (seq_len == 1):
  flash_decoding(Q, KV_cache, ...)

qk_scale = 1/sqrt(D); during prefill it is multiplied by 1.442695... in Qwen3Model.forward.

KV cache update:

```
combined_kv = cat([K, V], dim=-2)  # [B*S, 2*N_kv, D]
update_kv_buffer(combined_kv, cur_select_index, kv_buffer[layer])
```

kv_buffer[layer] shape: [T_max, 2*N_kv, D]

Output projection:

```
attn_out = F.linear(attn_output, o_proj_weight)
```

- o_proj_weight: [N * D, H]
- output shape: [B, S, H]

#### (3) Skip-RMSNorm + Residual (post-attention)

```
residual = residual + attn_out
attn_norm = RMSNorm(residual)
```

- post_attention_layernorm_weight: [H]

#### (4) MLP (FusedMLP)

SwiGLU path:

```
gate = gate_proj(attn_norm)  # [B,S,I]
up   = up_proj(attn_norm)    # [B,S,I]
mid  = SiLU(gate) * up       # swiglu_forward
out  = down_proj(mid)        # [B,S,H]
```

Weights:

- gate_proj.weight: [I, H]
- up_proj.weight: [I, H]
- down_proj.weight: [H, I]

Output shape: [B, S, H]

Layer returns (hidden_states, residual).

---

## 5. Final output

### 5.1 Final RMSNorm

```
h, _ = skip_rmsnorm(h, residual, norm_weight, eps)
```

- norm_weight: [H]

### 5.2 LM Head

```
logits = F.linear(h, lm_head_weight)
```

- lm_head_weight: [V, H]
- logits shape: [B, S, V]

---

## 6. Sampling (GenerateStreamText)

- Uses logits at the last position: logits[:, -1]
- If temperature > 0:
  - softmax
  - top-p sampling (sample_top_p)
- If temperature == 0: argmax

Default params:

- temperature = 0.6
- top_p = 0.9
- max_gen_len defaults to max_seq_len - 1

---

## 7. File map (suggested reading order)

- cli.py
- lite_llama/generate_stream.py
- lite_llama/executor/model_executor.py
- lite_llama/executor/mem_manager.py
- lite_llama/executor/req_tokens_manager.py
- lite_llama/models/qwen3.py
- lite_llama/models/RotaryEmbedding.py
- lite_llama/kernels/rope_emb.py
- lite_llama/kernels/flashattention2_nopad.py
- lite_llama/kernels/flashdecoding.py
- lite_llama/kernels/skip_rmsnorm.py
- lite_llama/kernels/swiglu.py

---

## 8. Qwen3-0.6B parameter table (fill from config.json)

```
model_type           = qwen3
V (vocab_size)       = ?
H (hidden_size)      = ?
I (intermediate)     = ?
L (num_layers)       = ?
N (num_heads)        = ?
N_kv (num_kv_heads)  = ?
D (head_dim)         = ?
S_max (max_seq_len)  = ?
rope_theta           = ?
rms_norm_eps         = ?
```

Replace the symbols in the sections above with your concrete numbers to get a fully numeric, per-layer description.
