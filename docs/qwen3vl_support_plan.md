# Qwen3VL Support Implementation Plan

## Overview

This document describes the plan to add Qwen3-VL (Vision-Language) model support to the lite_llama inference framework.

Qwen3VL is a multimodal model combining a custom Vision Transformer (ViT) encoder with a Qwen3-based language model, featuring M-RoPE (Multimodal Rotary Position Embedding) and DeepStack visual feature injection.

## Architecture Comparison: Llava vs Qwen3VL

| Component | Llava (existing) | Qwen3VL (to add) |
|---|---|---|
| Vision Encoder | CLIP ViT, 12 layers, standard Attention | Custom ViT, 27 layers, Conv3D PatchEmbed, Attention with RoPE + cu_seqlens |
| Vision Norm | LayerNorm | LayerNorm |
| Vision Output | Single hidden_states[layer] | Last + DeepStack (layers 8, 16, 24) |
| Projector | 2-layer MLP (Linear→GELU→Linear) | PatchMerger (spatial merge → Norm → Linear→GELU→Linear) |
| Text Model | LlamaModel (no QK norm) | Qwen3-like (with QK norm) |
| Position Encoding | Standard 1D RoPE | M-RoPE: 3D positions (T, H, W) with interleaved frequency pattern |
| Vision-Text Fusion | merge_input_ids_with_image_features | masked_scatter + DeepStack injection |
| DeepStack | None | Inject visual features from ViT layers 8/16/24 into LLM layers 0/1/2 |

## Implementation Phases

### Phase 1: Configuration (`models/model_config.py`)

Add three new dataclasses:
- `Qwen3VLVisionConfig`: Vision encoder configuration
  - depth=27, hidden_size=1152, num_heads=16
  - intermediate_size=4304, out_hidden_size=3584
  - patch_size=16, spatial_merge_size=2, temporal_patch_size=2
  - num_position_embeddings=2304, deepstack_visual_indexes=(8, 16, 24)
- `Qwen3VLTextConfig`: Text model configuration (Qwen3 compatible)
- `Qwen3VLConfig`: Top-level multimodal config
  - image_token_id=151655, video_token_id=151656
  - vision_start_token_id=151652, vision_end_token_id=151653
  - Nested vision_config and text_config

### Phase 2: M-RoPE (`models/RotaryEmbedding.py`)

Add `Qwen3VLTextRotaryEmbedding`:
- Accepts 3D position_ids of shape (3, batch, seq_len)
- Three dimensions represent temporal, height, width positions
- `apply_interleaved_mrope()`: Interleaves frequencies as [THWTHWTHW...TT]
- `mrope_section` parameter (default [24, 20, 20]) controls dimension ratios
- `compute_3d_position_ids()`: Generates 3D positions from text + vision grid_thw

### Phase 3: Vision Encoder (`models/qwen3vl_vision.py`) — NEW FILE

Components:
- `Qwen3VLVisionMLP`: GELU-activated FFN for ViT blocks
- `Qwen3VLVisionPatchEmbed`: Conv3D patch embedding (supports video temporal patches)
- `Qwen3VLVisionRotaryEmbedding`: RoPE for vision attention
- `Qwen3VLVisionPatchMerger`: Spatial merge (2×2) + projection to LLM hidden_size
- `Qwen3VLVisionAttention`: Self-attention with RoPE, uses FlashAttention with cu_seqlens for variable-length sequences
- `Qwen3VLVisionBlock`: ViT block (LayerNorm → Attention → LayerNorm → MLP)
- `Qwen3VLVisionModel`: Full vision encoder
  - `fast_pos_embed_interpolate()`: Bilinear interpolation of 2D position embeddings
  - `rot_pos_emb()`: Generate RoPE frequencies from grid_thw
  - Outputs: `(last_hidden_state, pooler_output, deepstack_features)`

### Phase 4: Qwen3VL Main Model (`models/qwen3vl.py`) — NEW FILE

`Qwen3VLModel` wrapper:
- Owns `Qwen3VLVisionModel` (visual) and `Qwen3Model` (language_model)
- `get_image_features()`: Vision encode → return embeddings
- `get_rope_index()`: Compute 3D M-RoPE position IDs for full multimodal sequence
- `get_vision_position_ids()`: Compute 3D positions for vision token grids
- `compute_3d_position_ids()`: Orchestrate position computation
- `forward()`:
  1. Text embedding → inputs_embeds
  2. Vision encode → image_embeds + deepstack_embeds
  3. masked_scatter image_embeds into inputs_embeds
  4. Compute 3D position_ids → M-RoPE cos/sin
  5. For each decoder layer: attention + MLP + (DeepStack injection at layers 0,1,2)
  6. Final norm → lm_head → logits

Modify `Qwen3Model.forward()` to accept:
- `deepstack_visual_embeds`: List of visual features for early layers
- `visual_pos_masks`: Boolean mask indicating visual token positions

### Phase 5: Vision Attention Kernel (`kernels/`)

- Vision attention reuses existing `flash_attention2_no_pad` kernel (already supports b_start_loc/b_seq_len)
- No new kernels needed — all existing kernels (FlashAttention, FlashDecoding, RoPE, SwiGLU, SkipRMSNorm) work as-is

### Phase 6: Executor (`executor/model_executor.py`)

- Add `"qwen3_vl"` to model type registry in `_initialize_model()`
- Update `forward()` signature: add `pixel_values`, `image_grid_thw` parameters
- Vision tokens replace placeholder tokens in text sequence, so KV cache allocation unchanged

### Phase 7: Generation Pipeline (`qwen3vl_generate_stream.py`) — NEW FILE

Similar to `llava_generate_stream.py` with Qwen3VL-specific adaptations:
- Uses `Qwen3VLProcessor` for image preprocessing
- Passes `image_grid_thw` and `mm_token_type_ids` through the pipeline
- M-RoPE position computation during prefill; incremental update during decode via `rope_deltas`
- Vision encoding only during first prefill step

### Phase 8: Weight Conversion (`apply_weight_convert.py`)

Add HF → lite_llama weight key mapping for Qwen3VL:
- `visual.patch_embed.proj.*` → vision encoder Conv3D
- `visual.blocks.{N}.*` → ViT layers
- `visual.merger.*` → main projector
- `visual.deepstack_merger_list.{N}.*` → deepstack projectors
- `model.language_model.*` → Qwen3Model (reuses existing Qwen3 mapping)

### Phase 9: CLI Entry Point (`cli_qwen3vl.py`) — NEW FILE

Interactive chat CLI for Qwen3VL, similar to `cli_llava.py`.

## Key Technical Decisions

1. **Reuse Qwen3Model**: Qwen3VL's text decoder is architecturally compatible with existing `Qwen3DecoderLayer` (Pre-Norm, QK norm, SwiGLU MLP, GQA). Only M-RoPE and DeepStack injection differ.

2. **M-RoPE pre-computation**: M-RoPE complexity is isolated to cos/sin pre-computation in `RotaryEmbedding.forward()`. The Triton `rope_emb_forward` kernel receives standard cos/sin tensors and applies them unchanged.

3. **Vision encoder is stateless**: Unlike the LLM decoder, the vision encoder runs once per image during prefill and has no KV cache. It outputs fixed-size embeddings.

4. **DeepStack injection is additive**: Visual features from ViT layers 8/16/24 are simply added to the corresponding LLM hidden states at layers 0/1/2 at the positions where visual tokens reside.
