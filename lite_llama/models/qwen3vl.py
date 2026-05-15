"""Qwen3-VL Multimodal Model.

Wraps the Qwen3VLVisionModel vision encoder with a Qwen3Model language model,
adding M-RoPE position encoding and DeepStack visual feature injection.
"""

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from dataclasses import dataclass

from .model_config import Qwen3VLConfig
from .qwen3vl_vision import Qwen3VLVisionModel
from .qwen3 import Qwen3Model
from .RotaryEmbedding import Qwen3VLTextRotaryEmbedding
from ..executor.tp_utils import TPConfig


@dataclass
class Qwen3VLModelOutput:
    last_hidden_state: torch.Tensor
    logits: Optional[torch.Tensor] = None
    rope_deltas: Optional[torch.Tensor] = None


class Qwen3VLModel(nn.Module):
    """Qwen3-VL multimodal model for inference."""

    def __init__(self, config: Qwen3VLConfig, tp_config: TPConfig = None):
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config
        self.text_config = config.text_config

        # Vision encoder
        self.visual = Qwen3VLVisionModel(config.vision_config)

        # Language model (Qwen3-based, TP-aware)
        self.language_model = Qwen3Model(config.text_config, tp_config=tp_config)
        # Replace standard RoPE with M-RoPE for 3D position encoding
        self.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config.text_config)

        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.vision_start_token_id = config.vision_start_token_id
        self.vision_end_token_id = config.vision_end_token_id

        self.rope_deltas = None
        self.spatial_merge_size = config.vision_config.spatial_merge_size

    # ------------------------------------------------------------------
    # Vision encoding
    # ------------------------------------------------------------------
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run vision encoder and return (pooled_embeddings, deepstack_features).

        Args:
            pixel_values: (total_patches, C, temporal_patch, patch, patch)
            image_grid_thw: (num_images, 3)

        Returns:
            pooled: (total_output_tokens, out_hidden_size)
            deepstack_features: list of (total_output_tokens, out_hidden_size)
        """
        vision_output = self.visual(pixel_values, image_grid_thw)

        # Split pooled output per image
        split_sizes = (
            image_grid_thw.prod(-1) // self.spatial_merge_size**2
        ).tolist()

        # Also split deepstack features per image
        deepstack_features = []
        for feat in vision_output.deepstack_features:
            deepstack_features.append(feat)

        return vision_output.pooler_output, deepstack_features, split_sizes

    # ------------------------------------------------------------------
    # 3D Position computation for M-RoPE
    # ------------------------------------------------------------------
    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        device=None,
    ) -> torch.Tensor:
        """Compute 3D position IDs for vision tokens from a single image/video grid.

        Returns shape (3, sequence_length) with temporal, height, width positions.
        """
        spatial_merge_size = self.spatial_merge_size

        llm_grid_t = grid_thw[0].item()
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size

        position_temporal = torch.arange(llm_grid_t, device=device)
        position_width = torch.arange(llm_grid_w, device=device) + start_position
        position_height = torch.arange(llm_grid_h, device=device) + start_position

        position_width = position_width.repeat(llm_grid_h * llm_grid_t)
        position_height = position_height.repeat_interleave(llm_grid_w).repeat(llm_grid_t)
        position_temporal = (
            position_temporal.repeat_interleave(llm_grid_h * llm_grid_w) + start_position
        )

        vision_position_ids = torch.stack(
            [position_temporal, position_height, position_width], dim=0
        )
        return vision_position_ids

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 3D M-RoPE position IDs and rope deltas for the full multimodal sequence.

        Returns:
            position_ids: (3, batch_size, sequence_length)
            mrope_position_deltas: (batch_size, 1)
        """
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            )
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.spatial_merge_size
        mrope_position_deltas = []
        position_ids = torch.zeros(
            3, input_ids.shape[0], input_ids.shape[1],
            dtype=input_ids.dtype, device=input_ids.device,
        )

        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(
                enumerate(input_token_type.tolist()), lambda x: x[1]
            ):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:  # text
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device)
                        .view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:  # image=1 or video=2
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = (
                    llm_positions.to(position_ids.device)
                )
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(current_input_ids)
            )

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas

    # ------------------------------------------------------------------
    # Compute 3D position IDs
    # ------------------------------------------------------------------
    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None

        if has_multimodal and mm_token_type_ids is not None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
            return position_ids

        if self.rope_deltas is not None:
            # Decode stage: increment positions using rope_deltas
            batch_size, seq_length = inputs_embeds.shape[:2]
            position_ids = torch.arange(
                0, seq_length, device=inputs_embeds.device
            ).view(1, 1, -1).expand(3, batch_size, -1)
            delta = self.rope_deltas.to(device=inputs_embeds.device)
            position_ids = position_ids + delta
            return position_ids

        return None

    # ------------------------------------------------------------------
    # Placeholder mask helpers
    # ------------------------------------------------------------------
    def _get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        token_id: int,
    ) -> torch.Tensor:
        mask = input_ids == token_id
        return mask.unsqueeze(-1).expand_as(inputs_embeds)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        image_tensor: Optional[torch.Tensor] = None,
    ):
        """Forward pass for Qwen3-VL.

        Prefill stage: encodes images, merges visual embeddings with text,
        computes M-RoPE positions, runs LLM with DeepStack.
        Decode stage: text-only autoregressive generation.
        """
        # Support both image_tensor (legacy API) and pixel_values
        if pixel_values is None and image_tensor is not None:
            pixel_values = image_tensor

        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        batch_size, seq_len = input_ids.shape

        image_mask = None
        video_mask = None
        visual_pos_masks = None
        deepstack_visual_embeds = None

        # --- Prefill: multimodal encoding ---
        if seq_len > 1 and pixel_values is not None:
            if image_grid_thw is None:
                # Infer grid_thw: assume single image, compute from pixel shape
                # pixel_values: (N, C, T_patch, P, P) or (N, C, H, W)
                if pixel_values.ndim == 5:
                    n_patches = pixel_values.shape[0]
                    h = w = int(n_patches**0.5)
                    image_grid_thw = torch.tensor(
                        [[1, h, w]], device=pixel_values.device, dtype=torch.long
                    )
                else:
                    raise ValueError("image_grid_thw required for non-patch pixel_values")

            # Encode images
            pooled_embeds, deepstack_feats, split_sizes = self.get_image_features(
                pixel_values, image_grid_thw
            )

            # Merge visual embeddings into text embeddings
            image_mask = self._get_placeholder_mask(
                input_ids, inputs_embeds, self.image_token_id
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, pooled_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            )

            if deepstack_feats:
                visual_pos_masks = image_mask[..., 0]
                deepstack_visual_embeds = deepstack_feats

            # Auto-generate mm_token_type_ids if processor didn't provide them
            if mm_token_type_ids is None and image_grid_thw is not None:
                mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
                # Mark image tokens and vision boundary tokens as modality=1 (image)
                mm_token_type_ids[input_ids == self.image_token_id] = 1
                mm_token_type_ids[input_ids == self.vision_start_token_id] = 1
                mm_token_type_ids[input_ids == self.vision_end_token_id] = 1

            # Compute 3D position IDs for M-RoPE
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=None,
                mm_token_type_ids=mm_token_type_ids,
            )

        # --- Decode: offset 2D position with rope_deltas for M-RoPE ---
        if position_ids is not None and position_ids.ndim == 2 and self.rope_deltas is not None:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
            delta = self.rope_deltas.to(device=position_ids.device)
            position_ids = position_ids + delta

        # --- Decode: compute 3D position IDs from cached rope_deltas ---
        if position_ids is None and self.rope_deltas is not None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
            )

        # --- Fallback: simple 1D position IDs if M-RoPE unavailable ---
        if position_ids is None:
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(
                0, seq_len, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # --- Run language model ---
        output = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            atten_info=atten_info,
            inputs_embeds=inputs_embeds,
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_pos_masks=visual_pos_masks,
        )

        return output
