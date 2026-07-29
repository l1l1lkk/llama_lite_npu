"""Unfused Q/K RMSNorm followed by rotary position embedding.

This module is the explicit performance baseline for Qwen3 attention.  Keeping
the three existing Triton launches behind one stable API lets the benchmark and
model integration select the baseline or fused implementation by branch.
"""

from __future__ import annotations

import torch

from .rope_emb import rope_emb_forward
from .skip_rmsnorm import skip_rmsnorm


@torch.no_grad()
def qk_rmsnorm_rope_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply independent Q/K RMSNorm and then RoPE using three launches."""

    q, _ = skip_rmsnorm(q, None, q_weight, eps)
    k, _ = skip_rmsnorm(k, None, k_weight, eps)
    return rope_emb_forward(q, k, cos, sin, batch_size, seq_len)
