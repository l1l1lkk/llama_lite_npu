"""Fused Triton Q/K RMSNorm and rotary position embedding."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_rmsnorm_rope_kernel(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    q_weight_ptr,
    k_weight_ptr,
    cos_ptr,
    cos_batch_stride,
    cos_sequence_stride,
    sin_ptr,
    sin_batch_stride,
    sin_sequence_stride,
    sequence_length,
    eps,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    head_dim: tl.constexpr,
    padded_q_heads: tl.constexpr,
    padded_k_heads: tl.constexpr,
    padded_half_dim: tl.constexpr,
):
    token_id = tl.program_id(0)
    batch_id = token_id // sequence_length
    position_id = token_id % sequence_length

    half_offsets = tl.arange(0, padded_half_dim)
    half_mask = half_offsets < head_dim // 2
    cos_row = tl.load(
        cos_ptr
        + batch_id * cos_batch_stride
        + position_id * cos_sequence_stride
        + half_offsets,
        mask=half_mask,
        other=0.0,
    )
    sin_row = tl.load(
        sin_ptr
        + batch_id * sin_batch_stride
        + position_id * sin_sequence_stride
        + half_offsets,
        mask=half_mask,
        other=0.0,
    )

    q_head_offsets = tl.arange(0, padded_q_heads)[:, None]
    q_dim_offsets = half_offsets[None, :]
    q_mask = (q_head_offsets < q_heads) & (q_dim_offsets < head_dim // 2)
    q_base = q_ptr + token_id * q_row_stride
    q_first_offsets = q_head_offsets * head_dim + q_dim_offsets
    q_second_offsets = q_first_offsets + head_dim // 2
    q_first = tl.load(q_base + q_first_offsets, mask=q_mask, other=0.0).to(
        tl.float32
    )
    q_second = tl.load(q_base + q_second_offsets, mask=q_mask, other=0.0).to(
        tl.float32
    )
    q_variance = tl.sum(q_first * q_first + q_second * q_second, axis=1) / head_dim
    q_rrms = 1.0 / tl.sqrt(q_variance + eps)
    q_weight_first = tl.load(
        q_weight_ptr + half_offsets, mask=half_mask, other=0.0
    )
    q_weight_second = tl.load(
        q_weight_ptr + half_offsets + head_dim // 2,
        mask=half_mask,
        other=0.0,
    )
    q_first = (q_first * q_rrms[:, None]).to(q_ptr.dtype.element_ty)
    q_second = (q_second * q_rrms[:, None]).to(q_ptr.dtype.element_ty)
    q_first = q_first * q_weight_first[None, :]
    q_second = q_second * q_weight_second[None, :]
    q_rotated_first = q_first * cos_row[None, :] - q_second * sin_row[None, :]
    q_rotated_second = q_second * cos_row[None, :] + q_first * sin_row[None, :]
    tl.store(q_base + q_first_offsets, q_rotated_first, mask=q_mask)
    tl.store(q_base + q_second_offsets, q_rotated_second, mask=q_mask)

    k_head_offsets = tl.arange(0, padded_k_heads)[:, None]
    k_dim_offsets = half_offsets[None, :]
    k_mask = (k_head_offsets < k_heads) & (k_dim_offsets < head_dim // 2)
    k_base = k_ptr + token_id * k_row_stride
    k_first_offsets = k_head_offsets * head_dim + k_dim_offsets
    k_second_offsets = k_first_offsets + head_dim // 2
    k_first = tl.load(k_base + k_first_offsets, mask=k_mask, other=0.0).to(
        tl.float32
    )
    k_second = tl.load(k_base + k_second_offsets, mask=k_mask, other=0.0).to(
        tl.float32
    )
    k_variance = tl.sum(k_first * k_first + k_second * k_second, axis=1) / head_dim
    k_rrms = 1.0 / tl.sqrt(k_variance + eps)
    k_weight_first = tl.load(
        k_weight_ptr + half_offsets, mask=half_mask, other=0.0
    )
    k_weight_second = tl.load(
        k_weight_ptr + half_offsets + head_dim // 2,
        mask=half_mask,
        other=0.0,
    )
    k_first = (k_first * k_rrms[:, None]).to(k_ptr.dtype.element_ty)
    k_second = (k_second * k_rrms[:, None]).to(k_ptr.dtype.element_ty)
    k_first = k_first * k_weight_first[None, :]
    k_second = k_second * k_weight_second[None, :]
    k_rotated_first = k_first * cos_row[None, :] - k_second * sin_row[None, :]
    k_rotated_second = k_second * cos_row[None, :] + k_first * sin_row[None, :]
    tl.store(k_base + k_first_offsets, k_rotated_first, mask=k_mask)
    tl.store(k_base + k_second_offsets, k_rotated_second, mask=k_mask)


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
    """Normalize Q/K per head and apply RoPE in one in-place Triton launch."""

    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("q and k must have shape [batch * sequence, heads, head_dim]")
    tokens, q_heads, head_dim = q.shape
    k_tokens, k_heads, k_head_dim = k.shape
    if tokens != batch_size * seq_len or k_tokens != tokens:
        raise ValueError("q/k token count must equal batch_size * seq_len")
    if k_head_dim != head_dim:
        raise ValueError("q and k must use the same head dimension")
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    if q_weight.numel() != head_dim or k_weight.numel() != head_dim:
        raise ValueError("RMSNorm weight length must equal head dimension")
    if cos.shape[:2] != (batch_size, seq_len) or sin.shape != cos.shape:
        raise ValueError("cos/sin must have shape [batch, sequence, head_dim]")
    if cos.shape[-1] < head_dim:
        raise ValueError("cos/sin last dimension must cover head_dim")

    q = q.contiguous()
    k = k.contiguous()
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    padded_q_heads = triton.next_power_of_2(q_heads)
    padded_k_heads = triton.next_power_of_2(k_heads)
    padded_half_dim = triton.next_power_of_2(head_dim // 2)
    _qk_rmsnorm_rope_kernel[(tokens,)](
        q,
        q.stride(0),
        k,
        k.stride(0),
        q_weight,
        k_weight,
        cos,
        cos.stride(0),
        cos.stride(1),
        sin,
        sin.stride(0),
        sin.stride(1),
        seq_len,
        eps,
        q_heads=q_heads,
        k_heads=k_heads,
        head_dim=head_dim,
        padded_q_heads=padded_q_heads,
        padded_k_heads=padded_k_heads,
        padded_half_dim=padded_half_dim,
    )
    return q, k
