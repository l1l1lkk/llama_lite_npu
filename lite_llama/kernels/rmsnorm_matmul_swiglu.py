"""Triton implementation for the decode FFN fusion boundary.

The execution order is skip-RMSNorm, packed gate/up MatMul, then SwiGLU.
Decode uses one Triton kernel while prefill keeps the tuned unfused path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .skip_rmsnorm import skip_rmsnorm
from .swiglu_fused import swiglu_packed_forward


_MAX_FUSED_DECODE_ROWS = 16
_MAX_HIDDEN_SIZE = 8192


@triton.jit
def _rmsnorm_matmul_swiglu_decode_kernel(
    x_ptr,
    residual_ptr,
    norm_weight_ptr,
    gate_up_weight_ptr,
    output_ptr,
    new_residual_ptr,
    rows,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_RMS: tl.constexpr,
):
    """Fuse decode skip-RMSNorm, packed gate/up GEMM, and SwiGLU."""

    row_block = tl.program_id(0)
    column_block = tl.program_id(1)

    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    rms_offsets = tl.arange(0, BLOCK_RMS)
    rms_mask = (
        (row_offsets[:, None] < rows)
        & (rms_offsets[None, :] < HIDDEN_SIZE)
    )
    x = tl.load(
        x_ptr + row_offsets[:, None] * HIDDEN_SIZE + rms_offsets[None, :],
        mask=rms_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(
            residual_ptr
            + row_offsets[:, None] * HIDDEN_SIZE
            + rms_offsets[None, :],
            mask=rms_mask,
            other=0.0,
        ).to(tl.float32)
        x += residual

    # Only the first N tile publishes the residual. All N tiles read the old
    # input buffers, so there is no in-place read/write race.
    tl.store(
        new_residual_ptr
        + row_offsets[:, None] * HIDDEN_SIZE
        + rms_offsets[None, :],
        x,
        mask=rms_mask & (column_block == 0),
    )
    variance = tl.sum(x * x, axis=1) / HIDDEN_SIZE
    inverse_rms = 1.0 / tl.sqrt(variance + eps)

    column_offsets = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    gate_accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, HIDDEN_SIZE, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        matrix_mask = (
            (row_offsets[:, None] < rows)
            & (k_offsets[None, :] < HIDDEN_SIZE)
        )
        values = tl.load(
            x_ptr + row_offsets[:, None] * HIDDEN_SIZE + k_offsets[None, :],
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_RESIDUAL:
            values += tl.load(
                residual_ptr
                + row_offsets[:, None] * HIDDEN_SIZE
                + k_offsets[None, :],
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
        norm_weight = tl.load(
            norm_weight_ptr + k_offsets[None, :],
            mask=k_offsets[None, :] < HIDDEN_SIZE,
            other=0.0,
        ).to(tl.float32)
        normalized = (
            values * inverse_rms[:, None] * norm_weight
        ).to(tl.float16)

        weight_mask = (
            (k_offsets[:, None] < HIDDEN_SIZE)
            & (column_offsets[None, :] < INTERMEDIATE_SIZE)
        )
        gate_weight = tl.load(
            gate_up_weight_ptr
            + column_offsets[None, :] * HIDDEN_SIZE
            + k_offsets[:, None],
            mask=weight_mask,
            other=0.0,
        )
        up_weight = tl.load(
            gate_up_weight_ptr
            + (INTERMEDIATE_SIZE + column_offsets[None, :]) * HIDDEN_SIZE
            + k_offsets[:, None],
            mask=weight_mask,
            other=0.0,
        )
        gate_accumulator = tl.dot(
            normalized, gate_weight, acc=gate_accumulator
        )
        up_accumulator = tl.dot(
            normalized, up_weight, acc=up_accumulator
        )

    activated = (
        gate_accumulator
        * tl.sigmoid(gate_accumulator)
        * up_accumulator
    )
    output_mask = (
        (row_offsets[:, None] < rows)
        & (column_offsets[None, :] < INTERMEDIATE_SIZE)
    )
    tl.store(
        output_ptr
        + row_offsets[:, None] * INTERMEDIATE_SIZE
        + column_offsets[None, :],
        activated,
        mask=output_mask,
    )


def _validate_inputs(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
) -> None:
    if x.ndim < 2:
        raise ValueError("RMSNorm-MatMul-SwiGLU input must have at least two dimensions")
    hidden_size = x.shape[-1]
    if weight.shape != (hidden_size,):
        raise ValueError(
            f"RMSNorm weight must have shape ({hidden_size},), received {weight.shape}"
        )
    if gate_up_weight.ndim != 2 or gate_up_weight.shape[1] != hidden_size:
        raise ValueError(
            "Packed gate/up weight must have shape "
            f"(2 * intermediate_size, {hidden_size}), received {gate_up_weight.shape}"
        )
    if gate_up_weight.shape[0] % 2:
        raise ValueError("Packed gate/up weight output dimension must be even")
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"Residual must have shape {x.shape}, received {residual.shape}"
        )
    tensors = (weight, gate_up_weight) if residual is None else (
        residual,
        weight,
        gate_up_weight,
    )
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("All RMSNorm-MatMul-SwiGLU tensors must share a device")
    if any(tensor.dtype != x.dtype for tensor in tensors):
        raise ValueError("All RMSNorm-MatMul-SwiGLU tensors must share a dtype")


def rmsnorm_matmul_swiglu_forward(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the fused decode kernel, with the exact unfused prefill fallback."""

    _validate_inputs(x, residual, weight, gate_up_weight)

    rows = x.numel() // x.shape[-1]
    use_fused_decode = (
        x.device.type == "npu"
        and x.dtype == torch.float16
        and x.shape[-2] == 1
        and rows <= _MAX_FUSED_DECODE_ROWS
        and x.shape[-1] <= _MAX_HIDDEN_SIZE
    )
    if use_fused_decode:
        x_2d = x.contiguous().view(rows, x.shape[-1])
        residual_2d = (
            residual.contiguous().view_as(x_2d) if residual is not None else x_2d
        )
        norm_weight = weight.contiguous()
        packed_weight = gate_up_weight.contiguous()
        intermediate_size = packed_weight.shape[0] // 2
        output = torch.empty(
            (rows, intermediate_size), dtype=x.dtype, device=x.device
        )
        new_residual = torch.empty_like(x_2d)
        block_m = 8
        block_n = 64
        block_k = 32
        block_rms = triton.next_power_of_2(x.shape[-1])
        grid = (
            triton.cdiv(rows, block_m),
            triton.cdiv(intermediate_size, block_n),
        )
        _rmsnorm_matmul_swiglu_decode_kernel[grid](
            x_2d,
            residual_2d,
            norm_weight,
            packed_weight,
            output,
            new_residual,
            rows,
            eps,
            HAS_RESIDUAL=residual is not None,
            HIDDEN_SIZE=x.shape[-1],
            INTERMEDIATE_SIZE=intermediate_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            BLOCK_RMS=block_rms,
        )
        return (
            output.view(*x.shape[:-1], intermediate_size),
            new_residual.view_as(x),
        )

    if x.device.type != "npu":
        new_residual = x if residual is None else x + residual
        variance = new_residual.float().square().mean(dim=-1, keepdim=True)
        normalized = (
            new_residual.float() * torch.rsqrt(variance + eps) * weight.float()
        ).to(x.dtype)
    else:
        normalized, new_residual = skip_rmsnorm(x, residual, weight, eps)

    gate_up = F.linear(normalized, gate_up_weight)
    return swiglu_packed_forward(gate_up), new_residual
