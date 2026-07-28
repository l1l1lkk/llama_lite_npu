"""Packed-layout SwiGLU implemented as a two-dimensional Triton kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_MAX_BLOCK_SIZE = 8192


@triton.jit
def _swiglu_packed_kernel(
    gate_up_ptr,
    output_ptr,
    gate_up_row_stride,
    output_row_stride,
    feature_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse SiLU(gate) * up for a packed ``[gate, up]`` input row."""

    row = tl.program_id(0).to(tl.int64)
    column_block = tl.program_id(1).to(tl.int64)
    offsets = column_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < feature_width

    gate_row = gate_up_ptr + row * gate_up_row_stride
    up_row = gate_row + feature_width
    output_row = output_ptr + row * output_row_stride

    gate = tl.load(gate_row + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_row + offsets, mask=mask, other=0.0)
    output = gate * tl.sigmoid(gate) * up
    tl.store(output_row + offsets, output, mask=mask)


def _block_size(feature_width: int, row_count: int) -> int:
    """Choose a UB-safe tile for latency-oriented and throughput-oriented rows."""

    max_block_size = 4096 if row_count <= 32 else _MAX_BLOCK_SIZE
    return min(max_block_size, triton.next_power_of_2(feature_width))


def _validate_pair(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"SwiGLU inputs must have the same shape: {a.shape} != {b.shape}")
    if a.device != b.device:
        raise ValueError(f"SwiGLU inputs must share a device: {a.device} != {b.device}")
    if a.dtype != b.dtype:
        raise ValueError(f"SwiGLU inputs must share a dtype: {a.dtype} != {b.dtype}")


def swiglu_packed_forward(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU to ``[gate, up]`` packed along the final dimension."""

    if gate_up.ndim < 1:
        raise ValueError("Packed SwiGLU input must have at least one dimension")
    packed_width = gate_up.shape[-1]
    if packed_width % 2:
        raise ValueError(f"Packed SwiGLU width must be even, received {packed_width}")

    if gate_up.device.type == "npu":
        if gate_up.stride(-1) != 1:
            gate_up = gate_up.contiguous()
        feature_width = packed_width // 2
        gate_up_rows = gate_up.view(-1, packed_width)
        output = torch.empty(
            (gate_up_rows.shape[0], feature_width),
            dtype=gate_up.dtype,
            device=gate_up.device,
        )
        block_size = _block_size(feature_width, gate_up_rows.shape[0])
        grid = (
            gate_up_rows.shape[0],
            triton.cdiv(feature_width, block_size),
        )
        _swiglu_packed_kernel[grid](
            gate_up_rows,
            output,
            gate_up_rows.stride(-2),
            output.stride(-2),
            feature_width=feature_width,
            BLOCK_SIZE=block_size,
        )
        return output.view(*gate_up.shape[:-1], feature_width)

    gate, up = gate_up.chunk(2, dim=-1)
    gate_fp32 = gate.float()
    return (gate_fp32 * torch.sigmoid(gate_fp32) * up.float()).to(gate.dtype)


def swiglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compatibility path for callers that still produce separate tensors.

    Dense Qwen3 uses :func:`swiglu_packed_forward` directly so its packed linear
    projection feeds the Triton kernel without a runtime copy. Other models keep
    a correct two-input API and pay one explicit packing operation.
    """

    _validate_pair(a, b)
    return swiglu_packed_forward(torch.cat((a, b), dim=-1))
