"""Interleaved packed SwiGLU implemented as a one-dimensional Triton kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_MAX_BLOCK_SIZE = 4096


@triton.jit
def _swiglu_packed_kernel(
    gate_up_ptr,
    output_ptr,
    output_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse SiLU(gate) * up for ``[gate0, up0, gate1, up1, ...]``."""

    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )
    mask = offsets < output_elements
    packed_offsets = offsets * 2
    gate = tl.load(gate_up_ptr + packed_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_up_ptr + packed_offsets + 1, mask=mask, other=0.0)
    output = gate * tl.sigmoid(gate) * up
    tl.store(output_ptr + offsets, output, mask=mask)


def _block_size(output_elements: int) -> int:
    """Choose a UB-safe tile while keeping small launches compact."""

    return min(_MAX_BLOCK_SIZE, triton.next_power_of_2(output_elements))


def _validate_pair(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"SwiGLU inputs must have the same shape: {a.shape} != {b.shape}")
    if a.device != b.device:
        raise ValueError(f"SwiGLU inputs must share a device: {a.device} != {b.device}")
    if a.dtype != b.dtype:
        raise ValueError(f"SwiGLU inputs must share a dtype: {a.dtype} != {b.dtype}")


def swiglu_packed_forward(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU to interleaved gate/up pairs along the final dimension."""

    if gate_up.ndim < 1:
        raise ValueError("Packed SwiGLU input must have at least one dimension")
    packed_width = gate_up.shape[-1]
    if packed_width % 2:
        raise ValueError(f"Packed SwiGLU width must be even, received {packed_width}")

    if gate_up.device.type == "npu":
        if gate_up.stride(-1) != 1:
            gate_up = gate_up.contiguous()
        feature_width = packed_width // 2
        output_shape = (*gate_up.shape[:-1], feature_width)
        output = torch.empty(
            output_shape,
            dtype=gate_up.dtype,
            device=gate_up.device,
        )
        output_elements = output.numel()
        block_size = _block_size(output_elements)
        grid = (triton.cdiv(output_elements, block_size),)
        _swiglu_packed_kernel[grid](
            gate_up,
            output,
            output_elements=output_elements,
            BLOCK_SIZE=block_size,
        )
        return output

    gate = gate_up[..., 0::2]
    up = gate_up[..., 1::2]
    gate_fp32 = gate.float()
    return (gate_fp32 * torch.sigmoid(gate_fp32) * up.float()).to(gate.dtype)


def swiglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compatibility path for callers that still produce separate tensors.

    Dense Qwen3 uses :func:`swiglu_packed_forward` directly so its packed linear
    projection feeds the Triton kernel without a runtime copy. Other models keep
    a correct two-input API and pay one explicit packing operation.
    """

    _validate_pair(a, b)
    interleaved = torch.stack((a, b), dim=-1).flatten(-2)
    return swiglu_packed_forward(interleaved)
