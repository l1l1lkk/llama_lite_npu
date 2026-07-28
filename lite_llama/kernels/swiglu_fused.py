"""Packed-layout SwiGLU backed by the native Ascend CANN operator."""

from __future__ import annotations

import torch


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
        import torch_npu

        return torch_npu.npu_swiglu(gate_up, dim=-1)

    gate, up = gate_up.chunk(2, dim=-1)
    gate_fp32 = gate.float()
    return (gate_fp32 * torch.sigmoid(gate_fp32) * up.float()).to(gate.dtype)


def swiglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compatibility path for callers that still produce separate tensors.

    Dense Qwen3 uses :func:`swiglu_packed_forward` directly so its packed linear
    projection feeds the CANN kernel without a runtime copy. Other models keep a
    correct two-input API and pay one explicit packing operation.
    """

    _validate_pair(a, b)
    return swiglu_packed_forward(torch.cat((a, b), dim=-1))
