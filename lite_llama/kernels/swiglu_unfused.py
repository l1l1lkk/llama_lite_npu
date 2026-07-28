"""Unfused SwiGLU reference used by the isolated performance baseline branch."""

import torch


def swiglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute ``SiLU(a) * b`` as separate eager PyTorch operations.

    The intermediate arithmetic intentionally uses FP32, matching the numerical
    contract of the Triton implementation.  This function is deliberately not
    optimized: it provides a stable, auditable baseline for kernel-count,
    memory-traffic, and end-to-end comparisons.
    """

    if a.shape != b.shape:
        raise ValueError(f"SwiGLU inputs must have the same shape: {a.shape} != {b.shape}")
    if a.device != b.device:
        raise ValueError(f"SwiGLU inputs must share a device: {a.device} != {b.device}")
    if a.dtype != b.dtype:
        raise ValueError(f"SwiGLU inputs must share a dtype: {a.dtype} != {b.dtype}")

    a_fp32 = a.float()
    b_fp32 = b.float()
    output = a_fp32 * torch.sigmoid(a_fp32) * b_fp32
    return output.to(dtype=a.dtype)
