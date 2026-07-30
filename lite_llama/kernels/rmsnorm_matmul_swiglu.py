"""Reference implementation for the decode FFN fusion boundary.

The execution order is skip-RMSNorm, packed gate/up MatMul, then SwiGLU.
Keeping the three launches behind one API gives the Triton implementation an
exactly matched semantic baseline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .skip_rmsnorm import skip_rmsnorm
from .swiglu_fused import swiglu_packed_forward


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
    """Run the unfused reference chain and return activation plus new residual."""

    _validate_inputs(x, residual, weight, gate_up_weight)

    if x.device.type == "npu":
        normalized, new_residual = skip_rmsnorm(x, residual, weight, eps)
    else:
        new_residual = x if residual is None else x + residual
        variance = new_residual.float().square().mean(dim=-1, keepdim=True)
        normalized = (
            new_residual.float() * torch.rsqrt(variance + eps) * weight.float()
        ).to(x.dtype)

    gate_up = F.linear(normalized, gate_up_weight)
    return swiglu_packed_forward(gate_up), new_residual
