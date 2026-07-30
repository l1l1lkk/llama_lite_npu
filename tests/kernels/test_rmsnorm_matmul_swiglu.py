import torch
import torch.nn.functional as F

from lite_llama.kernels.rmsnorm_matmul_swiglu import (
    rmsnorm_matmul_swiglu_forward,
)


def _reference(x, residual, weight, gate_up_weight, eps):
    new_residual = x if residual is None else x + residual
    variance = new_residual.float().square().mean(dim=-1, keepdim=True)
    normalized = (
        new_residual.float() * torch.rsqrt(variance + eps) * weight.float()
    ).to(x.dtype)
    gate, up = F.linear(normalized, gate_up_weight).chunk(2, dim=-1)
    return F.silu(gate.float()).mul(up.float()).to(x.dtype), new_residual


@torch.no_grad()
def test_unfused_reference_matches_torch():
    torch.manual_seed(7)
    x = torch.randn(2, 1, 16, dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(16, dtype=x.dtype)
    gate_up_weight = torch.randn(24, 16, dtype=x.dtype)

    actual, actual_residual = rmsnorm_matmul_swiglu_forward(
        x, residual, weight, gate_up_weight, 1e-6
    )
    expected, expected_residual = _reference(
        x, residual, weight, gate_up_weight, 1e-6
    )

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual_residual, expected_residual)


@torch.no_grad()
def test_unfused_reference_supports_missing_residual():
    x = torch.randn(1, 3, 8, dtype=torch.float32)
    weight = torch.ones(8)
    gate_up_weight = torch.randn(20, 8)

    actual, actual_residual = rmsnorm_matmul_swiglu_forward(
        x, None, weight, gate_up_weight
    )
    expected, expected_residual = _reference(
        x, None, weight, gate_up_weight, 1e-5
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_residual, expected_residual)


def test_unfused_reference_rejects_odd_packed_width():
    x = torch.randn(1, 1, 8)
    try:
        rmsnorm_matmul_swiglu_forward(
            x, None, torch.ones(8), torch.randn(19, 8)
        )
    except ValueError as error:
        assert "must be even" in str(error)
    else:
        raise AssertionError("odd packed width must be rejected")
