"""NPU correctness and latency benchmark for the decode FFN fusion boundary."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

from lite_llama.kernels import rmsnorm_matmul_swiglu_forward


def synchronize() -> None:
    torch.npu.synchronize()


def reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    new_residual = x.float() + residual.float()
    variance = new_residual.square().mean(dim=-1, keepdim=True)
    normalized = (
        new_residual
        * torch.rsqrt(variance + eps)
        * norm_weight.float()
    ).to(x.dtype)
    gate, up = F.linear(normalized, gate_up_weight).chunk(2, dim=-1)
    output = F.silu(gate.float()).mul(up.float()).to(x.dtype)
    return output, new_residual.to(x.dtype)


def tensor_error(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float | bool]:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    absolute = (actual_fp32 - expected_fp32).abs()
    relative = absolute / expected_fp32.abs().clamp_min(1e-6)
    return {
        "max_abs": float(absolute.max().item()),
        "max_rel": float(relative.max().item()),
        "allclose_atol_5e-2_rtol_5e-2": bool(
            torch.allclose(actual_fp32, expected_fp32, atol=5e-2, rtol=5e-2)
        ),
    }


@torch.no_grad()
def run_shape(
    batch: int,
    hidden_size: int,
    intermediate_size: int,
    warmup: int,
    iterations: int,
    eps: float,
) -> dict:
    device = torch.device("npu")
    dtype = torch.float16
    torch.manual_seed(20260730 + batch)
    torch_npu.npu.manual_seed_all(20260730 + batch)
    x = torch.randn(
        batch, 1, hidden_size, device=device, dtype=dtype
    )
    residual_seed = torch.randn(
        batch, 1, hidden_size, device=device, dtype=dtype
    )
    norm_weight = torch.randn(
        hidden_size, device=device, dtype=dtype
    )
    gate_up_weight = torch.randn(
        2 * intermediate_size,
        hidden_size,
        device=device,
        dtype=dtype,
    ) / hidden_size**0.5

    expected, expected_residual = reference(
        x, residual_seed, norm_weight, gate_up_weight, eps
    )
    actual, actual_residual = rmsnorm_matmul_swiglu_forward(
        x, residual_seed.clone(), norm_weight, gate_up_weight, eps
    )
    synchronize()

    warmup_residuals = [
        residual_seed.clone() for _ in range(warmup)
    ]
    for residual in warmup_residuals:
        rmsnorm_matmul_swiglu_forward(
            x, residual, norm_weight, gate_up_weight, eps
        )
    synchronize()

    timed_residuals = [
        residual_seed.clone() for _ in range(iterations)
    ]
    latencies_ms: list[float] = []
    for residual in timed_residuals:
        synchronize()
        start = time.perf_counter()
        rmsnorm_matmul_swiglu_forward(
            x, residual, norm_weight, gate_up_weight, eps
        )
        synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    sorted_ms = sorted(latencies_ms)
    p95_index = min(len(sorted_ms) - 1, int(len(sorted_ms) * 0.95))
    return {
        "batch": batch,
        "sequence_length": 1,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "latency_ms": {
            "median": statistics.median(latencies_ms),
            "mean": statistics.fmean(latencies_ms),
            "min": min(latencies_ms),
            "p95": sorted_ms[p95_index],
        },
        "output_error": tensor_error(actual, expected),
        "residual_error": tensor_error(actual_residual, expected_residual),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [
        run_shape(
            batch,
            args.hidden_size,
            args.intermediate_size,
            args.warmup,
            args.iterations,
            args.eps,
        )
        for batch in (int(value) for value in args.batches.split(","))
    ]
    payload = {
        "device": str(torch.npu.get_device_name(torch.device("npu"))),
        "torch_version": torch.__version__,
        "shapes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
