#!/usr/bin/env python3
"""Benchmark the branch-selected Q/K RMSNorm + RoPE backend on Ascend NPU."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch_npu

from lite_llama.kernels import qk_rmsnorm_rope_forward


@dataclass(frozen=True)
class ShapeCase:
    family: str
    batch: int
    sequence_length: int
    q_heads: int
    k_heads: int
    head_dimension: int


DEFAULT_CASES = (
    ShapeCase("batch", 1, 1, 16, 4, 128),
    ShapeCase("batch", 4, 1, 16, 4, 128),
    ShapeCase("batch", 16, 1, 16, 4, 128),
    ShapeCase("batch", 32, 1, 16, 4, 128),
    ShapeCase("sequence", 1, 16, 16, 4, 128),
    ShapeCase("sequence", 1, 128, 16, 4, 128),
    ShapeCase("sequence", 1, 512, 16, 4, 128),
    ShapeCase("sequence", 1, 2048, 16, 4, 128),
    ShapeCase("head_dimension", 4, 128, 16, 4, 64),
    ShapeCase("head_dimension", 4, 128, 16, 4, 128),
    ShapeCase("head_dimension", 4, 128, 16, 4, 256),
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def synchronize(device: torch.device) -> None:
    if device.type != "npu":
        raise ValueError(f"This benchmark requires an NPU, received {device}")
    torch.npu.synchronize(device)


def rmsnorm_rope_reference(
    value: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    variance = value.float().square().mean(dim=-1, keepdim=True)
    normalized = (value.float() * torch.rsqrt(variance + eps)).to(value.dtype)
    normalized = normalized * weight
    half = value.shape[-1] // 2
    first = normalized[..., :half]
    second = normalized[..., half:]
    cos_half = cos[..., :half].reshape(-1, 1, half)
    sin_half = sin[..., :half].reshape(-1, 1, half)
    return torch.cat(
        (
            first * cos_half - second * sin_half,
            second * cos_half + first * sin_half,
        ),
        dim=-1,
    )


def error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    relative_error_floor: float,
) -> dict:
    difference = (actual.float() - expected.float()).abs()
    expected_abs = expected.float().abs()
    relative_mask = expected_abs > relative_error_floor
    max_relative_error = (
        (difference[relative_mask] / expected_abs[relative_mask]).max().item()
        if relative_mask.any()
        else 0.0
    )
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "max_relative_error": max_relative_error,
        "allclose": torch.allclose(
            actual.float(), expected.float(), atol=atol, rtol=rtol
        ),
    }


def time_samples(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    warmup: int,
    samples: int,
    inner_iterations: int,
) -> list[float]:
    for _ in range(warmup):
        fn()
    synchronize(device)
    timings_ms = []
    for _ in range(samples):
        synchronize(device)
        started = time.perf_counter_ns()
        for _ in range(inner_iterations):
            fn()
        synchronize(device)
        timings_ms.append(
            (time.perf_counter_ns() - started) / 1_000_000 / inner_iterations
        )
    return timings_ms


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    results = []
    for case in DEFAULT_CASES:
        tokens = case.batch * case.sequence_length
        q = torch.randn(
            (tokens, case.q_heads, case.head_dimension),
            device=device,
            dtype=dtype,
        )
        k = torch.randn(
            (tokens, case.k_heads, case.head_dimension),
            device=device,
            dtype=dtype,
        )
        q_weight = torch.randn(case.head_dimension, device=device, dtype=dtype)
        k_weight = torch.randn(case.head_dimension, device=device, dtype=dtype)
        angles = torch.randn(
            (case.batch, case.sequence_length, case.head_dimension // 2),
            device=device,
            dtype=torch.float32,
        )
        cos_half = angles.cos().to(dtype)
        sin_half = angles.sin().to(dtype)
        cos = torch.cat((cos_half, cos_half), dim=-1)
        sin = torch.cat((sin_half, sin_half), dim=-1)

        operation = lambda: qk_rmsnorm_rope_forward(
            q,
            k,
            q_weight,
            k_weight,
            cos,
            sin,
            case.batch,
            case.sequence_length,
            args.eps,
        )
        expected_q = rmsnorm_rope_reference(q, q_weight, cos, sin, args.eps)
        expected_k = rmsnorm_rope_reference(k, k_weight, cos, sin, args.eps)
        actual_q, actual_k = operation()
        synchronize(device)
        q_error = error_metrics(
            actual_q,
            expected_q,
            atol=args.atol,
            rtol=args.rtol,
            relative_error_floor=args.relative_error_floor,
        )
        k_error = error_metrics(
            actual_k,
            expected_k,
            atol=args.atol,
            rtol=args.rtol,
            relative_error_floor=args.relative_error_floor,
        )

        total_elements = q.numel() + k.numel()
        inner_iterations = max(
            1,
            min(
                args.max_inner_iterations,
                args.target_elements_per_sample // total_elements,
            ),
        )
        timings_ms = time_samples(
            operation,
            device=device,
            warmup=args.warmup,
            samples=args.samples,
            inner_iterations=inner_iterations,
        )
        mean_ms = statistics.fmean(timings_ms)
        stdev_ms = statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0
        row = {
            **asdict(case),
            "dtype": args.dtype,
            "total_elements": total_elements,
            "inner_iterations": inner_iterations,
            "samples": args.samples,
            "mean_ms": mean_ms,
            "p50_ms": percentile(timings_ms, 0.50),
            "p90_ms": percentile(timings_ms, 0.90),
            "p99_ms": percentile(timings_ms, 0.99),
            "stdev_ms": stdev_ms,
            "cv_percent": (stdev_ms / mean_ms * 100.0) if mean_ms else 0.0,
            "q_max_abs_error": q_error["max_abs_error"],
            "q_mean_abs_error": q_error["mean_abs_error"],
            "q_max_relative_error": q_error["max_relative_error"],
            "k_max_abs_error": k_error["max_abs_error"],
            "k_mean_abs_error": k_error["mean_abs_error"],
            "k_max_relative_error": k_error["max_relative_error"],
            "allclose": q_error["allclose"] and k_error["allclose"],
        }
        results.append(row)
        print(json.dumps(row, sort_keys=True))

    failures = [row for row in results if not row["allclose"]]
    report = {
        "schema_version": 1,
        "metadata": {
            "git_branch": git_value("branch", "--show-current"),
            "git_commit": git_value("rev-parse", "HEAD"),
            "backend_module": qk_rmsnorm_rope_forward.__module__,
            "device": args.device,
            "physical_device": args.physical_device,
            "dtype": args.dtype,
            "torch_version": torch.__version__,
            "torch_npu_version": torch_npu.__version__,
            "python_version": platform.python_version(),
            "seed": args.seed,
            "warmup": args.warmup,
            "samples": args.samples,
            "eps": args.eps,
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "correctness": {
            "passed": not failures,
            "failed_cases": len(failures),
            "total_cases": len(results),
        },
        "results": results,
    }
    if failures:
        raise RuntimeError(f"{len(failures)} RMSNorm + RoPE cases failed")
    return report


def write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["results"][0]))
        writer.writeheader()
        writer.writerows(report["results"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--max-inner-iterations", type=int, default=20)
    parser.add_argument("--target-elements-per-sample", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--relative-error-floor", type=float, default=1e-2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    write_report(report, args.output)
    print(
        json.dumps(
            {
                "status": "pass",
                "cases": report["correctness"]["total_cases"],
                "output": str(args.output.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
