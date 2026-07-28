#!/usr/bin/env python3
"""Benchmark the branch-selected SwiGLU backend on an Ascend NPU."""

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

from lite_llama.kernels import swiglu_forward

try:
    from lite_llama.kernels import swiglu_packed_forward
except ImportError:
    swiglu_packed_forward = None


@dataclass(frozen=True)
class ShapeCase:
    family: str
    batch: int
    sequence_length: int
    feature_dimension: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.batch, self.sequence_length, self.feature_dimension


DEFAULT_CASES = (
    ShapeCase("batch", 1, 1, 6144),
    ShapeCase("batch", 4, 1, 6144),
    ShapeCase("batch", 16, 1, 6144),
    ShapeCase("batch", 32, 1, 6144),
    ShapeCase("sequence", 1, 16, 6144),
    ShapeCase("sequence", 1, 128, 6144),
    ShapeCase("sequence", 1, 512, 6144),
    ShapeCase("sequence", 1, 2048, 6144),
    ShapeCase("feature_dimension", 4, 128, 64),
    ShapeCase("feature_dimension", 4, 128, 128),
    ShapeCase("feature_dimension", 4, 128, 256),
    ShapeCase("feature_dimension", 4, 128, 512),
    ShapeCase("model_width", 4, 128, 6144),
    ShapeCase("model_width", 1, 128, 25600),
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


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_fp32 = a.float()
    return (a_fp32 * torch.sigmoid(a_fp32) * b.float()).to(a.dtype)


def time_samples(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    samples: int,
    inner_iterations: int,
) -> list[float]:
    for _ in range(warmup):
        fn()
    synchronize(device)

    timings_ms: list[float] = []
    for _ in range(samples):
        synchronize(device)
        started = time.perf_counter_ns()
        for _ in range(inner_iterations):
            fn()
        synchronize(device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        timings_ms.append(elapsed_ms / inner_iterations)
    return timings_ms


def git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def dtype_from_name(name: str) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return values[name]


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    dtype = dtype_from_name(args.dtype)

    results: list[dict] = []
    for case in DEFAULT_CASES:
        a = torch.randn(case.shape, device=device, dtype=dtype)
        b = torch.randn(case.shape, device=device, dtype=dtype)
        packed = torch.cat((a, b), dim=-1) if swiglu_packed_forward else None
        operation = (
            (lambda: swiglu_packed_forward(packed))
            if swiglu_packed_forward
            else (lambda: swiglu_forward(a, b))
        )
        expected = reference(a, b)
        actual = operation()
        synchronize(device)

        difference = (actual.float() - expected.float()).abs()
        expected_abs = expected.float().abs()
        max_abs_error = difference.max().item()
        mean_abs_error = difference.mean().item()
        relative_mask = expected_abs > args.relative_error_floor
        if relative_mask.any():
            max_relative_error = (
                difference[relative_mask] / expected_abs[relative_mask]
            ).max().item()
        else:
            max_relative_error = 0.0
        allclose = torch.allclose(
            actual.float(),
            expected.float(),
            atol=args.atol,
            rtol=args.rtol,
        )

        numel = actual.numel()
        inner_iterations = max(
            1,
            min(args.max_inner_iterations, args.target_elements_per_sample // numel),
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
            "numel": numel,
            "inner_iterations": inner_iterations,
            "samples": args.samples,
            "mean_ms": mean_ms,
            "p50_ms": percentile(timings_ms, 0.50),
            "p90_ms": percentile(timings_ms, 0.90),
            "p99_ms": percentile(timings_ms, 0.99),
            "stdev_ms": stdev_ms,
            "cv_percent": (stdev_ms / mean_ms * 100.0) if mean_ms else 0.0,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "max_relative_error": max_relative_error,
            "allclose": allclose,
        }
        results.append(row)
        print(json.dumps(row, sort_keys=True))
        del a, b, packed, actual, expected, difference, expected_abs

    failures = [row for row in results if not row["allclose"]]
    report = {
        "schema_version": 1,
        "metadata": {
            "git_branch": git_value("branch", "--show-current"),
            "git_commit": git_value("rev-parse", "HEAD"),
            "backend_module": swiglu_forward.__module__,
            "input_layout": "packed_gate_up" if swiglu_packed_forward else "separate_gate_up",
            "device": args.device,
            "physical_device": args.physical_device,
            "dtype": args.dtype,
            "torch_version": torch.__version__,
            "torch_npu_version": torch_npu.__version__,
            "python_version": platform.python_version(),
            "seed": args.seed,
            "warmup": args.warmup,
            "samples": args.samples,
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
        raise RuntimeError(f"{len(failures)} SwiGLU correctness cases failed")
    return report


def write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
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
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-3)
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
