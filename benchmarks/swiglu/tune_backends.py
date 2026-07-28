#!/usr/bin/env python3
"""Compare eager, Triton-tiled, and packed CANN SwiGLU candidates."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from benchmarks.swiglu.tune_tiles import candidate as triton_candidate


def unfused(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_fp32 = a.float()
    return (a_fp32 * torch.sigmoid(a_fp32) * b.float()).to(a.dtype)


def packed_cann(gate_up: torch.Tensor) -> torch.Tensor:
    return torch_npu.npu_swiglu(gate_up, dim=-1)


def cat_cann(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return packed_cann(torch.cat((a, b), dim=-1))


def benchmark(fn, *, device: torch.device, warmup: int, samples: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize(device)
    values = []
    for _ in range(samples):
        torch.npu.synchronize(device)
        started = time.perf_counter_ns()
        fn()
        torch.npu.synchronize(device)
        values.append((time.perf_counter_ns() - started) / 1_000_000)
    return statistics.fmean(values), statistics.stdev(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.npu.set_device(device)
    shapes = [(16, 1, 6144), (4, 128, 6144), (1, 128, 25600)]
    rows = []
    for shape in shapes:
        a = torch.randn(shape, device=device, dtype=torch.float16)
        b = torch.randn(shape, device=device, dtype=torch.float16)
        packed = torch.cat((a, b), dim=-1)
        expected = unfused(a, b)
        backends = {
            "unfused": lambda: unfused(a, b),
            "triton_tile_4096": lambda: triton_candidate(a, b, 4096),
            "cat_cann": lambda: cat_cann(a, b),
            "packed_cann": lambda: packed_cann(packed),
        }
        for name, fn in backends.items():
            actual = fn()
            torch.npu.synchronize(device)
            difference = (actual.float() - expected.float()).abs()
            allclose = torch.allclose(
                actual.float(),
                expected.float(),
                atol=2e-3,
                rtol=2e-3,
            )
            mean_ms, stdev_ms = benchmark(
                fn,
                device=device,
                warmup=args.warmup,
                samples=args.samples,
            )
            row = {
                "backend": name,
                "batch": shape[0],
                "sequence_length": shape[1],
                "feature_dimension": shape[2],
                "mean_ms": mean_ms,
                "stdev_ms": stdev_ms,
                "cv_percent": stdev_ms / mean_ms * 100.0,
                "max_abs_error": difference.max().item(),
                "mean_abs_error": difference.mean().item(),
                "allclose": allclose,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
        del a, b, packed, expected

    report = {
        "schema_version": 1,
        "physical_device": args.physical_device,
        "device": args.device,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
