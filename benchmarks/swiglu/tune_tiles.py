#!/usr/bin/env python3
"""Measure candidate 2-D tile sizes for the Ascend SwiGLU kernel."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _candidate_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_row_stride,
    b_row_stride,
    c_row_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    column_block = tl.program_id(1).to(tl.int64)
    offsets = column_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    a_row = tl.load(
        a_ptr + row * a_row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    b_row = tl.load(
        b_ptr + row * b_row_stride + offsets,
        mask=mask,
        other=0.0,
    )
    output = a_row * tl.sigmoid(a_row) * b_row
    tl.store(
        c_ptr + row * c_row_stride + offsets,
        output,
        mask=mask,
    )


def candidate(a: torch.Tensor, b: torch.Tensor, block_size: int) -> torch.Tensor:
    shape = a.shape
    n_cols = shape[-1]
    a_rows = a.view(-1, n_cols)
    b_rows = b.view(-1, n_cols)
    output = torch.empty_like(a_rows)
    grid = (a_rows.shape[0], triton.cdiv(n_cols, block_size))
    _candidate_kernel[grid](
        a_rows,
        b_rows,
        output,
        a_rows.stride(-2),
        b_rows.stride(-2),
        output.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=block_size,
    )
    return output.view(shape)


def benchmark(
    fn,
    *,
    device: torch.device,
    warmup: int,
    samples: int,
) -> tuple[float, float]:
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


def parse_shape(value: str) -> tuple[int, int, int]:
    shape = tuple(int(field) for field in value.lower().split("x"))
    if len(shape) != 3 or any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError("shape must be positive BxSxD")
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        default=[],
        help="Repeatable BxSxD shape. Defaults cover decode and prefill.",
    )
    parser.add_argument("--tiles", default="256,512,1024,2048,4096")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.npu.set_device(device)
    shapes = args.shape or [(16, 1, 6144), (4, 128, 6144), (1, 128, 25600)]
    tiles = [int(value) for value in args.tiles.split(",")]
    rows = []
    for shape in shapes:
        a = torch.randn(shape, device=device, dtype=torch.float16)
        b = torch.randn(shape, device=device, dtype=torch.float16)
        a_fp32 = a.float()
        expected = (a_fp32 * torch.sigmoid(a_fp32) * b.float()).to(a.dtype)
        for tile in tiles:
            actual = candidate(a, b, tile)
            torch.npu.synchronize(device)
            max_abs_error = (actual.float() - expected.float()).abs().max().item()
            allclose = torch.allclose(
                actual.float(),
                expected.float(),
                atol=2e-3,
                rtol=2e-3,
            )
            mean_ms, stdev_ms = benchmark(
                lambda: candidate(a, b, tile),
                device=device,
                warmup=args.warmup,
                samples=args.samples,
            )
            row = {
                "batch": shape[0],
                "sequence_length": shape[1],
                "feature_dimension": shape[2],
                "tile": tile,
                "mean_ms": mean_ms,
                "stdev_ms": stdev_ms,
                "cv_percent": stdev_ms / mean_ms * 100.0,
                "max_abs_error": max_abs_error,
                "allclose": allclose,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
        del a, b, a_fp32, expected

    report = {
        "schema_version": 1,
        "physical_device": args.physical_device,
        "device": args.device,
        "tiles": tiles,
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
