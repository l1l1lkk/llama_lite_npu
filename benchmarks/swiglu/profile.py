#!/usr/bin/env python3
"""Capture an Ascend NPU profile for the branch-selected SwiGLU backend."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
import torch_npu

from lite_llama.kernels import swiglu_forward


def git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def parse_shape(value: str) -> tuple[int, int, int]:
    fields = value.lower().split("x")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("shape must be BxSxD, for example 4x128x6144")
    shape = tuple(int(field) for field in fields)
    if any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--shape", type=parse_shape, default=(4, 128, 6144))
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-name", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--active-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(args.seed)
    a = torch.randn(args.shape, device=device, dtype=dtype)
    b = torch.randn(args.shape, device=device, dtype=dtype)

    for _ in range(args.warmup):
        swiglu_forward(a, b)
    torch.npu.synchronize(device)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=True,
        data_simplification=False,
    )
    handler = torch_npu.profiler.tensorboard_trace_handler(
        str(args.output_dir),
        worker_name=args.worker_name,
        analyse_flag=True,
        async_mode=False,
    )
    schedule = torch_npu.profiler.schedule(
        wait=1,
        warmup=1,
        active=args.active_steps,
        repeat=1,
    )
    total_steps = 2 + args.active_steps
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        experimental_config=experimental_config,
    ) as profiler:
        for _ in range(total_steps):
            swiglu_forward(a, b)
            profiler.step()
    torch.npu.synchronize(device)

    metadata = {
        "schema_version": 1,
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "backend_module": swiglu_forward.__module__,
        "device": args.device,
        "physical_device": args.physical_device,
        "shape": args.shape,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "active_steps": args.active_steps,
        "profiler_level": "Level1",
        "aic_metrics": "PipeUtilization",
        "l2_cache": True,
    }
    (args.output_dir / "capture-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": "pass", "output_dir": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
