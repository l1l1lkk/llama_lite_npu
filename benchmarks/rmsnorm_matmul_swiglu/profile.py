"""Capture an Ascend NPU profile for the branch-selected FFN boundary."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
import torch_npu

from lite_llama.kernels import rmsnorm_matmul_swiglu_forward


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-name", required=True)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--active-steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("npu")
    torch.npu.set_device(device)
    torch.manual_seed(20260730)
    x = torch.randn(
        args.batch, 1, args.hidden_size, device=device, dtype=torch.float16
    )
    residuals = [
        torch.randn_like(x)
        for _ in range(args.warmup + args.active_steps + 8)
    ]
    norm_weight = torch.randn(
        args.hidden_size, device=device, dtype=torch.float16
    )
    gate_up_weight = torch.randn(
        2 * args.intermediate_size,
        args.hidden_size,
        device=device,
        dtype=torch.float16,
    ) / args.hidden_size**0.5
    step = 0

    def operation():
        nonlocal step
        result = rmsnorm_matmul_swiglu_forward(
            x,
            residuals[step],
            norm_weight,
            gate_up_weight,
            1e-6,
        )
        step += 1
        return result

    for _ in range(args.warmup):
        operation()
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
        wait=1, warmup=1, active=args.active_steps, repeat=1
    )
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
        for _ in range(args.active_steps + 2):
            operation()
            profiler.step()
    torch.npu.synchronize(device)

    metadata = {
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "physical_device": args.physical_device,
        "batch": args.batch,
        "sequence_length": 1,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "active_steps": args.active_steps,
        "profiler_level": "Level1",
        "aic_metrics": "PipeUtilization",
        "l2_cache": True,
    }
    (args.output_dir / "capture-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
