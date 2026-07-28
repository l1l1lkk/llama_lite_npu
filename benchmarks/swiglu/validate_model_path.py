#!/usr/bin/env python3
"""Validate packed Gate/Up projection and SwiGLU with real Qwen3 weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from lite_llama.kernels import swiglu_packed_forward


DEFAULT_SHAPES = ((1, 1, 2048), (4, 1, 2048), (1, 128, 2048))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (actual.float() - expected.float()).abs()
    expected_abs = expected.float().abs()
    relative_mask = expected_abs > 1e-2
    max_relative_error = (
        (difference[relative_mask] / expected_abs[relative_mask]).max().item()
        if relative_mask.any()
        else 0.0
    )
    cosine = F.cosine_similarity(
        actual.float().reshape(1, -1),
        expected.float().reshape(1, -1),
    ).item()
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "max_relative_error_above_1e-2": max_relative_error,
        "cosine_similarity": cosine,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260728)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    state = torch.load(str(args.checkpoint), mmap=True, map_location="cpu")
    prefix = f"layers.{args.layer}.mlp"
    gate = state[f"{prefix}.gate_proj.weight"]
    up = state[f"{prefix}.up_proj.weight"]
    down = state[f"{prefix}.down_proj.weight"]
    if gate.shape != up.shape:
        raise ValueError(f"gate/up shapes differ: {gate.shape} != {up.shape}")
    if gate.shape[0] % args.tp_size:
        raise ValueError("intermediate dimension is not divisible by TP size")
    shard_width = gate.shape[0] // args.tp_size
    start = args.tp_rank * shard_width
    end = start + shard_width
    gate = gate[start:end].to(device=device, dtype=torch.float16)
    up = up[start:end].to(device=device, dtype=torch.float16)
    down = down[:, start:end].to(device=device, dtype=torch.float16)
    packed_weight = torch.cat((gate, up), dim=0)

    rows = []
    for shape in DEFAULT_SHAPES:
        x = torch.randn(shape, device=device, dtype=torch.float16)
        baseline_gate = F.linear(x, gate)
        baseline_up = F.linear(x, up)
        gate_fp32 = baseline_gate.float()
        baseline_activation = (
            gate_fp32 * torch.sigmoid(gate_fp32) * baseline_up.float()
        ).to(torch.float16)
        baseline_down = F.linear(baseline_activation, down)

        packed_projection = F.linear(x, packed_weight)
        fused_activation = swiglu_packed_forward(packed_projection)
        fused_down = F.linear(fused_activation, down)
        torch.npu.synchronize(device)

        activation_metrics = tensor_metrics(fused_activation, baseline_activation)
        down_metrics = tensor_metrics(fused_down, baseline_down)
        row = {
            "shape": shape,
            "activation": activation_metrics,
            "down_projection": down_metrics,
            "activation_allclose": torch.allclose(
                fused_activation.float(),
                baseline_activation.float(),
                atol=args.atol,
                rtol=args.rtol,
            ),
            "down_projection_allclose": torch.allclose(
                fused_down.float(),
                baseline_down.float(),
                atol=args.atol,
                rtol=args.rtol,
            ),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    passed = all(
        row["activation_allclose"] and row["down_projection_allclose"]
        for row in rows
    )
    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "physical_device": args.physical_device,
        "device": args.device,
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "layer": args.layer,
        "atol": args.atol,
        "rtol": args.rtol,
        "passed": passed,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError("packed Qwen3 model-path validation failed")
    print(json.dumps({"status": "pass", "cases": len(rows)}))


if __name__ == "__main__":
    main()
