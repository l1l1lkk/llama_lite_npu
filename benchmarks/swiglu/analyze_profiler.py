#!/usr/bin/env python3
"""Summarize matched baseline/fused Ascend profiler captures."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


RATIO_FIELDS = (
    "aiv_vec_ratio",
    "aiv_scalar_ratio",
    "aiv_mte2_ratio",
    "aiv_mte3_ratio",
)


def find_one(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise ValueError(f"expected one {filename} under {root}, found {len(matches)}")
    return matches[0]


def summarize_capture(root: Path) -> dict:
    kernel_path = find_one(root, "kernel_details.csv")
    with kernel_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if (row.get("Step Id") or "").strip()
        ]
    step_ids = sorted({int(row["Step Id"]) for row in rows})
    if not step_ids:
        raise ValueError(f"capture contains no scheduled steps: {kernel_path}")

    durations = [float(row["Duration(us)"]) for row in rows]
    type_counts: dict[str, int] = {}
    for row in rows:
        type_counts[row["Type"]] = type_counts.get(row["Type"], 0) + 1

    weighted_ratios = {}
    total_aiv_time = sum(float(row["aiv_time(us)"] or 0.0) for row in rows)
    for field in RATIO_FIELDS:
        weighted_ratios[field] = (
            sum(
                float(row["aiv_time(us)"] or 0.0) * float(row[field] or 0.0)
                for row in rows
            )
            / total_aiv_time
            if total_aiv_time
            else 0.0
        )

    dominant_ratio = max(weighted_ratios, key=weighted_ratios.get)
    classification = {
        "aiv_mte2_ratio": "memory_access",
        "aiv_mte3_ratio": "memory_write",
        "aiv_vec_ratio": "vector_compute",
        "aiv_scalar_ratio": "scalar_compute",
    }[dominant_ratio]
    return {
        "capture_root": str(root.resolve()),
        "kernel_details": str(kernel_path.resolve()),
        "active_steps": len(step_ids),
        "kernel_count": len(rows),
        "kernels_per_step": len(rows) / len(step_ids),
        "device_time_per_step_us": sum(durations) / len(step_ids),
        "kernel_duration_mean_us": statistics.fmean(durations),
        "kernel_types": type_counts,
        "weighted_pipe_ratios": weighted_ratios,
        "dominant_pipe": dominant_ratio,
        "classification": classification,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-prefill", type=Path, required=True)
    parser.add_argument("--baseline-decode", type=Path, required=True)
    parser.add_argument("--fused-prefill", type=Path, required=True)
    parser.add_argument("--fused-decode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    captures = {
        "prefill": {
            "baseline": summarize_capture(args.baseline_prefill),
            "fused": summarize_capture(args.fused_prefill),
        },
        "decode": {
            "baseline": summarize_capture(args.baseline_decode),
            "fused": summarize_capture(args.fused_decode),
        },
    }
    comparisons = {}
    for phase, values in captures.items():
        before = values["baseline"]
        after = values["fused"]
        comparisons[phase] = {
            "kernel_count_reduction_percent": (
                1.0 - after["kernels_per_step"] / before["kernels_per_step"]
            )
            * 100.0,
            "device_time_reduction_percent": (
                1.0
                - after["device_time_per_step_us"] / before["device_time_per_step_us"]
            )
            * 100.0,
            "speedup": (
                before["device_time_per_step_us"] / after["device_time_per_step_us"]
            ),
        }
    report = {
        "schema_version": 1,
        "captures": captures,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "phases": list(comparisons)}))


if __name__ == "__main__":
    main()
