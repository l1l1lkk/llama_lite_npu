#!/usr/bin/env python3
"""Compare unfused and fused SwiGLU benchmark reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KEY_FIELDS = ("family", "batch", "sequence_length", "feature_dimension", "dtype")


def key(row: dict) -> tuple:
    return tuple(row[field] for field in KEY_FIELDS)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load(args.baseline)
    fused = load(args.fused)
    baseline_rows = {key(row): row for row in baseline["results"]}
    fused_rows = {key(row): row for row in fused["results"]}
    if baseline_rows.keys() != fused_rows.keys():
        raise ValueError("baseline and fused reports contain different shape matrices")

    rows = []
    for case_key in baseline_rows:
        before = baseline_rows[case_key]
        after = fused_rows[case_key]
        speedup = before["mean_ms"] / after["mean_ms"]
        p50_speedup = before["p50_ms"] / after["p50_ms"]
        rows.append(
            {
                **{field: before[field] for field in KEY_FIELDS},
                "baseline_mean_ms": before["mean_ms"],
                "fused_mean_ms": after["mean_ms"],
                "latency_reduction_percent": (1.0 - 1.0 / speedup) * 100.0,
                "speedup": speedup,
                "baseline_p50_ms": before["p50_ms"],
                "fused_p50_ms": after["p50_ms"],
                "p50_latency_reduction_percent": (
                    1.0 - 1.0 / p50_speedup
                )
                * 100.0,
                "p50_speedup": p50_speedup,
                "baseline_cv_percent": before["cv_percent"],
                "fused_cv_percent": after["cv_percent"],
                "fused_max_abs_error": after["max_abs_error"],
                "fused_allclose": after["allclose"],
            }
        )

    summary = {
        "schema_version": 1,
        "baseline": baseline["metadata"],
        "fused": fused["metadata"],
        "all_correct": all(row["fused_allclose"] for row in rows),
        "min_speedup": min(row["speedup"] for row in rows),
        "geometric_mean_speedup": (
            __import__("math").prod(row["speedup"] for row in rows) ** (1.0 / len(rows))
        ),
        "max_speedup": max(row["speedup"] for row in rows),
        "min_p50_speedup": min(row["p50_speedup"] for row in rows),
        "geometric_mean_p50_speedup": (
            __import__("math").prod(row["p50_speedup"] for row in rows)
            ** (1.0 / len(rows))
        ),
        "max_p50_speedup": max(row["p50_speedup"] for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "pass", "cases": len(rows)}))


if __name__ == "__main__":
    main()
