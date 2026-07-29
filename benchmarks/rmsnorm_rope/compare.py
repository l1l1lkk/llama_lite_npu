#!/usr/bin/env python3
"""Compare matched baseline and fused RMSNorm + RoPE benchmark reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KEY_FIELDS = (
    "family",
    "batch",
    "sequence_length",
    "q_heads",
    "k_heads",
    "head_dimension",
    "dtype",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def key(row: dict) -> tuple:
    return tuple(row[field] for field in KEY_FIELDS)


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
        raise ValueError("baseline and fused shape matrices differ")
    rows = []
    for case_key in baseline_rows:
        before = baseline_rows[case_key]
        after = fused_rows[case_key]
        row = {field: before[field] for field in KEY_FIELDS}
        row.update(
            {
                "baseline_p50_ms": before["p50_ms"],
                "fused_p50_ms": after["p50_ms"],
                "latency_reduction_percent": (
                    1.0 - after["p50_ms"] / before["p50_ms"]
                )
                * 100.0,
                "speedup": before["p50_ms"] / after["p50_ms"],
                "baseline_cv_percent": before["cv_percent"],
                "fused_cv_percent": after["cv_percent"],
                "fused_q_max_abs_error": after["q_max_abs_error"],
                "fused_k_max_abs_error": after["k_max_abs_error"],
                "allclose": before["allclose"] and after["allclose"],
            }
        )
        rows.append(row)
    report = {
        "schema_version": 1,
        "baseline_metadata": baseline["metadata"],
        "fused_metadata": fused["metadata"],
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "pass", "cases": len(rows)}))


if __name__ == "__main__":
    main()
