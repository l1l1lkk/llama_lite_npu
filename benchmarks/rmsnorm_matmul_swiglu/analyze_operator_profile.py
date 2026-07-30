"""Summarize matched operator-level Ascend profiler captures."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def find_one(root: Path, name: str) -> Path:
    matches = list(root.glob(f"**/{name}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {matches}")
    return matches[0]


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "").strip()
    return float(value) if value and value != "N/A" else 0.0


def summarize(root: Path) -> dict:
    rows = list(csv.DictReader(find_one(root, "kernel_details.csv").open()))
    by_name: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_name.setdefault(row["Name"], []).append(row)
    return {
        "capture_metadata": json.loads(
            (root / "capture-metadata.json").read_text(encoding="utf-8")
        ),
        "kernel_count": len(rows),
        "kernels": {
            name: {
                "count": len(group),
                "duration_us_mean": statistics.fmean(
                    as_float(row, "Duration(us)") for row in group
                ),
                "duration_us_median": statistics.median(
                    as_float(row, "Duration(us)") for row in group
                ),
                "cube_utilization_percent_mean": statistics.fmean(
                    as_float(row, "cube_utilization(%)") for row in group
                ),
                "aic_mac_ratio_mean": statistics.fmean(
                    as_float(row, "aic_mac_ratio") for row in group
                ),
                "aiv_mte2_ratio_mean": statistics.fmean(
                    as_float(row, "aiv_mte2_ratio") for row in group
                ),
                "aiv_scalar_ratio_mean": statistics.fmean(
                    as_float(row, "aiv_scalar_ratio") for row in group
                ),
            }
            for name, group in sorted(by_name.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--separated", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "separated": summarize(args.separated),
        "fused_triton": summarize(args.fused),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "output": str(args.output)}))


if __name__ == "__main__":
    main()
