#!/usr/bin/env python3
"""Aggregate and compare baseline/fused streaming benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = (
    "ttft_ms_mean",
    "tpot_ms_mean",
    "e2e_ms_mean",
    "output_throughput_tokens_per_s",
)


def load_results(root: Path) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    for path in sorted(root.rglob("result.json")):
        report = json.loads(path.read_text())
        if int(report["metadata"]["repeat"]) <= 0:
            continue
        case_id = report["metadata"]["case_id"]
        results.setdefault(case_id, []).append(report)
    if not results:
        raise ValueError(f"no result.json files under {root}")
    return results


def aggregate(reports: list[dict]) -> dict:
    first = reports[0]
    aggregate_row = {
        "case_id": first["metadata"]["case_id"],
        "repeats": len(reports),
        "concurrency": first["metadata"]["concurrency"],
        "requests_per_repeat": first["metadata"]["submitted_requests"],
        "max_tokens": first["metadata"]["max_tokens"],
        "input_tokens_min": min(
            report["summary"]["input_tokens_min"] for report in reports
        ),
        "input_tokens_max": max(
            report["summary"]["input_tokens_max"] for report in reports
        ),
    }
    for metric in METRICS:
        aggregate_row[metric] = statistics.fmean(
            report["summary"][metric] for report in reports
        )
    output_hashes = [
        [row["output_sha256"] for row in report["requests"]]
        for report in reports
    ]
    aggregate_row["repeat_outputs_identical"] = all(
        hashes == output_hashes[0] for hashes in output_hashes[1:]
    )
    aggregate_row["output_hashes"] = output_hashes[0]
    return aggregate_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--fused-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = {
        case_id: aggregate(reports)
        for case_id, reports in load_results(args.baseline_dir).items()
    }
    fused = {
        case_id: aggregate(reports)
        for case_id, reports in load_results(args.fused_dir).items()
    }
    if baseline.keys() != fused.keys():
        raise ValueError("baseline and fused case matrices differ")

    rows = []
    for case_id in sorted(baseline):
        before = baseline[case_id]
        after = fused[case_id]
        row = {
            "case_id": case_id,
            "repeats": before["repeats"],
            "concurrency": before["concurrency"],
            "requests_per_repeat": before["requests_per_repeat"],
            "max_tokens": before["max_tokens"],
            "input_tokens_min": before["input_tokens_min"],
            "input_tokens_max": before["input_tokens_max"],
            "baseline_ttft_ms": before["ttft_ms_mean"],
            "fused_ttft_ms": after["ttft_ms_mean"],
            "ttft_reduction_percent": (
                1.0 - after["ttft_ms_mean"] / before["ttft_ms_mean"]
            )
            * 100.0,
            "baseline_tpot_ms": before["tpot_ms_mean"],
            "fused_tpot_ms": after["tpot_ms_mean"],
            "tpot_reduction_percent": (
                1.0 - after["tpot_ms_mean"] / before["tpot_ms_mean"]
            )
            * 100.0,
            "baseline_e2e_ms": before["e2e_ms_mean"],
            "fused_e2e_ms": after["e2e_ms_mean"],
            "e2e_reduction_percent": (
                1.0 - after["e2e_ms_mean"] / before["e2e_ms_mean"]
            )
            * 100.0,
            "baseline_output_tokens_per_s": before[
                "output_throughput_tokens_per_s"
            ],
            "fused_output_tokens_per_s": after["output_throughput_tokens_per_s"],
            "throughput_gain_percent": (
                after["output_throughput_tokens_per_s"]
                / before["output_throughput_tokens_per_s"]
                - 1.0
            )
            * 100.0,
            "baseline_repeat_outputs_identical": before[
                "repeat_outputs_identical"
            ],
            "fused_repeat_outputs_identical": after["repeat_outputs_identical"],
            "cross_branch_outputs_identical": (
                before["output_hashes"] == after["output_hashes"]
            ),
            "cross_branch_exact_output_matches": sum(
                baseline_hash == fused_hash
                for baseline_hash, fused_hash in zip(
                    before["output_hashes"],
                    after["output_hashes"],
                )
            ),
            "cross_branch_total_outputs": len(before["output_hashes"]),
        }
        row["cross_branch_exact_output_rate"] = (
            row["cross_branch_exact_output_matches"]
            / row["cross_branch_total_outputs"]
        )
        rows.append(row)

    report = {
        "schema_version": 1,
        "all_cross_branch_outputs_identical": all(
            row["cross_branch_outputs_identical"] for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "pass", "cases": len(rows)}))


if __name__ == "__main__":
    main()
