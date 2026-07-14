#!/usr/bin/env python3
"""Compare a candidate length-matrix result with a matching baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LOWER_IS_BETTER = ("client_ttft_ms", "client_tpot_ms")
HIGHER_IS_BETTER = ("client_output_throughput_tok_s",)


def compare(baseline: dict, candidate: dict) -> dict:
    invalid = []
    if baseline.get("fingerprint") != candidate.get("fingerprint"):
        invalid.append("fingerprint_mismatch")
    baseline_cells = baseline.get("cells", {})
    candidate_cells = candidate.get("cells", {})
    if set(baseline_cells) != set(candidate_cells):
        invalid.append("cell_set_mismatch")
    hard_fail = []
    rows = []
    for cell_id in sorted(set(baseline_cells) & set(candidate_cells)):
        base_cell, cand_cell = baseline_cells[cell_id], candidate_cells[cell_id]
        gates = cand_cell.get("gates", {})
        for name in ("strict_input", "strict_output", "zero_failed", "graph_valid"):
            if gates.get(name) is not True:
                hard_fail.append(f"{cell_id}:{name}")
        for metric in LOWER_IS_BETTER + HIGHER_IS_BETTER:
            base_metric = base_cell["metrics"][metric]
            cand_metric = cand_cell["metrics"][metric]
            base_mean = float(base_metric["mean"])
            cand_mean = float(cand_metric["mean"])
            cv = abs(float(base_metric.get("cv", 0)))
            threshold = max(0.10, 3 * cv)
            relative_change = cand_mean / base_mean - 1
            if metric in LOWER_IS_BETTER:
                regressed = relative_change > threshold
            else:
                regressed = relative_change < -threshold
            rows.append({
                "cell_id": cell_id,
                "metric": metric,
                "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                "baseline_mean": base_mean,
                "candidate_mean": cand_mean,
                "baseline_cv": cv,
                "threshold_fraction": threshold,
                "relative_change_fraction": relative_change,
                "status": "regression" if regressed else "pass",
            })
    regressions = [row for row in rows if row["status"] == "regression"]
    if invalid:
        status = "invalid"
    elif hard_fail:
        status = "hard_fail"
    elif regressions:
        status = "regression"
    else:
        status = "pass"
    return {
        "schema_version": 1,
        "status": status,
        "invalid_reasons": invalid,
        "hard_fail_reasons": hard_fail,
        "performance_regressions": regressions,
        "comparisons": rows,
        "threshold_rule": "max(10%, 3 * baseline CV)",
        "scope": "same-server empirical regression gate, not a cross-hardware SLA",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect", choices=("pass", "regression", "hard_fail", "invalid"))
    args = parser.parse_args()
    report = compare(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
    )
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    if args.expect and report["status"] != args.expect:
        raise SystemExit(f"expected {args.expect}, got {report['status']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
