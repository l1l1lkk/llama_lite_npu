#!/usr/bin/env python3
"""Summarize the unfused/fused x NPU Graph off/on serving ablation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


IMPLEMENTATIONS = ("baseline", "fused-triton")
GRAPH_MODES = ("off", "on")
CASES = ("p128-c1-o32", "p512-c1-o32")
METRICS = ("ttft_ms", "tpot_ms", "e2e_ms")


def stable(value: float, digits: int = 9) -> float:
    return round(float(value), digits)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def improvement_percent(reference: float, candidate: float) -> float:
    return stable((reference - candidate) / reference * 100.0)


def aggregate_cell(root: Path, implementation: str, graph_mode: str, case: str) -> dict:
    paths = sorted(
        (root / implementation / f"graph-{graph_mode}" / case).glob(
            "repeat-*/result.json"
        )
    )
    if not paths:
        raise FileNotFoundError(
            f"no result files for {implementation}/graph-{graph_mode}/{case}"
        )
    reports = [read_json(path) for path in paths]
    rows = [row for report in reports for row in report["requests"]]
    result = {
        "repeats": len(reports),
        "requests": len(rows),
        "throughput_tokens_per_s_mean": stable(
            statistics.fmean(
                report["summary"]["output_throughput_tokens_per_s"]
                for report in reports
            )
        ),
    }
    for metric in METRICS:
        request_values = [float(row[metric]) for row in rows]
        repeat_means = [
            float(report["summary"][f"{metric}_mean"]) for report in reports
        ]
        result[f"{metric}_mean"] = stable(statistics.fmean(request_values))
        result[f"{metric}_repeat_means"] = [stable(value) for value in repeat_means]
        result[f"{metric}_repeat_cv_percent"] = stable(
            statistics.stdev(repeat_means)
            / statistics.fmean(repeat_means)
            * 100.0
            if len(repeat_means) > 1
            else 0.0
        )
    return result


def graph_counters(root: Path, implementation: str, graph_mode: str) -> dict:
    server_root = root / implementation / f"graph-{graph_mode}" / "server"
    start = read_json(server_root / "formal-start-stats.json")["npu_graph"]
    end = read_json(server_root / "formal-end-stats.json")["npu_graph"]
    return {
        "formal_start": start,
        "formal_end": end,
        "formal_delta": {key: int(end[key]) - int(start[key]) for key in start},
    }


def load_outputs(
    root: Path, implementation: str, graph_mode: str, case: str
) -> dict[tuple[str, int], str]:
    outputs = {}
    paths = sorted(
        (root / implementation / f"graph-{graph_mode}" / case).glob(
            "repeat-*/result.json"
        )
    )
    for path in paths:
        for row in read_json(path)["requests"]:
            outputs[(path.parent.name, int(row["request_id"]))] = row["output_sha256"]
    return outputs


def parity(left: dict, right: dict) -> dict:
    keys = sorted(left.keys() & right.keys())
    matched = sum(left[key] == right[key] for key in keys)
    return {
        "matched": matched,
        "compared": len(keys),
        "match_percent": matched / len(keys) * 100.0 if keys else 0.0,
    }


def latest_csv(root: Path, name: str) -> Path:
    matches = sorted(root.glob(f"**/{name}_*.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} CSV under {root}, found {matches}")
    return matches[0]


def summarize_profiler(root: Path) -> dict:
    profiler_root = root / "profiler"
    if not profiler_root.exists():
        return {"status": "not_present"}

    modes = {}
    for graph_mode in GRAPH_MODES:
        mode_root = profiler_root / f"graph-{graph_mode}"
        request = read_json(mode_root / "profiled-request.json")["summary"]
        before = read_json(mode_root / "pre-profile-stats.json")["npu_graph"]
        after = read_json(mode_root / "post-profile-stats.json")["npu_graph"]

        api_rows = list(csv.DictReader(latest_csv(mode_root, "api_statistic").open()))
        api_by_name = {row["API Name"]: row for row in api_rows}
        op_rows = list(csv.DictReader(latest_csv(mode_root, "op_statistic").open()))
        modes[graph_mode] = {
            "profiled_request": {
                metric: float(request[f"{metric}_mean"]) for metric in METRICS
            },
            "graph_delta": {
                key: int(after[key]) - int(before[key]) for key in before
            },
            "runtime_api": {
                name: {
                    "time_us": float(api_by_name[name]["Time(us)"]),
                    "count": int(api_by_name[name]["Count"]),
                    "avg_us": float(api_by_name[name]["Avg(us)"]),
                }
                for name in ("FftsPlusTaskLaunch", "EventRecord", "ContextGetCurrent")
            },
            "device_ops": {
                "count": sum(int(row["Count"]) for row in op_rows),
                "total_time_us": stable(
                    sum(float(row["Total Time(us)"]) for row in op_rows), digits=6
                ),
            },
        }

    reductions = {}
    for api_name in ("FftsPlusTaskLaunch", "EventRecord", "ContextGetCurrent"):
        off = modes["off"]["runtime_api"][api_name]
        on = modes["on"]["runtime_api"][api_name]
        reductions[api_name] = {
            "count_reduction_percent": improvement_percent(
                float(off["count"]), float(on["count"])
            ),
            "time_reduction_percent": improvement_percent(
                off["time_us"], on["time_us"]
            ),
        }
    return {"status": "pass", "modes": modes, "runtime_api_reductions": reductions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()

    cells = {
        implementation: {
            graph_mode: {
                case: aggregate_cell(root, implementation, graph_mode, case)
                for case in CASES
            }
            for graph_mode in GRAPH_MODES
        }
        for implementation in IMPLEMENTATIONS
    }
    comparisons = {"graph_on_vs_off": {}, "fusion_vs_baseline": {}, "joint": {}}
    for implementation in IMPLEMENTATIONS:
        comparisons["graph_on_vs_off"][implementation] = {}
        for case in CASES:
            off = cells[implementation]["off"][case]
            on = cells[implementation]["on"][case]
            comparisons["graph_on_vs_off"][implementation][case] = {
                metric: improvement_percent(
                    off[f"{metric}_mean"], on[f"{metric}_mean"]
                )
                for metric in METRICS
            }
    for graph_mode in GRAPH_MODES:
        comparisons["fusion_vs_baseline"][graph_mode] = {}
        for case in CASES:
            baseline = cells["baseline"][graph_mode][case]
            fused = cells["fused-triton"][graph_mode][case]
            comparisons["fusion_vs_baseline"][graph_mode][case] = {
                metric: improvement_percent(
                    baseline[f"{metric}_mean"], fused[f"{metric}_mean"]
                )
                for metric in METRICS
            }
    for case in CASES:
        baseline_off = cells["baseline"]["off"][case]
        fused_on = cells["fused-triton"]["on"][case]
        comparisons["joint"][case] = {
            metric: improvement_percent(
                baseline_off[f"{metric}_mean"], fused_on[f"{metric}_mean"]
            )
            for metric in METRICS
        }

    parity_results = {}
    for case in CASES:
        baseline_off = load_outputs(root, "baseline", "off", case)
        baseline_on = load_outputs(root, "baseline", "on", case)
        fused_off = load_outputs(root, "fused-triton", "off", case)
        fused_on = load_outputs(root, "fused-triton", "on", case)
        parity_results[case] = {
            "baseline_graph_on_vs_off": parity(baseline_off, baseline_on),
            "fused_graph_on_vs_off": parity(fused_off, fused_on),
            "fused_vs_baseline_graph_off": parity(baseline_off, fused_off),
            "fused_vs_baseline_graph_on": parity(baseline_on, fused_on),
        }

    report = {
        "schema_version": 1,
        "cells": cells,
        "comparisons_percent": comparisons,
        "graph_counters": {
            implementation: {
                graph_mode: graph_counters(root, implementation, graph_mode)
                for graph_mode in GRAPH_MODES
            }
            for implementation in IMPLEMENTATIONS
        },
        "output_parity": parity_results,
        "profiler": summarize_profiler(root),
    }
    with (root / "comparison.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")

    with (root / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "implementation",
                "graph_mode",
                "case",
                "requests",
                "ttft_ms_mean",
                "tpot_ms_mean",
                "e2e_ms_mean",
                "throughput_tokens_per_s_mean",
            ]
        )
        for implementation in IMPLEMENTATIONS:
            for graph_mode in GRAPH_MODES:
                for case in CASES:
                    cell = cells[implementation][graph_mode][case]
                    writer.writerow(
                        [
                            implementation,
                            graph_mode,
                            case,
                            cell["requests"],
                            f"{cell['ttft_ms_mean']:.6f}",
                            f"{cell['tpot_ms_mean']:.6f}",
                            f"{cell['e2e_ms_mean']:.6f}",
                            f"{cell['throughput_tokens_per_s_mean']:.6f}",
                        ]
                    )
    print(json.dumps({"status": "pass", "result_root": str(root)}))


if __name__ == "__main__":
    main()
