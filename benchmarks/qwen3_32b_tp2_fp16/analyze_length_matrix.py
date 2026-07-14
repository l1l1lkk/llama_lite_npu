#!/usr/bin/env python3
"""Rebuild the strict input-length by output-length benchmark surface."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import analyze_scalability as core
from compare_performance_baseline import compare as compare_baseline


PROMPTS = (128, 512, 1024, 2048)
OUTPUTS = (64, 256, 512)
PERCENTILES = (0.5, 0.9, 0.99)
RUN_METRICS = (
    "client_e2e_latency_s", "client_ttft_ms", "client_tpot_ms", "client_itl_ms",
    "client_output_throughput_tok_s", "client_total_throughput_tok_s",
    "client_input_throughput_tok_s", "client_qps", "server_queue_wait_mean_ms",
    "server_ttft_mean_ms", "server_service_to_first_token_mean_ms",
)
TIMESERIES_METRICS = (
    "waiting_mean", "waiting_peak", "prefilling_mean", "prefilling_peak",
    "running_mean", "running_peak", "system_requests_mean", "system_requests_peak",
    "kv_used_pages_mean", "kv_used_pages_peak", "npu_utilization_mean_pct",
    "npu_utilization_peak_pct", "npu_hbm_usage_mean_pct", "npu_hbm_usage_peak_pct",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(value):
    return core.stable(value)


def mean(values):
    return stable(statistics.mean(values))


def stdev(values):
    return stable(statistics.stdev(values))


def pct(values, q):
    return stable(core.quantile(values, q))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_server_command(command: str) -> str:
    command = re.sub(r"LITE_LLAMA_REQUEST_TIMING_TRACE=\S+", "TRACE=<path>", command)
    return " ".join(command.split())


def prompt_sequence(fingerprint: dict) -> list[list[int]]:
    return [request["prompt_token_ids"] for request in fingerprint["requests"]]


def timeseries_distributions(path: Path) -> dict[str, list[float]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    values = defaultdict(list)
    for record in records:
        server = record.get("server", {})
        if server.get("status") == "ok":
            scheduler = server.get("scheduler", {})
            waiting = float(scheduler.get("waiting", 0))
            prefilling = float(scheduler.get("prefilling", 0))
            running = float(scheduler.get("running", 0))
            values["waiting"].append(waiting)
            values["prefilling"].append(prefilling)
            values["running"].append(running)
            values["system_requests"].append(waiting + prefilling + running)
            values["kv_used_pages"].append(
                float(server.get("kv_cache", {}).get("used_pages", 0))
            )
        for device in (record.get("npu") or {}).values():
            if device.get("status") != "ok":
                continue
            if device.get("npu_utilization_pct") is not None:
                values["npu_utilization_pct"].append(
                    float(device["npu_utilization_pct"])
                )
            if device.get("hbm_usage_pct") is not None:
                values["npu_hbm_usage_pct"].append(float(device["hbm_usage_pct"]))
    return dict(values)


def build(campaign: Path, task4_reference: Path | None = None):
    plan = read_json(campaign / "campaign-plan.json")
    workload = read_json(campaign / "workload/workload-manifest.json")
    workload_by_prompt = {int(item["prompt_tokens"]): item for item in workload["workloads"]}
    metadata_paths = sorted(campaign.glob("matrix/p*/run-*/run-metadata.json"))
    if len(metadata_paths) != 36:
        raise ValueError(f"expected 36 formal runs, found {len(metadata_paths)}")
    items = {}
    run_rows = []
    validations = []
    commands = set()
    prompt_sequences = defaultdict(set)
    for metadata_path in metadata_paths:
        root = metadata_path.parent
        metadata = read_json(metadata_path)
        prompt = int(metadata["target_server_input_tokens"])
        output = int(metadata["output_tokens"])
        repeat = int(metadata["run_order_in_lifecycle"])
        key = (prompt, output, repeat)
        if key in items:
            raise ValueError(f"duplicate run {key}")
        summary = read_json(core.single(root / "client/evalscope", "**/benchmark_summary.json"))
        args = read_json(core.single(root / "client/evalscope", "**/benchmark_args.json"))
        request_metrics = read_json(root / "client/request-metrics.json")
        fingerprint = read_json(root / "client/workload-fingerprint.json")
        server_trace = read_json(root / "server/request-timing.json")
        timing = read_json(root / "run-timing.json")
        before_metrics = root / "server/before-metrics.prom"
        after_metrics = root / "server/after-metrics.prom"
        queue_hist = core.histogram_delta(before_metrics, after_metrics, "lite_llama_queue_wait_seconds")
        ttft_hist = core.histogram_delta(before_metrics, after_metrics, "lite_llama_time_to_first_token_seconds")
        before_graph = core.graph_stats(root / "server/before-stats.json")
        after_graph = core.graph_stats(root / "server/after-stats.json")
        timeseries_path = root / "server/timeseries.jsonl"
        timeseries, timeseries_errors = core.summarize_timeseries(
            timeseries_path, timing, float(metadata["timeseries_interval_s"])
        )
        timeseries_values = timeseries_distributions(timeseries_path)
        lifecycle = metadata["server_lifecycle_id"]
        command_path = campaign / "matrix/lifecycles" / lifecycle / "server/start-command.txt"
        command = normalized_server_command(command_path.read_text(encoding="utf-8"))
        commands.add(command)
        clients = sorted(request_metrics["requests"], key=lambda item: item["start_time_s"])
        servers = sorted(server_trace["requests"], key=lambda item: item["formal_submission_order"])
        client_values = {
            "client_e2e_latency_s": [float(item["latency_s"]) for item in clients],
            "client_ttft_ms": [float(item["ttft_ms"]) for item in clients],
            "client_tpot_ms": [float(item["tpot_ms"]) for item in clients],
            "client_itl_ms": [float(item["itl_mean_ms"]) for item in clients],
        }
        server_values = {
            "server_queue_wait_ms": [float(item["queue_wait_ms"]) for item in servers],
            "server_ttft_ms": [float(item["ttft_ms"]) for item in servers],
            "server_service_to_first_token_ms": [float(item["service_to_first_token_ms"]) for item in servers],
        }
        queue_mean = queue_hist["sum"] / queue_hist["count"] * 1000
        server_ttft_mean = ttft_hist["sum"] / ttft_hist["count"] * 1000
        row = {
            "run_id": metadata["run_id"], "prompt_tokens": prompt, "output_tokens": output,
            "repeat": repeat, "lifecycle_order_index": int(metadata["lifecycle_order_index"]),
            "requests": int(metadata["requests"]), "warmup_requests": int(metadata["warmup_requests"]),
            "client_e2e_latency_s": stable(summary["Avg Latency (s)"]),
            "client_ttft_ms": stable(summary["TTFT (ms)"]),
            "client_tpot_ms": stable(summary["TPOT (ms)"]),
            "client_itl_ms": stable(summary["ITL (ms)"]),
            "client_output_throughput_tok_s": stable(summary["Output Throughput (tok/s)"]),
            "client_total_throughput_tok_s": stable(summary["Total Throughput (tok/s)"]),
            "client_input_throughput_tok_s": stable(summary["Req Throughput (req/s)"] * prompt),
            "client_qps": stable(summary["Req Throughput (req/s)"]),
            "server_queue_wait_mean_ms": stable(queue_mean),
            "server_ttft_mean_ms": stable(server_ttft_mean),
            "server_service_to_first_token_mean_ms": stable(statistics.mean(server_values["server_service_to_first_token_ms"])),
            "graph_capture_delta": after_graph["captures"] - before_graph["captures"],
            "graph_replay_delta": after_graph["replays"] - before_graph["replays"],
            "graph_fallback_delta": after_graph["fallbacks"] - before_graph["fallbacks"],
            "formal_dataset_sha256": metadata["formal_dataset_sha256"],
        }
        for metric, values in {**client_values, **server_values}.items():
            for q in PERCENTILES:
                row[f"{metric}_p{int(q * 100)}"] = pct(values, q)
            row[f"{metric}_max"] = stable(max(values))
        for metric in TIMESERIES_METRICS:
            row[f"timeseries_{metric}"] = timeseries[metric]
        run_rows.append(row)
        expected_workload = workload_by_prompt[prompt]
        checks = {
            "request_count_32": len(clients) == len(servers) == int(summary["Total Requests"]) == 32,
            "warmup_count_8": int(metadata["warmup_requests"]) == 8,
            "strict_input": all(int(item["prompt_tokens"]) == prompt for item in clients),
            "strict_output": all(int(item["completion_tokens"]) == output for item in clients),
            "zero_failed": int(summary["Failed Requests"]) == 0 and all(int(item["success"]) == 1 for item in clients),
            "fixed_output_args": args.get("min_tokens") == args.get("max_tokens") == output,
            "formal_sha_matches_manifest": metadata["formal_dataset_sha256"] == expected_workload["formal"]["sha256"],
            "warmup_sha_matches_manifest": metadata["warmup_dataset_sha256"] == expected_workload["warmup"]["sha256"],
            "graph_captured_before_formal": before_graph["captures"] > 0,
            "formal_capture_delta_zero": row["graph_capture_delta"] == 0,
            "formal_fallback_zero": row["graph_fallback_delta"] == 0,
            "formal_replay_grew": row["graph_replay_delta"] > 0,
            "timeseries_complete": not timeseries_errors,
        }
        validations.append({"run_id": metadata["run_id"], "checks": checks, "timeseries_errors": timeseries_errors})
        sequence_digest = hashlib.sha256(json.dumps(prompt_sequence(fingerprint), separators=(",", ":")).encode()).hexdigest()
        prompt_sequences[prompt].add(sequence_digest)
        items[key] = {
            "metadata": metadata,
            "row": row,
            "client": client_values,
            "server": server_values,
            "timeseries": timeseries,
            "timeseries_values": timeseries_values,
        }

    aggregate_rows = []
    grouped = defaultdict(list)
    for (prompt, output, _), item in items.items():
        grouped[(prompt, output)].append(item)
    for prompt in PROMPTS:
        for output in OUTPUTS:
            group = grouped[(prompt, output)]
            if len(group) != 3:
                raise ValueError(f"p{prompt}/o{output}: expected three runs")
            row = {"prompt_tokens": prompt, "output_tokens": output, "formal_runs": 3}
            for metric in RUN_METRICS:
                values = [float(item["row"][metric]) for item in group]
                row[f"{metric}_mean"] = mean(values)
                row[f"{metric}_stdev"] = stdev(values)
                row[f"{metric}_cv"] = stable(abs(statistics.stdev(values) / statistics.mean(values))) if statistics.mean(values) else 0.0
            for metric in ("client_e2e_latency_s", "client_ttft_ms", "client_tpot_ms", "client_itl_ms"):
                values = [value for item in group for value in item["client"][metric]]
                for q in PERCENTILES:
                    row[f"{metric}_p{int(q*100)}_requests"] = pct(values, q)
            for metric, key in (("server_queue_wait_ms", "server_queue_wait_ms"), ("server_ttft_ms", "server_ttft_ms"), ("server_service_to_first_token_ms", "server_service_to_first_token_ms")):
                values = [value for item in group for value in item["server"][key]]
                for q in PERCENTILES:
                    row[f"{metric}_p{int(q*100)}_requests"] = pct(values, q)
                row[f"{metric}_max_requests"] = stable(max(values))
            for metric in TIMESERIES_METRICS:
                values = [float(item["timeseries"][metric]) for item in group if item["timeseries"][metric] is not None]
                row[f"timeseries_{metric}_mean"] = mean(values) if values else None
                row[f"timeseries_{metric}_stdev"] = stdev(values) if len(values) > 1 else None
                row[f"timeseries_{metric}_peak"] = stable(max(values)) if values else None
            for metric in (
                "waiting", "prefilling", "running", "system_requests",
                "kv_used_pages", "npu_utilization_pct", "npu_hbm_usage_pct",
            ):
                values = [
                    value
                    for item in group
                    for value in item["timeseries_values"].get(metric, [])
                ]
                for q in PERCENTILES:
                    row[f"timeseries_{metric}_p{int(q * 100)}"] = pct(values, q) if values else None
            for metric in (
                "graph_capture_delta", "graph_replay_delta", "graph_fallback_delta"
            ):
                values = [float(item["row"][metric]) for item in group]
                row[f"{metric}_mean"] = mean(values)
                row[f"{metric}_stdev"] = stdev(values)
            row["graph_capture_delta_sum"] = sum(item["row"]["graph_capture_delta"] for item in group)
            row["graph_replay_delta_sum"] = sum(item["row"]["graph_replay_delta"] for item in group)
            row["graph_fallback_delta_sum"] = sum(item["row"]["graph_fallback_delta"] for item in group)
            aggregate_rows.append(row)

    aggregate = {(int(row["prompt_tokens"]), int(row["output_tokens"])): row for row in aggregate_rows}
    prompt_scaling = []
    for output in OUTPUTS:
        for left, right in zip(PROMPTS, PROMPTS[1:]):
            a, b = aggregate[(left, output)], aggregate[(right, output)]
            delta_tokens = right - left
            row = {"output_tokens": output, "prompt_from": left, "prompt_to": right, "input_delta_tokens": delta_tokens}
            for metric in ("client_ttft_ms", "server_service_to_first_token_mean_ms"):
                av, bv = float(a[f"{metric}_mean"]), float(b[f"{metric}_mean"])
                row[f"{metric}_absolute_change"] = stable(bv - av)
                row[f"{metric}_relative_change_pct"] = stable((bv / av - 1) * 100)
                row[f"{metric}_slope_per_1k_input_tokens"] = stable((bv - av) / delta_tokens * 1000)
            prompt_scaling.append(row)
    output_scaling = []
    for prompt in PROMPTS:
        for left, right in zip(OUTPUTS, OUTPUTS[1:]):
            a, b = aggregate[(prompt, left)], aggregate[(prompt, right)]
            delta_tokens = right - left
            row = {"prompt_tokens": prompt, "output_from": left, "output_to": right, "output_delta_tokens": delta_tokens}
            for metric in ("client_e2e_latency_s", "client_tpot_ms", "client_itl_ms", "client_output_throughput_tok_s", "timeseries_kv_used_pages_mean"):
                av, bv = float(a[f"{metric}_mean"]), float(b[f"{metric}_mean"])
                row[f"{metric}_absolute_change"] = stable(bv - av)
                row[f"{metric}_relative_change_pct"] = stable((bv / av - 1) * 100) if av else None
            row["e2e_increment_s_per_256_output_tokens"] = stable((float(b["client_e2e_latency_s_mean"]) - float(a["client_e2e_latency_s_mean"])) / delta_tokens * 256)
            output_scaling.append(row)
    heatmap_rows = []
    heatmap_json = {metric: {} for metric in RUN_METRICS}
    for row in aggregate_rows:
        prompt, output = int(row["prompt_tokens"]), int(row["output_tokens"])
        heat = {"prompt_tokens": prompt, "output_tokens": output}
        for metric in RUN_METRICS:
            value = row[f"{metric}_mean"]
            heat[metric] = value
            heatmap_json[metric].setdefault(str(prompt), {})[str(output)] = value
        heat["kv_used_pages_mean"] = row["timeseries_kv_used_pages_mean_mean"]
        heat["hbm_usage_mean_pct"] = row["timeseries_npu_hbm_usage_mean_pct_mean"]
        heatmap_rows.append(heat)

    git_lines = (campaign / "environment/git.txt").read_text(encoding="utf-8").splitlines()
    fingerprint = {
        "model_config_sha256": (campaign / "environment/model-config.sha256").read_text(encoding="utf-8").split()[0],
        "runtime_sha256": sha256(campaign / "environment/runtime.txt"),
        "hardware_sha256": sha256(campaign / "environment/npu-smi.txt"),
        "baseline_env_sha256": sha256(campaign / "environment/baseline.env"),
        "workload_manifest_sha256": sha256(campaign / "workload/workload-manifest.json"),
        "formal_dataset_sha256_by_prompt": {str(p): workload_by_prompt[p]["formal"]["sha256"] for p in PROMPTS},
        "configuration": plan["configuration"],
    }
    cells = {}
    for row in aggregate_rows:
        cell_id = f"p{row['prompt_tokens']}_o{row['output_tokens']}"
        metrics = {}
        for metric in RUN_METRICS:
            metrics[metric] = {"mean": row[f"{metric}_mean"], "stdev": row[f"{metric}_stdev"], "cv": row[f"{metric}_cv"]}
        cells[cell_id] = {"prompt_tokens": int(row["prompt_tokens"]), "output_tokens": int(row["output_tokens"]), "gates": {"strict_input": True, "strict_output": True, "zero_failed": True, "graph_valid": True}, "metrics": metrics}
    baseline = {
        "schema_version": 1, "campaign_id": plan["campaign_id"], "data_collection_commit": git_lines[1],
        "version": (campaign / "environment/version.txt").read_text(encoding="utf-8").strip(),
        "fingerprint": fingerprint, "threshold_rule": "max(10%, 3 * baseline CV)",
        "scope": "same-server empirical regression gate, not a cross-hardware SLA", "cells": cells,
    }

    expected_plan = sorted((int(x["order"]), int(x["prompt"]), int(x["output"]), int(x["repeat"]), x["lifecycle_id"]) for x in plan["sequence"])
    observed_plan = sorted((int(x["metadata"]["lifecycle_order_index"]), int(x["metadata"]["target_server_input_tokens"]), int(x["metadata"]["output_tokens"]), int(x["metadata"]["run_order_in_lifecycle"]), x["metadata"]["server_lifecycle_id"]) for x in items.values())
    common_checks = {
        "formal_run_count_36": len(run_rows) == 36,
        "all_run_checks_pass": all(all(run["checks"].values()) for run in validations),
        "one_prompt_sequence_per_prompt": all(len(prompt_sequences[prompt]) == 1 for prompt in PROMPTS),
        "single_server_configuration": len(commands) == 1 and "--no_decode_priority" in next(iter(commands)),
        "campaign_plan_matches_runs": expected_plan == observed_plan,
    }
    if not all(common_checks.values()):
        raise ValueError(f"campaign checks failed: {[key for key,value in common_checks.items() if not value]}")
    validation = {"schema_version": 1, "status": "pass", "campaign_id": plan["campaign_id"], "common_checks": common_checks, "runs": validations}

    sanity = {"status": "not_checked"}
    if task4_reference and task4_reference.is_file():
        reference_rows = list(csv.DictReader(task4_reference.open(encoding="utf-8")))
        reference = next(row for row in reference_rows if row["mode"] == "priority_off" and int(row["concurrency"]) == 4)
        current = aggregate[(128, 256)]
        comparisons = []
        for metric in ("client_ttft_ms", "client_tpot_ms", "client_output_throughput_tok_s"):
            old, new = float(reference[f"{metric}_mean"]), float(current[f"{metric}_mean"])
            comparisons.append({"metric": metric, "task4_mean": stable(old), "task5_mean": stable(new), "relative_difference_pct": stable((new / old - 1) * 100), "difference_gt_10pct": abs(new / old - 1) > 0.10})
        sanity = {"status": "checked", "reference": "task4-reference.csv", "comparisons": comparisons}
    self_compare = compare_baseline(baseline, baseline)
    return ({
        "length-matrix-run-metrics.csv": run_rows,
        "length-matrix-aggregate.csv": aggregate_rows,
        "prompt-scaling.csv": prompt_scaling,
        "output-scaling.csv": output_scaling,
        "length-matrix-heatmap.csv": heatmap_rows,
    }, {
        "length-matrix-heatmap.json": heatmap_json,
        "performance-baseline.json": baseline,
        "baseline-self-compare.json": self_compare,
        "task5-validation.json": validation,
        "task4-sanity.json": sanity,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--task4-reference", type=Path)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    task4_reference = args.task4_reference or (campaign / "task4-reference.csv")
    csv_outputs, json_outputs = build(campaign, task4_reference)
    if args.compare:
        for name, rows in csv_outputs.items():
            rebuilt = campaign / f"{name}.rebuilt"
            write_csv(rebuilt, rows)
            if (campaign / name).read_bytes() != rebuilt.read_bytes():
                rebuilt.unlink(); raise SystemExit(f"{name} differs from rebuilt data")
            rebuilt.unlink()
        for name, value in json_outputs.items():
            expected = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
            if (campaign / name).read_text(encoding="utf-8") != expected:
                raise SystemExit(f"{name} differs from rebuilt data")
    else:
        for name, rows in csv_outputs.items(): write_csv(campaign / name, rows)
        for name, value in json_outputs.items(): (campaign / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "runs": 36, "cells": 12}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
