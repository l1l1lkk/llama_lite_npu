#!/usr/bin/env python3
"""Rebuild the strict Decode Priority on/off paired benchmark analysis."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import analyze_scalability as core


MODES = ("priority_on", "priority_off")
CONCURRENCIES = (1, 2, 4, 8, 16)
PERCENTILES = (0.5, 0.9, 0.99)
RUN_METRICS = (
    "client_e2e_latency_s", "client_ttft_ms", "client_tpot_ms", "client_itl_ms",
    "client_output_throughput_tok_s", "client_total_throughput_tok_s", "client_qps",
    "server_queue_wait_mean_ms", "server_ttft_mean_ms",
    "estimated_nonqueue_first_token_ms", "queue_share_of_server_ttft_pct",
)
TIMESERIES_METRICS = (
    "waiting_mean", "waiting_peak", "prefilling_mean", "prefilling_peak",
    "running_mean", "running_peak", "system_requests_mean", "system_requests_peak",
    "kv_used_pages_mean", "kv_used_pages_peak", "npu_utilization_mean_pct",
    "npu_utilization_peak_pct", "npu_hbm_usage_mean_pct", "npu_hbm_usage_peak_pct",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def single(root: Path, pattern: str) -> Path:
    return core.single(root, pattern)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalized_pair_args(path: Path) -> dict:
    return {
        key: value for key, value in read_json(path).items()
        if key not in {"name", "outputs_dir"}
    }


def normalized_server_command(command: str) -> str:
    command = re.sub(r"LITE_LLAMA_REQUEST_TIMING_TRACE=\S+", "TRACE=<path>", command)
    command = re.sub(r"--(?:no_)?decode_priority", "<decode_priority_flag>", command)
    return " ".join(command.split())


def mean(values):
    return core.stable(statistics.mean(values))


def stdev(values):
    return core.stable(statistics.stdev(values))


def pct(values, q):
    return core.stable(core.quantile(values, q))


def build(campaign: Path):
    plan = read_json(campaign / "campaign-plan.json")
    metadata_paths = sorted(campaign.glob("priority_*/p*/run-*/run-metadata.json"))
    if len(metadata_paths) != 30:
        raise ValueError(f"expected 30 formal runs, found {len(metadata_paths)}")

    run_rows = []
    fairness_run_rows = []
    validation_runs = []
    items = {}
    formal_shas = set()

    for metadata_path in metadata_paths:
        root = metadata_path.parent
        metadata = read_json(metadata_path)
        mode = metadata["benchmark_variant"]
        concurrency = int(metadata["concurrency"])
        repeat = int(metadata["run_order_in_lifecycle"])
        if mode not in MODES:
            raise ValueError(f"unexpected mode: {mode}")
        key = (concurrency, repeat, mode)
        if key in items:
            raise ValueError(f"duplicate run: {key}")

        summary = read_json(single(root / "client/evalscope", "**/benchmark_summary.json"))
        args_path = single(root / "client/evalscope", "**/benchmark_args.json")
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
        timeseries, timeseries_errors = core.summarize_timeseries(
            root / "server/timeseries.jsonl", timing, float(metadata["timeseries_interval_s"])
        )
        lifecycle = metadata["server_lifecycle_id"]
        command_path = campaign / mode / "lifecycles" / lifecycle / "server/start-command.txt"
        command = command_path.read_text(encoding="utf-8")

        client = request_metrics["requests"]
        server = server_trace["requests"]
        client_by_submit = sorted(client, key=lambda item: item["start_time_s"])
        server_by_submit = sorted(server, key=lambda item: item["formal_submission_order"])
        server_queue = [float(item["queue_wait_ms"]) for item in server_by_submit]
        server_ttft = [float(item["ttft_ms"]) for item in server_by_submit]
        server_service = [float(item["service_to_first_token_ms"]) for item in server_by_submit]
        client_values = {
            "client_e2e_latency_s": [float(item["latency_s"]) for item in client_by_submit],
            "client_ttft_ms": [float(item["ttft_ms"]) for item in client_by_submit],
            "client_tpot_ms": [float(item["tpot_ms"]) for item in client_by_submit],
            "client_itl_ms": [float(item["itl_mean_ms"]) for item in client_by_submit],
        }
        queue_mean = queue_hist["sum"] / queue_hist["count"] * 1000
        ttft_mean = ttft_hist["sum"] / ttft_hist["count"] * 1000
        row = {
            "run_id": metadata["run_id"], "mode": mode, "concurrency": concurrency,
            "repeat": repeat, "lifecycle_order_index": int(metadata["lifecycle_order_index"]),
            "requests": int(metadata["requests"]), "warmup_requests": int(metadata["warmup_requests"]),
            "client_e2e_latency_s": core.stable(summary["Avg Latency (s)"]),
            "client_ttft_ms": core.stable(summary["TTFT (ms)"]),
            "client_tpot_ms": core.stable(summary["TPOT (ms)"]),
            "client_itl_ms": core.stable(summary["ITL (ms)"]),
            "client_output_throughput_tok_s": core.stable(summary["Output Throughput (tok/s)"]),
            "client_total_throughput_tok_s": core.stable(summary["Total Throughput (tok/s)"]),
            "client_qps": core.stable(summary["Req Throughput (req/s)"]),
            "server_queue_wait_mean_ms": core.stable(queue_mean),
            "server_ttft_mean_ms": core.stable(ttft_mean),
            "estimated_nonqueue_first_token_ms": core.stable(max(0.0, ttft_mean - queue_mean)),
            "queue_share_of_server_ttft_pct": core.stable(queue_mean / ttft_mean * 100),
            "graph_capture_delta": after_graph["captures"] - before_graph["captures"],
            "graph_replay_delta": after_graph["replays"] - before_graph["replays"],
            "graph_fallback_delta": after_graph["fallbacks"] - before_graph["fallbacks"],
        }
        for metric, values in client_values.items():
            for q in PERCENTILES:
                row[f"{metric}_p{int(q * 100)}"] = pct(values, q)
        for metric, values in (
            ("server_queue_wait_ms", server_queue),
            ("server_ttft_ms", server_ttft),
            ("server_service_to_first_token_ms", server_service),
        ):
            for q in PERCENTILES:
                row[f"{metric}_p{int(q * 100)}"] = pct(values, q)
            row[f"{metric}_max"] = core.stable(max(values))
        run_rows.append(row)

        half = len(server_by_submit) // 2
        first_queue, second_queue = server_queue[:half], server_queue[half:]
        first_ttft, second_ttft = server_ttft[:half], server_ttft[half:]
        fairness = {
            "run_id": metadata["run_id"], "mode": mode, "concurrency": concurrency,
            "repeat": repeat,
            "queue_wait_p99_ms": pct(server_queue, 0.99),
            "queue_wait_max_ms": core.stable(max(server_queue)),
            "ttft_p99_ms": pct(server_ttft, 0.99),
            "ttft_max_ms": core.stable(max(server_ttft)),
            "first_half_queue_mean_ms": mean(first_queue),
            "second_half_queue_mean_ms": mean(second_queue),
            "second_minus_first_queue_ms": core.stable(mean(second_queue) - mean(first_queue)),
            "first_half_ttft_mean_ms": mean(first_ttft),
            "second_half_ttft_mean_ms": mean(second_ttft),
            "second_minus_first_ttft_ms": core.stable(mean(second_ttft) - mean(first_ttft)),
            "requests_queue_wait_ge_1s": sum(value >= 1000 for value in server_queue),
        }
        fairness_run_rows.append(fairness)

        checks = {
            "request_count_32": len(client) == len(server) == int(summary["Total Requests"]) == 32,
            "strict_input_128": all(int(item["prompt_tokens"]) == 128 for item in client),
            "strict_output_256": all(int(item["completion_tokens"]) == 256 for item in client),
            "zero_failed": int(summary["Failed Requests"]) == 0 and all(int(item["success"]) == 1 for item in client),
            "server_trace_complete": all(item.get("queue_wait_ms") is not None and item.get("ttft_ms") is not None for item in server),
            "graph_captured_before_formal": before_graph["captures"] > 0,
            "formal_capture_delta_zero": row["graph_capture_delta"] == 0,
            "formal_fallback_zero": row["graph_fallback_delta"] == 0,
            "formal_replay_grew": row["graph_replay_delta"] > 0,
            "timeseries_complete": not timeseries_errors,
        }
        validation_runs.append({"run_id": metadata["run_id"], "checks": checks, "timeseries_errors": timeseries_errors})
        formal_shas.add(metadata["formal_dataset_sha256"])
        items[key] = {
            "metadata": metadata, "row": row, "fairness": fairness,
            "client_values": client_values, "server_queue": server_queue,
            "server_ttft": server_ttft, "server_service": server_service,
            "timeseries": timeseries, "fingerprint": core.canonical_requests(fingerprint),
            "args": normalized_pair_args(args_path), "command": command,
        }

    aggregate_rows = []
    fairness_aggregate_rows = []
    grouped = defaultdict(list)
    for (concurrency, repeat, mode), item in items.items():
        grouped[(mode, concurrency)].append(item)
    for mode in MODES:
        for concurrency in CONCURRENCIES:
            group = grouped[(mode, concurrency)]
            if len(group) != 3:
                raise ValueError(f"expected three runs for {mode} c{concurrency}")
            aggregate = {"mode": mode, "concurrency": concurrency, "formal_runs": 3}
            for metric in RUN_METRICS:
                values = [float(item["row"][metric]) for item in group]
                aggregate[f"{metric}_mean"] = mean(values)
                aggregate[f"{metric}_stdev"] = stdev(values)
            for metric in ("client_e2e_latency_s", "client_ttft_ms", "client_tpot_ms", "client_itl_ms"):
                values = [value for item in group for value in item["client_values"][metric]]
                for q in PERCENTILES:
                    aggregate[f"{metric}_p{int(q * 100)}_requests"] = pct(values, q)
            for metric, key in (
                ("server_queue_wait_ms", "server_queue"),
                ("server_ttft_ms", "server_ttft"),
                ("server_service_to_first_token_ms", "server_service"),
            ):
                values = [value for item in group for value in item[key]]
                for q in PERCENTILES:
                    aggregate[f"{metric}_p{int(q * 100)}_requests"] = pct(values, q)
                aggregate[f"{metric}_max_requests"] = core.stable(max(values))
            for metric in TIMESERIES_METRICS:
                values = [float(item["timeseries"][metric]) for item in group if item["timeseries"][metric] is not None]
                aggregate[f"timeseries_{metric}_mean"] = mean(values) if values else None
                aggregate[f"timeseries_{metric}_peak"] = core.stable(max(values)) if values else None
            aggregate["graph_capture_delta_sum"] = sum(item["row"]["graph_capture_delta"] for item in group)
            aggregate["graph_replay_delta_sum"] = sum(item["row"]["graph_replay_delta"] for item in group)
            aggregate["graph_fallback_delta_sum"] = sum(item["row"]["graph_fallback_delta"] for item in group)
            aggregate_rows.append(aggregate)

            fair_group = [item["fairness"] for item in group]
            fairness_aggregate_rows.append({
                "mode": mode, "concurrency": concurrency,
                "queue_wait_p99_ms_mean": mean([float(item["queue_wait_p99_ms"]) for item in fair_group]),
                "queue_wait_max_ms_max": core.stable(max(float(item["queue_wait_max_ms"]) for item in fair_group)),
                "ttft_p99_ms_mean": mean([float(item["ttft_p99_ms"]) for item in fair_group]),
                "ttft_max_ms_max": core.stable(max(float(item["ttft_max_ms"]) for item in fair_group)),
                "second_minus_first_queue_ms_mean": mean([float(item["second_minus_first_queue_ms"]) for item in fair_group]),
                "second_minus_first_ttft_ms_mean": mean([float(item["second_minus_first_ttft_ms"]) for item in fair_group]),
                "requests_queue_wait_ge_1s_total": sum(int(item["requests_queue_wait_ge_1s"]) for item in fair_group),
            })

    paired_rows = []
    pair_checks = []
    latency_metrics = (
        "client_e2e_latency_s", "client_ttft_ms", "client_tpot_ms", "client_itl_ms",
        "server_queue_wait_mean_ms", "server_ttft_mean_ms", "estimated_nonqueue_first_token_ms",
    )
    throughput_metrics = (
        "client_output_throughput_tok_s", "client_total_throughput_tok_s", "client_qps",
    )
    compare_keys = (
        "graph", "target_server_input_tokens", "evalscope_prompt_tokens", "min_tokens",
        "output_tokens", "concurrency", "requests", "warmup_requests", "dataset_kind",
        "formal_dataset_sha256", "warmup_dataset_sha256", "seed", "dataset_offset",
        "warmup_dataset_offset", "temperature", "top_p", "sampling",
    )
    for concurrency in CONCURRENCIES:
        for repeat in (1, 2, 3):
            on = items[(concurrency, repeat, "priority_on")]
            off = items[(concurrency, repeat, "priority_off")]
            row = {"pair_id": f"c{concurrency}_r{repeat}", "concurrency": concurrency, "repeat": repeat}
            for metric in latency_metrics:
                on_value, off_value = float(on["row"][metric]), float(off["row"][metric])
                row[f"{metric}_on"] = core.stable(on_value)
                row[f"{metric}_off"] = core.stable(off_value)
                row[f"{metric}_on_over_off"] = core.stable(on_value / off_value) if off_value else None
                row[f"{metric}_off_reduction_pct"] = core.stable((1 - off_value / on_value) * 100) if on_value else None
            for metric in throughput_metrics:
                on_value, off_value = float(on["row"][metric]), float(off["row"][metric])
                row[f"{metric}_on"] = core.stable(on_value)
                row[f"{metric}_off"] = core.stable(off_value)
                row[f"{metric}_off_over_on"] = core.stable(off_value / on_value)
                row[f"{metric}_off_gain_pct"] = core.stable((off_value / on_value - 1) * 100)
            paired_rows.append(row)
            checks = {
                "metadata_equal_except_mode": all(on["metadata"].get(key) == off["metadata"].get(key) for key in compare_keys),
                "evalscope_args_equal": on["args"] == off["args"],
                "prompt_multiset_equal": on["fingerprint"] == off["fingerprint"],
                "server_command_only_flag_differs": normalized_server_command(on["command"]) == normalized_server_command(off["command"]),
            }
            pair_checks.append({"pair_id": row["pair_id"], "checks": checks})

    comparison_rows = []
    causal = []
    aggregate_by_key = {(row["mode"], int(row["concurrency"])): row for row in aggregate_rows}
    for concurrency in CONCURRENCIES:
        on = aggregate_by_key[("priority_on", concurrency)]
        off = aggregate_by_key[("priority_off", concurrency)]
        row = {"concurrency": concurrency}
        for metric in latency_metrics:
            on_value, off_value = float(on[f"{metric}_mean"]), float(off[f"{metric}_mean"])
            row[f"{metric}_on_mean"] = core.stable(on_value)
            row[f"{metric}_off_mean"] = core.stable(off_value)
            row[f"{metric}_ratio_of_means_on_over_off"] = core.stable(on_value / off_value) if off_value else None
            row[f"{metric}_off_reduction_pct"] = core.stable((1 - off_value / on_value) * 100) if on_value else None
        for metric in throughput_metrics:
            on_value, off_value = float(on[f"{metric}_mean"]), float(off[f"{metric}_mean"])
            row[f"{metric}_on_mean"] = core.stable(on_value)
            row[f"{metric}_off_mean"] = core.stable(off_value)
            row[f"{metric}_ratio_of_means_off_over_on"] = core.stable(off_value / on_value)
            row[f"{metric}_off_gain_pct"] = core.stable((off_value / on_value - 1) * 100)
        comparison_rows.append(row)
        queue_reduction = float(row["server_queue_wait_mean_ms_off_reduction_pct"])
        tpot_change = (float(off["client_tpot_ms_mean"]) / float(on["client_tpot_ms_mean"]) - 1) * 100
        itl_change = (float(off["client_itl_ms_mean"]) / float(on["client_itl_ms_mean"]) - 1) * 100
        throughput_gain = float(row["client_output_throughput_tok_s_off_gain_pct"])
        causal.append({
            "concurrency": concurrency,
            "queue_wait_reduction_pct": core.stable(queue_reduction),
            "decode_priority_is_major_causal_factor": queue_reduction >= 80,
            "tpot_change_off_vs_on_pct": core.stable(tpot_change),
            "tpot_interference_cost_ge_10pct": tpot_change >= 10,
            "itl_change_off_vs_on_pct": core.stable(itl_change),
            "itl_interference_cost_ge_10pct": itl_change >= 10,
            "output_throughput_gain_pct": core.stable(throughput_gain),
            "output_throughput_gain_ge_15pct": throughput_gain >= 15,
        })

    plan_observed = sorted(
        (int(item["metadata"]["lifecycle_order_index"]), item["metadata"]["benchmark_variant"], int(item["metadata"]["concurrency"]), int(item["metadata"]["run_order_in_lifecycle"]), item["metadata"]["server_lifecycle_id"])
        for item in items.values()
    )
    plan_expected = sorted(
        (int(item["order"]), item["mode"], int(item["concurrency"]), int(item["repeat"]), item["lifecycle_id"])
        for item in plan["sequence"]
    )
    common_checks = {
        "formal_run_count_30": len(run_rows) == 30,
        "one_frozen_dataset_sha": formal_shas == {"11b2c0d603619092f7d2a35eae53cafc5cb32d4e7d642a429e29a41758a428ba"},
        "all_run_checks_pass": all(all(item["checks"].values()) for item in validation_runs),
        "all_15_pairs_match": len(pair_checks) == 15 and all(all(item["checks"].values()) for item in pair_checks),
        "campaign_plan_matches_runs": plan_observed == plan_expected,
    }
    if not all(common_checks.values()):
        raise ValueError(f"campaign checks failed: {[key for key, value in common_checks.items() if not value]}")
    validation = {
        "schema_version": 1, "status": "pass", "campaign_id": campaign.name,
        "common_checks": common_checks, "runs": validation_runs, "pairs": pair_checks,
    }
    causal_report = {
        "schema_version": 1,
        "preregistered_thresholds": {
            "major_causal_queue_reduction_pct": 80,
            "decode_interference_cost_pct": 10,
            "output_throughput_gain_pct": 15,
        },
        "results": causal,
    }
    return ({
        "decode-priority-run-metrics.csv": run_rows,
        "decode-priority-aggregate.csv": aggregate_rows,
        "decode-priority-paired-ratios.csv": paired_rows,
        "decode-priority-comparison.csv": comparison_rows,
        "decode-priority-fairness.csv": fairness_run_rows + fairness_aggregate_rows,
    }, {
        "decode-priority-causal.json": causal_report,
        "task4-validation.json": validation,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    csv_outputs, json_outputs = build(campaign)
    if args.compare:
        for name, rows in csv_outputs.items():
            rebuilt = campaign / f"{name}.rebuilt"
            write_csv(rebuilt, rows)
            if (campaign / name).read_bytes() != rebuilt.read_bytes():
                rebuilt.unlink()
                raise SystemExit(f"{name} differs from rebuilt data")
            rebuilt.unlink()
        for name, value in json_outputs.items():
            expected = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
            if (campaign / name).read_text(encoding="utf-8") != expected:
                raise SystemExit(f"{name} differs from rebuilt data")
    else:
        for name, rows in csv_outputs.items():
            write_csv(campaign / name, rows)
        for name, value in json_outputs.items():
            (campaign / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "runs": 30, "pairs": 15}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
