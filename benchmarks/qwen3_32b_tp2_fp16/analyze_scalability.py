#!/usr/bin/env python3
"""Validate and summarize the fixed-workload Graph-on scalability campaign."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


CONCURRENCIES = (1, 2, 4, 8, 16)
PERCENTILES = (0.5, 0.9, 0.99)
IGNORED_ARGS = {"name", "outputs_dir", "parallel"}


def stable(value: float | int | None):
    return None if value is None else round(float(value), 12)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def single(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"{root}: expected one {pattern}, found {len(matches)}")
    return matches[0]


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires values")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def prom_samples(path: Path, metric: str) -> list[tuple[dict[str, str], float]]:
    result = []
    pattern = re.compile(rf"^{re.escape(metric)}(?:\{{([^}}]*)\}})?\s+([-+0-9.eE]+)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1) or ""))
        result.append((labels, float(match.group(2))))
    return result


def sample_value(path: Path, metric: str, required: dict[str, str]) -> float:
    matches = [
        value
        for labels, value in prom_samples(path, metric)
        if all(labels.get(key) == expected for key, expected in required.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one {metric}{required}, found {len(matches)}")
    return matches[0]


def counter_delta(before: Path, after: Path, metric: str, labels: dict[str, str]) -> float:
    return sample_value(after, metric, labels) - sample_value(before, metric, labels)


def histogram_delta(before: Path, after: Path, metric: str) -> dict:
    required = {"endpoint": "chat"}
    before_buckets = {
        labels["le"]: value
        for labels, value in prom_samples(before, metric + "_bucket")
        if labels.get("endpoint") == "chat"
    }
    after_buckets = {
        labels["le"]: value
        for labels, value in prom_samples(after, metric + "_bucket")
        if labels.get("endpoint") == "chat"
    }
    buckets = {
        boundary: after_buckets[boundary] - before_buckets.get(boundary, 0.0)
        for boundary in after_buckets
    }
    count = counter_delta(before, after, metric + "_count", required)
    total_sum = counter_delta(before, after, metric + "_sum", required)
    return {"buckets": buckets, "count": count, "sum": total_sum}


def merge_histograms(histograms: list[dict]) -> dict:
    boundaries = set().union(*(histogram["buckets"] for histogram in histograms))
    return {
        "buckets": {
            boundary: sum(histogram["buckets"].get(boundary, 0.0) for histogram in histograms)
            for boundary in boundaries
        },
        "count": sum(histogram["count"] for histogram in histograms),
        "sum": sum(histogram["sum"] for histogram in histograms),
    }


def histogram_quantile(histogram: dict, q: float) -> float:
    count = float(histogram["count"])
    if count <= 0:
        raise ValueError("histogram has no observations")
    parsed = sorted(
        (
            math.inf if boundary == "+Inf" else float(boundary),
            float(cumulative),
        )
        for boundary, cumulative in histogram["buckets"].items()
    )
    target = q * count
    lower_bound = 0.0
    lower_count = 0.0
    for upper_bound, cumulative in parsed:
        if cumulative >= target:
            if math.isinf(upper_bound):
                return lower_bound
            in_bucket = cumulative - lower_count
            if in_bucket <= 0:
                return upper_bound
            fraction = (target - lower_count) / in_bucket
            return lower_bound + fraction * (upper_bound - lower_bound)
        lower_bound = upper_bound
        lower_count = cumulative
    return lower_bound


def graph_stats(path: Path) -> dict[str, int]:
    graph = read_json(path).get("npu_graph", {})
    return {
        key: int(graph.get(key, 0))
        for key in ("capture_attempts", "captures", "replays", "fallbacks")
    }


def canonical_requests(fingerprint: dict) -> list[dict]:
    requests = []
    for request in fingerprint["requests"]:
        requests.append({key: value for key, value in request.items() if key != "index"})
    return sorted(
        requests,
        key=lambda value: json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ),
    )


def normalized_server_command(command: str) -> str:
    return " ".join(command.split())


def normalized_evalscope_args(path: Path) -> dict:
    return {
        key: value
        for key, value in read_json(path).items()
        if key not in IGNORED_ARGS
    }


def summarize_timeseries(path: Path, timing: dict, interval: float) -> tuple[dict, list[str]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    errors = []
    monotonic = [float(record["monotonic_s"]) for record in records]
    if len(records) < 2:
        errors.append("fewer_than_two_samples")
    if any(right <= left for left, right in zip(monotonic, monotonic[1:])):
        errors.append("timestamps_not_strictly_increasing")
    server_records = [record for record in records if record["server"].get("status") == "ok"]
    if len(server_records) != len(records):
        errors.append("server_scrape_error")
    duration = float(timing["formal_duration_s"])
    expected_min = max(2, int(duration / interval * 0.5))
    if len(records) < expected_min:
        errors.append("insufficient_sample_coverage")
    gaps = [right - left for left, right in zip(monotonic, monotonic[1:])]
    schedulers = [record["server"].get("scheduler", {}) for record in server_records]
    waiting = [float(value.get("waiting", 0)) for value in schedulers]
    prefilling = [float(value.get("prefilling", 0)) for value in schedulers]
    running = [float(value.get("running", 0)) for value in schedulers]
    system = [a + b + c for a, b, c in zip(waiting, prefilling, running)]
    kv_used = [
        float(record["server"].get("kv_cache", {}).get("used_pages", 0))
        for record in server_records
    ]
    graph_replays = [
        float(record["server"].get("npu_graph", {}).get("replays", 0))
        for record in server_records
    ]
    npu_util = []
    npu_hbm = []
    for record in records:
        for device in (record.get("npu") or {}).values():
            if device.get("status") != "ok":
                continue
            if device.get("npu_utilization_pct") is not None:
                npu_util.append(float(device["npu_utilization_pct"]))
            if device.get("hbm_usage_pct") is not None:
                npu_hbm.append(float(device["hbm_usage_pct"]))
    def mean(values):
        return stable(statistics.mean(values)) if values else None
    def peak(values):
        return stable(max(values)) if values else None
    return ({
        "timeseries_samples": len(records),
        "timeseries_server_ok_samples": len(server_records),
        "timeseries_duration_s": stable(monotonic[-1] - monotonic[0]) if len(monotonic) > 1 else 0.0,
        "timeseries_max_gap_s": peak(gaps),
        "waiting_mean": mean(waiting),
        "waiting_peak": peak(waiting),
        "prefilling_mean": mean(prefilling),
        "prefilling_peak": peak(prefilling),
        "running_mean": mean(running),
        "running_peak": peak(running),
        "system_requests_mean": mean(system),
        "system_requests_peak": peak(system),
        "kv_used_pages_mean": mean(kv_used),
        "kv_used_pages_peak": peak(kv_used),
        "timeseries_graph_replay_delta": stable(max(graph_replays) - min(graph_replays)) if graph_replays else None,
        "npu_observations": len(npu_util),
        "npu_utilization_mean_pct": mean(npu_util),
        "npu_utilization_peak_pct": peak(npu_util),
        "npu_hbm_usage_mean_pct": mean(npu_hbm),
        "npu_hbm_usage_peak_pct": peak(npu_hbm),
    }, errors)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(campaign: Path) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    campaign_plan = read_json(campaign / "campaign-plan.json")
    metadata_paths = sorted(campaign.glob("on/p*/run-*/run-metadata.json"))
    if len(metadata_paths) != 15:
        raise ValueError(f"expected 15 runs, found {len(metadata_paths)}")
    run_rows = []
    timeseries_rows = []
    validations = []
    grouped = defaultdict(list)
    formal_dataset_sha = set()
    canonical_workloads = []
    normalized_args = []
    server_commands = []
    for metadata_path in metadata_paths:
        root = metadata_path.parent
        metadata = read_json(metadata_path)
        concurrency = int(metadata["concurrency"])
        formal_dataset_sha.add(metadata["formal_dataset_sha256"])
        summary = read_json(single(root / "client/evalscope", "**/benchmark_summary.json"))
        request_metrics = read_json(root / "client/request-metrics.json")
        fingerprint = read_json(root / "client/workload-fingerprint.json")
        canonical_workloads.append(canonical_requests(fingerprint))
        args_path = single(root / "client/evalscope", "**/benchmark_args.json")
        normalized_args.append(normalized_evalscope_args(args_path))
        before_metrics = root / "server/before-metrics.prom"
        after_metrics = root / "server/after-metrics.prom"
        queue_hist = histogram_delta(before_metrics, after_metrics, "lite_llama_queue_wait_seconds")
        server_ttft_hist = histogram_delta(before_metrics, after_metrics, "lite_llama_time_to_first_token_seconds")
        before_graph = graph_stats(root / "server/before-stats.json")
        after_graph = graph_stats(root / "server/after-stats.json")
        timing = read_json(root / "run-timing.json")
        timeseries, timeseries_errors = summarize_timeseries(
            root / "server/timeseries.jsonl",
            timing,
            float(metadata["timeseries_interval_s"]),
        )
        lifecycle = metadata["server_lifecycle_id"]
        command_path = campaign / "on/lifecycles" / lifecycle / "server/start-command.txt"
        server_commands.append(normalized_server_command(command_path.read_text(encoding="utf-8")))
        client_requests = request_metrics["requests"]
        request_values = {
            "client_e2e_latency_s": [float(request["latency_s"]) for request in client_requests],
            "client_ttft_ms": [float(request["ttft_ms"]) for request in client_requests],
            "client_tpot_ms": [float(request["tpot_ms"]) for request in client_requests],
            "client_itl_ms": [float(request["itl_mean_ms"]) for request in client_requests],
        }
        server_queue_mean_ms = queue_hist["sum"] / queue_hist["count"] * 1000
        server_ttft_mean_ms = server_ttft_hist["sum"] / server_ttft_hist["count"] * 1000
        row = {
            "run_id": metadata["run_id"],
            "concurrency": concurrency,
            "repeat": int(metadata["run_order_in_lifecycle"]),
            "lifecycle_order_index": int(metadata["lifecycle_order_index"]),
            "requests": int(metadata["requests"]),
            "warmup_requests": int(metadata["warmup_requests"]),
            "client_e2e_latency_s": stable(summary["Avg Latency (s)"]),
            "client_ttft_ms": stable(summary["TTFT (ms)"]),
            "client_tpot_ms": stable(summary["TPOT (ms)"]),
            "client_itl_ms": stable(summary["ITL (ms)"]),
            "client_output_throughput_tok_s": stable(summary["Output Throughput (tok/s)"]),
            "client_total_throughput_tok_s": stable(summary["Total Throughput (tok/s)"]),
            "client_qps": stable(summary["Req Throughput (req/s)"]),
            "server_queue_wait_mean_ms": stable(server_queue_mean_ms),
            "server_ttft_mean_ms": stable(server_ttft_mean_ms),
            "estimated_nonqueue_first_token_ms": stable(max(0.0, server_ttft_mean_ms - server_queue_mean_ms)),
            "queue_share_of_server_ttft_pct": stable(server_queue_mean_ms / server_ttft_mean_ms * 100),
            "little_law_qps_x_e2e": stable(float(summary["Req Throughput (req/s)"]) * float(summary["Avg Latency (s)"])),
            "graph_capture_delta": after_graph["captures"] - before_graph["captures"],
            "graph_replay_delta": after_graph["replays"] - before_graph["replays"],
            "graph_fallback_delta": after_graph["fallbacks"] - before_graph["fallbacks"],
        }
        for metric, values in request_values.items():
            for q in PERCENTILES:
                row[f"{metric}_p{int(q * 100)}"] = stable(quantile(values, q))
        for metric, histogram in (
            ("server_queue_wait_ms", queue_hist),
            ("server_ttft_ms", server_ttft_hist),
        ):
            for q in PERCENTILES:
                row[f"{metric}_p{int(q * 100)}_bucket_estimate"] = stable(
                    histogram_quantile(histogram, q) * 1000
                )
        run_rows.append(row)
        timeseries_rows.append({"run_id": metadata["run_id"], "concurrency": concurrency, **timeseries})
        checks = {
            "graph_captured_before_formal": before_graph["captures"] > 0,
            "formal_capture_delta_zero": row["graph_capture_delta"] == 0,
            "formal_fallback_zero": row["graph_fallback_delta"] == 0,
            "formal_replay_grew": row["graph_replay_delta"] > 0,
            "request_count_32": int(summary["Total Requests"]) == 32 == len(client_requests),
            "strict_input_128": all(int(request["prompt_tokens"]) == 128 for request in client_requests),
            "strict_output_256": all(int(request["completion_tokens"]) == 256 for request in client_requests),
            "zero_failed": int(summary["Failed Requests"]) == 0 and all(int(request["success"]) == 1 for request in client_requests),
            "timeseries_complete": not timeseries_errors,
        }
        validations.append({
            "run_id": metadata["run_id"],
            "checks": checks,
            "timeseries_errors": timeseries_errors,
            "before_graph": before_graph,
            "after_graph": after_graph,
        })
        grouped[concurrency].append({
            "row": row,
            "request_values": request_values,
            "queue_hist": queue_hist,
            "server_ttft_hist": server_ttft_hist,
            "timeseries": timeseries,
        })
    if set(grouped) != set(CONCURRENCIES) or any(len(grouped[c]) != 3 for c in CONCURRENCIES):
        raise ValueError("expected three runs for each concurrency 1,2,4,8,16")
    workload_equal = all(value == canonical_workloads[0] for value in canonical_workloads[1:])
    args_equal_except_concurrency = all(value == normalized_args[0] for value in normalized_args[1:])
    commands_equal = all(value == server_commands[0] for value in server_commands[1:])
    aggregate_rows = []
    run_metrics = (
        "client_e2e_latency_s",
        "client_ttft_ms",
        "client_tpot_ms",
        "client_itl_ms",
        "client_output_throughput_tok_s",
        "client_total_throughput_tok_s",
        "client_qps",
        "server_queue_wait_mean_ms",
        "server_ttft_mean_ms",
        "estimated_nonqueue_first_token_ms",
        "queue_share_of_server_ttft_pct",
        "little_law_qps_x_e2e",
    )
    for concurrency in CONCURRENCIES:
        group = grouped[concurrency]
        aggregate = {"concurrency": concurrency, "formal_runs": len(group)}
        for metric in run_metrics:
            values = [float(item["row"][metric]) for item in group]
            aggregate[f"{metric}_mean"] = stable(statistics.mean(values))
            aggregate[f"{metric}_stdev"] = stable(statistics.stdev(values))
            for q in PERCENTILES:
                aggregate[f"{metric}_p{int(q * 100)}_across_runs"] = stable(quantile(values, q))
        for metric in ("client_e2e_latency_s", "client_ttft_ms", "client_tpot_ms", "client_itl_ms"):
            values = [value for item in group for value in item["request_values"][metric]]
            for q in PERCENTILES:
                aggregate[f"{metric}_p{int(q * 100)}_requests"] = stable(quantile(values, q))
        for metric, key in (
            ("server_queue_wait_ms", "queue_hist"),
            ("server_ttft_ms", "server_ttft_hist"),
        ):
            histogram = merge_histograms([item[key] for item in group])
            for q in PERCENTILES:
                aggregate[f"{metric}_p{int(q * 100)}_bucket_estimate"] = stable(
                    histogram_quantile(histogram, q) * 1000
                )
        for metric in (
            "waiting_mean", "waiting_peak", "prefilling_mean", "prefilling_peak",
            "running_mean", "running_peak", "system_requests_mean", "system_requests_peak",
            "kv_used_pages_mean", "kv_used_pages_peak", "npu_utilization_mean_pct",
            "npu_utilization_peak_pct", "npu_hbm_usage_mean_pct", "npu_hbm_usage_peak_pct",
        ):
            values = [float(item["timeseries"][metric]) for item in group if item["timeseries"][metric] is not None]
            aggregate[f"timeseries_{metric}_mean"] = stable(statistics.mean(values)) if values else None
            aggregate[f"timeseries_{metric}_peak"] = stable(max(values)) if values else None
        aggregate_rows.append(aggregate)
    saturation_pairs = []
    first_saturation = None
    for prior, current in zip(aggregate_rows, aggregate_rows[1:]):
        throughput_gain = (
            current["client_output_throughput_tok_s_mean"]
            / prior["client_output_throughput_tok_s_mean"] - 1
        ) * 100
        ttft_growth = (current["client_ttft_ms_mean"] / prior["client_ttft_ms_mean"] - 1) * 100
        queue_prior = prior["server_queue_wait_mean_ms_mean"]
        queue_current = current["server_queue_wait_mean_ms_mean"]
        queue_growth = (
            (queue_current / queue_prior - 1) * 100 if queue_prior > 0 else math.inf
        )
        saturated = throughput_gain < 15 and (ttft_growth > 50 or queue_growth > 50)
        record = {
            "from_concurrency": prior["concurrency"],
            "to_concurrency": current["concurrency"],
            "output_throughput_gain_pct": stable(throughput_gain),
            "mean_ttft_growth_pct": stable(ttft_growth),
            "mean_queue_wait_growth_pct": stable(queue_growth),
            "criterion_met": saturated,
        }
        saturation_pairs.append(record)
        if saturated and first_saturation is None:
            first_saturation = current["concurrency"]
    saturation = {
        "criterion": "concurrency doubles; output throughput gain <15%; and mean TTFT or mean queue wait growth >50%",
        "first_saturation_concurrency": first_saturation,
        "pairs": saturation_pairs,
    }
    common_checks = {
        "formal_run_count_15": len(run_rows) == 15,
        "one_formal_dataset_sha": len(formal_dataset_sha) == 1,
        "observed_prompt_multiset_equal": workload_equal,
        "evalscope_args_equal_except_concurrency": args_equal_except_concurrency,
        "server_commands_identical": commands_equal,
        "campaign_plan_matches_runs": sorted(
            (
                int(item["order"]),
                int(item["concurrency"]),
                int(item["repeat"]),
                item["lifecycle_id"],
            )
            for item in campaign_plan["sequence"]
        ) == sorted(
            (
                int(read_json(path)["lifecycle_order_index"]),
                int(read_json(path)["concurrency"]),
                int(read_json(path)["run_order_in_lifecycle"]),
                read_json(path)["server_lifecycle_id"],
            )
            for path in metadata_paths
        ),
        "all_run_checks_pass": all(all(item["checks"].values()) for item in validations),
    }
    if not all(common_checks.values()):
        raise ValueError(f"campaign checks failed: {[key for key, value in common_checks.items() if not value]}")
    validation = {
        "schema_version": 1,
        "status": "pass",
        "campaign_id": campaign.name,
        "common_checks": common_checks,
        "formal_dataset_sha256": next(iter(formal_dataset_sha)),
        "runs": validations,
    }
    return ({
        "scalability-run-metrics.csv": run_rows,
        "scalability-aggregate.csv": aggregate_rows,
        "timeseries-summary.csv": timeseries_rows,
    }, {
        "saturation-analysis.json": saturation,
        "task3-validation.json": validation,
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
            temporary = campaign / (name + ".rebuilt")
            write_csv(temporary, rows)
            if (campaign / name).read_bytes() != temporary.read_bytes():
                temporary.unlink()
                raise SystemExit(f"{name} differs from rebuilt data")
            temporary.unlink()
        for name, value in json_outputs.items():
            serialized = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
            if (campaign / name).read_text(encoding="utf-8") != serialized:
                raise SystemExit(f"{name} differs from rebuilt data")
    else:
        for name, rows in csv_outputs.items():
            write_csv(campaign / name, rows)
        for name, value in json_outputs.items():
            (campaign / name).write_text(
                json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    print(json.dumps({"status": "pass", "runs": 15, "concurrencies": list(CONCURRENCIES)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
