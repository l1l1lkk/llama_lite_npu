#!/usr/bin/env python3
"""Build a traceable CSV from EvalScope summaries and server snapshots."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def metric_value(path: Path, name: str, labels: str = "") -> float | None:
    if not path.exists():
        return None
    prefix = name + labels
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix + " "):
            return float(line.rsplit(" ", 1)[1])
    return None


def delta(before: Path, after: Path, name: str, labels: str = "") -> float | None:
    a = metric_value(before, name, labels)
    b = metric_value(after, name, labels)
    if b is None:
        return None
    return b - (a or 0.0)


def nested_delta(before: Path, after: Path, section: str, key: str) -> float | None:
    if not after.exists():
        return None
    after_data = json.loads(after.read_text(encoding="utf-8"))
    before_data = (
        json.loads(before.read_text(encoding="utf-8")) if before.exists() else {}
    )
    return float(after_data.get(section, {}).get(key, 0)) - float(
        before_data.get(section, {}).get(key, 0)
    )


def stable_float(value: float) -> float:
    """Remove irrelevant last-bit differences across CPU architectures."""
    return round(float(value), 12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    metadata_paths = sorted(args.campaign_dir.glob("*/p*/run-*/run-metadata.json"))
    for metadata_path in metadata_paths:
        run_root = metadata_path.parent
        summaries = list((run_root / "client" / "evalscope").glob("**/benchmark_summary.json"))
        if len(summaries) != 1:
            print(f"skip {run_root}: expected one summary, found {len(summaries)}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary_path = summaries[0]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        before = run_root / "server" / "before-metrics.prom"
        after = run_root / "server" / "after-metrics.prom"
        before_stats = run_root / "server" / "before-stats.json"
        after_stats = run_root / "server" / "after-stats.json"
        success_delta = delta(
            before, after, "lite_llama_requests_total", '{endpoint="chat",status="success"}'
        )
        queue_sum = delta(
            before, after, "lite_llama_queue_wait_seconds_sum", '{endpoint="chat"}'
        )
        queue_count = delta(
            before, after, "lite_llama_queue_wait_seconds_count", '{endpoint="chat"}'
        )
        avg_input = summary.get("Avg Input Tokens")
        avg_output = summary.get("Avg Output Tokens")
        rows.append({
            **metadata,
            "client_total_requests": summary.get("Total Requests"),
            "client_success_requests": summary.get("Success Requests"),
            "client_failed_requests": summary.get("Failed Requests"),
            "client_success_rate": (
                summary.get("Success Requests", 0) / summary.get("Total Requests", 1)
            ),
            "client_avg_input_tokens": avg_input,
            "client_avg_output_tokens": avg_output,
            "strict_input_length_valid": (
                avg_input is not None
                and abs(avg_input - metadata["target_server_input_tokens"]) <= 0.1
            ),
            "strict_output_length_valid": (
                avg_output is not None
                and abs(avg_output - metadata["output_tokens"]) <= 0.1
            ),
            "client_e2e_latency_s": summary.get("Avg Latency (s)"),
            "client_ttft_ms": summary.get("TTFT (ms)"),
            "client_tpot_ms": summary.get("TPOT (ms)"),
            "client_itl_ms": summary.get("ITL (ms)"),
            "client_output_throughput_tok_s": summary.get("Output Throughput (tok/s)"),
            "client_total_throughput_tok_s": summary.get("Total Throughput (tok/s)"),
            "client_qps": summary.get("Req Throughput (req/s)"),
            "server_success_delta": success_delta,
            "server_queue_wait_mean_ms": (
                1000 * queue_sum / queue_count
                if queue_sum is not None and queue_count not in (None, 0) else None
            ),
            "server_graph_captures_delta": nested_delta(
                before_stats, after_stats, "npu_graph", "captures"
            ),
            "server_graph_replays_delta": nested_delta(
                before_stats, after_stats, "npu_graph", "replays"
            ),
            "server_graph_fallbacks_delta": nested_delta(
                before_stats, after_stats, "npu_graph", "fallbacks"
            ),
            "raw_summary": summary_path.relative_to(args.campaign_dir).as_posix(),
            "raw_server_before": before.relative_to(args.campaign_dir).as_posix(),
            "raw_server_after": after.relative_to(args.campaign_dir).as_posix(),
        })
    if not rows:
        raise SystemExit("no complete runs found")
    output = args.output or args.campaign_dir / "summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("benchmark_variant", row["graph"]), row["target_server_input_tokens"], row["output_tokens"], row["concurrency"])].append(row)
    aggregates = []
    include_variant = any("benchmark_variant" in row for row in rows)
    metric_names = (
        "client_e2e_latency_s",
        "client_ttft_ms",
        "client_tpot_ms",
        "client_itl_ms",
        "client_output_throughput_tok_s",
        "client_total_throughput_tok_s",
        "client_qps",
        "server_queue_wait_mean_ms",
    )
    for (variant, prompt, output_tokens, concurrency), group in sorted(grouped.items()):
        aggregate = {
            "graph": group[0]["graph"],
            "target_server_input_tokens": prompt,
            "output_tokens": output_tokens,
            "concurrency": concurrency,
            "formal_repeats": len(group),
            "strict_input_valid_runs": sum(bool(r["strict_input_length_valid"]) for r in group),
            "strict_output_valid_runs": sum(bool(r["strict_output_length_valid"]) for r in group),
            "client_success_requests": sum(r["client_success_requests"] for r in group),
            "client_failed_requests": sum(r["client_failed_requests"] for r in group),
            "server_graph_replays_delta": sum((r["server_graph_replays_delta"] or 0) for r in group),
            "server_graph_fallbacks_delta": sum((r["server_graph_fallbacks_delta"] or 0) for r in group),
            "run_ids": ";".join(r["run_id"] for r in group),
        }
        if include_variant:
            aggregate = {
                "graph": aggregate.pop("graph"),
                "benchmark_variant": variant,
                **aggregate,
            }
        for name in metric_names:
            values = [float(r[name]) for r in group if r[name] is not None]
            aggregate[name + "_mean"] = (
                stable_float(statistics.mean(values)) if values else None
            )
            aggregate[name + "_stdev"] = (
                stable_float(statistics.stdev(values))
                if len(values) > 1
                else 0.0 if values else None
            )
        aggregates.append(aggregate)
    aggregate_path = output.with_name("aggregate.csv")
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    print(aggregate_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
