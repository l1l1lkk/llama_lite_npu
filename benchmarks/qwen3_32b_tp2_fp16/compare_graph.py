#!/usr/bin/env python3
"""Validate paired Graph on/off runs and calculate paired effects."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = {
    "e2e_latency_s": ("client_e2e_latency_s", "latency"),
    "ttft_ms": ("client_ttft_ms", "latency"),
    "tpot_ms": ("client_tpot_ms", "latency"),
    "itl_ms": ("client_itl_ms", "latency"),
    "output_throughput_tok_s": (
        "client_output_throughput_tok_s",
        "throughput",
    ),
    "total_throughput_tok_s": (
        "client_total_throughput_tok_s",
        "throughput",
    ),
    "qps": ("client_qps", "throughput"),
    "queue_wait_ms": ("server_queue_wait_mean_ms", "latency"),
}
PAIR_FIELDS = (
    "target_server_input_tokens",
    "evalscope_prompt_tokens",
    "min_tokens",
    "output_tokens",
    "concurrency",
    "requests",
    "warmup_requests",
    "warmup_dataset_offset",
    "separate_warmup",
    "dataset_kind",
    "formal_dataset_sha256",
    "warmup_dataset_sha256",
    "seed",
    "dataset_offset",
    "temperature",
    "top_p",
    "sampling",
    "evalscope_version",
)
IGNORED_EVALSCOPE_ARGS = {"name", "outputs_dir"}


def stable(value: float) -> float:
    return round(float(value), 12)


def sha256_json(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def single(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"{root}: expected one {pattern}, found {len(matches)}")
    return matches[0]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_args(path: Path) -> dict:
    data = read_json(path)
    return {key: value for key, value in data.items() if key not in IGNORED_EVALSCOPE_ARGS}


def normalized_server_command(command: str) -> str:
    return " ".join(
        command.replace("--no_compiled_model", "<GRAPH_FLAG>")
        .replace("--compiled_model", "<GRAPH_FLAG>")
        .split()
    )


def graph_stats(path: Path) -> dict[str, int]:
    stats = read_json(path).get("npu_graph", {})
    return {
        key: int(stats.get(key, 0))
        for key in ("capture_attempts", "captures", "replays", "fallbacks")
    }


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["run_id"]: row for row in csv.DictReader(handle)}


def metric_effect(on: float, off: float, kind: str) -> tuple[float, float, str]:
    if kind == "latency":
        return stable(off / on), stable((off - on) / off * 100), "off/on"
    return stable(on / off), stable((on - off) / off * 100), "on/off"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(campaign: Path) -> tuple[list[dict], list[dict], dict]:
    summary = read_summary(campaign / "summary.csv")
    grouped = defaultdict(dict)
    metadata_paths = sorted([
        *campaign.glob("on/p*/run-*/run-metadata.json"),
        *campaign.glob("off/p*/run-*/run-metadata.json"),
    ])
    for path in metadata_paths:
        metadata = read_json(path)
        grouped[metadata["pair_id"]][metadata["graph"]] = (path.parent, metadata)
    if not grouped:
        raise ValueError("no paired run metadata found")

    on_command = (campaign / "on/server/start-command.txt").read_text(encoding="utf-8")
    off_command = (campaign / "off/server/start-command.txt").read_text(encoding="utf-8")
    normalized_on_command = normalized_server_command(on_command)
    normalized_off_command = normalized_server_command(off_command)
    if normalized_on_command != normalized_off_command:
        raise ValueError("server commands differ beyond the Graph flag")

    pair_rows = []
    validation_pairs = []
    for pair_id, modes in sorted(grouped.items()):
        if set(modes) != {"on", "off"}:
            raise ValueError(f"{pair_id}: expected on/off, got {sorted(modes)}")
        on_root, on_meta = modes["on"]
        off_root, off_meta = modes["off"]
        checks = {}
        checks["metadata_fields_equal"] = all(
            on_meta.get(field) == off_meta.get(field) for field in PAIR_FIELDS
        )

        on_args_path = single(on_root / "client/evalscope", "**/benchmark_args.json")
        off_args_path = single(off_root / "client/evalscope", "**/benchmark_args.json")
        on_args = normalized_args(on_args_path)
        off_args = normalized_args(off_args_path)
        checks["evalscope_args_equal"] = on_args == off_args

        on_fingerprint = read_json(on_root / "client/workload-fingerprint.json")
        off_fingerprint = read_json(off_root / "client/workload-fingerprint.json")
        on_warmup = read_json(on_root / "client/warmup/workload-fingerprint.json")
        off_warmup = read_json(off_root / "client/warmup/workload-fingerprint.json")
        checks["formal_prompt_tokens_equal"] = (
            on_fingerprint["requests"] == off_fingerprint["requests"]
        )
        checks["warmup_prompt_tokens_equal"] = on_warmup["requests"] == off_warmup["requests"]
        checks["formal_prompt_hash_equal"] = (
            on_fingerprint["prompt_sequence_sha256"]
            == off_fingerprint["prompt_sequence_sha256"]
        )
        checks["warmup_prompt_hash_equal"] = (
            on_warmup["prompt_sequence_sha256"]
            == off_warmup["prompt_sequence_sha256"]
        )

        graph_evidence = {}
        for mode, (root, _) in modes.items():
            pre = graph_stats(root / "server/pre-warmup-stats.json")
            before = graph_stats(root / "server/before-stats.json")
            after = graph_stats(root / "server/after-stats.json")
            graph_evidence[mode] = {"pre_warmup": pre, "before_formal": before, "after_formal": after}
        on_graph = graph_evidence["on"]
        off_graph = graph_evidence["off"]
        checks["graph_on_captured_before_formal"] = (
            on_graph["before_formal"]["captures"] > 0
            and on_graph["before_formal"]["replays"] > 0
        )
        checks["graph_on_no_formal_capture"] = (
            on_graph["after_formal"]["captures"]
            == on_graph["before_formal"]["captures"]
        )
        checks["graph_on_no_fallback"] = all(
            snapshot["fallbacks"] == 0 for snapshot in on_graph.values()
        )
        checks["graph_off_all_zero"] = all(
            value == 0
            for snapshot in off_graph.values()
            for value in snapshot.values()
        )
        if not all(checks.values()):
            failed = [key for key, value in checks.items() if not value]
            raise ValueError(f"{pair_id}: pairing checks failed: {failed}")

        on_summary = summary[on_meta["run_id"]]
        off_summary = summary[off_meta["run_id"]]
        pair_row = {
            "pair_id": pair_id,
            "concurrency": int(on_meta["concurrency"]),
            "repeat": int(pair_id.rsplit("r", 1)[1]),
            "on_run_id": on_meta["run_id"],
            "off_run_id": off_meta["run_id"],
            "requests_per_run": int(on_meta["requests"]),
            "warmup_requests": int(on_meta["warmup_requests"]),
            "seed": int(on_meta["seed"]),
            "dataset_offset": int(on_meta["dataset_offset"]),
            "prompt_sequence_sha256": on_fingerprint["prompt_sequence_sha256"],
        }
        for metric, (column, kind) in METRICS.items():
            on_value = float(on_summary[column])
            off_value = float(off_summary[column])
            ratio, effect, definition = metric_effect(on_value, off_value, kind)
            pair_row[f"{metric}_on"] = on_value
            pair_row[f"{metric}_off"] = off_value
            pair_row[f"{metric}_ratio"] = ratio
            pair_row[f"{metric}_effect_pct"] = effect
            pair_row[f"{metric}_ratio_definition"] = definition
        pair_rows.append(pair_row)
        validation_pairs.append({
            "pair_id": pair_id,
            "checks": checks,
            "formal_args_sha256": sha256_json(on_args),
            "formal_prompt_sequence_sha256": on_fingerprint["prompt_sequence_sha256"],
            "warmup_prompt_sequence_sha256": on_warmup["prompt_sequence_sha256"],
            "graph_evidence": graph_evidence,
        })

    comparison_rows = []
    for concurrency in sorted({int(row["concurrency"]) for row in pair_rows}):
        group = [row for row in pair_rows if int(row["concurrency"]) == concurrency]
        for metric, (_, kind) in METRICS.items():
            on_values = [float(row[f"{metric}_on"]) for row in group]
            off_values = [float(row[f"{metric}_off"]) for row in group]
            ratios = [float(row[f"{metric}_ratio"]) for row in group]
            ratio, effect, definition = metric_effect(
                statistics.mean(on_values), statistics.mean(off_values), kind
            )
            comparison_rows.append({
                "concurrency": concurrency,
                "metric": metric,
                "kind": kind,
                "on_mean": stable(statistics.mean(on_values)),
                "on_stdev": stable(statistics.stdev(on_values)),
                "off_mean": stable(statistics.mean(off_values)),
                "off_stdev": stable(statistics.stdev(off_values)),
                "paired_ratio_mean": stable(statistics.mean(ratios)),
                "paired_ratio_stdev": stable(statistics.stdev(ratios)),
                "ratio_of_means": ratio,
                "effect_pct_from_means": effect,
                "ratio_definition": definition,
            })

    validation = {
        "schema_version": 1,
        "status": "pass",
        "campaign_id": campaign.name,
        "pair_count": len(pair_rows),
        "normalized_server_command": normalized_on_command,
        "server_commands_equal_except_graph_flag": True,
        "pairs": validation_pairs,
    }
    return pair_rows, comparison_rows, validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    pair_rows, comparison_rows, validation = build(campaign)
    outputs = {
        campaign / "paired-ratios.csv": pair_rows,
        campaign / "graph-comparison.csv": comparison_rows,
    }
    validation_path = campaign / "pair-validation.json"
    if args.compare:
        for path, rows in outputs.items():
            temporary = path.with_suffix(path.suffix + ".rebuilt")
            write_csv(temporary, rows)
            if path.read_bytes() != temporary.read_bytes():
                temporary.unlink()
                raise SystemExit(f"{path.name} differs from rebuilt data")
            temporary.unlink()
        serialized = json.dumps(validation, indent=2, ensure_ascii=False) + "\n"
        if validation_path.read_text(encoding="utf-8") != serialized:
            raise SystemExit("pair-validation.json differs from rebuilt data")
    else:
        for path, rows in outputs.items():
            write_csv(path, rows)
        validation_path.write_text(
            json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": "pass", "pairs": len(pair_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
