#!/usr/bin/env python3
"""Measure streaming TTFT/TPOT for a frozen chat workload."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import requests


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def git_value(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_workload(path: Path) -> list[list[dict]]:
    messages = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    if not messages:
        raise ValueError(f"workload is empty: {path}")
    return messages


def run_request(
    *,
    request_id: int,
    messages: list[dict],
    endpoint: str,
    model: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at = None
    token_times = []
    content_parts = []
    usage = None
    event_count = 0
    with requests.post(
        endpoint,
        json=payload,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            event_count += 1
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            for choice in choices:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    token_times.append(now)
                    content_parts.append(content)
    finished = time.perf_counter()

    if first_token_at is None:
        raise RuntimeError(f"request {request_id} produced no streamed content")
    if usage is None:
        raise RuntimeError(f"request {request_id} produced no usage event")
    output_tokens = int(usage["completion_tokens"])
    if output_tokens != max_tokens:
        raise RuntimeError(
            f"request {request_id} output token mismatch: "
            f"{output_tokens} != {max_tokens}"
        )
    tpot_ms = (
        (finished - first_token_at) * 1000.0 / (output_tokens - 1)
        if output_tokens > 1
        else 0.0
    )
    content = "".join(content_parts)
    return {
        "request_id": request_id,
        "success": True,
        "input_tokens": int(usage["prompt_tokens"]),
        "output_tokens": output_tokens,
        "ttft_ms": (first_token_at - started) * 1000.0,
        "tpot_ms": tpot_ms,
        "e2e_ms": (finished - started) * 1000.0,
        "stream_content_events": len(token_times),
        "sse_events": event_count,
        "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "output_text": content,
    }


def summarize(rows: list[dict], wall_time_s: float) -> dict:
    result = {
        "requests": len(rows),
        "failed_requests": 0,
        "wall_time_s": wall_time_s,
        "output_throughput_tokens_per_s": (
            sum(row["output_tokens"] for row in rows) / wall_time_s
        ),
    }
    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [float(row[metric]) for row in rows]
        result[f"{metric}_mean"] = statistics.fmean(values)
        result[f"{metric}_p50"] = percentile(values, 0.50)
        result[f"{metric}_p90"] = percentile(values, 0.90)
        result[f"{metric}_max"] = max(values)
    result["input_tokens_min"] = min(row["input_tokens"] for row in rows)
    result["input_tokens_max"] = max(row["input_tokens"] for row in rows)
    result["output_tokens_min"] = min(row["output_tokens"] for row in rows)
    result["output_tokens_max"] = max(row["output_tokens"] for row in rows)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen3-1.7B")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--server-branch", required=True)
    parser.add_argument("--server-commit", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload = load_workload(args.workload)
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"

    for index in range(args.warmup_requests):
        run_request(
            request_id=-(index + 1),
            messages=workload[index % len(workload)],
            endpoint=endpoint,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency,
    ) as executor:
        futures = [
            executor.submit(
                run_request,
                request_id=index,
                messages=workload[index % len(workload)],
                endpoint=endpoint,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            for index in range(args.requests)
        ]
        rows = [future.result() for future in futures]
    wall_time_s = time.perf_counter() - started
    rows.sort(key=lambda row: row["request_id"])

    report = {
        "schema_version": 1,
        "metadata": {
            "case_id": args.case_id,
            "repeat": args.repeat,
            "client_branch": git_value(args.checkout, "branch", "--show-current"),
            "client_commit": git_value(args.checkout, "rev-parse", "HEAD"),
            "server_branch": args.server_branch,
            "server_commit": args.server_commit,
            "base_url": args.base_url,
            "model": args.model,
            "workload": str(args.workload.resolve()),
            "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
            "concurrency": args.concurrency,
            "submitted_requests": args.requests,
            "warmup_requests": args.warmup_requests,
            "max_tokens": args.max_tokens,
        },
        "summary": summarize(rows, wall_time_s),
        "requests": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", **report["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
