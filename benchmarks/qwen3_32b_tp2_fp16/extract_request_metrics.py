#!/usr/bin/env python3
"""Export compact per-request timing fields from an EvalScope SQLite result."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path


def find_single(root: Path, name: str) -> Path:
    matches = list(root.glob(f"**/{name}"))
    if len(matches) != 1:
        raise ValueError(f"{root}: expected one {name}, found {len(matches)}")
    return matches[0]


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract(evalscope_root: Path) -> dict:
    database = find_single(evalscope_root, "benchmark_data.db")
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "select rowid, request, start_time, completed_time, latency, "
            "first_chunk_latency, inter_token_latencies, prompt_tokens, "
            "completion_tokens, success, time_per_output_token "
            "from result order by rowid"
        ).fetchall()
    finally:
        connection.close()
    requests = []
    for row in rows:
        (
            rowid,
            request_json,
            start_time,
            completed_time,
            latency,
            first_chunk_latency,
            itl_json,
            prompt_tokens,
            completion_tokens,
            success,
            tpot,
        ) = row
        body = json.loads(request_json)
        itls = [float(value) for value in json.loads(itl_json or "[]")]
        requests.append({
            "completion_order": int(rowid),
            "request_sha256": canonical_sha256(body),
            "start_time_s": float(start_time),
            "completed_time_s": float(completed_time),
            "latency_s": float(latency),
            "ttft_ms": float(first_chunk_latency) * 1000,
            "tpot_ms": float(tpot) * 1000,
            "itl_mean_ms": statistics.mean(itls) * 1000 if itls else None,
            "itl_count": len(itls),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "success": int(success),
        })
    return {
        "schema_version": 1,
        "source_database": database.relative_to(evalscope_root).as_posix(),
        "request_count": len(requests),
        "requests": requests,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evalscope_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = extract(args.evalscope_root.resolve())
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "requests": report["request_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
