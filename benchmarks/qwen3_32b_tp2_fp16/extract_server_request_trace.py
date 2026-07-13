#!/usr/bin/env python3
"""Extract the formal request window from a benchmark-only server trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract(path: Path, requests: int) -> dict:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    successful = [record for record in records if record.get("status") == "success"]
    if len(successful) < requests:
        raise ValueError(f"trace has {len(successful)} successful records, need {requests}")
    formal = sorted(successful[-requests:], key=lambda record: record["submission_order"])
    for index, record in enumerate(formal, 1):
        record["formal_submission_order"] = index
        if any(record.get(key) is None for key in ("queue_wait_ms", "ttft_ms", "service_to_first_token_ms")):
            raise ValueError(f"formal request {index} is missing timing fields")
    return {
        "schema_version": 1,
        "source_trace": path.name,
        "request_count": len(formal),
        "requests": formal,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = extract(args.trace.resolve(), args.requests)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "requests": report["request_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
