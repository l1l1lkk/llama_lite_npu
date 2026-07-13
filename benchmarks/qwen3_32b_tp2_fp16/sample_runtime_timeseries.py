#!/usr/bin/env python3
"""Sample benchmark-only server/NPU state without changing inference semantics."""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


RUNNING = True


def stop(_signum, _frame) -> None:
    global RUNNING
    RUNNING = False


def fetch_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_npu(output: str) -> dict[str, float | int | None]:
    values: dict[str, float | int | None] = {}
    fields = {
        "HBM Usage Rate(%)": "hbm_usage_pct",
        "Aicore Usage Rate(%)": "aicore_usage_pct",
        "Aivector Usage Rate(%)": "aivector_usage_pct",
        "NPU Utilization(%)": "npu_utilization_pct",
    }
    for label, key in fields.items():
        match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)", output)
        values[key] = float(match.group(1)) if match else None
    return values


def sample_npus(ids: list[str], timeout: float) -> dict[str, dict]:
    samples = {}
    for device_id in ids:
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "usages", "-i", device_id],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
            samples[device_id] = {"status": "ok", **parse_npu(result.stdout)}
        except (OSError, subprocess.SubprocessError) as error:
            samples[device_id] = {
                "status": "error",
                "error": type(error).__name__,
            }
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--npu-interval", type=float, default=5.0)
    parser.add_argument("--npu-devices", default="6,7")
    parser.add_argument("--request-timeout", type=float, default=0.8)
    args = parser.parse_args()
    if args.interval <= 0 or args.npu_interval <= 0:
        raise SystemExit("sampling intervals must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    devices = [value.strip() for value in args.npu_devices.split(",") if value.strip()]
    next_tick = time.monotonic()
    next_npu = next_tick
    index = 0
    with args.output.open("w", encoding="utf-8", buffering=1) as handle:
        while RUNNING:
            now = time.monotonic()
            record = {
                "schema_version": 1,
                "sample_index": index,
                "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                "monotonic_s": now,
                "server": {"status": "error"},
                "npu": None,
            }
            try:
                snapshot = fetch_json(
                    args.server_url.rstrip("/") + "/debug/stats",
                    args.request_timeout,
                )
                record["server"] = {
                    "status": "ok",
                    "scheduler": snapshot.get("scheduler", {}),
                    "kv_cache": snapshot.get("kv_cache", {}),
                    "npu_graph": snapshot.get("npu_graph", {}),
                    "requests": snapshot.get("requests", {}),
                    "failures": snapshot.get("failures", {}),
                }
            except Exception as error:  # evidence must retain scrape failures
                record["server"] = {
                    "status": "error",
                    "error": type(error).__name__,
                }
            if now >= next_npu:
                record["npu"] = sample_npus(devices, max(1.0, args.request_timeout * 3))
                next_npu = now + args.npu_interval
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            index += 1
            next_tick += args.interval
            delay = max(0.0, next_tick - time.monotonic())
            if delay:
                time.sleep(delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
