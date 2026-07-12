#!/usr/bin/env python3
"""Verify bundle hashes and rebuild summaries without server-only artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def equivalent(expected: list[dict[str, str]], rebuilt: list[dict[str, str]]) -> bool:
    if len(expected) != len(rebuilt):
        return False
    for expected_row, rebuilt_row in zip(expected, rebuilt):
        if expected_row.keys() != rebuilt_row.keys():
            return False
        for key in expected_row:
            left = expected_row[key].replace("\\", "/")
            right = rebuilt_row[key].replace("\\", "/")
            if left == right:
                continue
            try:
                if math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12):
                    continue
            except ValueError:
                pass
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "campaign-manifest.json").read_text(encoding="utf-8"))
    for record in manifest["included_files"]:
        path = bundle / record["path"]
        if not path.is_file():
            raise SystemExit(f"missing included file: {record['path']}")
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise SystemExit(f"hash/size mismatch: {record['path']}")
    expected_summary = rows(bundle / "summary.csv")
    expected_aggregate = rows(bundle / "aggregate.csv")
    script = Path(__file__).with_name("summarize.py").resolve()
    with tempfile.TemporaryDirectory(prefix="qwen3-benchmark-bundle-") as temp:
        copied = Path(temp) / bundle.name
        shutil.copytree(bundle, copied)
        subprocess.run([sys.executable, str(script), str(copied)], check=True)
        if (copied / "pair-validation.json").is_file():
            compare_script = Path(__file__).with_name("compare_graph.py").resolve()
            subprocess.run(
                [sys.executable, str(compare_script), str(copied), "--compare"],
                check=True,
            )
        rebuilt_summary = rows(copied / "summary.csv")
        rebuilt_aggregate = rows(copied / "aggregate.csv")
    if not equivalent(expected_summary, rebuilt_summary):
        raise SystemExit("rebuilt summary.csv differs from tracked bundle")
    if not equivalent(expected_aggregate, rebuilt_aggregate):
        raise SystemExit("rebuilt aggregate.csv differs from tracked bundle")
    if len(expected_summary) != manifest["formal_run_count"]:
        raise SystemExit("formal run count does not match summary.csv")
    print(json.dumps({
        "status": "ok",
        "formal_runs": len(expected_summary),
        "aggregate_rows": len(expected_aggregate),
        "verified_included_files": len(manifest["included_files"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
