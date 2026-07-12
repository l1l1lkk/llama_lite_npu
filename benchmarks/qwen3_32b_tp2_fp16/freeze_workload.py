#!/usr/bin/env python3
"""Freeze accepted EvalScope request messages into deterministic JSONL datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def single(root: Path, name: str) -> Path:
    matches = list(root.glob(f"**/{name}"))
    if len(matches) != 1:
        raise ValueError(f"{root}: expected one {name}, found {len(matches)}")
    return matches[0]


def request_messages(database: Path) -> list[list[dict]]:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "select request from result order by rowid"
        ).fetchall()
    finally:
        connection.close()
    messages = []
    for (request_json,) in rows:
        request = json.loads(request_json)
        value = request.get("messages")
        if not isinstance(value, list) or not value:
            raise ValueError(f"request has no messages in {database}")
        messages.append(value)
    return messages


def write_dataset(path: Path, messages: list[list[dict]]) -> None:
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in messages
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_campaign", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    source = args.source_campaign.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing dataset directory: {output}")
    output.mkdir(parents=True)
    records = []
    for metadata_path in sorted(source.glob("on/p*/run-*/run-metadata.json")):
        run_root = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pair_id = metadata["pair_id"]
        for role, evalscope_root, expected_count in (
            ("formal", run_root / "client/evalscope", metadata["requests"]),
            (
                "warmup",
                run_root / "client/warmup/evalscope",
                metadata["warmup_requests"],
            ),
        ):
            database = single(evalscope_root, "benchmark_data.db")
            messages = request_messages(database)
            if len(messages) != int(expected_count):
                raise ValueError(
                    f"{pair_id} {role}: {len(messages)} requests, "
                    f"expected {expected_count}"
                )
            dataset_path = output / f"{pair_id}-{role}.jsonl"
            write_dataset(dataset_path, messages)
            records.append({
                "pair_id": pair_id,
                "role": role,
                "request_count": len(messages),
                "dataset_file": dataset_path.name,
                "dataset_size_bytes": dataset_path.stat().st_size,
                "dataset_sha256": sha256(dataset_path),
                "source_database": database.relative_to(source).as_posix(),
                "source_database_size_bytes": database.stat().st_size,
                "source_database_sha256": sha256(database),
            })
    manifest = {
        "schema_version": 1,
        "source_campaign": source.name,
        "dataset_kind": "line_by_line",
        "pair_count": len(records) // 2,
        "files": records,
    }
    (output / "workload-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pairs": manifest["pair_count"], "files": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
