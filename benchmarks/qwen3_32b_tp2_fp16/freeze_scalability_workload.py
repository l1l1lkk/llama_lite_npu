#!/usr/bin/env python3
"""Build two deterministic 32-request JSONL sets from audited frozen inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=32)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")
    source_paths = sorted(source_root.glob("*.jsonl"))
    if not source_paths:
        raise SystemExit("no source JSONL files")
    unique = []
    seen = set()
    sources = []
    for path in source_paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        sources.append({
            "path": path.relative_to(source_root.parent.parent).as_posix(),
            "sha256": sha256(path),
            "line_count": len(lines),
        })
        for line in lines:
            value = json.loads(line)
            if not isinstance(value, list):
                raise ValueError(f"{path}: each line must be a message list")
            normalized = canonical(value)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(value)
    required = args.requests * 2
    if len(unique) < required:
        raise SystemExit(f"need {required} unique requests, found {len(unique)}")
    output_root.mkdir(parents=True)
    outputs = []
    for name, values in (
        ("p128_o256_n32-formal.jsonl", unique[:args.requests]),
        ("p128_o256_n32-warmup.jsonl", unique[args.requests:required]),
    ):
        path = output_root / name
        path.write_text(
            "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
            encoding="utf-8",
        )
        outputs.append({"path": name, "request_count": len(values), "sha256": sha256(path)})
    manifest = {
        "schema_version": 1,
        "workload_id": "qwen3_32b_p128_o256_n32_scalability_v1",
        "selection": "first 64 unique message lists in sorted source path and line order",
        "sources": sources,
        "outputs": outputs,
    }
    (output_root / "workload-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "unique_inputs": len(unique), "outputs": outputs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
