"""Create and rebuild compact canonical v2 evidence bundles."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


FORBIDDEN_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".html", ".prof"}
FORBIDDEN_NAMES = {"server.log", "stdout.log"}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_bundle(root: str | Path, campaign: Mapping[str, Any]) -> Path:
    root = Path(root)
    for relative in ("environment", "workload", "runs", "derived", "diagnostics"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _json_dump(root / "campaign.json", campaign)
    _json_dump(root / "diagnostics/index.json", {"schema_version": 2, "entries": []})
    _json_dump(root / "omitted.json", [])
    return root


def rebuild_aggregate(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    request_files = sorted(root.glob("runs/*/*/repeat-*/client/requests.json"))
    requests: list[dict[str, Any]] = []
    for path in request_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"request evidence must be a list: {path}")
        requests.extend(value)
    if not requests:
        raise ValueError("bundle has no request-level evidence")
    success = [item for item in requests if item.get("success")]
    aggregate = {
        "schema_version": 2,
        "request_count": len(requests),
        "success_count": len(success),
        "failed_count": len(requests) - len(success),
        "mean_input_tokens": sum(float(item["input_tokens"]) for item in requests) / len(requests),
        "mean_output_tokens": sum(float(item["output_tokens"]) for item in requests) / len(requests),
        "mean_e2e_s": sum(float(item["e2e_s"]) for item in requests) / len(requests),
        "mean_ttft_ms": sum(float(item["ttft_ms"]) for item in requests) / len(requests),
        "source_request_files": [path.relative_to(root).as_posix() for path in request_files],
    }
    _json_dump(root / "derived/aggregate.json", aggregate)
    csv_path = root / "derived/aggregate.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in aggregate if key not in {"schema_version", "source_request_files"}])
        writer.writeheader()
        writer.writerow({key: value for key, value in aggregate.items() if key not in {"schema_version", "source_request_files"}})
    return aggregate


def build_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    root_resolved = root.resolve()
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"):
        resolved = path.resolve()
        if path.is_symlink() or (resolved != root_resolved and root_resolved not in resolved.parents):
            raise ValueError(f"bundle evidence must not escape through a symlink: {path}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.lower() in FORBIDDEN_NAMES:
            raise ValueError(f"bulky artifact must be represented in omitted.json: {path}")
        entries.append({"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = {
        "schema_version": 2,
        "manifest_self_excluded": True,
        "file_count": len(entries),
        "size_bytes": sum(item["size_bytes"] for item in entries),
        "files": entries,
    }
    _json_dump(root / "manifest.json", manifest)
    return manifest
