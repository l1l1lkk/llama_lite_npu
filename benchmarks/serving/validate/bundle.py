"""Offline, defensive SHA/size validation for canonical v2 bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from ..bundle import rebuild_aggregate


SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
MANIFEST_FIELDS = {
    "schema_version",
    "manifest_self_excluded",
    "file_count",
    "size_bytes",
    "files",
}
ENTRY_FIELDS = {"path", "size_bytes", "sha256"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str, *, reject_bom: bool = False) -> tuple[Any | None, list[dict[str, Any]]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, [{"kind": f"{label}_read_error", "detail": str(exc)}]
    if raw.startswith(b"\xef\xbb\xbf"):
        if reject_bom:
            return None, [{"kind": f"{label}_bom_not_allowed"}]
        raw = raw[3:]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, [{"kind": f"{label}_encoding_invalid", "offset": exc.start}]
    try:
        return json.loads(text), []
    except json.JSONDecodeError as exc:
        return None, [{"kind": f"{label}_json_invalid", "line": exc.lineno, "column": exc.colno}]


def _canonical_relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    if value == "manifest.json" or value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        return None
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    normalized = PurePosixPath(value).as_posix()
    return value if normalized == value else None


def _validate_omitted(root: Path) -> list[dict[str, Any]]:
    path = root / "omitted.json"
    if not path.is_file():
        return [{"kind": "omitted_metadata_missing"}]
    omitted, errors = _read_json(path, "omitted")
    if errors:
        return errors
    if not isinstance(omitted, list):
        return [{"kind": "omitted_metadata_invalid"}]
    required = {"kind", "path", "size_bytes", "sha256"}
    for index, entry in enumerate(omitted):
        if not isinstance(entry, dict) or not required.issubset(entry):
            errors.append({"kind": "omitted_entry_invalid", "index": index})
            continue
        size = entry["size_bytes"]
        sha = entry["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            errors.append({"kind": "omitted_entry_invalid", "index": index})
        elif not isinstance(sha, str) or SHA256_RE.fullmatch(sha) is None:
            errors.append({"kind": "omitted_entry_invalid", "index": index})
    return errors


def validate_bundle(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    errors: list[dict[str, Any]] = []
    if not root.exists():
        return {"schema_version": 2, "status": "fail", "errors": [{"kind": "bundle_root_missing"}]}
    if not root.is_dir():
        return {"schema_version": 2, "status": "fail", "errors": [{"kind": "bundle_root_not_directory"}]}

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {"schema_version": 2, "status": "fail", "errors": [{"kind": "manifest_missing"}]}
    manifest, parse_errors = _read_json(manifest_path, "manifest", reject_bom=True)
    if parse_errors:
        return {"schema_version": 2, "status": "fail", "errors": parse_errors}
    if not isinstance(manifest, dict):
        return {"schema_version": 2, "status": "fail", "errors": [{"kind": "manifest_not_mapping"}]}

    missing_fields = sorted(MANIFEST_FIELDS - manifest.keys())
    for field in missing_fields:
        errors.append({"kind": field if field.startswith("manifest_") else f"manifest_{field}"})
    if "schema_version" in manifest and manifest["schema_version"] != 2:
        errors.append({"kind": "manifest_schema_version"})
    if "manifest_self_excluded" in manifest and manifest["manifest_self_excluded"] is not True:
        errors.append({"kind": "manifest_self_excluded"})

    files = manifest.get("files")
    if "files" in manifest and not isinstance(files, list):
        errors.append({"kind": "manifest_files_type"})
        files = []
    elif files is None:
        files = []
    file_count = manifest.get("file_count")
    if "file_count" in manifest and (
        isinstance(file_count, bool) or not isinstance(file_count, int) or file_count < 0
    ):
        errors.append({"kind": "manifest_file_count_type"})
    elif "file_count" in manifest and file_count != len(files):
        errors.append({"kind": "manifest_file_count", "declared": file_count, "actual": len(files)})
    size_bytes = manifest.get("size_bytes")
    if "size_bytes" in manifest and (
        isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0
    ):
        errors.append({"kind": "manifest_size_bytes_type"})

    valid_entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    declared_size = 0
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            errors.append({"kind": "manifest_entry_type", "index": index})
            continue
        if not ENTRY_FIELDS.issubset(entry):
            errors.append({"kind": "manifest_entry_fields", "index": index})
            continue
        relative = _canonical_relative_path(entry.get("path"))
        if relative is None:
            errors.append({"kind": "manifest_path_invalid", "index": index, "path": entry.get("path")})
            continue
        if relative in seen:
            errors.append({"kind": "manifest_duplicate_path", "index": index, "path": relative})
        seen.add(relative)
        entry_size = entry.get("size_bytes")
        entry_sha = entry.get("sha256")
        if isinstance(entry_size, bool) or not isinstance(entry_size, int) or entry_size < 0:
            errors.append({"kind": "manifest_entry_size_type", "index": index, "path": relative})
            continue
        if not isinstance(entry_sha, str) or SHA256_RE.fullmatch(entry_sha) is None:
            errors.append({"kind": "manifest_entry_sha256", "index": index, "path": relative})
            continue
        declared_size += entry_size
        valid_entries.append(entry)
    if isinstance(size_bytes, int) and not isinstance(size_bytes, bool) and size_bytes != declared_size:
        errors.append({"kind": "manifest_size_bytes", "declared": size_bytes, "actual": declared_size})

    errors.extend(_validate_omitted(root))
    expected_paths = {entry["path"] for entry in valid_entries}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    for missing in sorted(expected_paths - actual_paths):
        errors.append({"kind": "file_missing", "path": missing})
    for unexpected in sorted(actual_paths - expected_paths):
        errors.append({"kind": "untracked_evidence", "path": unexpected})

    root_resolved = root.resolve()
    for entry in valid_entries:
        path = root / entry["path"]
        try:
            resolved = path.resolve()
        except OSError as exc:
            errors.append({"kind": "file_resolve_error", "path": entry["path"], "detail": str(exc)})
            continue
        if resolved != root_resolved and root_resolved not in resolved.parents:
            errors.append({"kind": "manifest_path_escape", "path": entry["path"]})
            continue
        if not path.is_file():
            continue
        try:
            actual_sha = _sha256(path)
            actual_size = path.stat().st_size
        except OSError as exc:
            errors.append({"kind": "file_read_error", "path": entry["path"], "detail": str(exc)})
            continue
        if actual_sha != entry["sha256"]:
            errors.append({"kind": "sha256_mismatch", "path": entry["path"]})
        if actual_size != entry["size_bytes"]:
            errors.append({"kind": "size_mismatch", "path": entry["path"]})

    return {
        "schema_version": 2,
        "status": "pass" if not errors else "fail",
        "checked_files": len(valid_entries),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_bundle(args.root)
        if args.rebuild and report["status"] == "pass":
            report = dict(report)
            report["aggregate"] = rebuild_aggregate(args.root)
            report = {**validate_bundle(args.root), "aggregate": report["aggregate"]}
    except Exception as exc:  # CLI boundary: invalid evidence must never emit a traceback.
        report = {
            "schema_version": 2,
            "status": "fail",
            "errors": [{"kind": "validator_internal_error", "detail": f"{type(exc).__name__}: {exc}"}],
        }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
