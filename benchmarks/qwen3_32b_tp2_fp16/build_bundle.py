#!/usr/bin/env python3
"""Create a compact, auditable Git bundle from a server benchmark campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT_FILES = ("summary.csv", "aggregate.csv")
ENVIRONMENT_FILES = (
    "ascend-env.txt",
    "baseline.env",
    "git.txt",
    "model-config.json",
    "model-config.sha256",
    "npu-smi.txt",
    "runtime.txt",
)
LIFECYCLE_FILES = (
    "start-command.txt",
    "health.json",
    "start-stats.json",
    "start-metrics.prom",
    "stop-stats.json",
    "stop-metrics.prom",
)
RUN_DIRECT_FILES = (
    "run-metadata.json",
    "client/command.txt",
    "client/exit-code.txt",
    "server/before-metrics.prom",
    "server/after-metrics.prom",
    "server/before-stats.json",
    "server/after-stats.json",
)
EVALSCOPE_FILES = (
    "benchmark_args.json",
    "benchmark_summary.json",
    "benchmark_percentile.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def copy_relative(source_root: Path, output_root: Path, relative: Path) -> None:
    source = source_root / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_campaign", type=Path)
    parser.add_argument("output_bundle", type=Path)
    args = parser.parse_args()
    source = args.source_campaign.resolve()
    output = args.output_bundle.resolve()
    if not source.is_dir():
        raise SystemExit(f"source campaign does not exist: {source}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing bundle: {output}")
    output.mkdir(parents=True)

    selected: set[Path] = set()
    run_ids: list[str] = []
    for name in ROOT_FILES:
        selected.add(Path(name))
    for name in ENVIRONMENT_FILES:
        selected.add(Path("environment") / name)
    for graph in ("on", "off"):
        for name in LIFECYCLE_FILES:
            candidate = Path(graph) / "server" / name
            if (source / candidate).is_file():
                selected.add(candidate)

    for metadata_path in sorted(source.glob("on/p*/run-*/run-metadata.json")) + sorted(
        source.glob("off/p*/run-*/run-metadata.json")
    ):
        run_root = metadata_path.parent
        summaries = list((run_root / "client" / "evalscope").glob("**/benchmark_summary.json"))
        if len(summaries) != 1:
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_ids.append(metadata["run_id"])
        relative_run = run_root.relative_to(source)
        for name in RUN_DIRECT_FILES:
            selected.add(relative_run / name)
        evalscope_root = summaries[0].parent
        for name in EVALSCOPE_FILES:
            candidate = evalscope_root / name
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            selected.add(candidate.relative_to(source))

    for relative in sorted(selected):
        copy_relative(source, output, relative)

    source_files = sorted(path for path in source.rglob("*") if path.is_file())
    omitted = [file_record(path, source) for path in source_files if path.relative_to(source) not in selected]
    included = [file_record(output / relative, output) for relative in sorted(selected)]
    try:
        git_head = subprocess.check_output(
            ["git", "-C", str(source.parent.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = "see environment/git.txt"
    manifest = {
        "schema_version": 1,
        "manifest_self_excluded_from_hash_index": True,
        "campaign_id": source.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_campaign": str(source),
        "git_head": git_head,
        "formal_run_count": len(run_ids),
        "run_ids": run_ids,
        "included_file_count": len(included),
        "included_size_bytes": sum(int(item["size_bytes"]) for item in included),
        "included_files": included,
        "omitted_file_count": len(omitted),
        "omitted_size_bytes": sum(int(item["size_bytes"]) for item in omitted),
        "omitted_files": omitted,
        "regenerate_commands": {
            "server_campaign": (
                "bash benchmarks/qwen3_32b_tp2_fp16/collect_env.sh <campaign>; "
                "bash benchmarks/qwen3_32b_tp2_fp16/start_server.sh on|off <campaign>; "
                "bash benchmarks/qwen3_32b_tp2_fp16/run_case.sh <campaign> <on|off> "
                "<target_prompt> <evalscope_prompt> 256 <concurrency> <requests>"
            ),
            "summaries": (
                "python benchmarks/qwen3_32b_tp2_fp16/summarize.py "
                "benchmarks/results/20260711_qwen3_32b_tp2_fp16"
            ),
            "bundle_validation": (
                "python benchmarks/qwen3_32b_tp2_fp16/validate_bundle.py "
                "benchmarks/results/20260711_qwen3_32b_tp2_fp16"
            ),
        },
    }
    (output / "campaign-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "formal_runs": len(run_ids),
        "included_files": len(included) + 1,
        "included_size_bytes": manifest["included_size_bytes"],
        "omitted_files": len(omitted),
        "omitted_size_bytes": manifest["omitted_size_bytes"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
