#!/usr/bin/env python3
"""验证固定输入/输出 token workload，并生成可复算的逐 run 证据。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def single(path_root: Path, pattern: str) -> Path:
    matches = list(path_root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"{path_root}: expected one {pattern}, found {len(matches)}"
        )
    return matches[0]


def exact(value, expected) -> bool:
    return value is not None and abs(float(value) - float(expected)) < 1e-9


def validate_run(metadata_path: Path, campaign: Path) -> dict[str, object]:
    run_root = metadata_path.parent
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary_path = single(
        run_root / "client" / "evalscope", "**/benchmark_summary.json"
    )
    percentile_path = single(
        run_root / "client" / "evalscope", "**/benchmark_percentile.json"
    )
    args_path = single(
        run_root / "client" / "evalscope", "**/benchmark_args.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    percentiles = json.loads(percentile_path.read_text(encoding="utf-8"))
    benchmark_args = json.loads(args_path.read_text(encoding="utf-8"))
    exit_code = int(
        (run_root / "client" / "exit-code.txt").read_text(encoding="utf-8")
    )
    target_input = int(metadata["target_server_input_tokens"])
    target_output = int(metadata["output_tokens"])
    input_values = [row.get("Input tokens") for row in percentiles]
    output_values = [row.get("Output tokens") for row in percentiles]
    fingerprint_path = run_root / "client" / "workload-fingerprint.json"
    fingerprint = (
        json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if fingerprint_path.is_file()
        else None
    )
    checks = {
        "exit_code_zero": exit_code == 0,
        "zero_failed_requests": summary.get("Failed Requests") == 0,
        "all_requests_succeeded": (
            summary.get("Success Requests") == summary.get("Total Requests")
            == metadata.get("requests")
        ),
        "args_fixed_output": (
            benchmark_args.get("min_tokens") == target_output
            and benchmark_args.get("max_tokens") == target_output
        ),
        "args_greedy": (
            exact(benchmark_args.get("temperature"), 0)
            and exact(benchmark_args.get("top_p"), 1)
        ),
        "strict_input_length": (
            exact(summary.get("Avg Input Tokens"), target_input)
            and bool(input_values)
            and all(exact(value, target_input) for value in input_values)
        ),
        "strict_output_length": (
            exact(summary.get("Avg Output Tokens"), target_output)
            and bool(output_values)
            and all(exact(value, target_output) for value in output_values)
        ),
    }
    if fingerprint is not None:
        checks["request_level_fingerprint_valid"] = (
            fingerprint.get("request_count") == metadata.get("requests")
            and all(
                len(request.get("prompt_token_ids", [])) == target_input
                and request.get("prompt_tokens") == target_input
                and request.get("completion_tokens") == target_output
                and request.get("success") == 1
                for request in fingerprint.get("requests", [])
            )
        )
    if metadata.get("separate_warmup"):
        checks["formal_evalscope_warmup_zero_when_separate"] = exact(
            benchmark_args.get("warmup_num"), 0
        )
    result = {
        "run_id": metadata["run_id"],
        "graph": metadata["graph"],
        "concurrency": metadata["concurrency"],
        "requests": metadata["requests"],
        "seed": metadata["seed"],
        "dataset_offset": metadata["dataset_offset"],
        "target_input_tokens": target_input,
        "target_output_tokens": target_output,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
        "raw_summary": summary_path.relative_to(campaign).as_posix(),
        "raw_percentile": percentile_path.relative_to(campaign).as_posix(),
        "raw_args": args_path.relative_to(campaign).as_posix(),
    }
    if fingerprint is not None:
        result["prompt_sequence_sha256"] = fingerprint.get(
            "prompt_sequence_sha256"
        )
    if "benchmark_variant" in metadata:
        result["benchmark_variant"] = metadata["benchmark_variant"]
    return result


def build_report(campaign: Path) -> dict[str, object]:
    metadata_paths = sorted(campaign.glob("*/p*/run-*/run-metadata.json"))
    runs = [validate_run(path, campaign) for path in metadata_paths]
    if not runs:
        raise ValueError(f"no runs found under {campaign}")
    cases: dict[str, dict[str, object]] = {}
    for run in runs:
        key = f"{run.get('benchmark_variant', run['graph'])}:c{run['concurrency']}"
        case = cases.setdefault(key, {"runs": 0, "passed": 0, "run_ids": []})
        case["runs"] += 1
        case["passed"] += run["status"] == "pass"
        case["run_ids"].append(run["run_id"])
    return {
        "schema_version": 1,
        "campaign_id": campaign.name,
        "status": "pass" if all(run["status"] == "pass" for run in runs) else "fail",
        "formal_run_count": len(runs),
        "cases": cases,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    report = build_report(campaign)
    output = args.output or campaign / "strict-validation.json"
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.compare:
        if not output.is_file() or output.read_text(encoding="utf-8") != serialized:
            raise SystemExit("strict-validation.json does not match rebuilt data")
    else:
        output.write_text(serialized, encoding="utf-8")
    print(json.dumps({"status": report["status"], "runs": len(report["runs"])}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
