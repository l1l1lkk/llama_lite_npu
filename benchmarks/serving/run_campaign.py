"""Plan canonical serving benchmark campaigns through one stable entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adapters import get_adapter
from .evalscope_client import build_evalscope_command
from .schema import CANONICAL_ENDPOINT_PATH, REPO_ROOT, CampaignSpec, load_campaign, load_model
from .workload.fingerprint import fingerprint_file


def _load_workload_contract(workload: dict[str, Any]) -> dict[str, Any]:
    manifest_path = REPO_ROOT / str(workload["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    formal_path = REPO_ROOT / str(workload["formal_dataset"])
    warmup_path = REPO_ROOT / str(workload["warmup_dataset"])
    actual_formal_sha = fingerprint_file(formal_path)
    actual_warmup_sha = fingerprint_file(warmup_path)
    formal_lines = sum(1 for line in formal_path.read_text(encoding="utf-8").splitlines() if line.strip())
    warmup_lines = sum(1 for line in warmup_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if int(manifest.get("schema_version", -1)) != 2:
        raise ValueError("workload manifest schema_version must be 2")
    if manifest.get("workload_id") != workload["id"]:
        raise ValueError("workload_id does not match workload.json")
    if int(manifest.get("seed", -1)) != int(workload["seed"]):
        raise ValueError("workload seed does not match workload.json")
    if manifest.get("request_order") != workload["request_order"]:
        raise ValueError("workload request_order does not match workload.json")
    for field in ("formal_offset", "warmup_offset"):
        if int(manifest.get(field, -1)) != int(workload[field]):
            raise ValueError(f"workload {field} does not match workload.json")
    if actual_formal_sha != str(manifest["formal_sha256"]).lower():
        raise ValueError("formal workload SHA256 does not match workload.json")
    if actual_warmup_sha != str(manifest["warmup_sha256"]).lower():
        raise ValueError("warmup workload SHA256 does not match workload.json")
    if int(manifest["formal_requests"]) != formal_lines:
        raise ValueError("formal workload line count does not match workload.json")
    if int(manifest["warmup_requests"]) != warmup_lines:
        raise ValueError("warmup workload line count does not match workload.json")
    return {
        "kind": "line_by_line",
        "workload_id": manifest["workload_id"],
        "formal": str(workload["formal_dataset"]),
        "warmup": str(workload["warmup_dataset"]),
        "formal_sha256": actual_formal_sha,
        "warmup_sha256": actual_warmup_sha,
        "formal_available_requests": formal_lines,
        "warmup_available_requests": warmup_lines,
        "actual_token_calibrated": bool(manifest["actual_token_calibrated"]),
        "strict_publishable": bool(manifest["strict_publishable"]),
    }


def build_campaign_plan(framework: str, campaign: str | Path) -> dict[str, Any]:
    spec: CampaignSpec = load_campaign(campaign)
    if framework not in spec.frameworks:
        raise ValueError(f"framework {framework} is not enabled by campaign {spec.campaign_id}")
    model = load_model(spec.model)
    adapter = get_adapter(framework)
    workload = spec.workload
    dataset_contract = _load_workload_contract(dict(workload))
    contract = {
        "api": "openai",
        "endpoint_path": CANONICAL_ENDPOINT_PATH,
        "served_model": model["served_name"],
        "tokenizer": model["tokenizer"],
        "dataset": dataset_contract,
        "seed": int(workload["seed"]),
        "formal_offset": int(workload["formal_offset"]),
        "warmup_offset": int(workload["warmup_offset"]),
        "request_order": str(workload["request_order"]),
        "stream": spec.stream,
        "sampling": dict(spec.sampling),
        "fixed_output": dict(spec.fixed_output),
        "total_timeout_s": spec.total_timeout_s,
    }
    result_root = Path(spec.result_root) / spec.campaign_id
    runs: list[dict[str, Any]] = []
    for case in spec.cases:
        if case.formal_requests + contract["formal_offset"] > dataset_contract["formal_available_requests"]:
            raise ValueError(f"case {case.case_id} formal_requests exceeds frozen dataset")
        if case.warmup_requests + contract["warmup_offset"] > dataset_contract["warmup_available_requests"]:
            raise ValueError(f"case {case.case_id} warmup_requests exceeds frozen dataset")
        for repeat in range(1, spec.repeats + 1):
            run_id = f"{spec.campaign_id}_{case.case_id}_r{repeat}"
            relative = Path("runs") / framework / case.case_id / f"repeat-{repeat:02d}"
            run = {
                "run_id": run_id,
                "case_id": case.case_id,
                "repeat": repeat,
                "prompt_tokens": case.prompt_tokens,
                "output_tokens": case.output_tokens,
                "concurrency": case.concurrency,
                "formal_requests": case.formal_requests,
                "warmup_requests": case.warmup_requests,
                "lifecycle": spec.lifecycle,
                "result_path": (result_root / relative).as_posix(),
            }
            run["warmup_command"] = build_evalscope_command(
                contract,
                run,
                base_url=adapter.base_url,
                output_dir=(result_root / relative / "client/warmup/evalscope").as_posix(),
                phase="warmup",
            )
            run["formal_command"] = build_evalscope_command(
                contract,
                run,
                base_url=adapter.base_url,
                output_dir=(result_root / relative / "client/evalscope").as_posix(),
            )
            runs.append(run)
    return {
        "schema_version": 2,
        "mode": "dry-run",
        "campaign_id": spec.campaign_id,
        "kind": spec.kind,
        "framework": framework,
        "model": model,
        "client_contract": contract,
        "graph_contract": dict(spec.graph),
        "failure_gates": dict(spec.failure_gates),
        "result_root": result_root.as_posix(),
        "adapter": adapter.launch_spec(model),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", required=True, choices=("lite_llama", "vllm_ascend"))
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("Phase 2 supports planning only; pass --dry-run")
    plan = build_campaign_plan(args.framework, args.campaign)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
