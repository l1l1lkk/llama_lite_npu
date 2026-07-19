"""Plan canonical serving benchmark campaigns through one stable entry point."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from .adapters import get_adapter
from .evalscope_client import build_evalscope_command
from .schema import (
    CANONICAL_ENDPOINT_PATH,
    REPO_ROOT,
    CampaignSpec,
    load_campaign,
    load_model,
    validate_base_url,
)
from .workload.fingerprint import fingerprint_file


_RUNTIME_PLACEHOLDER = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
_RUNTIME_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SAFE_RUNTIME_PATH = re.compile(r"^/[A-Za-z0-9._/+:=-]+$")


def _declared_runtime_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, Mapping):
        for item in value.values():
            names.update(_declared_runtime_names(item))
    elif isinstance(value, list):
        for item in value:
            names.update(_declared_runtime_names(item))
    elif isinstance(value, str):
        match = _RUNTIME_PLACEHOLDER.fullmatch(value)
        if match:
            names.add(match.group(1))
        elif "${" in value:
            raise ValueError("runtime value placeholders must occupy the complete field")
    return names


def _validate_runtime_path(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"runtime value {name} must not be empty")
    if not _SAFE_RUNTIME_PATH.fullmatch(value):
        raise ValueError(f"runtime value {name} contains unsafe characters")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or value.startswith("//")
        or path.as_posix() != value
        or ".." in path.parts
    ):
        raise ValueError(f"runtime value {name} must be a normalized Linux absolute path")
    return value


def parse_runtime_value_args(
    arguments: list[str] | tuple[str, ...], *, declared_names: set[str]
) -> dict[str, str]:
    """Parse repeatable NAME=VALUE bindings without shell expansion."""
    values: dict[str, str] = {}
    for argument in arguments:
        if "=" not in argument:
            raise ValueError("runtime value must use NAME=VALUE")
        name, value = argument.split("=", 1)
        if not _RUNTIME_NAME.fullmatch(name):
            raise ValueError(f"runtime value has invalid name: {name}")
        if name not in declared_names:
            raise ValueError(f"runtime value is not declared by the model: {name}")
        if name in values:
            raise ValueError(f"runtime value is duplicated: {name}")
        values[name] = _validate_runtime_path(name, value)
    return values


def _resolve_model_runtime_values(
    model: dict[str, Any],
    framework: str,
    runtime_values: Mapping[str, str] | None,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    declared_names = _declared_runtime_names(model)
    for name, value in (runtime_values or {}).items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ValueError("runtime value names and paths must be strings")
    supplied = parse_runtime_value_args(
        [f"{name}={value}" for name, value in (runtime_values or {}).items()],
        declared_names=declared_names,
    )
    resolved = deepcopy(model)
    fields = (
        resolved["logical_identity"],
        resolved["representations"][framework],
    )
    required_names: set[str] = set()
    used: dict[str, str] = {}
    for mapping in fields:
        field_name = "tokenizer_path" if mapping is resolved["logical_identity"] else "path"
        raw = mapping[field_name]
        match = _RUNTIME_PLACEHOLDER.fullmatch(str(raw))
        if not match:
            continue
        name = match.group(1)
        required_names.add(name)
        if name in supplied:
            mapping[field_name] = supplied[name]
            used[name] = supplied[name]
    unresolved = sorted(required_names - used.keys())
    return resolved, dict(sorted(used.items())), unresolved


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


def _model_plan(model: dict[str, Any], framework: str) -> dict[str, Any]:
    return {
        "served_name": model["served_name"],
        "logical_identity": dict(model["logical_identity"]),
        "representation": dict(model["representations"][framework]),
        "equivalence": dict(model["equivalence"]),
    }


def _eligibility(
    spec: CampaignSpec,
    dataset: dict[str, Any],
    model: dict[str, Any],
    runtime_environment: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if spec.kind != "performance":
        reasons.append("campaign_kind_not_performance")
    if not spec.publishable:
        reasons.append("campaign_not_publishable")
    if not spec.aggregation_allowed:
        reasons.append("aggregation_not_allowed")
    if not dataset["actual_token_calibrated"]:
        reasons.append("workload_not_token_calibrated")
    if not dataset["strict_publishable"]:
        reasons.append("workload_not_strict_publishable")
    if spec.fixed_output.get("capability_probe_required") and not spec.fixed_output.get("capability_probed", False):
        reasons.append("fixed_output_not_probed")
    if model["equivalence"].get("status") != "verified" or not model["equivalence"].get("valid_for_causal"):
        reasons.append("checkpoint_equivalence_unverified")
    if spec.kind == "performance" and runtime_environment.get("fingerprint_status") != "verified":
        reasons.append("environment_fingerprint_unverified")
    return not reasons, reasons


def build_campaign_plan(
    framework: str,
    campaign: str | Path,
    *,
    base_url_override: str | None = None,
    output_root: str | Path | None = None,
    runtime_values: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    spec: CampaignSpec = load_campaign(campaign)
    if framework not in spec.frameworks:
        raise ValueError(f"framework {framework} is not enabled by campaign {spec.campaign_id}")
    model, resolved_runtime_values, unresolved_runtime_values = _resolve_model_runtime_values(
        load_model(spec.model), framework, runtime_values
    )
    adapter = get_adapter(framework)
    runtime = validate_base_url(base_url_override or adapter.base_url)
    workload = spec.workload
    dataset_contract = _load_workload_contract(dict(workload))
    contract = {
        "api": "openai",
        "endpoint_path": CANONICAL_ENDPOINT_PATH,
        "served_model": model["served_name"],
        "tokenizer": model["logical_identity"]["tokenizer_path"],
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
    result_root = (
        Path(output_root)
        if output_root is not None
        else Path(spec.result_root) / spec.campaign_id / framework
    )
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
                base_url=runtime["base_url"],
                output_dir=(result_root / relative / "client/warmup/evalscope").as_posix(),
                phase="warmup",
            )
            run["formal_command"] = build_evalscope_command(
                contract,
                run,
                base_url=runtime["base_url"],
                output_dir=(result_root / relative / "client/evalscope").as_posix(),
            )
            runs.append(run)
    performance_eligible, eligibility_reasons = _eligibility(
        spec,
        dataset_contract,
        model,
        dict(adapter.config["runtime_environment"]),
    )
    return {
        "schema_version": 2,
        "mode": "dry-run",
        "campaign_id": spec.campaign_id,
        "kind": spec.kind,
        "purpose": spec.purpose,
        "comparison_scope": spec.comparison_scope,
        "evidence_class": "diagnostic" if spec.kind == "capability" else "candidate",
        "publishable": spec.publishable,
        "aggregation_allowed": spec.aggregation_allowed,
        "performance_eligible": performance_eligible,
        "performance_eligibility_reasons": eligibility_reasons,
        "result_namespace": spec.result_namespace,
        "runtime_bindings": resolved_runtime_values,
        "unresolved_runtime_values": unresolved_runtime_values,
        "capability_execution_ready": not unresolved_runtime_values,
        "framework": framework,
        "model": _model_plan(model, framework),
        "client_contract": contract,
        "client_environment": dict(spec.client_environment),
        "runtime": runtime,
        "environment_contract": dict(adapter.config["runtime_environment"]),
        "graph_contract": dict(spec.graph),
        "failure_gates": dict(spec.failure_gates),
        "result_root": result_root.as_posix(),
        "adapter": adapter.launch_spec(model, base_url=runtime["base_url"]),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", required=True, choices=("lite_llama", "vllm_ascend"))
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-url", help="Override the framework server origin for this plan")
    parser.add_argument("--output-root", help="Override the exact diagnostic/result root without creating it")
    parser.add_argument(
        "--runtime-value",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Bind one model-declared runtime path; repeat for each required value",
    )
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("the canonical runner currently supports planning only; pass --dry-run")
    try:
        spec = load_campaign(args.campaign)
        model = load_model(spec.model)
        runtime_values = parse_runtime_value_args(
            args.runtime_value,
            declared_names=_declared_runtime_names(model),
        )
        plan = build_campaign_plan(
            args.framework,
            args.campaign,
            base_url_override=args.base_url,
            output_root=args.output_root,
            runtime_values=runtime_values,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
