"""Typed configuration schema for canonical serving benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ENDPOINT_PATH = "/v1/chat/completions"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration does not exist: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _require(mapping: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - mapping.keys())
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    prompt_tokens: int
    output_tokens: int
    concurrency: int
    formal_requests: int
    warmup_requests: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CaseSpec":
        _require(
            value,
            {"id", "prompt_tokens", "output_tokens", "concurrency", "formal_requests", "warmup_requests"},
            "case",
        )
        case = cls(
            case_id=str(value["id"]),
            prompt_tokens=int(value["prompt_tokens"]),
            output_tokens=int(value["output_tokens"]),
            concurrency=int(value["concurrency"]),
            formal_requests=int(value["formal_requests"]),
            warmup_requests=int(value["warmup_requests"]),
        )
        if min(case.prompt_tokens, case.output_tokens, case.concurrency, case.formal_requests) <= 0:
            raise ValueError(f"case {case.case_id} has a non-positive formal field")
        if case.warmup_requests < 0 or case.formal_requests < case.concurrency:
            raise ValueError(f"case {case.case_id} has invalid request counts")
        return case


@dataclass(frozen=True)
class CampaignSpec:
    schema_version: int
    campaign_id: str
    kind: str
    model: str
    frameworks: tuple[str, ...]
    workload: Mapping[str, Any]
    cases: tuple[CaseSpec, ...]
    repeats: int
    lifecycle: str
    stream: bool
    sampling: Mapping[str, float]
    fixed_output: Mapping[str, Any]
    graph: Mapping[str, Any]
    failure_gates: Mapping[str, Any]
    result_root: str
    total_timeout_s: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CampaignSpec":
        required = {
            "schema_version",
            "campaign_id",
            "kind",
            "model",
            "frameworks",
            "workload",
            "cases",
            "repeats",
            "lifecycle",
            "stream",
            "sampling",
            "fixed_output",
            "graph",
            "failure_gates",
            "result_root",
        }
        _require(value, required, "campaign")
        if int(value["schema_version"]) != 2:
            raise ValueError("only campaign schema_version=2 is supported")
        if value["kind"] == "performance" and "semantic_accuracy" in value:
            raise ValueError("semantic accuracy must be reported in a separate campaign")
        frameworks = tuple(str(item) for item in value["frameworks"])
        if not frameworks or len(set(frameworks)) != len(frameworks):
            raise ValueError("frameworks must be a non-empty unique list")
        cases = tuple(CaseSpec.from_mapping(item) for item in value["cases"])
        if not cases or len({item.case_id for item in cases}) != len(cases):
            raise ValueError("cases must be a non-empty list with unique ids")
        repeats = int(value["repeats"])
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        workload = value["workload"]
        _require(
            workload,
            {
                "id",
                "formal_dataset",
                "warmup_dataset",
                "manifest",
                "seed",
                "formal_offset",
                "warmup_offset",
                "request_order",
            },
            "workload",
        )
        fixed_output = value["fixed_output"]
        _require(fixed_output, {"strategy", "capability_probe_required"}, "fixed_output")
        graph = value["graph"]
        _require(graph, {"mode", "causal_claim", "required"}, "graph")
        return cls(
            schema_version=2,
            campaign_id=str(value["campaign_id"]),
            kind=str(value["kind"]),
            model=str(value["model"]),
            frameworks=frameworks,
            workload=workload,
            cases=cases,
            repeats=repeats,
            lifecycle=str(value["lifecycle"]),
            stream=bool(value["stream"]),
            sampling={"temperature": float(value["sampling"]["temperature"]), "top_p": float(value["sampling"]["top_p"])},
            fixed_output=fixed_output,
            graph=graph,
            failure_gates=value["failure_gates"],
            result_root=str(value["result_root"]),
            total_timeout_s=int(value.get("total_timeout_s", 21600)),
        )


def _resolve_config(value: str | Path, category: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    name = path.name if path.suffix else f"{path.name}.yaml"
    candidate = REPO_ROOT / "benchmarks" / "configs" / category / name
    if candidate.exists():
        return candidate
    raise ValueError(f"unknown {category.rstrip('s')} configuration: {value}")


def load_campaign(value: str | Path) -> CampaignSpec:
    return CampaignSpec.from_mapping(_read_yaml(_resolve_config(value, "campaigns")))


def load_model(value: str | Path) -> dict[str, Any]:
    model = _read_yaml(_resolve_config(value, "models"))
    _require(model, {"schema_version", "model_id", "served_name", "checkpoint", "tokenizer", "dtype", "tensor_parallel_size"}, "model")
    if int(model["schema_version"]) != 2:
        raise ValueError("only model schema_version=2 is supported")
    return model


def load_framework(value: str | Path) -> dict[str, Any]:
    framework = _read_yaml(_resolve_config(value, "frameworks"))
    _require(framework, {"schema_version", "framework", "base_url", "endpoint_path", "health_path", "launch", "metrics"}, "framework")
    if int(framework["schema_version"]) != 2:
        raise ValueError("only framework schema_version=2 is supported")
    if framework["endpoint_path"] != CANONICAL_ENDPOINT_PATH:
        raise ValueError(
            f"framework endpoint_path must match canonical endpoint {CANONICAL_ENDPOINT_PATH}"
        )
    return framework
