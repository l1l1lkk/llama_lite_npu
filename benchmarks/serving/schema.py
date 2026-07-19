"""Typed configuration schema for canonical serving benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ENDPOINT_PATH = "/v1/chat/completions"
CAMPAIGN_KINDS = {"capability", "performance", "accuracy"}
COMPARISON_SCOPES = {"capability_only", "production_stack", "controlled_stack"}


def validate_base_url(value: str) -> dict[str, Any]:
    """Validate and normalize a runtime server origin without an API path."""
    try:
        parsed = urlsplit(str(value))
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid base URL: {value}") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("base URL scheme must be http or https")
    if not parsed.hostname or port is None or not (1 <= port <= 65535):
        raise ValueError("base URL must contain a host and explicit valid port")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base URL must not contain credentials")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain path, query, or fragment")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return {
        "base_url": f"{parsed.scheme}://{host}:{port}",
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "port": port,
    }


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
    purpose: str
    comparison_scope: str
    publishable: bool
    aggregation_allowed: bool
    result_namespace: str
    client_profile_required: bool
    client_environment: Mapping[str, str]
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
            "purpose",
            "comparison_scope",
            "publishable",
            "aggregation_allowed",
            "result_namespace",
            "client_profile_required",
            "client_environment",
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
        kind = str(value["kind"])
        comparison_scope = str(value["comparison_scope"])
        if kind not in CAMPAIGN_KINDS:
            raise ValueError(f"unsupported campaign kind: {kind}")
        if comparison_scope not in COMPARISON_SCOPES:
            raise ValueError(f"unsupported comparison_scope: {comparison_scope}")
        if not isinstance(value["client_profile_required"], bool):
            raise ValueError("client_profile_required must be boolean")
        if kind == "performance" and "semantic_accuracy" in value:
            raise ValueError("semantic accuracy must be reported in a separate campaign")
        if kind == "capability":
            if comparison_scope != "capability_only":
                raise ValueError("capability campaigns require comparison_scope=capability_only")
            if bool(value["publishable"]) or bool(value["aggregation_allowed"]):
                raise ValueError("capability campaigns cannot be publishable or aggregatable")
            if str(value["result_namespace"]) != "diagnostics":
                raise ValueError("capability campaigns require result_namespace=diagnostics")
            if not bool(value["client_profile_required"]):
                raise ValueError("capability campaigns require client_profile_required=true")
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
            kind=kind,
            purpose=str(value["purpose"]),
            comparison_scope=comparison_scope,
            publishable=bool(value["publishable"]),
            aggregation_allowed=bool(value["aggregation_allowed"]),
            result_namespace=str(value["result_namespace"]),
            client_profile_required=bool(value["client_profile_required"]),
            client_environment={str(key): str(item) for key, item in value["client_environment"].items()},
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
    _require(
        model,
        {"schema_version", "served_name", "logical_identity", "representations", "equivalence"},
        "model",
    )
    if int(model["schema_version"]) != 2:
        raise ValueError("only model schema_version=2 is supported")
    _require(
        model["logical_identity"],
        {
            "model_id",
            "config_sha256",
            "tokenizer_sha256",
            "tokenizer_path",
            "dtype",
            "tensor_parallel_size",
            "max_sequence_length",
        },
        "model.logical_identity",
    )
    for framework in ("lite_llama", "vllm_ascend"):
        if framework not in model["representations"]:
            raise ValueError(f"model representation is missing {framework}")
        _require(
            model["representations"][framework],
            {"path", "format", "fingerprint", "provenance"},
            f"model.representations.{framework}",
        )
    _require(model["equivalence"], {"status", "valid_for_causal"}, "model.equivalence")
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
    validate_base_url(str(framework["base_url"]))
    _require(framework, {"runtime_environment"}, "framework")
    return framework
