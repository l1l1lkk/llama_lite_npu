"""Cross-run and cross-framework campaign plan validation."""

from __future__ import annotations

from typing import Any, Mapping


CLIENT_CONTRACT_FIELDS = (
    "api",
    "endpoint_path",
    "served_model",
    "tokenizer",
    "dataset",
    "seed",
    "formal_offset",
    "warmup_offset",
    "request_order",
    "stream",
    "sampling",
    "fixed_output",
    "total_timeout_s",
)

MODEL_IDENTITY_FIELDS = (
    "model_id",
    "checkpoint",
    "config_sha256",
    "tokenizer",
    "dtype",
    "tensor_parallel_size",
)


def comparable_client_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    contract = plan["client_contract"]
    return {field: contract.get(field) for field in CLIENT_CONTRACT_FIELDS}


def validate_paired_plans(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    left_contract = comparable_client_contract(left)
    right_contract = comparable_client_contract(right)
    for field in CLIENT_CONTRACT_FIELDS:
        if left_contract[field] != right_contract[field]:
            errors.append(f"client_contract.{field}")
    for field in MODEL_IDENTITY_FIELDS:
        if left["model"].get(field) != right["model"].get(field):
            errors.append(f"model.{field}")
    left_runs = [{key: run[key] for key in run if key not in {"formal_command", "warmup_command", "result_path"}} for run in left["runs"]]
    right_runs = [{key: run[key] for key in run if key not in {"formal_command", "warmup_command", "result_path"}} for run in right["runs"]]
    if left_runs != right_runs:
        errors.append("run_matrix_mismatch")
    return {"schema_version": 2, "status": "pass" if not errors else "fail", "errors": errors}
