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
    "config_sha256",
    "tokenizer_sha256",
    "tokenizer_config_sha256",
    "model_index_sha256",
    "dtype",
    "tensor_parallel_size",
    "max_sequence_length",
)


def comparable_client_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    contract = plan["client_contract"]
    return {field: contract.get(field) for field in CLIENT_CONTRACT_FIELDS}


def validate_paired_plans(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    eligibility_errors: list[str] = []
    left_contract = comparable_client_contract(left)
    right_contract = comparable_client_contract(right)
    for field in CLIENT_CONTRACT_FIELDS:
        if left_contract[field] != right_contract[field]:
            errors.append(f"client_contract.{field}")
    environment_keys = sorted(set(left.get("client_environment", {})) | set(right.get("client_environment", {})))
    for field in environment_keys:
        if left.get("client_environment", {}).get(field) != right.get("client_environment", {}).get(field):
            errors.append(f"client_environment.{field}")
    for field in MODEL_IDENTITY_FIELDS:
        if left["model"]["logical_identity"].get(field) != right["model"]["logical_identity"].get(field):
            errors.append(f"model.logical_identity.{field}")
    if left.get("comparison_scope") != right.get("comparison_scope"):
        errors.append("comparison_scope")
    left_runs = [{key: run[key] for key in run if key not in {"formal_command", "warmup_command", "result_path"}} for run in left["runs"]]
    right_runs = [{key: run[key] for key in run if key not in {"formal_command", "warmup_command", "result_path"}} for run in right["runs"]]
    if left_runs != right_runs:
        errors.append("run_matrix_mismatch")
    scope = left.get("comparison_scope")
    equivalence = left["model"].get("equivalence", {})
    if equivalence != right["model"].get("equivalence", {}):
        errors.append("model.equivalence")
    if equivalence.get("status") != "verified" or not equivalence.get("valid_for_causal"):
        eligibility_errors.append("checkpoint_equivalence_unverified")
    if scope == "controlled_stack":
        stack_fields = ("cann", "torch", "torch_npu")
        for field in stack_fields:
            if left.get("environment_contract", {}).get(field) != right.get("environment_contract", {}).get(field):
                eligibility_errors.append(f"environment_contract.{field}")
        if left.get("environment_contract", {}).get("fingerprint_status") != "verified":
            eligibility_errors.append("environment_contract.left_fingerprint_unverified")
        if right.get("environment_contract", {}).get("fingerprint_status") != "verified":
            eligibility_errors.append("environment_contract.right_fingerprint_unverified")
    elif scope == "production_stack":
        if left.get("environment_contract", {}).get("fingerprint_status") != "verified":
            eligibility_errors.append("environment_contract.left_fingerprint_unverified")
        if right.get("environment_contract", {}).get("fingerprint_status") != "verified":
            eligibility_errors.append("environment_contract.right_fingerprint_unverified")
    if scope == "capability_only":
        eligibility_errors.append("capability_only")
    if not left.get("performance_eligible", False) or not right.get("performance_eligible", False):
        eligibility_errors.append("plan_not_performance_eligible")
    return {
        "schema_version": 2,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "client_workload_contract_match": not any(item.startswith("client_") or item == "run_matrix_mismatch" for item in errors),
        "logical_identity_match": not any(item.startswith("model.logical_identity") for item in errors),
        "representation_equivalence": equivalence.get("status", "unverified"),
        "comparison_eligible": not errors and not eligibility_errors,
        "eligibility_errors": sorted(set(eligibility_errors)),
    }
