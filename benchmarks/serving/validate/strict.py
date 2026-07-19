"""Strict correctness gates for one serving benchmark run."""

from __future__ import annotations

from typing import Any, Mapping


GRAPH_STATUSES = {"pass", "fail", "unsupported"}


def validate_strict_run(
    expected: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    semantic_accuracy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hard_failures: list[str] = []
    invalid_reasons: list[str] = []

    if not expected.get("fixed_output_capability_probed"):
        invalid_reasons.append("fixed_output_capability_not_probed")
    if not expected.get("actual_token_calibrated"):
        invalid_reasons.append("actual_token_length_not_calibrated")

    requests = evidence.get("requests", [])
    expected_ids = list(expected["request_ids"])
    actual_ids = [item.get("request_id") for item in requests]
    if actual_ids != expected_ids:
        hard_failures.append("request_count_or_order")
    for request in requests:
        request_id = request.get("request_id", "unknown")
        if not request.get("success", False):
            hard_failures.append(f"failed_request:{request_id}")
        if request.get("input_tokens") != expected["input_tokens"]:
            hard_failures.append(f"input_tokens:{request_id}")
        if request.get("output_tokens") != expected["output_tokens"]:
            hard_failures.append(f"output_tokens:{request_id}")

    for field in ("workload_fingerprint", "environment_fingerprint"):
        if evidence.get(field) != expected.get(field):
            hard_failures.append(field)

    graph = dict(evidence.get("graph", {}))
    graph_status = graph.get("status")
    if graph_status not in GRAPH_STATUSES:
        invalid_reasons.append("graph_status_missing_or_unknown")
        graph = {"status": "unsupported"}
    graph_expected = expected.get("graph", {})
    if graph.get("status") == "unsupported" and (graph_expected.get("required") or graph_expected.get("causal_claim")):
        invalid_reasons.append("graph_evidence_unsupported")
    elif graph.get("status") == "fail":
        hard_failures.append("graph_status")
    elif graph.get("status") == "pass":
        if graph.get("fallback_delta") != 0:
            hard_failures.append("graph_fallback")
        if graph_expected.get("required") or graph_expected.get("causal_claim"):
            if graph.get("replay_delta", 0) <= 0:
                hard_failures.append("graph_replay")
            if graph.get("capture_delta") != 0:
                hard_failures.append("graph_capture_during_formal")

    if hard_failures:
        status = "fail"
    elif invalid_reasons:
        status = "invalid"
    else:
        status = "pass"
    return {
        "schema_version": 2,
        "status": status,
        "performance_correctness": {"status": status, "hard_failures": hard_failures, "invalid_reasons": invalid_reasons},
        "hard_failures": hard_failures,
        "invalid_reasons": invalid_reasons,
        "graph": graph,
        "semantic_accuracy": dict(semantic_accuracy) if semantic_accuracy is not None else None,
    }
