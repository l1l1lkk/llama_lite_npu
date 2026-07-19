from benchmarks.serving.validate.strict import validate_strict_run


def _expected(**updates):
    value = {
        "request_ids": ["r0", "r1"],
        "input_tokens": 128,
        "output_tokens": 64,
        "workload_fingerprint": "workload-sha",
        "environment_fingerprint": "environment-sha",
        "fixed_output_capability_probed": True,
        "actual_token_calibrated": True,
        "graph": {"required": True, "causal_claim": False},
    }
    value.update(updates)
    return value


def _evidence(**updates):
    value = {
        "requests": [
            {"request_id": "r0", "success": True, "input_tokens": 128, "output_tokens": 64},
            {"request_id": "r1", "success": True, "input_tokens": 128, "output_tokens": 64},
        ],
        "workload_fingerprint": "workload-sha",
        "environment_fingerprint": "environment-sha",
        "graph": {"status": "pass", "capture_delta": 0, "fallback_delta": 0, "replay_delta": 2},
    }
    value.update(updates)
    return value


def test_strict_pass_keeps_accuracy_separate():
    report = validate_strict_run(_expected(), _evidence(), semantic_accuracy={"metric": "exact_match", "value": 0.5})

    assert report["status"] == "pass"
    assert report["performance_correctness"]["status"] == "pass"
    assert report["semantic_accuracy"] == {"metric": "exact_match", "value": 0.5}


def test_token_and_fingerprint_mismatch_are_hard_failures():
    evidence = _evidence(workload_fingerprint="wrong")
    evidence["requests"][1]["output_tokens"] = 63
    report = validate_strict_run(_expected(), evidence)

    assert report["status"] == "fail"
    assert "workload_fingerprint" in report["hard_failures"]
    assert "output_tokens:r1" in report["hard_failures"]


def test_graph_unsupported_is_invalid_for_causal_campaign():
    expected = _expected(graph={"required": True, "causal_claim": True})
    evidence = _evidence(graph={"status": "unsupported"})
    report = validate_strict_run(expected, evidence)

    assert report["status"] == "invalid"
    assert report["graph"]["status"] == "unsupported"


def test_environment_mismatch_and_graph_fallback_are_hard_failures():
    evidence = _evidence(environment_fingerprint="other")
    evidence["graph"]["fallback_delta"] = 1
    evidence["graph"]["replay_delta"] = 0
    report = validate_strict_run(_expected(), evidence)

    assert report["status"] == "fail"
    assert "environment_fingerprint" in report["hard_failures"]
    assert "graph_fallback" in report["hard_failures"]
    assert "graph_replay" in report["hard_failures"]


def test_optional_production_graph_fallback_is_still_a_hard_failure():
    expected = _expected(graph={"required": False, "causal_claim": False})
    evidence = _evidence(
        graph={"status": "pass", "capture_delta": 0, "fallback_delta": 3, "replay_delta": 0}
    )

    report = validate_strict_run(expected, evidence)

    assert report["status"] == "fail"
    assert "graph_fallback" in report["hard_failures"]


def test_optional_unsupported_graph_remains_unsupported_without_fake_zeroes():
    expected = _expected(graph={"required": False, "causal_claim": False})
    report = validate_strict_run(expected, _evidence(graph={"status": "unsupported"}))

    assert report["status"] == "pass"
    assert report["graph"] == {"status": "unsupported"}


def test_unprobed_fixed_output_cannot_pass():
    report = validate_strict_run(_expected(fixed_output_capability_probed=False), _evidence())
    assert report["status"] == "invalid"
