import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.serving.run_campaign import (
    build_campaign_plan,
    parse_runtime_value_args,
)
from benchmarks.serving.schema import validate_base_url
from benchmarks.serving.validate.campaign import validate_paired_plans


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "benchmarks/configs/campaigns/capability_smoke.yaml"


def _argument(command, name):
    return command[command.index(name) + 1]


def test_runtime_base_url_resolves_client_health_and_launch_port():
    lite = build_campaign_plan("lite_llama", CAMPAIGN)
    vllm = build_campaign_plan(
        "vllm_ascend", CAMPAIGN, base_url_override="http://127.0.0.1:18000"
    )

    assert lite["runtime"]["base_url"] == "http://127.0.0.1:8213"
    assert lite["adapter"]["health_url"] == "http://127.0.0.1:8213/health"
    assert _argument(lite["adapter"]["launch_command"], "--port") == "8213"
    assert _argument(lite["runs"][0]["formal_command"], "--url").endswith(
        ":8213/v1/chat/completions"
    )
    assert vllm["runtime"]["base_url"] == "http://127.0.0.1:18000"
    assert vllm["adapter"]["health_url"] == "http://127.0.0.1:18000/health"
    assert _argument(vllm["adapter"]["launch_command"], "--port") == "18000"
    assert _argument(vllm["runs"][0]["formal_command"], "--url").endswith(
        ":18000/v1/chat/completions"
    )


@pytest.mark.parametrize(
    "value",
    (
        "ftp://127.0.0.1:18000",
        "http://127.0.0.1",
        "http://user:pass@127.0.0.1:18000",
        "http://127.0.0.1:18000/path",
        "http://127.0.0.1:18000?x=1",
        "http://127.0.0.1:18000#fragment",
        "http://127.0.0.1:99999",
    ),
)
def test_runtime_base_url_rejects_ambiguous_or_unsafe_values(value):
    with pytest.raises(ValueError, match="base URL"):
        validate_base_url(value)


def test_client_environment_is_common_and_field_level_comparable():
    lite = build_campaign_plan("lite_llama", CAMPAIGN)
    vllm = build_campaign_plan("vllm_ascend", CAMPAIGN)

    assert lite["client_environment"] == {
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"
    }
    assert lite["client_environment"] == vllm["client_environment"]
    drifted = deepcopy(vllm)
    drifted["client_environment"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    report = validate_paired_plans(lite, drifted)
    assert report["status"] == "fail"
    assert "client_environment.TORCH_DEVICE_BACKEND_AUTOLOAD" in report["errors"]


def test_capability_campaign_is_diagnostic_only_and_unpublishable(tmp_path):
    root = tmp_path / "must-not-exist"
    plan = build_campaign_plan("lite_llama", CAMPAIGN, output_root=root)

    assert plan["kind"] == "capability"
    assert plan["comparison_scope"] == "capability_only"
    assert plan["evidence_class"] == "diagnostic"
    assert plan["publishable"] is False
    assert plan["aggregation_allowed"] is False
    assert plan["performance_eligible"] is False
    assert plan["result_namespace"] == "diagnostics"
    assert plan["result_root"].startswith(root.as_posix())
    assert not root.exists()


def test_framework_specific_checkpoint_representations_remain_diagnostic():
    lite = build_campaign_plan("lite_llama", CAMPAIGN)
    vllm = build_campaign_plan("vllm_ascend", CAMPAIGN)

    assert lite["model"]["representation"]["format"] == "custom_pth"
    assert vllm["model"]["representation"]["format"] == "hf_safetensors"
    assert lite["model"]["representation"]["path"] != vllm["model"]["representation"]["path"]
    assert lite["model"]["equivalence"]["status"] == "unverified"
    assert lite["performance_eligible"] is False
    assert validate_paired_plans(lite, vllm)["status"] == "pass"


@pytest.mark.parametrize("field", ("config_sha256", "tokenizer_sha256"))
def test_logical_model_identity_drift_is_a_paired_failure(field):
    lite = build_campaign_plan("lite_llama", CAMPAIGN)
    vllm = deepcopy(build_campaign_plan("vllm_ascend", CAMPAIGN))
    vllm["model"]["logical_identity"][field] = "different"

    report = validate_paired_plans(lite, vllm)
    assert report["status"] == "fail"
    assert f"model.logical_identity.{field}" in report["errors"]


def test_controlled_stack_requires_shared_environment_fingerprint():
    lite = build_campaign_plan("lite_llama", CAMPAIGN)
    vllm = build_campaign_plan("vllm_ascend", CAMPAIGN)
    lite = deepcopy(lite)
    vllm = deepcopy(vllm)
    lite["comparison_scope"] = "controlled_stack"
    vllm["comparison_scope"] = "controlled_stack"
    lite["environment_contract"] = {"cann": "8.5", "torch": "2.7.1", "torch_npu": "2.7.1"}
    vllm["environment_contract"] = {"cann": "9.0", "torch": "2.10", "torch_npu": "2.10"}

    report = validate_paired_plans(lite, vllm)
    assert report["comparison_eligible"] is False
    assert "environment_contract.cann" in report["eligibility_errors"]


def test_production_stack_allows_native_versions_but_requires_fingerprints():
    lite = deepcopy(build_campaign_plan("lite_llama", CAMPAIGN))
    vllm = deepcopy(build_campaign_plan("vllm_ascend", CAMPAIGN))
    for plan in (lite, vllm):
        plan["comparison_scope"] = "production_stack"
        plan["performance_eligible"] = True
        plan["model"]["equivalence"] = {
            "status": "verified",
            "valid_for_causal": True,
        }

    report = validate_paired_plans(lite, vllm)
    assert report["status"] == "pass"
    assert report["comparison_eligible"] is False
    assert "environment_contract.left_fingerprint_unverified" in report["eligibility_errors"]
    assert "environment_contract.right_fingerprint_unverified" in report["eligibility_errors"]


def test_cli_output_root_and_override_are_plan_only(tmp_path):
    root = tmp_path / "diagnostics"
    command = [
        sys.executable,
        "-m",
        "benchmarks.serving.run_campaign",
        "--framework",
        "vllm_ascend",
        "--campaign",
        str(CAMPAIGN),
        "--base-url",
        "http://127.0.0.1:18000",
        "--output-root",
        str(root),
        "--dry-run",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["runtime"]["port"] == 18000
    assert plan["performance_eligible"] is False
    assert not root.exists()


def test_retired_p0_smoke_alias_fails_explicitly():
    with pytest.raises(ValueError, match="unknown campaign configuration"):
        build_campaign_plan("lite_llama", "p0_smoke")


LITE_BINDINGS = {
    "QWEN3_32B_LITE_CHECKPOINT": "/fixtures/lite/Qwen3-32B",
    "QWEN3_32B_TOKENIZER": "/fixtures/hf/Qwen3-32B",
}
VLLM_BINDINGS = {
    "QWEN3_32B_HF_CHECKPOINT": "/fixtures/hf/Qwen3-32B",
    "QWEN3_32B_TOKENIZER": "/fixtures/hf/Qwen3-32B",
}


@pytest.mark.parametrize(
    ("framework", "expected"),
    (
        ("lite_llama", ["QWEN3_32B_LITE_CHECKPOINT", "QWEN3_32B_TOKENIZER"]),
        ("vllm_ascend", ["QWEN3_32B_HF_CHECKPOINT", "QWEN3_32B_TOKENIZER"]),
    ),
)
def test_unbound_plan_preserves_placeholders_and_lists_only_framework_requirements(framework, expected):
    plan = build_campaign_plan(framework, CAMPAIGN)
    assert plan["capability_execution_ready"] is False
    assert plan["unresolved_runtime_values"] == expected
    assert plan["runtime_bindings"] == {}
    assert "${" in plan["model"]["representation"]["path"]
    assert "${" in _argument(plan["runs"][0]["formal_command"], "--tokenizer-path")


@pytest.mark.parametrize(
    ("framework", "bindings", "checkpoint"),
    (
        ("lite_llama", LITE_BINDINGS, "/fixtures/lite/Qwen3-32B"),
        ("vllm_ascend", VLLM_BINDINGS, "/fixtures/hf/Qwen3-32B"),
    ),
)
def test_bound_plan_resolves_checkpoint_and_tokenizer_without_changing_evidence_class(
    framework, bindings, checkpoint, client_profile_path
):
    plan = build_campaign_plan(
        framework, CAMPAIGN, runtime_values=bindings, client_profile=client_profile_path
    )
    launch = plan["adapter"]["launch_command"]
    client = plan["runs"][0]["formal_command"]
    assert plan["capability_execution_ready"] is True
    assert plan["unresolved_runtime_values"] == []
    assert plan["runtime_bindings"] == bindings
    assert plan["model"]["representation"]["path"] == checkpoint
    assert checkpoint in launch
    assert _argument(client, "--tokenizer-path") == "/fixtures/hf/Qwen3-32B"
    assert "${" not in "\n".join(launch + client)
    assert plan["evidence_class"] == "diagnostic"
    assert plan["performance_eligible"] is False


def test_paired_bound_plans_require_same_public_tokenizer_but_allow_checkpoint_difference(client_profile_path):
    lite = build_campaign_plan(
        "lite_llama", CAMPAIGN, runtime_values=LITE_BINDINGS, client_profile=client_profile_path
    )
    vllm = build_campaign_plan(
        "vllm_ascend", CAMPAIGN, runtime_values=VLLM_BINDINGS, client_profile=client_profile_path
    )
    assert validate_paired_plans(lite, vllm)["status"] == "pass"

    drifted = deepcopy(vllm)
    drifted["client_contract"]["tokenizer"] = "/fixtures/other/Qwen3-32B"
    report = validate_paired_plans(lite, drifted)
    assert report["status"] == "fail"
    assert "client_contract.tokenizer" in report["errors"]


@pytest.mark.parametrize(
    "arguments",
    (
        ["missing-equals"],
        ["UNKNOWN=/fixtures/model"],
        ["QWEN3_32B_TOKENIZER="],
        ["QWEN3_32B_TOKENIZER=relative/model"],
        ["QWEN3_32B_TOKENIZER=/fixtures/model;touch-x"],
        ["QWEN3_32B_TOKENIZER=/fixtures/$(touch x)"],
        ["QWEN3_32B_TOKENIZER=/fixtures/`touch x`"],
        ["QWEN3_32B_TOKENIZER=/fixtures/model\nnext"],
        ["QWEN3_32B_TOKENIZER=/fixtures/model with-space"],
        ["QWEN3_32B_TOKENIZER=/fixtures/model*"],
        ["QWEN3_32B_TOKENIZER=/fixtures/../escape"],
        ["QWEN3_32B_TOKENIZER=/fixtures/a", "QWEN3_32B_TOKENIZER=/fixtures/b"],
    ),
)
def test_runtime_binding_parser_rejects_malformed_unknown_duplicate_or_unsafe_values(arguments):
    with pytest.raises(ValueError, match="runtime value"):
        parse_runtime_value_args(arguments, declared_names={
            "QWEN3_32B_LITE_CHECKPOINT",
            "QWEN3_32B_HF_CHECKPOINT",
            "QWEN3_32B_TOKENIZER",
        })


def test_cli_runtime_values_are_repeatable_and_serialized(tmp_path, client_profile_path):
    root = tmp_path / "diagnostics"
    command = [
        sys.executable,
        "-m",
        "benchmarks.serving.run_campaign",
        "--framework",
        "lite_llama",
        "--campaign",
        str(CAMPAIGN),
        "--runtime-value",
        "QWEN3_32B_LITE_CHECKPOINT=/fixtures/lite/Qwen3-32B",
        "--runtime-value",
        "QWEN3_32B_TOKENIZER=/fixtures/hf/Qwen3-32B",
        "--output-root",
        str(root),
        "--client-profile",
        str(client_profile_path),
        "--dry-run",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["capability_execution_ready"] is True
    assert plan["runtime_bindings"] == LITE_BINDINGS
    assert not root.exists()


@pytest.mark.parametrize(
    "runtime_arguments",
    (
        ["UNKNOWN=/fixtures/model"],
        ["QWEN3_32B_TOKENIZER="],
        ["QWEN3_32B_TOKENIZER=relative/model"],
        ["QWEN3_32B_TOKENIZER=/fixtures/$(touch-x)"],
        [
            "QWEN3_32B_TOKENIZER=/fixtures/a",
            "QWEN3_32B_TOKENIZER=/fixtures/b",
        ],
    ),
)
def test_cli_rejects_invalid_runtime_values_without_traceback(runtime_arguments):
    command = [
        sys.executable,
        "-m",
        "benchmarks.serving.run_campaign",
        "--framework",
        "lite_llama",
        "--campaign",
        str(CAMPAIGN),
    ]
    for value in runtime_arguments:
        command.extend(("--runtime-value", value))
    command.append("--dry-run")
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 2
    assert "runtime value" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_mapping_api_rejects_non_string_runtime_values():
    with pytest.raises(ValueError, match="must be strings"):
        build_campaign_plan(
            "lite_llama",
            CAMPAIGN,
            runtime_values={"QWEN3_32B_TOKENIZER": Path("/fixtures/model")},
        )
