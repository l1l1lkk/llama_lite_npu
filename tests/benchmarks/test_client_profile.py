import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from benchmarks.serving.client_profile import (
    compute_profile_fingerprint,
    load_client_profile,
)
from benchmarks.serving.run_campaign import build_campaign_plan
from benchmarks.serving.validate.campaign import validate_paired_plans


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "benchmarks/configs/campaigns/capability_smoke.yaml"
LITE_BINDINGS = {
    "QWEN3_32B_LITE_CHECKPOINT": "/fixtures/lite/Qwen3-32B",
    "QWEN3_32B_TOKENIZER": "/fixtures/hf/Qwen3-32B",
}
VLLM_BINDINGS = {
    "QWEN3_32B_HF_CHECKPOINT": "/fixtures/hf/Qwen3-32B",
    "QWEN3_32B_TOKENIZER": "/fixtures/hf/Qwen3-32B",
}


def verified_profile() -> dict:
    profile = {
        "schema_version": 1,
        "status": "verified",
        "client_id": "evalscope-client-py310-e1.8.0-ms1.36.3-tf5.5.3",
        "policy": "cpu_isolated_no_torch",
        "python_executable": "/opt/evalscope-client/bin/python",
        "python_prefix": "/opt/evalscope-client",
        "python_base_prefix": "/usr",
        "evalscope_executable": "/opt/evalscope-client/bin/evalscope",
        "required_environment": {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        "packages": {
            "python": "3.10.12",
            "evalscope": "1.8.0",
            "modelscope": "1.36.3",
            "transformers": "5.5.3",
            "accelerate": None,
            "torch": None,
            "torch_npu": None,
        },
        "requirements_lock": {
            "path": "/opt/evalscope-client/requirements.lock",
            "sha256": "1" * 64,
        },
        "installed_distribution_fingerprint": {
            "algorithm": "sha256-dist-info-metadata-record-v1",
            "sha256": "2" * 64,
        },
        "evalscope_perf": {
            "help_status": "pass",
            "flags": [
                "--api", "--dataset", "--dataset-offset", "--dataset-path",
                "--max-tokens", "--min-tokens", "--model", "--name",
                "--no-test-connection", "--no-timestamp", "--number",
                "--outputs-dir", "--parallel", "--seed", "--stream",
                "--temperature", "--tokenizer-path", "--top-p",
                "--total-timeout", "--url", "--warmup-num",
            ],
        },
        "tokenizer": {
            "resolved_path": "/fixtures/hf/Qwen3-32B",
            "config_sha256": "3" * 64,
            "tokenizer_sha256": "4" * 64,
            "tokenizer_config_sha256": "5" * 64,
            "cpu_load_status": "pass",
            "class": "Qwen2TokenizerFast",
            "vocab_size": 151643,
        },
        "generated_at": "2026-07-19T12:00:00Z",
        "preflight": {"tool_code_sha256": "6" * 64},
    }
    profile["overall_fingerprint_sha256"] = compute_profile_fingerprint(profile)
    return profile


def write_profile(tmp_path: Path, profile: dict | None = None) -> Path:
    path = tmp_path / "client-profile.json"
    path.write_text(json.dumps(profile or verified_profile()), encoding="utf-8")
    return path


def test_missing_profile_is_unready_and_never_uses_bare_evalscope():
    plan = build_campaign_plan("lite_llama", CAMPAIGN, runtime_values=LITE_BINDINGS)

    assert plan["client_profile_status"] == "unresolved"
    assert plan["capability_execution_ready"] is False
    assert "client_profile_missing" in plan["execution_readiness_reasons"]
    assert plan["runs"][0]["formal_command"][0] == "__CLIENT_PROFILE_REQUIRED__"
    assert plan["client_preflight_command"] is None


@pytest.mark.parametrize("path", ("relative/profile.json", "C:/unsafe/$(touch-x).json"))
def test_client_profile_path_rejects_relative_or_dangerous_values(path):
    with pytest.raises(ValueError, match="client profile path"):
        load_client_profile(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda p: p.update(status="unverified"), "status"),
        (lambda p: p.update(overall_fingerprint_sha256="bad"), "64 lowercase hex"),
        (lambda p: p.update(evalscope_executable="relative/evalscope"), "absolute"),
        (lambda p: p.update(evalscope_executable="/opt/evalscope;touch-x"), "unsafe"),
        (lambda p: p.update(python_prefix="/opt/sibling-client"), "bin"),
        (lambda p: p["requirements_lock"].update(sha256="0" * 64), "fingerprint"),
        (lambda p: p.update(unexpected="value"), "unexpected fields"),
    ),
)
def test_malformed_unverified_or_tampered_profile_is_rejected(tmp_path, mutation, message):
    profile = verified_profile()
    mutation(profile)
    path = write_profile(tmp_path, profile)

    with pytest.raises(ValueError, match=message):
        load_client_profile(path)


def test_verified_profile_binds_absolute_client_and_readiness_depends_on_model(tmp_path):
    path = write_profile(tmp_path)
    unbound = build_campaign_plan("lite_llama", CAMPAIGN, client_profile=path)
    bound = build_campaign_plan(
        "lite_llama", CAMPAIGN, client_profile=path, runtime_values=LITE_BINDINGS
    )

    assert unbound["capability_execution_ready"] is False
    assert unbound["execution_readiness_reasons"] == ["model_runtime_values_unresolved"]
    assert bound["capability_execution_ready"] is True
    assert bound["client_profile_status"] == "verified"
    assert bound["runs"][0]["formal_command"][0] == "/opt/evalscope-client/bin/evalscope"
    assert bound["client_preflight_command"]["argv"][0] == "/opt/evalscope-client/bin/python"
    assert bound["client_preflight_command"]["environment"] == {
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"
    }
    assert bound["client_environment"] == {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
    assert bound["performance_eligible"] is False


@pytest.mark.parametrize(
    ("path", "value", "error"),
    (
        (("evalscope_executable",), "/opt/other/bin/evalscope", "client_profile.evalscope_executable"),
        (("python_prefix",), "/opt/sibling-client", "client_profile.python_prefix"),
        (("packages", "evalscope"), "1.7.1", "client_profile.packages.evalscope"),
        (("packages", "modelscope"), "1.37.0", "client_profile.packages.modelscope"),
        (("packages", "transformers"), "5.8.0", "client_profile.packages.transformers"),
        (("requirements_lock", "sha256"), "7" * 64, "client_profile.requirements_lock.sha256"),
        (("tokenizer", "tokenizer_sha256"), "8" * 64, "client_profile.tokenizer.tokenizer_sha256"),
        (("overall_fingerprint_sha256",), "9" * 64, "client_profile.overall_fingerprint_sha256"),
    ),
)
def test_paired_profile_drift_is_field_level(tmp_path, path, value, error):
    profile_path = write_profile(tmp_path)
    lite = build_campaign_plan(
        "lite_llama", CAMPAIGN, client_profile=profile_path, runtime_values=LITE_BINDINGS
    )
    vllm = build_campaign_plan(
        "vllm_ascend", CAMPAIGN, client_profile=profile_path, runtime_values=VLLM_BINDINGS
    )
    assert validate_paired_plans(lite, vllm)["status"] == "pass"

    drifted = deepcopy(vllm)
    target = drifted["client_profile"]
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    report = validate_paired_plans(lite, drifted)
    assert report["status"] == "fail"
    assert error in report["errors"]


def test_profile_environment_conflict_is_rejected(tmp_path):
    campaign = yaml.safe_load(CAMPAIGN.read_text(encoding="utf-8"))
    campaign["client_environment"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(campaign), encoding="utf-8")

    with pytest.raises(ValueError, match="client environment conflict"):
        build_campaign_plan(
            "lite_llama", campaign_path, client_profile=write_profile(tmp_path)
        )
