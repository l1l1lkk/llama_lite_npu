from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.serving.evalscope_client import build_evalscope_command
from benchmarks.serving.run_campaign import build_campaign_plan
from benchmarks.serving.validate.campaign import validate_paired_plans


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "benchmarks/configs/campaigns/p0_smoke.yaml"


def _normalized_contract(plan):
    return {
        key: plan["client_contract"][key]
        for key in (
            "api",
            "endpoint_path",
            "dataset",
            "seed",
            "formal_offset",
            "warmup_offset",
            "stream",
            "sampling",
            "fixed_output",
            "request_order",
        )
    }


def test_two_frameworks_share_one_client_contract():
    lite = build_campaign_plan("lite_llama", CAMPAIGN)
    vllm = build_campaign_plan("vllm_ascend", CAMPAIGN)

    assert _normalized_contract(lite) == _normalized_contract(vllm)
    assert lite["adapter"]["launch_command"] != vllm["adapter"]["launch_command"]
    assert lite["client_contract"]["endpoint_path"] == "/v1/chat/completions"
    assert validate_paired_plans(lite, vllm)["status"] == "pass"


def test_evalscope_command_is_generated_only_from_contract():
    plan = build_campaign_plan("lite_llama", CAMPAIGN)
    run = plan["runs"][0]
    command = build_evalscope_command(
        plan["client_contract"],
        run,
        base_url=plan["adapter"]["base_url"],
        output_dir="synthetic-output",
    )

    assert command[:2] == ["evalscope", "perf"]
    assert command[command.index("--dataset-path") + 1].endswith("formal.jsonl")
    assert command[command.index("--min-tokens") + 1] == "64"
    assert command[command.index("--max-tokens") + 1] == "64"
    assert "--stream" in command
    assert "--seed" in command
    assert "--dataset-offset" in command


def test_independent_warmup_uses_its_own_dataset_count_and_zero_offset():
    plan = build_campaign_plan("lite_llama", CAMPAIGN)
    run = plan["runs"][0]
    command = run["warmup_command"]

    assert command[command.index("--dataset-path") + 1].endswith("warmup.jsonl")
    assert command[command.index("--number") + 1] == "2"
    assert command[command.index("--dataset-offset") + 1] == "0"


@pytest.mark.parametrize(
    ("field", "value"),
    (("tokenizer", "DIFFERENT"), ("served_model", "DIFFERENT"), ("total_timeout_s", 1)),
)
def test_paired_plan_reports_client_field_drift(field, value):
    left = build_campaign_plan("lite_llama", CAMPAIGN)
    right = build_campaign_plan("vllm_ascend", CAMPAIGN)
    right = deepcopy(right)
    right["client_contract"][field] = value

    report = validate_paired_plans(left, right)

    assert report["status"] == "fail"
    assert f"client_contract.{field}" in report["errors"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_id", "different"),
        ("checkpoint", "different"),
        ("config_sha256", "different"),
        ("tokenizer", "different"),
        ("dtype", "bf16"),
        ("tensor_parallel_size", 1),
    ),
)
def test_paired_plan_reports_model_identity_drift(field, value):
    left = build_campaign_plan("lite_llama", CAMPAIGN)
    right = deepcopy(build_campaign_plan("vllm_ascend", CAMPAIGN))
    right["model"][field] = value

    report = validate_paired_plans(left, right)

    assert report["status"] == "fail"
    assert f"model.{field}" in report["errors"]
