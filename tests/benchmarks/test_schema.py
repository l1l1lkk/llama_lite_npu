import hashlib
import json
from pathlib import Path

import pytest
import yaml

from benchmarks.serving.schema import CampaignSpec, load_campaign
from benchmarks.serving.run_campaign import build_campaign_plan


ROOT = Path(__file__).resolve().parents[2]


def test_loads_canonical_campaign_contract():
    spec = load_campaign(ROOT / "benchmarks/configs/campaigns/capability_smoke.yaml")

    assert isinstance(spec, CampaignSpec)
    assert spec.campaign_id == "capability_smoke"
    assert spec.kind == "capability"
    assert spec.comparison_scope == "capability_only"
    assert spec.frameworks == ("lite_llama", "vllm_ascend")
    assert spec.stream is True
    assert spec.sampling == {"temperature": 0.0, "top_p": 1.0}
    assert spec.fixed_output["strategy"] == "min_equals_max"
    assert spec.fixed_output["capability_probe_required"] is True
    assert spec.fixed_output["capability_probed"] is False
    assert spec.lifecycle == "independent_per_run"
    assert spec.repeats == 1
    assert spec.cases[0].formal_requests == 4
    assert spec.cases[0].warmup_requests == 2
    plan = build_campaign_plan("lite_llama", ROOT / "benchmarks/configs/campaigns/capability_smoke.yaml")
    assert plan["client_contract"]["dataset"]["actual_token_calibrated"] is False
    assert plan["client_contract"]["dataset"]["strict_publishable"] is False


def test_schema_rejects_accuracy_mixed_into_performance(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
schema_version: 2
campaign_id: bad
kind: performance
purpose: reject mixed accuracy
comparison_scope: production_stack
publishable: true
aggregation_allowed: true
result_namespace: results
client_profile_required: true
client_environment: {TORCH_DEVICE_BACKEND_AUTOLOAD: "0"}
model: qwen3_32b_tp2_fp16
frameworks: [lite_llama]
semantic_accuracy: {score: 1.0}
workload: {id: x, formal_dataset: x.jsonl, warmup_dataset: y.jsonl, manifest: workload.json, seed: 42, formal_offset: 0, warmup_offset: 0, request_order: frozen_jsonl}
cases: [{id: c1, prompt_tokens: 128, output_tokens: 64, concurrency: 1, formal_requests: 1, warmup_requests: 1}]
repeats: 1
lifecycle: independent_per_run
stream: true
sampling: {temperature: 0.0, top_p: 1.0}
fixed_output: {strategy: min_equals_max, capability_probe_required: true}
graph: {mode: production, causal_claim: false, required: false}
failure_gates: {strict_tokens: true}
result_root: benchmarks/results
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="semantic accuracy"):
        load_campaign(path)


def test_plan_rejects_case_requests_beyond_frozen_dataset(tmp_path):
    formal = tmp_path / "formal.jsonl"
    warmup = tmp_path / "warmup.jsonl"
    formal.write_text('{"messages":[]}\n{"messages":[]}\n', encoding="utf-8")
    warmup.write_text('{"messages":[]}\n', encoding="utf-8")
    manifest = tmp_path / "workload.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "workload_id": "too-small",
                "seed": 42,
                "request_order": "frozen_jsonl",
                "formal_offset": 0,
                "warmup_offset": 0,
                "formal_requests": 2,
                "warmup_requests": 1,
                "formal_sha256": hashlib.sha256(formal.read_bytes()).hexdigest(),
                "warmup_sha256": hashlib.sha256(warmup.read_bytes()).hexdigest(),
                "actual_token_calibrated": False,
                "strict_publishable": False,
            }
        ),
        encoding="utf-8",
    )
    campaign = tmp_path / "campaign.yaml"
    value = yaml.safe_load((ROOT / "benchmarks/configs/campaigns/capability_smoke.yaml").read_text(encoding="utf-8"))
    value["workload"].update(
        {
            "id": "too-small",
            "formal_dataset": str(formal),
            "warmup_dataset": str(warmup),
            "manifest": str(manifest),
        }
    )
    value["cases"][0]["formal_requests"] = 3
    campaign.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(ValueError, match="formal_requests exceeds frozen dataset"):
        build_campaign_plan("lite_llama", campaign)
