import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from benchmarks.serving.adapters import get_adapter
from benchmarks.serving.schema import load_framework


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "benchmarks/configs/campaigns/p0_smoke.yaml"


def test_adapters_declare_capabilities_without_fabricating_graph_zeroes():
    lite = get_adapter("lite_llama")
    vllm = get_adapter("vllm_ascend")

    assert lite.metric_capabilities()["graph_counters"] == "supported"
    assert vllm.metric_capabilities()["graph_counters"] == "unsupported"
    assert "graph_capture_delta" not in vllm.metric_capabilities()


def test_dry_run_is_stable_json_and_has_no_result_side_effect(tmp_path):
    command = [
        sys.executable,
        "-m",
        "benchmarks.serving.run_campaign",
        "--framework",
        "lite_llama",
        "--campaign",
        str(CAMPAIGN),
        "--dry-run",
    ]
    before = sorted((ROOT / "benchmarks/results").iterdir())
    first = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    after = sorted((ROOT / "benchmarks/results").iterdir())

    assert json.loads(first.stdout) == json.loads(second.stdout)
    assert before == after
    assert json.loads(first.stdout)["mode"] == "dry-run"


def test_framework_config_rejects_endpoint_drift_before_dry_run(tmp_path):
    source = ROOT / "benchmarks/configs/frameworks/lite_llama.yaml"
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    value["endpoint_path"] = "/v1/completions"
    path = tmp_path / "drift.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical endpoint"):
        load_framework(path)
