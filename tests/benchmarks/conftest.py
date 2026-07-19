import json

import pytest

from benchmarks.serving.client_profile import compute_profile_fingerprint


@pytest.fixture
def client_profile_path(tmp_path):
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
    path = tmp_path / "client-profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path
