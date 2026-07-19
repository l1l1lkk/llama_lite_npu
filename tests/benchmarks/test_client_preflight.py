import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.serving.client_preflight import (
    REQUIRED_EVALSCOPE_FLAGS,
    generate_verified_profile,
    run_preflight_before_launch,
    verify_existing_profile,
)
from benchmarks.serving.client_profile import load_client_profile


class SyntheticProbe:
    def __init__(self, *, imports=True, tokenizer=True, help_flags=None, packages=None, identity=None):
        self.imports = imports
        self.tokenizer = tokenizer
        self.help_flags = set(help_flags or REQUIRED_EVALSCOPE_FLAGS)
        self.packages = packages
        self.identity = identity or {
            "executable": "/opt/client/bin/python",
            "prefix": "/opt/client",
            "base_prefix": "/usr",
        }

    def is_file(self, path):
        return str(path) != "/missing"

    def current_python_identity(self):
        return dict(self.identity)

    def package_versions(self):
        return self.packages or {
            "python": "3.10.12",
            "evalscope": "1.8.0",
            "modelscope": "1.36.3",
            "transformers": "5.5.3",
            "accelerate": None,
            "torch": None,
            "torch_npu": None,
        }

    def validate_imports(self):
        if not self.imports:
            raise RuntimeError("synthetic import failure")

    def load_tokenizer(self, path):
        if not self.tokenizer:
            raise RuntimeError("synthetic tokenizer failure")
        return {"class": "Qwen2TokenizerFast", "vocab_size": 151643}

    def evalscope_help_flags(self, executable, environment):
        return self.help_flags

    def distribution_fingerprint(self):
        return "2" * 64


def fixture_files(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_text("evalscope==1.8.0\n", encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name, content in (
        ("config.json", "{}"),
        ("tokenizer.json", '{"version":"1"}'),
        ("tokenizer_config.json", "{}"),
    ):
        (tokenizer / name).write_text(content, encoding="utf-8")
    return lock, tokenizer


def test_synthetic_preflight_generates_stable_verified_profile(tmp_path):
    lock, tokenizer = fixture_files(tmp_path)
    kwargs = dict(
        client_id="synthetic-client",
        python_executable="/opt/client/bin/python",
        python_prefix="/opt/client",
        evalscope_executable="/opt/client/bin/evalscope",
        requirements_lock=lock,
        tokenizer_path=tokenizer,
        probe=SyntheticProbe(),
        generated_at="2026-07-19T12:00:00Z",
    )
    first = generate_verified_profile(**kwargs)
    second = generate_verified_profile(**{**kwargs, "generated_at": "2026-07-20T12:00:00Z"})

    assert first["status"] == "verified"
    assert first["overall_fingerprint_sha256"] == second["overall_fingerprint_sha256"]
    assert first["packages"]["accelerate"] is None
    assert first["packages"]["torch"] is None
    assert first["packages"]["torch_npu"] is None
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(first), encoding="utf-8")
    assert load_client_profile(path)["overall_fingerprint_sha256"] == first["overall_fingerprint_sha256"]


@pytest.mark.parametrize(
    ("probe", "message"),
    (
        (SyntheticProbe(imports=False), "import failure"),
        (SyntheticProbe(tokenizer=False), "tokenizer failure"),
        (SyntheticProbe(help_flags={"--url"}), "missing required EvalScope flags"),
    ),
)
def test_failed_preflight_does_not_write_verified_profile(tmp_path, probe, message):
    lock, tokenizer = fixture_files(tmp_path)
    output = tmp_path / "profile.json"
    with pytest.raises(RuntimeError, match=message):
        generate_verified_profile(
            client_id="synthetic-client",
            python_executable="/opt/client/bin/python",
            python_prefix="/opt/client",
            evalscope_executable="/opt/client/bin/evalscope",
            requirements_lock=lock,
            tokenizer_path=tokenizer,
            probe=probe,
            output_path=output,
        )
    assert not output.exists()


def test_cpu_isolated_policy_rejects_torch_accelerate_or_torch_npu(tmp_path):
    lock, tokenizer = fixture_files(tmp_path)
    packages = dict(SyntheticProbe().package_versions())
    packages["accelerate"] = "1.6.0"
    with pytest.raises(RuntimeError, match="requires absent packages: accelerate"):
        generate_verified_profile(
            client_id="synthetic-client",
            python_executable="/opt/client/bin/python",
            python_prefix="/opt/client",
            evalscope_executable="/opt/client/bin/evalscope",
            requirements_lock=lock,
            tokenizer_path=tokenizer,
            probe=SyntheticProbe(packages=packages),
        )


def test_preflight_cli_failure_is_structured_json_without_traceback():
    completed = subprocess.run(
        [sys.executable, "-m", "benchmarks.serving.client_preflight"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert json.loads(completed.stdout)["status"] == "failed"
    assert "Traceback" not in completed.stderr


def test_preflight_failure_blocks_adapter_launch():
    calls = []

    result = run_preflight_before_launch(
        lambda: 1,
        lambda: calls.append("launch") or 0,
    )

    assert result == {"preflight_rc": 1, "launch_rc": None, "launch_invocations": 0}
    assert calls == []


def test_symlinked_selected_venv_path_is_preserved_without_resolve(tmp_path):
    lock, tokenizer = fixture_files(tmp_path)
    probe = SyntheticProbe(identity={
        "executable": "/envs/client-a/bin/python",
        "prefix": "/envs/client-a",
        "base_prefix": "/usr",
        "resolved_executable": "/usr/bin/python3.10",
    })
    profile = generate_verified_profile(
        client_id="symlinked-venv",
        python_executable="/envs/client-a/bin/python",
        python_prefix="/envs/client-a",
        evalscope_executable="/envs/client-a/bin/evalscope",
        requirements_lock=lock,
        tokenizer_path=tokenizer,
        probe=probe,
    )
    assert profile["python_executable"] == "/envs/client-a/bin/python"
    assert profile["python_prefix"] == "/envs/client-a"
    assert profile["python_base_prefix"] == "/usr"


def test_sibling_venv_is_rejected_even_if_it_shares_base_interpreter(tmp_path):
    lock, tokenizer = fixture_files(tmp_path)
    probe = SyntheticProbe(identity={
        "executable": "/envs/client-b/bin/python",
        "prefix": "/envs/client-b",
        "base_prefix": "/usr",
        "resolved_executable": "/usr/bin/python3.10",
    })
    with pytest.raises(RuntimeError, match="selected client Python executable"):
        generate_verified_profile(
            client_id="client-a",
            python_executable="/envs/client-a/bin/python",
            python_prefix="/envs/client-a",
            evalscope_executable="/envs/client-a/bin/evalscope",
            requirements_lock=lock,
            tokenizer_path=tokenizer,
            probe=probe,
        )


def test_base_interpreter_cannot_impersonate_selected_venv(tmp_path):
    lock, tokenizer = fixture_files(tmp_path)
    probe = SyntheticProbe(identity={
        "executable": "/usr/bin/python",
        "prefix": "/usr",
        "base_prefix": "/usr",
    })
    with pytest.raises(RuntimeError, match="virtual environment"):
        generate_verified_profile(
            client_id="base-python",
            python_executable="/usr/bin/python",
            python_prefix="/usr",
            evalscope_executable="/usr/bin/evalscope",
            requirements_lock=lock,
            tokenizer_path=tokenizer,
            probe=probe,
        )


@pytest.mark.parametrize(
    ("identity", "message"),
    (
        ({"executable": "/envs/client-b/bin/python", "prefix": "/envs/client-a", "base_prefix": "/usr"}, "selected client Python executable"),
        ({"executable": "/envs/client-a/bin/python", "prefix": "/envs/client-b", "base_prefix": "/usr"}, "Python prefix differs"),
    ),
)
def test_live_verify_detects_executable_and_prefix_drift(tmp_path, identity, message):
    lock, tokenizer = fixture_files(tmp_path)
    output = tmp_path / "profile.json"
    generate_verified_profile(
        client_id="client-a",
        python_executable="/envs/client-a/bin/python",
        python_prefix="/envs/client-a",
        evalscope_executable="/envs/client-a/bin/evalscope",
        requirements_lock=lock,
        tokenizer_path=tokenizer,
        probe=SyntheticProbe(identity={
            "executable": "/envs/client-a/bin/python",
            "prefix": "/envs/client-a",
            "base_prefix": "/usr",
        }),
        output_path=output,
    )
    with pytest.raises(RuntimeError, match=message):
        verify_existing_profile(
            output,
            probe=SyntheticProbe(identity=identity),
        )


def test_preflight_success_allows_exactly_one_adapter_launch():
    calls = []

    result = run_preflight_before_launch(
        lambda: 0,
        lambda: calls.append("launch") or 0,
    )

    assert result == {"preflight_rc": 0, "launch_rc": 0, "launch_invocations": 1}
    assert calls == ["launch"]
