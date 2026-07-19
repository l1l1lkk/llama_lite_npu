import json
import importlib.metadata
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.serving.client_preflight import (
    RealClientProbe,
    REQUIRED_EVALSCOPE_FLAGS,
    generate_verified_profile,
    run_preflight_before_launch,
    verify_existing_profile,
)
from benchmarks.serving.client_profile import load_client_profile


PERF_REQUIREMENTS = ["fastapi>=0.100", "sse-starlette>=1.6", "uvicorn>=0.20"]
PERF_MODULES = ["evalscope.perf.main", "evalscope.perf.plugin.api.openai_api"]


class SyntheticProbe:
    def __init__(
        self, *, imports=True, tokenizer=True, help_flags=None, packages=None,
        identity=None, perf_error=None, perf_requirements=None, perf_modules=None,
    ):
        self.imports = imports
        self.tokenizer = tokenizer
        self.help_flags = set(help_flags or REQUIRED_EVALSCOPE_FLAGS)
        self.packages = packages
        self.perf_error = perf_error
        self.perf_requirements = list(perf_requirements or PERF_REQUIREMENTS)
        self.perf_modules = list(perf_modules or PERF_MODULES)
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

    def evalscope_perf_contract(self):
        if self.perf_error:
            raise RuntimeError(self.perf_error)
        return {
            "extra": "perf",
            "entrypoint_import_status": "pass",
            "entrypoint_modules": self.perf_modules,
            "applicable_requirements": self.perf_requirements,
        }

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


@pytest.mark.parametrize(
    ("error", "message"),
    (
        ("EvalScope metadata does not provide extra: perf", "does not provide extra"),
        ("missing EvalScope perf requirement: uvicorn", "missing.*uvicorn"),
        ("EvalScope perf requirement version mismatch: uvicorn 0.19", "version mismatch"),
        ("EvalScope perf entrypoint import failed: No module named uvicorn", "entrypoint import failed"),
    ),
)
def test_perf_contract_failures_block_profile_and_launch(tmp_path, error, message):
    lock, tokenizer = fixture_files(tmp_path)
    output = tmp_path / "profile.json"
    probe = SyntheticProbe(perf_error=error)

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
    result = run_preflight_before_launch(lambda: 1, lambda: pytest.fail("launch invoked"))
    assert result["launch_invocations"] == 0
    assert not output.exists()


def test_positive_profile_freezes_help_imports_and_metadata_requirements(tmp_path):
    lock, tokenizer = fixture_files(tmp_path)
    profile = generate_verified_profile(
        client_id="synthetic-client",
        python_executable="/opt/client/bin/python",
        python_prefix="/opt/client",
        evalscope_executable="/opt/client/bin/evalscope",
        requirements_lock=lock,
        tokenizer_path=tokenizer,
        probe=SyntheticProbe(),
    )

    assert profile["schema_version"] == 2
    assert profile["evalscope_perf"]["extra"] == "perf"
    assert profile["evalscope_perf"]["entrypoint_import_status"] == "pass"
    assert profile["evalscope_perf"]["entrypoint_modules"] == PERF_MODULES
    assert profile["evalscope_perf"]["applicable_requirements"] == PERF_REQUIREMENTS


class FakeEvalScopeDistribution:
    def __init__(self, extras, requirements):
        self.metadata = self
        self.requires = requirements
        self._extras = extras

    def get_all(self, key):
        return self._extras if key == "Provides-Extra" else None


def install_real_perf_probe_fakes(monkeypatch, *, extras=("perf",), versions=None, import_error=None):
    requirements = [
        "fastapi>=0.100; extra == 'perf'",
        "sse-starlette>=1.6; extra == 'perf'",
        "uvicorn>=0.20; extra == 'perf'",
        "ignored-extra>=1; extra == 'other'",
    ]
    versions = versions or {
        "fastapi": "0.115.0",
        "sse-starlette": "2.1.0",
        "uvicorn": "0.30.0",
    }
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: FakeEvalScopeDistribution(extras, requirements),
    )

    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", version)

    def import_module(name):
        if import_error:
            raise ModuleNotFoundError(import_error)
        return object()

    monkeypatch.setattr("benchmarks.serving.client_preflight.importlib.import_module", import_module)


def test_real_perf_contract_is_metadata_derived_and_imports_entrypoints(monkeypatch):
    install_real_perf_probe_fakes(monkeypatch)

    contract = RealClientProbe().evalscope_perf_contract()

    assert contract["extra"] == "perf"
    assert contract["entrypoint_import_status"] == "pass"
    assert contract["entrypoint_modules"] == PERF_MODULES
    assert contract["applicable_requirements"] == [
        'fastapi>=0.100; extra == "perf"',
        'sse-starlette>=1.6; extra == "perf"',
        'uvicorn>=0.20; extra == "perf"',
    ]


def test_real_perf_contract_rejects_missing_declared_extra(monkeypatch):
    install_real_perf_probe_fakes(monkeypatch, extras=())

    with pytest.raises(RuntimeError, match="does not provide extra"):
        RealClientProbe().evalscope_perf_contract()


def test_real_perf_contract_rejects_missing_dependency(monkeypatch):
    install_real_perf_probe_fakes(
        monkeypatch,
        versions={"fastapi": "0.115.0", "sse-starlette": "2.1.0"},
    )
    with pytest.raises(RuntimeError, match="missing.*uvicorn"):
        RealClientProbe().evalscope_perf_contract()


def test_real_perf_contract_rejects_wrong_dependency_version(monkeypatch):
    install_real_perf_probe_fakes(
        monkeypatch,
        versions={"fastapi": "0.115.0", "sse-starlette": "2.1.0", "uvicorn": "0.19.0"},
    )
    with pytest.raises(RuntimeError, match="version mismatch.*uvicorn"):
        RealClientProbe().evalscope_perf_contract()


def test_real_perf_contract_rejects_actual_entrypoint_import_failure(monkeypatch):
    install_real_perf_probe_fakes(monkeypatch, import_error="No module named 'uvicorn'")

    with pytest.raises(RuntimeError, match="entrypoint import failed.*uvicorn"):
        RealClientProbe().evalscope_perf_contract()


def test_live_verify_rejects_perf_contract_drift(tmp_path):
    lock, tokenizer = fixture_files(tmp_path)
    output = tmp_path / "profile.json"
    generate_verified_profile(
        client_id="synthetic-client",
        python_executable="/opt/client/bin/python",
        python_prefix="/opt/client",
        evalscope_executable="/opt/client/bin/evalscope",
        requirements_lock=lock,
        tokenizer_path=tokenizer,
        probe=SyntheticProbe(),
        output_path=output,
    )

    with pytest.raises(RuntimeError, match="fingerprint differs"):
        verify_existing_profile(
            output,
            probe=SyntheticProbe(perf_requirements=[*PERF_REQUIREMENTS, "uvloop>=0.19"]),
        )


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
