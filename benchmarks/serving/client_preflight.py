"""Generate and revalidate a CPU-only EvalScope client profile."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
from typing import Any, Mapping, Protocol

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from .client_profile import (
    CLIENT_PACKAGE_FIELDS,
    CLIENT_PROFILE_SCHEMA_VERSION,
    compute_profile_fingerprint,
    load_client_profile,
    validate_client_profile,
    validate_linux_absolute_path,
)


REQUIRED_EVALSCOPE_FLAGS = frozenset(
    {
        "--api",
        "--dataset",
        "--dataset-offset",
        "--dataset-path",
        "--max-tokens",
        "--min-tokens",
        "--model",
        "--name",
        "--no-test-connection",
        "--no-timestamp",
        "--number",
        "--outputs-dir",
        "--parallel",
        "--seed",
        "--stream",
        "--temperature",
        "--tokenizer-path",
        "--top-p",
        "--total-timeout",
        "--url",
        "--warmup-num",
    }
)
REQUIRED_CLIENT_ENVIRONMENT = {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
PERF_EXTRA = "perf"
PERF_ENTRYPOINT_MODULES = (
    "evalscope.perf.main",
    "evalscope.perf.plugin.api.openai_api",
)
MINIMUM_PERF_RUNTIME_DISTRIBUTIONS = frozenset({"fastapi", "sse-starlette", "uvicorn"})


class ClientProbe(Protocol):
    def is_file(self, path: str | Path) -> bool: ...
    def package_versions(self) -> Mapping[str, str | None]: ...
    def validate_imports(self) -> None: ...
    def evalscope_perf_contract(self) -> Mapping[str, Any]: ...
    def load_tokenizer(self, path: str | Path) -> Mapping[str, Any]: ...
    def evalscope_help_flags(self, executable: str, environment: Mapping[str, str]) -> set[str]: ...
    def distribution_fingerprint(self) -> str: ...
    def current_python_identity(self) -> Mapping[str, str]: ...


class RealClientProbe:
    """Read-only probes executed inside the selected client Python."""

    def is_file(self, path: str | Path) -> bool:
        return Path(path).is_file()

    def current_python_identity(self) -> Mapping[str, str]:
        # abspath/normpath preserve the selected venv path; realpath/resolve would
        # collapse a venv symlink to the shared base interpreter.
        normalize = lambda value: os.path.abspath(os.path.normpath(value))
        return {
            "executable": normalize(sys.executable),
            "prefix": normalize(sys.prefix),
            "base_prefix": normalize(getattr(sys, "base_prefix", sys.prefix)),
        }

    def package_versions(self) -> Mapping[str, str | None]:
        versions: dict[str, str | None] = {"python": platform.python_version()}
        for name in CLIENT_PACKAGE_FIELDS:
            if name == "python":
                continue
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        return versions

    def validate_imports(self) -> None:
        from modelscope import AutoTokenizer as ModelScopeAutoTokenizer  # noqa: F401
        from transformers import AutoTokenizer as TransformersAutoTokenizer  # noqa: F401

    @staticmethod
    def _normalized_requirement(requirement: Requirement) -> str:
        name = canonicalize_name(requirement.name)
        extras = ""
        if requirement.extras:
            extras = "[" + ",".join(sorted(canonicalize_name(extra) for extra in requirement.extras)) + "]"
        target = f" @ {requirement.url}" if requirement.url else str(requirement.specifier)
        marker = f"; {requirement.marker}" if requirement.marker else ""
        return f"{name}{extras}{target}{marker}"

    def evalscope_perf_contract(self) -> Mapping[str, Any]:
        """Validate the installed metadata closure and import the real perf entrypoints."""
        distribution = importlib.metadata.distribution("evalscope")
        provided_extras = {
            canonicalize_name(item)
            for item in (distribution.metadata.get_all("Provides-Extra") or [])
        }
        if PERF_EXTRA not in provided_extras:
            raise RuntimeError("EvalScope metadata does not provide extra: perf")

        marker_environment = default_environment()
        marker_environment["extra"] = PERF_EXTRA
        applicable: list[str] = []
        applicable_names: set[str] = set()
        for raw_requirement in distribution.requires or []:
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None and not requirement.marker.evaluate(marker_environment):
                continue
            normalized_name = canonicalize_name(requirement.name)
            try:
                installed_version = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"missing EvalScope perf requirement: {normalized_name} ({requirement})"
                ) from exc
            if requirement.specifier and Version(installed_version) not in requirement.specifier:
                raise RuntimeError(
                    "EvalScope perf requirement version mismatch: "
                    f"{normalized_name} {installed_version} does not satisfy {requirement.specifier}"
                )
            applicable.append(self._normalized_requirement(requirement))
            applicable_names.add(normalized_name)

        missing_runtime = sorted(MINIMUM_PERF_RUNTIME_DISTRIBUTIONS - applicable_names)
        if missing_runtime:
            raise RuntimeError(
                "EvalScope perf marker evaluation omitted required runtime dependencies: "
                + ", ".join(missing_runtime)
            )
        try:
            for module in PERF_ENTRYPOINT_MODULES:
                importlib.import_module(module)
        except Exception as exc:
            raise RuntimeError(f"EvalScope perf entrypoint import failed: {exc}") from exc
        return {
            "extra": PERF_EXTRA,
            "entrypoint_import_status": "pass",
            "entrypoint_modules": list(PERF_ENTRYPOINT_MODULES),
            "applicable_requirements": sorted(set(applicable)),
        }

    def load_tokenizer(self, path: str | Path) -> Mapping[str, Any]:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        return {"class": type(tokenizer).__name__, "vocab_size": int(len(tokenizer))}

    def evalscope_help_flags(self, executable: str, environment: Mapping[str, str]) -> set[str]:
        process = subprocess.run(
            [executable, "perf", "--help"],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, **environment},
            shell=False,
        )
        if process.returncode != 0:
            raise RuntimeError(f"EvalScope perf help failed with rc={process.returncode}")
        return set(re.findall(r"--[a-z0-9-]+", process.stdout))

    def distribution_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name in ("evalscope", "modelscope", "transformers"):
            distribution = importlib.metadata.distribution(name)
            for filename in ("METADATA", "RECORD"):
                content = distribution.read_text(filename)
                if content is None:
                    raise RuntimeError(f"missing {name} dist-info {filename}")
                digest.update(f"{name}/{filename}\0".encode("utf-8"))
                digest.update(content.encode("utf-8"))
        return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_profile(path: Path, profile: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def generate_verified_profile(
    *,
    client_id: str,
    python_executable: str,
    python_prefix: str,
    evalscope_executable: str,
    requirements_lock: str | Path,
    tokenizer_path: str | Path,
    probe: ClientProbe | None = None,
    generated_at: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run all CPU gates and emit a verified profile only after they pass."""
    probe = probe or RealClientProbe()
    if isinstance(probe, RealClientProbe) and os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
        raise RuntimeError("client preflight requires TORCH_DEVICE_BACKEND_AUTOLOAD=0 before imports")
    validate_linux_absolute_path(python_executable, label="python executable")
    validate_linux_absolute_path(python_prefix, label="Python environment prefix")
    validate_linux_absolute_path(evalscope_executable, label="EvalScope executable")
    if PurePosixPath(python_executable).parent != PurePosixPath(python_prefix) / "bin":
        raise RuntimeError("selected client Python executable must be inside python_prefix/bin")
    if PurePosixPath(evalscope_executable).parent != PurePosixPath(python_prefix) / "bin":
        raise RuntimeError("selected EvalScope executable must be inside python_prefix/bin")
    if not probe.is_file(python_executable):
        raise RuntimeError(f"python executable is not a file: {python_executable}")
    if not probe.is_file(evalscope_executable):
        raise RuntimeError(f"EvalScope executable is not a file: {evalscope_executable}")
    identity = dict(probe.current_python_identity())
    missing_identity = sorted({"executable", "prefix", "base_prefix"} - identity.keys())
    if missing_identity:
        raise RuntimeError(f"Python identity probe omitted fields: {', '.join(missing_identity)}")
    for field in ("executable", "prefix", "base_prefix"):
        validate_linux_absolute_path(identity[field], label=f"live Python {field}")
    if identity["executable"] != python_executable:
        raise RuntimeError(
            "preflight must run with the selected client Python executable: "
            f"{identity['executable']} != {python_executable}"
        )
    if identity["prefix"] != python_prefix:
        raise RuntimeError(
            f"preflight Python prefix differs from selected environment: {identity['prefix']} != {python_prefix}"
        )
    if identity["prefix"] == identity["base_prefix"]:
        raise RuntimeError("selected client Python must belong to a virtual environment, not the base interpreter")
    lock = Path(requirements_lock)
    tokenizer = Path(tokenizer_path)
    if not lock.is_file():
        raise RuntimeError(f"requirements lock is not a file: {lock}")
    if not tokenizer.is_dir():
        raise RuntimeError(f"tokenizer path is not a directory: {tokenizer}")
    key_files = {
        "config_sha256": tokenizer / "config.json",
        "tokenizer_sha256": tokenizer / "tokenizer.json",
        "tokenizer_config_sha256": tokenizer / "tokenizer_config.json",
    }
    missing = [path.name for path in key_files.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"tokenizer identity files are missing: {', '.join(sorted(missing))}")

    packages = dict(probe.package_versions())
    missing_package_fields = sorted(set(CLIENT_PACKAGE_FIELDS) - packages.keys())
    if missing_package_fields:
        raise RuntimeError(f"package probe omitted fields: {', '.join(missing_package_fields)}")
    present = [name for name in ("accelerate", "torch", "torch_npu") if packages[name] is not None]
    if present:
        raise RuntimeError(f"cpu-isolated policy requires absent packages: {', '.join(present)}")
    for name in ("evalscope", "modelscope", "transformers"):
        if not packages[name]:
            raise RuntimeError(f"required package is absent: {name}")

    probe.validate_imports()
    perf_contract = dict(probe.evalscope_perf_contract())
    tokenizer_result = dict(probe.load_tokenizer(tokenizer))
    flags = set(probe.evalscope_help_flags(evalscope_executable, REQUIRED_CLIENT_ENVIRONMENT))
    missing_flags = sorted(REQUIRED_EVALSCOPE_FLAGS - flags)
    if missing_flags:
        raise RuntimeError(f"missing required EvalScope flags: {', '.join(missing_flags)}")
    distribution_sha = probe.distribution_fingerprint()
    if not re.fullmatch(r"[0-9a-f]{64}", distribution_sha):
        raise RuntimeError("installed distribution fingerprint is not SHA256")

    tool_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    timestamp = generated_at or __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat().replace("+00:00", "Z")
    profile: dict[str, Any] = {
        "schema_version": CLIENT_PROFILE_SCHEMA_VERSION,
        "status": "verified",
        "client_id": client_id,
        "policy": "cpu_isolated_no_torch",
        "python_executable": python_executable,
        "python_prefix": python_prefix,
        "python_base_prefix": identity["base_prefix"],
        "evalscope_executable": evalscope_executable,
        "required_environment": dict(REQUIRED_CLIENT_ENVIRONMENT),
        "packages": {name: packages[name] for name in CLIENT_PACKAGE_FIELDS},
        "requirements_lock": {"path": str(lock), "sha256": _sha256_file(lock)},
        "installed_distribution_fingerprint": {
            "algorithm": "sha256-dist-info-metadata-record-v1",
            "sha256": distribution_sha,
        },
        "evalscope_perf": {
            **perf_contract,
            "help_status": "pass",
            "flags": sorted(flags),
        },
        "tokenizer": {
            "resolved_path": str(tokenizer.resolve()),
            **{name: _sha256_file(path) for name, path in key_files.items()},
            "cpu_load_status": "pass",
            "class": str(tokenizer_result["class"]),
            "vocab_size": int(tokenizer_result["vocab_size"]),
        },
        "generated_at": timestamp,
        "preflight": {"tool_code_sha256": tool_sha},
    }
    profile["overall_fingerprint_sha256"] = compute_profile_fingerprint(profile)
    validated = validate_client_profile(profile)
    if output_path is not None:
        _write_profile(Path(output_path), validated)
    return validated


def verify_existing_profile(path: str | Path, *, probe: ClientProbe | None = None) -> dict[str, Any]:
    expected = load_client_profile(path)
    actual = generate_verified_profile(
        client_id=expected["client_id"],
        python_executable=expected["python_executable"],
        python_prefix=expected["python_prefix"],
        evalscope_executable=expected["evalscope_executable"],
        requirements_lock=expected["requirements_lock"]["path"],
        tokenizer_path=expected["tokenizer"]["resolved_path"],
        probe=probe,
        generated_at=expected["generated_at"],
    )
    if actual["overall_fingerprint_sha256"] != expected["overall_fingerprint_sha256"]:
        raise RuntimeError("live client preflight fingerprint differs from verified profile")
    return actual


def run_preflight_before_launch(preflight, launch) -> dict[str, int | None]:
    """Machine-enforced ordering contract for future non-dry-run orchestration."""
    preflight_rc = int(preflight())
    if preflight_rc != 0:
        return {"preflight_rc": preflight_rc, "launch_rc": None, "launch_invocations": 0}
    launch_rc = int(launch())
    return {"preflight_rc": 0, "launch_rc": launch_rc, "launch_invocations": 1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-profile")
    parser.add_argument("--client-id")
    parser.add_argument("--python-executable")
    parser.add_argument("--python-prefix")
    parser.add_argument("--evalscope-executable")
    parser.add_argument("--requirements-lock")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        if args.verify_profile:
            profile = verify_existing_profile(args.verify_profile)
        else:
            required = (
                args.client_id,
                args.python_executable,
                args.python_prefix,
                args.evalscope_executable,
                args.requirements_lock,
                args.tokenizer_path,
                args.output,
            )
            if any(item is None for item in required):
                raise ValueError("profile generation requires all client, executable, lock, tokenizer, and output arguments")
            profile = generate_verified_profile(
                client_id=args.client_id,
                python_executable=args.python_executable,
                python_prefix=args.python_prefix,
                evalscope_executable=args.evalscope_executable,
                requirements_lock=args.requirements_lock,
                tokenizer_path=args.tokenizer_path,
                output_path=args.output,
            )
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"status": "failed", "errors": [str(exc)]}, sort_keys=True))
        return 1
    print(json.dumps({"status": "verified", "overall_fingerprint_sha256": profile["overall_fingerprint_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
