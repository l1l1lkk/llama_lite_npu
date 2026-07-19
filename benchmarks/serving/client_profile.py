"""Strict schema and fingerprint helpers for the canonical EvalScope client."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping


CLIENT_PROFILE_SCHEMA_VERSION = 1
CLIENT_PROFILE_REQUIRED = "__CLIENT_PROFILE_REQUIRED__"
CLIENT_PACKAGE_FIELDS = (
    "python",
    "evalscope",
    "modelscope",
    "transformers",
    "accelerate",
    "torch",
    "torch_npu",
)
OPTIONAL_CPU_PACKAGES = ("accelerate", "torch", "torch_npu")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_POSIX_PATH = re.compile(r"^/[A-Za-z0-9._/+:=-]+$")
_UNSAFE_PATH_CHARS = set(";`$\r\n()[]{}*?&|<>!")


def _require(mapping: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - mapping.keys())
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")


def validate_linux_absolute_path(value: Any, *, label: str) -> str:
    """Validate a normalized Linux path that is safe as one argv element."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty Linux absolute path")
    path = PurePosixPath(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be a normalized Linux absolute path")
    if not _SAFE_POSIX_PATH.fullmatch(value):
        raise ValueError(f"{label} contains unsafe characters")
    if value.startswith("//") or path.as_posix() != value or ".." in path.parts:
        raise ValueError(f"{label} must be a normalized Linux absolute path")
    return value


def validate_profile_file_path(value: str | Path) -> Path:
    """Validate a host-native absolute JSON path without shell semantics."""
    raw = str(value)
    if not raw or any(char in raw for char in _UNSAFE_PATH_CHARS):
        raise ValueError("client profile path contains unsafe characters")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("client profile path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"client profile cannot be resolved: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"client profile is not a file: {resolved}")
    return resolved


def _validate_identity_path(value: Any, *, label: str) -> str:
    """Accept canonical target Linux paths and native absolute synthetic fixtures."""
    if not isinstance(value, str) or not value or any(char in value for char in _UNSAFE_PATH_CHARS):
        raise ValueError(f"{label} contains unsafe characters")
    if _SAFE_POSIX_PATH.fullmatch(value):
        return validate_linux_absolute_path(value, label=label)
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be absolute")
    return str(path)


def _validate_hex64(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValueError(f"{label} must be 64 lowercase hex characters")
    return value


def _fingerprint_payload(profile: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(profile)
    value.pop("overall_fingerprint_sha256", None)
    # Timestamps are audit metadata, not client identity.
    value.pop("generated_at", None)
    value.pop("source_path", None)
    return value


def compute_profile_fingerprint(profile: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _fingerprint_payload(profile), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_client_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("client profile must be a JSON mapping")
    required = {
        "schema_version",
        "status",
        "client_id",
        "policy",
        "python_executable",
        "python_prefix",
        "python_base_prefix",
        "evalscope_executable",
        "required_environment",
        "packages",
        "requirements_lock",
        "installed_distribution_fingerprint",
        "evalscope_perf",
        "tokenizer",
        "generated_at",
        "preflight",
        "overall_fingerprint_sha256",
    }
    _require(value, required, "client profile")
    unexpected = sorted(set(value) - required)
    if unexpected:
        raise ValueError(f"client profile has unexpected fields: {', '.join(unexpected)}")
    if not isinstance(value["schema_version"], int) or isinstance(value["schema_version"], bool) or value["schema_version"] != CLIENT_PROFILE_SCHEMA_VERSION:
        raise ValueError("only client profile schema_version=1 is supported")
    if value["status"] != "verified":
        raise ValueError("client profile status must be verified")
    if not isinstance(value["client_id"], str) or not value["client_id"]:
        raise ValueError("client profile client_id must be a non-empty string")
    if value["policy"] != "cpu_isolated_no_torch":
        raise ValueError("client profile policy must be cpu_isolated_no_torch")
    validate_linux_absolute_path(value["python_executable"], label="client profile python_executable")
    validate_linux_absolute_path(value["python_prefix"], label="client profile python_prefix")
    validate_linux_absolute_path(value["python_base_prefix"], label="client profile python_base_prefix")
    validate_linux_absolute_path(value["evalscope_executable"], label="client profile evalscope_executable")
    prefix = PurePosixPath(value["python_prefix"])
    executable_parent = PurePosixPath(value["python_executable"]).parent
    evalscope_parent = PurePosixPath(value["evalscope_executable"]).parent
    if executable_parent != prefix / "bin":
        raise ValueError("client profile python_executable must be inside python_prefix/bin")
    if evalscope_parent != prefix / "bin":
        raise ValueError("client profile evalscope_executable must be inside python_prefix/bin")
    if value["python_prefix"] == value["python_base_prefix"]:
        raise ValueError("client profile must identify a virtual environment, not the base interpreter")

    environment = value["required_environment"]
    if not isinstance(environment, Mapping) or environment.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
        raise ValueError("client profile requires TORCH_DEVICE_BACKEND_AUTOLOAD=0")
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in environment.items()):
        raise ValueError("client profile required_environment must contain string pairs")

    packages = value["packages"]
    if not isinstance(packages, Mapping):
        raise ValueError("client profile packages must be a mapping")
    _require(packages, set(CLIENT_PACKAGE_FIELDS), "client profile packages")
    extra_packages = sorted(set(packages) - set(CLIENT_PACKAGE_FIELDS))
    if extra_packages:
        raise ValueError(f"client profile packages has unexpected fields: {', '.join(extra_packages)}")
    for name in CLIENT_PACKAGE_FIELDS:
        version = packages[name]
        if version is not None and (not isinstance(version, str) or not version):
            raise ValueError(f"client profile packages.{name} must be a version string or null")
    present = [name for name in OPTIONAL_CPU_PACKAGES if packages[name] is not None]
    if present:
        raise ValueError(f"cpu-isolated client profile requires absent packages: {', '.join(present)}")

    lock = value["requirements_lock"]
    if not isinstance(lock, Mapping):
        raise ValueError("client profile requirements_lock must be a mapping")
    _require(lock, {"path", "sha256"}, "client profile requirements_lock")
    if set(lock) != {"path", "sha256"}:
        raise ValueError("client profile requirements_lock has unexpected fields")
    _validate_identity_path(lock["path"], label="client profile requirements_lock.path")
    _validate_hex64(lock["sha256"], label="client profile requirements_lock.sha256")

    distribution = value["installed_distribution_fingerprint"]
    if not isinstance(distribution, Mapping):
        raise ValueError("client profile installed_distribution_fingerprint must be a mapping")
    _require(distribution, {"algorithm", "sha256"}, "client profile installed_distribution_fingerprint")
    if set(distribution) != {"algorithm", "sha256"}:
        raise ValueError("client profile installed_distribution_fingerprint has unexpected fields")
    if distribution["algorithm"] != "sha256-dist-info-metadata-record-v1":
        raise ValueError("unsupported installed distribution fingerprint algorithm")
    _validate_hex64(distribution["sha256"], label="client profile installed_distribution_fingerprint.sha256")

    perf = value["evalscope_perf"]
    if not isinstance(perf, Mapping) or perf.get("help_status") != "pass":
        raise ValueError("client profile EvalScope perf help status must be pass")
    flags = perf.get("flags")
    if set(perf) != {"help_status", "flags"}:
        raise ValueError("client profile evalscope_perf has unexpected fields")
    if not isinstance(flags, list) or not flags or len(flags) != len(set(flags)):
        raise ValueError("client profile EvalScope flags must be a non-empty unique list")
    if not all(isinstance(flag, str) and flag.startswith("--") for flag in flags):
        raise ValueError("client profile EvalScope flags contain an invalid item")

    tokenizer = value["tokenizer"]
    if not isinstance(tokenizer, Mapping):
        raise ValueError("client profile tokenizer must be a mapping")
    _require(
        tokenizer,
        {
            "resolved_path",
            "config_sha256",
            "tokenizer_sha256",
            "tokenizer_config_sha256",
            "cpu_load_status",
            "class",
            "vocab_size",
        },
        "client profile tokenizer",
    )
    expected_tokenizer_fields = {
        "resolved_path", "config_sha256", "tokenizer_sha256",
        "tokenizer_config_sha256", "cpu_load_status", "class", "vocab_size",
    }
    if set(tokenizer) != expected_tokenizer_fields:
        raise ValueError("client profile tokenizer has unexpected fields")
    _validate_identity_path(tokenizer["resolved_path"], label="client profile tokenizer.resolved_path")
    for field in ("config_sha256", "tokenizer_sha256", "tokenizer_config_sha256"):
        _validate_hex64(tokenizer[field], label=f"client profile tokenizer.{field}")
    if tokenizer["cpu_load_status"] != "pass":
        raise ValueError("client profile tokenizer CPU load status must be pass")
    if not isinstance(tokenizer["class"], str) or not tokenizer["class"]:
        raise ValueError("client profile tokenizer class must be a non-empty string")
    if not isinstance(tokenizer["vocab_size"], int) or tokenizer["vocab_size"] <= 0:
        raise ValueError("client profile tokenizer vocab_size must be positive")

    if not isinstance(value["generated_at"], str) or not value["generated_at"]:
        raise ValueError("client profile generated_at must be a non-empty string")
    preflight = value["preflight"]
    if not isinstance(preflight, Mapping):
        raise ValueError("client profile preflight must be a mapping")
    _require(preflight, {"tool_code_sha256"}, "client profile preflight")
    if set(preflight) != {"tool_code_sha256"}:
        raise ValueError("client profile preflight has unexpected fields")
    _validate_hex64(preflight["tool_code_sha256"], label="client profile preflight.tool_code_sha256")
    _validate_hex64(value["overall_fingerprint_sha256"], label="client profile overall_fingerprint_sha256")
    expected = compute_profile_fingerprint(value)
    if value["overall_fingerprint_sha256"] != expected:
        raise ValueError("client profile fingerprint does not match profile content")
    return json.loads(json.dumps(value))


def load_client_profile(path: str | Path) -> dict[str, Any]:
    resolved = validate_profile_file_path(path)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"client profile is not valid UTF-8 JSON: {resolved}") from exc
    profile = validate_client_profile(value)
    profile["source_path"] = str(resolved)
    return profile


def profile_for_plan(profile: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "schema_version",
        "status",
        "client_id",
        "policy",
        "python_executable",
        "python_prefix",
        "python_base_prefix",
        "evalscope_executable",
        "required_environment",
        "packages",
        "requirements_lock",
        "installed_distribution_fingerprint",
        "evalscope_perf",
        "tokenizer",
        "generated_at",
        "preflight",
        "overall_fingerprint_sha256",
        "source_path",
    )
    return {field: json.loads(json.dumps(profile[field])) for field in fields if field in profile}
