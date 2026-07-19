"""Pure descriptions of framework-specific server lifecycles."""

from __future__ import annotations

from abc import ABC
from typing import Any, Mapping

from ..schema import load_framework, validate_base_url


class FrameworkAdapter(ABC):
    name: str

    def __init__(self) -> None:
        self.config = load_framework(self.name)

    @property
    def base_url(self) -> str:
        return str(self.config["base_url"])

    def metric_capabilities(self) -> dict[str, str]:
        return {str(key): str(value) for key, value in self.config["metrics"].items()}

    def launch_spec(self, model: Mapping[str, Any], *, base_url: str | None = None) -> dict[str, Any]:
        runtime = validate_base_url(base_url or self.base_url)
        representation = model["representations"][self.name]
        identity = model["logical_identity"]
        substitutions = {
            "checkpoint": str(representation["path"]),
            "served_name": str(model["served_name"]),
            "dtype": str(identity["dtype"]),
            "tensor_parallel_size": str(identity["tensor_parallel_size"]),
            "port": str(runtime["port"]),
        }
        command = []
        for item in self.config["launch"]["command"]:
            rendered = str(item)
            for key, replacement in substitutions.items():
                rendered = rendered.replace("{" + key + "}", replacement)
            command.append(rendered)
        environment = {str(key): str(value) for key, value in self.config["launch"].get("environment", {}).items()}
        return {
            "framework": self.name,
            "base_url": runtime["base_url"],
            "health_url": runtime["base_url"] + str(self.config["health_path"]),
            "launch_command": command,
            "environment": environment,
            "metric_capabilities": self.metric_capabilities(),
            "runtime_environment": dict(self.config["runtime_environment"]),
            "lifecycle_actions": ["preflight", "start", "health_check", "warmup", "formal", "stop"],
        }
