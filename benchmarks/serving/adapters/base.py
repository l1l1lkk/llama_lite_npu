"""Pure descriptions of framework-specific server lifecycles."""

from __future__ import annotations

from abc import ABC
from typing import Any, Mapping

from ..schema import load_framework


class FrameworkAdapter(ABC):
    name: str

    def __init__(self) -> None:
        self.config = load_framework(self.name)

    @property
    def base_url(self) -> str:
        return str(self.config["base_url"])

    def metric_capabilities(self) -> dict[str, str]:
        return {str(key): str(value) for key, value in self.config["metrics"].items()}

    def launch_spec(self, model: Mapping[str, Any]) -> dict[str, Any]:
        substitutions = {
            "checkpoint": str(model["checkpoint"]),
            "served_name": str(model["served_name"]),
            "dtype": str(model["dtype"]),
            "tensor_parallel_size": str(model["tensor_parallel_size"]),
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
            "base_url": self.base_url,
            "health_url": self.base_url.rstrip("/") + str(self.config["health_path"]),
            "launch_command": command,
            "environment": environment,
            "metric_capabilities": self.metric_capabilities(),
            "lifecycle_actions": ["preflight", "start", "health_check", "warmup", "formal", "stop"],
        }
