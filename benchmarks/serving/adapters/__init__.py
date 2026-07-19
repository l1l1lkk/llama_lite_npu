"""Framework adapter registry."""

from .base import FrameworkAdapter
from .lite_llama import LiteLlamaAdapter
from .vllm_ascend import VllmAscendAdapter


def get_adapter(name: str) -> FrameworkAdapter:
    adapters = {
        "lite_llama": LiteLlamaAdapter,
        "vllm_ascend": VllmAscendAdapter,
    }
    try:
        return adapters[name]()
    except KeyError as exc:
        raise ValueError(f"unsupported framework: {name}") from exc


__all__ = ["FrameworkAdapter", "get_adapter"]
