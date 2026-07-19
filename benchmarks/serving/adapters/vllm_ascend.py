"""vLLM-Ascend server lifecycle description."""

from .base import FrameworkAdapter


class VllmAscendAdapter(FrameworkAdapter):
    name = "vllm_ascend"
