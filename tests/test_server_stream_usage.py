import asyncio
import json
import sys
import types
import unittest
from unittest.mock import patch

lite_llama_module = types.ModuleType("lite_llama")
utils_module = types.ModuleType("lite_llama.utils")
device_module = types.ModuleType("lite_llama.utils.device")
device_module.get_device = lambda device=None: device or "cpu"
observability_module = types.ModuleType("lite_llama.observability")


class _FakeInferenceMetrics:
    def sync_runtime(self, executor):
        return None

    def render(self):
        return b""

    @property
    def content_type(self):
        return "text/plain"

    def snapshot(self):
        return {}


observability_module.InferenceMetrics = _FakeInferenceMetrics
sys.modules.setdefault("lite_llama", lite_llama_module)
sys.modules.setdefault("lite_llama.utils", utils_module)
sys.modules.setdefault("lite_llama.utils.device", device_module)
sys.modules.setdefault("lite_llama.observability", observability_module)

import server


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class _FakeGenerator:
    tokenizer = _FakeTokenizer()

    def text_completion_stream(self, prompts, **kwargs):
        yield [{"generation": "hello"}]
        yield [{"generation": "hello world"}]


def _collect_stream(generator):
    async def collect():
        return [chunk async for chunk in generator]

    return asyncio.run(collect())


def _decode_sse(chunk):
    return json.loads(chunk.removeprefix("data: ").strip())


class StreamUsageTest(unittest.TestCase):
    def test_stream_chat_returns_usage_chunk_when_requested(self):
        with (
            patch.object(server, "_generator", _FakeGenerator()),
            patch.object(server, "_is_tp", False),
            patch.object(server, "_is_vl", False),
        ):
            request = server.ChatCompletionRequest(
                messages=[server.ChatMessage(role="user", content="ignored")],
                stream=True,
                stream_options={"include_usage": True},
            )

            chunks = _collect_stream(
                server._stream_chat("one two three", [], request, "chatcmpl-test")
            )

        usage_chunk = _decode_sse(chunks[-2])
        self.assertEqual(usage_chunk["choices"], [])
        self.assertEqual(
            usage_chunk["usage"],
            {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        )
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()
