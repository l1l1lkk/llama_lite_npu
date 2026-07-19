import asyncio
import json
import sys
import types
import unittest
from queue import Queue
from types import SimpleNamespace
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


class _FakeBatchRequest:
    def __init__(self, events, generated_token_ids, *, finished=True):
        self.outputs = Queue()
        for event in events:
            self.outputs.put(SimpleNamespace(**event))
        self.generated_token_ids = list(generated_token_ids)
        self.finished = finished
        self.cancel_count = 0

    def cancel(self):
        self.cancel_count += 1


def _continuous_chunks(prompt, request, request_id, batch_request):
    with (
        patch.object(
            server, "_submit_continuous_request", return_value=batch_request
        ),
        patch.object(server, "_generator", _FakeGenerator()),
        patch.object(server, "_model_name", "Qwen3-32B"),
    ):
        return _collect_stream(
            server._stream_continuous_chat(prompt, request, request_id)
        )


def _chunk_kinds(chunks):
    kinds = []
    decoded = []
    for chunk in chunks:
        if chunk == "data: [DONE]\n\n":
            kinds.append("done")
            continue
        payload = _decode_sse(chunk)
        decoded.append(payload)
        if "error" in payload:
            kinds.append("error")
        elif payload.get("usage") is not None:
            kinds.append("usage")
        elif payload["choices"][0]["finish_reason"] is not None:
            kinds.append("finish")
        else:
            kinds.append("content")
    return kinds, decoded


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

    def _request(self, stream_options=...):
        kwargs = {}
        if stream_options is not ...:
            kwargs["stream_options"] = stream_options
        return server.ChatCompletionRequest(
            messages=[server.ChatMessage(role="user", content="ignored")],
            stream=True,
            **kwargs,
        )

    def _successful_batch(self, token_ids=(101, 102)):
        return _FakeBatchRequest(
            [
                {
                    "delta": "hello",
                    "error": None,
                    "finished": False,
                    "finish_reason": None,
                },
                {
                    "delta": " world",
                    "error": None,
                    "finished": True,
                    "finish_reason": "length",
                },
            ],
            token_ids,
        )

    def test_continuous_usage_order_and_identity(self):
        chunks = _continuous_chunks(
            "one two three",
            self._request({"include_usage": True}),
            "chatcmpl-continuous",
            self._successful_batch(),
        )
        kinds, decoded = _chunk_kinds(chunks)
        self.assertEqual(kinds, ["content", "content", "finish", "usage", "done"])
        finish = decoded[-2]
        usage = decoded[-1]
        self.assertEqual(finish["choices"][0]["finish_reason"], "length")
        self.assertEqual(usage["choices"], [])
        self.assertEqual(usage["id"], finish["id"])
        self.assertEqual(usage["model"], finish["model"])
        self.assertEqual(
            usage["usage"],
            {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )

    def test_continuous_false_and_absent_omit_usage(self):
        for request in (
            self._request({"include_usage": False}),
            self._request(),
        ):
            chunks = _continuous_chunks(
                "one two three",
                request,
                "chatcmpl-no-usage",
                self._successful_batch(),
            )
            kinds, _ = _chunk_kinds(chunks)
            self.assertEqual(kinds, ["content", "content", "finish", "done"])

    def test_continuous_request_token_counts_are_isolated(self):
        first = _continuous_chunks(
            "one two",
            self._request({"include_usage": True}),
            "chatcmpl-first",
            self._successful_batch((1,)),
        )
        second = _continuous_chunks(
            "one two three four",
            self._request({"include_usage": True}),
            "chatcmpl-second",
            self._successful_batch((1, 2, 3)),
        )
        _, first_decoded = _chunk_kinds(first)
        _, second_decoded = _chunk_kinds(second)
        self.assertEqual(first_decoded[-1]["usage"], {
            "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3,
        })
        self.assertEqual(second_decoded[-1]["usage"], {
            "prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7,
        })

    def test_continuous_error_cancels_without_success_usage(self):
        batch = _FakeBatchRequest(
            [{"delta": "", "error": "synthetic failure", "finished": False,
              "finish_reason": None}],
            (1, 2),
            finished=False,
        )
        chunks = _continuous_chunks(
            "one two three",
            self._request({"include_usage": True}),
            "chatcmpl-error",
            batch,
        )
        kinds, decoded = _chunk_kinds(chunks)
        self.assertEqual(kinds, ["error"])
        self.assertNotIn("usage", decoded[0])
        self.assertEqual(batch.cancel_count, 1)


if __name__ == "__main__":
    unittest.main()
