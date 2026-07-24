import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from queue import Queue
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_server_module():
    package = ModuleType("lite_llama")
    utils = ModuleType("lite_llama.utils")
    device = ModuleType("lite_llama.utils.device")
    device.get_device = lambda: "cpu"
    observability = ModuleType("lite_llama.observability")
    observability.InferenceMetrics = type(
        "InferenceMetrics", (), {"__init__": lambda self: None}
    )
    tracing = ModuleType("lite_llama.tracing")
    tracing.TraceManager = type(
        "TraceManager",
        (),
        {
            "__init__": lambda self: setattr(self, "enabled", False),
            "allows": lambda self, level: False,
            "close": lambda self: None,
            "snapshot": lambda self, **kwargs: {
                "enabled": False,
                "events": [],
            },
        },
    )
    tracing.TraceLifecycleObserver = lambda manager: manager
    tracing.ObserverHub = lambda *observers: observers[0]
    tracing.TracingBackend = lambda backend, manager, rank=0: backend
    tracing.install_generator_layer_hooks = (
        lambda generator, manager: None
    )
    tracing.rank_output_path = (
        lambda output_path, rank, tensor_parallel: output_path
    )
    stubs = {
        "lite_llama": package,
        "lite_llama.utils": utils,
        "lite_llama.utils.device": device,
        "lite_llama.observability": observability,
        "lite_llama.tracing": tracing,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        module = load_module("server_min_tokens_test", "server.py")
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


server = load_server_module()


class RequestSchemaMinTokensTest(unittest.TestCase):
    def test_chat_and_completion_accept_fixed_length(self):
        chat = server.ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=256,
            min_tokens=256,
        )
        completion = server.CompletionRequest(
            prompt="hello", max_tokens=256, min_tokens=256
        )
        self.assertEqual(chat.min_tokens, 256)
        self.assertEqual(completion.min_tokens, 256)

    def test_min_tokens_defaults_to_legacy_behavior(self):
        request = server.CompletionRequest(prompt="hello", max_tokens=8)
        self.assertEqual(request.min_tokens, 0)

    def test_negative_and_over_max_values_are_rejected(self):
        with self.assertRaises(ValueError):
            server.CompletionRequest(prompt="hello", min_tokens=-1)
        with self.assertRaises(ValueError):
            server.CompletionRequest(
                prompt="hello", max_tokens=8, min_tokens=9
            )


class SamplingMinTokensTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampling = load_module(
            "lite_llama_sampling_min_tokens_test", "lite_llama/sampling.py"
        )

    def test_greedy_masks_eos_per_row(self):
        logits = torch.tensor(
            [[1.0, 9.0, 3.0], [1.0, 9.0, 3.0]], dtype=torch.float32
        )
        sampled = self.sampling.sample_next_token(
            logits,
            temperature=[0.0, 0.0],
            blocked_token_ids=[[1], []],
        )
        self.assertEqual(sampled.tolist(), [2, 1])

    def test_mask_uses_global_ids_for_vocab_shards(self):
        shard = torch.tensor([[1.0, 9.0, 3.0]], dtype=torch.float32)
        masked = self.sampling.mask_blocked_token_logits(
            shard, [[4]], vocab_start_index=3
        )
        self.assertTrue(torch.isneginf(masked[0, 1]))
        self.assertEqual(shard[0, 1].item(), 9.0)

    def test_mask_validates_batch_rows(self):
        with self.assertRaises(ValueError):
            self.sampling.mask_blocked_token_logits(
                torch.zeros((2, 4)), [[]]
            )


class BatchRequestMinTokensTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batching = load_module(
            "lite_llama_batching_min_tokens_test",
            "lite_llama/continuous_batching.py",
        )

    def test_batch_requests_support_different_min_tokens(self):
        first = self.batching.BatchRequest(
            "first", [1], 4, 0.0, 1.0, min_tokens=4
        )
        second = self.batching.BatchRequest(
            "second", [2], 4, 0.0, 1.0, min_tokens=0
        )
        self.assertEqual([first.min_tokens, second.min_tokens], [4, 0])

    def test_batch_request_rejects_invalid_min_tokens(self):
        with self.assertRaises(ValueError):
            self.batching.BatchRequest(
                "negative", [1], 4, 0.0, 1.0, min_tokens=-1
            )
        with self.assertRaises(ValueError):
            self.batching.BatchRequest(
                "over", [1], 4, 0.0, 1.0, min_tokens=5
            )

    def test_eos_is_allowed_after_minimum_is_reached(self):
        request = self.batching.BatchRequest(
            "request", [1], 4, 0.0, 1.0, min_tokens=1
        )
        request.accept_token(7, eos_token_id=99, decode_tokens=str)
        self.assertFalse(request.finished)
        request.accept_token(99, eos_token_id=99, decode_tokens=str)
        self.assertTrue(request.finished)
        self.assertEqual(request.finish_reason, "stop")

    def test_model_backend_masks_eos_until_each_row_reaches_minimum(self):
        sampling = load_module(
            "lite_llama.sampling", "lite_llama/sampling.py"
        )
        package = ModuleType("lite_llama")
        backend = self.batching.ContinuousBatchModelBackend.__new__(
            self.batching.ContinuousBatchModelBackend
        )
        backend.executor = SimpleNamespace(logits_are_sharded=False)
        backend.tokenizer = SimpleNamespace(eos_token_id=1)
        backend.sampling_candidate_k = 8
        backend.metrics = None
        backend._device_tokens = {}
        backend._device_positions = {}
        first = self.batching.BatchRequest(
            "first", [1], 4, 0.0, 1.0, min_tokens=1
        )
        second = self.batching.BatchRequest(
            "second", [2], 4, 0.0, 1.0, min_tokens=0
        )
        first.model_request_id = 0
        second.model_request_id = 1
        logits = torch.tensor(
            [[1.0, 9.0, 3.0], [1.0, 9.0, 3.0]], dtype=torch.float32
        )
        with patch.dict(
            sys.modules,
            {"lite_llama": package, "lite_llama.sampling": sampling},
        ):
            sampled = backend._sample_device(logits, [first, second])
            backend._remember_sampled_tokens([first, second], sampled)
            next_sampled = backend._sample_device(logits[:1], [first])
        self.assertEqual(sampled.tolist(), [2, 1])
        self.assertEqual(first.sampled_token_count, 1)
        self.assertEqual(next_sampled.tolist(), [1])


class StreamingMinTokensTest(unittest.IsolatedAsyncioTestCase):
    async def test_chat_stream_forwards_min_tokens(self):
        batch_request = SimpleNamespace(
            outputs=Queue(),
            finished=True,
            generated_token_ids=[],
            cancel=lambda: None,
        )
        batch_request.outputs.put(
            SimpleNamespace(
                delta="", error=None, finished=True, finish_reason="length"
            )
        )
        request = server.ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=8,
            min_tokens=8,
            stream=True,
        )
        with patch.object(
            server, "_submit_continuous_request", return_value=batch_request
        ) as submit:
            chunks = [
                chunk
                async for chunk in server._stream_continuous_chat(
                    "hello", request, "request-id"
                )
            ]
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertEqual(submit.call_args.args[5], 8)

    async def test_completion_stream_forwards_min_tokens(self):
        batch_request = SimpleNamespace(
            outputs=Queue(), finished=True, cancel=lambda: None
        )
        batch_request.outputs.put(
            SimpleNamespace(delta="", error=None, finished=True)
        )
        request = server.CompletionRequest(
            prompt="hello",
            max_tokens=8,
            min_tokens=8,
            stream=True,
        )
        with patch.object(
            server, "_submit_continuous_request", return_value=batch_request
        ) as submit:
            chunks = [
                chunk
                async for chunk in server._stream_continuous_completion(
                    "hello", request, "request-id"
                )
            ]
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertEqual(submit.call_args.args[5], 8)


class TpControlMinTokensTest(unittest.TestCase):
    def test_prefill_round_trip_keeps_per_request_min_tokens(self):
        module = load_module(
            "lite_llama_tp_control_min_tokens_test",
            "lite_llama/executor/tp_control.py",
        )
        requests = [
            SimpleNamespace(
                control_id=1,
                prompt_tokens=[10],
                max_new_tokens=8,
                min_tokens=8,
                temperature=0.0,
                top_p=1.0,
            ),
            SimpleNamespace(
                control_id=2,
                prompt_tokens=[20],
                max_new_tokens=8,
                min_tokens=0,
                temperature=0.0,
                top_p=1.0,
            ),
        ]
        decoded = module.decode_command(*module.encode_prefill(requests))
        self.assertEqual(decoded.min_tokens, [8, 0])


if __name__ == "__main__":
    unittest.main()
