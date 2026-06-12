import unittest
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "lite_llama"
    / "continuous_batching.py"
)


def load_batching_module():
    spec = importlib.util.spec_from_file_location(
        "lite_llama_continuous_batching_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeBackend:
    def __init__(self):
        self.prefill_tokens = {}
        self.decode_tokens = {}
        self.prefill_calls = []
        self.decode_calls = []
        self.released = []

    def prefill(self, requests):
        self.prefill_calls.append([request.request_id for request in requests])
        return [
            self.prefill_tokens[request.request_id].pop(0)
            for request in requests
        ]

    def decode(self, requests):
        self.decode_calls.append([request.request_id for request in requests])
        return [
            self.decode_tokens[request.request_id].pop(0)
            for request in requests
        ]

    def release(self, requests):
        self.released.extend(request.request_id for request in requests)


class ContinuousBatchSchedulerTest(unittest.TestCase):
    def _make_scheduler(self, max_batch_size=4):
        ContinuousBatchScheduler = (
            load_batching_module().ContinuousBatchScheduler
        )

        backend = FakeBackend()
        scheduler = ContinuousBatchScheduler(
            backend=backend,
            max_batch_size=max_batch_size,
            eos_token_id=99,
            decode_tokens=lambda token_ids: "".join(
                f"<{token_id}>" for token_id in token_ids
            ),
        )
        return scheduler, backend

    def test_new_request_joins_existing_decode_batch(self):
        scheduler, backend = self._make_scheduler()
        backend.prefill_tokens = {"a": [10], "b": [20]}
        backend.decode_tokens = {"a": [11, 99], "b": [21, 99]}

        request_a = scheduler.submit(
            request_id="a",
            prompt_tokens=[1, 2],
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )
        scheduler.step()
        self.assertEqual(request_a.generated_token_ids, [10])

        request_b = scheduler.submit(
            request_id="b",
            prompt_tokens=[3, 4],
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )
        scheduler.step()

        self.assertEqual(backend.prefill_calls, [["a"], ["b"]])
        self.assertEqual(backend.decode_calls, [["a"]])
        self.assertEqual(request_a.generated_token_ids, [10, 11])
        self.assertEqual(request_b.generated_token_ids, [20])

        scheduler.step()
        self.assertTrue(request_a.finished)
        self.assertEqual(request_a.finish_reason, "stop")
        self.assertFalse(request_b.finished)
        self.assertIn("a", backend.released)

    def test_max_tokens_releases_request_independently(self):
        scheduler, backend = self._make_scheduler()
        backend.prefill_tokens = {"short": [7], "long": [8]}
        backend.decode_tokens = {"long": [9, 99]}

        short = scheduler.submit(
            request_id="short",
            prompt_tokens=[1],
            max_new_tokens=1,
            temperature=0.0,
            top_p=1.0,
        )
        long = scheduler.submit(
            request_id="long",
            prompt_tokens=[2],
            max_new_tokens=4,
            temperature=0.0,
            top_p=1.0,
        )
        scheduler.step()

        self.assertTrue(short.finished)
        self.assertFalse(long.finished)
        self.assertEqual(backend.released, ["short"])

        scheduler.step()
        self.assertEqual(long.generated_token_ids, [8, 9])

    def test_pending_requests_obey_batch_capacity(self):
        scheduler, backend = self._make_scheduler(max_batch_size=1)
        backend.prefill_tokens = {"a": [10], "b": [20]}
        backend.decode_tokens = {"a": [99]}

        scheduler.submit("a", [1], 4, 0.0, 1.0)
        scheduler.submit("b", [2], 4, 0.0, 1.0)
        scheduler.step()

        self.assertEqual(backend.prefill_calls, [["a"]])
        self.assertEqual(scheduler.pending_count, 1)

        scheduler.step()
        self.assertEqual(backend.decode_calls, [["a"]])
        scheduler.step()
        self.assertEqual(backend.prefill_calls, [["a"], ["b"]])

    def test_stream_delta_uses_the_complete_decoded_prefix(self):
        request = load_batching_module().BatchRequest(
            request_id="utf8",
            prompt_tokens=[1],
            max_new_tokens=4,
            temperature=0.0,
            top_p=1.0,
        )
        decoded = {
            (10,): "你",
            (10, 11): "你好",
        }

        for token_id in (10, 11):
            request.accept_token(
                token_id,
                eos_token_id=99,
                decode_tokens=lambda ids: decoded[tuple(ids)],
            )

        self.assertEqual(request.outputs.get_nowait().delta, "你")
        self.assertEqual(request.outputs.get_nowait().delta, "好")


class FakeExecutor:
    def __init__(self):
        self.device = "cpu"
        self.use_paged_attn = True
        self.next_request_id = 0
        self.lengths = {}
        self.req_tokens_manager = SimpleNamespace(
            req_token_count=self.lengths
        )
        self.prefill_batches = []
        self.decode_batches = []
        self.released = []
        self.active_request_ids = ()

    def reserve_paged_requests(self, prompt_lengths):
        result = []
        for length in prompt_lengths:
            req_idx = self.next_request_id
            self.next_request_id += 1
            self.lengths[req_idx] = length
            result.append(req_idx)
        return tuple(result)

    def activate_paged_prefill_batch(self, request_ids, prompt_length):
        self.active_request_ids = tuple(request_ids)
        self.prefill_batches.append((tuple(request_ids), prompt_length))

    def activate_paged_decode_batch(self, request_ids):
        self.active_request_ids = tuple(request_ids)
        self.decode_batches.append(tuple(request_ids))

    def extend_paged_requests(self, request_ids):
        for req_idx in request_ids:
            self.lengths[req_idx] += 1

    def release_paged_request_ids(self, request_ids):
        self.released.extend(request_ids)
        for req_idx in request_ids:
            self.lengths.pop(req_idx, None)

    def forward(self, input_ids, position_ids):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, 128))
        for row, req_idx in enumerate(self.active_request_ids):
            logits[row, -1, 40 + req_idx] = 10
        return logits


class ContinuousBatchModelBackendTest(unittest.TestCase):
    def test_prefill_groups_requests_by_prompt_length(self):
        module = load_batching_module()
        executor = FakeExecutor()
        tokenizer = SimpleNamespace(
            eos_token_id=99,
            decode=lambda ids, skip_special_tokens=True: str(ids[0]),
        )
        generator = SimpleNamespace(
            model_executor=executor,
            tokenizer=tokenizer,
        )
        backend = module.ContinuousBatchModelBackend(generator)
        requests = [
            module.BatchRequest("a", [1, 2], 4, 0.0, 1.0),
            module.BatchRequest("b", [3], 4, 0.0, 1.0),
            module.BatchRequest("c", [4, 5], 4, 0.0, 1.0),
        ]

        tokens = backend.prefill(requests)

        self.assertEqual(tokens, [40, 41, 42])
        self.assertEqual(
            executor.prefill_batches,
            [((0, 2), 2), ((1,), 1)],
        )
        self.assertEqual([r.model_request_id for r in requests], [0, 1, 2])

    def test_decode_rebuilds_dynamic_batch_metadata(self):
        module = load_batching_module()
        executor = FakeExecutor()
        generator = SimpleNamespace(
            model_executor=executor,
            tokenizer=SimpleNamespace(eos_token_id=99),
        )
        backend = module.ContinuousBatchModelBackend(generator)
        requests = [
            module.BatchRequest("a", [1], 4, 0.0, 1.0),
            module.BatchRequest("b", [2], 4, 0.0, 1.0),
        ]
        backend.prefill(requests)
        for request, token in zip(requests, (40, 41)):
            request.generated_token_ids.append(token)

        tokens = backend.decode([requests[1]])

        self.assertEqual(tokens, [41])
        self.assertEqual(executor.decode_batches[-1], (1,))
        self.assertEqual(executor.lengths[1], 3)


if __name__ == "__main__":
    unittest.main()
