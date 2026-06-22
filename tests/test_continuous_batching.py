import unittest
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "lite_llama"
    / "continuous_batching.py"
)


def load_batching_module():
    package_root = MODULE_PATH.parent
    package = sys.modules.get("lite_llama")
    if package is None:
        package = ModuleType("lite_llama")
        sys.modules["lite_llama"] = package
    package.__path__ = [str(package_root)]
    spec = importlib.util.spec_from_file_location(
        "lite_llama_continuous_batching_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeBackend:
    def __init__(self):
        self.max_context_tokens = None
        self.max_prefill_tokens = None
        self.prefill_tokens = {}
        self.prefill_chunk_tokens = {}
        self.decode_tokens = {}
        self.prefill_calls = []
        self.prefill_chunk_calls = []
        self.decode_calls = []
        self.released = []
        self.prefill_errors = []
        self.reserved_lengths = []

    def prefill(self, requests):
        self.prefill_calls.append([request.request_id for request in requests])
        if self.prefill_errors:
            raise self.prefill_errors.pop(0)
        return [
            self.prefill_tokens[request.request_id].pop(0)
            for request in requests
        ]

    def prefill_chunk(self, requests, chunk_size):
        self.prefill_chunk_calls.append(
            [(request.request_id, request.prefill_cursor, chunk_size) for request in requests]
        )
        results = []
        for request in requests:
            request.prefill_cursor = min(
                len(request.model_context_tokens),
                request.prefill_cursor + int(chunk_size),
            )
            if request.prefill_cursor >= len(request.model_context_tokens):
                results.append(self.prefill_chunk_tokens[request.request_id].pop(0))
            else:
                results.append(None)
        return results

    def decode(self, requests):
        self.decode_calls.append([request.request_id for request in requests])
        return [
            self.decode_tokens[request.request_id].pop(0)
            for request in requests
        ]

    def release(self, requests):
        self.released.extend(request.request_id for request in requests)

    def preempt(self, requests):
        self.release(requests)


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

    def test_submit_rejects_prompt_at_context_capacity(self):
        scheduler, backend = self._make_scheduler()
        scheduler.max_context_tokens = 4
        backend.max_context_tokens = 4

        with self.assertRaisesRegex(ValueError, "prompt length exceeds"):
            scheduler.submit("too-long", [1, 2, 3, 4], 1, 0.0, 1.0)

    def test_submit_rejects_prompt_plus_generation_over_context_capacity(self):
        scheduler, backend = self._make_scheduler()
        scheduler.max_context_tokens = 4
        backend.max_context_tokens = 4

        with self.assertRaisesRegex(ValueError, "prompt plus generation"):
            scheduler.submit("too-long", [1, 2, 3], 2, 0.0, 1.0)

    def test_context_full_active_request_finishes_before_decode(self):
        scheduler, backend = self._make_scheduler()
        scheduler.max_context_tokens = 3
        backend.decode_tokens = {"full": [99]}
        request = scheduler.submit("full", [1], 2, 0.0, 1.0)
        request.generated_token_ids = [10, 11]
        scheduler._pending.clear()
        scheduler._active = [request]

        scheduler.step()

        self.assertTrue(request.finished)
        self.assertEqual(request.finish_reason, "length")
        self.assertEqual(backend.decode_calls, [])
        self.assertEqual(backend.released, ["full"])

    def test_prefill_token_budget_limits_admission(self):
        scheduler, backend = self._make_scheduler(max_batch_size=4)
        scheduler.max_prefill_tokens = 4
        backend.prefill_tokens = {"a": [10], "b": [20], "c": [30]}
        backend.decode_tokens = {"a": [99], "b": [99], "c": [99]}

        scheduler.submit("a", [1, 2, 3], 4, 0.0, 1.0)
        scheduler.submit("b", [4, 5, 6], 4, 0.0, 1.0)
        scheduler.submit("c", [7, 8], 4, 0.0, 1.0)

        scheduler.step()

        self.assertEqual(backend.prefill_calls, [["a"]])
        self.assertEqual(scheduler.pending_count, 2)

        scheduler.step()
        self.assertEqual(backend.prefill_calls, [["a"], ["b"]])

    def test_prefill_token_budget_defaults_from_backend(self):
        module = load_batching_module()
        backend = FakeBackend()
        backend.max_prefill_tokens = 4
        scheduler = module.ContinuousBatchScheduler(
            backend=backend,
            max_batch_size=4,
            eos_token_id=99,
            decode_tokens=lambda token_ids: "".join(
                f"<{token_id}>" for token_id in token_ids
            ),
        )
        backend.prefill_tokens = {"a": [10], "b": [20], "c": [30]}
        backend.decode_tokens = {"a": [99], "b": [99], "c": [99]}

        scheduler.submit("a", [1, 2, 3], 4, 0.0, 1.0)
        scheduler.submit("b", [4, 5, 6], 4, 0.0, 1.0)
        scheduler.submit("c", [7, 8], 4, 0.0, 1.0)

        scheduler.step()

        self.assertEqual(scheduler.max_prefill_tokens, 4)
        self.assertEqual(backend.prefill_calls, [["a"]])
        self.assertEqual(scheduler.pending_count, 2)

    def test_prefill_token_budget_admits_one_oversized_request(self):
        scheduler, backend = self._make_scheduler(max_batch_size=4)
        scheduler.max_prefill_tokens = 2
        backend.prefill_tokens = {"long": [10], "short": [20]}
        backend.decode_tokens = {"long": [99]}

        scheduler.submit("long", [1, 2, 3, 4], 4, 0.0, 1.0)
        scheduler.submit("short", [5], 4, 0.0, 1.0)

        scheduler.step()

        self.assertEqual(backend.prefill_calls, [["long"]])
        self.assertEqual(scheduler.pending_count, 1)

    def test_chunked_prefill_starts_long_prompt_without_waiting_for_full_budget(self):
        module = load_batching_module()
        backend = FakeBackend()
        scheduler = module.ContinuousBatchScheduler(
            backend=backend,
            max_batch_size=2,
            eos_token_id=99,
            decode_tokens=lambda token_ids: "".join(
                f"<{token_id}>" for token_id in token_ids
            ),
            max_prefill_tokens=4,
            chunked_prefill=True,
            prefill_chunk_size=4,
        )
        backend.prefill_chunk_tokens = {"long": [10], "short": [20]}

        scheduler.submit("long", [1, 2, 3, 4, 5, 6, 7, 8], 4, 0.0, 1.0)
        scheduler.submit("short", [9, 10], 4, 0.0, 1.0)

        scheduler.step()

        self.assertEqual(backend.prefill_chunk_calls, [[("long", 0, 4)]])
        self.assertEqual(scheduler.pending_count, 1)
        self.assertEqual(scheduler.prefilling_count, 1)

    def test_chunked_prefill_executes_long_prompt_across_scheduler_ticks(self):
        module = load_batching_module()
        backend = FakeBackend()
        scheduler = module.ContinuousBatchScheduler(
            backend=backend,
            max_batch_size=2,
            eos_token_id=99,
            decode_tokens=lambda token_ids: "".join(
                f"<{token_id}>" for token_id in token_ids
            ),
            max_prefill_tokens=2,
            chunked_prefill=True,
            prefill_chunk_size=2,
        )
        backend.prefill_chunk_tokens = {"long": [10]}

        request = scheduler.submit("long", [1, 2, 3, 4, 5], 4, 0.0, 1.0)

        scheduler.step()
        self.assertEqual(request.generated_token_ids, [])
        self.assertEqual(scheduler.prefilling_count, 1)

        scheduler.step()
        self.assertEqual(request.generated_token_ids, [])
        self.assertEqual(scheduler.prefilling_count, 1)

        scheduler.step()
        self.assertEqual(request.generated_token_ids, [10])
        self.assertEqual(scheduler.prefilling_count, 0)
        self.assertEqual(
            backend.prefill_chunk_calls,
            [[("long", 0, 2)], [("long", 2, 2)], [("long", 4, 2)]],
        )

    def test_decode_token_budget_limits_active_decode_rows(self):
        scheduler, backend = self._make_scheduler(max_batch_size=4)
        scheduler.max_decode_tokens = 1
        backend.prefill_tokens = {"a": [10], "b": [20]}
        backend.decode_tokens = {"a": [11, 99], "b": [21, 99]}

        scheduler.submit("a", [1], 4, 0.0, 1.0)
        scheduler.submit("b", [2], 4, 0.0, 1.0)
        scheduler.step()
        scheduler.step()

        self.assertEqual(backend.decode_calls, [["a"]])
        scheduler.step()
        self.assertEqual(backend.decode_calls, [["a"], ["b"]])

    def test_incremental_decoder_limits_normal_decode_window(self):
        module = load_batching_module()
        request = module.BatchRequest(
            request_id="bounded",
            prompt_tokens=[1],
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
        )
        decoded_lengths = []

        def decode(ids):
            decoded_lengths.append(len(ids))
            return "".join(chr(96 + token_id) for token_id in ids)

        for token_id in range(1, 17):
            request.accept_token(
                token_id,
                eos_token_id=99,
                decode_tokens=decode,
            )

        self.assertLessEqual(max(decoded_lengths), 9)
        self.assertEqual(request.decoded_text, "abcdefghijklmnop")

    def test_stream_delta_uses_the_complete_decoded_prefix(self):
        request = load_batching_module().BatchRequest(
            request_id="utf8",
            prompt_tokens=[1],
            max_new_tokens=4,
            temperature=0.0,
            top_p=1.0,
        )
        decoded = {
            (10,): "?",
            (10, 11): "??",
        }

        for token_id in (10, 11):
            request.accept_token(
                token_id,
                eos_token_id=99,
                decode_tokens=lambda ids: decoded[tuple(ids)],
            )

        self.assertEqual(request.outputs.get_nowait().delta, "?")
        self.assertEqual(request.outputs.get_nowait().delta, "?")

    def test_kv_capacity_error_preempts_active_request_and_requeues_new_request(self):
        module = load_batching_module()
        scheduler, backend = self._make_scheduler(max_batch_size=2)
        backend.prefill_tokens = {"active": [10], "new": [20]}
        backend.decode_tokens = {"active": [11]}

        active = scheduler.submit("active", [1, 2], 4, 0.0, 1.0)
        scheduler.step()
        self.assertEqual(active.generated_token_ids, [10])

        backend.prefill_errors.append(module.KVCacheCapacityError("Paged KV capacity exhausted"))
        new = scheduler.submit("new", [3, 4], 4, 0.0, 1.0)

        self.assertTrue(scheduler.step())
        self.assertEqual(backend.released, ["active"])
        self.assertEqual(active.preemptions, 1)
        self.assertEqual(scheduler.pending_count, 2)
        self.assertIsNone(new.model_request_id)

    def test_preempted_request_prefill_uses_prompt_plus_generated_context(self):
        module = load_batching_module()
        request = module.BatchRequest("r", [1, 2], 4, 0.0, 1.0)
        request.generated_token_ids.extend([10, 11])

        self.assertEqual(request.model_context_tokens, [1, 2, 10, 11])


class FakeExecutor:
    def __init__(self):
        self.device = "cpu"
        self.use_paged_attn = True
        self.max_prefill_tokens = None
        self.next_request_id = 0
        self.lengths = {}
        self.req_tokens_manager = SimpleNamespace(
            req_token_count=self.lengths
        )
        self.prefill_batches = []
        self.decode_batches = []
        self.released = []
        self.active_request_ids = ()
        self.forward_inputs = []
        self.logits_are_sharded = False
        self.reserved_lengths = []
        self.packed_prefill_batches = []
        self.sample_indices = None

    def reserve_paged_requests(self, prompt_lengths, reserved_lengths=None):
        self.reserved_lengths.append(
            tuple(
                reserved_lengths
                if reserved_lengths is not None
                else prompt_lengths
            )
        )
        result = []
        for length in prompt_lengths:
            req_idx = self.next_request_id
            self.next_request_id += 1
            self.lengths[req_idx] = length
            result.append(req_idx)
        return tuple(result)

    def ensure_paged_request_capacity(self, req_idx, total_tokens):
        return None

    def activate_paged_prefill_batch(self, request_ids, prompt_length):
        self.active_request_ids = tuple(request_ids)
        self.prefill_batches.append((tuple(request_ids), prompt_length))

    def activate_paged_packed_prefill_batch(self, request_ids, prompt_lengths):
        self.active_request_ids = tuple(request_ids)
        start = 0
        starts = []
        sample_indices = []
        for length in prompt_lengths:
            starts.append(start)
            sample_indices.append(start + int(length) - 1)
            start += int(length)
        self.sample_indices = tuple(sample_indices)
        self.packed_prefill_batches.append(
            (tuple(request_ids), tuple(prompt_lengths), tuple(starts))
        )
        position_ids = [
            position
            for length in prompt_lengths
            for position in range(int(length))
        ]
        return (
            torch.tensor(position_ids, dtype=torch.long),
            torch.tensor(sample_indices, dtype=torch.long),
        )

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

    def share_paged_request_from_cache(self, prompt_tokens):
        return None

    def store_paged_request_prefix(self, prompt_tokens, req_idx, first_token_id):
        pass

    def forward(self, input_ids, position_ids):
        self.forward_inputs.append(
            (input_ids.detach().clone(), position_ids.detach().clone())
        )
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, 128))
        if batch_size == 1 and self.sample_indices is not None:
            for req_idx, token_index in zip(self.active_request_ids, self.sample_indices):
                logits[0, token_index, 40 + req_idx] = 10
            self.sample_indices = None
        else:
            for row, req_idx in enumerate(self.active_request_ids):
                logits[row, -1, 40 + req_idx] = 10
        return logits


class ContinuousBatchModelBackendTest(unittest.TestCase):
    def test_prefill_packs_mixed_length_requests_into_one_forward(self):
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
        self.assertEqual(executor.prefill_batches, [])
        self.assertEqual(
            executor.packed_prefill_batches,
            [((0, 1, 2), (2, 1, 2), (0, 2, 3))],
        )
        input_ids, position_ids = executor.forward_inputs[0]
        self.assertEqual(input_ids.tolist(), [[1, 2, 3, 4, 5]])
        self.assertEqual(position_ids.tolist(), [[0, 1, 0, 0, 1]])
        self.assertEqual([r.model_request_id for r in requests], [0, 1, 2])

    def test_prefill_splits_packed_batches_by_backend_token_budget(self):
        module = load_batching_module()
        executor = FakeExecutor()
        executor.max_prefill_tokens = 3
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
            module.BatchRequest("b", [3, 4], 4, 0.0, 1.0),
            module.BatchRequest("c", [5], 4, 0.0, 1.0),
        ]

        tokens = backend.prefill(requests)

        self.assertEqual(tokens, [40, 41, 42])
        self.assertEqual(
            executor.packed_prefill_batches,
            [((1, 2), (2, 1), (0, 2))],
        )
        self.assertEqual(executor.prefill_batches, [((0,), 2)])
        self.assertEqual(len(executor.forward_inputs), 2)

    def test_single_prompt_over_prefill_budget_falls_back_to_incremental_replay(self):
        module = load_batching_module()
        executor = FakeExecutor()
        executor.max_prefill_tokens = 2
        generator = SimpleNamespace(
            model_executor=executor,
            tokenizer=SimpleNamespace(eos_token_id=99),
        )
        backend = module.ContinuousBatchModelBackend(generator)
        request = module.BatchRequest("long", [1, 2, 3, 4], 4, 0.0, 1.0)

        tokens = backend.prefill([request])

        self.assertEqual(tokens, [40])
        self.assertEqual(executor.reserved_lengths, [(5,)])
        self.assertEqual(executor.prefill_batches, [])
        self.assertEqual(executor.packed_prefill_batches, [])
        self.assertEqual(
            [input_ids.tolist() for input_ids, _ in executor.forward_inputs],
            [[[1]], [[2]], [[3]], [[4]]],
        )

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

    def test_decode_reuses_device_token_and_position_state(self):
        module = load_batching_module()
        executor = FakeExecutor()
        generator = SimpleNamespace(
            model_executor=executor,
            tokenizer=SimpleNamespace(eos_token_id=99),
        )
        backend = module.ContinuousBatchModelBackend(generator)
        request = module.BatchRequest("a", [1, 2], 4, 0.0, 1.0)

        self.assertEqual(backend.prefill([request]), [40])
        request.generated_token_ids[:] = [77]
        executor.lengths[0] = 999

        backend.decode([request])

        decode_input, decode_positions = executor.forward_inputs[-1]
        self.assertEqual(decode_input.tolist(), [[40]])
        self.assertEqual(decode_positions.tolist(), [[2]])

    def test_worker_backend_keeps_sampled_tokens_on_device_without_host_result(self):
        module = load_batching_module()
        executor = FakeExecutor()
        generator = SimpleNamespace(
            model_executor=executor,
            tokenizer=SimpleNamespace(eos_token_id=99),
        )
        backend = module.ContinuousBatchModelBackend(
            generator, return_host_tokens=False
        )
        request = module.BatchRequest("worker", [1], 4, 0.0, 1.0)

        result = backend.prefill([request])

        self.assertEqual(result, [])
        self.assertEqual(
            backend._device_tokens[request.model_request_id].tolist(), 40
        )

    def test_exact_prefix_cache_hit_skips_prefill_forward(self):
        module = load_batching_module()

        class PrefixCacheExecutor(FakeExecutor):
            def __init__(self):
                super().__init__()
                self.cache = {}
                self.shared = []

            def share_paged_request_from_cache(self, prompt_tokens):
                key = tuple(prompt_tokens)
                if key not in self.cache:
                    return None
                req_idx = self.next_request_id
                self.next_request_id += 1
                length, token_id = self.cache[key]
                self.lengths[req_idx] = length
                self.shared.append((req_idx, key))
                return req_idx, token_id

            def store_paged_request_prefix(self, prompt_tokens, req_idx, first_token_id):
                self.cache[tuple(prompt_tokens)] = (self.lengths[req_idx], first_token_id)

        executor = PrefixCacheExecutor()
        generator = SimpleNamespace(
            model_executor=executor,
            tokenizer=SimpleNamespace(eos_token_id=99),
        )
        backend = module.ContinuousBatchModelBackend(generator)
        first = module.BatchRequest("first", [1, 2, 3], 4, 0.0, 1.0)
        second = module.BatchRequest("second", [1, 2, 3], 4, 0.0, 1.0)

        self.assertEqual(backend.prefill([first]), [40])
        self.assertEqual(backend.prefill([second]), [40])

        self.assertEqual(len(executor.forward_inputs), 1)
        self.assertEqual(executor.reserved_lengths, [(3,)])
        self.assertEqual(executor.shared, [(1, (1, 2, 3))])
        self.assertEqual(second.model_request_id, 1)
        self.assertEqual(backend._device_positions[1].tolist(), 3)

    def test_partial_prefix_cache_is_disabled_by_default(self):
        module = load_batching_module()

        class PartialPrefixExecutor(FakeExecutor):
            def __init__(self):
                super().__init__()
                self.partial_lookup_calls = 0

            def share_paged_prefix_from_cache(self, prompt_tokens):
                self.partial_lookup_calls += 1
                return (1, 4, None)

        executor = PartialPrefixExecutor()
        generator = SimpleNamespace(
            model_executor=executor,
            tokenizer=SimpleNamespace(eos_token_id=99),
        )
        backend = module.ContinuousBatchModelBackend(generator)
        request = module.BatchRequest("partial-disabled", [1, 2, 3, 4, 5, 6], 4, 0.0, 1.0)

        tokens = backend.prefill([request])

        self.assertEqual(tokens, [40])
        self.assertEqual(executor.partial_lookup_calls, 0)
        forwarded_token_ids = [
            input_ids.tolist()[0]
            for input_ids, _ in executor.forward_inputs
        ]
        self.assertEqual(forwarded_token_ids, [[1, 2, 3, 4, 5, 6]])

    def test_partial_prefix_cache_hit_replays_only_suffix_tokens(self):
        module = load_batching_module()

        class PartialPrefixExecutor(FakeExecutor):
            def __init__(self):
                super().__init__()
                self.next_request_id = 3
                self.lengths[0] = 4
                self.lengths[1] = 4
                self.partial_hits = {
                    (1, 2, 3, 4, 5, 6): (1, 4, None),
                }
                self.extended = []

            def share_paged_prefix_from_cache(self, prompt_tokens):
                return self.partial_hits.get(tuple(prompt_tokens))

            def extend_paged_requests(self, request_ids):
                self.extended.append(tuple(request_ids))
                super().extend_paged_requests(request_ids)

        executor = PartialPrefixExecutor()
        generator = SimpleNamespace(
            model_executor=executor,
            tokenizer=SimpleNamespace(eos_token_id=99),
        )
        backend = module.ContinuousBatchModelBackend(
            generator, enable_partial_prefix_cache=True
        )
        request = module.BatchRequest("partial", [1, 2, 3, 4, 5, 6], 4, 0.0, 1.0)

        tokens = backend.prefill([request])

        self.assertEqual(tokens, [41])
        self.assertEqual(request.model_request_id, 1)
        forwarded_token_ids = [
            input_ids.tolist()[0][0]
            for input_ids, _ in executor.forward_inputs
        ]
        forwarded_positions = [
            position_ids.tolist()[0][0]
            for _, position_ids in executor.forward_inputs
        ]
        self.assertEqual(forwarded_token_ids, [5, 6])
        self.assertEqual(forwarded_positions, [4, 5])

    def test_prefill_chunk_replays_multiple_requests_as_decode_micro_batches(self):
        module = load_batching_module()
        executor = FakeExecutor()
        generator = SimpleNamespace(
            model_executor=executor,
            tokenizer=SimpleNamespace(eos_token_id=99),
        )
        backend = module.ContinuousBatchModelBackend(generator)
        request_a = module.BatchRequest("a", [1, 2, 3], 4, 0.0, 1.0)
        request_b = module.BatchRequest("b", [4, 5], 4, 0.0, 1.0)

        first = backend.prefill_chunk([request_a, request_b], chunk_size=2)

        self.assertEqual(first, [None, 41])
        self.assertEqual(request_a.prefill_cursor, 2)
        self.assertEqual(request_b.prefill_cursor, 2)
        self.assertEqual(executor.forward_inputs[0][0].tolist(), [[1], [4]])
        self.assertEqual(executor.forward_inputs[0][1].tolist(), [[0], [0]])
        self.assertEqual(executor.forward_inputs[1][0].tolist(), [[2], [5]])
        self.assertEqual(executor.forward_inputs[1][1].tolist(), [[1], [1]])

        second = backend.prefill_chunk([request_a], chunk_size=2)

        self.assertEqual(second, [40])
        self.assertEqual(request_a.prefill_cursor, 3)
        self.assertEqual(executor.forward_inputs[-1][0].tolist(), [[3]])
        self.assertEqual(executor.forward_inputs[-1][1].tolist(), [[2]])

    def test_exact_prefix_cache_is_disabled_for_sampling_requests(self):
        module = load_batching_module()

        class PrefixCacheExecutor(FakeExecutor):
            def __init__(self):
                super().__init__()
                self.cache = {}

            def share_paged_request_from_cache(self, prompt_tokens):
                if tuple(prompt_tokens) in self.cache:
                    return self.next_request_id, self.cache[tuple(prompt_tokens)]
                return None

            def store_paged_request_prefix(self, prompt_tokens, req_idx, first_token_id):
                self.cache[tuple(prompt_tokens)] = first_token_id

        executor = PrefixCacheExecutor()
        generator = SimpleNamespace(
            model_executor=executor,
            tokenizer=SimpleNamespace(eos_token_id=99),
        )
        backend = module.ContinuousBatchModelBackend(generator)
        first = module.BatchRequest("first", [1, 2, 3], 4, 0.6, 0.9)
        second = module.BatchRequest("second", [1, 2, 3], 4, 0.6, 0.9)

        self.assertEqual(backend.prefill([first]), [40])
        self.assertEqual(backend.prefill([second]), [41])

        self.assertEqual(len(executor.forward_inputs), 2)
        self.assertEqual(executor.cache, {})


if __name__ == "__main__":
    unittest.main()
