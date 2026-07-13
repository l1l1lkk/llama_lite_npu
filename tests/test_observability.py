import unittest
import importlib.util
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

from prometheus_client import CollectorRegistry, generate_latest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "lite_llama" / "observability.py"
)
SPEC = importlib.util.spec_from_file_location(
    "lite_llama_observability_test", MODULE_PATH
)
OBSERVABILITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OBSERVABILITY)
InferenceMetrics = OBSERVABILITY.InferenceMetrics
classify_failure = OBSERVABILITY.classify_failure


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class ObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.registry = CollectorRegistry()
        self.metrics = InferenceMetrics(
            registry=self.registry,
            clock=self.clock,
            include_process_metrics=False,
        )

    def _render(self):
        return generate_latest(self.registry).decode("utf-8")

    def test_failure_categories_are_stable_and_low_cardinality(self):
        cases = (
            (RuntimeError("continuous batching waiting queue is full"), "queue_full"),
            (ValueError("prompt length exceeds model context capacity"), "context_limit"),
            (RuntimeError("Paged KV allocation failed"), "kv_capacity"),
            (RuntimeError("HCCL watchdog timeout"), "hccl"),
            (RuntimeError("ACL stream synchronize failed"), "npu_runtime"),
            (RuntimeError("NPU graph capture failed"), "graph_failure"),
            (RuntimeError("client cancelled request"), "client_cancelled"),
            (ValueError("temperature must be non-negative"), "invalid_request"),
            (RuntimeError("model forward failed"), "model_execution"),
            (Exception("unexpected"), "internal"),
        )
        for error, expected in cases:
            with self.subTest(error=error):
                self.assertEqual(classify_failure(error), expected)

    def test_request_lifecycle_emits_latency_and_token_metrics(self):
        request = SimpleNamespace(
            request_id="chatcmpl-test",
            endpoint="chat",
            prompt_tokens=[1, 2, 3],
            finish_reason=None,
        )
        self.metrics.on_request_submitted(request)
        self.clock.advance(0.25)
        self.metrics.on_request_admitted(request)
        self.clock.advance(0.50)
        self.metrics.on_token(request)
        self.clock.advance(0.10)
        self.metrics.on_token(request)
        self.clock.advance(0.15)
        request.finish_reason = "stop"
        self.metrics.on_request_finished(request)

        text = self._render()
        self.assertIn(
            'lite_llama_requests_total{endpoint="chat",status="success"} 1.0',
            text,
        )
        self.assertIn(
            'lite_llama_prompt_tokens_total{endpoint="chat"} 3.0',
            text,
        )
        self.assertIn(
            'lite_llama_generated_tokens_total{endpoint="chat"} 2.0',
            text,
        )
        self.assertIn(
            'lite_llama_queue_wait_seconds_sum{endpoint="chat"} 0.25',
            text,
        )
        self.assertIn(
            'lite_llama_time_to_first_token_seconds_sum{endpoint="chat"} 0.75',
            text,
        )
        self.assertIn(
            'lite_llama_inter_token_latency_seconds_sum{endpoint="chat"}',
            text,
        )
        self.assertIn(
            'lite_llama_request_latency_seconds_sum{endpoint="chat"} 1.0',
            text,
        )

    def test_optional_request_trace_records_exact_queue_and_ttft(self):
        request = SimpleNamespace(
            request_id="trace-test", endpoint="chat", prompt_tokens=[1],
            finish_reason=None,
        )
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "trace.jsonl"
            self.metrics.configure_request_timing_trace(trace)
            self.metrics.on_request_submitted(request)
            self.clock.advance(0.2)
            self.metrics.on_request_admitted(request)
            self.clock.advance(0.3)
            self.metrics.on_token(request)
            self.clock.advance(0.4)
            self.metrics.on_request_finished(request)
            record = json.loads(trace.read_text(encoding="utf-8"))
        self.assertEqual(record["submission_order"], 1)
        self.assertAlmostEqual(record["queue_wait_ms"], 200.0)
        self.assertAlmostEqual(record["ttft_ms"], 500.0)
        self.assertAlmostEqual(record["service_to_first_token_ms"], 300.0)

    def test_failures_and_rejections_are_attributed(self):
        request = SimpleNamespace(
            request_id="cmpl-test",
            endpoint="completion",
            prompt_tokens=[1],
            finish_reason="error",
        )
        self.metrics.on_request_submitted(request)
        self.clock.advance(0.2)
        self.metrics.on_request_failed(
            request, RuntimeError("Paged KV allocation failed")
        )
        self.metrics.on_request_rejected(
            "chat", RuntimeError("continuous batching waiting queue is full")
        )

        text = self._render()
        self.assertIn(
            'lite_llama_request_failures_total{endpoint="completion",reason="kv_capacity"} 1.0',
            text,
        )
        self.assertIn(
            'lite_llama_request_failures_total{endpoint="chat",reason="queue_full"} 1.0',
            text,
        )

    def test_unknown_endpoint_is_normalized_to_avoid_high_cardinality(self):
        request = SimpleNamespace(
            request_id="arbitrary-endpoint",
            endpoint="tenant-12345",
            prompt_tokens=[1],
            finish_reason="stop",
        )
        self.metrics.on_request_submitted(request)
        self.metrics.on_request_finished(request)

        text = self._render()
        self.assertIn(
            'lite_llama_requests_total{endpoint="unknown",status="success"} 1.0',
            text,
        )
        self.assertNotIn('endpoint="tenant-12345"', text)


    def test_sampling_path_counters_are_exported(self):
        stats = SimpleNamespace(
            candidate_rows=4,
            fallback_rows=1,
            full_logit_gather_batches=1,
        )
        self.metrics.on_sampling_stats(stats)

        text = self._render()
        self.assertIn('lite_llama_sampling_candidate_rows_total 4.0', text)
        self.assertIn('lite_llama_sampling_fallback_rows_total 1.0', text)
        self.assertIn('lite_llama_sampling_full_logit_gather_batches_total 1.0', text)

    def test_scheduler_and_runtime_snapshots_are_exported(self):
        self.metrics.update_scheduler(waiting=3, prefilling=2, running=4)
        executor = SimpleNamespace(
            req_tokens_manager=SimpleNamespace(
                page_mgr=SimpleNamespace(num_pages=100, num_free_pages=40)
            ),
            graph_runner=SimpleNamespace(
                capture_attempt_count=5,
                capture_count=4,
                replay_count=90,
                fallback_count=1,
            ),
        )
        self.metrics.sync_runtime(executor)

        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["scheduler"]["waiting"], 3)
        self.assertEqual(snapshot["kv_cache"]["used_pages"], 60)
        self.assertEqual(snapshot["npu_graph"]["replays"], 90)

        text = self._render()
        self.assertIn("lite_llama_waiting_requests 3.0", text)
        self.assertIn("lite_llama_kv_pages_used 60.0", text)
        self.assertIn("lite_llama_graph_replays_total 90.0", text)

        self.metrics.sync_runtime(executor)
        text = self._render()
        self.assertIn("lite_llama_graph_replays_total 90.0", text)


if __name__ == "__main__":
    unittest.main()
