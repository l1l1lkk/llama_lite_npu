import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server.py"


class ServerContinuousBatchingContractTest(unittest.TestCase):
    def setUp(self):
        self.source = SERVER.read_text(encoding="utf-8")
        self.module = ast.parse(self.source)

    def test_server_exposes_scheduler_capacity_arguments(self):
        for option in (
            "--continuous_batching",
            "--no_continuous_batching",
            "--max_batch_size",
            "--max_seq_len",
            "--max_waiting_requests",
            "--scheduler_poll_ms",
        ):
            self.assertIn(option, self.source)

    def test_server_passes_max_seq_len_to_generator(self):
        self.assertIn("max_seq_len: int = 1024", self.source)
        self.assertIn("max_seq_len=max_seq_len", self.source)
        self.assertIn("max_seq_len=args.max_seq_len", self.source)
        self.assertIn("Max seq len:", self.source)

    def test_text_endpoints_submit_to_shared_scheduler(self):
        self.assertIn("_submit_continuous_request", self.source)
        self.assertIn("_stream_continuous_chat", self.source)
        self.assertIn("_wait_continuous_chat", self.source)

    def test_server_exposes_prometheus_and_debug_metrics(self):
        self.assertIn('app.get("/metrics")', self.source)
        self.assertIn('app.get("/debug/stats")', self.source)
        self.assertIn("_metrics.render()", self.source)
        self.assertIn("_metrics.snapshot()", self.source)

    def test_server_injects_metrics_and_request_endpoint(self):
        self.assertIn("metrics=_metrics", self.source)
        self.assertIn('endpoint="chat"', self.source)
        self.assertIn('endpoint="completion"', self.source)
        self.assertIn("_sync_runtime_metrics", self.source)

    def test_server_exposes_expert_parallel_mode(self):
        self.assertIn("--moe_parallel_mode", self.source)
        self.assertIn(
            "moe_parallel_mode=args.moe_parallel_mode",
            self.source,
        )

    def test_server_exposes_partial_prefix_cache_switch(self):
        self.assertIn("--partial_prefix_cache", self.source)
        self.assertIn("partial_prefix_cache=args.partial_prefix_cache", self.source)
        self.assertIn("enable_partial_prefix_cache=_partial_prefix_cache", self.source)

    def test_tp_worker_uses_step_level_batch_commands(self):
        self.assertIn("_tp_continuous_worker_loop", self.source)
        self.assertIn("prefill", self.source)
        self.assertIn("decode", self.source)
        self.assertIn("release", self.source)

    def test_continuous_batching_uses_cpu_store_command_channel(self):
        self.assertIn("StoreCommandChannel", self.source)
        start = self.source.index("class _TpCoordinatedContinuousBackend")
        end = self.source.index("def _tp_continuous_worker_loop")
        coordinator_source = self.source[start:end]
        self.assertNotIn("broadcast_object_list", coordinator_source)
        self.assertNotIn("TensorCommandChannel", coordinator_source)

    def test_tp_coordinator_tracks_worker_known_control_ids(self):
        start = self.source.index("class _TpCoordinatedContinuousBackend")
        end = self.source.index("def _tp_continuous_worker_loop")
        coordinator_source = self.source[start:end]
        self.assertIn("_worker_known_control_ids", coordinator_source)
        self.assertIn("prefill state was not mirrored", coordinator_source)
        self.assertIn("prepare_prefill_chunk", coordinator_source)
        self.assertIn("wait_ack", coordinator_source)

    def test_tp_worker_ignores_release_for_unknown_control_ids(self):
        worker_start = self.source.index("def _tp_continuous_worker_loop")
        worker_source = self.source[worker_start:]
        self.assertIn('if command.operation == "release":', worker_source)
        self.assertIn("continue", worker_source)

    def test_tp_worker_acks_success_and_failure(self):
        worker_start = self.source.index("def _tp_continuous_worker_loop")
        worker_source = self.source[worker_start:]
        self.assertIn("receive_with_sequence", worker_source)
        self.assertIn("channel.ack(sequence)", worker_source)
        self.assertIn("channel.ack(sequence, ok=False", worker_source)

    def test_tp_decode_carries_state_snapshot(self):
        start = self.source.index("class _TpCoordinatedContinuousBackend")
        worker_start = self.source.index("def _tp_continuous_worker_loop")
        coordinator_source = self.source[start:worker_start]
        worker_source = self.source[worker_start:]

        self.assertIn("encode_decode_state", coordinator_source)
        self.assertIn('"decode_state"', worker_source)
        self.assertIn("expected_seq_lens", worker_source)
        self.assertIn("TP worker decode_state mismatch", worker_source)

    def test_tp_worker_decode_state_host_validation_is_debug_gated(self):
        worker_start = self.source.index("def _tp_continuous_worker_loop")
        worker_source = self.source[worker_start:]

        self.assertIn("LLAMA_LITE_NPU_VALIDATE_TP_DECODE_STATE", worker_source)
        self.assertIn(
            'if command.operation == "decode_state" and validate_decode_state:',
            worker_source,
        )
        guarded_start = worker_source.index(
            'if command.operation == "decode_state" and validate_decode_state:'
        )
        guarded_end = worker_source.index("backend.decode(worker_requests)")
        guarded_source = worker_source[guarded_start:guarded_end]
        self.assertIn(".cpu()", guarded_source)



    def test_server_exposes_adaptive_chunked_prefill_flags(self):
        server_text = self.source
        self.assertIn("--chunked_prefill_policy", server_text)
        self.assertIn("--chunked_prefill_min_tokens", server_text)
        self.assertIn("chunked_prefill_policy=args.chunked_prefill_policy", server_text)
        self.assertIn("chunked_prefill_min_tokens=args.chunked_prefill_min_tokens", server_text)

if __name__ == "__main__":
    unittest.main()
