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
            "--max_waiting_requests",
            "--scheduler_poll_ms",
        ):
            self.assertIn(option, self.source)

    def test_text_endpoints_submit_to_shared_scheduler(self):
        self.assertIn("_submit_continuous_request", self.source)
        self.assertIn("_stream_continuous_chat", self.source)
        self.assertIn("_wait_continuous_chat", self.source)

    def test_server_exposes_expert_parallel_mode(self):
        self.assertIn("--moe_parallel_mode", self.source)
        self.assertIn(
            "moe_parallel_mode=args.moe_parallel_mode",
            self.source,
        )

    def test_tp_worker_uses_step_level_batch_commands(self):
        self.assertIn("_tp_continuous_worker_loop", self.source)
        self.assertIn("prefill", self.source)
        self.assertIn("decode", self.source)
        self.assertIn("release", self.source)

    def test_continuous_batching_uses_tensor_command_channel(self):
        self.assertIn("TensorCommandChannel", self.source)
        start = self.source.index("class _TpCoordinatedContinuousBackend")
        end = self.source.index("def _tp_continuous_worker_loop")
        coordinator_source = self.source[start:end]
        self.assertNotIn("broadcast_object_list", coordinator_source)


if __name__ == "__main__":
    unittest.main()
