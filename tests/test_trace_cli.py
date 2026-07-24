import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "lite_llama" / "trace_cli.py"
SPEC = importlib.util.spec_from_file_location(
    "lite_llama_trace_cli_test", MODULE_PATH
)
TRACE_CLI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACE_CLI
SPEC.loader.exec_module(TRACE_CLI)


class TraceCliTests(unittest.TestCase):
    def test_layer_run_disables_graph_by_default(self):
        with patch.object(TRACE_CLI.sys, "executable", "python"):
            command = TRACE_CLI.build_server_command(
                trace_level="layer",
                trace_output="trace.jsonl",
                trace_buffer_events=1234,
                server_args=(
                    "--",
                    "--checkpoints_dir",
                    "/models/qwen",
                    "--port",
                    "8213",
                ),
            )
        self.assertIn("--trace", command)
        self.assertIn("--trace_level", command)
        self.assertIn("--no_compiled_model", command)
        self.assertIn("--trace_output", command)
        self.assertIn("trace.jsonl", command)

    def test_explicit_graph_choice_is_preserved(self):
        command = TRACE_CLI.build_server_command(
            trace_level="layer",
            trace_output=None,
            trace_buffer_events=50_000,
            server_args=(
                "--checkpoints_dir",
                "/models/qwen",
                "--compiled_model",
            ),
        )
        self.assertIn("--compiled_model", command)
        self.assertNotIn("--no_compiled_model", command)

    def test_checkpoint_argument_is_required(self):
        with self.assertRaisesRegex(ValueError, "--checkpoints_dir"):
            TRACE_CLI.build_server_command(
                trace_level="request",
                trace_output=None,
                trace_buffer_events=50_000,
                server_args=("--port", "8213"),
            )

    def test_viewer_is_self_contained_and_exposes_live_controls(self):
        html = (
            ROOT / "lite_llama" / "trace_ui" / "index.html"
        ).read_text(encoding="utf-8")
        self.assertIn("LiteLlama Trace", html)
        self.assertIn("new EventSource", html)
        self.assertIn("trace/snapshot", html)
        self.assertIn('id="timeline"', html)
        self.assertIn('id="layers"', html)
        self.assertNotIn("<script src=", html)


if __name__ == "__main__":
    unittest.main()
