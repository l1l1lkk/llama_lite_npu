import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "qwen3_32b_tp2_fp16"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BENCHMARK / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class WorkloadFingerprintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load(
            "workload_fingerprint_test", "extract_workload_fingerprint.py"
        )

    def test_canonical_hash_is_key_order_independent(self):
        self.assertEqual(
            self.module.canonical_sha256({"a": 1, "b": 2}),
            self.module.canonical_sha256({"b": 2, "a": 1}),
        )

    def test_last_user_text_matches_server_behavior(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "last"},
        ]
        self.assertEqual(self.module.last_user_text(messages), "last")


class GraphComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load("graph_comparison_test", "compare_graph.py")

    def test_server_command_normalization_changes_only_graph_flag(self):
        on = "python server.py --page_size 16 --compiled_model --continuous_batching"
        off = "python server.py --page_size 16 --no_compiled_model --continuous_batching"
        self.assertEqual(
            self.module.normalized_server_command(on),
            self.module.normalized_server_command(off),
        )
        changed = off.replace("--page_size 16", "--page_size 32")
        self.assertNotEqual(
            self.module.normalized_server_command(on),
            self.module.normalized_server_command(changed),
        )

    def test_latency_and_throughput_ratios_have_explicit_directions(self):
        latency = self.module.metric_effect(8.0, 10.0, "latency")
        throughput = self.module.metric_effect(12.0, 10.0, "throughput")
        self.assertEqual(latency, (1.25, 20.0, "off/on"))
        self.assertEqual(throughput, (1.2, 20.0, "on/off"))


class GraphAblationScriptContractTest(unittest.TestCase):
    def test_run_script_separates_warmup_from_formal_metrics(self):
        source = (BENCHMARK / "run_case.sh").read_text(encoding="utf-8")
        self.assertIn("SEPARATE_WARMUP", source)
        self.assertIn("WARMUP_DATASET_OFFSET_OVERRIDE", source)
        self.assertIn("pre-warmup-stats.json", source)
        self.assertIn("FORMAL_WARMUP=0", source)
        self.assertIn("workload-fingerprint.json", source)

    def test_bundle_keeps_pairing_evidence(self):
        source = (BENCHMARK / "build_bundle.py").read_text(encoding="utf-8")
        for evidence in (
            "pair-validation.json",
            "paired-ratios.csv",
            "graph-comparison.csv",
            "pre-warmup-stats.json",
            "workload-fingerprint.json",
        ):
            self.assertIn(evidence, source)


if __name__ == "__main__":
    unittest.main()
