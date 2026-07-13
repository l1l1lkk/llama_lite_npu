import hashlib
import importlib.util
import json
import sqlite3
import sys
import tempfile
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


class RuntimeSamplerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load("runtime_sampler_test", "sample_runtime_timeseries.py")

    def test_parse_npu_fields(self):
        parsed = self.module.parse_npu(
            "HBM Usage Rate(%) : 12\nAicore Usage Rate(%) : 73\n"
            "Aivector Usage Rate(%) : 4\nNPU Utilization(%) : 81\n"
        )
        self.assertEqual(parsed["hbm_usage_pct"], 12.0)
        self.assertEqual(parsed["npu_utilization_pct"], 81.0)


class RequestMetricExtractorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load("request_metric_extractor_test", "extract_request_metrics.py")

    def test_compact_request_metric_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            database = root / "benchmark_data.db"
            connection = sqlite3.connect(database)
            connection.execute(
                "create table result(request text,start_time real,inter_token_latencies text,"
                "success integer,response_messages text,completed_time real,latency real,"
                "first_chunk_latency real,prompt_tokens integer,completion_tokens integer,"
                "max_gpu_memory_cost real,time_per_output_token real)"
            )
            connection.execute(
                "insert into result values(?,?,?,?,?,?,?,?,?,?,?,?)",
                (json.dumps({"messages": []}), 1.0, "[0.1,0.2]", 1, "", 2.0,
                 1.0, 0.25, 128, 256, 0.0, 0.003),
            )
            connection.commit()
            connection.close()
            report = self.module.extract(root)
            self.assertEqual(report["request_count"], 1)
            self.assertEqual(report["requests"][0]["ttft_ms"], 250.0)
            self.assertAlmostEqual(report["requests"][0]["itl_mean_ms"], 150.0)


class ServerRequestTraceExtractorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load("server_request_trace_extractor_test", "extract_server_request_trace.py")

    def test_extracts_last_formal_window_in_submission_order(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "trace.jsonl"
            records = [
                {"submission_order": index, "status": "success", "queue_wait_ms": index,
                 "ttft_ms": index + 10, "service_to_first_token_ms": 10, "e2e_ms": 20}
                for index in range(1, 7)
            ]
            trace.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            report = self.module.extract(trace, 4)
        self.assertEqual([item["submission_order"] for item in report["requests"]], [3, 4, 5, 6])
        self.assertEqual([item["formal_submission_order"] for item in report["requests"]], [1, 2, 3, 4])


class ScalabilityAnalyzerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load("scalability_analyzer_test", "analyze_scalability.py")

    def test_quantile_interpolates(self):
        self.assertEqual(self.module.quantile([1, 2, 3], 0.5), 2)
        self.assertEqual(self.module.quantile([1, 3], 0.5), 2)

    def test_histogram_quantile_interpolates_bucket(self):
        histogram = {
            "count": 10,
            "sum": 5,
            "buckets": {"0.5": 5, "1.0": 10, "+Inf": 10},
        }
        self.assertEqual(self.module.histogram_quantile(histogram, 0.5), 0.5)
        self.assertEqual(self.module.histogram_quantile(histogram, 0.75), 0.75)


class ScalabilityContractTest(unittest.TestCase):
    def test_workload_manifest_and_hashes(self):
        root = BENCHMARK / "datasets" / "20260713_scalability"
        manifest = json.loads((root / "workload-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["workload_id"], "qwen3_32b_p128_o256_n32_scalability_v1")
        for output in manifest["outputs"]:
            path = root / output["path"]
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 32)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), output["sha256"])

    def test_run_and_lifecycle_scripts_keep_evidence(self):
        run = (BENCHMARK / "run_case.sh").read_text(encoding="utf-8")
        start = (BENCHMARK / "start_server.sh").read_text(encoding="utf-8")
        stop = (BENCHMARK / "stop_server.sh").read_text(encoding="utf-8")
        for value in (
            "sample_runtime_timeseries.py",
            "extract_request_metrics.py",
            "run-timing.json",
            "timeseries-exit-code.txt",
            "server_lifecycle_id",
        ):
            self.assertIn(value, run)
        self.assertIn("SERVER_LIFECYCLE_ID", start)
        self.assertIn("SERVER_LIFECYCLE_ID", stop)
        self.assertIn("DECODE_PRIORITY_MODE", start)
        self.assertIn("RESULT_NAMESPACE", run)
        self.assertIn("extract_server_request_trace.py", run)

    def test_campaign_plan_is_fixed_and_complete(self):
        plan = json.loads(
            (BENCHMARK / "campaigns/20260713_graph_scalability.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(plan["sequence_locked_before_remaining_formal_runs"])
        self.assertEqual(len(plan["sequence"]), 15)
        self.assertEqual(
            {(item["concurrency"], item["repeat"]) for item in plan["sequence"]},
            {(concurrency, repeat) for concurrency in (1, 2, 4, 8, 16) for repeat in (1, 2, 3)},
        )

    def test_decode_priority_campaign_plan_is_paired_and_interleaved(self):
        plan = json.loads(
            (BENCHMARK / "campaigns/20260713_decode_priority_ablation.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(plan["sequence_locked_before_formal_runs"])
        self.assertEqual(len(plan["sequence"]), 30)
        observed = {
            (item["concurrency"], item["repeat"], item["mode"])
            for item in plan["sequence"]
        }
        expected = {
            (concurrency, repeat, mode)
            for concurrency in (1, 2, 4, 8, 16)
            for repeat in (1, 2, 3)
            for mode in ("priority_on", "priority_off")
        }
        self.assertEqual(observed, expected)
        for concurrency in (1, 2, 4, 8, 16):
            first_modes = []
            for repeat in (1, 2, 3):
                pair = sorted(
                    (item for item in plan["sequence"] if item["concurrency"] == concurrency and item["repeat"] == repeat),
                    key=lambda item: item["order"],
                )
                first_modes.append(pair[0]["mode"])
            self.assertNotEqual(first_modes[0], first_modes[1])


if __name__ == "__main__":
    unittest.main()
