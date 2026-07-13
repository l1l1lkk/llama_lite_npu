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


if __name__ == "__main__":
    unittest.main()
