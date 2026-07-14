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

    def test_length_matrix_plan_and_workloads_are_complete(self):
        plan = json.loads(
            (BENCHMARK / "campaigns/20260714_length_matrix.json").read_text(encoding="utf-8")
        )
        self.assertTrue(plan["sequence_locked_before_formal_runs"])
        self.assertEqual(len(plan["sequence"]), 36)
        self.assertEqual(
            {(item["prompt"], item["output"], item["repeat"]) for item in plan["sequence"]},
            {(prompt, output, repeat) for prompt in (128, 512, 1024, 2048)
             for output in (64, 256, 512) for repeat in (1, 2, 3)},
        )
        watchdog = plan["collection_watchdog"]
        self.assertEqual(watchdog["stall_no_progress_seconds"], 1200)
        self.assertEqual(watchdog["hard_review_seconds"], 21600)
        self.assertEqual(watchdog["applies_from_order"], 2)
        self.assertIn("do not automatically terminate", watchdog["hard_review_action"])
        runner = (BENCHMARK / "run_length_matrix_campaign.sh").read_text(encoding="utf-8")
        for signal in ("success", "generated", "replays"):
            self.assertIn(signal, runner)
        self.assertIn("no_progress_seconds >= stall_no_progress_seconds", runner)
        self.assertIn("HARD_REVIEW_REQUIRED", runner)
        self.assertNotIn("timeout --signal", runner)
        root = BENCHMARK / "datasets/20260714_length_matrix"
        manifest = json.loads((root / "workload-manifest.json").read_text(encoding="utf-8"))
        for workload in manifest["workloads"]:
            for kind, count in (("formal", 32), ("warmup", 8)):
                record = workload[kind]
                path = root / record["path"]
                self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), count)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])


class PerformanceBaselineCompareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load("performance_baseline_compare_test", "compare_performance_baseline.py")

    @staticmethod
    def fixture():
        metrics = {
            "client_ttft_ms": {"mean": 100.0, "stdev": 2.0, "cv": 0.02},
            "client_tpot_ms": {"mean": 10.0, "stdev": 0.2, "cv": 0.02},
            "client_output_throughput_tok_s": {"mean": 50.0, "stdev": 1.0, "cv": 0.02},
        }
        return {
            "fingerprint": {"hardware": "same", "workload": "same"},
            "cells": {"p128_o256": {"gates": {
                "strict_input": True, "strict_output": True,
                "zero_failed": True, "graph_valid": True,
            }, "metrics": metrics}},
        }

    def test_self_compare_passes(self):
        fixture = self.fixture()
        self.assertEqual(self.module.compare(fixture, fixture)["status"], "pass")

    def test_regression_and_invalid_are_separate(self):
        baseline = self.fixture()
        candidate = json.loads(json.dumps(baseline))
        candidate["cells"]["p128_o256"]["metrics"]["client_ttft_ms"]["mean"] = 120
        self.assertEqual(self.module.compare(baseline, candidate)["status"], "regression")
        candidate = json.loads(json.dumps(baseline))
        candidate["fingerprint"]["hardware"] = "different"
        self.assertEqual(self.module.compare(baseline, candidate)["status"], "invalid")

    def test_direction_and_cv_thresholds(self):
        baseline = self.fixture()
        baseline["cells"]["p128_o256"]["metrics"]["client_tpot_ms"]["cv"] = 0.05
        candidate = json.loads(json.dumps(baseline))
        candidate["cells"]["p128_o256"]["metrics"]["client_tpot_ms"]["mean"] = 11.4
        self.assertEqual(self.module.compare(baseline, candidate)["status"], "pass")
        candidate["cells"]["p128_o256"]["metrics"]["client_tpot_ms"]["mean"] = 11.6
        self.assertEqual(self.module.compare(baseline, candidate)["status"], "regression")
        candidate = json.loads(json.dumps(baseline))
        candidate["cells"]["p128_o256"]["metrics"]["client_output_throughput_tok_s"]["mean"] = 44
        self.assertEqual(self.module.compare(baseline, candidate)["status"], "regression")

    def test_correctness_failure_is_hard_fail(self):
        baseline = self.fixture()
        candidate = json.loads(json.dumps(baseline))
        candidate["cells"]["p128_o256"]["gates"]["graph_valid"] = False
        self.assertEqual(self.module.compare(baseline, candidate)["status"], "hard_fail")


class LengthMatrixAnalyzerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(BENCHMARK))
        try:
            cls.module = load("length_matrix_analyzer_test", "analyze_length_matrix.py")
        finally:
            sys.path.pop(0)

    def test_timeseries_distributions_keep_scheduler_kv_and_npu_samples(self):
        records = [
            {
                "server": {
                    "status": "ok",
                    "scheduler": {"waiting": 1, "prefilling": 2, "running": 3},
                    "kv_cache": {"used_pages": 7},
                },
                "npu": {
                    "6": {"status": "ok", "npu_utilization_pct": 80, "hbm_usage_pct": 90},
                    "7": {"status": "ok", "npu_utilization_pct": 82, "hbm_usage_pct": 91},
                },
            }
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "timeseries.jsonl"
            path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
            values = self.module.timeseries_distributions(path)
        self.assertEqual(values["system_requests"], [6.0])
        self.assertEqual(values["kv_used_pages"], [7.0])
        self.assertEqual(values["npu_utilization_pct"], [80.0, 82.0])
        self.assertEqual(values["npu_hbm_usage_pct"], [90.0, 91.0])

    def test_prompt_multiset_digest_ignores_concurrent_completion_order(self):
        first = {
            "requests": [
                {"prompt_token_ids_sha256": "prompt-a"},
                {"prompt_token_ids_sha256": "prompt-b"},
            ]
        }
        second = {"requests": list(reversed(first["requests"]))}
        self.assertEqual(
            self.module.prompt_multiset_digest(first),
            self.module.prompt_multiset_digest(second),
        )


if __name__ == "__main__":
    unittest.main()
