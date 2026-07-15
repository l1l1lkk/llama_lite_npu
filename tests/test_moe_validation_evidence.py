"""Fail-closed integrity checks for the tracked MoE validation evidence."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = (
    REPO_ROOT
    / "docs"
    / "validation_data"
    / "20260715_qwen3_30b_a3b_moe_runtime"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(name: str):
    with (EVIDENCE_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(name: str):
    with (EVIDENCE_DIR / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class MoeValidationEvidenceTest(unittest.TestCase):
    def test_manifest_covers_every_payload_with_exact_size_and_hash(self):
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", "docs/validation_data"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8").split("\0")
        evidence_paths = sorted(
            path for path in tracked if path.endswith((".csv", ".json"))
        )
        self.assertEqual(
            evidence_paths,
            [
                "docs/validation_data/20260715_qwen3_30b_a3b_moe_runtime/"
                "layer-oracle-metrics.csv",
                "docs/validation_data/20260715_qwen3_30b_a3b_moe_runtime/"
                "manifest.json",
                "docs/validation_data/20260715_qwen3_30b_a3b_moe_runtime/"
                "router-oracle.csv",
                "docs/validation_data/20260715_qwen3_30b_a3b_moe_runtime/"
                "selected-expert-ownership.csv",
                "docs/validation_data/20260715_qwen3_30b_a3b_moe_runtime/"
                "summary.json",
            ],
        )

        for path in evidence_paths:
            output = subprocess.run(
                ["git", "check-attr", "text", "eol", "--", path],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout.splitlines()
            self.assertEqual(
                output,
                [f"{path}: text: set", f"{path}: eol: lf"],
                path,
            )

        manifest = _load_json("manifest.json")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            manifest["source_commit"],
            "84239ed2e2afd9a277fdc2a5becab8623c1d31c0",
        )
        self.assertEqual(
            manifest["durable_test_commit"],
            "0beba7de621cc98efc22c22c26a54b426b2bc84d",
        )
        self.assertEqual(
            manifest["source_diagnostics_manifest_sha256"],
            {
                "phase4a": (
                    "1b3028449c0eb22335013c2e3be3bb4814d2c63fdc6165f2"
                    "19791efbe471ebe0"
                ),
                "phase4b": (
                    "4304b7c4c8fa744f28853c87cf49e04f60dd64af1a123f62"
                    "a8a6836dee74169e"
                ),
                "phase4b_d1": (
                    "92649b6e54fd616d3a469e43ebb68d2caef661da1a7b214a"
                    "7cf9dedf0a6a843e"
                ),
                "phase4b_d2": (
                    "ba7be8d6741ff1b50e3f8f9fcb12a7e56c0f0a3e024f27"
                    "a6c14e86550c011029"
                ),
                "phase4c2a": (
                    "e94e27be8717f43d20b43651b830933fe61bac23a78d259f"
                    "5bf66bb786317a8c"
                ),
            },
        )

        entries = {entry["path"]: entry for entry in manifest["files"]}
        expected = {
            "layer-oracle-metrics.csv",
            "router-oracle.csv",
            "selected-expert-ownership.csv",
            "summary.json",
        }
        actual = {
            path.name
            for path in EVIDENCE_DIR.iterdir()
            if path.is_file() and path.name != "manifest.json"
        }
        self.assertEqual(set(entries), expected)
        self.assertEqual(actual, expected)
        for name, entry in entries.items():
            path = EVIDENCE_DIR / name
            self.assertEqual(path.stat().st_size, entry["size"], name)
            self.assertEqual(_sha256(path), entry["sha256"], name)

    def test_summary_freezes_model_phases_and_correctness_contract(self):
        summary = _load_json("summary.json")
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["artifact_kind"], "compact derived evidence")
        self.assertEqual(
            summary["source_commit"],
            "84239ed2e2afd9a277fdc2a5becab8623c1d31c0",
        )
        self.assertEqual(
            summary["validation_source_commit"], summary["source_commit"]
        )
        self.assertEqual(
            summary["durable_test_commit"],
            "0beba7de621cc98efc22c22c26a54b426b2bc84d",
        )
        source_hashes = summary["source_manifests"]
        manifest_hashes = _load_json("manifest.json")[
            "source_diagnostics_manifest_sha256"
        ]
        self.assertEqual(
            source_hashes,
            {
                "phase4a_sha256sums_self": manifest_hashes["phase4a"],
                "phase4b_sha256sums_self": manifest_hashes["phase4b"],
                "phase4b_d1_sha256sums_self": manifest_hashes["phase4b_d1"],
                "phase4b_d2_sha256sums_self": manifest_hashes["phase4b_d2"],
                "phase4c2a_sha256sums_self": manifest_hashes["phase4c2a"],
            },
        )
        model = summary["model"]
        self.assertEqual(model["name"], "Qwen3-30B-A3B")
        self.assertEqual(model["model_type"], "qwen3_moe")
        self.assertEqual(
            [model["hidden_size"], model["num_experts"], model["top_k"]],
            [2048, 128, 8],
        )
        self.assertEqual(model["checkpoint_artifact_dtype"], "bfloat16")
        self.assertEqual(model["runtime_dtype"], "float16")
        self.assertEqual(
            model["checkpoint_sha256"],
            "d9456e599c153a1b62c74ce4da9e796dbe58c683c3335358b6258af737b59fe1",
        )
        self.assertEqual(
            model["config_sha256"],
            "2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297",
        )

        phase4a = summary["phase4a"]
        historical = phase4a["historical_suite_before_phase4c1"]
        self.assertEqual(
            [historical["devices"], historical["tests_per_device"]], [2, 5]
        )
        self.assertEqual(
            [historical["failed"], historical["errors"], historical["skipped"]],
            [0, 0, 0],
        )
        self.assertTrue(historical["graph_capture_replay_executed"])
        durable = phase4a["durable_generic_boundary_test"]
        self.assertEqual(durable["expected_suite_tests_per_device"], 6)
        self.assertEqual(durable["server_rerun_status"], "passed_phase4c2a")
        self.assertEqual(
            [
                durable["devices"],
                durable["tests_per_device"],
                durable["passed_per_device"],
            ],
            [2, 6, 6],
        )
        self.assertEqual(
            [durable["failed"], durable["errors"], durable["skipped"]],
            [0, 0, 0],
        )
        self.assertTrue(durable["generic_boundary_executed"])
        self.assertTrue(durable["graph_capture_replay_executed"])

        validator = summary["release_validator"]
        self.assertEqual(validator["commit"], summary["durable_test_commit"])
        self.assertEqual(validator["exit_code"], 0)
        self.assertEqual(validator["discovered_tests"], 161)
        self.assertEqual(validator["skipped"], 1)
        self.assertEqual(
            validator["skip_reason"],
            "matplotlib is required only for the optional matmul visualization benchmark",
        )
        self.assertEqual(validator["npu_test_ids_collected"], 0)
        self.assertEqual(
            validator["npu_suite_execution"], "explicit_separate_commands"
        )
        self.assertIn(
            "The CPU-friendly release validator does not collect tests/npu; "
            "the hardware gate requires explicit per-device NPU suite commands.",
            summary["limitations"],
        )

        phase4b = summary["phase4b"]
        self.assertTrue(phase4b["comparisons"]["A_vs_C_token_exact"])
        self.assertTrue(phase4b["comparisons"]["A_vs_C_text_exact"])
        self.assertFalse(phase4b["comparisons"]["A_vs_B_token_exact"])
        self.assertEqual(
            phase4b["comparisons"]["A_vs_B_first_difference_token_index"],
            15,
        )
        self.assertEqual(phase4b["A"]["tokens"], phase4b["C"]["tokens"])
        self.assertEqual(
            phase4b["A"]["text_sha256"], phase4b["C"]["text_sha256"]
        )
        differences = [
            index
            for index, pair in enumerate(
                zip(phase4b["A"]["tokens"], phase4b["B"]["tokens"])
            )
            if pair[0] != pair[1]
        ]
        self.assertEqual(differences, [15])
        self.assertEqual(
            phase4b["C"]["graph_counters"],
            {"attempts": 1, "captured": 1, "replays": 15, "fallbacks": 0},
        )

        near_tie = summary["phase4b_d1"]["near_tie"]
        self.assertTrue(near_tie["supported"])
        self.assertEqual(near_tie["first_difference_token_index"], 15)
        self.assertEqual(near_tie["winner_margin"], 0.03125)
        self.assertEqual(near_tie["cross_mode_logit_delta"], 0.03125)

        d2 = summary["phase4b_d2"]
        self.assertEqual(d2["classification"], "BOTH_PATHS_REFERENCE_ALIGNED")
        self.assertTrue(d2["independent_oracle"])
        self.assertEqual(d2["forbidden_production_symbol_hits"], 0)
        self.assertEqual(d2["target_layers"], [0, 32, 37, 47])
        self.assertEqual(
            d2["tolerances"]["local_and_post"],
            {"rtol": 0.01, "atol": 0.01},
        )
        self.assertEqual(
            d2["tolerances"]["partition_sum"],
            {"rtol": 0.00001, "atol": 0.000001},
        )
        self.assertEqual(
            d2["gate_counts"],
            {
                "metrics_rows": 72,
                "router_rows": 8,
                "ownership_rows": 128,
                "local": 16,
                "strict_partition_sum": 8,
                "post": 8,
                "router_selected_set_failures": 0,
                "ownership_failures": 0,
                "nonfinite_metric_rows": 0,
            },
        )

    def test_csv_evidence_has_complete_rows_and_all_required_gates(self):
        metrics = _load_csv("layer-oracle-metrics.csv")
        routers = _load_csv("router-oracle.csv")
        ownership = _load_csv("selected-expert-ownership.csv")
        self.assertEqual([len(metrics), len(routers), len(ownership)], [72, 8, 128])

        comparisons = Counter(row["comparison"] for row in metrics)
        self.assertEqual(
            comparisons["captured_local_vs_independent_partial"], 16
        )
        self.assertEqual(
            comparisons["independent_partial_sum_vs_exec_oracle"], 8
        )
        self.assertEqual(comparisons["captured_post_vs_exec_oracle"], 8)
        required = {
            "captured_local_vs_independent_partial",
            "independent_partial_sum_vs_exec_oracle",
            "captured_post_vs_exec_oracle",
        }
        required_rows = [row for row in metrics if row["comparison"] in required]
        self.assertEqual(len(required_rows), 32)
        self.assertTrue(all(row["gate_pass"] == "True" for row in required_rows))
        self.assertEqual(
            sum(
                row["reference_finite"] != "True"
                or row["actual_finite"] != "True"
                for row in metrics
            ),
            0,
        )

        self.assertEqual(
            sum(row["selected_set_equal"] != "True" for row in routers), 0
        )
        self.assertEqual(
            sum(row["boundary_explainable"] != "True" for row in routers), 0
        )
        order_differences = [
            row for row in routers if row["selected_order_equal"] != "True"
        ]
        self.assertEqual(
            [(row["layer"], row["mode"]) for row in order_differences],
            [("32", "B")],
        )

        self.assertEqual(sum(row["ownership_ok"] != "True" for row in ownership), 0)
        self.assertEqual(
            {(row["layer"], row["mode"]) for row in ownership},
            {
                (str(layer), mode)
                for layer in (0, 32, 37, 47)
                for mode in ("A", "B")
            },
        )


if __name__ == "__main__":
    unittest.main()
