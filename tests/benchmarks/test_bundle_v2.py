import json
import shutil
import subprocess
import sys

from benchmarks.serving.bundle import build_manifest, initialize_bundle, rebuild_aggregate
from benchmarks.serving.validate.bundle import validate_bundle


def _make_bundle(root):
    request_dir = root / "runs/lite_llama/p128_o64_c1/repeat-01/client"
    request_dir.mkdir(parents=True)
    (root / "campaign.json").write_text('{"schema_version":2,"campaign_id":"synthetic"}\n', encoding="utf-8")
    (request_dir / "requests.json").write_text(
        json.dumps(
            [
                {"request_id": "r0", "success": True, "input_tokens": 128, "output_tokens": 64, "e2e_s": 1.0, "ttft_ms": 100.0},
                {"request_id": "r1", "success": True, "input_tokens": 128, "output_tokens": 64, "e2e_s": 1.2, "ttft_ms": 120.0},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "omitted.json").write_text(
        json.dumps([{"kind": "server_log", "path": "/server/server.log", "size_bytes": 1000, "sha256": "a" * 64}]) + "\n",
        encoding="utf-8",
    )
    rebuild_aggregate(root)
    build_manifest(root)


def test_bundle_validates_offline_and_rebuilds_aggregate(tmp_path):
    source = tmp_path / "source"
    _make_bundle(source)
    offline = tmp_path / "offline"
    shutil.copytree(source, offline)

    assert validate_bundle(offline)["status"] == "pass"
    rebuilt = rebuild_aggregate(offline)
    assert rebuilt["request_count"] == 2
    assert rebuilt["success_count"] == 2
    assert rebuilt["mean_output_tokens"] == 64.0


def test_bundle_detects_tampering(tmp_path):
    root = tmp_path / "bundle"
    _make_bundle(root)
    evidence = next(root.glob("runs/**/requests.json"))
    evidence.write_text("[]\n", encoding="utf-8")

    report = validate_bundle(root)
    assert report["status"] == "fail"
    assert report["errors"][0]["kind"] == "sha256_mismatch"


def test_initialize_bundle_creates_canonical_layers(tmp_path):
    root = initialize_bundle(tmp_path / "new", {"schema_version": 2, "campaign_id": "synthetic"})
    assert (root / "campaign.json").is_file()
    assert (root / "environment").is_dir()
    assert (root / "workload").is_dir()
    assert (root / "runs").is_dir()
    assert (root / "derived").is_dir()
    assert (root / "diagnostics/index.json").is_file()
    assert (root / "omitted.json").is_file()


def test_bundle_rejects_malformed_omitted_metadata(tmp_path):
    root = tmp_path / "bundle"
    _make_bundle(root)
    (root / "omitted.json").write_text('[{"kind":"server_log"}]\n', encoding="utf-8")
    build_manifest(root)

    report = validate_bundle(root)
    assert report["status"] == "fail"
    assert {item["kind"] for item in report["errors"]} == {"omitted_entry_invalid"}


def test_empty_manifest_is_structured_failure_and_cli_has_no_traceback(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    report = validate_bundle(root)
    assert report["status"] == "fail"
    assert "manifest_schema_version" in {item["kind"] for item in report["errors"]}

    completed = subprocess.run(
        [sys.executable, "-m", "benchmarks.serving.validate.bundle", str(root)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert json.loads(completed.stdout)["status"] == "fail"
    assert "Traceback" not in completed.stderr


def test_bad_json_and_bom_are_structured_failures(tmp_path):
    for name, content, expected in (
        ("bad", b"{not-json", "manifest_json_invalid"),
        ("bom", b"\xef\xbb\xbf{}", "manifest_bom_not_allowed"),
    ):
        root = tmp_path / name
        root.mkdir()
        (root / "manifest.json").write_bytes(content)
        report = validate_bundle(root)
        assert report["status"] == "fail"
        assert expected in {item["kind"] for item in report["errors"]}


def test_duplicate_path_and_summary_mismatch_are_structured_failures(tmp_path):
    root = tmp_path / "duplicate"
    root.mkdir()
    entry = {"path": "omitted.json", "size_bytes": 3, "sha256": "a" * 64}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "manifest_self_excluded": True,
                "file_count": 99,
                "size_bytes": 999,
                "files": [entry, entry],
            }
        ),
        encoding="utf-8",
    )
    report = validate_bundle(root)
    kinds = {item["kind"] for item in report["errors"]}
    assert report["status"] == "fail"
    assert {"manifest_duplicate_path", "manifest_file_count", "manifest_size_bytes"}.issubset(kinds)


def test_manifest_path_escape_is_rejected_without_reading_outside(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    root = tmp_path / "escape"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "manifest_self_excluded": True,
                "file_count": 1,
                "size_bytes": 6,
                "files": [{"path": "../outside.txt", "size_bytes": 6, "sha256": "x" * 64}],
            }
        ),
        encoding="utf-8",
    )
    report = validate_bundle(root)
    assert report["status"] == "fail"
    assert "manifest_path_invalid" in {item["kind"] for item in report["errors"]}


def test_missing_root_and_non_mapping_manifest_are_structured_failures(tmp_path):
    missing = validate_bundle(tmp_path / "missing")
    assert missing["status"] == "fail"
    assert missing["errors"] == [{"kind": "bundle_root_missing"}]

    root = tmp_path / "list-manifest"
    root.mkdir()
    (root / "manifest.json").write_text("[]\n", encoding="utf-8")
    report = validate_bundle(root)
    assert report["status"] == "fail"
    assert report["errors"] == [{"kind": "manifest_not_mapping"}]


def test_noncanonical_manifest_paths_are_rejected(tmp_path):
    for index, bad_path in enumerate(("C:/outside", "/absolute", "dir\\file", "manifest.json", "a//b")):
        root = tmp_path / f"bad-path-{index}"
        root.mkdir()
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "manifest_self_excluded": True,
                    "file_count": 1,
                    "size_bytes": 0,
                    "files": [{"path": bad_path, "size_bytes": 0, "sha256": "a" * 64}],
                }
            ),
            encoding="utf-8",
        )
        report = validate_bundle(root)
        assert report["status"] == "fail"
        assert "manifest_path_invalid" in {item["kind"] for item in report["errors"]}
