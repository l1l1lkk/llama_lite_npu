import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = ROOT / "docs/benchmarks"


def test_current_benchmark_docs_are_utf8_and_local_links_resolve():
    markdown_link = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for path in sorted(DOC_ROOT.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        assert "�" not in text
        for target in markdown_link.findall(text):
            if target.startswith(("http://", "https://", "#")):
                continue
            relative = target.split("#", 1)[0]
            assert (path.parent / relative).resolve().exists(), f"broken link in {path}: {target}"


def test_docs_index_has_one_current_entry_and_legacy_pages_are_demoted():
    index = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    assert len(re.findall(r"^- \[Benchmark 总入口\]\(benchmarks/README\.md\)", index, re.MULTILINE)) == 1
    assert "Historical/unverified" in (ROOT / "docs/vllm_ascend_benchmark.md").read_text(encoding="utf-8")
    assert "historical v1" in (ROOT / "docs/benchmark_results/README.md").read_text(encoding="utf-8")


def test_execution_guide_keeps_bundle_rebuild_and_phase3b_contract():
    text = (DOC_ROOT / "execution.md").read_text(encoding="utf-8")
    for marker in (
        "validate.bundle",
        "--rebuild",
        "manifest.json",
        "omitted.json",
        "runs/<framework>",
        "performance correctness",
        "semantic accuracy",
        "8213",
        "18000",
        "--client-profile",
        "client_preflight_command",
        "__CLIENT_PROFILE_REQUIRED__",
        "cpu_isolated_no_torch",
        "python_prefix",
        "samefile",
        "evalscope.perf.main",
        "EvalScope[perf]",
        "schema v1",
        "request_count=0",
    ):
        assert marker in text
