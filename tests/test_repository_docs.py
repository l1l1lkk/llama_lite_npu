import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryDocumentationTests(unittest.TestCase):
    def test_readmes_have_required_github_sections(self):
        english = (ROOT / "README.md").read_text(encoding="utf-8")
        chinese = (ROOT / "README_CN.md").read_text(encoding="utf-8")

        self.assertIn("[中文](README_CN.md)", english)
        for heading in (
            "## Architecture",
            "## Quick Start",
            "## Benchmarks",
            "## Capability Matrix",
            "## Observability",
        ):
            self.assertIn(heading, english)
        for heading in (
            "## 架构概览",
            "## 快速启动",
            "## 性能数据",
            "## 能力矩阵",
            "## 可观测性",
        ):
            self.assertIn(heading, chinese)

    def test_readmes_publish_selected_evalscope_results(self):
        english = (ROOT / "README.md").read_text(encoding="utf-8")
        chinese = (ROOT / "README_CN.md").read_text(encoding="utf-8")

        for value in (
            "24.7089",
            "63.3523",
            "57.7176",
            "50.158",
            "10.2149",
            "49.6692",
            "4.862x",
        ):
            self.assertIn(value, english)
            self.assertIn(value, chinese)

    def test_release_version_is_synchronized(self):
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        english = (ROOT / "README.md").read_text(encoding="utf-8")
        chinese = (ROOT / "README_CN.md").read_text(encoding="utf-8")

        self.assertEqual(version, "0.0.15rc2")
        self.assertIn("0.0.15rc2", english)
        self.assertIn("0.0.15rc2", chinese)
        self.assertTrue((ROOT / "docs/releases/v0.0.15rc2.md").exists())
        self.assertTrue((ROOT / "docs/releases/v0.0.15rc1.md").exists())
        self.assertTrue((ROOT / "docs/releases/v0.0.14rc1.md").exists())
        self.assertTrue((ROOT / "docs/releases/v0.0.13rc3.md").exists())
        self.assertTrue((ROOT / "docs/releases/v0.0.13rc2.md").exists())
        self.assertTrue((ROOT / "docs/releases/v0.0.11rc1.md").exists())
        self.assertTrue((ROOT / "docs/observability.md").exists())

    def test_versioning_distinguishes_release_impact(self):
        versioning = (ROOT / "docs/versioning.md").read_text(encoding="utf-8")
        self.assertIn("功能大更新", versioning)
        self.assertIn("小功能或 Bug 修复", versioning)
        self.assertIn("0.0.13rc2 -> 0.0.13rc3", versioning)
        self.assertIn("benchmark 数据与测试脚本", versioning)
        self.assertIn("release/<VERSION>", versioning)
        self.assertIn("v<VERSION>", versioning)
        self.assertIn("annotated tag", versioning)

        release = (ROOT / "docs/releases/v0.0.15rc2.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("release/0.0.15rc2", release)
        self.assertIn("v0.0.15rc2", release)

    def test_core_modules_have_module_docstrings(self):
        modules = (
            "lite_llama/continuous_batching.py",
            "lite_llama/executor/model_executor.py",
            "lite_llama/executor/npu_graph.py",
            "lite_llama/executor/paged_attention.py",
            "lite_llama/models/qwen3_moe.py",
            "lite_llama/observability.py",
        )
        for relative_path in modules:
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            docstring = ast.get_docstring(ast.parse(source))
            self.assertIsNotNone(docstring, relative_path)
            self.assertGreaterEqual(len(docstring), 80, relative_path)


if __name__ == "__main__":
    unittest.main()
