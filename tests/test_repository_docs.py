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
        ):
            self.assertIn(heading, english)
        for heading in (
            "## 系统架构",
            "## 快速开始",
            "## 性能数据",
            "## 能力矩阵",
        ):
            self.assertIn(heading, chinese)

    def test_readmes_publish_selected_evalscope_results(self):
        english = (ROOT / "README.md").read_text(encoding="utf-8")
        chinese = (ROOT / "README_CN.md").read_text(encoding="utf-8")

        for value in ("24.7089", "63.3523", "57.7176", "50.158"):
            self.assertIn(value, english)
            self.assertIn(value, chinese)

    def test_release_version_is_synchronized(self):
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        english = (ROOT / "README.md").read_text(encoding="utf-8")
        chinese = (ROOT / "README_CN.md").read_text(encoding="utf-8")

        self.assertEqual(version, "0.0.10rc2")
        self.assertIn("0.0.10rc2", english)
        self.assertIn("0.0.10rc2", chinese)
        self.assertTrue((ROOT / "docs/releases/v0.0.10rc2.md").exists())

    def test_core_modules_have_module_docstrings(self):
        modules = (
            "lite_llama/continuous_batching.py",
            "lite_llama/executor/model_executor.py",
            "lite_llama/executor/npu_graph.py",
            "lite_llama/executor/paged_attention.py",
            "lite_llama/models/qwen3_moe.py",
        )
        for relative_path in modules:
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            docstring = ast.get_docstring(ast.parse(source))
            self.assertIsNotNone(docstring, relative_path)
            self.assertGreaterEqual(len(docstring), 80, relative_path)


if __name__ == "__main__":
    unittest.main()
