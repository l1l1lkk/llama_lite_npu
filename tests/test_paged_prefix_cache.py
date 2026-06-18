import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_EXECUTOR = ROOT / "lite_llama" / "executor" / "model_executor.py"


class BlockLevelPrefixCacheContractTest(unittest.TestCase):
    def setUp(self):
        self.source = MODEL_EXECUTOR.read_text(encoding="utf-8")
        self.module = ast.parse(self.source)

    def test_partial_prefix_cache_uses_block_cache_map(self):
        self.assertIn("_paged_block_prefix_cache", self.source)
        self.assertIn("_paged_block_key", self.source)
        self.assertIn("_store_paged_prompt_blocks", self.source)

    def test_partial_prefix_lookup_does_not_scan_full_prompt_cache(self):
        method = next(
            node
            for node in ast.walk(self.module)
            if isinstance(node, ast.FunctionDef)
            and node.name == "share_paged_prefix_from_cache"
        )
        method_source = ast.get_source_segment(self.source, method)
        self.assertNotIn("_paged_prefix_cache.items()", method_source)
        self.assertIn("_paged_block_prefix_cache.get", method_source)


if __name__ == "__main__":
    unittest.main()
