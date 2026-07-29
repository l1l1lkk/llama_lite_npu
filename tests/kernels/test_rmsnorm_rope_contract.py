import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class RMSNormRoPEContractTest(unittest.TestCase):
    def test_qwen3_uses_branch_selected_backend(self):
        source = (ROOT / "lite_llama/models/qwen3.py").read_text()
        self.assertIn("qk_rmsnorm_rope_forward(", source)
        self.assertNotIn("xq, _ = skip_rmsnorm(xq", source)

    def test_baseline_keeps_three_existing_launches(self):
        source = (
            ROOT / "lite_llama/kernels/rmsnorm_rope_unfused.py"
        ).read_text()
        tree = ast.parse(source)
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("skip_rmsnorm", called)
        self.assertIn("rope_emb_forward", called)

    def test_shape_matrix_covers_requested_axes(self):
        source = (ROOT / "benchmarks/rmsnorm_rope/benchmark.py").read_text()
        tree = ast.parse(source)
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertTrue({"batch", "sequence", "head_dimension"} <= constants)


if __name__ == "__main__":
    unittest.main()
