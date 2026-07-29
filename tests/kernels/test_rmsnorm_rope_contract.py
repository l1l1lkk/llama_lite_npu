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

    def test_fused_backend_uses_one_triton_kernel(self):
        init_source = (ROOT / "lite_llama/kernels/__init__.py").read_text()
        self.assertIn(
            "from .rmsnorm_rope_fused import qk_rmsnorm_rope_forward",
            init_source,
        )
        source = (ROOT / "lite_llama/kernels/rmsnorm_rope_fused.py").read_text()
        tree = ast.parse(source)
        imports = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        functions = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        self.assertIn("triton", imports)
        self.assertTrue(
            any(node.name == "_qk_rmsnorm_rope_kernel" for node in functions)
        )
        self.assertNotIn("skip_rmsnorm(", source)
        self.assertNotIn("rope_emb_forward(", source)

    def test_shape_matrix_covers_requested_axes(self):
        source = (ROOT / "benchmarks/rmsnorm_rope/benchmark.py").read_text()
        tree = ast.parse(source)
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertTrue({"batch", "sequence", "head_dimension"} <= constants)

    def test_fused_grid_limit_is_per_token(self):
        source = (
            ROOT / "lite_llama/executor/model_executor.py"
        ).read_text(encoding="utf-8")
        method = source.split("def _infer_safe_prefill_tokens", 1)[1].split(
            "def _get_max_avaliable_tokens", 1
        )[0]
        self.assertIn("return safe_grid_rows", method)
        self.assertNotIn("safe_grid_rows // local_q_heads", method)


if __name__ == "__main__":
    unittest.main()
