import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class SwiGLUBenchmarkContractTest(unittest.TestCase):
    def test_unfused_backend_is_selected(self):
        source = (ROOT / "lite_llama/kernels/__init__.py").read_text()
        self.assertIn(
            "from .swiglu_unfused import swiglu_forward",
            source,
        )

    def test_unfused_reference_uses_fp32_intermediates(self):
        source = (ROOT / "lite_llama/kernels/swiglu_unfused.py").read_text()
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertGreaterEqual(
            sum(call.func.attr == "float" for call in calls),
            2,
        )
        self.assertTrue(any(call.func.attr == "sigmoid" for call in calls))

    def test_shape_matrix_covers_requested_axes(self):
        source = (ROOT / "benchmarks/swiglu/benchmark.py").read_text()
        tree = ast.parse(source)
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertTrue({"batch", "sequence", "feature_dimension"} <= constants)


if __name__ == "__main__":
    unittest.main()
