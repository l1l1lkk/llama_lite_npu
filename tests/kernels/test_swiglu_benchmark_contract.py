import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class SwiGLUBenchmarkContractTest(unittest.TestCase):
    def test_fused_backend_is_selected(self):
        source = (ROOT / "lite_llama/kernels/__init__.py").read_text()
        self.assertIn(
            "from .swiglu_fused import swiglu_forward, swiglu_packed_forward",
            source,
        )

    def test_fused_backend_uses_triton_kernel(self):
        source = (ROOT / "lite_llama/kernels/swiglu_fused.py").read_text()
        tree = ast.parse(source)
        imports = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        attributes = [
            node for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        ]
        functions = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        self.assertIn("triton", imports)
        self.assertTrue(
            any(node.name == "_swiglu_packed_kernel" for node in functions)
        )
        self.assertFalse(any(node.attr == "npu_swiglu" for node in attributes))

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
