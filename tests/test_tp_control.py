import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "lite_llama"
    / "executor"
    / "tp_control.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "lite_llama_tp_control_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorCommandCodecTest(unittest.TestCase):
    def test_prefill_round_trip(self):
        module = load_module()
        requests = [
            SimpleNamespace(
                control_id=4,
                prompt_tokens=[10, 11],
                max_new_tokens=32,
                temperature=0.6,
                top_p=0.9,
            ),
            SimpleNamespace(
                control_id=9,
                prompt_tokens=[20],
                max_new_tokens=16,
                temperature=0.0,
                top_p=1.0,
            ),
        ]

        encoded = module.encode_prefill(requests)
        decoded = module.decode_command(*encoded)

        self.assertEqual(decoded.operation, "prefill")
        self.assertEqual(decoded.control_ids, [4, 9])
        self.assertEqual(decoded.prompt_tokens, [[10, 11], [20]])
        self.assertEqual(decoded.max_new_tokens, [32, 16])
        self.assertEqual(decoded.temperatures, [0.6, 0.0])
        self.assertEqual(decoded.top_ps, [0.9, 1.0])

    def test_decode_release_and_shutdown_round_trip(self):
        module = load_module()

        for operation, encoder in (
            ("decode", module.encode_decode),
            ("release", module.encode_release),
        ):
            decoded = module.decode_command(*encoder([3, 8]))
            self.assertEqual(decoded.operation, operation)
            self.assertEqual(decoded.control_ids, [3, 8])

        shutdown = module.decode_command(*module.encode_shutdown())
        self.assertEqual(shutdown.operation, "shutdown")
        self.assertEqual(shutdown.control_ids, [])


if __name__ == "__main__":
    unittest.main()
