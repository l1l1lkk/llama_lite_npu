import unittest

import torch

from lite_llama.executor.model_executor import _pack_dense_swiglu_state_dict


class PackedSwiGLUStateDictTest(unittest.TestCase):
    def test_gate_and_up_weights_are_packed_in_split_order(self):
        gate = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        up = gate + 100
        state = {
            "layers.0.mlp.gate_proj.weight": gate,
            "layers.0.mlp.up_proj.weight": up,
            "layers.0.mlp.down_proj.weight": torch.ones(4, 3),
        }

        packed = _pack_dense_swiglu_state_dict(state)

        self.assertNotIn("layers.0.mlp.gate_proj.weight", packed)
        self.assertNotIn("layers.0.mlp.up_proj.weight", packed)
        torch.testing.assert_close(
            packed["layers.0.mlp.gate_up_proj.weight"],
            torch.cat((gate, up), dim=0),
        )

    def test_shape_mismatch_is_rejected(self):
        state = {
            "layers.0.mlp.gate_proj.weight": torch.ones(3, 4),
            "layers.0.mlp.up_proj.weight": torch.ones(4, 4),
        }
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            _pack_dense_swiglu_state_dict(state)


if __name__ == "__main__":
    unittest.main()
