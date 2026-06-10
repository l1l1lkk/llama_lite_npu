import importlib.util
import ast
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


model_config = _load_module(
    "qwen3_moe_model_config", "lite_llama/models/model_config.py"
)
tp_utils = _load_module("qwen3_moe_tp_utils", "lite_llama/executor/tp_utils.py")


class Qwen3MoeConfigTest(unittest.TestCase):
    def test_official_config_fields_are_mapped(self):
        config = model_config.Qwen3MoeConfig.from_dict(
            {
                "model_type": "qwen3_moe",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 48,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 768,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "norm_topk_prob": True,
            }
        )

        self.assertEqual(config.model_type, "qwen3_moe")
        self.assertEqual(config.num_layers, 48)
        self.assertEqual(config.num_heads, 32)
        self.assertEqual(config.num_kv_heads, 4)
        self.assertEqual(config.num_experts, 128)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.moe_intermediate_size, 768)
        self.assertTrue(config.norm_topk_prob)

    def test_invalid_tensor_parallel_divisibility_is_rejected(self):
        config = model_config.Qwen3MoeConfig(
            num_heads=32,
            num_kv_heads=4,
            moe_intermediate_size=768,
        )

        with self.assertRaisesRegex(ValueError, "num_kv_heads"):
            config.validate_tensor_parallel(8)


class Qwen3MoeTensorParallelTest(unittest.TestCase):
    def test_gate_and_up_are_sharded_separately_before_recombining(self):
        # Layout is [gate rows..., up rows...].
        weight = torch.arange(2 * 8 * 3).reshape(1, 16, 3)
        rank0 = tp_utils.shard_moe_gate_up(weight, intermediate_size=8, tp=tp_utils.TPConfig(2, 0))
        rank1 = tp_utils.shard_moe_gate_up(weight, intermediate_size=8, tp=tp_utils.TPConfig(2, 1))

        self.assertEqual(rank0.shape, (1, 8, 3))
        self.assertTrue(torch.equal(rank0[:, :4], weight[:, 0:4]))
        self.assertTrue(torch.equal(rank0[:, 4:], weight[:, 8:12]))
        self.assertTrue(torch.equal(rank1[:, :4], weight[:, 4:8]))
        self.assertTrue(torch.equal(rank1[:, 4:], weight[:, 12:16]))

    def test_down_projection_is_sharded_on_intermediate_axis(self):
        weight = torch.arange(2 * 3 * 8).reshape(2, 3, 8)
        rank1 = tp_utils.shard_moe_down(weight, tp_utils.TPConfig(2, 1))

        self.assertEqual(rank1.shape, (2, 3, 4))
        self.assertTrue(torch.equal(rank1, weight[:, :, 4:8]))


class Qwen3MoeExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.moe = _load_module("qwen3_moe_layers", "lite_llama/models/moe.py")

    def test_router_returns_normalized_topk_probabilities(self):
        router = self.moe.Qwen3MoeTopKRouter(
            hidden_size=3,
            num_experts=4,
            top_k=2,
            norm_topk_prob=True,
            dtype=torch.float32,
        )
        router.weight.data.copy_(
            torch.tensor(
                [
                    [2.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )

        logits, weights, indices = router(torch.tensor([[1.0, 0.0, 0.0]]))

        self.assertEqual(logits.shape, (1, 4))
        self.assertEqual(indices.tolist(), [[0, 1]])
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))

    def test_sparse_block_matches_explicit_expert_reference(self):
        block = self.moe.Qwen3SparseMoeBlock(
            hidden_size=2,
            num_experts=2,
            top_k=1,
            intermediate_size=2,
            norm_topk_prob=True,
            dtype=torch.float32,
        )
        block.gate.weight.data.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        block.experts.gate_up_weight.data.copy_(
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
                    [[0.5, 0.0], [0.0, 0.5], [1.0, 0.0], [0.0, 1.0]],
                ]
            )
        )
        block.experts.down_weight.data.copy_(
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[2.0, 0.0], [0.0, 2.0]],
                ]
            )
        )
        inputs = torch.tensor([[[2.0, 1.0], [-2.0, 1.0]]])

        output = block(inputs)

        expected_rows = []
        for token, expert_id in zip(inputs.view(-1, 2), (0, 1)):
            gate_up = torch.nn.functional.linear(
                token, block.experts.gate_up_weight[expert_id]
            )
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = torch.nn.functional.silu(gate) * up
            expected_rows.append(
                torch.nn.functional.linear(
                    hidden, block.experts.down_weight[expert_id]
                )
            )
        expected = torch.stack(expected_rows).view_as(inputs)
        torch.testing.assert_close(output, expected)


class Qwen3MoeSwiGLUStrideTest(unittest.TestCase):
    def test_kernel_accepts_independent_input_and_output_row_strides(self):
        source = (ROOT / "lite_llama/kernels/swiglu.py").read_text()
        module = ast.parse(source)
        kernel = next(
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_swiglu_forward_kernel"
        )
        argument_names = [arg.arg for arg in kernel.args.args]

        self.assertIn("a_row_stride", argument_names)
        self.assertIn("b_row_stride", argument_names)
        self.assertIn("c_row_stride", argument_names)


class Qwen3MoeWeightConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = _load_module(
            "qwen3_moe_weight_utils",
            "lite_llama/utils/qwen3_moe_weights.py",
        )

    def test_experts_are_stacked_in_numeric_order_and_gate_up_are_fused(self):
        state = {
            "model.layers.0.mlp.gate.weight": torch.full((2, 3), 9.0),
        }
        for expert_id in range(2):
            prefix = f"model.layers.0.mlp.experts.{expert_id}"
            state[f"{prefix}.gate_proj.weight"] = torch.full(
                (2, 3), 10.0 + expert_id
            )
            state[f"{prefix}.up_proj.weight"] = torch.full(
                (2, 3), 20.0 + expert_id
            )
            state[f"{prefix}.down_proj.weight"] = torch.full(
                (3, 2), 30.0 + expert_id
            )

        converted = self.weights.stack_qwen3_moe_weights(
            state, num_layers=1, num_experts=2
        )

        self.assertTrue(
            torch.equal(
                converted["layers.0.mlp.gate.weight"],
                state["model.layers.0.mlp.gate.weight"],
            )
        )
        gate_up = converted["layers.0.mlp.experts.gate_up_weight"]
        down = converted["layers.0.mlp.experts.down_weight"]
        self.assertEqual(gate_up.shape, (2, 4, 3))
        self.assertEqual(down.shape, (2, 3, 2))
        self.assertTrue(torch.all(gate_up[0, :2] == 10.0))
        self.assertTrue(torch.all(gate_up[0, 2:] == 20.0))
        self.assertTrue(torch.all(gate_up[1, :2] == 11.0))
        self.assertTrue(torch.all(gate_up[1, 2:] == 21.0))

    def test_missing_expert_weight_raises_clear_error(self):
        state = {
            "model.layers.0.mlp.gate.weight": torch.zeros(2, 3),
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(2, 3),
            "model.layers.0.mlp.experts.0.up_proj.weight": torch.zeros(2, 3),
        }

        with self.assertRaisesRegex(
            KeyError, "layer 0 expert 0.*down_proj"
        ):
            self.weights.stack_qwen3_moe_weights(
                state, num_layers=1, num_experts=1
            )

    def test_conversion_can_release_source_expert_tensors(self):
        state = {
            "model.layers.0.mlp.gate.weight": torch.zeros(1, 1),
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(1, 1),
            "model.layers.0.mlp.experts.0.up_proj.weight": torch.zeros(1, 1),
            "model.layers.0.mlp.experts.0.down_proj.weight": torch.zeros(1, 1),
        }

        self.weights.stack_qwen3_moe_weights(
            state, num_layers=1, num_experts=1, consume=True
        )

        self.assertNotIn(
            "model.layers.0.mlp.experts.0.gate_proj.weight", state
        )
        self.assertNotIn("model.layers.0.mlp.gate.weight", state)


if __name__ == "__main__":
    unittest.main()
