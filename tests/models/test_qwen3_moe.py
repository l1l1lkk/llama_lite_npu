import importlib.util
import ast
import os
import sys
import types
import unittest
from unittest import mock
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

    def test_runtime_moe_weights_are_transposed_to_gmm_native_layout(self):
        gate_up = torch.arange(2 * 16 * 3).reshape(2, 16, 3)
        down = torch.arange(2 * 3 * 8).reshape(2, 3, 8)
        tp = tp_utils.TPConfig(2, 1)

        runtime_gate_up = tp_utils.prepare_moe_gate_up_for_gmm(
            gate_up, intermediate_size=8, tp=tp
        )
        runtime_down = tp_utils.prepare_moe_down_for_gmm(down, tp=tp)

        expected_gate_up = tp_utils.shard_moe_gate_up(
            gate_up, intermediate_size=8, tp=tp
        ).transpose(1, 2).contiguous()
        expected_down = tp_utils.shard_moe_down(
            down, tp
        ).transpose(1, 2).contiguous()
        self.assertEqual(runtime_gate_up.shape, (2, 3, 8))
        self.assertEqual(runtime_down.shape, (2, 4, 3))
        self.assertTrue(runtime_gate_up.is_contiguous())
        self.assertTrue(runtime_down.is_contiguous())
        self.assertTrue(torch.equal(runtime_gate_up, expected_gate_up))
        self.assertTrue(torch.equal(runtime_down, expected_down))

    def test_expert_parallel_slices_complete_experts(self):
        gate_up = torch.arange(4 * 8 * 3).reshape(4, 8, 3)
        down = torch.arange(4 * 3 * 4).reshape(4, 3, 4)
        tp = tp_utils.TPConfig(
            world_size=2,
            rank=1,
            moe_parallel_mode="ep",
        )

        runtime_gate_up = tp_utils.prepare_moe_gate_up_for_ep(gate_up, tp)
        runtime_down = tp_utils.prepare_moe_down_for_ep(down, tp)

        self.assertEqual(runtime_gate_up.shape, (2, 3, 8))
        self.assertEqual(runtime_down.shape, (2, 4, 3))
        self.assertTrue(
            torch.equal(
                runtime_gate_up,
                gate_up[2:4].transpose(1, 2).contiguous(),
            )
        )
        self.assertTrue(
            torch.equal(
                runtime_down,
                down[2:4].transpose(1, 2).contiguous(),
            )
        )


class Qwen3MoeExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routed = _load_module(
            "qwen3_moe_routed_gemv",
            "lite_llama/kernels/moe_routed_gemv.py",
        )
        cls.moe = _load_module("qwen3_moe_layers", "lite_llama/models/moe.py")
        cls.routing = _load_module(
            "qwen3_moe_routing", "lite_llama/kernels/moe_routing.py"
        )

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
            ).transpose(1, 2)
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
                token,
                block.experts.gate_up_weight[expert_id].transpose(0, 1),
            )
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = torch.nn.functional.silu(gate) * up
            expected_rows.append(
                torch.nn.functional.linear(
                    hidden,
                    block.experts.down_weight[expert_id].transpose(0, 1),
                )
            )
        expected = torch.stack(expected_rows).view_as(inputs)
        torch.testing.assert_close(output, expected)

    def test_reference_routing_groups_gathers_and_scatters(self):
        hidden_states = torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        )
        selected_experts = torch.tensor([[2, 0], [1, 2], [0, 2]])
        routing_weights = torch.tensor(
            [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]]
        )

        plan = self.routing.prepare_moe_routing_reference(
            hidden_states,
            selected_experts,
            routing_weights,
            num_experts=3,
        )

        self.assertEqual(plan.expert_counts.tolist(), [2, 1, 3])
        self.assertEqual(plan.group_list.tolist(), [2, 3, 6])
        self.assertEqual(plan.sorted_token_ids.tolist(), [0, 2, 1, 0, 1, 2])
        torch.testing.assert_close(
            plan.routed_states,
            hidden_states[plan.sorted_token_ids],
        )

        expert_output = plan.routed_states + 1.0
        actual = self.routing.finalize_moe_routing_reference(
            expert_output,
            plan.sorted_token_ids,
            plan.sorted_weights,
            num_tokens=hidden_states.shape[0],
        )
        expected = torch.zeros_like(hidden_states)
        for token_id in range(hidden_states.shape[0]):
            for slot in range(selected_experts.shape[1]):
                expected[token_id] += (
                    hidden_states[token_id] + 1.0
                ) * routing_weights[token_id, slot]
        torch.testing.assert_close(actual, expected)

    def test_local_routing_keeps_only_owned_experts(self):
        hidden_states = torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        )
        selected_experts = torch.tensor([[0, 3], [1, 2], [3, 0]])
        routing_weights = torch.tensor(
            [[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]]
        )

        plan = self.routing.prepare_moe_routing_local_reference(
            hidden_states,
            selected_experts,
            routing_weights,
            expert_start=2,
            local_num_experts=2,
        )

        self.assertEqual(plan.expert_counts.tolist(), [1, 2])
        self.assertEqual(plan.group_list.tolist(), [1, 3])
        self.assertEqual(plan.sorted_token_ids.tolist(), [1, 0, 2])
        torch.testing.assert_close(
            plan.routed_states,
            hidden_states[plan.sorted_token_ids],
        )

    def test_grouped_path_matches_eager_reference(self):
        experts = self.moe.Qwen3MoeExperts(
            hidden_size=3,
            num_experts=3,
            intermediate_size=2,
            dtype=torch.float32,
        )
        torch.manual_seed(7)
        experts.gate_up_weight.data.normal_()
        experts.down_weight.data.normal_()
        hidden_states = torch.randn(4, 3)
        selected_experts = torch.tensor([[0, 2], [1, 0], [2, 1], [2, 0]])
        routing_weights = torch.tensor(
            [[0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.5, 0.5]]
        )

        def fake_grouped_matmul(x, weight, group_list):
            outputs = []
            start = 0
            for expert_id, end in enumerate(group_list.tolist()):
                outputs.append(x[start:end] @ weight[expert_id])
                start = end
            return torch.cat(outputs, dim=0)

        with mock.patch.object(
            experts, "_run_grouped_matmul", side_effect=fake_grouped_matmul
        ) as grouped_matmul:
            grouped = experts._forward_grouped_local(
                hidden_states, selected_experts, routing_weights
            )
        eager = experts._forward_eager_local(
            hidden_states, selected_experts, routing_weights
        )

        self.assertEqual(grouped_matmul.call_count, 2)
        first_weight = grouped_matmul.call_args_list[0].args[1]
        second_weight = grouped_matmul.call_args_list[1].args[1]
        self.assertEqual(first_weight.shape, (3, 3, 4))
        self.assertEqual(second_weight.shape, (3, 2, 3))
        torch.testing.assert_close(grouped, eager, rtol=1e-5, atol=1e-5)

    def test_routed_gemv_path_matches_eager_reference(self):
        experts = self.moe.Qwen3MoeExperts(
            hidden_size=3,
            num_experts=3,
            intermediate_size=2,
            dtype=torch.float32,
        )
        torch.manual_seed(17)
        experts.gate_up_weight.data.normal_()
        experts.down_weight.data.normal_()
        hidden_states = torch.randn(4, 3)
        selected_experts = torch.tensor(
            [[0, 2], [1, 0], [2, 1], [2, 0]]
        )
        routing_weights = torch.tensor(
            [[0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.5, 0.5]]
        )

        routed = experts._forward_routed_local(
            hidden_states, selected_experts, routing_weights
        )
        eager = experts._forward_eager_local(
            hidden_states, selected_experts, routing_weights
        )

        torch.testing.assert_close(routed, eager, rtol=1e-5, atol=1e-5)

    def test_auto_backend_selects_routed_gemv_for_small_decode_batch(self):
        experts = self.moe.Qwen3MoeExperts(
            hidden_size=3,
            num_experts=3,
            intermediate_size=2,
            dtype=torch.float32,
        )
        experts.routed_gemv_max_assignments = 32

        self.assertTrue(
            experts._should_use_routed_gemv(
                num_tokens=4, top_k=2, device_type="npu"
            )
        )
        self.assertFalse(
            experts._should_use_routed_gemv(
                num_tokens=17, top_k=2, device_type="npu"
            )
        )
        self.assertFalse(
            experts._should_use_routed_gemv(
                num_tokens=4, top_k=2, device_type="cpu"
            )
        )

    def test_expert_parallel_partial_outputs_sum_to_dense_reference(self):
        torch.manual_seed(31)
        hidden_states = torch.randn(3, 4)
        selected_experts = torch.tensor([[0, 3], [1, 2], [3, 0]])
        routing_weights = torch.tensor(
            [[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]]
        )
        global_gate_up = torch.randn(4, 4, 6)
        global_down = torch.randn(4, 3, 4)

        dense = self.routed.routed_expert_matmul_reference(
            hidden_states,
            selected_experts,
            routing_weights,
            global_gate_up,
            global_down,
        )
        rank0 = self.routed.routed_expert_matmul_reference(
            hidden_states,
            selected_experts,
            routing_weights,
            global_gate_up[:2],
            global_down[:2],
            expert_start=0,
            local_num_experts=2,
        )
        rank1 = self.routed.routed_expert_matmul_reference(
            hidden_states,
            selected_experts,
            routing_weights,
            global_gate_up[2:],
            global_down[2:],
            expert_start=2,
            local_num_experts=2,
        )

        torch.testing.assert_close(rank0 + rank1, dense)

    def test_expert_parallel_runtime_owns_only_local_experts(self):
        tp = tp_utils.TPConfig(
            world_size=2,
            rank=1,
            moe_parallel_mode="ep",
        )
        experts = self.moe.Qwen3MoeExperts(
            hidden_size=4,
            num_experts=4,
            intermediate_size=3,
            tp_config=tp,
            dtype=torch.float32,
        )

        self.assertEqual(experts.expert_start, 2)
        self.assertEqual(experts.expert_end, 4)
        self.assertEqual(experts.local_num_experts, 2)
        self.assertEqual(experts.local_intermediate_size, 3)
        self.assertEqual(experts.gate_up_weight.shape, (2, 4, 6))
        self.assertEqual(experts.down_weight.shape, (2, 3, 4))

    def test_torch_npu_gmm_uses_tensor_split_and_cumulative_groups(self):
        calls = []

        def fake_npu_grouped_matmul(**kwargs):
            calls.append(kwargs)
            return [kwargs["x"][0]]

        fake_torch_npu = types.SimpleNamespace(
            npu_grouped_matmul=fake_npu_grouped_matmul
        )
        x = torch.randn(3, 2)
        weight = torch.randn(2, 2, 4)
        group_list = torch.tensor([1, 3], dtype=torch.int64)

        with mock.patch.dict(
            sys.modules, {"torch_npu": fake_torch_npu}
        ):
            result = self.moe.Qwen3MoeExperts._run_grouped_matmul(
                x, weight, group_list
            )

        self.assertIs(result, x)
        self.assertEqual(calls[0]["split_item"], 2)
        self.assertEqual(calls[0]["group_type"], 0)
        self.assertEqual(calls[0]["group_list_type"], 0)
        self.assertIs(calls[0]["x"][0], x)
        self.assertIs(calls[0]["weight"][0], weight)
        self.assertIs(calls[0]["group_list"], group_list)

    def test_auto_backend_keeps_cpu_on_eager_path(self):
        experts = self.moe.Qwen3MoeExperts(
            hidden_size=2,
            num_experts=2,
            intermediate_size=2,
            dtype=torch.float32,
        )
        experts.gate_up_weight.data.zero_()
        experts.down_weight.data.zero_()
        hidden_states = torch.ones(1, 2)
        selected_experts = torch.tensor([[0]])
        routing_weights = torch.ones(1, 1)

        with mock.patch.object(
            experts,
            "_forward_grouped_local",
            side_effect=AssertionError("CPU must not use GMM"),
        ):
            output = experts(
                hidden_states, selected_experts, routing_weights
            )

        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_alignment_validation_reports_numeric_error(self):
        experts = self.moe.Qwen3MoeExperts(
            hidden_size=2,
            num_experts=2,
            intermediate_size=2,
            dtype=torch.float32,
        )

        with self.assertRaisesRegex(
            RuntimeError, "MoE GMM alignment failed.*max_abs_diff"
        ):
            experts._validate_local_outputs(
                torch.zeros(2, 2),
                torch.ones(2, 2),
                rtol=1e-2,
                atol=1e-2,
            )

    def test_backend_environment_rejects_unknown_value(self):
        with mock.patch.dict(
            os.environ, {"LITE_LLAMA_MOE_BACKEND": "unknown"}
        ):
            with self.assertRaisesRegex(ValueError, "LITE_LLAMA_MOE_BACKEND"):
                self.moe.Qwen3MoeExperts(
                    hidden_size=2,
                    num_experts=2,
                    intermediate_size=2,
                )

    def test_backend_environment_accepts_routed_gemv(self):
        with mock.patch.dict(
            os.environ, {"LITE_LLAMA_MOE_BACKEND": "routed_gemv"}
        ):
            experts = self.moe.Qwen3MoeExperts(
                hidden_size=2,
                num_experts=2,
                intermediate_size=2,
            )
        self.assertEqual(experts.backend, "routed_gemv")

    def test_npu_routing_source_has_no_host_visible_tolist(self):
        source = (
            ROOT / "lite_llama/kernels/moe_routing.py"
        ).read_text(encoding="utf-8")
        module = ast.parse(source)
        functions = {
            node.name: node
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in ("prepare_moe_routing_npu", "finalize_moe_routing_npu"):
            self.assertIn(name, functions)
            function_source = ast.get_source_segment(source, functions[name])
            self.assertNotIn(".tolist(", function_source)

        kernel_names = {
            node.name
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("_moe_")
        }
        self.assertIn("_moe_count_and_gather_kernel", kernel_names)
        self.assertIn("_moe_weighted_scatter_kernel", kernel_names)

    def test_ascend_gather_does_not_consume_atomic_add_return_value(self):
        source = (
            ROOT / "lite_llama/kernels/moe_routing.py"
        ).read_text(encoding="utf-8")
        module = ast.parse(source)
        gather_kernel = next(
            node
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_moe_count_and_gather_kernel"
        )

        atomic_assignments = [
            node
            for node in ast.walk(gather_kernel)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "atomic_add"
        ]
        self.assertEqual(
            atomic_assignments,
            [],
            "Ascend Triton cannot consume the old value returned by "
            "tl.atomic_add during TTIR-to-Linalg conversion",
        )


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
