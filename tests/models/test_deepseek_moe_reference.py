import ast
import unittest
from pathlib import Path

import torch

from tests.reference.deepseek_moe_reference import (
    DeepSeekRoutingSpec,
    deepseek_moe_reference,
    deepseek_route_reference,
    deepseek_routed_experts_reference,
    deepseek_shared_expert_reference,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "tests/reference/deepseek_moe_reference.py"


def _v2_spec(**overrides):
    values = {
        "num_experts": 4,
        "experts_per_token": 2,
        "num_groups": 2,
        "topk_groups": 1,
        "score_func": "softmax",
        "topk_method": "group_limited_greedy",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
    }
    values.update(overrides)
    return DeepSeekRoutingSpec(**values)


def _v3_spec(**overrides):
    values = {
        "num_experts": 4,
        "experts_per_token": 2,
        "num_groups": 2,
        "topk_groups": 1,
        "score_func": "sigmoid",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
    }
    values.update(overrides)
    return DeepSeekRoutingSpec(**values)


def _tp_gate_up_slice(weight, start, end):
    gate, up = weight.chunk(2, dim=-1)
    return torch.cat((gate[..., start:end], up[..., start:end]), dim=-1)


class DeepSeekMoeRoutingReferenceTest(unittest.TestCase):
    def test_empty_routing_and_full_moe_shapes(self):
        hidden = torch.empty(0, 3)
        router_weight = torch.randn(4, 3)
        gate_up = torch.randn(4, 3, 4)
        down = torch.randn(4, 2, 3)

        output, logits, weights, expert_ids = deepseek_moe_reference(
            hidden,
            router_weight,
            gate_up,
            down,
            _v2_spec(),
        )

        self.assertEqual(output.shape, (0, 3))
        self.assertEqual(logits.shape, (0, 4))
        self.assertEqual(weights.shape, (0, 2))
        self.assertEqual(expert_ids.shape, (0, 2))
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(expert_ids.dtype, torch.int64)

    def test_single_and_multi_token_legal_matrix(self):
        generator = torch.Generator().manual_seed(610)
        cases = (
            (1, 4, 1, 2, 1),
            (5, 4, 2, 2, 1),
            (3, 8, 4, 4, 2),
        )
        for tokens, experts, top_k, groups, top_groups in cases:
            with self.subTest(
                tokens=tokens,
                experts=experts,
                top_k=top_k,
            ):
                hidden = torch.randn(tokens, 3, generator=generator)
                router_weight = torch.randn(experts, 3, generator=generator)
                logits, weights, expert_ids = deepseek_route_reference(
                    hidden,
                    router_weight,
                    _v2_spec(
                        num_experts=experts,
                        experts_per_token=top_k,
                        num_groups=groups,
                        topk_groups=top_groups,
                    ),
                )
                self.assertEqual(logits.shape, (tokens, experts))
                self.assertEqual(weights.shape, (tokens, top_k))
                self.assertEqual(expert_ids.shape, (tokens, top_k))
                self.assertTrue(torch.isfinite(weights).all().item())

    def test_router_uses_fp32_and_global_int64_ids(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                hidden = torch.tensor([[2.0, 1.0]], dtype=dtype)
                router_weight = torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, 0.0],
                        [0.0, -1.0],
                    ],
                    dtype=dtype,
                )

                logits, weights, expert_ids = deepseek_route_reference(
                    hidden,
                    router_weight,
                    _v2_spec(),
                )

                self.assertEqual(logits.dtype, torch.float32)
                self.assertEqual(weights.dtype, torch.float32)
                self.assertEqual(expert_ids.dtype, torch.int64)

    def test_softmax_routing_matches_explicit_scores(self):
        hidden = torch.tensor([[2.0, 1.0]])
        router_weight = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        )
        logits, weights, expert_ids = deepseek_route_reference(
            hidden,
            router_weight,
            _v2_spec(),
        )

        expected_scores = torch.softmax(logits, dim=-1)
        self.assertEqual(set(expert_ids[0].tolist()), {0, 1})
        expected = expected_scores.gather(1, expert_ids)
        expected = expected / expected.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(weights, expected)

    def test_correction_bias_changes_selection_only(self):
        hidden = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        router_weight = torch.eye(4)
        correction_bias = torch.tensor([0.0, 0.0, 3.0, 0.0])
        spec = _v3_spec(
            experts_per_token=1,
            norm_topk_prob=False,
            routed_scaling_factor=2.0,
        )

        logits, weights, expert_ids = deepseek_route_reference(
            hidden,
            router_weight,
            spec,
            correction_bias=correction_bias,
        )

        self.assertEqual(expert_ids.item(), 2)
        original_score = torch.sigmoid(logits)[0, 2] * 2.0
        biased_score = (torch.sigmoid(logits)[0, 2] + 3.0) * 2.0
        torch.testing.assert_close(weights[0, 0], original_score)
        self.assertNotEqual(weights[0, 0].item(), biased_score.item())

    def test_group_limited_greedy_uses_group_max(self):
        hidden = torch.tensor([[10.0, 0.0, 6.0, 6.0, 1.0, 1.0]])
        router_weight = torch.eye(6)
        spec = _v2_spec(
            num_experts=6,
            experts_per_token=2,
            num_groups=3,
            topk_groups=1,
        )

        _, _, expert_ids = deepseek_route_reference(
            hidden,
            router_weight,
            spec,
        )

        self.assertEqual(set(expert_ids[0].tolist()), {0, 1})

    def test_noaux_tc_uses_group_top2_sum(self):
        hidden = torch.tensor([[4.0, -4.0, 1.0, 1.0, -2.0, -2.0]])
        router_weight = torch.eye(6)
        spec = _v3_spec(
            num_experts=6,
            experts_per_token=2,
            num_groups=3,
            topk_groups=1,
        )

        _, _, expert_ids = deepseek_route_reference(
            hidden,
            router_weight,
            spec,
            correction_bias=torch.zeros(6),
        )

        self.assertEqual(set(expert_ids[0].tolist()), {2, 3})

    def test_norm_and_routed_scaling_factor_are_ordered(self):
        hidden = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
        router_weight = torch.eye(4)
        raw_spec = _v2_spec(
            norm_topk_prob=False,
            routed_scaling_factor=2.5,
        )
        norm_spec = _v2_spec(
            norm_topk_prob=True,
            routed_scaling_factor=2.5,
        )

        logits, raw_weights, raw_ids = deepseek_route_reference(
            hidden,
            router_weight,
            raw_spec,
        )
        _, norm_weights, norm_ids = deepseek_route_reference(
            hidden,
            router_weight,
            norm_spec,
        )

        self.assertTrue(torch.equal(raw_ids, norm_ids))
        scores = torch.softmax(logits, dim=-1).gather(1, raw_ids)
        torch.testing.assert_close(raw_weights, scores * 2.5)
        torch.testing.assert_close(
            norm_weights.sum(dim=-1),
            torch.tensor([2.5]),
        )

    def test_k_boundary_has_positive_margin(self):
        hidden = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        router_weight = torch.eye(4)
        spec = _v2_spec(num_groups=1, topk_groups=1)

        logits, _, expert_ids = deepseek_route_reference(
            hidden,
            router_weight,
            spec,
        )

        scores = torch.softmax(logits, dim=-1)
        sorted_scores = scores.sort(dim=-1, descending=True).values
        margin = sorted_scores[0, 1] - sorted_scores[0, 2]
        self.assertEqual(set(expert_ids[0].tolist()), {0, 1})
        self.assertGreater(margin.item(), 0.0)

    def test_tie_contract_does_not_bind_group_or_slot_order(self):
        hidden = torch.zeros(1, 4)
        router_weight = torch.eye(4)

        _, weights, expert_ids = deepseek_route_reference(
            hidden,
            router_weight,
            _v2_spec(),
        )

        selected_set = set(expert_ids[0].tolist())
        self.assertIn(selected_set, ({0, 1}, {2, 3}))
        torch.testing.assert_close(weights[0], torch.tensor([0.5, 0.5]))

    def test_non_contiguous_extreme_finite_input_is_deterministic(self):
        hidden = torch.tensor(
            [
                [1.0e4, -1.0e4],
                [5.0e3, -5.0e3],
                [-2.0e3, 2.0e3],
                [1.0, -1.0],
            ]
        ).t()
        self.assertFalse(hidden.is_contiguous())
        router_weight = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        first = deepseek_route_reference(hidden, router_weight, _v2_spec())
        second = deepseek_route_reference(hidden, router_weight, _v2_spec())

        for first_tensor, second_tensor in zip(first, second):
            self.assertTrue(torch.equal(first_tensor, second_tensor))
        self.assertTrue(torch.isfinite(first[0]).all().item())
        self.assertTrue(torch.isfinite(first[1]).all().item())

    def test_configuration_and_bias_contract_fail_closed(self):
        invalid_specs = (
            {"num_experts": 0},
            {"num_experts": 5, "num_groups": 2},
            {"topk_groups": 3},
            {"experts_per_token": 3},
            {"score_func": "sqrtsoftplus"},
            {"topk_method": "greedy"},
            {"score_func": "sigmoid"},
            {"topk_method": "noaux_tc"},
            {
                "num_experts": 2,
                "experts_per_token": 1,
                "num_groups": 2,
                "topk_groups": 1,
                "score_func": "sigmoid",
                "topk_method": "noaux_tc",
            },
            {"routed_scaling_factor": float("inf")},
        )
        for overrides in invalid_specs:
            with self.subTest(overrides=overrides):
                with self.assertRaises((TypeError, ValueError)):
                    _v2_spec(**overrides)

        hidden = torch.ones(1, 2)
        router_weight = torch.ones(4, 2)
        with self.assertRaisesRegex(ValueError, "requires correction_bias"):
            deepseek_route_reference(
                hidden,
                router_weight,
                _v3_spec(),
            )
        with self.assertRaisesRegex(ValueError, "does not accept"):
            deepseek_route_reference(
                hidden,
                router_weight,
                _v2_spec(),
                correction_bias=torch.zeros(4),
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            deepseek_route_reference(
                hidden,
                router_weight,
                _v3_spec(),
                correction_bias=torch.zeros(3),
            )

    def test_non_finite_values_are_out_of_contract(self):
        hidden = torch.tensor([[float("nan"), 0.0]])
        router_weight = torch.ones(4, 2)
        with self.assertRaisesRegex(ValueError, "finite"):
            deepseek_route_reference(hidden, router_weight, _v2_spec())

    def test_sigmoid_normalization_underflow_fails_closed_only_when_enabled(self):
        hidden = torch.tensor([[1.0]])
        router_weight = torch.full((4, 1), -1000.0)
        correction_bias = torch.zeros(4)

        with self.assertRaisesRegex(
            ValueError,
            "normalization sum must be positive and finite",
        ):
            deepseek_route_reference(
                hidden,
                router_weight,
                _v3_spec(norm_topk_prob=True),
                correction_bias=correction_bias,
            )

        _, unnormalized_weights, expert_ids = deepseek_route_reference(
            hidden,
            router_weight,
            _v3_spec(norm_topk_prob=False),
            correction_bias=correction_bias,
        )
        self.assertEqual(expert_ids.shape, (1, 2))
        self.assertTrue(torch.isfinite(unnormalized_weights).all().item())
        torch.testing.assert_close(
            unnormalized_weights,
            torch.zeros_like(unnormalized_weights),
        )

    def test_routing_weight_scaling_overflow_fails_closed(self):
        with self.assertRaisesRegex(
            ValueError,
            "routing weights must remain finite after scaling",
        ):
            deepseek_route_reference(
                torch.ones(1, 1),
                torch.arange(4, dtype=torch.float32).reshape(4, 1),
                _v2_spec(routed_scaling_factor=1.0e40),
            )


class DeepSeekMoeExpertReferenceTest(unittest.TestCase):
    def test_hot_expert_and_empty_experts(self):
        generator = torch.Generator().manual_seed(611)
        hidden = torch.randn(3, 3, generator=generator)
        expert_ids = torch.zeros(3, 2, dtype=torch.int64)
        routing_weights = torch.full((3, 2), 0.5)
        gate_up = torch.randn(4, 3, 4, generator=generator)
        down = torch.randn(4, 2, 3, generator=generator)
        changed_gate_up = gate_up.clone()
        changed_down = down.clone()
        changed_gate_up[1:] = 1000.0
        changed_down[1:] = -1000.0

        original = deepseek_routed_experts_reference(
            hidden,
            expert_ids,
            routing_weights,
            gate_up,
            down,
        )
        changed = deepseek_routed_experts_reference(
            hidden,
            expert_ids,
            routing_weights,
            changed_gate_up,
            changed_down,
        )

        torch.testing.assert_close(original, changed)

    def test_shared_zero_one_and_combined_multiple_width(self):
        generator = torch.Generator().manual_seed(612)
        hidden = torch.randn(4, 3, generator=generator)
        first_gate_up = torch.randn(3, 4, generator=generator)
        second_gate_up = torch.randn(3, 4, generator=generator)
        first_down = torch.randn(2, 3, generator=generator)
        second_down = torch.randn(2, 3, generator=generator)
        first_gate, first_up = first_gate_up.chunk(2, dim=-1)
        second_gate, second_up = second_gate_up.chunk(2, dim=-1)
        combined_gate_up = torch.cat(
            (first_gate, second_gate, first_up, second_up),
            dim=-1,
        )
        combined_down = torch.cat((first_down, second_down), dim=0)

        zero = deepseek_shared_expert_reference(hidden)
        first = deepseek_shared_expert_reference(
            hidden,
            first_gate_up,
            first_down,
        )
        second = deepseek_shared_expert_reference(
            hidden,
            second_gate_up,
            second_down,
        )
        combined = deepseek_shared_expert_reference(
            hidden,
            combined_gate_up,
            combined_down,
        )

        torch.testing.assert_close(zero, torch.zeros_like(hidden))
        torch.testing.assert_close(combined, first + second)

    def test_full_moe_adds_routed_and_shared_outputs(self):
        generator = torch.Generator().manual_seed(613)
        hidden = torch.randn(3, 3, generator=generator)
        router_weight = torch.randn(4, 3, generator=generator)
        routed_gate_up = torch.randn(4, 3, 4, generator=generator)
        routed_down = torch.randn(4, 2, 3, generator=generator)
        shared_gate_up = torch.randn(3, 6, generator=generator)
        shared_down = torch.randn(3, 3, generator=generator)

        output, logits, weights, expert_ids = deepseek_moe_reference(
            hidden,
            router_weight,
            routed_gate_up,
            routed_down,
            _v2_spec(),
            shared_gate_up_weight=shared_gate_up,
            shared_down_weight=shared_down,
        )
        routed = deepseek_routed_experts_reference(
            hidden,
            expert_ids,
            weights,
            routed_gate_up,
            routed_down,
        )
        shared = deepseek_shared_expert_reference(
            hidden,
            shared_gate_up,
            shared_down,
        )

        torch.testing.assert_close(output, routed + shared)
        self.assertEqual(logits.shape, (3, 4))

    def test_tensor_parallel_routed_partial_sum(self):
        generator = torch.Generator().manual_seed(614)
        hidden = torch.randn(4, 3, generator=generator)
        expert_ids = torch.tensor(
            [[0, 1], [2, 3], [1, 3], [0, 2]],
            dtype=torch.int64,
        )
        routing_weights = torch.tensor(
            [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.5, 0.5]]
        )
        gate_up = torch.randn(4, 3, 8, generator=generator)
        down = torch.randn(4, 4, 3, generator=generator)
        full = deepseek_routed_experts_reference(
            hidden,
            expert_ids,
            routing_weights,
            gate_up,
            down,
        )
        partials = []
        for start, end in ((0, 2), (2, 4)):
            partials.append(
                deepseek_routed_experts_reference(
                    hidden,
                    expert_ids,
                    routing_weights,
                    _tp_gate_up_slice(gate_up, start, end),
                    down[:, start:end],
                )
            )

        torch.testing.assert_close(
            partials[0] + partials[1],
            full,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_tensor_parallel_shared_partial_sum(self):
        generator = torch.Generator().manual_seed(615)
        hidden = torch.randn(4, 3, generator=generator)
        gate_up = torch.randn(3, 8, generator=generator)
        down = torch.randn(4, 3, generator=generator)
        full = deepseek_shared_expert_reference(hidden, gate_up, down)
        partials = []
        for start, end in ((0, 2), (2, 4)):
            partials.append(
                deepseek_shared_expert_reference(
                    hidden,
                    _tp_gate_up_slice(gate_up.unsqueeze(0), start, end)[0],
                    down[start:end],
                )
            )

        torch.testing.assert_close(
            partials[0] + partials[1],
            full,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_expert_parallel_ownership_and_local_sum(self):
        generator = torch.Generator().manual_seed(616)
        hidden = torch.randn(4, 3, generator=generator)
        expert_ids = torch.tensor(
            [[0, 1], [2, 3], [1, 2], [3, 0]],
            dtype=torch.int64,
        )
        routing_weights = torch.tensor(
            [[0.6, 0.4], [0.3, 0.7], [0.8, 0.2], [0.5, 0.5]]
        )
        gate_up = torch.randn(4, 3, 4, generator=generator)
        down = torch.randn(4, 2, 3, generator=generator)
        full = deepseek_routed_experts_reference(
            hidden,
            expert_ids,
            routing_weights,
            gate_up,
            down,
        )
        rank_zero = deepseek_routed_experts_reference(
            hidden,
            expert_ids,
            routing_weights,
            gate_up[:2],
            down[:2],
            expert_start=0,
        )
        rank_one = deepseek_routed_experts_reference(
            hidden,
            expert_ids,
            routing_weights,
            gate_up[2:],
            down[2:],
            expert_start=2,
        )

        ownership = {
            expert_id: int(expert_id < 2) + int(expert_id >= 2)
            for expert_id in expert_ids.unique().tolist()
        }
        self.assertEqual(set(ownership.values()), {1})
        torch.testing.assert_close(rank_zero + rank_one, full)

    def test_reference_is_inference_only(self):
        hidden = torch.randn(2, 3, requires_grad=True)
        router_weight = torch.randn(4, 3, requires_grad=True)
        gate_up = torch.randn(4, 3, 4, requires_grad=True)
        down = torch.randn(4, 2, 3, requires_grad=True)

        result = deepseek_moe_reference(
            hidden,
            router_weight,
            gate_up,
            down,
            _v2_spec(),
        )

        for tensor in result:
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)

    def test_weight_shape_and_shared_pair_errors_fail_closed(self):
        hidden = torch.ones(1, 3)
        expert_ids = torch.zeros(1, 1, dtype=torch.int64)
        routing_weights = torch.ones(1, 1)
        with self.assertRaisesRegex(ValueError, "rank-3"):
            deepseek_routed_experts_reference(
                hidden,
                expert_ids,
                routing_weights,
                torch.ones(3, 4),
                torch.ones(1, 2, 3),
            )
        with self.assertRaisesRegex(ValueError, "paired"):
            deepseek_shared_expert_reference(
                hidden,
                torch.ones(3, 4),
                None,
            )


class DeepSeekMoeReferenceIndependenceTest(unittest.TestCase):
    def test_reference_has_no_production_import_or_symbol_dependency(self):
        source = REFERENCE_PATH.read_text(encoding="utf-8")
        module = ast.parse(source)
        imported_roots = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])

        self.assertLessEqual(
            imported_roots,
            {"__future__", "dataclasses", "math", "typing", "torch"},
        )
        for forbidden in (
            "lite_llama",
            "SoftmaxTopKRouter",
            "Qwen3MoeTopKRouter",
            "RoutedExpertExecutor",
            "Qwen3MoeExperts",
            "shard_moe",
            "prepare_moe",
            "grouped_matmul",
            "routed_gemv",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
