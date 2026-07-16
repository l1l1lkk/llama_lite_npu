import ast
import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import torch

from tests.reference.deepseek_moe_reference import (
    DeepSeekRoutingSpec,
    deepseek_route_reference,
)
from tests.reference.moe_reference import route_topk_reference


ROOT = Path(__file__).resolve().parents[2]
MOE_PATH = ROOT / "lite_llama/models/moe.py"


def _load_production_moe():
    name = "deepseek_moe_router_contract_production"
    spec = importlib.util.spec_from_file_location(name, MOE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_router(production, spec, hidden_size, dtype=torch.float32):
    return production.DeepSeekGroupedTopKRouter(
        hidden_size=hidden_size,
        num_experts=spec.num_experts,
        top_k=spec.experts_per_token,
        num_groups=spec.num_groups,
        topk_groups=spec.topk_groups,
        score_func=spec.score_func,
        topk_method=spec.topk_method,
        norm_topk_prob=spec.norm_topk_prob,
        routed_scaling_factor=spec.routed_scaling_factor,
        dtype=dtype,
    )


def _assert_matches_reference(
    test_case,
    router,
    hidden_states,
    reference_spec,
    *,
    correction_bias=None,
    rtol=0.0,
    atol=0.0,
):
    expected_logits, expected_weights, expected_ids = (
        deepseek_route_reference(
            hidden_states,
            router.weight.detach(),
            reference_spec,
            correction_bias=correction_bias,
        )
    )
    result = router(hidden_states)
    logits, weights, selected_experts = result

    test_case.assertIs(result.router_logits, logits)
    test_case.assertIs(result.routing_weights, weights)
    test_case.assertIs(result.selected_experts, selected_experts)
    test_case.assertEqual(
        result._fields,
        ("router_logits", "routing_weights", "selected_experts"),
    )
    test_case.assertTrue(torch.equal(selected_experts, expected_ids))
    test_case.assertEqual(selected_experts.dtype, torch.int64)
    torch.testing.assert_close(
        logits.float(),
        expected_logits,
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        weights.float(),
        expected_weights,
        rtol=rtol,
        atol=atol,
    )
    return result


class DeepSeekMoeRouterContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.production = _load_production_moe()

    def test_grouped_topk_config_is_immutable(self):
        config = self.production.GroupedTopKConfig(
            num_groups=2,
            topk_groups=1,
            score_func="softmax",
            topk_method="group_limited_greedy",
        )

        self.assertEqual(config.num_groups, 2)
        self.assertEqual(config.topk_groups, 1)
        self.assertTrue(config.norm_topk_prob)
        self.assertEqual(config.routed_scaling_factor, 1.0)
        with self.assertRaises(AttributeError):
            config.num_groups = 4

    def test_config_and_constructor_reject_unaudited_combinations(self):
        config_cases = (
            {"num_groups": 0},
            {"topk_groups": 3},
            {"score_func": "sqrtsoftplus"},
            {"topk_method": "static_hash"},
            {"score_func": "sigmoid"},
            {"topk_method": "noaux_tc"},
            {"routed_scaling_factor": float("inf")},
        )
        defaults = {
            "num_groups": 2,
            "topk_groups": 1,
            "score_func": "softmax",
            "topk_method": "group_limited_greedy",
        }
        for overrides in config_cases:
            values = dict(defaults)
            values.update(overrides)
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    self.production.GroupedTopKConfig(**values)

        router_cases = (
            {"num_experts": 5},
            {"top_k": 3},
            {
                "num_experts": 2,
                "top_k": 1,
                "num_groups": 2,
                "score_func": "sigmoid",
                "topk_method": "noaux_tc",
            },
        )
        router_defaults = {
            "hidden_size": 3,
            "num_experts": 4,
            "top_k": 2,
            "num_groups": 2,
            "topk_groups": 1,
            "score_func": "softmax",
            "topk_method": "group_limited_greedy",
            "dtype": torch.float32,
        }
        for overrides in router_cases:
            values = dict(router_defaults)
            values.update(overrides)
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    self.production.DeepSeekGroupedTopKRouter(**values)

    def test_v2_fp32_matrix_matches_independent_reference(self):
        generator = torch.Generator().manual_seed(620)
        cases = (
            (0, 4, 2, 2, 1, True, 1.0),
            (1, 4, 1, 2, 1, False, 2.0),
            (5, 8, 4, 4, 2, True, 2.5),
        )
        for tokens, experts, top_k, groups, top_groups, norm, scale in cases:
            with self.subTest(
                tokens=tokens,
                experts=experts,
                top_k=top_k,
            ):
                spec = DeepSeekRoutingSpec(
                    num_experts=experts,
                    experts_per_token=top_k,
                    num_groups=groups,
                    topk_groups=top_groups,
                    score_func="softmax",
                    topk_method="group_limited_greedy",
                    norm_topk_prob=norm,
                    routed_scaling_factor=scale,
                )
                router = _make_router(self.production, spec, hidden_size=4)
                with torch.no_grad():
                    router.weight.copy_(
                        torch.randn(experts, 4, generator=generator)
                    )
                hidden_states = torch.randn(tokens, 4, generator=generator)

                result = _assert_matches_reference(
                    self,
                    router,
                    hidden_states,
                    spec,
                )
                self.assertEqual(result.router_logits.shape, (tokens, experts))
                self.assertEqual(result.routing_weights.shape, (tokens, top_k))
                self.assertEqual(result.selected_experts.shape, (tokens, top_k))

    def test_v3_fp32_matrix_matches_independent_reference(self):
        generator = torch.Generator().manual_seed(621)
        cases = (
            (1, 4, 1, 2, 1, False, 2.0),
            (6, 8, 3, 4, 2, True, 2.5),
        )
        for tokens, experts, top_k, groups, top_groups, norm, scale in cases:
            with self.subTest(tokens=tokens, experts=experts):
                spec = DeepSeekRoutingSpec(
                    num_experts=experts,
                    experts_per_token=top_k,
                    num_groups=groups,
                    topk_groups=top_groups,
                    score_func="sigmoid",
                    topk_method="noaux_tc",
                    norm_topk_prob=norm,
                    routed_scaling_factor=scale,
                )
                router = _make_router(self.production, spec, hidden_size=4)
                correction_bias = torch.randn(
                    experts,
                    generator=generator,
                ) * 0.1
                with torch.no_grad():
                    router.weight.copy_(
                        torch.randn(experts, 4, generator=generator)
                    )
                    router.e_score_correction_bias.copy_(correction_bias)
                hidden_states = torch.randn(tokens, 4, generator=generator)

                _assert_matches_reference(
                    self,
                    router,
                    hidden_states,
                    spec,
                    correction_bias=correction_bias,
                )

    def test_correction_bias_changes_selection_not_combine_weight(self):
        spec = DeepSeekRoutingSpec(
            num_experts=4,
            experts_per_token=1,
            num_groups=2,
            topk_groups=1,
            score_func="sigmoid",
            topk_method="noaux_tc",
            norm_topk_prob=False,
            routed_scaling_factor=2.0,
        )
        router = _make_router(self.production, spec, hidden_size=4)
        correction_bias = torch.tensor([0.0, 0.0, 3.0, 0.0])
        with torch.no_grad():
            router.weight.copy_(torch.eye(4))
            router.e_score_correction_bias.copy_(correction_bias)
        hidden_states = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

        result = _assert_matches_reference(
            self,
            router,
            hidden_states,
            spec,
            correction_bias=correction_bias,
        )

        self.assertEqual(result.selected_experts.item(), 2)
        unbiased = torch.sigmoid(result.router_logits.float())[0, 2] * 2.0
        biased = (torch.sigmoid(result.router_logits.float())[0, 2] + 3.0) * 2.0
        torch.testing.assert_close(result.routing_weights[0, 0], unbiased)
        self.assertNotEqual(result.routing_weights[0, 0].item(), biased.item())

    def test_norm_and_route_scale_match_reference(self):
        hidden_states = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
        for norm_topk_prob in (False, True):
            with self.subTest(norm_topk_prob=norm_topk_prob):
                spec = DeepSeekRoutingSpec(
                    num_experts=4,
                    experts_per_token=2,
                    num_groups=1,
                    topk_groups=1,
                    score_func="softmax",
                    topk_method="group_limited_greedy",
                    norm_topk_prob=norm_topk_prob,
                    routed_scaling_factor=2.5,
                )
                router = _make_router(self.production, spec, hidden_size=4)
                with torch.no_grad():
                    router.weight.copy_(torch.eye(4))

                result = _assert_matches_reference(
                    self,
                    router,
                    hidden_states,
                    spec,
                )
                if norm_topk_prob:
                    torch.testing.assert_close(
                        result.routing_weights.sum(dim=-1),
                        torch.tensor([2.5]),
                    )

    def test_fp16_and_bfloat16_stable_margin_match_reference(self):
        spec = DeepSeekRoutingSpec(
            num_experts=4,
            experts_per_token=2,
            num_groups=1,
            topk_groups=1,
            score_func="softmax",
            topk_method="group_limited_greedy",
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
        hidden_values = torch.tensor([[8.0, 4.0, 1.0, -3.0]])
        for dtype, tolerance in (
            (torch.float16, 2.0e-3),
            (torch.bfloat16, 1.0e-2),
        ):
            with self.subTest(dtype=dtype):
                router = _make_router(
                    self.production,
                    spec,
                    hidden_size=4,
                    dtype=dtype,
                )
                with torch.no_grad():
                    router.weight.copy_(torch.eye(4, dtype=dtype))
                hidden_states = hidden_values.to(dtype)

                result = _assert_matches_reference(
                    self,
                    router,
                    hidden_states,
                    spec,
                    rtol=tolerance,
                    atol=tolerance,
                )
                self.assertEqual(set(result.selected_experts[0].tolist()), {0, 1})
                self.assertEqual(result.router_logits.dtype, dtype)
                self.assertEqual(result.routing_weights.dtype, dtype)

    def test_tie_contract_locks_set_and_weight_properties_only(self):
        spec = DeepSeekRoutingSpec(
            num_experts=4,
            experts_per_token=2,
            num_groups=2,
            topk_groups=1,
            score_func="softmax",
            topk_method="group_limited_greedy",
        )
        router = _make_router(self.production, spec, hidden_size=4)
        with torch.no_grad():
            router.weight.copy_(torch.eye(4))

        result = router(torch.zeros(1, 4))

        selected_set = set(result.selected_experts[0].tolist())
        self.assertIn(selected_set, ({0, 1}, {2, 3}))
        torch.testing.assert_close(
            result.routing_weights[0].float(),
            torch.tensor([0.5, 0.5]),
        )

    def test_state_dict_distinguishes_v2_and_v3_payloads(self):
        v2_spec = DeepSeekRoutingSpec(
            num_experts=4,
            experts_per_token=2,
            num_groups=2,
            topk_groups=1,
            score_func="softmax",
            topk_method="group_limited_greedy",
        )
        v3_spec = DeepSeekRoutingSpec(
            num_experts=4,
            experts_per_token=2,
            num_groups=2,
            topk_groups=1,
            score_func="sigmoid",
            topk_method="noaux_tc",
        )
        v2_router = _make_router(self.production, v2_spec, hidden_size=3)
        v3_router = _make_router(self.production, v3_spec, hidden_size=3)

        self.assertEqual(set(v2_router.state_dict()), {"weight"})
        self.assertEqual(
            set(v3_router.state_dict()),
            {"weight", "e_score_correction_bias"},
        )
        self.assertEqual(
            v3_router.e_score_correction_bias.shape,
            (4,),
        )
        self.assertEqual(
            v3_router.e_score_correction_bias.dtype,
            torch.float32,
        )

        state = {
            "weight": torch.randn(4, 3),
            "e_score_correction_bias": torch.randn(4),
        }
        v3_router.load_state_dict(state)
        torch.testing.assert_close(v3_router.weight, state["weight"])
        torch.testing.assert_close(
            v3_router.e_score_correction_bias,
            state["e_score_correction_bias"],
        )

    def test_qwen_router_signature_state_and_output_remain_compatible(self):
        self.assertEqual(
            inspect.signature(self.production.SoftmaxTopKRouter),
            inspect.signature(self.production.Qwen3MoeTopKRouter),
        )
        softmax = self.production.SoftmaxTopKRouter(
            hidden_size=4,
            num_experts=4,
            top_k=2,
            dtype=torch.float32,
        )
        qwen = self.production.Qwen3MoeTopKRouter(
            hidden_size=4,
            num_experts=4,
            top_k=2,
            dtype=torch.float32,
        )
        weight = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        hidden_states = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        with torch.no_grad():
            softmax.weight.copy_(weight)
            qwen.weight.copy_(weight)
        expected = route_topk_reference(
            hidden_states,
            weight,
            top_k=2,
            norm_topk_prob=True,
        )

        softmax_result = softmax(hidden_states)
        qwen_result = qwen(hidden_states)
        self.assertEqual(set(softmax.state_dict()), {"weight"})
        self.assertEqual(set(qwen.state_dict()), {"weight"})
        for softmax_tensor, qwen_tensor, expected_tensor in zip(
            softmax_result,
            qwen_result,
            expected,
        ):
            torch.testing.assert_close(softmax_tensor, expected_tensor)
            torch.testing.assert_close(qwen_tensor, expected_tensor)

    def test_direct_file_loader_exposes_dependency_light_router(self):
        self.assertEqual(self.production.__package__, "")
        self.assertTrue(
            issubclass(
                self.production.DeepSeekGroupedTopKRouter,
                torch.nn.Module,
            )
        )
        self.assertEqual(
            self.production.RoutingResult._fields,
            ("router_logits", "routing_weights", "selected_experts"),
        )

    def test_grouped_router_hot_path_has_no_host_sync_or_dynamic_adapter(self):
        source = MOE_PATH.read_text(encoding="utf-8")
        module = ast.parse(source)
        router_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == "DeepSeekGroupedTopKRouter"
        )
        forward = next(
            node
            for node in router_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        forward_source = ast.get_source_segment(source, forward)

        for forbidden in (
            ".item(",
            ".tolist(",
            ".cpu(",
            ".numpy(",
            "print(",
            "logger",
            "warning",
            "isinstance(",
        ):
            self.assertNotIn(forbidden, forward_source)

        conditions = [
            ast.unparse(node.test)
            for node in ast.walk(forward)
            if isinstance(node, ast.If)
        ]
        self.assertEqual(len(conditions), 4)
        for condition in conditions:
            self.assertIn("self.grouped_config", condition)
            self.assertNotIn("hidden_states", condition)
            self.assertNotIn("router_logits", condition)
            self.assertNotIn("routing_weights", condition)


if __name__ == "__main__":
    unittest.main()
