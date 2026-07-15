import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

from tests.reference.moe_reference import (
    expert_forward_reference,
    moe_forward_reference,
    route_topk_reference,
)


ROOT = Path(__file__).resolve().parents[2]


def _load_production_moe():
    name = "qwen3_moe_reference_contract_production"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "lite_llama/models/moe.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MoeReferenceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.production = _load_production_moe()

    @staticmethod
    def _make_block(
        *,
        hidden_size=4,
        num_experts=4,
        intermediate_size=3,
        top_k=2,
        norm_topk_prob=True,
        seed=101,
    ):
        block = MoeReferenceContractTest.production.Qwen3SparseMoeBlock(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=intermediate_size,
            norm_topk_prob=norm_topk_prob,
            dtype=torch.float32,
        )
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            block.gate.weight.copy_(
                torch.randn(
                    num_experts,
                    hidden_size,
                    generator=generator,
                )
            )
            block.experts.gate_up_weight.copy_(
                torch.randn(
                    num_experts,
                    hidden_size,
                    2 * intermediate_size,
                    generator=generator,
                )
            )
            block.experts.down_weight.copy_(
                torch.randn(
                    num_experts,
                    intermediate_size,
                    hidden_size,
                    generator=generator,
                )
            )
        return block

    def _assert_block_matches_reference(self, block, hidden_states):
        expected, logits, weights, expert_ids = moe_forward_reference(
            hidden_states.reshape(-1, block.hidden_size),
            block.gate.weight.detach(),
            block.experts.gate_up_weight.detach(),
            block.experts.down_weight.detach(),
            top_k=block.gate.top_k,
            norm_topk_prob=block.gate.norm_topk_prob,
        )
        actual = block(hidden_states)

        torch.testing.assert_close(
            actual,
            expected.reshape_as(actual),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(block.last_router_logits, logits)
        return weights, expert_ids

    def test_router_matrix_matches_current_production_contract(self):
        for tokens in (0, 1, 5):
            for num_experts in (1, 2, 4, 8):
                top_k_values = sorted({1, min(2, num_experts), num_experts})
                for top_k in top_k_values:
                    for norm_topk_prob in (False, True):
                        with self.subTest(
                            tokens=tokens,
                            num_experts=num_experts,
                            top_k=top_k,
                            norm_topk_prob=norm_topk_prob,
                        ):
                            generator = torch.Generator().manual_seed(
                                1000
                                + tokens * 100
                                + num_experts * 10
                                + top_k
                                + int(norm_topk_prob)
                            )
                            hidden_states = torch.randn(
                                tokens,
                                3,
                                generator=generator,
                            )
                            router_weight = torch.randn(
                                num_experts,
                                3,
                                generator=generator,
                            )
                            router_weight += torch.arange(
                                num_experts,
                                dtype=torch.float32,
                            ).unsqueeze(1) * 1e-4

                            expected = route_topk_reference(
                                hidden_states,
                                router_weight,
                                top_k=top_k,
                                norm_topk_prob=norm_topk_prob,
                            )
                            router = self.production.Qwen3MoeTopKRouter(
                                hidden_size=3,
                                num_experts=num_experts,
                                top_k=top_k,
                                norm_topk_prob=norm_topk_prob,
                                dtype=torch.float32,
                            )
                            router.weight.data.copy_(router_weight)
                            actual = router(hidden_states)

                            torch.testing.assert_close(actual[0], expected[0])
                            torch.testing.assert_close(actual[1], expected[1])
                            self.assertTrue(torch.equal(actual[2], expected[2]))
                            self.assertEqual(actual[1].dtype, actual[0].dtype)

    def test_reference_module_has_no_production_moe_imports(self):
        reference_path = ROOT / "tests/reference/moe_reference.py"
        source = reference_path.read_text(encoding="utf-8")
        module = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        self.assertEqual(
            imported_modules,
            {"__future__", "typing", "torch", "torch.nn.functional"},
        )
        for forbidden in (
            "lite_llama",
            "Qwen3MoeTopKRouter",
            "Qwen3MoeExperts",
            "moe_routing",
            "moe_routed_gemv",
            "grouped_matmul",
        ):
            self.assertNotIn(forbidden, source)

    def test_norm_topk_flag_controls_selected_weight_sum(self):
        hidden_states = torch.tensor([[1.0, 0.5]])
        router_weight = torch.tensor(
            [[2.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        )

        _, unnormalized, _ = route_topk_reference(
            hidden_states,
            router_weight,
            top_k=2,
            norm_topk_prob=False,
        )
        _, normalized, _ = route_topk_reference(
            hidden_states,
            router_weight,
            top_k=2,
            norm_topk_prob=True,
        )

        self.assertLess(unnormalized.sum().item(), 1.0)
        torch.testing.assert_close(normalized.sum(), torch.tensor(1.0))

    def test_empty_block_returns_empty_flat_token_shape(self):
        block = self._make_block()
        hidden_states = torch.empty(0, 4)

        weights, expert_ids = self._assert_block_matches_reference(
            block,
            hidden_states,
        )

        self.assertEqual(block(hidden_states).shape, (0, 4))
        self.assertEqual(weights.shape, (0, 2))
        self.assertEqual(expert_ids.shape, (0, 2))

    def test_single_and_multiple_tokens_match_independent_oracle(self):
        for tokens in (1, 7):
            for norm_topk_prob in (False, True):
                with self.subTest(
                    tokens=tokens,
                    norm_topk_prob=norm_topk_prob,
                ):
                    block = self._make_block(
                        norm_topk_prob=norm_topk_prob,
                        seed=200 + tokens + int(norm_topk_prob),
                    )
                    generator = torch.Generator().manual_seed(300 + tokens)
                    hidden_states = torch.randn(
                        tokens,
                        4,
                        generator=generator,
                    )
                    self._assert_block_matches_reference(block, hidden_states)

    def test_hot_expert_and_unselected_experts_match_oracle(self):
        block = self._make_block(top_k=1)
        with torch.no_grad():
            block.gate.weight.zero_()
            block.gate.weight[0].fill_(2.0)
            block.gate.weight[1:].fill_(-2.0)
        hidden_states = torch.ones(6, 4)

        _, expert_ids = self._assert_block_matches_reference(
            block,
            hidden_states,
        )

        self.assertTrue(torch.equal(expert_ids, torch.zeros_like(expert_ids)))
        self.assertEqual(set(expert_ids.flatten().tolist()), {0})

    def test_balanced_routing_hits_every_expert(self):
        block = self._make_block(
            hidden_size=4,
            num_experts=4,
            top_k=1,
        )
        with torch.no_grad():
            block.gate.weight.copy_(torch.eye(4) * 10.0)
        hidden_states = torch.eye(4)

        _, expert_ids = self._assert_block_matches_reference(
            block,
            hidden_states,
        )

        self.assertEqual(set(expert_ids.flatten().tolist()), {0, 1, 2, 3})

    def test_noncontiguous_hidden_input_matches_contiguous_input_and_oracle(self):
        block = self._make_block()
        generator = torch.Generator().manual_seed(401)
        storage = torch.randn(5, 8, generator=generator)
        hidden_states = storage[:, ::2]
        self.assertFalse(hidden_states.is_contiguous())

        expected, _, _, _ = moe_forward_reference(
            hidden_states,
            block.gate.weight.detach(),
            block.experts.gate_up_weight.detach(),
            block.experts.down_weight.detach(),
            top_k=2,
            norm_topk_prob=True,
        )
        noncontiguous_output = block(hidden_states)
        contiguous_output = block(hidden_states.contiguous())

        torch.testing.assert_close(noncontiguous_output, expected)
        torch.testing.assert_close(noncontiguous_output, contiguous_output)

    def test_extreme_finite_logits_remain_finite(self):
        hidden_states = torch.tensor([[1.0], [-1.0]])
        router_weight = torch.tensor([[1e4], [-1e4], [0.0], [1.0]])

        expected = route_topk_reference(
            hidden_states,
            router_weight,
            top_k=2,
            norm_topk_prob=True,
        )
        router = self.production.Qwen3MoeTopKRouter(
            hidden_size=1,
            num_experts=4,
            top_k=2,
            norm_topk_prob=True,
            dtype=torch.float32,
        )
        router.weight.data.copy_(router_weight)
        actual = router(hidden_states)

        self.assertTrue(torch.isfinite(actual[0]).all())
        self.assertTrue(torch.isfinite(actual[1]).all())
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_router_uses_fp32_softmax_and_casts_weights_back(self):
        router = self.production.Qwen3MoeTopKRouter(
            hidden_size=2,
            num_experts=4,
            top_k=2,
            norm_topk_prob=True,
            dtype=torch.bfloat16,
        )
        router.weight.data.copy_(
            torch.tensor(
                [[2.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                dtype=torch.bfloat16,
            )
        )
        hidden_states = torch.tensor(
            [[1.0, 0.5]],
            dtype=torch.bfloat16,
        )
        observed_softmax_dtypes = []
        original_softmax = torch.nn.functional.softmax

        def record_softmax_dtype(tensor, dim):
            observed_softmax_dtypes.append(tensor.dtype)
            return original_softmax(tensor, dim=dim)

        with mock.patch.object(
            self.production.F,
            "softmax",
            side_effect=record_softmax_dtype,
        ):
            logits, weights, _ = router(hidden_states)

        self.assertEqual(observed_softmax_dtypes, [torch.float32])
        self.assertEqual(logits.dtype, torch.bfloat16)
        self.assertEqual(weights.dtype, logits.dtype)

    def test_non_tie_routing_is_deterministic_for_fixed_inputs(self):
        block = self._make_block(seed=501)
        hidden_states = torch.randn(
            6,
            4,
            generator=torch.Generator().manual_seed(502),
        )

        output1 = block(hidden_states)
        logits1, weights1, ids1 = block.gate(hidden_states)
        output2 = block(hidden_states)
        logits2, weights2, ids2 = block.gate(hidden_states)

        self.assertTrue(torch.equal(ids1, ids2))
        self.assertTrue(torch.equal(logits1, logits2))
        self.assertTrue(torch.equal(weights1, weights2))
        self.assertTrue(torch.equal(output1, output2))

    def test_tied_router_does_not_bind_expert_id_order(self):
        block = self._make_block(num_experts=4, top_k=2)
        with torch.no_grad():
            block.gate.weight.zero_()
            block.experts.gate_up_weight.copy_(
                block.experts.gate_up_weight[0:1].expand_as(
                    block.experts.gate_up_weight
                )
            )
            block.experts.down_weight.copy_(
                block.experts.down_weight[0:1].expand_as(
                    block.experts.down_weight
                )
            )
        hidden_states = torch.randn(
            3,
            4,
            generator=torch.Generator().manual_seed(601),
        )

        weights, expert_ids = self._assert_block_matches_reference(
            block,
            hidden_states,
        )

        self.assertTrue(((0 <= expert_ids) & (expert_ids < 4)).all())
        for row in expert_ids:
            self.assertEqual(len(set(row.tolist())), 2)
        torch.testing.assert_close(
            weights,
            torch.full_like(weights, 0.5),
        )

    def test_tensor_parallel_intermediate_partials_sum_to_full_oracle(self):
        generator = torch.Generator().manual_seed(701)
        hidden_states = torch.randn(5, 4, generator=generator)
        router_weight = torch.randn(4, 4, generator=generator)
        gate_up_weight = torch.randn(4, 4, 8, generator=generator)
        down_weight = torch.randn(4, 4, 4, generator=generator)
        _, routing_weights, expert_ids = route_topk_reference(
            hidden_states,
            router_weight,
            top_k=2,
            norm_topk_prob=True,
        )
        full = expert_forward_reference(
            hidden_states,
            expert_ids,
            routing_weights,
            gate_up_weight,
            down_weight,
        )

        partials = []
        intermediate_size = down_weight.shape[1]
        for rank in range(2):
            start = rank * (intermediate_size // 2)
            end = start + intermediate_size // 2
            local_gate_up = torch.cat(
                (
                    gate_up_weight[:, :, start:end],
                    gate_up_weight[
                        :,
                        :,
                        intermediate_size + start : intermediate_size + end,
                    ],
                ),
                dim=-1,
            )
            local_down = down_weight[:, start:end, :]
            partials.append(
                expert_forward_reference(
                    hidden_states,
                    expert_ids,
                    routing_weights,
                    local_gate_up,
                    local_down,
                )
            )

        torch.testing.assert_close(
            partials[0] + partials[1],
            full,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_expert_parallel_local_contributions_sum_to_full_oracle(self):
        generator = torch.Generator().manual_seed(801)
        hidden_states = torch.randn(6, 4, generator=generator)
        router_weight = torch.randn(4, 4, generator=generator)
        gate_up_weight = torch.randn(4, 4, 6, generator=generator)
        down_weight = torch.randn(4, 3, 4, generator=generator)
        _, routing_weights, expert_ids = route_topk_reference(
            hidden_states,
            router_weight,
            top_k=2,
            norm_topk_prob=True,
        )
        full = expert_forward_reference(
            hidden_states,
            expert_ids,
            routing_weights,
            gate_up_weight,
            down_weight,
        )
        rank0 = expert_forward_reference(
            hidden_states,
            expert_ids,
            routing_weights,
            gate_up_weight[:2],
            down_weight[:2],
            expert_start=0,
            local_num_experts=2,
        )
        rank1 = expert_forward_reference(
            hidden_states,
            expert_ids,
            routing_weights,
            gate_up_weight[2:],
            down_weight[2:],
            expert_start=2,
            local_num_experts=2,
        )

        torch.testing.assert_close(rank0 + rank1, full)

    def test_reference_is_inference_only(self):
        hidden_states = torch.randn(2, 3, requires_grad=True)
        router_weight = torch.randn(2, 3, requires_grad=True)
        gate_up_weight = torch.randn(2, 3, 4, requires_grad=True)
        down_weight = torch.randn(2, 2, 3, requires_grad=True)

        output, logits, weights, _ = moe_forward_reference(
            hidden_states,
            router_weight,
            gate_up_weight,
            down_weight,
            top_k=1,
            norm_topk_prob=True,
        )

        self.assertFalse(output.requires_grad)
        self.assertFalse(logits.requires_grad)
        self.assertFalse(weights.requires_grad)

    def test_reference_rejects_invalid_top_k(self):
        hidden_states = torch.empty(0, 3)
        router_weight = torch.empty(4, 3)

        for top_k in (0, 5):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "top_k"):
                    route_topk_reference(
                        hidden_states,
                        router_weight,
                        top_k=top_k,
                        norm_topk_prob=True,
                    )


if __name__ == "__main__":
    unittest.main()
