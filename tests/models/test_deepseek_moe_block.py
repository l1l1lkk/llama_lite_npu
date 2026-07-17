import ast
import hashlib
import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from tests.reference.deepseek_moe_reference import (
    DeepSeekRoutingSpec,
    deepseek_moe_reference,
    deepseek_shared_expert_reference,
)


ROOT = Path(__file__).resolve().parents[2]
MOE_PATH = ROOT / "lite_llama/models/moe.py"


def _load_production_moe():
    name = "deepseek_moe_block_contract_production"
    spec = importlib.util.spec_from_file_location(name, MOE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tp_config(world_size, rank, mode):
    return types.SimpleNamespace(
        world_size=world_size,
        rank=rank,
        moe_parallel_mode=mode,
        enabled=world_size > 1,
    )


def _slice_shared_gate_up(weight, start, end):
    gate, up = weight.chunk(2, dim=-1)
    return torch.cat((gate[:, start:end], up[:, start:end]), dim=-1)


def _routing_spec(version):
    if version == "v2":
        return DeepSeekRoutingSpec(
            num_experts=4,
            experts_per_token=2,
            num_groups=2,
            topk_groups=1,
            score_func="softmax",
            topk_method="group_limited_greedy",
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
    return DeepSeekRoutingSpec(
        num_experts=4,
        experts_per_token=2,
        num_groups=2,
        topk_groups=1,
        score_func="sigmoid",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=1.25,
    )


def _make_block(production, spec, hidden_size=3, dtype=torch.float32):
    return production.DeepSeekMoeBlock(
        hidden_size=hidden_size,
        num_experts=spec.num_experts,
        top_k=spec.experts_per_token,
        intermediate_size=2,
        shared_intermediate_size=3,
        num_groups=spec.num_groups,
        topk_groups=spec.topk_groups,
        score_func=spec.score_func,
        topk_method=spec.topk_method,
        norm_topk_prob=spec.norm_topk_prob,
        routed_scaling_factor=spec.routed_scaling_factor,
        dtype=dtype,
    )


def _load_block_weights(block, generator):
    with torch.no_grad():
        block.gate.weight.copy_(
            torch.randn(block.gate.weight.shape, generator=generator)
        )
        if hasattr(block.gate, "e_score_correction_bias"):
            block.gate.e_score_correction_bias.copy_(
                torch.randn(
                    block.gate.e_score_correction_bias.shape,
                    generator=generator,
                )
                * 0.05
            )
        block.experts.gate_up_weight.copy_(
            torch.randn(
                block.experts.gate_up_weight.shape,
                generator=generator,
            )
        )
        block.experts.down_weight.copy_(
            torch.randn(
                block.experts.down_weight.shape,
                generator=generator,
            )
        )
        block.shared_experts.gate_up_weight.copy_(
            torch.randn(
                block.shared_experts.gate_up_weight.shape,
                generator=generator,
            )
        )
        block.shared_experts.down_weight.copy_(
            torch.randn(
                block.shared_experts.down_weight.shape,
                generator=generator,
            )
        )


class DeepSeekMoeBlockContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.production = _load_production_moe()

    def test_shared_expert_placement_world_one_and_tp_slices(self):
        placement = self.production.SharedExpertPlacement.from_config(
            shared_intermediate_size=8,
        )
        self.assertEqual(
            tuple(placement),
            ("tp", 1, 0, 8, 8, 0, 8, False),
        )
        with self.assertRaises(AttributeError):
            placement.rank = 1

        expected = ((0, 0, 4), (1, 4, 8))
        for rank, start, end in expected:
            with self.subTest(rank=rank):
                placement = self.production.SharedExpertPlacement.from_config(
                    shared_intermediate_size=8,
                    tp_config=_tp_config(2, rank, "tp"),
                )
                self.assertEqual(placement.local_intermediate_size, 4)
                self.assertEqual(placement.intermediate_start, start)
                self.assertEqual(placement.intermediate_end, end)
                self.assertTrue(placement.reduce_output)

    def test_shared_expert_placement_ep_replicates_complete_expert(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                placement = self.production.SharedExpertPlacement.from_config(
                    shared_intermediate_size=7,
                    tp_config=_tp_config(2, rank, "ep"),
                )
                self.assertEqual(placement.parallel_mode, "ep")
                self.assertEqual(placement.local_intermediate_size, 7)
                self.assertEqual(placement.intermediate_start, 0)
                self.assertEqual(placement.intermediate_end, 7)
                self.assertFalse(placement.reduce_output)

    def test_shared_expert_placement_rejects_invalid_configuration(self):
        cases = (
            (
                dict(
                    shared_intermediate_size=8,
                    tp_config=_tp_config(2, 0, "unknown"),
                ),
                "moe_parallel_mode",
            ),
            (
                dict(
                    shared_intermediate_size=8,
                    tp_config=_tp_config(0, 0, "tp"),
                ),
                "world_size",
            ),
            (
                dict(
                    shared_intermediate_size=8,
                    tp_config=_tp_config(2, 2, "tp"),
                ),
                "rank",
            ),
            (
                dict(shared_intermediate_size=0),
                "shared_intermediate_size",
            ),
            (
                dict(
                    shared_intermediate_size=7,
                    tp_config=_tp_config(2, 0, "tp"),
                ),
                "divisible",
            ),
        )
        for kwargs, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.production.SharedExpertPlacement.from_config(**kwargs)

    def test_shared_local_fp32_matches_independent_reference(self):
        generator = torch.Generator().manual_seed(701)
        hidden = torch.randn(5, 3, generator=generator)
        module = self.production.SharedExpertMLP(
            hidden_size=3,
            shared_intermediate_size=4,
            dtype=torch.float32,
        )
        with torch.no_grad():
            module.gate_up_weight.copy_(
                torch.randn(3, 8, generator=generator)
            )
            module.down_weight.copy_(
                torch.randn(4, 3, generator=generator)
            )
        expected = deepseek_shared_expert_reference(
            hidden,
            module.gate_up_weight.detach(),
            module.down_weight.detach(),
        )
        actual = module(hidden)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(
            set(module.state_dict()),
            {"gate_up_weight", "down_weight"},
        )

    def test_tp_shared_partials_sum_to_independent_full_reference(self):
        generator = torch.Generator().manual_seed(702)
        hidden = torch.randn(4, 3, generator=generator)
        full_gate_up = torch.randn(3, 8, generator=generator)
        full_down = torch.randn(4, 3, generator=generator)
        expected = deepseek_shared_expert_reference(
            hidden,
            full_gate_up,
            full_down,
        )
        partials = []
        for rank, (start, end) in enumerate(((0, 2), (2, 4))):
            module = self.production.SharedExpertMLP(
                hidden_size=3,
                shared_intermediate_size=4,
                tp_config=_tp_config(2, rank, "tp"),
                dtype=torch.float32,
            )
            with torch.no_grad():
                module.gate_up_weight.copy_(
                    _slice_shared_gate_up(full_gate_up, start, end)
                )
                module.down_weight.copy_(full_down[start:end])
            local_expected = deepseek_shared_expert_reference(
                hidden,
                _slice_shared_gate_up(full_gate_up, start, end),
                full_down[start:end],
            )
            local_actual = module._forward_local(hidden)
            torch.testing.assert_close(
                local_actual,
                local_expected,
                rtol=1e-6,
                atol=1e-6,
            )
            partials.append(local_actual)
        torch.testing.assert_close(
            partials[0] + partials[1],
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_ep_shared_is_replicated_and_must_not_be_rank_summed(self):
        generator = torch.Generator().manual_seed(703)
        hidden = torch.randn(4, 3, generator=generator)
        gate_up = torch.randn(3, 8, generator=generator)
        down = torch.randn(4, 3, generator=generator)
        expected = deepseek_shared_expert_reference(hidden, gate_up, down)
        outputs = []
        for rank in range(2):
            module = self.production.SharedExpertMLP(
                hidden_size=3,
                shared_intermediate_size=4,
                tp_config=_tp_config(2, rank, "ep"),
                dtype=torch.float32,
            )
            with torch.no_grad():
                module.gate_up_weight.copy_(gate_up)
                module.down_weight.copy_(down)
            outputs.append(module(hidden))
            torch.testing.assert_close(
                outputs[-1],
                expected,
                rtol=1e-6,
                atol=1e-6,
            )
        self.assertFalse(
            torch.allclose(outputs[0] + outputs[1], expected)
        )

    def test_shared_reduce_helper_call_count_matches_ownership(self):
        hidden = torch.ones(2, 3)

        def make_module(world_size, rank, mode):
            module = self.production.SharedExpertMLP(
                hidden_size=3,
                shared_intermediate_size=4,
                tp_config=_tp_config(world_size, rank, mode),
                dtype=torch.float32,
            )
            with torch.no_grad():
                module.gate_up_weight.fill_(0.25)
                module.down_weight.fill_(0.5)
            return module

        tp_module = make_module(2, 0, "tp")
        tp_local = tp_module._forward_local(hidden)
        with mock.patch.object(
            self.production,
            "_shared_expert_all_reduce",
            side_effect=lambda tensor: tensor + 1,
        ) as reduce_mock:
            actual = tp_module(hidden)
        reduce_mock.assert_called_once()
        torch.testing.assert_close(reduce_mock.call_args.args[0], tp_local)
        torch.testing.assert_close(actual, tp_local + 1)

        for module in (make_module(2, 0, "ep"), make_module(1, 0, "tp")):
            with self.subTest(placement=module.placement):
                with mock.patch.object(
                    self.production,
                    "_shared_expert_all_reduce",
                    side_effect=lambda tensor: tensor + 1,
                ) as reduce_mock:
                    actual = module(hidden)
                reduce_mock.assert_not_called()
                torch.testing.assert_close(
                    actual,
                    module._forward_local(hidden),
                )

    def test_v2_and_v3_blocks_match_independent_full_moe_reference(self):
        for version, seed in (("v2", 704), ("v3", 705)):
            with self.subTest(version=version):
                generator = torch.Generator().manual_seed(seed)
                spec = _routing_spec(version)
                block = _make_block(self.production, spec)
                _load_block_weights(block, generator)
                hidden = torch.randn(5, 3, generator=generator)
                correction_bias = getattr(
                    block.gate,
                    "e_score_correction_bias",
                    None,
                )
                expected, expected_logits, _, _ = deepseek_moe_reference(
                    hidden,
                    block.gate.weight.detach(),
                    block.experts.gate_up_weight.detach(),
                    block.experts.down_weight.detach(),
                    spec,
                    correction_bias=correction_bias,
                    shared_gate_up_weight=(
                        block.shared_experts.gate_up_weight.detach()
                    ),
                    shared_down_weight=(
                        block.shared_experts.down_weight.detach()
                    ),
                )
                captured_logits = []
                handle = block.gate.register_forward_hook(
                    lambda _module, _inputs, output: captured_logits.append(
                        output.router_logits
                    )
                )
                try:
                    actual = block(hidden)
                finally:
                    handle.remove()
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-5,
                    atol=1e-6,
                )
                torch.testing.assert_close(
                    block.last_router_logits.float(),
                    expected_logits,
                )
                self.assertIs(block.last_router_logits, captured_logits[0])

    def test_block_restores_2d_3d_and_empty_shapes(self):
        generator = torch.Generator().manual_seed(706)
        block = _make_block(self.production, _routing_spec("v2"))
        _load_block_weights(block, generator)
        hidden_3d = torch.randn(2, 3, 3, generator=generator)
        flat_output = block(hidden_3d.reshape(-1, 3))
        output_3d = block(hidden_3d)
        self.assertEqual(output_3d.shape, hidden_3d.shape)
        torch.testing.assert_close(
            output_3d.reshape(-1, 3),
            flat_output,
        )

        empty = block(torch.empty(0, 3))
        self.assertEqual(empty.shape, (0, 3))
        self.assertEqual(block.last_router_logits.shape, (0, 4))

    def test_block_state_dict_keys_match_v2_and_v3_contract(self):
        common = {
            "gate.weight",
            "experts.gate_up_weight",
            "experts.down_weight",
            "shared_experts.gate_up_weight",
            "shared_experts.down_weight",
        }
        v2 = _make_block(self.production, _routing_spec("v2"))
        v3 = _make_block(self.production, _routing_spec("v3"))
        self.assertEqual(set(v2.state_dict()), common)
        self.assertEqual(
            set(v3.state_dict()),
            common | {"gate.e_score_correction_bias"},
        )

    def test_block_has_no_reduce_and_existing_classes_are_ast_stable(self):
        tree = ast.parse(MOE_PATH.read_text(encoding="utf-8"))
        classes = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }
        expected_hashes = {
            "RoutingResult": (
                "db31c71494c427aebd0ad41683add7d5aefabb36585fce309002614e357064d7"
            ),
            "GroupedTopKConfig": (
                "d34fc7077903aeb238d436d6323585b08c85ffeead780320405925751010a677"
            ),
            "ExpertPlacement": (
                "fde0e98285e682f9c717ee1e9f778c136523f346fc418661500d5c7de396dd3f"
            ),
            "SoftmaxTopKRouter": (
                "eb52ac48af02afce52b27a0b7c512dc43cdd5e54b8d4d37c65ed8e02e6573420"
            ),
            "Qwen3MoeTopKRouter": (
                "a672fa62d25e1a59813d70d38ccb111de034c1df8e25df7f5ba606a6ff4f2b07"
            ),
            "DeepSeekGroupedTopKRouter": (
                "44015d4eed3ce171c00114459c24d81de2f76252fafa3c76bd2ab198263d90bc"
            ),
            "RoutedExpertExecutor": (
                "c605e75330bcabec3f313d33dcce3057288e10b3c647ea0ff111669ce190159c"
            ),
            "Qwen3MoeExperts": (
                "3298abe2690dcb7e86bde0dd277f94333444e7757166f4d9603e66dbb2daa0d0"
            ),
            "Qwen3SparseMoeBlock": (
                "13c9e70cc1d9d66d8060f723eddd2f1d2712dcf58c15775d28d29308a24a3148"
            ),
        }
        for name, expected in expected_hashes.items():
            with self.subTest(name=name):
                actual = hashlib.sha256(
                    ast.dump(
                        classes[name],
                        include_attributes=False,
                    ).encode()
                ).hexdigest()
                self.assertEqual(actual, expected)

        block_forward = next(
            node
            for node in classes["DeepSeekMoeBlock"].body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        source = ast.unparse(block_forward)
        self.assertNotIn("tp_all_reduce", source)
        self.assertNotIn("_shared_expert_all_reduce", source)
        self.assertEqual(
            sum(isinstance(node, ast.Add) for node in ast.walk(block_forward)),
            1,
        )
        shared_forward = next(
            node
            for node in classes["SharedExpertMLP"].body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        self.assertEqual(
            ast.unparse(shared_forward).count("_shared_expert_all_reduce("),
            1,
        )
        self.assertNotIn(".item(", ast.unparse(shared_forward))
        self.assertNotIn(".tolist(", ast.unparse(shared_forward))

    def test_dependency_light_loader_exposes_complete_block_boundary(self):
        self.assertTrue(inspect.isclass(self.production.SharedExpertPlacement))
        self.assertTrue(inspect.isclass(self.production.SharedExpertMLP))
        self.assertTrue(inspect.isclass(self.production.DeepSeekMoeBlock))
        block = _make_block(self.production, _routing_spec("v2"))
        _load_block_weights(block, torch.Generator().manual_seed(707))
        self.assertEqual(block(torch.ones(1, 3)).shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
