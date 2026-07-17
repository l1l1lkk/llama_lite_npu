"""Ascend 910B3 correctness regression for DeepSeek-V2/V3 MoE.

Run this hardware gate explicitly on the server:

ASCEND_RT_VISIBLE_DEVICES=6 LITE_LLAMA_MOE_BACKEND=gmm \
  LITE_LLAMA_MOE_VALIDATE=1 python -m unittest \
  tests.npu.test_deepseek_moe -v

The test is a single-card, small-shape component regression.  It does not
exercise decode Graph, TP/EP collectives, checkpoint files, or a full model.
"""

import os
import unittest
from unittest import mock

import torch


try:
    import torch_npu

    HAS_NPU_GMM = (
        torch.npu.is_available()
        and hasattr(torch_npu, "npu_grouped_matmul")
    )
except (ImportError, AttributeError):
    HAS_NPU_GMM = False


T = 4
H = 64
E = 8
K = 2
I = 32
SHARED_I = 32
NUM_GROUPS = 2
TOPK_GROUPS = 1
RTOL = 1e-2
ATOL = 1e-2


def _routing_spec(version):
    from tests.reference.deepseek_moe_reference import DeepSeekRoutingSpec

    if version == "v2":
        return DeepSeekRoutingSpec(
            num_experts=E,
            experts_per_token=K,
            num_groups=NUM_GROUPS,
            topk_groups=TOPK_GROUPS,
            score_func="softmax",
            topk_method="group_limited_greedy",
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
    return DeepSeekRoutingSpec(
        num_experts=E,
        experts_per_token=K,
        num_groups=NUM_GROUPS,
        topk_groups=TOPK_GROUPS,
        score_func="sigmoid",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )


def _component_config(version):
    from lite_llama.models.model_config import DeepSeekMoeConfig

    is_v3 = version == "v3"
    return DeepSeekMoeConfig(
        architectures=[
            "DeepseekV3ForCausalLM" if is_v3 else "DeepseekV2ForCausalLM"
        ],
        model_type="deepseek_v3" if is_v3 else "deepseek_v2",
        hidden_size=H,
        num_layers=2,
        intermediate_size=128,
        num_experts=E,
        num_experts_per_tok=K,
        moe_intermediate_size=I,
        num_shared_experts=1,
        score_func="sigmoid" if is_v3 else "softmax",
        topk_method="noaux_tc" if is_v3 else "group_limited_greedy",
        num_groups=NUM_GROUPS,
        topk_groups=TOPK_GROUPS,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        torch_dtype="bfloat16",
    )


def _desired_logits(version):
    if version == "v2":
        return torch.tensor(
            [
                [4.0, 3.0, 1.0, 0.0, 0.0, -1.0, -2.0, -3.0],
                [0.0, -1.0, -2.0, -3.0, 4.0, 3.0, 1.0, 0.0],
                [1.0, 4.0, 3.0, 0.0, 0.0, -1.0, -2.0, -3.0],
                [0.0, -1.0, -2.0, -3.0, 1.0, 0.0, 4.0, 3.0],
            ],
            dtype=torch.float32,
        )
    return torch.tensor(
        [
            [4.0, 3.0, 1.0, 0.0, 1.5, 1.0, 0.0, -1.0],
            [1.0, 4.0, 3.0, 0.0, 1.0, 1.5, 0.0, -1.0],
            [1.0, 0.0, 4.0, 3.0, -1.0, 0.5, 1.5, 1.0],
            [1.0, 0.0, 3.0, 4.0, -1.0, 0.0, 1.0, 1.5],
        ],
        dtype=torch.float32,
    )


def _case_tensors(version, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.zeros(T, H, dtype=torch.float32)
    hidden[:, :T] = torch.eye(T, dtype=torch.float32)
    router_weight = torch.zeros(E, H, dtype=torch.float32)
    router_weight[:, :T] = _desired_logits(version).transpose(0, 1)

    def quantized_randn(shape):
        return (
            torch.randn(shape, generator=generator, dtype=torch.float32)
            .mul_(0.2)
            .to(dtype)
            .contiguous()
        )

    canonical = {
        "layers.0.mlp.gate.weight": router_weight.to(dtype).contiguous(),
        "layers.0.mlp.experts.gate_up_weight": quantized_randn(
            (E, 2 * I, H)
        ),
        "layers.0.mlp.experts.down_weight": quantized_randn((E, H, I)),
        "layers.0.mlp.shared_experts.gate_up_weight": quantized_randn(
            (2 * SHARED_I, H)
        ),
        "layers.0.mlp.shared_experts.down_weight": quantized_randn(
            (H, SHARED_I)
        ),
    }
    if version == "v3":
        canonical["layers.0.mlp.gate.e_score_correction_bias"] = torch.tensor(
            [-2.0] * 4 + [2.0] * 4,
            dtype=torch.float32,
        )
    return hidden.to(dtype).contiguous(), canonical


def _runtime_reference_weights(canonical):
    return {
        "router": canonical["layers.0.mlp.gate.weight"],
        "routed_gate_up": canonical[
            "layers.0.mlp.experts.gate_up_weight"
        ].permute(0, 2, 1).contiguous(),
        "routed_down": canonical[
            "layers.0.mlp.experts.down_weight"
        ].transpose(1, 2).contiguous(),
        "shared_gate_up": canonical[
            "layers.0.mlp.shared_experts.gate_up_weight"
        ].transpose(0, 1).contiguous(),
        "shared_down": canonical[
            "layers.0.mlp.shared_experts.down_weight"
        ].transpose(0, 1).contiguous(),
    }


def _selection_margins(logits, selected_experts, version, correction_bias):
    original = (
        torch.softmax(logits.float(), dim=-1)
        if version == "v2"
        else torch.sigmoid(logits.float())
    )
    selection = original if correction_bias is None else original + correction_bias
    grouped = selection.reshape(T, NUM_GROUPS, E // NUM_GROUPS)
    if version == "v2":
        group_scores = grouped.max(dim=-1).values
    else:
        group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    group_margin = (
        group_scores.topk(2, dim=-1).values[:, 0]
        - group_scores.topk(2, dim=-1).values[:, 1]
    )

    expert_margin = []
    experts_per_group = E // NUM_GROUPS
    for token in range(T):
        group_id = int(selected_experts[token, 0]) // experts_per_group
        values = grouped[token, group_id].sort(descending=True).values
        expert_margin.append(values[K - 1] - values[K])
    return group_margin, torch.stack(expert_margin)


class DeepSeekMoeFixtureContractTest(unittest.TestCase):
    def test_reference_signals_exceed_absolute_tolerance(self):
        from tests.reference.deepseek_moe_reference import (
            deepseek_moe_reference,
            deepseek_routed_experts_reference,
            deepseek_shared_expert_reference,
        )

        cases = (
            ("v2", torch.float16, 20260717),
            ("v2", torch.bfloat16, 20260818),
            ("v3", torch.float16, 20260919),
            ("v3", torch.bfloat16, 20261020),
        )
        for version, dtype, seed in cases:
            spec = _routing_spec(version)
            hidden, canonical = _case_tensors(version, dtype, seed)
            weights = _runtime_reference_weights(canonical)
            correction_bias = canonical.get(
                "layers.0.mlp.gate.e_score_correction_bias"
            )
            full, _logits, routing_weights, selected_experts = (
                deepseek_moe_reference(
                    hidden,
                    weights["router"],
                    weights["routed_gate_up"],
                    weights["routed_down"],
                    spec,
                    correction_bias=correction_bias,
                    shared_gate_up_weight=weights["shared_gate_up"],
                    shared_down_weight=weights["shared_down"],
                )
            )
            routed = deepseek_routed_experts_reference(
                hidden,
                selected_experts,
                routing_weights,
                weights["routed_gate_up"],
                weights["routed_down"],
            )
            shared = deepseek_shared_expert_reference(
                hidden,
                weights["shared_gate_up"],
                weights["shared_down"],
            )

            for signal_name, expected in (
                ("routed", routed),
                ("shared", shared),
                ("full", full),
            ):
                with self.subTest(
                    version=version,
                    dtype=str(dtype),
                    signal=signal_name,
                ):
                    signal_max = float(expected.abs().max())
                    self.assertGreater(
                        signal_max,
                        2 * ATOL,
                        f"{signal_name} signal max {signal_max} is too small",
                    )
                    with self.assertRaises(AssertionError):
                        torch.testing.assert_close(
                            torch.zeros_like(expected),
                            expected,
                            rtol=RTOL,
                            atol=ATOL,
                        )


@unittest.skipUnless(HAS_NPU_GMM, "requires torch_npu GMM on an NPU")
class DeepSeekMoeNPURegressionTest(unittest.TestCase):
    def setUp(self):
        self._old_environment = {
            name: os.environ.get(name)
            for name in ("LITE_LLAMA_MOE_BACKEND", "LITE_LLAMA_MOE_VALIDATE")
        }
        os.environ["LITE_LLAMA_MOE_BACKEND"] = "gmm"
        os.environ["LITE_LLAMA_MOE_VALIDATE"] = "1"
        torch.npu.set_device(0)

    def tearDown(self):
        for name, value in self._old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_v2_v3_fp16_bf16_match_independent_reference(self):
        from lite_llama.models.deepseek_moe import (
            build_deepseek_moe_block,
            prepare_deepseek_moe_layer_state,
        )
        from tests.reference.deepseek_moe_reference import (
            deepseek_moe_reference,
            deepseek_route_reference,
            deepseek_routed_experts_reference,
            deepseek_shared_expert_reference,
        )

        cases = (
            ("v2", torch.float16, 20260717),
            ("v2", torch.bfloat16, 20260818),
            ("v3", torch.float16, 20260919),
            ("v3", torch.bfloat16, 20261020),
        )
        for version, dtype, seed in cases:
            with self.subTest(version=version, dtype=str(dtype), seed=seed):
                config = _component_config(version)
                spec = _routing_spec(version)
                hidden_cpu, canonical = _case_tensors(version, dtype, seed)
                expected_weights = _runtime_reference_weights(canonical)
                correction_bias = canonical.get(
                    "layers.0.mlp.gate.e_score_correction_bias"
                )

                runtime_state = prepare_deepseek_moe_layer_state(
                    canonical,
                    config,
                    layer_index=0,
                )
                torch.testing.assert_close(
                    runtime_state["experts.gate_up_weight"],
                    expected_weights["routed_gate_up"],
                    rtol=0.0,
                    atol=0.0,
                )
                torch.testing.assert_close(
                    runtime_state["experts.down_weight"],
                    expected_weights["routed_down"],
                    rtol=0.0,
                    atol=0.0,
                )
                torch.testing.assert_close(
                    runtime_state["shared_experts.gate_up_weight"],
                    expected_weights["shared_gate_up"],
                    rtol=0.0,
                    atol=0.0,
                )
                torch.testing.assert_close(
                    runtime_state["shared_experts.down_weight"],
                    expected_weights["shared_down"],
                    rtol=0.0,
                    atol=0.0,
                )

                block = build_deepseek_moe_block(
                    config,
                    layer_index=0,
                    dtype=dtype,
                )
                incompatible = block.load_state_dict(runtime_state, strict=True)
                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])
                expected_keys = {
                    "gate.weight",
                    "experts.gate_up_weight",
                    "experts.down_weight",
                    "shared_experts.gate_up_weight",
                    "shared_experts.down_weight",
                }
                if version == "v3":
                    expected_keys.add("gate.e_score_correction_bias")
                    self.assertEqual(
                        block.gate.e_score_correction_bias.dtype,
                        torch.float32,
                    )
                self.assertEqual(set(block.state_dict()), expected_keys)
                block = block.npu()
                hidden_npu = hidden_cpu.npu()

                reference = deepseek_moe_reference(
                    hidden_cpu,
                    expected_weights["router"],
                    expected_weights["routed_gate_up"],
                    expected_weights["routed_down"],
                    spec,
                    correction_bias=correction_bias,
                    shared_gate_up_weight=expected_weights["shared_gate_up"],
                    shared_down_weight=expected_weights["shared_down"],
                )
                reference_full, reference_logits, reference_routing, reference_ids = (
                    reference
                )
                reference_routed = deepseek_routed_experts_reference(
                    hidden_cpu,
                    reference_ids,
                    reference_routing,
                    expected_weights["routed_gate_up"],
                    expected_weights["routed_down"],
                )
                reference_shared = deepseek_shared_expert_reference(
                    hidden_cpu,
                    expected_weights["shared_gate_up"],
                    expected_weights["shared_down"],
                )

                with torch.inference_mode():
                    production_routing = block.gate(hidden_npu)
                torch.npu.synchronize()
                self.assertEqual(
                    production_routing.router_logits.shape,
                    (T, E),
                )
                self.assertEqual(
                    production_routing.router_logits.dtype,
                    dtype,
                )
                self.assertTrue(
                    torch.isfinite(production_routing.router_logits).all()
                )
                self.assertEqual(
                    production_routing.routing_weights.shape,
                    (T, K),
                )
                self.assertEqual(
                    production_routing.routing_weights.dtype,
                    dtype,
                )
                self.assertTrue(
                    torch.isfinite(production_routing.routing_weights).all()
                )
                self.assertEqual(
                    production_routing.selected_experts.shape,
                    (T, K),
                )
                self.assertEqual(
                    production_routing.selected_experts.dtype,
                    torch.int64,
                )
                torch.testing.assert_close(
                    production_routing.router_logits.cpu().float(),
                    reference_logits,
                    rtol=RTOL,
                    atol=ATOL,
                )
                torch.testing.assert_close(
                    production_routing.routing_weights.cpu().float(),
                    reference_routing,
                    rtol=RTOL,
                    atol=ATOL,
                )
                torch.testing.assert_close(
                    production_routing.selected_experts.cpu(),
                    reference_ids,
                    rtol=0.0,
                    atol=0.0,
                )

                group_margin, expert_margin = _selection_margins(
                    reference_logits,
                    reference_ids,
                    version,
                    correction_bias,
                )
                self.assertGreater(float(group_margin.min()), 0.0)
                self.assertGreater(float(expert_margin.min()), 0.0)

                if version == "v3":
                    zero_bias_route = deepseek_route_reference(
                        hidden_cpu,
                        expected_weights["router"],
                        spec,
                        correction_bias=torch.zeros(E, dtype=torch.float32),
                    )
                    self.assertFalse(
                        torch.equal(zero_bias_route[2], reference_ids)
                    )
                    original_scores = torch.sigmoid(reference_logits)
                    original_selected = original_scores.gather(1, reference_ids)
                    expected_combine = original_selected / original_selected.sum(
                        dim=-1,
                        keepdim=True,
                    )
                    torch.testing.assert_close(
                        reference_routing,
                        expected_combine,
                        rtol=1e-6,
                        atol=1e-6,
                    )
                    biased_selected = (
                        original_scores + correction_bias
                    ).gather(1, reference_ids)
                    biased_combine = biased_selected / biased_selected.sum(
                        dim=-1,
                        keepdim=True,
                    )
                    self.assertGreater(
                        float((biased_combine - reference_routing).abs().max()),
                        0.01,
                    )

                captured_logits = []
                hook = block.gate.register_forward_hook(
                    lambda _module, _inputs, output: captured_logits.append(
                        output.router_logits
                    )
                )
                try:
                    with (
                        mock.patch.object(
                            torch_npu,
                            "npu_grouped_matmul",
                            wraps=torch_npu.npu_grouped_matmul,
                        ) as gmm_mock,
                        mock.patch.object(
                            block.experts,
                            "_forward_grouped_local",
                            wraps=block.experts._forward_grouped_local,
                        ) as grouped_mock,
                        mock.patch.object(
                            block.experts,
                            "_forward_eager_local",
                            wraps=block.experts._forward_eager_local,
                        ) as eager_mock,
                        mock.patch.object(
                            block.experts,
                            "_use_grouped_backend",
                            wraps=block.experts._use_grouped_backend,
                        ) as backend_mock,
                        torch.inference_mode(),
                    ):
                        direct = block.experts._forward_grouped_local(
                            hidden_npu,
                            production_routing.selected_experts,
                            production_routing.routing_weights,
                        )
                        public = block.experts(
                            hidden_npu,
                            production_routing.selected_experts,
                            production_routing.routing_weights,
                        )
                        shared = block.shared_experts._forward_local(hidden_npu)
                        full = block(hidden_npu)
                    torch.npu.synchronize()

                    self.assertEqual(block.experts.backend, "gmm")
                    self.assertEqual(gmm_mock.call_count, 6)
                    self.assertEqual(grouped_mock.call_count, 3)
                    self.assertEqual(backend_mock.call_count, 2)
                    self.assertEqual(eager_mock.call_count, 2)
                    self.assertIs(block.last_router_logits, captured_logits[-1])
                finally:
                    hook.remove()

                for actual, expected in (
                    (direct, reference_routed),
                    (public, reference_routed),
                    (shared, reference_shared),
                    (full, reference_full),
                ):
                    self.assertEqual(actual.shape, (T, H))
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(
                        actual.cpu().float(),
                        expected,
                        rtol=RTOL,
                        atol=ATOL,
                    )
                torch.testing.assert_close(
                    block.last_router_logits.cpu().float(),
                    reference_logits,
                    rtol=RTOL,
                    atol=ATOL,
                )


if __name__ == "__main__":
    unittest.main()
