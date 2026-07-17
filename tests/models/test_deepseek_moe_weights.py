import ast
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from tests.reference.deepseek_moe_reference import (
    DeepSeekRoutingSpec,
    deepseek_moe_reference,
)


ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = ROOT / "lite_llama/utils/deepseek_moe_weights.py"
TP_UTILS_PATH = ROOT / "lite_llama/executor/tp_utils.py"
MOE_PATH = ROOT / "lite_llama/models/moe.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_state(
    *,
    layers=(3,),
    num_experts=2,
    hidden_size=3,
    intermediate_size=2,
    shared_intermediate_size=4,
    use_bias=False,
    dtype=torch.float32,
):
    state = {
        "model.layers.4.mlp.gate_proj.weight": torch.full(
            (2, hidden_size),
            -44.0,
            dtype=dtype,
        ),
        "unrelated.weight": torch.tensor([99.0], dtype=dtype),
    }
    for layer_id in layers:
        prefix = f"model.layers.{layer_id}.mlp"
        state[f"{prefix}.gate.weight"] = (
            torch.arange(
                num_experts * hidden_size,
                dtype=dtype,
            ).reshape(num_experts, hidden_size)
            + layer_id * 1000
        )
        if use_bias:
            state[f"{prefix}.gate.e_score_correction_bias"] = (
                torch.arange(num_experts, dtype=torch.float16)
                + layer_id / 100
            )
        for expert_id in range(num_experts):
            expert_prefix = f"{prefix}.experts.{expert_id}"
            base = layer_id * 1000 + expert_id * 100
            state[f"{expert_prefix}.gate_proj.weight"] = (
                torch.arange(
                    intermediate_size * hidden_size,
                    dtype=dtype,
                ).reshape(intermediate_size, hidden_size)
                + base + 10
            )
            state[f"{expert_prefix}.up_proj.weight"] = (
                torch.arange(
                    intermediate_size * hidden_size,
                    dtype=dtype,
                ).reshape(intermediate_size, hidden_size)
                + base + 20
            )
            state[f"{expert_prefix}.down_proj.weight"] = (
                torch.arange(
                    hidden_size * intermediate_size,
                    dtype=dtype,
                ).reshape(hidden_size, intermediate_size)
                + base + 30
            )
        shared_prefix = f"{prefix}.shared_experts"
        state[f"{shared_prefix}.gate_proj.weight"] = (
            torch.arange(
                shared_intermediate_size * hidden_size,
                dtype=dtype,
            ).reshape(shared_intermediate_size, hidden_size)
            + layer_id * 1000
            + 40
        )
        state[f"{shared_prefix}.up_proj.weight"] = (
            torch.arange(
                shared_intermediate_size * hidden_size,
                dtype=dtype,
            ).reshape(shared_intermediate_size, hidden_size)
            + layer_id * 1000
            + 50
        )
        state[f"{shared_prefix}.down_proj.weight"] = (
            torch.arange(
                hidden_size * shared_intermediate_size,
                dtype=dtype,
            ).reshape(hidden_size, shared_intermediate_size)
            + layer_id * 1000
            + 60
        )
    return state


def _consumed_keys(layer_id, num_experts, use_bias):
    prefix = f"model.layers.{layer_id}.mlp"
    keys = [f"{prefix}.gate.weight"]
    if use_bias:
        keys.append(f"{prefix}.gate.e_score_correction_bias")
    for expert_id in range(num_experts):
        expert_prefix = f"{prefix}.experts.{expert_id}"
        keys.extend(
            (
                f"{expert_prefix}.gate_proj.weight",
                f"{expert_prefix}.up_proj.weight",
                f"{expert_prefix}.down_proj.weight",
            )
        )
    shared_prefix = f"{prefix}.shared_experts"
    keys.extend(
        (
            f"{shared_prefix}.gate_proj.weight",
            f"{shared_prefix}.up_proj.weight",
            f"{shared_prefix}.down_proj.weight",
        )
    )
    return keys


def _expected_canonical(state, layer_id, num_experts, use_bias):
    source = f"model.layers.{layer_id}.mlp"
    target = f"layers.{layer_id}.mlp"
    result = {f"{target}.gate.weight": state[f"{source}.gate.weight"]}
    if use_bias:
        result[f"{target}.gate.e_score_correction_bias"] = state[
            f"{source}.gate.e_score_correction_bias"
        ].float()
    gate_up = []
    down = []
    for expert_id in range(num_experts):
        expert = f"{source}.experts.{expert_id}"
        gate_up.append(
            torch.cat(
                (
                    state[f"{expert}.gate_proj.weight"],
                    state[f"{expert}.up_proj.weight"],
                ),
                dim=0,
            )
        )
        down.append(state[f"{expert}.down_proj.weight"])
    result[f"{target}.experts.gate_up_weight"] = torch.stack(
        gate_up,
        dim=0,
    )
    result[f"{target}.experts.down_weight"] = torch.stack(down, dim=0)
    shared = f"{source}.shared_experts"
    result[f"{target}.shared_experts.gate_up_weight"] = torch.cat(
        (
            state[f"{shared}.gate_proj.weight"],
            state[f"{shared}.up_proj.weight"],
        ),
        dim=0,
    )
    result[f"{target}.shared_experts.down_weight"] = state[
        f"{shared}.down_proj.weight"
    ]
    return result


def _assert_state_equal(test_case, actual, expected):
    test_case.assertEqual(list(actual), list(expected))
    for key in expected:
        test_case.assertTrue(
            torch.equal(actual[key], expected[key]),
            key,
        )


class DeepSeekMoeWeightContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = _load_module(
            "deepseek_moe_weight_utils",
            WEIGHTS_PATH,
        )
        cls.tp_utils = _load_module(
            "deepseek_moe_weight_tp_utils",
            TP_UTILS_PATH,
        )
        cls.production = _load_module(
            "deepseek_moe_weight_block_production",
            MOE_PATH,
        )

    def test_v2_single_layer_keys_layout_values_and_numeric_order(self):
        state = _make_state(layers=(3,), num_experts=3)
        converted = self.weights.stack_deepseek_moe_weights(
            state,
            moe_layer_indices=(3,),
            num_experts=3,
            use_correction_bias=False,
        )
        expected = _expected_canonical(state, 3, 3, False)
        _assert_state_equal(self, converted, expected)
        self.assertEqual(
            converted["layers.3.mlp.experts.gate_up_weight"].shape,
            (3, 4, 3),
        )
        self.assertEqual(
            converted["layers.3.mlp.experts.down_weight"].shape,
            (3, 3, 2),
        )
        self.assertEqual(
            converted["layers.3.mlp.shared_experts.gate_up_weight"].shape,
            (8, 3),
        )

    def test_v3_bias_is_required_and_canonicalized_to_fp32(self):
        state = _make_state(layers=(3,), use_bias=True)
        source_bias = state[
            "model.layers.3.mlp.gate.e_score_correction_bias"
        ]
        converted = self.weights.stack_deepseek_moe_weights(
            state,
            moe_layer_indices=(3,),
            num_experts=2,
            use_correction_bias=True,
        )
        bias = converted["layers.3.mlp.gate.e_score_correction_bias"]
        self.assertEqual(bias.dtype, torch.float32)
        torch.testing.assert_close(bias, source_bias.float())

        missing = _make_state(layers=(3,), use_bias=False)
        with self.assertRaisesRegex(KeyError, "layer 3.*correction_bias"):
            self.weights.stack_deepseek_moe_weights(
                missing,
                moe_layer_indices=(3,),
                num_experts=2,
                use_correction_bias=True,
            )

        v2_state = _make_state(layers=(3,), use_bias=True)
        unused_bias = v2_state[
            "model.layers.3.mlp.gate.e_score_correction_bias"
        ]
        v2_converted = self.weights.stack_deepseek_moe_weights(
            v2_state,
            moe_layer_indices=(3,),
            num_experts=2,
            use_correction_bias=False,
            consume=True,
        )
        self.assertNotIn(
            "layers.3.mlp.gate.e_score_correction_bias",
            v2_converted,
        )
        self.assertIs(
            v2_state["model.layers.3.mlp.gate.e_score_correction_bias"],
            unused_bias,
        )

    def test_noncontiguous_layer_order_and_dense_weights_are_untouched(self):
        state = _make_state(layers=(3, 5))
        dense = state["model.layers.4.mlp.gate_proj.weight"]
        converted = self.weights.stack_deepseek_moe_weights(
            state,
            moe_layer_indices=(5, 3),
            num_experts=2,
            use_correction_bias=False,
        )
        self.assertEqual(
            list(converted),
            list(_expected_canonical(state, 5, 2, False))
            + list(_expected_canonical(state, 3, 2, False)),
        )
        self.assertIs(state["model.layers.4.mlp.gate_proj.weight"], dense)
        self.assertIn("unrelated.weight", state)

    def test_consume_false_preserves_source_keys_and_tensor_identity(self):
        state = _make_state(layers=(3,))
        keys_before = list(state)
        identities = {key: id(value) for key, value in state.items()}
        converted = self.weights.stack_deepseek_moe_weights(
            state,
            moe_layer_indices=(3,),
            num_experts=2,
            use_correction_bias=False,
            consume=False,
        )
        self.assertEqual(list(state), keys_before)
        self.assertEqual(
            {key: id(value) for key, value in state.items()},
            identities,
        )
        self.assertIs(
            converted["layers.3.mlp.gate.weight"],
            state["model.layers.3.mlp.gate.weight"],
        )
        self.assertIs(
            converted["layers.3.mlp.shared_experts.down_weight"],
            state["model.layers.3.mlp.shared_experts.down_proj.weight"],
        )

    def test_consume_true_deletes_only_consumed_keys_after_success(self):
        state = _make_state(layers=(3, 5))
        unrelated = state["unrelated.weight"]
        dense = state["model.layers.4.mlp.gate_proj.weight"]
        self.weights.stack_deepseek_moe_weights(
            state,
            moe_layer_indices=(3,),
            num_experts=2,
            use_correction_bias=False,
            consume=True,
        )
        for key in _consumed_keys(3, 2, False):
            self.assertNotIn(key, state)
        for key in _consumed_keys(5, 2, False):
            self.assertIn(key, state)
        self.assertIs(state["unrelated.weight"], unrelated)
        self.assertIs(state["model.layers.4.mlp.gate_proj.weight"], dense)

    def test_consume_failure_is_atomic(self):
        state = _make_state(layers=(3, 5))
        del state["model.layers.5.mlp.shared_experts.down_proj.weight"]
        keys_before = list(state)
        identities = {key: id(value) for key, value in state.items()}
        with self.assertRaisesRegex(KeyError, "layer 5.*shared_experts.*down"):
            self.weights.stack_deepseek_moe_weights(
                state,
                moe_layer_indices=(3, 5),
                num_experts=2,
                use_correction_bias=False,
                consume=True,
            )
        self.assertEqual(list(state), keys_before)
        self.assertEqual(
            {key: id(value) for key, value in state.items()},
            identities,
        )

    def test_missing_router_expert_shared_and_bias_fail_clearly(self):
        cases = (
            ("model.layers.3.mlp.gate.weight", "layer 3.*gate.weight", False),
            (
                "model.layers.3.mlp.experts.1.up_proj.weight",
                "layer 3 expert 1.*up_proj",
                False,
            ),
            (
                "model.layers.3.mlp.shared_experts.gate_proj.weight",
                "layer 3.*shared_experts.*gate_proj",
                False,
            ),
            (
                "model.layers.3.mlp.gate.e_score_correction_bias",
                "layer 3.*correction_bias",
                True,
            ),
        )
        for key, message, use_bias in cases:
            with self.subTest(key=key):
                state = _make_state(layers=(3,), use_bias=use_bias)
                del state[key]
                with self.assertRaisesRegex(KeyError, message):
                    self.weights.stack_deepseek_moe_weights(
                        state,
                        moe_layer_indices=(3,),
                        num_experts=2,
                        use_correction_bias=use_bias,
                    )

    def test_shape_type_dtype_and_expert_count_errors_fail_closed(self):
        mutations = (
            (
                "router non-tensor",
                "model.layers.3.mlp.gate.weight",
                "not a tensor",
                TypeError,
                "torch.Tensor",
            ),
            (
                "router expert count",
                "model.layers.3.mlp.gate.weight",
                torch.zeros(3, 3),
                ValueError,
                "shape.*2",
            ),
            (
                "gate/up mismatch",
                "model.layers.3.mlp.experts.0.up_proj.weight",
                torch.zeros(3, 3),
                ValueError,
                "gate/up shapes differ",
            ),
            (
                "expert down mismatch",
                "model.layers.3.mlp.experts.0.down_proj.weight",
                torch.zeros(3, 3),
                ValueError,
                "down_proj.*shape",
            ),
            (
                "shared mismatch",
                "model.layers.3.mlp.shared_experts.down_proj.weight",
                torch.zeros(3, 3),
                ValueError,
                "shared_experts.*down_proj.*shape",
            ),
            (
                "nonfloating",
                "model.layers.3.mlp.experts.0.gate_proj.weight",
                torch.zeros(2, 3, dtype=torch.int64),
                TypeError,
                "floating point",
            ),
            (
                "dtype mismatch",
                "model.layers.3.mlp.experts.0.gate_proj.weight",
                torch.zeros(2, 3, dtype=torch.float64),
                ValueError,
                "dtype/device mismatch",
            ),
        )
        for name, key, value, error, message in mutations:
            with self.subTest(name=name):
                state = _make_state(layers=(3,))
                state[key] = value
                with self.assertRaisesRegex(error, message):
                    self.weights.stack_deepseek_moe_weights(
                        state,
                        moe_layer_indices=(3,),
                        num_experts=2,
                        use_correction_bias=False,
                    )

        state = _make_state(layers=(3,))
        state["model.layers.3.mlp.gate.weight"] = torch.zeros(2, 3, 1)
        with self.assertRaisesRegex(ValueError, "gate.weight.*rank 2"):
            self.weights.stack_deepseek_moe_weights(
                state,
                moe_layer_indices=(3,),
                num_experts=2,
                use_correction_bias=False,
            )

        state = _make_state(layers=(3,))
        for projection in ("gate_proj", "up_proj"):
            state[
                f"model.layers.3.mlp.experts.0.{projection}.weight"
            ] = torch.zeros(2, 4)
        with self.assertRaisesRegex(ValueError, "expert 0.*shape.*3"):
            self.weights.stack_deepseek_moe_weights(
                state,
                moe_layer_indices=(3,),
                num_experts=2,
                use_correction_bias=False,
            )

        state = _make_state(layers=(3,))
        for projection in ("gate_proj", "up_proj"):
            state[
                f"model.layers.3.mlp.experts.1.{projection}.weight"
            ] = torch.zeros(3, 3)
        state[
            "model.layers.3.mlp.experts.1.down_proj.weight"
        ] = torch.zeros(3, 3)
        with self.assertRaisesRegex(ValueError, "expert 1.*intermediate size"):
            self.weights.stack_deepseek_moe_weights(
                state,
                moe_layer_indices=(3,),
                num_experts=2,
                use_correction_bias=False,
            )

    def test_layer_index_and_api_configuration_fail_closed(self):
        state = _make_state(layers=(3,))
        for indices, message in (
            ((), "must not be empty"),
            ((3, 3), "duplicate layer 3"),
            ((-1,), "non-negative"),
            ((True,), "non-negative"),
        ):
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(ValueError, message):
                    self.weights.stack_deepseek_moe_weights(
                        state,
                        moe_layer_indices=indices,
                        num_experts=2,
                        use_correction_bias=False,
                    )
        with self.assertRaisesRegex(TypeError, "use_correction_bias"):
            self.weights.stack_deepseek_moe_weights(
                state,
                moe_layer_indices=(3,),
                num_experts=2,
                use_correction_bias="sqrtsoftplus",
            )

    def test_world_one_shared_layout_matches_d1_runtime_parameters(self):
        gate_up = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        down = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        tp = self.tp_utils.TPConfig()
        runtime_gate_up = self.tp_utils.prepare_shared_expert_gate_up(
            gate_up,
            4,
            tp,
        )
        runtime_down = self.tp_utils.prepare_shared_expert_down(down, tp)
        self.assertTrue(torch.equal(runtime_gate_up, gate_up.T.contiguous()))
        self.assertTrue(torch.equal(runtime_down, down.T.contiguous()))
        self.assertEqual(runtime_gate_up.shape, (3, 8))
        self.assertEqual(runtime_down.shape, (4, 3))

    def test_tp2_shared_slices_and_reconstructs_canonical_layout(self):
        gate_up = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        down = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        runtime_gate_up = []
        runtime_down = []
        for rank in range(2):
            tp = self.tp_utils.TPConfig(
                world_size=2,
                rank=rank,
                moe_parallel_mode="tp",
            )
            gate_local = self.tp_utils.prepare_shared_expert_gate_up(
                gate_up,
                4,
                tp,
            )
            down_local = self.tp_utils.prepare_shared_expert_down(down, tp)
            start = rank * 2
            expected_gate = torch.cat(
                (gate_up[start : start + 2], gate_up[4 + start : 6 + start]),
                dim=0,
            ).T.contiguous()
            self.assertTrue(torch.equal(gate_local, expected_gate))
            self.assertTrue(
                torch.equal(
                    down_local,
                    down[:, start : start + 2].T.contiguous(),
                )
            )
            runtime_gate_up.append(gate_local)
            runtime_down.append(down_local)

        gate_parts = [weight.T.chunk(2, dim=0) for weight in runtime_gate_up]
        reconstructed_gate_up = torch.cat(
            (
                torch.cat((gate_parts[0][0], gate_parts[1][0]), dim=0),
                torch.cat((gate_parts[0][1], gate_parts[1][1]), dim=0),
            ),
            dim=0,
        )
        reconstructed_down = torch.cat(
            (runtime_down[0].T, runtime_down[1].T),
            dim=1,
        )
        self.assertTrue(torch.equal(reconstructed_gate_up, gate_up))
        self.assertTrue(torch.equal(reconstructed_down, down))

    def test_ep2_shared_layout_is_fully_replicated_per_rank(self):
        gate_up = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        down = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        world_one = self.tp_utils.TPConfig()
        full_gate = self.tp_utils.prepare_shared_expert_gate_up(
            gate_up,
            4,
            world_one,
        )
        full_down = self.tp_utils.prepare_shared_expert_down(down, world_one)
        for rank in range(2):
            tp = self.tp_utils.TPConfig(
                world_size=2,
                rank=rank,
                moe_parallel_mode="ep",
            )
            self.assertTrue(
                torch.equal(
                    self.tp_utils.prepare_shared_expert_gate_up(
                        gate_up,
                        4,
                        tp,
                    ),
                    full_gate,
                )
            )
            self.assertTrue(
                torch.equal(
                    self.tp_utils.prepare_shared_expert_down(down, tp),
                    full_down,
                )
            )

    def test_shared_layout_invalid_configuration_and_shapes_fail(self):
        gate_up = torch.zeros(8, 3)
        down = torch.zeros(3, 4)
        cases = (
            (
                lambda: self.tp_utils.prepare_shared_expert_gate_up(
                    gate_up,
                    4,
                    self.tp_utils.TPConfig(
                        world_size=2,
                        rank=0,
                        moe_parallel_mode="unknown",
                    ),
                ),
                "moe_parallel_mode",
            ),
            (
                lambda: self.tp_utils.prepare_shared_expert_gate_up(
                    gate_up,
                    4,
                    self.tp_utils.TPConfig(world_size=0, rank=0),
                ),
                "world_size",
            ),
            (
                lambda: self.tp_utils.prepare_shared_expert_gate_up(
                    gate_up,
                    4,
                    self.tp_utils.TPConfig(world_size=2, rank=2),
                ),
                "rank",
            ),
            (
                lambda: self.tp_utils.prepare_shared_expert_gate_up(
                    torch.zeros(6, 3),
                    3,
                    self.tp_utils.TPConfig(world_size=2, rank=0),
                ),
                "divisible",
            ),
            (
                lambda: self.tp_utils.prepare_shared_expert_gate_up(
                    torch.zeros(7, 3),
                    4,
                    self.tp_utils.TPConfig(),
                ),
                "shape",
            ),
            (
                lambda: self.tp_utils.prepare_shared_expert_down(
                    torch.zeros(3, 4, 1),
                    self.tp_utils.TPConfig(),
                ),
                "shape",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    call()

    def test_v2_v3_canonical_to_runtime_strict_load_and_reference(self):
        for use_bias, seed in ((False, 801), (True, 802)):
            with self.subTest(use_bias=use_bias):
                state = _make_state(
                    layers=(3,),
                    num_experts=4,
                    hidden_size=3,
                    intermediate_size=2,
                    shared_intermediate_size=3,
                    use_bias=use_bias,
                )
                for key in _consumed_keys(3, 4, use_bias):
                    if not key.endswith("e_score_correction_bias"):
                        state[key] = state[key] * 1e-3
                canonical = self.weights.stack_deepseek_moe_weights(
                    state,
                    moe_layer_indices=(3,),
                    num_experts=4,
                    use_correction_bias=use_bias,
                )
                expected = _expected_canonical(state, 3, 4, use_bias)
                _assert_state_equal(self, canonical, expected)
                prefix = "layers.3.mlp."
                runtime = {
                    "gate.weight": canonical[prefix + "gate.weight"],
                    "experts.gate_up_weight": (
                        self.tp_utils.prepare_moe_gate_up_for_gmm(
                            canonical[prefix + "experts.gate_up_weight"],
                            2,
                            self.tp_utils.TPConfig(),
                        )
                    ),
                    "experts.down_weight": (
                        self.tp_utils.prepare_moe_down_for_gmm(
                            canonical[prefix + "experts.down_weight"],
                            self.tp_utils.TPConfig(),
                        )
                    ),
                    "shared_experts.gate_up_weight": (
                        self.tp_utils.prepare_shared_expert_gate_up(
                            canonical[
                                prefix + "shared_experts.gate_up_weight"
                            ],
                            3,
                            self.tp_utils.TPConfig(),
                        )
                    ),
                    "shared_experts.down_weight": (
                        self.tp_utils.prepare_shared_expert_down(
                            canonical[prefix + "shared_experts.down_weight"],
                            self.tp_utils.TPConfig(),
                        )
                    ),
                }
                if use_bias:
                    runtime["gate.e_score_correction_bias"] = canonical[
                        prefix + "gate.e_score_correction_bias"
                    ]
                spec = DeepSeekRoutingSpec(
                    num_experts=4,
                    experts_per_token=2,
                    num_groups=2,
                    topk_groups=1,
                    score_func="sigmoid" if use_bias else "softmax",
                    topk_method=(
                        "noaux_tc" if use_bias else "group_limited_greedy"
                    ),
                    norm_topk_prob=True,
                    routed_scaling_factor=1.0,
                )
                block = self.production.DeepSeekMoeBlock(
                    hidden_size=3,
                    num_experts=4,
                    top_k=2,
                    intermediate_size=2,
                    shared_intermediate_size=3,
                    num_groups=2,
                    topk_groups=1,
                    score_func=spec.score_func,
                    topk_method=spec.topk_method,
                    dtype=torch.float32,
                )
                load_result = block.load_state_dict(runtime, strict=True)
                self.assertEqual(load_result.missing_keys, [])
                self.assertEqual(load_result.unexpected_keys, [])
                if use_bias:
                    self.assertEqual(
                        block.gate.e_score_correction_bias.dtype,
                        torch.float32,
                    )
                generator = torch.Generator().manual_seed(seed)
                hidden = torch.randn(4, 3, generator=generator)
                expected_output, _, _, _ = deepseek_moe_reference(
                    hidden,
                    expected[prefix + "gate.weight"],
                    expected[prefix + "experts.gate_up_weight"].transpose(
                        1,
                        2,
                    ).contiguous(),
                    expected[prefix + "experts.down_weight"].transpose(
                        1,
                        2,
                    ).contiguous(),
                    spec,
                    correction_bias=(
                        expected.get(prefix + "gate.e_score_correction_bias")
                    ),
                    shared_gate_up_weight=expected[
                        prefix + "shared_experts.gate_up_weight"
                    ].T.contiguous(),
                    shared_down_weight=expected[
                        prefix + "shared_experts.down_weight"
                    ].T.contiguous(),
                )
                torch.testing.assert_close(
                    block(hidden),
                    expected_output,
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_existing_tp_functions_and_frozen_moe_files_are_unchanged(self):
        tree = ast.parse(TP_UTILS_PATH.read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        expected_hashes = {
            "get_tp_config": "7ef50a4b8d50f1655728493ee918bf477840f0bd05b26eb6c1fb41fdad1df8e3",
            "init_tp": "4c1422c80d2fc125b573fb3997b459d9f80d9a3838900cdbe4f0a935052a8b2f",
            "get_tp_group": "5f4f5820f5c4e9c9beeacd0777f5de2cf19f6fcf26a16966d5904b67520541d2",
            "tp_is_initialized": "2efbee9df4dc847766d11bde785596dfd2cf2c5eb644cd334729e96046914143",
            "tp_all_reduce": "32162e9b2c9c637c28b2f3bb4d04f0221c3c07bcd08ba2d33b18164b9448a4b2",
            "tp_all_gather": "708e5442d89386785dae1b4d570aaede498e0caf0b1e89491c8b1dfda79dbb56",
            "_shard_slice": "c8a70e56e07eb5969c36a1cee73ce3a282be127e519d6e52439c8d710d5ca510",
            "shard_attention_q": "c5c8878baa708082aa2605541481c3c12ca05a92165582ae98b5e33e3bd131a5",
            "shard_attention_kv": "d341013f4cc13133a08803b510d8c297db75f337d48fece3264b663329542275",
            "shard_attention_o": "511e52fe2f7c3698eb424c83b95f45c2231ebf616e195e923b01144260ead47c",
            "shard_ffn_gate_up": "a4c3a6a840373f0bc5244b9f9ba8b12b4f433e32f1efeaa3bbcca48852f04151",
            "shard_ffn_down": "ee31336e66e4dc85d51c3cacf0e1e9bff010a1552ba7a525568d27d662cb9020",
            "shard_moe_gate_up": "d869be9e7e8e8fb8da579217fb906e463cb114ed5f364f5d475126c5bbd38222",
            "shard_moe_down": "ea56cb2d33394ed383046c54086e0c7c67b4aef8a80bc340f2bd7adb87dc5e11",
            "prepare_moe_gate_up_for_gmm": "640b790779b22cb8cb411fbfbfbb99c72600524405d90bb03baf33bde69cae29",
            "prepare_moe_down_for_gmm": "2c4640081361b0cb68ddfdc489d2685c921a0f11be23c8184a9ef4c313da14a6",
            "shard_moe_experts": "b77a4e3fb4618f4c765081e4ee77ffdd0a53a41a793486d85f2d7c5204b8d802",
            "prepare_moe_gate_up_for_ep": "810f8d1acbf8b12c89b7496e58f367f5acab100dc4742febd569f8bf326894be",
            "prepare_moe_down_for_ep": "7b70ab6d924c71325990d00a85ff7702882c7200c40f3cdccd3c49e7fcae8cab",
            "shard_lm_head": "46dbf78f6f6b2199ba0acdc71c1da87fafeb4a1e6be6282e22e5ff4a060e5c41",
            "detect_tp_env": "31c15975b076df9bad9060acd1d5102f7dccd4d298f4344a2f1682a5b2c070a7",
        }
        for name, expected in expected_hashes.items():
            with self.subTest(name=name):
                actual = hashlib.sha256(
                    ast.dump(
                        functions[name],
                        include_attributes=False,
                    ).encode()
                ).hexdigest()
                self.assertEqual(actual, expected)

        frozen = {
            "lite_llama/models/moe.py": "ae44153220a76eb5234fbcafc6fc49504377b27b431b68db121cef5b6d09b8ca",
            "tests/reference/deepseek_moe_reference.py": "659e73c636d52e002321a91ce5956e3f125ed2b125bffd7cb801513ad9e33691",
            "tests/models/test_deepseek_moe_reference.py": "c027762442e0b44364225d512ad625e11946f25dd822a635a3f13f9460ca0fff",
            "tests/models/test_deepseek_moe_router.py": "588240fbaf9611bf0f7ec002bd949446b35453a7cbcd77af17badc08405953b4",
            "tests/models/test_deepseek_moe_block.py": "1a1ed0be143506866c633c9f3f0924edc56a01c4430fb2fcb3fe0e6d75e622e6",
        }
        for path, expected in frozen.items():
            with self.subTest(path=path):
                actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
