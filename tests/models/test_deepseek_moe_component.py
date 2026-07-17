import ast
import importlib.util
import inspect
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from tests.reference.deepseek_moe_reference import (
    DeepSeekRoutingSpec,
    deepseek_moe_reference,
    deepseek_route_reference,
    deepseek_shared_expert_reference,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "lite_llama/models/model_config.py"
COMPONENT_PATH = ROOT / "lite_llama/models/deepseek_moe.py"
MOE_PATH = ROOT / "lite_llama/models/moe.py"
TP_UTILS_PATH = ROOT / "lite_llama/executor/tp_utils.py"
WEIGHTS_PATH = ROOT / "lite_llama/utils/deepseek_moe_weights.py"

V2_LITE_SOURCE = (
    "https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/commit/"
    "604d5664dddd88a0433dbae533b7fe9472482de0"
)
V3_SOURCE = (
    "https://github.com/deepseek-ai/DeepSeek-V3/commit/"
    "9b4e9788e4a3a731f7567338ed15d3ec549ce03b"
)

V2_LITE_CONFIG = {
    "architectures": ["DeepseekV2ForCausalLM"],
    "model_type": "deepseek_v2",
    "hidden_size": 2048,
    "num_hidden_layers": 27,
    "intermediate_size": 10944,
    "n_routed_experts": 64,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 1408,
    "n_shared_experts": 2,
    "scoring_func": "softmax",
    "topk_method": "greedy",
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": False,
    "routed_scaling_factor": 1.0,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "torch_dtype": "bfloat16",
}

V3_CONFIG = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "model_type": "deepseek_v3",
    "hidden_size": 7168,
    "num_hidden_layers": 61,
    "intermediate_size": 18432,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "n_shared_experts": 1,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_group": 8,
    "topk_group": 4,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "first_k_dense_replace": 3,
    "moe_layer_freq": 1,
    "torch_dtype": "bfloat16",
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_component_modules():
    config = _load_module("deepseek_moe_component_config", CONFIG_PATH)
    moe = _load_module("deepseek_moe_component_moe", MOE_PATH)
    tp_utils = _load_module("deepseek_moe_component_tp_utils", TP_UTILS_PATH)
    weights = _load_module("deepseek_moe_component_weights", WEIGHTS_PATH)
    component = _load_module("deepseek_moe_component_production", COMPONENT_PATH)
    return config, moe, tp_utils, weights, component


def _small_config_dict(version, **overrides):
    if version == "v2":
        data = {
            "architectures": ["DeepseekV2ForCausalLM"],
            "model_type": "deepseek_v2",
            "hidden_size": 4,
            "num_hidden_layers": 6,
            "intermediate_size": 12,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 4,
            "n_shared_experts": 1,
            "scoring_func": "softmax",
            "topk_method": "greedy",
            "n_group": 2,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.0,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "torch_dtype": "bfloat16",
        }
    else:
        data = {
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 4,
            "num_hidden_layers": 6,
            "intermediate_size": 12,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 4,
            "n_shared_experts": 1,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "n_group": 2,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.25,
            "first_k_dense_replace": 3,
            "moe_layer_freq": 1,
            "torch_dtype": "bfloat16",
        }
    data.update(overrides)
    return data


def _make_hf_state(
    *,
    layers,
    num_experts=4,
    hidden_size=4,
    intermediate_size=4,
    shared_intermediate_size=4,
    use_bias=False,
    seed=700,
):
    generator = torch.Generator().manual_seed(seed)
    state = {
        "unrelated.weight": torch.randn(2, generator=generator),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(
            12, hidden_size, generator=generator
        ),
    }
    for layer_index in layers:
        prefix = f"model.layers.{layer_index}.mlp"
        state[f"{prefix}.gate.weight"] = torch.randn(
            num_experts, hidden_size, generator=generator
        ) * 0.2
        if use_bias:
            state[f"{prefix}.gate.e_score_correction_bias"] = torch.randn(
                num_experts, generator=generator
            ) * 0.02
        for expert_id in range(num_experts):
            expert = f"{prefix}.experts.{expert_id}"
            state[f"{expert}.gate_proj.weight"] = torch.randn(
                intermediate_size, hidden_size, generator=generator
            ) * 0.2
            state[f"{expert}.up_proj.weight"] = torch.randn(
                intermediate_size, hidden_size, generator=generator
            ) * 0.2
            state[f"{expert}.down_proj.weight"] = torch.randn(
                hidden_size, intermediate_size, generator=generator
            ) * 0.2
        shared = f"{prefix}.shared_experts"
        state[f"{shared}.gate_proj.weight"] = torch.randn(
            shared_intermediate_size, hidden_size, generator=generator
        ) * 0.2
        state[f"{shared}.up_proj.weight"] = torch.randn(
            shared_intermediate_size, hidden_size, generator=generator
        ) * 0.2
        state[f"{shared}.down_proj.weight"] = torch.randn(
            hidden_size, shared_intermediate_size, generator=generator
        ) * 0.2
    return state


def _routing_spec(config):
    return DeepSeekRoutingSpec(
        num_experts=config.num_experts,
        experts_per_token=config.num_experts_per_tok,
        num_groups=config.num_groups,
        topk_groups=config.topk_groups,
        score_func=config.score_func,
        topk_method=config.topk_method,
        norm_topk_prob=config.norm_topk_prob,
        routed_scaling_factor=config.routed_scaling_factor,
    )


def _reference_from_canonical(hidden, canonical, config, layer_index):
    prefix = f"layers.{layer_index}.mlp."
    return deepseek_moe_reference(
        hidden,
        canonical[prefix + "gate.weight"],
        canonical[prefix + "experts.gate_up_weight"].transpose(1, 2).contiguous(),
        canonical[prefix + "experts.down_weight"].transpose(1, 2).contiguous(),
        _routing_spec(config),
        correction_bias=canonical.get(
            prefix + "gate.e_score_correction_bias"
        ),
        shared_gate_up_weight=canonical[
            prefix + "shared_experts.gate_up_weight"
        ].transpose(0, 1).contiguous(),
        shared_down_weight=canonical[
            prefix + "shared_experts.down_weight"
        ].transpose(0, 1).contiguous(),
    )


def _ast_class_dump(source, class_name):
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    return ast.dump(node, include_attributes=False)


class DeepSeekMoeComponentContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.config_module,
            cls.moe,
            cls.tp_utils,
            cls.weights,
            cls.component,
        ) = _load_component_modules()

    def test_v2_lite_frozen_official_fields_aliases_and_schedule(self):
        config = self.config_module.DeepSeekMoeConfig.from_dict(
            {**V2_LITE_CONFIG, "unknown_attention_field": 99}
        )
        self.assertEqual(V2_LITE_SOURCE.rsplit("/", 1)[-1], "604d5664dddd88a0433dbae533b7fe9472482de0")
        self.assertEqual(config.model_type, "deepseek_v2")
        self.assertEqual(config.architectures, ["DeepseekV2ForCausalLM"])
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.num_layers, 27)
        self.assertEqual(config.intermediate_size, 10944)
        self.assertEqual(config.num_experts, 64)
        self.assertEqual(config.num_experts_per_tok, 6)
        self.assertEqual(config.moe_intermediate_size, 1408)
        self.assertEqual(config.num_shared_experts, 2)
        self.assertEqual(config.shared_intermediate_size, 2816)
        self.assertEqual(config.score_func, "softmax")
        self.assertEqual(config.topk_method, "group_limited_greedy")
        self.assertFalse(config.norm_topk_prob)
        self.assertFalse(config.uses_correction_bias)
        self.assertEqual(config.moe_layer_indices(), tuple(range(1, 27)))
        self.assertFalse(hasattr(config, "unknown_attention_field"))

    def test_v3_frozen_official_fields_and_first_three_dense_layers(self):
        config = self.config_module.DeepSeekMoeConfig.from_dict(V3_CONFIG)
        self.assertEqual(V3_SOURCE.rsplit("/", 1)[-1], "9b4e9788e4a3a731f7567338ed15d3ec549ce03b")
        self.assertEqual(config.model_type, "deepseek_v3")
        self.assertEqual(config.hidden_size, 7168)
        self.assertEqual(config.num_layers, 61)
        self.assertEqual(config.intermediate_size, 18432)
        self.assertEqual(config.num_experts, 256)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.moe_intermediate_size, 2048)
        self.assertEqual(config.shared_intermediate_size, 2048)
        self.assertEqual((config.num_groups, config.topk_groups), (8, 4))
        self.assertEqual(config.routed_scaling_factor, 2.5)
        self.assertTrue(config.uses_correction_bias)
        self.assertEqual(config.moe_layer_indices(), tuple(range(3, 61)))
        self.assertEqual(
            [config.is_moe_layer(index) for index in range(4)],
            [False, False, False, True],
        )

    def test_model_type_and_architectures_must_match_frozen_source(self):
        from_dict_cases = (
            (
                {
                    **V2_LITE_CONFIG,
                    "architectures": ["DeepseekV3ForCausalLM"],
                },
                "deepseek_v2",
            ),
            (
                {
                    **V3_CONFIG,
                    "architectures": ["DeepseekV2ForCausalLM"],
                },
                "deepseek_v3",
            ),
            ({key: value for key, value in V2_LITE_CONFIG.items() if key != "architectures"}, "architectures"),
            ({key: value for key, value in V2_LITE_CONFIG.items() if key != "model_type"}, "model_type"),
            ({**V2_LITE_CONFIG, "architectures": []}, "architectures"),
            (
                {
                    **V2_LITE_CONFIG,
                    "architectures": [
                        "DeepseekV2ForCausalLM",
                        "DeepseekV3ForCausalLM",
                    ],
                },
                "architectures",
            ),
            (
                {
                    **V3_CONFIG,
                    "architectures": [
                        "DeepseekV3ForCausalLM",
                        "UnknownForCausalLM",
                    ],
                },
                "architectures",
            ),
        )
        for data, message in from_dict_cases:
            with self.subTest(path="from_dict", message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.config_module.DeepSeekMoeConfig.from_dict(data)

        direct_cases = (
            dict(
                model_type="deepseek_v2",
                architectures=["DeepseekV3ForCausalLM"],
            ),
            dict(
                model_type="deepseek_v3",
                architectures=["DeepseekV2ForCausalLM"],
                score_func="sigmoid",
                topk_method="noaux_tc",
            ),
        )
        for kwargs in direct_cases:
            with self.subTest(path="direct", model_type=kwargs["model_type"]):
                with self.assertRaisesRegex(ValueError, "architectures"):
                    self.config_module.DeepSeekMoeConfig(**kwargs)

        default = self.config_module.DeepSeekMoeConfig()
        self.assertEqual(default.model_type, "deepseek_v2")
        self.assertEqual(default.architectures, ["DeepseekV2ForCausalLM"])

    def test_from_dict_requires_every_audited_source_field(self):
        required_fields = (
            "architectures",
            "model_type",
            "hidden_size",
            "intermediate_size",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "topk_method",
            "norm_topk_prob",
            "routed_scaling_factor",
            "first_k_dense_replace",
            "moe_layer_freq",
            "torch_dtype",
        )
        for field_name in required_fields:
            with self.subTest(kind="field", field=field_name):
                data = dict(V3_CONFIG)
                del data[field_name]
                with self.assertRaisesRegex(ValueError, field_name):
                    self.config_module.DeepSeekMoeConfig.from_dict(data)

        alias_groups = (
            ("num_hidden_layers", "num_layers"),
            ("n_routed_experts", "num_experts"),
            ("n_shared_experts", "num_shared_experts"),
            ("scoring_func", "score_func"),
            ("n_group", "num_groups"),
            ("topk_group", "topk_groups"),
        )
        for alias, canonical in alias_groups:
            with self.subTest(kind="alias_group", alias=alias, canonical=canonical):
                data = dict(V3_CONFIG)
                del data[alias]
                with self.assertRaisesRegex(ValueError, f"{alias}.*{canonical}"):
                    self.config_module.DeepSeekMoeConfig.from_dict(data)

        misspelled = dict(V3_CONFIG)
        misspelled["routed_scaling_facter"] = misspelled.pop(
            "routed_scaling_factor"
        )
        with self.assertRaisesRegex(ValueError, "routed_scaling_factor"):
            self.config_module.DeepSeekMoeConfig.from_dict(misspelled)

    def test_from_dict_rejects_conflicting_alias_and_canonical_values(self):
        alias_groups = (
            ("num_hidden_layers", "num_layers", 61, 60),
            ("n_routed_experts", "num_experts", 256, 128),
            ("n_shared_experts", "num_shared_experts", 1, 2),
            ("scoring_func", "score_func", "sigmoid", "softmax"),
            ("n_group", "num_groups", 8, 4),
            ("topk_group", "topk_groups", 4, 3),
        )
        for alias, canonical, same_value, conflicting_value in alias_groups:
            with self.subTest(kind="same", alias=alias, canonical=canonical):
                same = {**V3_CONFIG, canonical: same_value}
                config = self.config_module.DeepSeekMoeConfig.from_dict(same)
                self.assertEqual(getattr(config, canonical), same_value)
            with self.subTest(kind="conflict", alias=alias, canonical=canonical):
                conflict = {**V3_CONFIG, canonical: conflicting_value}
                with self.assertRaisesRegex(
                    ValueError,
                    f"{alias}.*{canonical}|{canonical}.*{alias}",
                ):
                    self.config_module.DeepSeekMoeConfig.from_dict(conflict)

    def test_zero_based_frequency_schedule_and_out_of_range(self):
        config = self.config_module.DeepSeekMoeConfig.from_dict(
            _small_config_dict("v2", moe_layer_freq=2)
        )
        self.assertEqual(config.moe_layer_indices(), (2, 4))
        for index in (-1, 0, 1, 3, 5, 6, True):
            with self.subTest(index=index):
                self.assertFalse(config.is_moe_layer(index))

    def test_config_fails_closed_for_unaudited_and_invalid_values(self):
        cases = (
            ({**_small_config_dict("v2"), "model_type": "deepseek_v4"}, "V4"),
            ({**_small_config_dict("v2"), "static_hash": [1, 2]}, "hash"),
            ({**_small_config_dict("v2"), "scoring_func": "sqrtsoftplus"}, "requires"),
            ({**_small_config_dict("v2"), "topk_method": "static_hash"}, "requires"),
            ({**_small_config_dict("v2"), "scoring_func": "sigmoid"}, "requires"),
            ({**_small_config_dict("v3"), "topk_method": "group_limited_greedy"}, "requires"),
            ({**_small_config_dict("v2"), "n_routed_experts": 3}, "divisible"),
            ({**_small_config_dict("v2"), "num_experts_per_tok": 5}, "exceed"),
            ({**_small_config_dict("v2"), "topk_group": 3}, "topk_groups"),
            ({**_small_config_dict("v2"), "n_shared_experts": 0}, "positive"),
            ({**_small_config_dict("v2"), "routed_scaling_factor": float("inf")}, "finite"),
            ({**_small_config_dict("v2"), "moe_layer_freq": 0}, "positive"),
            ({**_small_config_dict("v2"), "first_k_dense_replace": 6}, "num_layers"),
            ({**_small_config_dict("v2"), "n_routed_experts": True}, "positive"),
            ({**_small_config_dict("v2"), "norm_topk_prob": 1}, "bool"),
            ({**_small_config_dict("v3"), "n_routed_experts": 2, "n_group": 2, "num_experts_per_tok": 1}, "two experts"),
            ({**_small_config_dict("v2"), "num_hidden_layers": 2, "first_k_dense_replace": 1, "moe_layer_freq": 3}, "at least one"),
        )
        for data, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    self.config_module.DeepSeekMoeConfig.from_dict(data)

    def test_parallel_validation_distinguishes_tp_and_ep_ownership(self):
        config = self.config_module.DeepSeekMoeConfig.from_dict(
            _small_config_dict(
                "v2",
                moe_intermediate_size=6,
                n_shared_experts=2,
            )
        )
        config.validate_moe_parallel(3, "tp")
        config.validate_moe_parallel(4, "ep")
        with self.assertRaisesRegex(ValueError, "moe_intermediate_size"):
            config.validate_moe_parallel(4, "tp")
        with self.assertRaisesRegex(ValueError, "num_experts"):
            config.validate_moe_parallel(3, "ep")
        with self.assertRaisesRegex(ValueError, "mode"):
            config.validate_moe_parallel(2, "unknown")

    def test_config_is_not_registered_as_a_full_model(self):
        registry_source = (
            ROOT / "lite_llama/executor/executor_struct.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("DeepSeekMoeConfig", registry_source)
        self.assertNotIn("deepseek_v2", registry_source)
        self.assertNotIn("deepseek_v3", registry_source)

    def test_factory_builds_only_scheduled_v2_v3_moe_blocks(self):
        expected_v2 = {
            "gate.weight",
            "experts.gate_up_weight",
            "experts.down_weight",
            "shared_experts.gate_up_weight",
            "shared_experts.down_weight",
        }
        for version, layer_index in (("v2", 1), ("v3", 3)):
            with self.subTest(version=version):
                config = self.config_module.DeepSeekMoeConfig.from_dict(
                    _small_config_dict(version)
                )
                block = self.component.build_deepseek_moe_block(
                    config,
                    layer_index,
                    dtype=torch.float32,
                )
                expected = set(expected_v2)
                if version == "v3":
                    expected.add("gate.e_score_correction_bias")
                self.assertEqual(set(block.state_dict()), expected)
                self.assertEqual(block.hidden_size, config.hidden_size)
                self.assertEqual(block.experts.layer_index, layer_index)
                self.assertEqual(block.experts.parallel_mode, "tp")
                self.assertEqual(
                    block.shared_experts.shared_intermediate_size,
                    config.shared_intermediate_size,
                )
                self.assertEqual(
                    block.gate.grouped_config.routed_scaling_factor,
                    config.routed_scaling_factor,
                )
        config = self.config_module.DeepSeekMoeConfig.from_dict(
            _small_config_dict("v3")
        )
        for bad_layer in (0, 2, 6):
            with self.subTest(layer=bad_layer):
                with self.assertRaisesRegex(ValueError, r"scheduled|\[0"):
                    self.component.build_deepseek_moe_block(config, bad_layer)

    def test_world_one_adapter_strict_loads_and_matches_reference(self):
        for version, layer_index, seed in (("v2", 1, 811), ("v3", 3, 812)):
            with self.subTest(version=version):
                config = self.config_module.DeepSeekMoeConfig.from_dict(
                    _small_config_dict(version)
                )
                hf_state = _make_hf_state(
                    layers=(layer_index,),
                    use_bias=config.uses_correction_bias,
                    seed=seed,
                )
                canonical = self.weights.stack_deepseek_moe_weights(
                    hf_state,
                    moe_layer_indices=(layer_index,),
                    num_experts=config.num_experts,
                    use_correction_bias=config.uses_correction_bias,
                )
                before = {key: id(value) for key, value in canonical.items()}
                runtime = self.component.prepare_deepseek_moe_layer_state(
                    canonical,
                    config,
                    layer_index,
                )
                self.assertEqual(before, {key: id(value) for key, value in canonical.items()})
                self.assertIs(
                    runtime["gate.weight"],
                    canonical[f"layers.{layer_index}.mlp.gate.weight"],
                )
                block = self.component.build_deepseek_moe_block(
                    config,
                    layer_index,
                    dtype=torch.float32,
                )
                result = block.load_state_dict(runtime, strict=True)
                self.assertEqual(result.missing_keys, [])
                self.assertEqual(result.unexpected_keys, [])
                hidden = torch.randn(5, 4, generator=torch.Generator().manual_seed(seed + 1))
                expected = _reference_from_canonical(
                    hidden, canonical, config, layer_index
                )[0]
                with torch.inference_mode():
                    actual = block(hidden)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                if config.uses_correction_bias:
                    self.assertIs(
                        runtime["gate.e_score_correction_bias"],
                        canonical[
                            f"layers.{layer_index}.mlp.gate.e_score_correction_bias"
                        ],
                    )
                    self.assertEqual(
                        block.gate.e_score_correction_bias.dtype,
                        torch.float32,
                    )

    def test_tp2_rank_states_strict_load_and_partials_sum_to_reference(self):
        config = self.config_module.DeepSeekMoeConfig.from_dict(
            _small_config_dict("v3")
        )
        layer_index = 3
        hf_state = _make_hf_state(layers=(layer_index,), use_bias=True, seed=821)
        canonical = self.weights.stack_deepseek_moe_weights(
            hf_state,
            moe_layer_indices=(layer_index,),
            num_experts=config.num_experts,
            use_correction_bias=True,
        )
        hidden = torch.randn(4, 4, generator=torch.Generator().manual_seed(822))
        expected, _, routing_weights, selected_experts = _reference_from_canonical(
            hidden, canonical, config, layer_index
        )
        routed_partials = []
        shared_partials = []
        for rank in range(2):
            tp = self.tp_utils.TPConfig(2, rank, moe_parallel_mode="tp")
            block = self.component.build_deepseek_moe_block(
                config, layer_index, tp_config=tp, dtype=torch.float32
            )
            runtime = self.component.prepare_deepseek_moe_layer_state(
                canonical, config, layer_index, tp_config=tp
            )
            result = block.load_state_dict(runtime, strict=True)
            self.assertEqual((result.missing_keys, result.unexpected_keys), ([], []))
            self.assertEqual(block.experts.gate_up_weight.shape, (4, 4, 4))
            self.assertEqual(block.shared_experts.gate_up_weight.shape, (4, 4))
            routed_partials.append(
                block.experts._forward_eager_local(
                    hidden, selected_experts, routing_weights
                )
            )
            shared_partials.append(block.shared_experts._forward_local(hidden))
        torch.testing.assert_close(
            sum(routed_partials) + sum(shared_partials),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_ep2_rank_states_own_routed_slice_and_replicate_shared(self):
        config = self.config_module.DeepSeekMoeConfig.from_dict(
            _small_config_dict("v2")
        )
        layer_index = 1
        hf_state = _make_hf_state(layers=(layer_index,), seed=831)
        canonical = self.weights.stack_deepseek_moe_weights(
            hf_state,
            moe_layer_indices=(layer_index,),
            num_experts=config.num_experts,
            use_correction_bias=False,
        )
        hidden = torch.randn(4, 4, generator=torch.Generator().manual_seed(832))
        expected, _, routing_weights, selected_experts = _reference_from_canonical(
            hidden, canonical, config, layer_index
        )
        routed_partials = []
        shared_outputs = []
        ownership = []
        for rank in range(2):
            tp = self.tp_utils.TPConfig(2, rank, moe_parallel_mode="ep")
            block = self.component.build_deepseek_moe_block(
                config, layer_index, tp_config=tp, dtype=torch.float32
            )
            runtime = self.component.prepare_deepseek_moe_layer_state(
                canonical, config, layer_index, tp_config=tp
            )
            block.load_state_dict(runtime, strict=True)
            ownership.extend(range(block.experts.expert_start, block.experts.expert_end))
            self.assertEqual(block.experts.gate_up_weight.shape, (2, 4, 8))
            self.assertEqual(block.shared_experts.gate_up_weight.shape, (4, 8))
            routed_partials.append(
                block.experts._forward_eager_local(
                    hidden, selected_experts, routing_weights
                )
            )
            shared_outputs.append(block.shared_experts._forward_local(hidden))
        self.assertEqual(ownership, [0, 1, 2, 3])
        torch.testing.assert_close(shared_outputs[0], shared_outputs[1])
        torch.testing.assert_close(
            sum(routed_partials) + shared_outputs[0],
            expected,
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertFalse(torch.allclose(sum(routed_partials) + sum(shared_outputs), expected))

    def test_multilayer_adapter_extracts_only_requested_layer(self):
        config = self.config_module.DeepSeekMoeConfig.from_dict(
            _small_config_dict("v2")
        )
        hf_state = _make_hf_state(layers=(1, 3), seed=841)
        canonical = self.weights.stack_deepseek_moe_weights(
            hf_state,
            moe_layer_indices=(1, 3),
            num_experts=4,
            use_correction_bias=False,
        )
        keys_before = tuple(canonical)
        identities = {key: id(value) for key, value in canonical.items()}
        runtime = self.component.prepare_deepseek_moe_layer_state(
            canonical, config, 3
        )
        self.assertEqual(tuple(canonical), keys_before)
        self.assertEqual(identities, {key: id(value) for key, value in canonical.items()})
        self.assertEqual(
            set(runtime),
            {
                "gate.weight",
                "experts.gate_up_weight",
                "experts.down_weight",
                "shared_experts.gate_up_weight",
                "shared_experts.down_weight",
            },
        )
        self.assertIs(runtime["gate.weight"], canonical["layers.3.mlp.gate.weight"])

    def test_v2_ignores_extra_bias_and_v3_requires_fp32_bias(self):
        v2 = self.config_module.DeepSeekMoeConfig.from_dict(_small_config_dict("v2"))
        state = _make_hf_state(layers=(1,), seed=851)
        canonical = self.weights.stack_deepseek_moe_weights(
            state,
            moe_layer_indices=(1,),
            num_experts=4,
            use_correction_bias=False,
        )
        extra_key = "layers.1.mlp.gate.e_score_correction_bias"
        canonical[extra_key] = torch.ones(4)
        runtime = self.component.prepare_deepseek_moe_layer_state(canonical, v2, 1)
        self.assertNotIn("gate.e_score_correction_bias", runtime)
        self.assertIn(extra_key, canonical)

        v3 = self.config_module.DeepSeekMoeConfig.from_dict(_small_config_dict("v3"))
        v3_state = _make_hf_state(layers=(3,), use_bias=True, seed=852)
        v3_canonical = self.weights.stack_deepseek_moe_weights(
            v3_state,
            moe_layer_indices=(3,),
            num_experts=4,
            use_correction_bias=True,
        )
        del v3_canonical["layers.3.mlp.gate.e_score_correction_bias"]
        with self.assertRaisesRegex(KeyError, "correction_bias"):
            self.component.prepare_deepseek_moe_layer_state(v3_canonical, v3, 3)

    def test_adapter_fails_closed_for_shape_dtype_and_parallel_mismatch(self):
        config = self.config_module.DeepSeekMoeConfig.from_dict(_small_config_dict("v2"))
        canonical = self.weights.stack_deepseek_moe_weights(
            _make_hf_state(layers=(1,), seed=861),
            moe_layer_indices=(1,),
            num_experts=4,
            use_correction_bias=False,
        )
        prefix = "layers.1.mlp."
        cases = []
        missing = dict(canonical)
        del missing[prefix + "gate.weight"]
        cases.append((missing, None, "missing"))
        wrong_shape = dict(canonical)
        wrong_shape[prefix + "experts.down_weight"] = torch.zeros(4, 4, 3)
        cases.append((wrong_shape, None, "shape"))
        wrong_dtype = dict(canonical)
        wrong_dtype[prefix + "shared_experts.down_weight"] = canonical[
            prefix + "shared_experts.down_weight"
        ].double()
        cases.append((wrong_dtype, None, "dtype"))
        for state, tp, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((KeyError, ValueError), message):
                    self.component.prepare_deepseek_moe_layer_state(
                        state, config, 1, tp_config=tp
                    )
        with self.assertRaisesRegex(ValueError, "moe_intermediate_size"):
            self.component.prepare_deepseek_moe_layer_state(
                canonical,
                config,
                1,
                tp_config=self.tp_utils.TPConfig(3, 0, moe_parallel_mode="tp"),
            )

    def test_direct_loader_api_signatures_and_dependency_boundary(self):
        self.assertEqual(
            tuple(inspect.signature(self.component.build_deepseek_moe_block).parameters),
            ("config", "layer_index", "tp_config", "dtype"),
        )
        self.assertEqual(
            tuple(inspect.signature(self.component.prepare_deepseek_moe_layer_state).parameters),
            ("canonical_state", "config", "layer_index", "tp_config"),
        )
        imports = {
            alias.name
            for node in ast.walk(ast.parse(COMPONENT_PATH.read_text(encoding="utf-8")))
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("transformers", imports)
        self.assertNotIn("accelerate", imports)
        self.assertFalse(any("attention" in name.lower() for name in self.component.__dict__))

    def test_existing_qwen_configs_and_frozen_moe_files_are_unchanged(self):
        baseline = subprocess.run(
            ["git", "show", "HEAD:lite_llama/models/model_config.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8")
        current = CONFIG_PATH.read_text(encoding="utf-8")
        for class_name in (
            "BaseConfig",
            "LlamaConfig",
            "Qwen2Config",
            "Qwen3Config",
            "Qwen3MoeConfig",
            "VisionConfig",
            "LlavaConfig",
            "Qwen3VLVisionConfig",
            "Qwen3VLConfig",
        ):
            with self.subTest(class_name=class_name):
                self.assertEqual(
                    _ast_class_dump(current, class_name),
                    _ast_class_dump(baseline, class_name),
                )
        frozen_paths = (
            "lite_llama/models/moe.py",
            "lite_llama/executor/tp_utils.py",
            "lite_llama/utils/deepseek_moe_weights.py",
            "tests/reference/deepseek_moe_reference.py",
            "tests/models/test_deepseek_moe_reference.py",
            "tests/models/test_deepseek_moe_router.py",
            "tests/models/test_deepseek_moe_block.py",
            "tests/models/test_deepseek_moe_weights.py",
        )
        result = subprocess.run(
            ["git", "diff", "--exit-code", "HEAD", "--", *frozen_paths],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
