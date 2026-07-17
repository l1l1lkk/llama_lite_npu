import ast
from dataclasses import fields
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from tests.reference.deepseek_moe_reference import (
    DeepSeekRoutingSpec,
    deepseek_moe_reference,
)


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT / "lite_llama/utils/deepseek_moe_checkpoint.py"
CONFIG_PATH = ROOT / "lite_llama/models/model_config.py"
WEIGHTS_PATH = ROOT / "lite_llama/utils/deepseek_moe_weights.py"
COMPONENT_PATH = ROOT / "lite_llama/models/deepseek_moe.py"
MOE_PATH = ROOT / "lite_llama/models/moe.py"
TP_UTILS_PATH = ROOT / "lite_llama/executor/tp_utils.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_checkpoint_modules():
    config = _load_module("deepseek_moe_component_config", CONFIG_PATH)
    sys.modules["deepseek_moe_checkpoint_config"] = config
    weights = _load_module("deepseek_moe_checkpoint_weights", WEIGHTS_PATH)
    moe = _load_module("deepseek_moe_component_moe", MOE_PATH)
    tp_utils = _load_module("deepseek_moe_component_tp_utils", TP_UTILS_PATH)
    component = _load_module("deepseek_moe_component_production", COMPONENT_PATH)
    checkpoint = _load_module("deepseek_moe_checkpoint_production", CHECKPOINT_PATH)
    return config, weights, moe, tp_utils, component, checkpoint


def _config_dict(version="v2", **overrides):
    if version == "v2":
        data = {
            "architectures": ["DeepseekV2ForCausalLM"],
            "model_type": "deepseek_v2",
            "scoring_func": "softmax",
            "topk_method": "greedy",
            "routed_scaling_factor": 1.0,
        }
    else:
        data = {
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "routed_scaling_factor": 1.25,
        }
    data.update(
        {
            "hidden_size": 4,
            "num_hidden_layers": 5,
            "intermediate_size": 12,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 3,
            "n_shared_experts": 1,
            "n_group": 2,
            "topk_group": 1,
            "norm_topk_prob": True,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "torch_dtype": "float32",
        }
    )
    data.update(overrides)
    return data


def _make_hf_state(
    *,
    layers=(1,),
    use_bias=False,
    dtype=torch.float32,
    seed=902,
):
    generator = torch.Generator().manual_seed(seed)
    state = {
        "unrelated.weight": torch.randn(2, generator=generator, dtype=dtype),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(
            12,
            4,
            generator=generator,
            dtype=dtype,
        ),
    }
    for layer_index in layers:
        prefix = f"model.layers.{layer_index}.mlp"
        state[f"{prefix}.gate.weight"] = torch.randn(
            4,
            4,
            generator=generator,
            dtype=dtype,
        ) * 0.2
        if use_bias:
            state[f"{prefix}.gate.e_score_correction_bias"] = torch.randn(
                4,
                generator=generator,
                dtype=torch.float16,
            ) * 0.02
        for expert_id in range(4):
            expert = f"{prefix}.experts.{expert_id}"
            state[f"{expert}.gate_proj.weight"] = torch.randn(
                3,
                4,
                generator=generator,
                dtype=dtype,
            ) * 0.2
            state[f"{expert}.up_proj.weight"] = torch.randn(
                3,
                4,
                generator=generator,
                dtype=dtype,
            ) * 0.2
            state[f"{expert}.down_proj.weight"] = torch.randn(
                4,
                3,
                generator=generator,
                dtype=dtype,
            ) * 0.2
        shared = f"{prefix}.shared_experts"
        state[f"{shared}.gate_proj.weight"] = torch.randn(
            3,
            4,
            generator=generator,
            dtype=dtype,
        ) * 0.2
        state[f"{shared}.up_proj.weight"] = torch.randn(
            3,
            4,
            generator=generator,
            dtype=dtype,
        ) * 0.2
        state[f"{shared}.down_proj.weight"] = torch.randn(
            4,
            3,
            generator=generator,
            dtype=dtype,
        ) * 0.2
    return state


def _required_keys(layer_index, *, use_bias):
    prefix = f"model.layers.{layer_index}.mlp"
    keys = [f"{prefix}.gate.weight"]
    if use_bias:
        keys.append(f"{prefix}.gate.e_score_correction_bias")
    for expert_id in range(4):
        expert = f"{prefix}.experts.{expert_id}"
        keys.extend(
            (
                f"{expert}.gate_proj.weight",
                f"{expert}.up_proj.weight",
                f"{expert}.down_proj.weight",
            )
        )
    shared = f"{prefix}.shared_experts"
    keys.extend(
        (
            f"{shared}.gate_proj.weight",
            f"{shared}.up_proj.weight",
            f"{shared}.down_proj.weight",
        )
    )
    return tuple(keys)


def _expected_canonical(state, layer_index, *, use_bias):
    source = f"model.layers.{layer_index}.mlp"
    target = f"layers.{layer_index}.mlp"
    expected = {f"{target}.gate.weight": state[f"{source}.gate.weight"]}
    if use_bias:
        expected[f"{target}.gate.e_score_correction_bias"] = state[
            f"{source}.gate.e_score_correction_bias"
        ].float()
    gate_up = []
    down = []
    for expert_id in range(4):
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
    expected[f"{target}.experts.gate_up_weight"] = torch.stack(gate_up)
    expected[f"{target}.experts.down_weight"] = torch.stack(down)
    shared = f"{source}.shared_experts"
    expected[f"{target}.shared_experts.gate_up_weight"] = torch.cat(
        (
            state[f"{shared}.gate_proj.weight"],
            state[f"{shared}.up_proj.weight"],
        ),
        dim=0,
    )
    expected[f"{target}.shared_experts.down_weight"] = state[
        f"{shared}.down_proj.weight"
    ]
    return expected


def _write_config(directory, data):
    (directory / "config.json").write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_index(directory, shards, *, weight_map=None):
    generated_map = {}
    for filename, tensors in shards.items():
        save_file(tensors, str(directory / filename))
        for key in tensors:
            generated_map[key] = filename
    payload = {"metadata": {"format": "pt"}, "weight_map": generated_map}
    if weight_map is not None:
        payload["weight_map"] = weight_map
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _recording_safe_open(real_safe_open):
    events = {"open": [], "get": [], "close": []}

    def recording(filename, *, framework, device):
        inner = real_safe_open(filename, framework=framework, device=device)

        class RecordingContext:
            def __enter__(self):
                events["open"].append(Path(filename).name)
                self.reader = inner.__enter__()
                return self

            def keys(self):
                return self.reader.keys()

            def get_tensor(self, key):
                events["get"].append((Path(filename).name, key))
                return self.reader.get_tensor(key)

            def __exit__(self, exc_type, exc, traceback):
                events["close"].append(Path(filename).name)
                return inner.__exit__(exc_type, exc, traceback)

        return RecordingContext()

    return recording, events


def _assert_state_equal(test_case, actual, expected):
    test_case.assertEqual(set(actual), set(expected))
    for key in expected:
        with test_case.subTest(key=key):
            test_case.assertEqual(actual[key].device.type, "cpu")
            test_case.assertEqual(actual[key].dtype, expected[key].dtype)
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


class DeepSeekMoeCheckpointContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.config_module,
            cls.weights,
            cls.moe,
            cls.tp_utils,
            cls.component,
            cls.checkpoint,
        ) = _load_checkpoint_modules()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _config(self, version="v2", **overrides):
        data = _config_dict(version, **overrides)
        _write_config(self.directory, data)
        return self.checkpoint.load_deepseek_moe_checkpoint_config(
            self.directory
        )

    def test_checkpoint_reader_api_and_dependency_light_loader(self):
        self.assertEqual(
            str(
                inspect.signature(
                    self.checkpoint.load_deepseek_moe_checkpoint_config
                )
            ),
            "(checkpoints_dir) -> 'DeepSeekMoeConfig'",
        )
        self.assertEqual(
            str(
                inspect.signature(
                    self.checkpoint.load_deepseek_moe_canonical_layer
                )
            ),
            "(checkpoints_dir, config: 'DeepSeekMoeConfig', "
            "layer_index: 'int') -> 'dict[str, torch.Tensor]'",
        )
        source = CHECKPOINT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertTrue(
            imported.isdisjoint(
                {"transformers", "accelerate", "lite_llama", "kernels", "server"}
            )
        )
        self.assertNotIn("except ImportError", source)

    def test_config_loader_uses_audited_v2_v3_contract(self):
        v2 = self._config("v2")
        self.assertEqual(v2.model_type, "deepseek_v2")
        self.assertEqual(v2.topk_method, "group_limited_greedy")
        _write_config(self.directory, _config_dict("v3"))
        v3 = self.checkpoint.load_deepseek_moe_checkpoint_config(self.directory)
        self.assertEqual(v3.model_type, "deepseek_v3")
        self.assertTrue(v3.uses_correction_bias)

    def test_file_config_and_passed_config_must_match_before_weight_io(self):
        cases = (("v3", "v2"), ("v2", "v3"))
        for index, (file_version, passed_version) in enumerate(cases):
            with self.subTest(
                file_version=file_version,
                passed_version=passed_version,
            ):
                case_dir = self.directory / f"mismatch-{index}"
                case_dir.mkdir()
                _write_config(case_dir, _config_dict(file_version))
                state = _make_hf_state(
                    layers=(1,),
                    use_bias=file_version == "v3",
                )
                save_file(state, str(case_dir / "model.safetensors"))
                passed = self.config_module.DeepSeekMoeConfig.from_dict(
                    _config_dict(passed_version)
                )
                recording, events = _recording_safe_open(
                    self.checkpoint.safe_open
                )
                with mock.patch.object(
                    self.checkpoint,
                    "safe_open",
                    recording,
                ), mock.patch.object(
                    self.checkpoint,
                    "_layer_file_mapping",
                ) as file_mapping:
                    with self.assertRaisesRegex(
                        ValueError,
                        "config.json.*match|match.*config.json",
                    ):
                        self.checkpoint.load_deepseek_moe_canonical_layer(
                            case_dir,
                            passed,
                            1,
                        )
                self.assertEqual(events["open"], [])
                self.assertEqual(events["get"], [])
                file_mapping.assert_not_called()

    def test_config_subclass_cannot_override_correction_bias_before_weight_io(self):
        class CorrectionBiasOverride(self.config_module.DeepSeekMoeConfig):
            @property
            def uses_correction_bias(self):
                return False

        file_config = self.config_module.DeepSeekMoeConfig.from_dict(
            _config_dict("v3")
        )
        overridden = CorrectionBiasOverride(
            **{
                item.name: getattr(file_config, item.name)
                for item in fields(file_config)
                if item.init
            }
        )
        _write_config(self.directory, _config_dict("v3"))
        save_file(
            _make_hf_state(layers=(1,), use_bias=True),
            str(self.directory / "model.safetensors"),
        )
        recording, events = _recording_safe_open(self.checkpoint.safe_open)
        with mock.patch.object(
            self.checkpoint,
            "safe_open",
            recording,
        ), mock.patch.object(
            self.checkpoint,
            "_layer_file_mapping",
            wraps=self.checkpoint._layer_file_mapping,
        ) as file_mapping:
            with self.assertRaisesRegex(
                TypeError,
                "exact.*DeepSeekMoeConfig|DeepSeekMoeConfig.*subclass",
            ):
                self.checkpoint.load_deepseek_moe_canonical_layer(
                    self.directory,
                    overridden,
                    1,
                )
        file_mapping.assert_not_called()
        self.assertEqual(events["open"], [])
        self.assertEqual(events["get"], [])

    def test_config_subclass_cannot_override_layer_schedule_before_weight_io(self):
        class LayerScheduleOverride(self.config_module.DeepSeekMoeConfig):
            def is_moe_layer(self, layer_index):
                return True

        file_config = self.config_module.DeepSeekMoeConfig.from_dict(
            _config_dict("v3")
        )
        overridden = LayerScheduleOverride(
            **{
                item.name: getattr(file_config, item.name)
                for item in fields(file_config)
                if item.init
            }
        )
        _write_config(self.directory, _config_dict("v3"))
        save_file(
            _make_hf_state(layers=(0,), use_bias=True),
            str(self.directory / "model.safetensors"),
        )
        recording, events = _recording_safe_open(self.checkpoint.safe_open)
        with mock.patch.object(
            self.checkpoint,
            "safe_open",
            recording,
        ), mock.patch.object(
            self.checkpoint,
            "_layer_file_mapping",
            wraps=self.checkpoint._layer_file_mapping,
        ) as file_mapping:
            with self.assertRaisesRegex(
                TypeError,
                "exact.*DeepSeekMoeConfig|DeepSeekMoeConfig.*subclass",
            ):
                self.checkpoint.load_deepseek_moe_canonical_layer(
                    self.directory,
                    overridden,
                    0,
                )
        file_mapping.assert_not_called()
        self.assertEqual(events["open"], [])
        self.assertEqual(events["get"], [])

    def test_stale_and_semantically_drifted_config_fail_before_weight_io(self):
        _write_config(self.directory, _config_dict("v2"))
        passed = self.checkpoint.load_deepseek_moe_checkpoint_config(
            self.directory
        )
        save_file(
            _make_hf_state(layers=(1,), use_bias=True),
            str(self.directory / "model.safetensors"),
        )
        drift_cases = (
            ("version", _config_dict("v3")),
            ("hidden_size", _config_dict("v2", hidden_size=8)),
            ("num_layers", _config_dict("v2", num_hidden_layers=6)),
            ("dense_intermediate", _config_dict("v2", intermediate_size=16)),
            ("experts", _config_dict("v2", n_routed_experts=8)),
            ("top_k", _config_dict("v2", num_experts_per_tok=1)),
            ("routed_intermediate", _config_dict("v2", moe_intermediate_size=6)),
            ("shared", _config_dict("v2", n_shared_experts=2)),
            ("groups", _config_dict("v2", n_group=1)),
            ("topk_groups", _config_dict("v2", topk_group=2)),
            ("norm", _config_dict("v2", norm_topk_prob=False)),
            ("scale", _config_dict("v2", routed_scaling_factor=2.0)),
            ("dense_schedule", _config_dict("v2", first_k_dense_replace=2)),
            ("frequency", _config_dict("v2", moe_layer_freq=2)),
            ("dtype", _config_dict("v2", torch_dtype="bfloat16")),
        )
        for name, file_config in drift_cases:
            with self.subTest(name=name):
                _write_config(self.directory, file_config)
                recording, events = _recording_safe_open(
                    self.checkpoint.safe_open
                )
                with mock.patch.object(
                    self.checkpoint,
                    "safe_open",
                    recording,
                ), mock.patch.object(
                    self.checkpoint,
                    "_layer_file_mapping",
                ) as file_mapping:
                    with self.assertRaisesRegex(
                        ValueError,
                        "config.json.*match|match.*config.json",
                    ):
                        self.checkpoint.load_deepseek_moe_canonical_layer(
                            self.directory,
                            passed,
                            1,
                        )
                self.assertEqual(events["open"], [])
                self.assertEqual(events["get"], [])
                file_mapping.assert_not_called()

    def test_equivalent_direct_and_canonical_configs_are_accepted(self):
        file_data = _config_dict("v2")
        _write_config(self.directory, file_data)
        state = _make_hf_state(layers=(1,))
        save_file(state, str(self.directory / "model.safetensors"))
        normalized = self.config_module.DeepSeekMoeConfig.from_dict(file_data)
        direct = self.config_module.DeepSeekMoeConfig(
            **{
                item.name: getattr(normalized, item.name)
                for item in fields(normalized)
                if item.init
            }
        )
        canonical_data = dict(file_data)
        for alias, canonical in normalized._ALIASES.items():
            canonical_data[canonical] = canonical_data.pop(alias)
        canonical_data["topk_method"] = "group_limited_greedy"
        canonical = self.config_module.DeepSeekMoeConfig.from_dict(
            canonical_data
        )
        self.assertEqual(direct, normalized)
        self.assertEqual(canonical, normalized)
        for passed in (direct, canonical):
            with self.subTest(kind=type(passed).__name__):
                result = self.checkpoint.load_deepseek_moe_canonical_layer(
                    self.directory,
                    passed,
                    1,
                )
                self.assertEqual(len(result), 5)

    def test_discovered_file_containment_rejects_outside_paths(self):
        root = self.directory.resolve()
        inside = root / "inside.json"
        inside.write_text("{}", encoding="utf-8")
        self.assertEqual(
            self.checkpoint._validated_discovered_file(
                root,
                inside,
                label="inside",
            ),
            inside.resolve(),
        )
        with tempfile.TemporaryDirectory() as outside_dir:
            outside = Path(outside_dir) / "outside.safetensors"
            outside.write_bytes(b"outside")
            for label in ("config.json", "safetensors index", "single shard"):
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, "escapes"):
                        self.checkpoint._validated_discovered_file(
                            root,
                            outside,
                            label=label,
                        )

    def test_config_index_and_single_discovery_use_containment_helper(self):
        config = self._config("v2")
        state = _make_hf_state(layers=(1,))
        save_file(state, str(self.directory / "model.safetensors"))
        real_helper = self.checkpoint._validated_discovered_file
        with mock.patch.object(
            self.checkpoint,
            "_validated_discovered_file",
            wraps=real_helper,
        ) as helper:
            self.checkpoint.load_deepseek_moe_checkpoint_config(self.directory)
            self.checkpoint.load_deepseek_moe_canonical_layer(
                self.directory,
                config,
                1,
            )
        labels = [call.kwargs["label"] for call in helper.call_args_list]
        self.assertIn("config.json", labels)
        self.assertIn("single safetensors file", labels)

        index_dir = self.directory / "indexed"
        index_dir.mkdir()
        _write_config(index_dir, _config_dict("v2"))
        _write_index(
            index_dir,
            {"model-00001-of-00001.safetensors": state},
        )
        with mock.patch.object(
            self.checkpoint,
            "_validated_discovered_file",
            wraps=real_helper,
        ) as helper:
            self.checkpoint.load_deepseek_moe_canonical_layer(
                index_dir,
                config,
                1,
            )
        labels = [call.kwargs["label"] for call in helper.call_args_list]
        self.assertIn("safetensors index", labels)

    def test_config_json_errors_and_v4_source_fail_closed(self):
        with self.assertRaisesRegex(FileNotFoundError, "config.json"):
            self.checkpoint.load_deepseek_moe_checkpoint_config(self.directory)
        (self.directory / "config.json").write_text("{bad", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid config JSON"):
            self.checkpoint.load_deepseek_moe_checkpoint_config(self.directory)
        (self.directory / "config.json").write_text(
            '{"model_type":"deepseek_v2","model_type":"deepseek_v3"}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.checkpoint.load_deepseek_moe_checkpoint_config(self.directory)
        cases = (
            ({**_config_dict("v2"), "model_type": "deepseek_v4"}, "V4"),
            (
                {
                    **_config_dict("v2"),
                    "architectures": ["DeepseekV3ForCausalLM"],
                },
                "architectures",
            ),
            (
                {
                    key: value
                    for key, value in _config_dict("v3").items()
                    if key != "hidden_size"
                },
                "hidden_size",
            ),
        )
        for data, message in cases:
            with self.subTest(message=message):
                _write_config(self.directory, data)
                with self.assertRaisesRegex(ValueError, message):
                    self.checkpoint.load_deepseek_moe_checkpoint_config(
                        self.directory
                    )

    def test_single_file_v2_reads_only_target_layer(self):
        config = self._config("v2")
        state = _make_hf_state(layers=(1, 3), use_bias=True)
        save_file(state, str(self.directory / "model.safetensors"))
        recording, events = _recording_safe_open(self.checkpoint.safe_open)
        with mock.patch.object(self.checkpoint, "safe_open", recording):
            actual = self.checkpoint.load_deepseek_moe_canonical_layer(
                self.directory,
                config,
                1,
            )
        expected = _expected_canonical(state, 1, use_bias=False)
        _assert_state_equal(self, actual, expected)
        self.assertEqual(events["open"], ["model.safetensors"])
        self.assertEqual(events["close"], ["model.safetensors"])
        self.assertEqual(
            [key for _, key in events["get"]],
            list(_required_keys(1, use_bias=False)),
        )
        self.assertNotIn(
            "model.layers.1.mlp.gate.e_score_correction_bias",
            [key for _, key in events["get"]],
        )
        self.assertFalse(any("layers.3" in key for _, key in events["get"]))
        self.assertFalse(any(key == "unrelated.weight" for _, key in events["get"]))

    def test_sharded_v3_opens_each_relevant_shard_once(self):
        config = self._config("v3")
        state = _make_hf_state(layers=(1, 3), use_bias=True)
        target = set(_required_keys(1, use_bias=True))
        keys = list(state)
        shard_a = {key: state[key] for key in keys if key in target and len(key) % 2}
        shard_b = {key: state[key] for key in keys if key in target and not len(key) % 2}
        shard_other = {key: state[key] for key in keys if key not in target}
        _write_index(
            self.directory,
            {
                "model-00001-of-00003.safetensors": shard_a,
                "model-00002-of-00003.safetensors": shard_b,
                "model-00003-of-00003.safetensors": shard_other,
            },
        )
        recording, events = _recording_safe_open(self.checkpoint.safe_open)
        with mock.patch.object(self.checkpoint, "safe_open", recording):
            actual = self.checkpoint.load_deepseek_moe_canonical_layer(
                self.directory,
                config,
                1,
            )
        _assert_state_equal(
            self,
            actual,
            _expected_canonical(state, 1, use_bias=True),
        )
        self.assertEqual(
            sorted(events["open"]),
            [
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
            ],
        )
        self.assertEqual(len(events["open"]), len(set(events["open"])))
        self.assertEqual(sorted(events["close"]), sorted(events["open"]))
        self.assertEqual({key for _, key in events["get"]}, target)
        self.assertEqual(
            actual["layers.1.mlp.gate.e_score_correction_bias"].dtype,
            torch.float32,
        )

    def test_canonical_layer_strict_loads_and_matches_reference(self):
        for version in ("v2", "v3"):
            with self.subTest(version=version):
                case_dir = self.directory / version
                case_dir.mkdir()
                _write_config(case_dir, _config_dict(version))
                config = self.checkpoint.load_deepseek_moe_checkpoint_config(
                    case_dir
                )
                state = _make_hf_state(
                    layers=(1,),
                    use_bias=version == "v3",
                )
                save_file(state, str(case_dir / "model.safetensors"))
                canonical = self.checkpoint.load_deepseek_moe_canonical_layer(
                    case_dir,
                    config,
                    1,
                )
                runtime = self.component.prepare_deepseek_moe_layer_state(
                    canonical,
                    config,
                    1,
                )
                block = self.component.build_deepseek_moe_block(
                    config,
                    1,
                    dtype=torch.float32,
                )
                result = block.load_state_dict(runtime, strict=True)
                self.assertEqual(result.missing_keys, [])
                self.assertEqual(result.unexpected_keys, [])
                prefix = "layers.1.mlp."
                spec = DeepSeekRoutingSpec(
                    num_experts=config.num_experts,
                    experts_per_token=config.num_experts_per_tok,
                    num_groups=config.num_groups,
                    topk_groups=config.topk_groups,
                    score_func=config.score_func,
                    topk_method=config.topk_method,
                    norm_topk_prob=config.norm_topk_prob,
                    routed_scaling_factor=config.routed_scaling_factor,
                )
                for token_count in (0, 3):
                    with self.subTest(
                        version=version,
                        token_count=token_count,
                    ):
                        hidden = torch.randn(token_count, 4)
                        expected, _, _, _ = deepseek_moe_reference(
                            hidden,
                            canonical[prefix + "gate.weight"],
                            canonical[
                                prefix + "experts.gate_up_weight"
                            ].transpose(1, 2).contiguous(),
                            canonical[
                                prefix + "experts.down_weight"
                            ].transpose(1, 2).contiguous(),
                            spec,
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
                        torch.testing.assert_close(
                            block(hidden),
                            expected,
                            rtol=1e-5,
                            atol=1e-6,
                        )

    def test_layer_index_must_be_scheduled_moe_layer(self):
        config = self._config("v2", first_k_dense_replace=2, moe_layer_freq=2)
        save_file(
            _make_hf_state(layers=(2,)),
            str(self.directory / "model.safetensors"),
        )
        cases = (
            (True, TypeError, "integer"),
            (-1, ValueError, "must be in"),
            (5, ValueError, "must be in"),
            (0, ValueError, "not a scheduled"),
            (3, ValueError, "not a scheduled"),
        )
        for layer_index, error, message in cases:
            with self.subTest(layer_index=layer_index):
                with self.assertRaisesRegex(error, message):
                    self.checkpoint.load_deepseek_moe_canonical_layer(
                        self.directory,
                        config,
                        layer_index,
                    )

    def test_multiple_files_without_index_and_index_ambiguity_fail(self):
        config = self._config("v2")
        state = _make_hf_state(layers=(1,))
        save_file(state, str(self.directory / "one.safetensors"))
        save_file(state, str(self.directory / "two.safetensors"))
        with self.assertRaisesRegex(ValueError, "multiple safetensors files"):
            self.checkpoint.load_deepseek_moe_canonical_layer(
                self.directory,
                config,
                1,
            )
        (self.directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {}}),
            encoding="utf-8",
        )
        (self.directory / "other.safetensors.index.json").write_text(
            json.dumps({"weight_map": {}}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "multiple safetensors indexes"):
            self.checkpoint.load_deepseek_moe_canonical_layer(
                self.directory,
                config,
                1,
            )

    def test_index_json_and_mapping_types_fail_closed(self):
        config = self._config("v2")
        cases = (
            ("[]", "JSON object"),
            ("{}", "missing weight_map"),
            ('{"weight_map": []}', "non-empty object"),
            ('{"weight_map": {"x": 3}}', "non-empty string"),
            (
                '{"weight_map":{"x":"a.safetensors",'
                '"x":"b.safetensors"}}',
                "duplicate JSON key",
            ),
        )
        for content, message in cases:
            with self.subTest(message=message):
                (self.directory / "model.safetensors.index.json").write_text(
                    content,
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    self.checkpoint.load_deepseek_moe_canonical_layer(
                        self.directory,
                        config,
                        1,
                    )
        (self.directory / "model.safetensors.index.json").unlink()
        (self.directory / "weights.safetensors.index.json").write_text(
            json.dumps({"weight_map": {}}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "must be named"):
            self.checkpoint.load_deepseek_moe_canonical_layer(
                self.directory,
                config,
                1,
            )

    def test_index_rejects_unsafe_missing_and_wrong_suffix_paths(self):
        config = self._config("v2")
        gate_key = _required_keys(1, use_bias=False)[0]
        cases = (
            (str((self.directory / "absolute.safetensors").resolve()), "unsafe"),
            ("../outside.safetensors", "unsafe"),
            ("weights.bin", "must end with"),
            ("missing.safetensors", "is missing"),
        )
        for shard_path, message in cases:
            with self.subTest(shard_path=shard_path):
                (self.directory / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {gate_key: shard_path}}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex((ValueError, FileNotFoundError), message):
                    self.checkpoint.load_deepseek_moe_canonical_layer(
                        self.directory,
                        config,
                        1,
                    )

    def test_index_requires_every_target_key_mapping(self):
        config = self._config("v2")
        state = _make_hf_state(layers=(1,))
        required = list(_required_keys(1, use_bias=False))
        shard = "model-00001-of-00001.safetensors"
        save_file(state, str(self.directory / shard))
        weight_map = {key: shard for key in required[:-1]}
        _write_index(self.directory, {}, weight_map=weight_map)
        with self.assertRaisesRegex(KeyError, required[-1]):
            self.checkpoint.load_deepseek_moe_canonical_layer(
                self.directory,
                config,
                1,
            )

    def test_mapped_missing_tensor_closes_context_without_return(self):
        config = self._config("v3")
        state = _make_hf_state(layers=(1,), use_bias=True)
        required = list(_required_keys(1, use_bias=True))
        missing = required[-1]
        del state[missing]
        shard = "model-00001-of-00001.safetensors"
        save_file(state, str(self.directory / shard))
        weight_map = {key: shard for key in required}
        _write_index(self.directory, {}, weight_map=weight_map)
        recording, events = _recording_safe_open(self.checkpoint.safe_open)
        with mock.patch.object(self.checkpoint, "safe_open", recording):
            with self.assertRaisesRegex(KeyError, missing):
                self.checkpoint.load_deepseek_moe_canonical_layer(
                    self.directory,
                    config,
                    1,
                )
        self.assertEqual(events["open"], [shard])
        self.assertEqual(events["close"], [shard])
        self.assertNotIn(missing, [key for _, key in events["get"]])

    def test_converter_rejects_bad_tensor_shape_dtype_and_missing_v3_bias(self):
        cases = []
        wrong_shape = _make_hf_state(layers=(1,))
        wrong_shape["model.layers.1.mlp.gate.weight"] = torch.randn(3, 4)
        cases.append(("v2", wrong_shape, "must have shape"))
        wrong_dtype = _make_hf_state(layers=(1,))
        wrong_dtype[
            "model.layers.1.mlp.experts.0.gate_proj.weight"
        ] = torch.ones(3, 4, dtype=torch.int64)
        cases.append(("v2", wrong_dtype, "floating point"))
        mixed_dtype = _make_hf_state(layers=(1,))
        mixed_dtype[
            "model.layers.1.mlp.experts.0.gate_proj.weight"
        ] = torch.ones(3, 4, dtype=torch.float64)
        cases.append(("v2", mixed_dtype, "dtype/device mismatch"))
        missing_bias = _make_hf_state(layers=(1,), use_bias=False)
        cases.append(("v3", missing_bias, "correction_bias"))

        for index, (version, state, message) in enumerate(cases):
            with self.subTest(version=version, message=message):
                case_dir = self.directory / f"case-{index}"
                case_dir.mkdir()
                config = self.config_module.DeepSeekMoeConfig.from_dict(
                    _config_dict(version)
                )
                _write_config(case_dir, _config_dict(version))
                save_file(state, str(case_dir / "model.safetensors"))
                with self.assertRaisesRegex(
                    (KeyError, TypeError, ValueError),
                    message,
                ):
                    self.checkpoint.load_deepseek_moe_canonical_layer(
                        case_dir,
                        config,
                        1,
                    )

    def test_frozen_production_and_qwen_files_are_unchanged(self):
        frozen = (
            "lite_llama/models/model_config.py",
            "lite_llama/models/deepseek_moe.py",
            "lite_llama/models/moe.py",
            "lite_llama/executor/tp_utils.py",
            "lite_llama/utils/deepseek_moe_weights.py",
            "apply_weight_convert.py",
            "lite_llama/executor/executor_struct.py",
            "lite_llama/executor/model_executor.py",
            "tests/reference/deepseek_moe_reference.py",
            "tests/models/test_deepseek_moe_reference.py",
            "tests/models/test_deepseek_moe_router.py",
            "tests/models/test_deepseek_moe_block.py",
            "tests/models/test_deepseek_moe_weights.py",
            "tests/models/test_deepseek_moe_component.py",
            "tests/models/test_moe_reference.py",
            "tests/models/test_qwen3_moe.py",
        )
        result = subprocess.run(
            ["git", "diff", "--exit-code", "HEAD", "--", *frozen],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            result.stdout.decode("utf-8") + result.stderr.decode("utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
