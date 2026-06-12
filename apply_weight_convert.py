#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_weight_convert.py
~~~~~~~~~~~~~~~~~~~~
将 Qwen-2/3、Llama、LLaVA 等 HuggingFace / PyTorch-bin 权重
整理为 lite_llama 框架的自定义格式模型权重的小工具。

Usage
-----
python lite_llama/apply_weight_convert.py /path/to/weights [--model-type qwen3] [--device cuda]

Author: harleyszhang (2025-06-08)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Mapping

import torch
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM,
                          LlavaConfig, LlavaForConditionalGeneration)

from lite_llama.utils.logger import get_logger
from lite_llama.utils.qwen3_moe_weights import stack_qwen3_moe_weights

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 通用工具函数
# --------------------------------------------------------------------------- #
def ensure_dir(path: Path) -> Path:
    """若目录不存在则创建，最后返回自身。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_state_dict(out_dir: Path, model_id: str, state: dict[str, torch.Tensor]) -> None:
    """保存 state_dict 并打印信息。"""
    torch.save(state, out_dir / f"{model_id}.pth", _use_new_zipfile_serialization=True)
    logger.info("✅ 已保存权重到 %s", out_dir / f"{model_id}.pth")


def copy_metadata(src: Path, dst: Path) -> None:
    """复制 *.json 与 tokenizer.model 等辅助文件。"""
    for file in src.glob("*.json"):
        shutil.copy2(file, dst)
    tok = src / "tokenizer.model"
    if tok.exists():
        shutil.copy2(tok, dst)


# --------------------------------------------------------------------------- #
# 修订后的 merge_kv_weights —— 只生成 kv_proj_weight，下划线风格
# --------------------------------------------------------------------------- #
def merge_kv_weights(state: dict[str, torch.Tensor],
                     prefix: str,
                     with_bias: bool = False) -> None:
    """
    将 K/V 投影合并为 kv_proj_weight / kv_proj_bias（可选），
    完全使用 **下划线** 键名，保证与 lite-llama 的 Qwen3 实现一致。
    同时若此前转换过留下旧的 kv_proj.weight，也会被删除。
    """
    # ---------- 1. 找到现有 K/V ----------
    # 两种候选命名：点号风格  layers.0.self_attn.k_proj.weight
    #            下划线风格 layers.0.self_attn.k_proj_weight
    candidates = [
        (f"{prefix}.k_proj.weight", f"{prefix}.v_proj.weight"),
        (f"{prefix}.k_proj_weight", f"{prefix}.v_proj_weight"),
    ]
    for k_key, v_key in candidates:
        if k_key in state and v_key in state:
            break
    else:  # 没有任何一对匹配
        return

    # ---------- 2. 合并权重 ----------
    fused_k = f"{prefix}.kv_proj_weight"            # 目标键（下划线）
    state[fused_k] = torch.cat([state[k_key], state[v_key]], dim=0)
    del state[k_key], state[v_key]

    # ---------- 3. 合并 bias（可选） ----------
    if with_bias:
        bias_cands = [
            (f"{prefix}.k_proj.bias",  f"{prefix}.v_proj.bias"),
            (f"{prefix}.k_proj_bias",  f"{prefix}.v_proj_bias"),
        ]
        for kb_key, vb_key in bias_cands:
            if kb_key in state and vb_key in state:
                fused_b = f"{prefix}.kv_proj_bias"
                state[fused_b] = torch.cat([state[kb_key], state[vb_key]], dim=0)
                del state[kb_key], state[vb_key]
                break

    # ---------- 4. 如有旧版 kv_proj.weight，顺带删掉 ----------
    old_key = f"{prefix}.kv_proj.weight"
    if old_key in state:
        del state[old_key]


def build_mapping(common: Mapping[str, str],
                  layer_tpl: Mapping[str, str],
                  num_layers: int) -> dict[str, str]:
    """根据层数展开模板映射表。"""
    mapping = dict(common)
    for i in range(num_layers):
        mapping.update({hf.format(i=i): custom.format(i=i) for hf, custom in layer_tpl.items()})
    return mapping

# --------------------------------------------------------------------------- #
# 具体各模型映射规则
# --------------------------------------------------------------------------- #
_SPEC = {
    # Qwen-2
    "qwen2": {
        "common": {
            "model.norm.weight":         "norm_weight",
            "model.embed_tokens.weight": "embed_tokens.weight",
            "lm_head.weight":            "lm_head_weight",
        },
        "layer": {
            # q_proj/k_proj/... 同下
            "model.layers.{i}.self_attn.q_proj.weight":  "layers.{i}.self_attn.q_proj_weight",
            "model.layers.{i}.self_attn.q_proj.bias":    "layers.{i}.self_attn.q_proj_bias",
            "model.layers.{i}.self_attn.k_proj.weight":  "layers.{i}.self_attn.k_proj_weight",
            "model.layers.{i}.self_attn.k_proj.bias":    "layers.{i}.self_attn.k_proj_bias",
            "model.layers.{i}.self_attn.v_proj.weight":  "layers.{i}.self_attn.v_proj_weight",
            "model.layers.{i}.self_attn.v_proj.bias":    "layers.{i}.self_attn.v_proj_bias",
            "model.layers.{i}.self_attn.o_proj.weight":  "layers.{i}.self_attn.o_proj_weight",
            "model.layers.{i}.mlp.gate_proj.weight":     "layers.{i}.mlp.gate_proj.weight",
            "model.layers.{i}.mlp.up_proj.weight":       "layers.{i}.mlp.up_proj.weight",
            "model.layers.{i}.mlp.down_proj.weight":     "layers.{i}.mlp.down_proj.weight",
            "model.layers.{i}.input_layernorm.weight":   "layers.{i}.input_layernorm_weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attention_layernorm_weight",
        },
        "merge_bias": True,
    },

    # Qwen-3
    "qwen3": {
        "common": {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight":         "norm_weight",
            "lm_head.weight":            "lm_head_weight",
        },
        "layer": {
            "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj_weight",
            "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj_weight",
            "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj_weight",
            "model.layers.{i}.self_attn.q_norm.weight": "layers.{i}.self_attn.q_norm_weight",
            "model.layers.{i}.self_attn.k_norm.weight": "layers.{i}.self_attn.k_norm_weight",
            "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj_weight",
            "model.layers.{i}.mlp.gate_proj.weight":    "layers.{i}.mlp.gate_proj.weight",
            "model.layers.{i}.mlp.up_proj.weight":      "layers.{i}.mlp.up_proj.weight",
            "model.layers.{i}.mlp.down_proj.weight":    "layers.{i}.mlp.down_proj.weight",
            "model.layers.{i}.input_layernorm.weight":  "layers.{i}.input_layernorm_weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attention_layernorm_weight",
        },
        "merge_bias": False,
    },

    # Qwen-3 MoE
    "qwen3_moe": {
        "common": {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight":         "norm_weight",
            "lm_head.weight":            "lm_head_weight",
        },
        "layer": {
            "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj_weight",
            "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj_weight",
            "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj_weight",
            "model.layers.{i}.self_attn.q_norm.weight": "layers.{i}.self_attn.q_norm_weight",
            "model.layers.{i}.self_attn.k_norm.weight": "layers.{i}.self_attn.k_norm_weight",
            "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj_weight",
            "model.layers.{i}.input_layernorm.weight":  "layers.{i}.input_layernorm_weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attention_layernorm_weight",
        },
        "merge_bias": False,
    },

    # Qwen3-VL (vision encoder + Qwen3 language model)
    "qwen3_vl": {
        "common": {
            # --- language model (Qwen3-based) ---
            "model.language_model.embed_tokens.weight":     "language_model.embed_tokens.weight",
            "model.language_model.norm.weight":              "language_model.norm_weight",
            "lm_head.weight":                                "language_model.lm_head_weight",
            # --- vision encoder ---
            "model.visual.patch_embed.proj.weight":          "visual.patch_embed.proj.weight",
            "model.visual.patch_embed.proj.bias":            "visual.patch_embed.proj.bias",
            "model.visual.pos_embed.weight":                 "visual.pos_embed.weight",
            "model.visual.merger.norm.weight":               "visual.merger.norm.weight",
            "model.visual.merger.norm.bias":                 "visual.merger.norm.bias",
            "model.visual.merger.linear_fc1.weight":         "visual.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc1.bias":           "visual.merger.linear_fc1.bias",
            "model.visual.merger.linear_fc2.weight":         "visual.merger.linear_fc2.weight",
            "model.visual.merger.linear_fc2.bias":           "visual.merger.linear_fc2.bias",
        },
        "layer": {
            # --- language model decoder layers ---
            "model.language_model.layers.{i}.self_attn.q_proj.weight":   "language_model.layers.{i}.self_attn.q_proj_weight",
            "model.language_model.layers.{i}.self_attn.k_proj.weight":   "language_model.layers.{i}.self_attn.k_proj_weight",
            "model.language_model.layers.{i}.self_attn.v_proj.weight":   "language_model.layers.{i}.self_attn.v_proj_weight",
            "model.language_model.layers.{i}.self_attn.q_norm.weight":   "language_model.layers.{i}.self_attn.q_norm_weight",
            "model.language_model.layers.{i}.self_attn.k_norm.weight":   "language_model.layers.{i}.self_attn.k_norm_weight",
            "model.language_model.layers.{i}.self_attn.o_proj.weight":   "language_model.layers.{i}.self_attn.o_proj_weight",
            "model.language_model.layers.{i}.mlp.gate_proj.weight":      "language_model.layers.{i}.mlp.gate_proj.weight",
            "model.language_model.layers.{i}.mlp.up_proj.weight":        "language_model.layers.{i}.mlp.up_proj.weight",
            "model.language_model.layers.{i}.mlp.down_proj.weight":      "language_model.layers.{i}.mlp.down_proj.weight",
            "model.language_model.layers.{i}.input_layernorm.weight":    "language_model.layers.{i}.input_layernorm_weight",
            "model.language_model.layers.{i}.post_attention_layernorm.weight": "language_model.layers.{i}.post_attention_layernorm_weight",
        },
        "vision_layer": {
            # --- vision encoder blocks ---
            "model.visual.blocks.{v}.norm1.weight":         "visual.blocks.{v}.norm1.weight",
            "model.visual.blocks.{v}.norm1.bias":           "visual.blocks.{v}.norm1.bias",
            "model.visual.blocks.{v}.norm2.weight":         "visual.blocks.{v}.norm2.weight",
            "model.visual.blocks.{v}.norm2.bias":           "visual.blocks.{v}.norm2.bias",
            "model.visual.blocks.{v}.attn.qkv.weight":      "visual.blocks.{v}.attn.qkv.weight",
            "model.visual.blocks.{v}.attn.qkv.bias":        "visual.blocks.{v}.attn.qkv.bias",
            "model.visual.blocks.{v}.attn.proj.weight":     "visual.blocks.{v}.attn.proj.weight",
            "model.visual.blocks.{v}.attn.proj.bias":       "visual.blocks.{v}.attn.proj.bias",
            "model.visual.blocks.{v}.mlp.linear_fc1.weight": "visual.blocks.{v}.mlp.linear_fc1.weight",
            "model.visual.blocks.{v}.mlp.linear_fc1.bias":  "visual.blocks.{v}.mlp.linear_fc1.bias",
            "model.visual.blocks.{v}.mlp.linear_fc2.weight": "visual.blocks.{v}.mlp.linear_fc2.weight",
            "model.visual.blocks.{v}.mlp.linear_fc2.bias":  "visual.blocks.{v}.mlp.linear_fc2.bias",
        },
        "deepstack_visual_indexes": (8, 16, 24),
        "merge_bias": False,
    },

    # Llama-HF
    "llama": {
        "common": {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight":         "norm_weight",
            "lm_head.weight":            "lm_head.weight",
        },
        "layer": {
            "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj.weight",
            "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj.weight",
            "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj.weight",
            "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj.weight",
            "model.layers.{i}.mlp.gate_proj.weight":    "layers.{i}.mlp.gate_proj.weight",
            "model.layers.{i}.mlp.up_proj.weight":      "layers.{i}.mlp.up_proj.weight",
            "model.layers.{i}.mlp.down_proj.weight":    "layers.{i}.mlp.down_proj.weight",
            "model.layers.{i}.input_layernorm.weight":  "layers.{i}.attention_norm_weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.ffn_norm_weight",
        },
        "merge_bias": False,
    },

    # Llama-bin（原 Fairseq/Llama.PTH 格式）
    "llama-bin": {
        "common": {
            "tok_embeddings.weight": "embed_tokens.weight",
            "norm.weight":           "norm_weight",
            "output.weight":         "lm_head.weight",
        },
        "layer": {
            "layers.{i}.attention.wq.weight": "layers.{i}.attention.q_proj.weight",
            "layers.{i}.attention.wk.weight": "layers.{i}.attention.k_proj.weight",
            "layers.{i}.attention.wv.weight": "layers.{i}.attention.v_proj.weight",
            "layers.{i}.attention.wo.weight": "layers.{i}.attention.o_proj.weight",
            "layers.{i}.feed_forward.w1.weight": "layers.{i}.feed_forward.gate_proj.weight",
            "layers.{i}.feed_forward.w3.weight": "layers.{i}.feed_forward.up_proj.weight",
            "layers.{i}.feed_forward.w2.weight": "layers.{i}.feed_forward.down_proj.weight",
            "layers.{i}.attention_norm.weight":  "layers.{i}.attention_norm_weight",
            "layers.{i}.ffn_norm.weight":        "layers.{i}.ffn_norm_weight",
        },
        "merge_bias": False,
    },

    # LLaVA-Llama
    "llava": {
        "common": {
            "language_model.model.embed_tokens.weight": "language_model.embed_tokens.weight",
            "language_model.model.norm.weight":         "language_model.norm_weight",
            "language_model.lm_head.weight":            "language_model.lm_head.weight",
        },
        "layer": {
            "language_model.model.layers.{i}.self_attn.q_proj.weight": "language_model.layers.{i}.self_attn.q_proj.weight",
            "language_model.model.layers.{i}.self_attn.k_proj.weight": "language_model.layers.{i}.self_attn.k_proj.weight",
            "language_model.model.layers.{i}.self_attn.v_proj.weight": "language_model.layers.{i}.self_attn.v_proj.weight",
            "language_model.model.layers.{i}.self_attn.o_proj.weight": "language_model.layers.{i}.self_attn.o_proj.weight",
            "language_model.model.layers.{i}.mlp.gate_proj.weight":    "language_model.layers.{i}.mlp.gate_proj.weight",
            "language_model.model.layers.{i}.mlp.up_proj.weight":      "language_model.layers.{i}.mlp.up_proj.weight",
            "language_model.model.layers.{i}.mlp.down_proj.weight":    "language_model.layers.{i}.mlp.down_proj.weight",
            "language_model.model.layers.{i}.input_layernorm.weight":  "language_model.layers.{i}.attention_norm_weight",
            "language_model.model.layers.{i}.post_attention_layernorm.weight": "language_model.layers.{i}.ffn_norm_weight",
        },
        "merge_bias": False,
    },
}

# --------------------------------------------------------------------------- #
# 核心转换逻辑
# --------------------------------------------------------------------------- #
def convert(checkpoints_dir: Path,
            hf_state: dict[str, torch.Tensor],
            model_type: str,
            layers_info: dict) -> dict[str, torch.Tensor]:
    """执行主转换流程并把结果保存到 my_weight/<model_id>/ 目录。"""
    spec = _SPEC[model_type]
    num_layers = layers_info["num_layers"]
    mapping = build_mapping(spec["common"], spec["layer"], num_layers)
    new_sd: dict[str, torch.Tensor] = {}

    # ---------- 1a. 重映射 LLM layers ----------
    for k, v in tqdm(hf_state.items(), desc=f"[{model_type}] 权重重映射"):
        if (ck := mapping.get(k)) is not None:
            new_sd[ck] = v
        else:
            logger.debug("忽略未映射参数 %s", k)

    # ---------- 1b. Qwen3VL: 映射 vision encoder ----------
    if model_type == "qwen3_vl":
        vision_depth = layers_info.get("vision_depth", 27)
        deepstack_indexes = layers_info.get("deepstack_indexes", [8, 16, 24])

        # Vision blocks
        for v_idx in range(vision_depth):
            for hf_pat, lit_pat in spec["vision_layer"].items():
                hf_key = hf_pat.format(v=v_idx)
                lit_key = lit_pat.format(v=v_idx)
                if hf_key in hf_state:
                    new_sd[lit_key] = hf_state[hf_key]

        # Deepstack mergers
        for ds_idx, vis_layer in enumerate(deepstack_indexes):
            for suffix in ("norm.weight", "norm.bias",
                           "linear_fc1.weight", "linear_fc1.bias",
                           "linear_fc2.weight", "linear_fc2.bias"):
                hf_key = f"model.visual.deepstack_merger_list.{ds_idx}.{suffix}"
                lit_key = f"visual.deepstack_merger_list.{ds_idx}.{suffix}"
                if hf_key in hf_state:
                    new_sd[lit_key] = hf_state[hf_key]

        logger.info("Vision blocks: %d, deepstack mergers: %d", vision_depth, len(deepstack_indexes))

    if model_type == "qwen3_moe":
        num_experts = layers_info["num_experts"]
        new_sd.update(
            stack_qwen3_moe_weights(
                hf_state,
                num_layers=num_layers,
                num_experts=num_experts,
                consume=True,
            )
        )
        logger.info(
            "Qwen3 MoE experts stacked: layers=%d experts_per_layer=%d",
            num_layers,
            num_experts,
        )

    # ---------- 2. 对 LLM 部分执行 KV 合并 ----------
    if model_type.startswith("qwen") or model_type.startswith("llama"):
        llm_prefix = "language_model." if model_type == "qwen3_vl" else ""
        for i in range(num_layers):
            prefix = f"{llm_prefix}layers.{i}.self_attn"
            merge_kv_weights(new_sd, prefix, with_bias=spec["merge_bias"])

    # ---------- 2b. 处理 tie_word_embeddings ----------
    if model_type == "qwen3_vl":
        lm_head_key = "language_model.lm_head_weight"
        embed_key = "language_model.embed_tokens.weight"
        if lm_head_key not in new_sd and embed_key in new_sd:
            logger.info("lm_head 与 embed_tokens 权重绑定，从 embed_tokens 复制")
            new_sd[lm_head_key] = new_sd[embed_key].clone()

    # ---------- 3. 保存 ----------
    script_root = Path(__file__).resolve().parent
    out_dir = ensure_dir(script_root / "my_weight" / checkpoints_dir.name)
    save_state_dict(out_dir, checkpoints_dir.name, new_sd)
    copy_metadata(checkpoints_dir, out_dir)

    logger.info("转换完成，共 %d 个参数", len(new_sd))
    return new_sd



# --------------------------------------------------------------------------- #
# CLI 辅助：由 config.json 判别模型类型
# --------------------------------------------------------------------------- #
def detect_model_type(checkpoints_dir: Path) -> str:
    """
    读取 config.json 中的 model_type 字段。
    若该字段在 _SPEC 中无法找到，则抛出错误提示。
    """
    cfg = AutoConfig.from_pretrained(checkpoints_dir, trust_remote_code=True)
    mtype = cfg.model_type.lower()
    # 某些模型可能需要额外归一化 / 映射
    alias = {
        "qwen2":     "qwen2",
        "qwen3":     "qwen3",
        "qwen3_moe": "qwen3_moe",
        "llama":     "llama",
        "llava":     "llava",
        "qwen3_vl":  "qwen3_vl",
    }.get(mtype, mtype)      # 默认原样返回
    if alias not in _SPEC:
        raise ValueError(f"暂不支持的 model_type '{mtype}'，请检查映射表")
    return alias


def load_hf_state(checkpoints_dir: Path,
                  model_type: str,
                  device: str = "cpu") -> dict[str, torch.Tensor]:
    """加载 HF / safetensors / bin 权重到 state_dict。"""
    # 优先尝试 safetensors 直接加载（速度更快、内存更低）
    safetensor_files = sorted(checkpoints_dir.glob("*.safetensors"))
    if safetensor_files:
        logger.info("检测到 %d 个 safetensors 文件，直接加载...", len(safetensor_files))
        from safetensors.torch import load_file as safetensors_load
        state_dict = {}
        for sf in tqdm(safetensor_files, desc="加载 safetensors"):
            state_dict.update(safetensors_load(str(sf), device=device))
        return state_dict

    # 回退到 HuggingFace 加载
    if model_type == "llava" or model_type == "qwen3_vl":
        model = (AutoModelForCausalLM
                 .from_pretrained(checkpoints_dir, torch_dtype=torch.float16,
                                  low_cpu_mem_usage=True, trust_remote_code=True)
                 .to(device))
    else:
        model = (AutoModelForCausalLM
                 .from_pretrained(checkpoints_dir, torch_dtype=torch.float16,
                                  low_cpu_mem_usage=True)
                 .to(device))
    return model.state_dict()


def get_num_layers(checkpoints_dir: Path, model_type: str) -> dict:
    """从 config 中提取层数信息。返回 dict 包含 num_layers (LLM) 和 vision_depth (ViT)。"""
    if model_type == "llava":
        cfg = LlavaConfig.from_pretrained(checkpoints_dir)
        return {"num_layers": cfg.text_config.num_hidden_layers}
    cfg = AutoConfig.from_pretrained(checkpoints_dir, trust_remote_code=True)
    if model_type == "qwen3_vl":
        vis_cfg = cfg.vision_config
        deepstack = getattr(vis_cfg, "deepstack_visual_indexes", None) or [8, 16, 24]
        return {
            "num_layers": cfg.text_config.num_hidden_layers,
            "vision_depth": getattr(vis_cfg, "depth", 27),
            "deepstack_indexes": list(deepstack),
        }
    result = {"num_layers": cfg.num_hidden_layers}
    if model_type == "qwen3_moe":
        result["num_experts"] = cfg.num_experts
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HF / bin checkpoints into Lite-LLaMA format.")
    parser.add_argument("checkpoints_dir", type=Path, help="模型权重目录")
    parser.add_argument("--model-type",
                        choices=_SPEC.keys(),
                        help="显式指定模型类型；默认根据目录名猜测")
    parser.add_argument("--device", default="cpu",
                        help="加载权重时使用的设备 (default: cpu，权重转换不需要 GPU)")
    args = parser.parse_args()

    ckpt_dir: Path = args.checkpoints_dir.resolve()
    
    # 1️⃣ **直接从 config.json 读取 model_type** ↓
    model_type = args.model_type or detect_model_type(ckpt_dir)
    logger.info("检测到 model_type = %s", model_type)

    # 2️⃣ 获取层数
    layers_info = get_num_layers(ckpt_dir, model_type)
    logger.info("层数信息 %s", layers_info)

    # 3️⃣ 加载权重并执行转换
    hf_sd = load_hf_state(ckpt_dir, model_type, device=args.device)
    convert(ckpt_dir, hf_sd, model_type, layers_info)

# tests/test_convert.py
import torch
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

# ---------- 公共工具 ----------
def dummy_state_dict_qwen(num_layers: int, hidden: int = 4):
    """构造一个最小化的 Qwen state_dict，仅包含 KV/O/W1/W2 ... 层权重。"""
    sd = {
        "model.norm.weight": torch.ones(hidden),
        "model.embed_tokens.weight": torch.zeros(10, hidden),
        "lm_head.weight": torch.zeros(hidden, 10),
    }
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        sd.update({
            f"{prefix}.self_attn.q_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.k_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.v_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.o_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.mlp.gate_proj.weight": torch.randn(2 * hidden, hidden),
            f"{prefix}.mlp.up_proj.weight":   torch.randn(2 * hidden, hidden),
            f"{prefix}.mlp.down_proj.weight": torch.randn(hidden, 2 * hidden),
            f"{prefix}.input_layernorm.weight": torch.ones(hidden),
            f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden),
        })
    return sd


def dummy_state_dict_llama(num_layers: int, hidden: int = 4):
    """构造最小化 Llama HF 权重，方便测试不进行 KV 合并。"""
    sd = {
        "model.norm.weight": torch.ones(hidden),
        "model.embed_tokens.weight": torch.zeros(10, hidden),
        "lm_head.weight": torch.zeros(hidden, 10),
    }
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        sd.update({
            f"{prefix}.self_attn.q_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.k_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.v_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.self_attn.o_proj.weight": torch.randn(hidden, hidden),
            f"{prefix}.mlp.gate_proj.weight": torch.randn(2 * hidden, hidden),
            f"{prefix}.mlp.up_proj.weight":   torch.randn(2 * hidden, hidden),
            f"{prefix}.mlp.down_proj.weight": torch.randn(hidden, 2 * hidden),
            f"{prefix}.input_layernorm.weight": torch.ones(hidden),
            f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden),
        })
    return sd


# ---------- fixtures ----------
@pytest.fixture(scope="function")
def tmp_ckpt_dir(tmp_path: Path):
    """创建临时 checkpoints 目录并写入最小 config.json。"""
    def _factory(model_type: str):
        ckpt = tmp_path / model_type
        ckpt.mkdir()
        (ckpt / "config.json").write_text(json.dumps({"model_type": model_type}))
        return ckpt
    return _factory


# ---------- 测试映射 ----------
@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
def test_mapping_and_num_params(tmp_ckpt_dir, model_type):
    ckpt_dir = tmp_ckpt_dir(model_type)
    num_layers = 2

    # 构造假权重
    sd = (dummy_state_dict_qwen(num_layers) if model_type.startswith("qwen")
          else dummy_state_dict_llama(num_layers))

    # 跑转换
    new_sd = convert(ckpt_dir, sd, model_type, num_layers)

    # 基础 key 应当存在
    assert "embed_tokens.weight" in new_sd
    assert "norm_weight" in new_sd


# ---------- Qwen KV 合并 ----------
def test_qwen_kv_merge(tmp_ckpt_dir):
    model_type = "qwen3"
    num_layers = 1
    ckpt_dir = tmp_ckpt_dir(model_type)
    sd = dummy_state_dict_qwen(num_layers)

    new_sd = convert(ckpt_dir, sd, model_type, num_layers)

    # KV fused weight 应出现
    kv_key = "layers.0.self_attn.kv_proj.weight"
    assert kv_key in new_sd

    # 原 K、V 不应保留
    assert "layers.0.self_attn.k_proj_weight" not in new_sd
    assert "layers.0.self_attn.v_proj_weight" not in new_sd

    # 维度检查：concat 后第一维应为 2*hidden
    hidden = sd["model.norm.weight"].numel()     # 4
    assert new_sd[kv_key].shape[0] == 2 * hidden


# ---------- Llama 不合并 ----------
def test_llama_no_kv_merge(tmp_ckpt_dir):
    model_type = "llama"
    num_layers = 1
    ckpt_dir = tmp_ckpt_dir(model_type)
    sd = dummy_state_dict_llama(num_layers)

    new_sd = convert(ckpt_dir, sd, model_type, num_layers)

    # KV fused 不存在
    assert "layers.0.self_attn.kv_proj.weight" not in new_sd
    # 原 K/V 仍然存在
    assert "layers.0.self_attn.k_proj.weight" in new_sd
    assert "layers.0.self_attn.v_proj.weight" in new_sd

if __name__ == "__main__":
    main()
