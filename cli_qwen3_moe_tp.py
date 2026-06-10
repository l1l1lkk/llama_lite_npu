"""Qwen3 MoE tensor-parallel interactive CLI.

Usage:
  ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
      --nproc_per_node=2 cli_qwen3_moe_tp.py \
      --checkpoints_dir /path/to/Qwen3-30B-A3B/ --page_size 16
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from typing import Optional

import torch
from rich.console import Console

from lite_llama.executor.tp_utils import detect_tp_env
from lite_llama.generate_stream import GenerateStreamText
from lite_llama.utils.prompt_templates import get_prompter


warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")


def _broadcast_string(value: str, src: int = 0) -> str:
    objects = [value]
    torch.distributed.broadcast_object_list(objects, src=src)
    return objects[0]


def main(
    *,
    checkpoints_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_gpu_num_blocks: Optional[int] = None,
    max_gen_len: int = 1024,
    page_size: int = 16,
    compiled_model: bool = True,
    enable_thinking: bool = True,
) -> None:
    tp = detect_tp_env()
    rank = tp.rank if tp else 0
    device = (
        f"npu:{rank}"
        if tp
        else (
            "npu:0"
            if hasattr(torch, "npu") and torch.npu.is_available()
            else "cpu"
        )
    )

    try:
        generator = GenerateStreamText(
            checkpoints_dir=checkpoints_dir,
            tokenizer_path=checkpoints_dir,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            compiled_model=compiled_model,
            page_size=page_size,
            device=device,
        )
    except Exception as exc:
        if rank == 0:
            print(f"Model loading failed: {exc}")
            traceback.print_exc()
        raise SystemExit(1) from exc

    torch.manual_seed(42)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed(42)
        if hasattr(torch.npu, "manual_seed_all"):
            torch.npu.manual_seed_all(42)

    prompter = get_prompter(
        "qwen3",
        checkpoints_dir,
        short_prompt=False,
        enable_thinking=enable_thinking,
    )
    console = Console() if rank == 0 else None

    if rank == 0:
        print(f"Model:       {checkpoints_dir}")
        print(f"TP size:     {tp.world_size if tp else 1}")
        print(f"Max seq len: {max_seq_len}")
        print(f"Page size:   {page_size}")
        print(
            "NPU Graph:   requested (MoE uses eager decode)"
            if compiled_model
            else "NPU Graph:   off"
        )
        print(f"Thinking:    {'on' if enable_thinking else 'off'}")

    while True:
        prompt = (
            input("\nEnter prompt (or 'exit' to quit):\n").strip()
            if rank == 0
            else ""
        )
        if tp:
            prompt = _broadcast_string(prompt)
        if prompt.lower() == "exit":
            break

        prompter.insert_prompt(prompt)
        if rank == 0:
            print("\nASSISTANT: ", end="", flush=True)

        completion = ""
        try:
            stream = generator.text_completion_stream(
                [prompter.model_input],
                temperature=temperature,
                top_p=top_p,
                max_gen_len=max_gen_len,
            )
            for batch_completions in stream:
                if rank == 0:
                    current = batch_completions[0]["generation"]
                    console.print(
                        f"[red]{current[len(completion):]}[/red]", end=""
                    )
                    completion = current
        except Exception as exc:
            if rank == 0:
                print(f"\nGeneration failed: {exc}")
                traceback.print_exc()
            continue

        if rank == 0:
            print("\n\n==================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LiteLlama Qwen3-30B-A3B TP CLI"
    )
    parser.add_argument("--checkpoints_dir", required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--max_gen_len", type=int, default=1024)
    parser.add_argument("--max_gpu_num_blocks", type=int, default=None)
    parser.add_argument(
        "--page_size",
        type=int,
        default=16,
        help="PagedAttention page size; use 0 to disable.",
    )
    parser.add_argument(
        "--compiled_model",
        dest="compiled_model",
        action="store_true",
        help="Request NPU Graph (MoE falls back to eager decode).",
    )
    parser.add_argument(
        "--no_compiled_model",
        dest="compiled_model",
        action="store_false",
        help="Disable NPU Graph.",
    )
    parser.add_argument(
        "--enable_thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode.",
    )
    parser.add_argument(
        "--disable_thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable Qwen3 thinking mode.",
    )
    parser.set_defaults(compiled_model=True, enable_thinking=True)
    args = parser.parse_args()
    main(**vars(args))
