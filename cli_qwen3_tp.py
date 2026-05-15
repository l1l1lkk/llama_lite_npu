"""Qwen3 Tensor Parallelism CLI entry point.

Usage (torchrun):
  torchrun --nproc_per_node=2 cli_qwen3_tp.py --checkpoints_dir /path/to/model

Usage (NPU):
  torchrun --nproc_per_node=2 cli_qwen3_tp.py --device npu --checkpoints_dir /path/to/model
"""

import argparse
import traceback
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt

import sys, os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

from lite_llama.executor.tp_utils import detect_tp_env
from lite_llama.generate_stream import GenerateStreamText
from lite_llama.utils.device import get_device


def main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gpu_num_blocks=None,
    max_gen_len: Optional[int] = 1024,
    compiled_model: bool = False,
    device: str = None,
    checkpoints_dir: str = None,
):
    # TP init: auto-detect from torchrun env
    tp_config = detect_tp_env()
    if tp_config is None:
        device = get_device(device)
    else:
        device = f"npu:{tp_config.rank}" if tp_config.is_npu else f"cuda:{tp_config.rank}"

    console = Console()
    console.print(f"[green]TP config: world_size={tp_config.world_size if tp_config else 1}, "
                  f"rank={tp_config.rank if tp_config else 0}, device={device}[/green]")

    if checkpoints_dir is None:
        console.print("[red]Please specify --checkpoints_dir[/red]")
        sys.exit(1)

    try:
        generator = GenerateStreamText(
            checkpoints_dir=checkpoints_dir,
            tokenizer_path=checkpoints_dir,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            compiled_model=compiled_model,
            device=device,
        )
    except Exception as e:
        console.print(f"[red]Model load failed: {e}[/red]")
        traceback.print_exc()
        sys.exit(1)

    # Only rank 0 handles interactive I/O
    if tp_config is not None and tp_config.rank != 0:
        console.print("[yellow]Rank %d waiting...[/yellow]" % tp_config.rank)
        # Non-rank-0 just stays alive
        import time
        while True:
            time.sleep(3600)

    while True:
        prompt = input("Enter prompt (or 'exit' to quit):\n").strip()
        if prompt.lower() == "exit":
            break

        print("\nASSISTANT: ", end="", flush=True)
        prompts = [prompt]

        stream = generator.text_completion_stream(
            prompts,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
        )

        completion = ""
        for batch_completions in stream:
            new_text = batch_completions[0]["generation"][len(completion):]
            completion = batch_completions[0]["generation"]
            print(new_text, end="", flush=True)
        print("\n\n==================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiteLlama Qwen3 TP CLI")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoints_dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_gen_len", type=int, default=1024)
    parser.add_argument("--max_gpu_num_blocks", type=int, default=None)
    args = parser.parse_args()
    main(
        temperature=args.temperature, top_p=args.top_p,
        max_seq_len=args.max_seq_len, max_gen_len=args.max_gen_len,
        max_gpu_num_blocks=args.max_gpu_num_blocks,
        device=args.device, checkpoints_dir=args.checkpoints_dir,
    )
