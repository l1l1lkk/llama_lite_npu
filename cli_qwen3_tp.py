"""Qwen3 Tensor Parallelism CLI — text-only interactive chat.

Usage:
  ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node=2 \
      cli_qwen3_tp.py --checkpoints_dir /path/to/Qwen3-32B/
"""

import torch
import argparse
import traceback
from typing import Optional

from rich.console import Console

import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

from lite_llama.generate_stream import GenerateStreamText
from lite_llama.executor.tp_utils import detect_tp_env


def _broadcast_string(s: str, src: int = 0, max_len: int = 4096) -> str:
    if s is None:
        s = ""
    encoded = s.encode("utf-8")[:max_len]
    dev = f"npu:{src}"
    data = torch.zeros(max_len, dtype=torch.int32, device=dev)
    for i, b in enumerate(encoded):
        data[i] = b
    torch.distributed.broadcast(data, src=src)
    decoded = bytes(data[data != 0].cpu().tolist()).decode("utf-8", errors="replace")
    return decoded


def main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gpu_num_blocks=None,
    max_gen_len: Optional[int] = 1024,
    compiled_model: bool = False,
    checkpoints_dir: str = None,
):
    tp = detect_tp_env()
    rank = tp.rank if tp else 0

    if checkpoints_dir is None:
        if rank == 0:
            print("[red]请指定 --checkpoints_dir 路径[/red]")
        sys.exit(1)

    try:
        generator = GenerateStreamText(
            checkpoints_dir=checkpoints_dir,
            tokenizer_path=checkpoints_dir,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            compiled_model=compiled_model,
            device=f"npu:{rank}" if tp else "cpu",
        )
    except Exception as e:
        if rank == 0:
            print(f"[red]模型加载失败: {e}[/red]")
            traceback.print_exc()
        sys.exit(1)

    torch.manual_seed(42)
    console = Console() if rank == 0 else None

    while True:
        if rank == 0:
            prompt = input("\nEnter prompt (or 'exit' to quit):\n").strip()
        else:
            prompt = ""

        if tp:
            prompt = _broadcast_string(prompt, src=0)

        if prompt.lower() == "exit":
            break

        if rank == 0:
            print("\nASSISTANT: ", end="", flush=True)

        try:
            stream = generator.text_completion_stream(
                [prompt],
                temperature=temperature, top_p=top_p, max_gen_len=max_gen_len,
            )
        except Exception as e:
            if rank == 0:
                print(f"\nGeneration failed: {e}")
                traceback.print_exc()
            continue

        completion = ""
        for batch_completions in stream:
            if rank == 0:
                new_text = batch_completions[0]["generation"][len(completion):]
                completion = batch_completions[0]["generation"]
                console.print(f"[red]{new_text}[/red]", end="")

        if rank == 0:
            print("\n\n==================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiteLlama Qwen3 TP CLI")
    parser.add_argument("--checkpoints_dir", type=str, default=None,
                        help="Path to converted weights")
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
        checkpoints_dir=args.checkpoints_dir,
    )
