"""Qwen3-VL Tensor Parallelism CLI — multimodal interactive chat.

Usage:
  ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node=2 \
      cli_qwen3vl_tp.py --checkpoints_dir /path/to/Qwen3-VL-32B-Instruct/
"""

import torch
import argparse
import traceback
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt

import sys, os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

from lite_llama.qwen3vl_generate_stream import Qwen3VLGeneratorStream
from lite_llama.utils.image_process import vis_images
from lite_llama.executor.tp_utils import detect_tp_env


def _broadcast_string(s: str, src: int = 0) -> str:
    """Broadcast a string from src to all ranks via object list."""
    objects = [s]
    torch.distributed.broadcast_object_list(objects, src=src)
    return objects[0]


def main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gpu_num_blocks=None,
    max_gen_len: Optional[int] = 512,
    compiled_model: bool = False,
    device: str = None,
    checkpoints_dir: str = None,
):
    tp = detect_tp_env()
    rank = tp.rank if tp else 0

    if checkpoints_dir is None:
        if rank == 0:
            print("[red]请指定 --checkpoints_dir 路径[/red]")
        sys.exit(1)

    try:
        generator = Qwen3VLGeneratorStream(
            checkpoints_dir=checkpoints_dir,
            tokenizer_path=checkpoints_dir,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            compiled_model=compiled_model,
            device=f"npu:{rank}" if tp else (device or "cpu"),
        )
    except Exception as e:
        if rank == 0:
            print(f"[red]模型加载失败: {e}[/red]")
            traceback.print_exc()
        sys.exit(1)

    # Same seed on all ranks for deterministic sampling
    torch.manual_seed(42)

    console = Console() if rank == 0 else None

    while True:
        # ---- Image input (rank 0 only) ----
        if rank == 0:
            console.print(
                "[bold green]Enter image path or URL (type 'exit' to quit):[/bold green]"
            )
            while True:
                image_input = Prompt.ask("Image")
                if os.path.isfile(image_input):
                    break
                elif image_input.strip().lower() == "exit":
                    break
                else:
                    console.print(f"[red]'{image_input}' is not a valid file path![/red]")

            image_input = image_input.strip()
        else:
            image_input = ""

        if tp:
            image_input = _broadcast_string(image_input, src=0)

        if image_input.lower() == "exit":
            break

        image_items = [image_input]

        if rank == 0:
            vis_images(image_items)

        # ---- Text prompt (rank 0 only) ----
        if rank == 0:
            input_prompt = Prompt.ask("[bold green]Prompt[/bold green]").strip()
        else:
            input_prompt = ""

        if tp:
            input_prompt = _broadcast_string(input_prompt, src=0)

        if input_prompt.lower() == "exit":
            break

        # ---- Generation (both ranks) ----
        try:
            stream = generator.text_completion_stream(
                [input_prompt], image_items,
                temperature=temperature, top_p=top_p, max_gen_len=max_gen_len,
            )
        except Exception as e:
            if rank == 0:
                console.print(f"[red]Generation failed: {e}[/red]")
                traceback.print_exc()
            continue

        if rank == 0:
            console.print("ASSISTANT: ", end="")

        completion = ""
        for batch_completions in stream:
            if rank == 0:
                next_text = batch_completions[0]["generation"][len(completion):]
                completion = batch_completions[0]["generation"]
                console.print(f"[red]{next_text}[/red]", end="")

        if rank == 0:
            console.print("\n[bold green]==================================[/bold green]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiteLlama Qwen3-VL TP CLI")
    parser.add_argument("--device", type=str, default=None,
                        help="Device. Auto-detect if not set.")
    parser.add_argument("--checkpoints_dir", type=str, default=None,
                        help="Path to converted weights")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--max_gpu_num_blocks", type=int, default=None)
    args = parser.parse_args()
    main(
        temperature=args.temperature, top_p=args.top_p,
        max_seq_len=args.max_seq_len, max_gen_len=args.max_gen_len,
        max_gpu_num_blocks=args.max_gpu_num_blocks,
        checkpoints_dir=args.checkpoints_dir,
    )
