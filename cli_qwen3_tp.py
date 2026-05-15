"""Qwen3 / Qwen3-VL Tensor Parallelism CLI.

Usage:
  torchrun --nproc_per_node=2 cli_qwen3_tp.py --checkpoints_dir /path/to/model

  Image + TP:
  ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node=2 \
      cli_qwen3_tp.py --checkpoints_dir /path/to/model --image /path/to/img.jpg
"""

import argparse
import traceback
import time as time_module
from typing import Optional

import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

from lite_llama.executor.tp_utils import detect_tp_env
from lite_llama.utils.device import get_device


def main_worker(rank: int, world_size: int, args):
    """All ranks run this jointly. Rank 0 handles I/O, all ranks compute."""
    tp_config = detect_tp_env()

    # Determine model type from checkpoint
    import json
    from pathlib import Path
    cfg_path = Path(args.checkpoints_dir) / "config.json"
    with open(cfg_path) as f:
        config = json.load(f)
    is_vl = config.get("model_type", "").lower() in ("qwen3_vl", "llava")

    # Load the right generator
    if is_vl:
        from lite_llama.qwen3vl_generate_stream import Qwen3VLGeneratorStream
        from PIL import Image
        generator = Qwen3VLGeneratorStream(
            checkpoints_dir=args.checkpoints_dir,
            tokenizer_path=args.checkpoints_dir,
            max_gpu_num_blocks=args.max_gpu_num_blocks,
            max_seq_len=args.max_seq_len,
            compiled_model=False,
            device=f"npu:{rank}" if tp_config else get_device(),
        )
    else:
        from lite_llama.generate_stream import GenerateStreamText
        generator = GenerateStreamText(
            checkpoints_dir=args.checkpoints_dir,
            tokenizer_path=args.checkpoints_dir,
            max_gpu_num_blocks=args.max_gpu_num_blocks,
            max_seq_len=args.max_seq_len,
            compiled_model=False,
            device=f"npu:{rank}" if tp_config else get_device(),
        )

    # Load image if provided
    image_items = []
    if is_vl and args.image:
        img = Image.open(args.image).convert("RGB")
        image_items = [img]
        if rank == 0:
            print(f"Image loaded: {args.image}")

    # Set same RNG seed on all ranks for deterministic sampling
    import torch
    torch.manual_seed(42)

    # Only rank 0 handles I/O
    while True:
        if rank == 0:
            sys.stdout.write("\nEnter prompt (or 'exit' to quit):\n")
            sys.stdout.flush()
            try:
                prompt = input().strip()
            except EOFError:
                break
        else:
            prompt = ""

        # Broadcast prompt from rank 0
        if tp_config:
            prompt = _broadcast_string(prompt, src=0)

        if prompt.lower() == "exit":
            break

        if rank == 0:
            sys.stdout.write("\nASSISTANT: ")
            sys.stdout.flush()

        # Both ranks execute generation — tp_all_reduce in model syncs them
        try:
            if is_vl:
                stream = generator.text_completion_stream(
                    [prompt], image_items,
                    temperature=args.temperature, top_p=args.top_p,
                    max_gen_len=args.max_gen_len,
                )
            else:
                stream = generator.text_completion_stream(
                    [prompt],
                    temperature=args.temperature, top_p=args.top_p,
                    max_gen_len=args.max_gen_len,
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
                sys.stdout.write(new_text)
                sys.stdout.flush()

        if rank == 0:
            print()


def _broadcast_string(s: str, src: int = 0, max_len: int = 4096) -> str:
    """Broadcast a string from src rank to all ranks."""
    import torch
    if s is None:
        s = ""
    # Pad/truncate to fixed length, broadcast as int32 tensor
    encoded = s.encode("utf-8")[:max_len]
    data = torch.zeros(max_len, dtype=torch.int32, device="cpu")
    for i, b in enumerate(encoded):
        data[i] = b
    torch.distributed.broadcast(data, src=src)
    decoded = bytes(data[data != 0].tolist()).decode("utf-8", errors="replace")
    return decoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiteLlama Qwen3/VL TP CLI")
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image", type=str, default=None,
                        help="Image path (for VL models)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--max_gpu_num_blocks", type=int, default=None)
    args = parser.parse_args()

    # Detect TP environment
    tp = detect_tp_env()
    rank = tp.rank if tp else 0
    world_size = tp.world_size if tp else 1

    if rank == 0:
        print(f"TP config: world_size={world_size}, rank={rank}")
        print(f"Model: {args.checkpoints_dir}")

    main_worker(rank, world_size, args)
