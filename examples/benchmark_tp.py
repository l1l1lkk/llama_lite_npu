"""Tensor Parallelism benchmark for Qwen3 / Qwen3-VL.

Measures throughput (tokens/s), prefill time, decode per-token latency,
and GPU memory usage under TP.

Usage (text model):
  ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node=2 \
      --master_addr=127.0.0.1 --master_port=29500 \
      examples/benchmark_tp.py \
      --checkpoints_dir /path/to/Qwen3-32B/ \
      --prompt_len 128 --batch_size 4 --max_gen_len 256 \
      --warmup 2 --iterations 10

Usage (VL model):
  ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node=2 \
      --master_addr=127.0.0.1 --master_port=29500 \
      examples/benchmark_tp.py --vl \
      --checkpoints_dir /path/to/Qwen3-VL-32B/ \
      --batch_size 1 --max_gen_len 128
"""

import argparse
import time
import sys
import os
from typing import Optional

import torch
import torch_npu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lite_llama.executor.tp_utils import detect_tp_env
from lite_llama.generate_stream import GenerateStreamText
from lite_llama.utils.prompt_templates import get_prompter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _broadcast_string(s: str, src: int = 0) -> str:
    objects = [s]
    torch.distributed.broadcast_object_list(objects, src=src)
    return objects[0]


def get_gpu_memory_info(device: str) -> dict:
    """Return used / total GPU memory in GB."""
    free, total = torch.npu.mem_get_info()
    used = total - free
    return {
        "used_gb": used / (1024**3),
        "total_gb": total / (1024**3),
        "free_gb": free / (1024**3),
    }


# ---------------------------------------------------------------------------
# Dummy image for VL benchmark
# ---------------------------------------------------------------------------
def make_dummy_image():
    from PIL import Image
    import io, base64
    # Minimal valid PNG: 1x1 white pixel
    img = Image.new("RGB", (336, 336), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Image.open(buf)


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------
def run_benchmark(
    generator,
    prompts: list[str],
    max_gen_len: int,
    temperature: float,
    top_p: float,
    is_vl: bool = False,
    dummy_image=None,
) -> dict:
    """Run generation and return timing stats."""
    # Warmup
    warmup_prompts = ["Hello"] * len(prompts)
    if is_vl:
        _ = generator.text_completion_stream(
            warmup_prompts, [dummy_image] * len(prompts),
            temperature=temperature, top_p=top_p, max_gen_len=5,
        )
    else:
        _ = generator.text_completion_stream(
            warmup_prompts, temperature=temperature, top_p=top_p, max_gen_len=5,
        )

    # --- Timed run ---
    torch.npu.synchronize()
    t_start = time.time()

    if is_vl:
        stream = generator.text_completion_stream(
            prompts, [dummy_image] * len(prompts),
            temperature=temperature, top_p=top_p, max_gen_len=max_gen_len,
        )
    else:
        stream = generator.text_completion_stream(
            prompts, temperature=temperature, top_p=top_p, max_gen_len=max_gen_len,
        )

    total_tokens = 0
    completions = []
    for batch_completions in stream:
        completions = batch_completions
        total_tokens += 1

    torch.npu.synchronize()
    t_end = time.time()

    elapsed = t_end - t_start
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    per_token_ms = (elapsed / total_tokens * 1000) if total_tokens > 0 else 0

    return {
        "tokens_generated": total_tokens,
        "total_time_s": elapsed,
        "throughput_tok_s": throughput,
        "per_token_ms": per_token_ms,
        "completions": completions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tp = detect_tp_env()
    rank = tp.rank if tp else 0
    world_size = tp.world_size if tp else 1

    parser = argparse.ArgumentParser(description="TP Benchmark")
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--vl", action="store_true", help="Vision-Language model")
    parser.add_argument("--prompt_len", type=int, default=128,
                        help="Prompt length in tokens (approximate)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of benchmark iterations")
    parser.add_argument(
        "--enable_thinking", dest="enable_thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode (default).",
    )
    parser.add_argument(
        "--disable_thinking", dest="enable_thinking",
        action="store_false",
        help="Disable Qwen3 thinking mode.",
    )
    parser.set_defaults(enable_thinking=True)
    args = parser.parse_args()

    # Print header (rank 0 only)
    if rank == 0:
        print("=" * 70)
        print(f"  TP Benchmark — world_size={world_size}")
        print(f"  Model:       {args.checkpoints_dir}")
        print(f"  Type:        {'Vision-Language' if args.vl else 'Text-only'}")
        print(f"  Prompt len:  ~{args.prompt_len} tokens")
        print(f"  Batch size:  {args.batch_size}")
        print(f"  Max gen len: {args.max_gen_len}")
        print(f"  Warmup:      {args.warmup}  |  Iterations: {args.iterations}")
        print("=" * 70)

    # --- Memory before loading ---
    mem_before = get_gpu_memory_info(f"npu:{rank}")

    # --- Load generator ---
    if args.vl:
        from lite_llama.qwen3vl_generate_stream import Qwen3VLGeneratorStream
        generator = Qwen3VLGeneratorStream(
            checkpoints_dir=args.checkpoints_dir,
            tokenizer_path=args.checkpoints_dir,
            max_seq_len=args.prompt_len + args.max_gen_len + 1024,
            device=f"npu:{rank}",
        )
        dummy_image = make_dummy_image()
    else:
        from lite_llama.generate_stream import GenerateStreamText
        generator = GenerateStreamText(
            checkpoints_dir=args.checkpoints_dir,
            tokenizer_path=args.checkpoints_dir,
            max_seq_len=args.prompt_len + args.max_gen_len + 1024,
            device=f"npu:{rank}",
        )
        dummy_image = None

    # --- Memory after loading ---
    torch.npu.synchronize()
    mem_after = get_gpu_memory_info(f"npu:{rank}")
    if rank == 0:
        print(f"\n  GPU memory — before load: {mem_before['used_gb']:.1f} GB")
        print(f"  GPU memory — after  load: {mem_after['used_gb']:.1f} GB")
        print(f"  Model weight approx:      {mem_after['used_gb'] - mem_before['used_gb']:.1f} GB")
        print()

    # --- Generate prompt ---
    # Use a fixed prompt to avoid variance
    prompt_text = (
        "Artificial intelligence has transformed many industries. "
        + "explain how it works in detail " * (args.prompt_len // 8)
    )
    if args.vl:
        prompts = [f"<image> Describe this image in detail. {prompt_text[:args.prompt_len]}"
                   for _ in range(args.batch_size)]
    else:
        # Wrap with Qwen3 ChatML template
        prompter = get_prompter(
            "qwen3", args.checkpoints_dir, short_prompt=False,
            enable_thinking=args.enable_thinking,
        )
        prompter.insert_prompt(prompt_text[:args.prompt_len])
        prompts = [prompter.model_input for _ in range(args.batch_size)]

    torch.manual_seed(42)

    # --- Warmup iterations ---
    if rank == 0:
        print(f"  Running {args.warmup} warmup iterations...")
    for i in range(args.warmup):
        _ = run_benchmark(
            generator, prompts, args.max_gen_len,
            args.temperature, args.top_p, args.vl, dummy_image,
        )
        if rank == 0:
            print(f"    Warmup {i+1}/{args.warmup} done")

    # --- Benchmark iterations ---
    if rank == 0:
        print(f"\n  Running {args.iterations} benchmark iterations...\n")

    results = []
    for i in range(args.iterations):
        torch.npu.synchronize()
        r = run_benchmark(
            generator, prompts, args.max_gen_len,
            args.temperature, args.top_p, args.vl, dummy_image,
        )
        results.append(r)
        if rank == 0:
            print(f"    Iter {i+1:2d}: {r['tokens_generated']:4d} tokens, "
                  f"{r['total_time_s']:.3f}s, "
                  f"{r['throughput_tok_s']:.1f} tok/s, "
                  f"{r['per_token_ms']:.2f} ms/tok")

    # --- Summary (rank 0) ---
    if rank != 0:
        return

    throughputs = [r["throughput_tok_s"] for r in results]
    latencies = [r["per_token_ms"] for r in results]
    times = [r["total_time_s"] for r in results]
    tokens_list = [r["tokens_generated"] for r in results]

    avg_throughput = sum(throughputs) / len(throughputs)
    avg_latency = sum(latencies) / len(latencies)
    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_list) / len(tokens_list)

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  World size:          {world_size}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Total GPU memory:    {mem_after['total_gb']:.1f} GB per card")
    print(f"  Model + KV cache:    {mem_after['used_gb'] - mem_before['used_gb']:.1f} GB")
    print(f"  Avg tokens/gen:      {avg_tokens:.0f}")
    print(f"  Avg time:            {avg_time:.3f} s")
    print(f"  Avg throughput:      {avg_throughput:.1f} tokens/s")
    print(f"  Avg per-token:       {avg_latency:.2f} ms")
    print(f"  Batch throughput:    {avg_throughput * args.batch_size:.1f} tokens/s")
    print("=" * 70)

    # Show sample output
    if results and results[0]["completions"]:
        sample = results[0]["completions"][0].get("generation", "")[:200]
        print(f"\n  Sample output (first 200 chars):\n  {sample}\n")


if __name__ == "__main__":
    main()
