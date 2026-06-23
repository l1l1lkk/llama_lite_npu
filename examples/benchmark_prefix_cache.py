#!/usr/bin/env python3
"""Benchmark OpenAI server Prefix Cache behavior.

This script intentionally targets ``server.py`` through the OpenAI-compatible
HTTP API. The rc4 Prefix Cache lives in the continuous-batching server backend,
so direct model benchmarks such as ``benchmark_tp.py`` do not exercise it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, Iterator

import requests


DEFAULT_PROMPT = (
    "Explain how artificial intelligence works in practical terms. "
    "Focus on model inference, token generation, and why cached prefixes can "
    "reduce time to first token."
)


@dataclass
class RequestResult:
    index: int
    success: bool
    latency_s: float
    ttft_s: float | None
    output_chars: int
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    error: str | None = None

    @property
    def output_tok_s(self) -> float | None:
        if not self.output_tokens or self.latency_s <= 0:
            return None
        return self.output_tokens / self.latency_s


def percentile(values: Iterable[float], pct: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return float("nan")
    rank = math.ceil((pct / 100.0) * len(sorted_values))
    rank = min(max(rank, 1), len(sorted_values))
    return sorted_values[rank - 1]


def iter_sse_payloads(chunks: Iterable[bytes]) -> Iterator[dict]:
    """Yield JSON payloads from OpenAI-style SSE byte chunks."""
    buffer = ""
    for raw in chunks:
        if not raw:
            continue
        buffer += raw.decode("utf-8")
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            for line in event.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if not payload or payload == "[DONE]":
                    continue
                yield json.loads(payload)


def build_prompt(
    *,
    dataset: str,
    index: int,
    prompt_len: int,
    same_prompt: str,
) -> str:
    if dataset == "same":
        return same_prompt
    base = (
        f"Request {index}: explain AI inference performance, KV cache, "
        f"scheduler behavior, and prefix cache measurement. "
    )
    words = []
    cursor = 0
    while len(words) < prompt_len:
        words.extend((base + f"unique-token-{index}-{cursor} ").split())
        cursor += 1
    return " ".join(words[:prompt_len])


def make_payload(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    stream: bool,
    include_usage: bool,
    enable_thinking: bool,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "enable_thinking": enable_thinking,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if stream and include_usage:
        payload["stream_options"] = {"include_usage": True}
    return payload


def run_one_request(
    *,
    index: int,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    stream: bool,
    timeout: float,
    api_key: str | None,
    enable_thinking: bool,
) -> RequestResult:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = make_payload(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
        include_usage=True,
        enable_thinking=enable_thinking,
    )
    start = time.perf_counter()
    first_token_at: float | None = None
    output_text = ""
    prompt_tokens = None
    output_tokens = None
    total_tokens = None
    try:
        if stream:
            with requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                for data in iter_sse_payloads(response.iter_content(chunk_size=None)):
                    if "error" in data:
                        raise RuntimeError(data["error"])
                    usage = data.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens")
                        output_tokens = usage.get("completion_tokens")
                        total_tokens = usage.get("total_tokens")
                    choices = data.get("choices") or []
                    for choice in choices:
                        delta = (choice.get("delta") or {}).get("content", "")
                        if delta:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            output_text += delta
        else:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            output_text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            first_token_at = None
        end = time.perf_counter()
        return RequestResult(
            index=index,
            success=True,
            latency_s=end - start,
            ttft_s=(first_token_at - start) if first_token_at is not None else None,
            output_chars=len(output_text),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
    except Exception as exc:
        end = time.perf_counter()
        return RequestResult(
            index=index,
            success=False,
            latency_s=end - start,
            ttft_s=None,
            output_chars=0,
            prompt_tokens=None,
            output_tokens=None,
            total_tokens=None,
            error=str(exc),
        )


def summarize(results: list[RequestResult]) -> str:
    successes = [result for result in results if result.success]
    failures = [result for result in results if not result.success]
    latencies = [result.latency_s for result in successes]
    ttfts = [result.ttft_s for result in successes if result.ttft_s is not None]
    output_tokens = [
        result.output_tokens for result in successes if result.output_tokens is not None
    ]
    total_tokens = [
        result.total_tokens for result in successes if result.total_tokens is not None
    ]
    lines = []
    lines.append("=" * 70)
    lines.append("Prefix Cache Benchmark Summary")
    lines.append("=" * 70)
    lines.append(f"Total / Success / Failed: {len(results)} / {len(successes)} / {len(failures)}")
    if successes:
        total_time = sum(latencies)
        wall_time = max(result.latency_s for result in successes)
        lines.append(f"Avg latency:             {statistics.mean(latencies):.4f} s")
        lines.append(f"P50 / P90 / P99 latency: {percentile(latencies, 50):.4f} / {percentile(latencies, 90):.4f} / {percentile(latencies, 99):.4f} s")
        if ttfts:
            lines.append(f"Avg TTFT:                {statistics.mean(ttfts):.4f} s")
            lines.append(f"P50 / P90 / P99 TTFT:    {percentile(ttfts, 50):.4f} / {percentile(ttfts, 90):.4f} / {percentile(ttfts, 99):.4f} s")
        if output_tokens:
            lines.append(f"Avg output tokens:       {statistics.mean(output_tokens):.1f}")
            lines.append(f"Output throughput:       {sum(output_tokens) / total_time:.2f} tok/s/request-time")
            lines.append(f"Wall output throughput:  {sum(output_tokens) / wall_time:.2f} tok/s/wall")
        if total_tokens:
            lines.append(f"Avg total tokens:        {statistics.mean(total_tokens):.1f}")
    if failures:
        lines.append("-" * 70)
        lines.append("Failures:")
        for result in failures[:5]:
            lines.append(f"  #{result.index}: {result.error}")
    lines.append("=" * 70)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark server Prefix Cache with same/random prompts."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8213/v1/chat/completions",
        help="OpenAI-compatible chat completions URL.",
    )
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--dataset", choices=("same", "random"), default="same")
    parser.add_argument("--number", type=int, default=20)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=21600)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--same-prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.set_defaults(stream=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = [
        build_prompt(
            dataset=args.dataset,
            index=index,
            prompt_len=args.prompt_len,
            same_prompt=args.same_prompt,
        )
        for index in range(args.number)
    ]

    print("=" * 70)
    print("Prefix Cache Benchmark")
    print("=" * 70)
    print(f"URL:         {args.url}")
    print(f"Model:       {args.model}")
    print(f"Dataset:     {args.dataset}")
    print(f"Requests:    {args.number}")
    print(f"Parallel:    {args.parallel}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p:       {args.top_p if args.top_p is not None else 'inactive/default'}")
    print(f"Stream:      {args.stream}")
    print(f"Thinking:    {'on' if args.enable_thinking else 'off'}")
    if args.dataset == "same" and args.temperature != 0:
        print("WARNING: Prefix Cache is disabled for temperature > 0.")
    print("=" * 70)

    results: list[RequestResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [
            pool.submit(
                run_one_request,
                index=index,
                url=args.url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=args.stream,
                timeout=args.timeout,
                api_key=args.api_key,
                enable_thinking=args.enable_thinking,
            )
            for index, prompt in enumerate(prompts)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            status = "ok" if result.success else "failed"
            ttft = f"{result.ttft_s:.3f}s" if result.ttft_s is not None else "n/a"
            out = result.output_tokens if result.output_tokens is not None else "n/a"
            print(
                f"Request {result.index + 1:>4}/{args.number}: "
                f"{status}, latency={result.latency_s:.3f}s, TTFT={ttft}, "
                f"out_tokens={out}"
            )
    results.sort(key=lambda item: item.index)
    print()
    print(summarize(results))


if __name__ == "__main__":
    main()
