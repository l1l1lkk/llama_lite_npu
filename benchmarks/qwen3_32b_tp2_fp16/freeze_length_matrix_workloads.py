#!/usr/bin/env python3
"""Freeze deterministic exact-token workloads for the length matrix campaign."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from transformers import AutoTokenizer


PROMPTS = (128, 512, 1024, 2048)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[list[dict]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_prompt_module(repo_root: Path):
    path = repo_root / "lite_llama" / "utils" / "prompt_templates.py"
    spec = importlib.util.spec_from_file_location("benchmark_prompt_templates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def token_count(tokenizer, prompt_module, messages: list[dict]) -> int:
    prompt = next(
        str(message.get("content", ""))
        for message in reversed(messages)
        if message.get("role") == "user"
    )
    prompter = prompt_module.get_prompter("qwen3", "", enable_thinking=True)
    prompter.insert_prompt(prompt)
    return len(tokenizer.encode(prompter.model_input, add_special_tokens=True))


def extend_to_target(tokenizer, prompt_module, messages: list[dict], target: int) -> list[dict]:
    result = [dict(message) for message in messages]
    if token_count(tokenizer, prompt_module, result) == target:
        return result
    user_indexes = [index for index, message in enumerate(result) if message.get("role") == "user"]
    if not user_indexes:
        raise ValueError("message list has no user message")
    index = user_indexes[-1]
    original = str(result[index]["content"])
    low, high = 0, target
    while low <= high:
        count = (low + high) // 2
        result[index]["content"] = original + " x" * count
        observed = token_count(tokenizer, prompt_module, result)
        if observed == target:
            return result
        if observed < target:
            low = count + 1
        else:
            high = count - 1
    for count in range(max(0, high - 4), min(target, low + 4) + 1):
        result[index]["content"] = original + " x" * count
        if token_count(tokenizer, prompt_module, result) == target:
            return result
    raise ValueError(f"could not construct exact {target}-token request")


def write_jsonl(path: Path, values: list[list[dict]]) -> None:
    path.write_bytes(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values).encode("utf-8")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--source-formal", type=Path, required=True)
    parser.add_argument("--source-warmup", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    prompt_module = load_prompt_module(Path(__file__).resolve().parents[2])
    formal_source = read_jsonl(args.source_formal)
    warmup_source = read_jsonl(args.source_warmup)[:8]
    if len(formal_source) != 32 or len(warmup_source) != 8:
        raise ValueError("expected 32 formal and at least 8 warmup source requests")
    output_root.mkdir(parents=True)
    workloads = []
    for prompt in PROMPTS:
        formal = [extend_to_target(tokenizer, prompt_module, value, prompt) for value in formal_source]
        warmup = [extend_to_target(tokenizer, prompt_module, value, prompt) for value in warmup_source]
        formal_path = output_root / f"p{prompt}_n32-formal.jsonl"
        warmup_path = output_root / f"p{prompt}_n8-warmup.jsonl"
        write_jsonl(formal_path, formal)
        write_jsonl(warmup_path, warmup)
        observed = [token_count(tokenizer, prompt_module, value) for value in formal + warmup]
        if observed != [prompt] * 40:
            raise ValueError(f"p{prompt}: strict token validation failed")
        workloads.append({
            "prompt_tokens": prompt,
            "formal": {"path": formal_path.name, "requests": 32, "sha256": sha256(formal_path)},
            "warmup": {"path": warmup_path.name, "requests": 8, "sha256": sha256(warmup_path)},
            "generation": "extend the corresponding audited p128 message with deterministic ' x' tokens",
        })
    manifest = {
        "schema_version": 1,
        "workload_id": "qwen3_32b_length_matrix_v1",
        "tokenizer_path": str(args.tokenizer_path),
        "source_formal_sha256": sha256(args.source_formal),
        "source_warmup_sha256": sha256(args.source_warmup),
        "selection": "all 32 formal requests and first 8 warmup requests; exact chat-template token count",
        "workloads": workloads,
    }
    (output_root / "workload-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "workloads": workloads}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
