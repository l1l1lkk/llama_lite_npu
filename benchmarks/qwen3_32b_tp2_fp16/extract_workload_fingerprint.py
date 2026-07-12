#!/usr/bin/env python3
"""Extract exact prompt-token evidence from an EvalScope SQLite result."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def find_single(root: Path, name: str) -> Path:
    matches = list(root.glob(f"**/{name}"))
    if len(matches) != 1:
        raise ValueError(f"{root}: expected one {name}, found {len(matches)}")
    return matches[0]


def last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return " ".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def load_prompt_module(repo_root: Path):
    path = repo_root / "lite_llama" / "utils" / "prompt_templates.py"
    spec = importlib.util.spec_from_file_location("benchmark_prompt_templates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract(
    evalscope_root: Path,
    *,
    tokenizer_path: Path,
    repo_root: Path,
    expected_prompt_tokens: int,
    expected_completion_tokens: int,
) -> dict[str, object]:
    from transformers import AutoTokenizer

    database = find_single(evalscope_root, "benchmark_data.db")
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "select rowid, request, prompt_tokens, completion_tokens, success "
            "from result order by rowid"
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"no request rows in {database}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=True
    )
    prompt_module = load_prompt_module(repo_root)
    requests = []
    all_token_ids = []
    for index, (_, request_json, prompt_tokens, completion_tokens, success) in enumerate(rows):
        body = json.loads(request_json)
        prompt = last_user_text(body.get("messages", []))
        prompter = prompt_module.get_prompter(
            "qwen3", "", enable_thinking=body.get("enable_thinking", True)
        )
        prompter.insert_prompt(prompt)
        token_ids = [
            int(token)
            for token in tokenizer.encode(
                prompter.model_input, add_special_tokens=True
            )
        ]
        if len(token_ids) != int(prompt_tokens):
            raise ValueError(
                f"request {index}: reconstructed {len(token_ids)} prompt tokens, "
                f"database reports {prompt_tokens}"
            )
        if int(prompt_tokens) != expected_prompt_tokens:
            raise ValueError(
                f"request {index}: prompt_tokens={prompt_tokens}, "
                f"expected={expected_prompt_tokens}"
            )
        if int(completion_tokens) != expected_completion_tokens:
            raise ValueError(
                f"request {index}: completion_tokens={completion_tokens}, "
                f"expected={expected_completion_tokens}"
            )
        if int(success) != 1:
            raise ValueError(f"request {index}: success={success}")
        selected_parameters = {
            key: body.get(key)
            for key in (
                "model",
                "max_tokens",
                "min_tokens",
                "seed",
                "stream",
                "temperature",
                "top_p",
                "enable_thinking",
            )
        }
        requests.append(
            {
                "index": index,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "success": int(success),
                "prompt_token_ids": token_ids,
                "prompt_token_ids_sha256": canonical_sha256(token_ids),
                "request_parameters": selected_parameters,
                "request_parameters_sha256": canonical_sha256(selected_parameters),
            }
        )
        all_token_ids.append(token_ids)
    return {
        "schema_version": 1,
        "source_database": database.relative_to(evalscope_root).as_posix(),
        "source_database_size_bytes": database.stat().st_size,
        "source_database_sha256": sha256_file(database),
        "request_count": len(requests),
        "expected_prompt_tokens": expected_prompt_tokens,
        "expected_completion_tokens": expected_completion_tokens,
        "prompt_sequence_sha256": canonical_sha256(all_token_ids),
        "requests": requests,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evalscope_root", type=Path)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-prompt-tokens", type=int, required=True)
    parser.add_argument("--expected-completion-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = extract(
        args.evalscope_root.resolve(),
        tokenizer_path=args.tokenizer_path.resolve(),
        repo_root=args.repo_root.resolve(),
        expected_prompt_tokens=args.expected_prompt_tokens,
        expected_completion_tokens=args.expected_completion_tokens,
    )
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"requests": report["request_count"], "status": "ok"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
