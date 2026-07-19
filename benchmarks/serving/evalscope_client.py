"""Build the single EvalScope/OpenAI client command used by all adapters."""

from __future__ import annotations

from typing import Any, Mapping


def build_evalscope_command(
    contract: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    base_url: str,
    output_dir: str,
    phase: str = "formal",
) -> list[str]:
    if phase not in {"formal", "warmup"}:
        raise ValueError("phase must be formal or warmup")
    is_warmup = phase == "warmup"
    dataset = contract["dataset"]["warmup" if is_warmup else "formal"]
    requests = run["warmup_requests" if is_warmup else "formal_requests"]
    offset = contract["warmup_offset" if is_warmup else "formal_offset"]
    command = [
        "evalscope",
        "perf",
        "--url",
        base_url.rstrip("/") + contract["endpoint_path"],
        "--api",
        contract["api"],
        "--model",
        contract["served_model"],
        "--tokenizer-path",
        contract["tokenizer"],
        "--dataset",
        "line_by_line",
        "--dataset-path",
        dataset,
        "--number",
        str(requests),
        "--parallel",
        str(run["concurrency"]),
        "--warmup-num",
        "0",
        "--min-tokens",
        str(run["output_tokens"]),
        "--max-tokens",
        str(run["output_tokens"]),
        "--temperature",
        str(contract["sampling"]["temperature"]),
        "--top-p",
        str(contract["sampling"]["top_p"]),
        "--seed",
        str(contract["seed"]),
        "--dataset-offset",
        str(offset),
        "--no-test-connection",
        "--total-timeout",
        str(contract["total_timeout_s"]),
        "--outputs-dir",
        output_dir,
        "--no-timestamp",
        "--name",
        f"{run['run_id']}_{phase}",
    ]
    if contract["stream"]:
        command.append("--stream")
    return command
