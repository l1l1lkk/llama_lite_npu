"""Validate provenance-aware streaming usage evidence.

EvalScope's persisted ``response_messages`` intentionally exclude usage-only
SSE events.  Consequently, only a captured raw SSE sequence may establish
whether the server emitted such an event.  Persisted messages and database
token counts remain useful diagnostics, but their wire provenance is unknown.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


def _result(
    status: str,
    source: str,
    token_source: str,
    errors: list[str],
    *,
    usage: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "wire_usage_status": status,
        "evidence_source": source,
        "token_source": token_source,
        "gate_eligible": status == "observed" and not errors,
        "errors": sorted(set(errors)),
    }
    if usage is not None:
        result["usage"] = dict(usage)
    return result


def _parse_raw_sse(raw_sse: Sequence[str]) -> tuple[list[Any], list[str]]:
    """Parse ordered decoded transport chunks without trusting their boundaries."""

    events: list[Any] = []
    errors: list[str] = []
    transport_chunks: list[str] = []
    event_delimiter = re.compile(r"(?:(?:\r\n)|\r|\n){2}")
    for raw_chunk in raw_sse:
        if not isinstance(raw_chunk, str):
            errors.append("sse.chunk_not_string")
            continue
        transport_chunks.append(raw_chunk)

    # The complete bounded probe is already in memory.  Join before delimiter
    # parsing so a trailing CR cannot be mistaken for a standalone line ending
    # when it is actually the first half of a CRLF in the next transport chunk.
    buffer = "".join(transport_chunks)
    while delimiter := event_delimiter.search(buffer):
        message = buffer[: delimiter.start()]
        buffer = buffer[delimiter.end() :]
        message = message.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not message:
            continue
        if not message.startswith("data:"):
            errors.append("sse.missing_data_prefix")
            continue
        payload = message.removeprefix("data:").strip()
        if payload == "[DONE]":
            events.append("[DONE]")
            continue
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            errors.append("sse.invalid_json")
            continue
        if not isinstance(decoded, Mapping):
            errors.append("sse.payload_not_object")
            continue
        events.append(decoded)
    if buffer:
        errors.append("sse.incomplete_event")
        if "[DONE]" in events:
            errors.append("sse.data_after_done")
    return events, errors


def _valid_token(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_stream_usage_evidence(
    *,
    raw_sse: Sequence[str] | None = None,
    persisted_response_messages: Sequence[Mapping[str, Any]] | None = None,
    db_prompt_tokens: int | None = None,
    db_completion_tokens: int | None = None,
) -> dict[str, Any]:
    """Classify streaming usage evidence without inventing wire provenance.

    ``raw_sse`` is an ordered sequence of already UTF-8-decoded transport
    chunks.  Chunk boundaries are arbitrary and need not align with SSE event,
    line, CRLF, or JSON boundaries.  Every event, including the final
    ``[DONE]``, must end with a complete SSE blank-line delimiter.

    The persisted arguments are intentionally accepted to make the ambiguity
    explicit.  Their values never change the result from ``unknown`` because
    EvalScope may populate the token fields from either a wire usage event or
    its tokenizer fallback.
    """

    if raw_sse is None:
        # Touch these values only to make the intentionally ignored inputs
        # explicit to type checkers and future maintainers.
        _ = (persisted_response_messages, db_prompt_tokens, db_completion_tokens)
        return _result(
            "unknown",
            "evalscope_persistence",
            "ambiguous_wire_or_fallback",
            [],
        )

    events, errors = _parse_raw_sse(raw_sse)
    if not events:
        errors.append("sse.empty")
        return _result("invalid", "raw_sse", "invalid", errors)

    done_indices = [index for index, event in enumerate(events) if event == "[DONE]"]
    if len(done_indices) != 1:
        errors.append("sse.done_count")
    elif done_indices[0] != len(events) - 1:
        errors.append("sse.done_not_last")
        errors.append("sse.data_after_done")

    payloads = [event for event in events if event != "[DONE]"]
    baseline_id = payloads[0].get("id") if payloads else None
    baseline_model = payloads[0].get("model") if payloads else None
    if not isinstance(baseline_id, str) or not baseline_id:
        errors.append("stream.id_missing")
    if not isinstance(baseline_model, str) or not baseline_model:
        errors.append("stream.model_missing")
    for payload in payloads[1:]:
        if payload.get("id") != baseline_id:
            errors.append("stream.id_mismatch")
        if payload.get("model") != baseline_model:
            errors.append("stream.model_mismatch")

    finish_indices: list[int] = []
    usage_indices: list[int] = []
    usage_value: Mapping[str, int] | None = None
    for index, event in enumerate(events):
        if event == "[DONE]":
            continue
        choices = event.get("choices")
        usage = event.get("usage")
        if usage is not None:
            usage_indices.append(index)
            if choices != []:
                errors.append("usage.choices_not_empty")
            if not isinstance(usage, Mapping):
                errors.append("usage.not_object")
                continue
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if not _valid_token(usage.get(field)):
                    errors.append(f"usage.{field}_invalid")
            if all(_valid_token(usage.get(field)) for field in (
                "prompt_tokens", "completion_tokens", "total_tokens"
            )):
                if usage["total_tokens"] != (
                    usage["prompt_tokens"] + usage["completion_tokens"]
                ):
                    errors.append("usage.total_tokens_mismatch")
                usage_value = usage
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping) and first.get("finish_reason") is not None:
                finish_indices.append(index)

    if len(finish_indices) != 1:
        errors.append("stream.finish_count")
    if len(usage_indices) > 1:
        errors.append("usage.duplicate")
    if usage_indices and finish_indices:
        if not (finish_indices[0] < usage_indices[0]):
            errors.append("usage.order")
        if done_indices and not (usage_indices[-1] < done_indices[0]):
            errors.append("usage.order")

    if errors:
        return _result("invalid", "raw_sse", "invalid", errors)
    if not usage_indices:
        return _result("missing", "raw_sse", "absent", [])
    return _result(
        "observed",
        "raw_sse",
        "wire_usage",
        [],
        usage=usage_value,
    )
