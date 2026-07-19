import json

import pytest

from benchmarks.serving.validate.stream_usage import validate_stream_usage_evidence


def _event(*, choices, stream_id="chatcmpl-one", model="Qwen3-32B", usage=None):
    event = {"id": stream_id, "model": model, "choices": choices}
    if usage is not None:
        event["usage"] = usage
    return f"data: {json.dumps(event)}\n\n"


def _valid_raw_sse():
    return [
        _event(choices=[{"delta": {"content": "x"}, "finish_reason": None}]),
        _event(choices=[{"delta": {}, "finish_reason": "length"}]),
        _event(
            choices=[],
            usage={"prompt_tokens": 51, "completion_tokens": 64, "total_tokens": 115},
        ),
        "data: [DONE]\n\n",
    ]


def test_persisted_evidence_is_unknown_not_missing_or_observed():
    result = validate_stream_usage_evidence(
        persisted_response_messages=[{"choices": [{"delta": {"content": "x"}}]}],
        db_prompt_tokens=51,
        db_completion_tokens=64,
    )
    assert result == {
        "wire_usage_status": "unknown",
        "evidence_source": "evalscope_persistence",
        "token_source": "ambiguous_wire_or_fallback",
        "gate_eligible": False,
        "errors": [],
    }


def test_raw_sse_observed_is_gate_eligible():
    result = validate_stream_usage_evidence(raw_sse=_valid_raw_sse())
    assert result["wire_usage_status"] == "observed"
    assert result["evidence_source"] == "raw_sse"
    assert result["token_source"] == "wire_usage"
    assert result["gate_eligible"] is True
    assert result["usage"] == {
        "prompt_tokens": 51,
        "completion_tokens": 64,
        "total_tokens": 115,
    }
    assert result["errors"] == []


def test_transport_chunk_boundaries_do_not_change_observed_result():
    wire = "".join(_valid_raw_sse())
    expected = validate_stream_usage_evidence(raw_sse=[wire])
    combined_events = validate_stream_usage_evidence(
        raw_sse=[wire[: wire.index("data: [DONE]")], wire[wire.index("data: [DONE]") :]]
    )
    fragmented_json = validate_stream_usage_evidence(
        raw_sse=[wire[:17], wire[17:53], wire[53:121], wire[121:]]
    )
    assert expected["wire_usage_status"] == "observed"
    assert combined_events == expected
    assert fragmented_json == expected


def test_crlf_delimiter_split_across_transport_chunks():
    wire = "".join(_valid_raw_sse()).replace("\n", "\r\n")
    split_at = wire.index("\r\n") + 1
    result = validate_stream_usage_evidence(
        raw_sse=[wire[:split_at], wire[split_at:]]
    )
    assert result["wire_usage_status"] == "observed"
    assert result["errors"] == []


@pytest.mark.parametrize(
    ("name", "delimiter"),
    [("lf", "\n\n"), ("crlf", "\r\n\r\n"), ("cr", "\r\r")],
)
def test_every_single_transport_split_matches_whole_wire(name, delimiter):
    wire = "".join(_valid_raw_sse()).replace("\n\n", delimiter)
    expected = validate_stream_usage_evidence(raw_sse=[wire])
    assert expected["wire_usage_status"] == "observed"
    if name == "crlf":
        pending_cr_split = len(wire) - 1
        assert wire[:pending_cr_split].endswith("\r")
        assert wire[pending_cr_split:].startswith("\n")
    for split_at in range(len(wire) + 1):
        actual = validate_stream_usage_evidence(
            raw_sse=[wire[:split_at], wire[split_at:]]
        )
        assert actual == expected, (name, split_at, actual)


@pytest.mark.parametrize(
    "delimiter",
    ["\n\n", "\r\n\r\n", "\r\r"],
)
def test_three_transport_fragments_match_whole_wire(delimiter):
    wire = "".join(_valid_raw_sse()).replace("\n\n", delimiter)
    expected = validate_stream_usage_evidence(raw_sse=[wire])
    json_split = wire.index('"choices"') + 4
    delimiter_split = wire.rfind(delimiter) + 1
    actual = validate_stream_usage_evidence(
        raw_sse=[wire[:json_split], wire[json_split:delimiter_split], wire[delimiter_split:]]
    )
    assert actual == expected


def test_unterminated_final_sse_event_is_invalid():
    wire = "".join(_valid_raw_sse())
    result = validate_stream_usage_evidence(raw_sse=[wire[:-1]])
    assert result["wire_usage_status"] == "invalid"
    assert "sse.incomplete_event" in result["errors"]


@pytest.mark.parametrize(
    "suffix",
    [
        "x",
        _event(choices=[{"delta": {"content": "late"}, "finish_reason": None}]),
    ],
)
def test_nonempty_data_after_done_is_invalid(suffix):
    wire = "".join(_valid_raw_sse())
    result = validate_stream_usage_evidence(raw_sse=[wire + suffix])
    assert result["wire_usage_status"] == "invalid"
    assert "sse.data_after_done" in result["errors"]


def test_complete_raw_sse_without_usage_is_missing():
    raw = _valid_raw_sse()
    del raw[-2]
    result = validate_stream_usage_evidence(raw_sse=raw)
    assert result["wire_usage_status"] == "missing"
    assert result["evidence_source"] == "raw_sse"
    assert result["token_source"] == "absent"
    assert result["gate_eligible"] is False
    assert result["errors"] == []


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (lambda raw: raw.insert(-1, raw[-2]), "usage.duplicate"),
        (lambda raw: raw.insert(0, raw.pop(-2)), "usage.order"),
        (
            lambda raw: raw.__setitem__(
                -2,
                _event(
                    choices=[],
                    stream_id="chatcmpl-other",
                    usage={"prompt_tokens": 51, "completion_tokens": 64,
                           "total_tokens": 115},
                ),
            ),
            "stream.id_mismatch",
        ),
        (
            lambda raw: raw.__setitem__(
                -2,
                _event(
                    choices=[],
                    model="different-model",
                    usage={"prompt_tokens": 51, "completion_tokens": 64,
                           "total_tokens": 115},
                ),
            ),
            "stream.model_mismatch",
        ),
        (
            lambda raw: raw.__setitem__(
                -2,
                _event(
                    choices=[],
                    usage={"prompt_tokens": 51, "completion_tokens": 64,
                           "total_tokens": 999},
                ),
            ),
            "usage.total_tokens_mismatch",
        ),
        (lambda raw: raw.__setitem__(-2, "data: {not-json}\n\n"), "sse.invalid_json"),
    ],
)
def test_invalid_raw_sse_has_field_level_error(mutate, expected_error):
    raw = _valid_raw_sse()
    mutate(raw)
    result = validate_stream_usage_evidence(raw_sse=raw)
    assert result["wire_usage_status"] == "invalid"
    assert result["gate_eligible"] is False
    assert expected_error in result["errors"]


def test_usage_tokens_must_be_non_negative_integers():
    raw = _valid_raw_sse()
    raw[-2] = _event(
        choices=[],
        usage={"prompt_tokens": True, "completion_tokens": -1, "total_tokens": 0},
    )
    result = validate_stream_usage_evidence(raw_sse=raw)
    assert "usage.prompt_tokens_invalid" in result["errors"]
    assert "usage.completion_tokens_invalid" in result["errors"]
