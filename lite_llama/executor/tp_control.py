"""Tensor-only control messages for TP continuous batching.

The hot scheduler path must not pickle Python request objects for every model
step.  Rank 0 encodes the small amount of scheduler metadata into fixed dtype
tensors, while worker ranks reconstruct only the state required to mirror the
model execution.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from typing import NamedTuple, Sequence


_OP_PREFILL = 1
_OP_DECODE = 2
_OP_RELEASE = 3
_OP_SHUTDOWN = 4
_OP_PREFILL_CHUNK = 5

_OPERATION_NAMES = {
    _OP_PREFILL: "prefill",
    _OP_DECODE: "decode",
    _OP_RELEASE: "release",
    _OP_SHUTDOWN: "shutdown",
    _OP_PREFILL_CHUNK: "prefill_chunk",
}


class DecodedCommand(NamedTuple):
    operation: str
    control_ids: list[int]
    prompt_tokens: list[list[int]]
    max_new_tokens: list[int]
    temperatures: list[float]
    top_ps: list[float]
    prefill_cursors: list[int] = []
    chunk_size: int = 0


def _header(operation: int, batch_size: int, ints, floats):
    return [operation, batch_size, len(ints), len(floats)]


def encode_prefill(requests):
    integers: list[int] = []
    floats: list[float] = []
    for request in requests:
        if request.control_id is None:
            raise RuntimeError("continuous batching request has no control_id")
        prompt_tokens = [int(token) for token in request.prompt_tokens]
        integers.extend(
            [
                int(request.control_id),
                int(request.max_new_tokens),
                len(prompt_tokens),
                *prompt_tokens,
            ]
        )
        floats.extend([float(request.temperature), float(request.top_p)])
    return _header(_OP_PREFILL, len(requests), integers, floats), integers, floats


def encode_prefill_chunk(requests, chunk_size: int):
    integers: list[int] = [int(chunk_size)]
    floats: list[float] = []
    for request in requests:
        if request.control_id is None:
            raise RuntimeError("continuous batching request has no control_id")
        prompt_tokens = [int(token) for token in request.prompt_tokens]
        integers.extend(
            [
                int(request.control_id),
                int(request.max_new_tokens),
                int(getattr(request, "prefill_cursor", 0)),
                len(prompt_tokens),
                *prompt_tokens,
            ]
        )
        floats.extend([float(request.temperature), float(request.top_p)])
    return _header(_OP_PREFILL_CHUNK, len(requests), integers, floats), integers, floats


def _encode_ids(operation: int, control_ids: Sequence[int]):
    integers = [int(control_id) for control_id in control_ids]
    return _header(operation, len(integers), integers, []), integers, []


def encode_decode(control_ids: Sequence[int]):
    return _encode_ids(_OP_DECODE, control_ids)


def encode_release(control_ids: Sequence[int]):
    return _encode_ids(_OP_RELEASE, control_ids)


def encode_shutdown():
    return _header(_OP_SHUTDOWN, 0, [], []), [], []


def _as_list(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return list(values)


def decode_command(header, integers, floats) -> DecodedCommand:
    header = [int(value) for value in _as_list(header)]
    integers = [int(value) for value in _as_list(integers)]
    floats = [float(value) for value in _as_list(floats)]
    if len(header) != 4:
        raise RuntimeError(f"invalid TP command header length: {len(header)}")

    operation_code, batch_size, integer_count, float_count = header
    operation = _OPERATION_NAMES.get(operation_code)
    if operation is None:
        raise RuntimeError(f"unknown TP command operation: {operation_code}")
    if integer_count != len(integers) or float_count != len(floats):
        raise RuntimeError("TP command payload length does not match header")

    prefill_cursors: list[int] = []
    chunk_size = 0
    if operation == "prefill":
        control_ids: list[int] = []
        prompt_tokens: list[list[int]] = []
        max_new_tokens: list[int] = []
        cursor = 0
        for _ in range(batch_size):
            if cursor + 3 > len(integers):
                raise RuntimeError("truncated TP prefill metadata")
            control_id, max_new, prompt_length = integers[cursor : cursor + 3]
            cursor += 3
            end = cursor + prompt_length
            if end > len(integers):
                raise RuntimeError("truncated TP prefill prompt")
            control_ids.append(control_id)
            max_new_tokens.append(max_new)
            prompt_tokens.append(integers[cursor:end])
            cursor = end
        if cursor != len(integers) or len(floats) != batch_size * 2:
            raise RuntimeError("invalid TP prefill payload")
        temperatures = floats[0::2]
        top_ps = floats[1::2]
    elif operation == "prefill_chunk":
        control_ids = []
        prompt_tokens = []
        max_new_tokens = []
        prefill_cursors = []
        if not integers:
            raise RuntimeError("missing TP prefill_chunk chunk size")
        chunk_size = int(integers[0])
        cursor = 1
        for _ in range(batch_size):
            if cursor + 4 > len(integers):
                raise RuntimeError("truncated TP prefill_chunk metadata")
            control_id, max_new, prefill_cursor, prompt_length = integers[
                cursor : cursor + 4
            ]
            cursor += 4
            end = cursor + prompt_length
            if end > len(integers):
                raise RuntimeError("truncated TP prefill_chunk prompt")
            control_ids.append(control_id)
            max_new_tokens.append(max_new)
            prefill_cursors.append(prefill_cursor)
            prompt_tokens.append(integers[cursor:end])
            cursor = end
        if cursor != len(integers) or len(floats) != batch_size * 2:
            raise RuntimeError("invalid TP prefill_chunk payload")
        temperatures = floats[0::2]
        top_ps = floats[1::2]
    else:
        if len(integers) != batch_size or floats:
            raise RuntimeError(f"invalid TP {operation} payload")
        control_ids = integers
        prompt_tokens = []
        max_new_tokens = []
        temperatures = []
        top_ps = []

    return DecodedCommand(
        operation=operation,
        control_ids=control_ids,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperatures=temperatures,
        top_ps=top_ps,
        prefill_cursors=prefill_cursors,
        chunk_size=chunk_size,
    )




def encode_store_payload(encoded) -> bytes:
    """Serialize a command without creating device tensors.

    This is used by the TP continuous-batching CPU control plane.  The payload
    is intentionally plain JSON because command metadata is tiny compared with
    model compute and must remain debuggable.
    """

    header, integers, floats = encoded
    return json.dumps(
        {
            "header": [int(value) for value in header],
            "integers": [int(value) for value in integers],
            "floats": [float(value) for value in floats],
        },
        separators=(",", ":"),
    ).encode("utf-8")


def decode_store_payload(payload) -> DecodedCommand:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    data = json.loads(payload)
    return decode_command(
        data.get("header", []),
        data.get("integers", []),
        data.get("floats", []),
    )


class StoreCommandChannel:
    """CPU-side TP command channel for continuous batching.

    HCCL collectives should not be used as an idle command queue: worker ranks
    may wait for long periods between HTTP requests, and long-lived NPU stream
    waits can hit Ascend runtime watchdog timeouts.  This channel uses
    ``torch.distributed.TCPStore`` for control metadata and lets ranks enter
    NPU/HCCL only when real model work starts.
    """

    def __init__(
        self,
        src: int = 0,
        store=None,
        prefix: str = "lite_llama_tp_cb",
        timeout_seconds: int | None = None,
    ) -> None:
        self.src = int(src)
        self.prefix = str(prefix).rstrip("/")
        self.sequence = 0
        self.store = store if store is not None else self._create_default_store(
            timeout_seconds=timeout_seconds
        )

    @staticmethod
    def _create_default_store(timeout_seconds: int | None = None):
        import torch

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "StoreCommandChannel requires an initialized torch.distributed process group"
            )
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        host = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        port = int(os.environ.get("LITE_LLAMA_TP_STORE_PORT", master_port + 2000))
        timeout = timedelta(seconds=int(timeout_seconds or os.environ.get("LITE_LLAMA_TP_STORE_TIMEOUT", "7200")))
        return torch.distributed.TCPStore(
            host,
            port,
            world_size,
            rank == 0,
            timeout,
        )

    def _key(self) -> str:
        return f"{self.prefix}/{self.sequence}"

    def send(self, encoded) -> None:
        self.store.set(self._key(), encode_store_payload(encoded))
        self.sequence += 1

    def receive(self) -> DecodedCommand:
        payload = self.store.get(self._key())
        self.sequence += 1
        return decode_store_payload(payload)


class TensorCommandChannel:
    """Broadcast continuous-batching commands with HCCL-compatible tensors."""

    def __init__(self, device, src: int = 0, group=None) -> None:
        self.device = device
        self.src = int(src)
        self.group = group

    def _broadcast(self, tensor) -> None:
        import torch

        torch.distributed.broadcast(
            tensor,
            src=self.src,
            group=self.group,
        )

    def send(self, encoded) -> None:
        import torch

        header, integers, floats = encoded
        header_tensor = torch.tensor(
            header, dtype=torch.int64, device=self.device
        )
        self._broadcast(header_tensor)
        if integers:
            self._broadcast(
                torch.tensor(
                    integers, dtype=torch.int64, device=self.device
                )
            )
        if floats:
            self._broadcast(
                torch.tensor(
                    floats, dtype=torch.float32, device=self.device
                )
            )

    def receive(self) -> DecodedCommand:
        import torch

        header = torch.empty(4, dtype=torch.int64, device=self.device)
        self._broadcast(header)
        header_values = header.cpu().tolist()
        integer_count = int(header_values[2])
        float_count = int(header_values[3])
        integers = torch.empty(
            integer_count, dtype=torch.int64, device=self.device
        )
        floats = torch.empty(
            float_count, dtype=torch.float32, device=self.device
        )
        if integer_count:
            self._broadcast(integers)
        if float_count:
            self._broadcast(floats)
        return decode_command(header_values, integers, floats)
