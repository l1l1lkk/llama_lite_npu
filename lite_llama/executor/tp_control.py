"""Tensor-only control messages for TP continuous batching.

The hot scheduler path must not pickle Python request objects for every model
step.  Rank 0 encodes the small amount of scheduler metadata into fixed dtype
tensors, while worker ranks reconstruct only the state required to mirror the
model execution.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence


_OP_PREFILL = 1
_OP_DECODE = 2
_OP_RELEASE = 3
_OP_SHUTDOWN = 4

_OPERATION_NAMES = {
    _OP_PREFILL: "prefill",
    _OP_DECODE: "decode",
    _OP_RELEASE: "release",
    _OP_SHUTDOWN: "shutdown",
}


class DecodedCommand(NamedTuple):
    operation: str
    control_ids: list[int]
    prompt_tokens: list[list[int]]
    max_new_tokens: list[int]
    temperatures: list[float]
    top_ps: list[float]


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
    )


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
