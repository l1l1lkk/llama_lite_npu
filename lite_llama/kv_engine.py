"""CPU-side scheduling metadata for KV cache planning.

This module intentionally does not own NPU KV tensors. It tracks logical block
metadata that higher-level schedulers can use before wiring reuse into the live
PagedAttention allocator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PrefixCacheMatch:
    matched_tokens: int
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class PrefillChunk:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


class KVBlockRefCounter:
    """Track logical KV block ownership and sharing reference counts."""

    def __init__(self, total_blocks: int) -> None:
        total_blocks = int(total_blocks)
        if total_blocks < 1:
            raise ValueError("total_blocks must be positive")
        self._refcounts = [0] * total_blocks

    @property
    def total_blocks(self) -> int:
        return len(self._refcounts)

    @property
    def available_count(self) -> int:
        return sum(1 for count in self._refcounts if count == 0)

    def refcount(self, block_id: int) -> int:
        return self._refcounts[self._check_block_id(block_id)]

    def acquire(self, count: int) -> tuple[int, ...]:
        count = int(count)
        if count < 1:
            raise ValueError("count must be positive")
        blocks = [
            block_id
            for block_id, refcount in enumerate(self._refcounts)
            if refcount == 0
        ][:count]
        if len(blocks) != count:
            raise RuntimeError(
                f"insufficient KV blocks: need {count}, available {len(blocks)}"
            )
        for block_id in blocks:
            self._refcounts[block_id] = 1
        return tuple(blocks)

    def add_ref(self, block_ids: Iterable[int]) -> None:
        for block_id in block_ids:
            checked = self._check_block_id(block_id)
            if self._refcounts[checked] == 0:
                raise ValueError(f"cannot add ref to free KV block {checked}")
            self._refcounts[checked] += 1

    def release(self, block_ids: Iterable[int]) -> None:
        for block_id in block_ids:
            checked = self._check_block_id(block_id)
            if self._refcounts[checked] == 0:
                raise ValueError(f"KV block {checked} refcount would become negative")
            self._refcounts[checked] -= 1

    def _check_block_id(self, block_id: int) -> int:
        block_id = int(block_id)
        if block_id < 0 or block_id >= len(self._refcounts):
            raise IndexError(f"KV block id out of range: {block_id}")
        return block_id


class PrefixCache:
    """Block-aligned prefix cache metadata.

    Keys are token blocks. Values are logical KV block ids. Matching stops at
    the first missing full block and never returns a partial block.
    """

    def __init__(self, block_size: int) -> None:
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._block_to_id: dict[tuple[int, ...], int] = {}

    def put(
        self,
        token_ids: Sequence[int],
        block_ids: Sequence[int],
    ) -> None:
        full_block_count = len(token_ids) // self.block_size
        if full_block_count == 0:
            return
        if len(block_ids) < full_block_count:
            raise ValueError("not enough block ids for full token blocks")
        for block_index in range(full_block_count):
            start = block_index * self.block_size
            end = start + self.block_size
            key = tuple(int(token_id) for token_id in token_ids[start:end])
            self._block_to_id[key] = int(block_ids[block_index])

    def match(self, token_ids: Sequence[int]) -> PrefixCacheMatch:
        block_ids: list[int] = []
        full_block_count = len(token_ids) // self.block_size
        for block_index in range(full_block_count):
            start = block_index * self.block_size
            end = start + self.block_size
            key = tuple(int(token_id) for token_id in token_ids[start:end])
            block_id = self._block_to_id.get(key)
            if block_id is None:
                break
            block_ids.append(block_id)
        return PrefixCacheMatch(
            matched_tokens=len(block_ids) * self.block_size,
            block_ids=tuple(block_ids),
        )


class ChunkedPrefillPlanner:
    """Split prompts into deterministic prefill chunks."""

    def __init__(self, chunk_size: int) -> None:
        chunk_size = int(chunk_size)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    def plan(self, prompt_length: int) -> tuple[PrefillChunk, ...]:
        prompt_length = int(prompt_length)
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        chunks = []
        for start in range(0, prompt_length, self.chunk_size):
            chunks.append(
                PrefillChunk(start=start, end=min(start + self.chunk_size, prompt_length))
            )
        return tuple(chunks)
