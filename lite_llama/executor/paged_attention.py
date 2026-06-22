"""PagedAttention: KV cache memory management with page-sized blocks.

Like vLLM's PagedAttention, KV cache is allocated in fixed-size pages
(blocks of tokens) instead of as one contiguous block per request.
This eliminates fragmentation and enables memory sharing.

The FlashDecoding kernel already uses b_req_tokens_table for indirect
KV cache indexing — no kernel changes needed!

Architecture:
  KV cache pool: [P0][P1][P2]...[P_{max_pages}]
                    each page = PAGE_SIZE tokens × (2*kv_heads*head_dim) × fp16
  Free list:     [P7, P12, P23, ...]  ← available pages
  Page table:    req0 → [P0, P1, P2, P5, P8]  ← non-contiguous
                 req1 → [P3, P4]
                 req2 → [P6, P7, P9, P10, P11]

Usage:
  from .paged_attention import PagedKVCacheManager
  mgr = PagedKVCacheManager(num_layers, num_kv_heads, head_dim, num_pages, page_size)
  page_indices = mgr.alloc(num_tokens)  # allocates enough pages
  mgr.free(page_indices)                # returns pages to free pool
"""

from __future__ import annotations

import torch
from typing import List, Optional, Tuple


class PagedKVCacheManager:
    """Page-based KV cache memory manager.

    Args:
        num_layers: Number of transformer layers
        num_kv_heads: Number of KV heads (per TP rank)
        head_dim: Head dimension
        num_pages: Total number of pages in the pool
        page_size: Tokens per page (typically 16-256)
        dtype: Data type (fp16)
        device: NPU device
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_pages: int,
        page_size: int = 16,
        dtype: torch.dtype = torch.float16,
        device: str = "npu:0",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_pages = num_pages
        self.page_size = page_size
        self.dtype = dtype
        self.device = device

        # Total tokens that can be stored
        self.max_tokens = num_pages * page_size
        self.max_num_tokens = self.max_tokens

        # Pre-allocate one big KV buffer: (num_pages * page_size, 2*kv_heads, head_dim)
        tokens_per_layer = num_pages * page_size
        self.gpu_kv_buffer = [
            torch.empty(
                (tokens_per_layer, 2 * num_kv_heads, head_dim),
                dtype=dtype, device=device,
            )
            for _ in range(num_layers)
        ]

        # Free page pool: True = free, False = allocated.  Refcounts are kept
        # on Host because page ownership is scheduler metadata, not model data.
        self.page_free = torch.ones(num_pages, dtype=torch.bool, device="cpu")
        self.page_refcount = torch.zeros(num_pages, dtype=torch.int32, device="cpu")
        self.num_free_pages = num_pages

    # ------------------------------------------------------------------
    # Allocation / Free
    # ------------------------------------------------------------------
    def alloc(self, num_tokens: int) -> Optional[torch.Tensor]:
        """Allocate pages for `num_tokens`. Returns list of page indices."""
        num_needed = (num_tokens + self.page_size - 1) // self.page_size
        if num_needed > self.num_free_pages:
            return None

        free_indices = torch.nonzero(self.page_free).squeeze(-1)[:num_needed]
        self.page_free[free_indices] = False
        self.page_refcount[free_indices] = 1
        self.num_free_pages -= num_needed
        return free_indices

    def add_ref(self, page_indices: torch.Tensor):
        """Add references to already allocated pages."""
        if len(page_indices) == 0:
            return
        page_indices = page_indices.to(device="cpu", dtype=torch.long)
        if torch.any(self.page_refcount[page_indices] <= 0):
            raise ValueError("cannot add ref to a free KV page")
        self.page_refcount[page_indices] += 1

    def free(self, page_indices: torch.Tensor):
        """Drop one reference and return pages to the pool at refcount zero."""
        if len(page_indices) == 0:
            return
        page_indices = page_indices.to(device="cpu", dtype=torch.long)
        if torch.any(self.page_refcount[page_indices] <= 0):
            raise ValueError("KV page refcount would become negative")
        self.page_refcount[page_indices] -= 1
        released = page_indices[self.page_refcount[page_indices] == 0]
        if len(released) > 0:
            self.page_free[released] = True
            self.num_free_pages += int(len(released))

    def free_all(self):
        """Reset all pages to free."""
        self.page_free[:] = True
        self.page_refcount[:] = 0
        self.num_free_pages = self.num_pages

    # ------------------------------------------------------------------
    # Token table builder
    # ------------------------------------------------------------------
    def build_token_table(
        self, page_indices: torch.Tensor, num_tokens: int,
        token_table: torch.Tensor, req_idx: int,
    ) -> int:
        """Fill token_table[req_idx, :] with physical token positions.

        Each page covers `page_size` tokens. The table maps logical token
        positions to physical positions in the KV cache buffer.

        Example (page_size=4, 10 tokens, pages [3, 7, 12]):
          logical: [0,   1,   2,   3, | 4,   5,   6,   7, | 8,   9 ]
          page:    [-----------3------|-------7--------|----12---]
          physical:[12,  13,  14,  15,| 28,  29,  30,  31,| 48,  49]
          token_table[req, 0:10] = [12, 13, 14, 15, 28, 29, 30, 31, 48, 49]

        Returns the number of tokens actually mapped.
        """
        num_pages = len(page_indices)
        mapped = 0
        for p_idx, page_id in enumerate(page_indices):
            start = p_idx * self.page_size
            end = min(start + self.page_size, num_tokens)
            if start >= num_tokens:
                break
            block_len = end - start
            phys_start = int(page_id.item()) * self.page_size
            token_table[req_idx, start:end] = torch.arange(
                phys_start, phys_start + block_len,
                dtype=torch.int32, device=token_table.device,
            )
            mapped += block_len
        return mapped


# ---------------------------------------------------------------------------
# Integration helpers for existing ReqTokensManager
# ---------------------------------------------------------------------------
class PagedReqTokensManager:
    """Paged version of ReqTokensManager — manages request → page mappings."""

    def __init__(
        self, max_requests: int, max_seq_len: int,
        page_manager: PagedKVCacheManager,
        device: str = "npu:0",
    ):
        self.max_requests = max_requests
        self.max_seq_len = max_seq_len
        self.page_mgr = page_manager
        self.device = device

        # b_req_tokens_table: (max_requests, max_seq_len)
        # Maps each request's logical token positions to physical KV cache positions
        self.b_req_tokens_table = torch.zeros(
            (max_requests, max_seq_len), dtype=torch.int32, device=device,
        )

        # Per-request state
        self.req_page_table: dict[int, torch.Tensor] = {}  # req_idx → page indices
        self.req_token_count: dict[int, int] = {}           # req_idx → num tokens
        self.req_active = torch.zeros(max_requests, dtype=torch.bool, device="cpu")
        self.free_req_indices = list(range(max_requests))

    def alloc_req(
        self, req_idx: int, num_tokens: int, reserved_tokens: int | None = None
    ) -> bool:
        """Allocate KV cache for an explicit request id.

        ``num_tokens`` is the logical sequence length visible to attention.
        ``reserved_tokens`` is the physical KV capacity to reserve up front.
        Chunked prefill uses this to reserve the full prompt capacity while
        exposing only the already-replayed prefix length to the model.
        """
        if req_idx not in self.free_req_indices:
            return False
        num_tokens = int(num_tokens)
        capacity_tokens = int(reserved_tokens) if reserved_tokens is not None else num_tokens
        if num_tokens < 1 or capacity_tokens < num_tokens:
            return False
        if capacity_tokens > self.max_seq_len:
            return False

        pages = self.page_mgr.alloc(capacity_tokens)
        if pages is None:
            return False

        self.free_req_indices.remove(req_idx)
        self.req_page_table[req_idx] = pages
        self.req_token_count[req_idx] = num_tokens
        self.req_active[req_idx] = True

        self.page_mgr.build_token_table(
            pages, num_tokens, self.b_req_tokens_table, req_idx,
        )
        return True

    def reserve_req(
        self, num_tokens: int, reserved_tokens: int | None = None
    ) -> Optional[int]:
        """Allocate the lowest free request id and its initial KV pages."""
        if not self.free_req_indices:
            return None
        req_idx = self.free_req_indices[0]
        if not self.alloc_req(req_idx, num_tokens, reserved_tokens):
            return None
        return req_idx

    def ensure_req_capacity(self, req_idx: int, total_tokens: int) -> bool:
        """Ensure a request has enough physical pages for ``total_tokens``.

        This does not change the logical token count.  New logical token
        mappings are still appended by ``extend_req`` as replay/decode
        progresses.
        """
        if req_idx not in self.req_page_table:
            return False
        total_tokens = int(total_tokens)
        if total_tokens < self.req_token_count[req_idx]:
            return True
        if total_tokens > self.max_seq_len:
            return False

        required_pages = (
            total_tokens + self.page_mgr.page_size - 1
        ) // self.page_mgr.page_size
        existing = self.req_page_table[req_idx]
        current_pages = len(existing)
        if required_pages <= current_pages:
            return True

        additional_pages = required_pages - current_pages
        additional = self.page_mgr.alloc(additional_pages * self.page_mgr.page_size)
        if additional is None:
            return False
        self.req_page_table[req_idx] = torch.cat([existing, additional])
        return True

    def share_req_from_pages(
        self, req_idx: int, page_indices: torch.Tensor, num_tokens: int
    ) -> bool:
        """Map a request to existing pages and increment their refcounts."""
        if req_idx not in self.free_req_indices:
            return False
        num_tokens = int(num_tokens)
        if num_tokens < 1 or num_tokens > self.max_seq_len:
            return False
        required_pages = (num_tokens + self.page_mgr.page_size - 1) // self.page_mgr.page_size
        if len(page_indices) < required_pages:
            return False

        pages = page_indices[:required_pages].to(device="cpu", dtype=torch.long)
        self.page_mgr.add_ref(pages)
        self.free_req_indices.remove(req_idx)
        self.req_page_table[req_idx] = pages
        self.req_token_count[req_idx] = num_tokens
        self.req_active[req_idx] = True
        self.page_mgr.build_token_table(
            pages, num_tokens, self.b_req_tokens_table, req_idx,
        )
        return True

    def reserve_shared_req(
        self, page_indices: torch.Tensor, num_tokens: int
    ) -> Optional[int]:
        """Allocate a request id that shares existing KV pages."""
        if not self.free_req_indices:
            return None
        req_idx = self.free_req_indices[0]
        if not self.share_req_from_pages(req_idx, page_indices, num_tokens):
            return None
        return req_idx

    def batch_metadata(
        self, req_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build device metadata for a dynamic decode batch.

        The method only reads host-managed request dictionaries. It does not
        copy an NPU tensor back to Host, so it is safe to use between decode
        steps when the active request set changes.
        """
        if not req_indices:
            raise ValueError("req_indices must not be empty")
        for req_idx in req_indices:
            if req_idx not in self.req_token_count:
                raise KeyError(f"request {req_idx} is not allocated")

        req_ids = torch.tensor(
            req_indices, dtype=torch.int32, device=self.device
        )
        seq_lens = torch.tensor(
            [self.req_token_count[req_idx] for req_idx in req_indices],
            dtype=torch.long,
            device=self.device,
        )
        last_indices = torch.stack(
            [
                self.b_req_tokens_table[
                    req_idx, self.req_token_count[req_idx] - 1
                ]
                for req_idx in req_indices
            ]
        ).to(dtype=torch.int32, device=self.device)
        return req_ids, seq_lens, last_indices

    def extend_req(self, req_idx: int, num_new_tokens: int) -> bool:
        """Extend a request by `num_new_tokens` (decode step)."""
        if num_new_tokens <= 0:
            return True

        cur_tokens = self.req_token_count[req_idx]
        new_total = cur_tokens + num_new_tokens
        if new_total > self.max_seq_len:
            return False

        # Check if we need more pages.  Use the actual reserved page count
        # instead of the logical token count so chunked prefill can reserve
        # full prompt capacity up front while replaying it incrementally.
        new_pages = (new_total + self.page_mgr.page_size - 1) // self.page_mgr.page_size
        current_pages = len(self.req_page_table[req_idx])
        if new_pages > current_pages:
            additional = self.page_mgr.alloc(
                (new_pages - current_pages) * self.page_mgr.page_size
            )
            if additional is None:
                return False
            existing = self.req_page_table[req_idx]
            self.req_page_table[req_idx] = torch.cat([existing, additional])

        # Append only the new logical-to-physical mappings. Rebuilding the
        # complete table on every token makes decode host work grow with the
        # context length and repeatedly synchronizes page ids back to Python.
        logical_positions = torch.arange(cur_tokens, new_total, dtype=torch.long)
        page_slots = torch.div(
            logical_positions, self.page_mgr.page_size, rounding_mode="floor"
        )
        offsets = logical_positions.remainder(self.page_mgr.page_size)
        page_ids = self.req_page_table[req_idx][page_slots]
        physical_positions = page_ids * self.page_mgr.page_size + offsets
        self.b_req_tokens_table[req_idx, cur_tokens:new_total] = physical_positions.to(
            device=self.b_req_tokens_table.device,
            dtype=self.b_req_tokens_table.dtype,
        )
        self.req_token_count[req_idx] = new_total
        return True

    def get_token_indices(
        self, req_idx: int, num_tokens: Optional[int] = None
    ) -> torch.Tensor:
        """Return physical KV token positions for a request."""
        if num_tokens is None:
            num_tokens = self.req_token_count[req_idx]
        return self.b_req_tokens_table[req_idx, :num_tokens]

    def free_req(self, req_idx: int):
        """Free a request and its pages."""
        if req_idx in self.req_page_table:
            self.page_mgr.free(self.req_page_table[req_idx])
            del self.req_page_table[req_idx]
            del self.req_token_count[req_idx]
        self.req_active[req_idx] = False
        if req_idx not in self.free_req_indices:
            self.free_req_indices.append(req_idx)
            self.free_req_indices.sort()

    def request_pages(self, req_idx: int) -> Tuple[int, ...]:
        """Return Host page ids owned by a request."""
        if req_idx not in self.req_page_table:
            raise KeyError(f"request {req_idx} is not allocated")
        return tuple(int(page_id) for page_id in self.req_page_table[req_idx].tolist())
