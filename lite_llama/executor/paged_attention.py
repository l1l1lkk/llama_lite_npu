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

        # Free page pool: True = free, False = allocated
        self.page_free = torch.ones(num_pages, dtype=torch.bool, device="cpu")
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
        self.num_free_pages -= num_needed
        return free_indices

    def free(self, page_indices: torch.Tensor):
        """Return pages to the free pool."""
        self.page_free[page_indices] = True
        self.num_free_pages += len(page_indices)

    def free_all(self):
        """Reset all pages to free."""
        self.page_free[:] = True
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

    def alloc_req(self, req_idx: int, num_tokens: int) -> bool:
        """Allocate KV cache for an explicit request id."""
        if req_idx not in self.free_req_indices:
            return False
        pages = self.page_mgr.alloc(num_tokens)
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

    def extend_req(self, req_idx: int, num_new_tokens: int) -> bool:
        """Extend a request by `num_new_tokens` (decode step)."""
        if num_new_tokens <= 0:
            return True

        cur_tokens = self.req_token_count[req_idx]
        new_total = cur_tokens + num_new_tokens
        if new_total > self.max_seq_len:
            return False

        # Check if we need more pages
        cur_pages = (cur_tokens + self.page_mgr.page_size - 1) // self.page_mgr.page_size
        new_pages = (new_total + self.page_mgr.page_size - 1) // self.page_mgr.page_size
        if new_pages > cur_pages:
            additional = self.page_mgr.alloc((new_pages - cur_pages) * self.page_mgr.page_size)
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
        self.free_req_indices.append(req_idx)
        self.free_req_indices.sort()
