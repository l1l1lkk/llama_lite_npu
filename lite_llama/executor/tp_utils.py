"""Tensor Parallelism utilities for lite_llama.

Megatron-style column/row sharding of attention and FFN weights,
with HCCL (NPU) / NCCL (CUDA) communication primitives.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# TP configuration
# ---------------------------------------------------------------------------
@dataclass
class TPConfig:
    world_size: int = 1
    rank: int = 0
    backend: str = "hccl"  # "hccl" for NPU, "nccl" for CUDA

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_npu(self) -> bool:
        return self.backend == "hccl"


# ---------------------------------------------------------------------------
# Process group management
# ---------------------------------------------------------------------------
_TP_GROUP = None
_TP_CONFIG: Optional[TPConfig] = None


def get_tp_config() -> TPConfig:
    global _TP_CONFIG
    if _TP_CONFIG is None:
        _TP_CONFIG = TPConfig()
    return _TP_CONFIG


def init_tp(world_size: int, rank: int, backend: str = "hccl") -> TPConfig:
    """Initialize tensor parallelism process group."""
    global _TP_GROUP, _TP_CONFIG

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
        )

    _TP_GROUP = torch.distributed.new_group(
        ranks=list(range(world_size)),
        backend=backend,
    )
    _TP_CONFIG = TPConfig(world_size=world_size, rank=rank, backend=backend)

    if _TP_CONFIG.is_npu:
        torch.npu.set_device(rank)
    else:
        torch.cuda.set_device(rank)

    return _TP_CONFIG


def get_tp_group():
    return _TP_GROUP


# ---------------------------------------------------------------------------
# Communication primitives
# ---------------------------------------------------------------------------
def tp_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce across TP group (sum + broadcast)."""
    cfg = get_tp_config()
    if not cfg.enabled:
        return tensor
    torch.distributed.all_reduce(tensor, group=_TP_GROUP)
    return tensor


def tp_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather across TP group (collect slices)."""
    cfg = get_tp_config()
    if not cfg.enabled:
        return tensor
    chunks = [torch.empty_like(tensor) for _ in range(cfg.world_size)]
    torch.distributed.all_gather(chunks, tensor, group=_TP_GROUP)
    return torch.cat(chunks, dim=dim)


# ---------------------------------------------------------------------------
# Weight sharding helpers
# ---------------------------------------------------------------------------
def _shard_slice(total: int, world_size: int, rank: int) -> slice:
    """Return slice for this rank's portion of `total` items."""
    per_rank = total // world_size
    start = rank * per_rank
    end = start + per_rank
    return slice(start, end)


def shard_attention_q(
    weight: torch.Tensor, tp: TPConfig
) -> torch.Tensor:
    """Column-shard Q projection: (out=num_heads*hd, in=hidden)."""
    if not tp.enabled:
        return weight
    return weight[_shard_slice(weight.shape[0], tp.world_size, tp.rank)].clone()


def shard_attention_kv(
    weight: torch.Tensor, num_kv_heads: int, head_dim: int, tp: TPConfig
) -> torch.Tensor:
    """Shard fused KV projection: (2*num_kv_heads*hd, hidden).
    KV layout: [K_heads..., V_heads...] each of size (num_kv_heads*hd,).
    """
    if not tp.enabled:
        return weight
    kv_per_head = num_kv_heads * head_dim
    k_sl = _shard_slice(kv_per_head, tp.world_size, tp.rank)
    v_sl = _shard_slice(kv_per_head, tp.world_size, tp.rank)
    sl_k = slice(k_sl.start, k_sl.stop)
    sl_v = slice(kv_per_head + v_sl.start, kv_per_head + v_sl.stop)
    return torch.cat([weight[sl_k], weight[sl_v]], dim=0).clone()


def shard_attention_o(
    weight: torch.Tensor, tp: TPConfig
) -> torch.Tensor:
    """Row-shard O projection: (in=hidden, out=num_heads*hd)."""
    if not tp.enabled:
        return weight
    return weight[:, _shard_slice(weight.shape[1], tp.world_size, tp.rank)].clone()


def shard_ffn_gate_up(
    weight: torch.Tensor, tp: TPConfig
) -> torch.Tensor:
    """Column-shard gate/up projection: (out=intermediate, in=hidden)."""
    if not tp.enabled:
        return weight
    return weight[_shard_slice(weight.shape[0], tp.world_size, tp.rank)].clone()


def shard_ffn_down(
    weight: torch.Tensor, tp: TPConfig
) -> torch.Tensor:
    """Row-shard down projection: (in=hidden, out=intermediate)."""
    if not tp.enabled:
        return weight
    return weight[:, _shard_slice(weight.shape[1], tp.world_size, tp.rank)].clone()


def shard_moe_gate_up(
    weight: torch.Tensor, intermediate_size: int, tp: TPConfig
) -> torch.Tensor:
    """Shard stacked MoE gate/up weights without mixing their row ranges.

    ``weight`` uses ``[expert, gate_then_up, hidden]`` layout.
    """
    if not tp.enabled:
        return weight
    if weight.ndim != 3 or weight.shape[1] != 2 * intermediate_size:
        raise ValueError(
            "MoE gate/up weight must have shape "
            f"[experts, {2 * intermediate_size}, hidden], got {tuple(weight.shape)}"
        )
    gate_slice = _shard_slice(intermediate_size, tp.world_size, tp.rank)
    up_slice = _shard_slice(intermediate_size, tp.world_size, tp.rank)
    gate = weight[:, gate_slice, :]
    up = weight[:, intermediate_size + up_slice.start : intermediate_size + up_slice.stop, :]
    return torch.cat((gate, up), dim=1).clone()


def shard_moe_down(weight: torch.Tensor, tp: TPConfig) -> torch.Tensor:
    """Shard stacked MoE down weights on their intermediate/input axis."""
    if not tp.enabled:
        return weight
    if weight.ndim != 3:
        raise ValueError(
            f"MoE down weight must be three-dimensional, got {tuple(weight.shape)}"
        )
    return weight[:, :, _shard_slice(weight.shape[2], tp.world_size, tp.rank)].clone()


def shard_lm_head(
    weight: torch.Tensor, tp: TPConfig
) -> torch.Tensor:
    """Column-shard lm_head: (vocab, hidden)."""
    if not tp.enabled:
        return weight
    return weight[_shard_slice(weight.shape[0], tp.world_size, tp.rank)].clone()


# ---------------------------------------------------------------------------
# Utility: detect distributed environment (torchrun / mp.spawn)
# ---------------------------------------------------------------------------
def detect_tp_env() -> Optional[TPConfig]:
    """Auto-detect TP config from torchrun environment variables."""
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    world_size = os.environ.get("WORLD_SIZE")
    if rank is None or world_size is None:
        return None
    rank = int(rank)
    world_size = int(world_size)
    if world_size <= 1:
        return None

    # Detect backend: NPU uses "hccl", CUDA uses "nccl"
    backend = "hccl" if hasattr(torch, "npu") and torch.npu.is_available() else "nccl"
    return init_tp(world_size=world_size, rank=rank, backend=backend)
