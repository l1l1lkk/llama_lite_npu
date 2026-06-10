"""NPU Graph support for fixed-shape decode replay.

FlashDecoding launches work in 128-token partitions. Decode graphs are cached
per ``(batch_size, sequence_bucket)`` so all sequence lengths inside the same
partition bucket can reuse one graph while the real lengths remain dynamic.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional

import torch


try:
    _NPU_GRAPH_AVAILABLE = (
        hasattr(torch.npu, "NPUGraph")
        and hasattr(torch.npu, "graph")
    )
except Exception:
    _NPU_GRAPH_AVAILABLE = False


logger = logging.getLogger(__name__)


def supports_decode_graph(model_type: str) -> bool:
    """Return whether the model has a static decode execution topology."""
    return model_type.lower() != "qwen3_moe"


@dataclass
class _CapturedGraph:
    graph: object
    output: torch.Tensor
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    cur_select_index: Optional[torch.Tensor]
    b_seq_len: Optional[torch.Tensor]
    b_req_idx: Optional[torch.Tensor]


class NpuGraphRunner:
    """Cache and replay decode graphs using FlashDecoding length buckets."""

    PARTITION_SIZE = 128

    def __init__(self, model):
        self.model = model
        self._graphs: dict[tuple[int, int], _CapturedGraph] = {}
        self._failed_keys: set[tuple[int, int]] = set()
        self._pool = (
            torch.npu.graph_pool_handle()
            if _NPU_GRAPH_AVAILABLE and hasattr(torch.npu, "graph_pool_handle")
            else None
        )
        self.capture_count = 0
        self.replay_count = 0
        self.fallback_count = 0

    @property
    def available(self) -> bool:
        return _NPU_GRAPH_AVAILABLE

    @property
    def captured(self) -> bool:
        return bool(self._graphs)

    @classmethod
    def sequence_bucket(cls, sequence_length: int) -> int:
        sequence_length = max(1, int(sequence_length))
        return (
            (sequence_length + cls.PARTITION_SIZE - 1) // cls.PARTITION_SIZE
        ) * cls.PARTITION_SIZE

    def graph_key(
        self, input_ids: torch.Tensor, sequence_length: int
    ) -> tuple[int, int]:
        return input_ids.shape[0], self.sequence_bucket(sequence_length)

    def capture(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info,
    ) -> bool:
        """Capture one graph for the current batch and sequence bucket."""
        if not _NPU_GRAPH_AVAILABLE:
            return False

        key = self.graph_key(input_ids, atten_info.max_actual_seq_len)
        if key in self._graphs:
            return True
        if key in self._failed_keys:
            return False

        try:
            # Graph capture requires stable storage. AttentionInfo is mutated by
            # KV allocation after every decode step, so retain independent
            # dynamic tensors while sharing the large KV and request tables.
            static_input_ids = input_ids.clone()
            static_position_ids = position_ids.clone()
            static_atten_info = copy.copy(atten_info)
            static_atten_info.cur_select_index = atten_info.cur_select_index.clone()
            static_atten_info.b_seq_len = atten_info.b_seq_len.clone()
            static_atten_info.b_req_idx = (
                atten_info.b_req_idx.clone()
                if getattr(atten_info, "b_req_idx", None) is not None
                else None
            )
            static_atten_info.max_actual_seq_len = key[1]

            _ = self.model.forward(
                static_input_ids, static_position_ids, static_atten_info
            )
            torch.npu.synchronize()

            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, pool=self._pool):
                graph_output = self.model.forward(
                    static_input_ids, static_position_ids, static_atten_info
                )

            self._graphs[key] = _CapturedGraph(
                graph=graph,
                output=graph_output,
                input_ids=static_input_ids,
                position_ids=static_position_ids,
                cur_select_index=static_atten_info.cur_select_index,
                b_seq_len=static_atten_info.b_seq_len,
                b_req_idx=static_atten_info.b_req_idx,
            )
            self.capture_count += 1
            logger.info("Captured NPU decode graph for batch=%d bucket=%d", *key)
            return True
        except Exception as exc:
            # A failed graph shape must not be captured again on every token.
            self._failed_keys.add(key)
            logger.warning(
                "NPU graph capture failed for batch=%d bucket=%d; "
                "using eager execution: %s",
                key[0],
                key[1],
                exc,
            )
            return False

    def replay(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info,
    ) -> Optional[torch.Tensor]:
        """Replay a captured bucket after updating its dynamic inputs."""
        key = self.graph_key(input_ids, atten_info.max_actual_seq_len)
        captured = self._graphs.get(key)
        if captured is None:
            return None

        captured.input_ids.copy_(input_ids)
        captured.position_ids.copy_(position_ids)
        if captured.cur_select_index is not None:
            captured.cur_select_index.copy_(atten_info.cur_select_index)
        if captured.b_seq_len is not None:
            captured.b_seq_len.copy_(atten_info.b_seq_len)
        if captured.b_req_idx is not None:
            captured.b_req_idx.copy_(atten_info.b_req_idx)

        captured.graph.replay()
        self.replay_count += 1
        return captured.output

    def __call__(self, input_ids, position_ids, atten_info):
        if not self.available:
            self.fallback_count += 1
            return self.model.forward(input_ids, position_ids, atten_info)

        key = self.graph_key(input_ids, atten_info.max_actual_seq_len)
        if key not in self._graphs and key not in self._failed_keys:
            self.capture(input_ids, position_ids, atten_info)

        result = self.replay(input_ids, position_ids, atten_info)
        if result is None:
            self.fallback_count += 1
            return self.model.forward(input_ids, position_ids, atten_info)
        return result
