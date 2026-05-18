"""NPU Graph support for decode-phase kernel launch batching.

Ascend NPU supports graph-like execution via the "task sink" mode
(ACL_TASK_SINK) which batches kernel launches at the driver level.

If task_sink is unavailable, falls back to normal execution.

Usage:
  from .npu_graph import NpuGraphRunner
  runner = NpuGraphRunner(model)
  runner.capture(batch_size, atten_info_template)
  # In decode loop:
  logits = runner.replay(input_ids, position_ids, atten_info)
"""

from __future__ import annotations

import torch
from typing import Optional

try:
    _NPU_GRAPH_AVAILABLE = hasattr(torch.npu, "set_option")
except Exception:
    _NPU_GRAPH_AVAILABLE = False


class NpuGraphRunner:
    """NPU graph runner for decode phase.

    Captures the full decode forward pass into a graph, replaying
    with updated inputs each step. This eliminates per-kernel launch
    overhead (~128 kernel launches per layer × 64 layers).
    """

    def __init__(self, model):
        self.model = model
        self._captured = False
        self._graph = None
        self._graph_inputs: dict = {}
        self._graph_output = None
        self._batch_size = 0

    @property
    def available(self) -> bool:
        return _NPU_GRAPH_AVAILABLE

    def capture(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info,
    ) -> bool:
        """Capture the decode forward pass. Returns True if successful."""
        if not _NPU_GRAPH_AVAILABLE:
            return False

        self._batch_size = input_ids.shape[0]

        try:
            # Enable task sink mode (NPU kernel launch batching)
            torch.npu.set_option({"ACL_TASK_SINK": "1"})

            # Warm up
            _ = self.model.forward(input_ids, position_ids, atten_info)
            torch.npu.synchronize()

            # Capture
            self._graph = torch.npu.graph()
            self._graph.capture_begin()
            self._graph_output = self.model.forward(
                input_ids, position_ids, atten_info,
            )
            self._graph.capture_end()

            # Store mutable input references for later .copy_()
            self._graph_inputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "cur_select_index": atten_info.cur_select_index,
                "b_req_idx": getattr(atten_info, "b_req_idx", None),
            }
            self._captured = True
            return True

        except Exception:
            # Graph capture failed — disable and fall through
            self._captured = False
            self._graph = None
            return False

    def replay(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info,
    ) -> Optional[torch.Tensor]:
        """Replay captured graph with new inputs."""
        if not self._captured or self._graph is None:
            return None

        # Update mutable inputs in-place
        self._graph_inputs["input_ids"].copy_(input_ids)
        self._graph_inputs["position_ids"].copy_(position_ids)

        if self._graph_inputs["cur_select_index"] is not None:
            self._graph_inputs["cur_select_index"].copy_(atten_info.cur_select_index)
        if self._graph_inputs["b_req_idx"] is not None:
            self._graph_inputs["b_req_idx"].copy_(atten_info.b_req_idx)

        # Replay
        self._graph.replay()
        torch.npu.synchronize()
        return self._graph_output

    def __call__(self, input_ids, position_ids, atten_info):
        result = self.replay(input_ids, position_ids, atten_info)
        if result is None:
            return self.model.forward(input_ids, position_ids, atten_info)
        return result
