"""Atlas 910B3 numerical checks for Qwen3 MoE GMM routing.

Run explicitly on the server:

ASCEND_RT_VISIBLE_DEVICES=0 python -m unittest \
  tests.npu.test_qwen3_moe_gmm -v
"""

import os
import unittest

import torch


try:
    import torch_npu

    HAS_NPU_GMM = (
        torch.npu.is_available()
        and hasattr(torch_npu, "npu_grouped_matmul")
    )
    HAS_NPU_GRAPH = (
        HAS_NPU_GMM
        and hasattr(torch.npu, "NPUGraph")
        and hasattr(torch.npu, "graph")
    )
except (ImportError, AttributeError):
    HAS_NPU_GMM = False
    HAS_NPU_GRAPH = False


@unittest.skipUnless(HAS_NPU_GMM, "requires torch_npu GMM on an NPU")
class Qwen3MoeGMMNPUTest(unittest.TestCase):
    def setUp(self):
        os.environ["LITE_LLAMA_MOE_BACKEND"] = "gmm"
        torch.npu.set_device(0)

    def tearDown(self):
        os.environ.pop("LITE_LLAMA_MOE_BACKEND", None)

    def test_grouped_experts_match_eager_local_output(self):
        from lite_llama.models.moe import Qwen3MoeExperts

        torch.manual_seed(17)
        experts = Qwen3MoeExperts(
            hidden_size=64,
            num_experts=8,
            intermediate_size=32,
            layer_index=3,
            dtype=torch.float16,
        ).npu()
        experts.gate_up_weight.data.normal_(mean=0.0, std=0.02)
        experts.down_weight.data.normal_(mean=0.0, std=0.02)
        hidden_states = torch.randn(
            19, 64, device="npu", dtype=torch.float16
        )
        selected_experts = torch.randint(
            0, 8, (19, 2), device="npu", dtype=torch.int64
        )
        routing_weights = torch.rand(
            19, 2, device="npu", dtype=torch.float16
        )
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        )

        with torch.no_grad():
            eager = experts._forward_eager_local(
                hidden_states, selected_experts, routing_weights
            )
            grouped = experts._forward_grouped_local(
                hidden_states, selected_experts, routing_weights
            )
        torch.npu.synchronize()

        torch.testing.assert_close(
            grouped, eager, rtol=1e-2, atol=1e-2
        )

    def test_validation_mode_checks_each_sparse_block(self):
        from lite_llama.models.moe import Qwen3SparseMoeBlock

        os.environ["LITE_LLAMA_MOE_VALIDATE"] = "1"
        try:
            block = Qwen3SparseMoeBlock(
                hidden_size=64,
                num_experts=8,
                top_k=2,
                intermediate_size=32,
                layer_index=5,
                dtype=torch.float16,
            ).npu()
            block.gate.weight.data.normal_(mean=0.0, std=0.02)
            block.experts.gate_up_weight.data.normal_(
                mean=0.0, std=0.02
            )
            block.experts.down_weight.data.normal_(mean=0.0, std=0.02)
            hidden_states = torch.randn(
                2, 7, 64, device="npu", dtype=torch.float16
            )

            with torch.no_grad():
                output = block(hidden_states)
            torch.npu.synchronize()

            self.assertEqual(output.shape, hidden_states.shape)
        finally:
            os.environ.pop("LITE_LLAMA_MOE_VALIDATE", None)

    def test_routed_gemv_matches_eager_local_output(self):
        from lite_llama.models.moe import Qwen3MoeExperts

        os.environ["LITE_LLAMA_MOE_BACKEND"] = "routed_gemv"
        try:
            torch.manual_seed(29)
            experts = Qwen3MoeExperts(
                hidden_size=64,
                num_experts=8,
                intermediate_size=32,
                layer_index=9,
                dtype=torch.float16,
            ).npu()
            experts.gate_up_weight.data.normal_(mean=0.0, std=0.02)
            experts.down_weight.data.normal_(mean=0.0, std=0.02)
            hidden_states = torch.randn(
                4, 64, device="npu", dtype=torch.float16
            )
            selected_experts = torch.randint(
                0, 8, (4, 2), device="npu", dtype=torch.int64
            )
            routing_weights = torch.rand(
                4, 2, device="npu", dtype=torch.float16
            )
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )

            with torch.no_grad():
                eager = experts._forward_eager_local(
                    hidden_states, selected_experts, routing_weights
                )
                routed = experts._forward_routed_local(
                    hidden_states, selected_experts, routing_weights
                )
            torch.npu.synchronize()

            torch.testing.assert_close(
                routed, eager, rtol=1e-2, atol=1e-2
            )
        finally:
            os.environ["LITE_LLAMA_MOE_BACKEND"] = "gmm"

    def test_routed_gemv_expert_parallel_computes_local_contribution(self):
        from lite_llama.executor.tp_utils import TPConfig
        from lite_llama.models.moe import Qwen3MoeExperts

        torch.manual_seed(37)
        global_gate_up = torch.randn(
            8, 64, 64, device="npu", dtype=torch.float16
        ) * 0.02
        global_down = torch.randn(
            8, 32, 64, device="npu", dtype=torch.float16
        ) * 0.02
        hidden_states = torch.randn(
            4, 64, device="npu", dtype=torch.float16
        )
        selected_experts = torch.tensor(
            [[0, 7], [2, 5], [6, 1], [4, 3]],
            device="npu",
            dtype=torch.int64,
        )
        routing_weights = torch.full(
            (4, 2), 0.5, device="npu", dtype=torch.float16
        )
        rank1 = Qwen3MoeExperts(
            hidden_size=64,
            num_experts=8,
            intermediate_size=32,
            tp_config=TPConfig(
                world_size=2,
                rank=1,
                moe_parallel_mode="ep",
            ),
            dtype=torch.float16,
        ).npu()
        rank1.gate_up_weight.data.copy_(global_gate_up[4:])
        rank1.down_weight.data.copy_(global_down[4:])

        with torch.no_grad():
            eager = rank1._forward_eager_local(
                hidden_states, selected_experts, routing_weights
            )
            routed = rank1._forward_routed_local(
                hidden_states, selected_experts, routing_weights
            )
        torch.npu.synchronize()

        torch.testing.assert_close(
            routed, eager, rtol=1e-2, atol=1e-2
        )

    @unittest.skipUnless(
        HAS_NPU_GRAPH, "requires torch_npu GMM and NPUGraph on an NPU"
    )
    def test_dynamic_routing_replays_inside_npu_graph(self):
        """Prove dynamic expert ids/group_list remain data, not graph shape."""
        from lite_llama.models.moe import Qwen3MoeExperts

        torch.manual_seed(23)
        experts = Qwen3MoeExperts(
            hidden_size=64,
            num_experts=8,
            intermediate_size=32,
            layer_index=7,
            dtype=torch.float16,
        ).npu()
        experts.gate_up_weight.data.normal_(mean=0.0, std=0.02)
        experts.down_weight.data.normal_(mean=0.0, std=0.02)

        static_hidden = torch.randn(
            4, 64, device="npu", dtype=torch.float16
        )
        static_experts = torch.tensor(
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            device="npu",
            dtype=torch.int64,
        )
        static_weights = torch.full(
            (4, 2), 0.5, device="npu", dtype=torch.float16
        )

        with torch.no_grad():
            _ = experts._forward_grouped_local(
                static_hidden, static_experts, static_weights
            )
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        pool = (
            torch.npu.graph_pool_handle()
            if hasattr(torch.npu, "graph_pool_handle")
            else None
        )
        with torch.no_grad(), torch.npu.graph(graph, pool=pool):
            graph_output = experts._forward_grouped_local(
                static_hidden, static_experts, static_weights
            )

        next_hidden = torch.randn_like(static_hidden)
        next_experts = torch.tensor(
            [[7, 6], [7, 5], [3, 1], [2, 0]],
            device="npu",
            dtype=torch.int64,
        )
        next_weights = torch.tensor(
            [[0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.55, 0.45]],
            device="npu",
            dtype=torch.float16,
        )
        static_hidden.copy_(next_hidden)
        static_experts.copy_(next_experts)
        static_weights.copy_(next_weights)
        graph.replay()
        torch.npu.synchronize()

        with torch.no_grad():
            eager_output = experts._forward_grouped_local(
                next_hidden, next_experts, next_weights
            )
        torch.testing.assert_close(
            graph_output, eager_output, rtol=1e-2, atol=1e-2
        )


if __name__ == "__main__":
    unittest.main()
