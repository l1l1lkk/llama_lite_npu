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
except (ImportError, AttributeError):
    HAS_NPU_GMM = False


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


if __name__ == "__main__":
    unittest.main()
