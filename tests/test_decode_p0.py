import importlib.util
import ast
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


paged_attention = _load_module(
    "decode_p0_paged_attention", "lite_llama/executor/paged_attention.py"
)
npu_graph = _load_module("decode_p0_npu_graph", "lite_llama/executor/npu_graph.py")


class PagedKvIncrementalUpdateTest(unittest.TestCase):
    def setUp(self):
        self.page_mgr = paged_attention.PagedKVCacheManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            num_pages=4,
            page_size=4,
            dtype=torch.float16,
            device="cpu",
        )
        self.req_mgr = paged_attention.PagedReqTokensManager(
            max_requests=1,
            max_seq_len=16,
            page_manager=self.page_mgr,
            device="cpu",
        )

    def test_extend_updates_only_new_mapping(self):
        self.assertTrue(self.req_mgr.alloc_req(0, 3))

        with patch.object(
            self.page_mgr,
            "build_token_table",
            side_effect=AssertionError("full token table rebuild"),
        ):
            self.assertTrue(self.req_mgr.extend_req(0, 1))

        self.assertEqual(self.req_mgr.req_token_count[0], 4)
        self.assertEqual(self.req_mgr.b_req_tokens_table[0, :4].tolist(), [0, 1, 2, 3])

    def test_extend_across_page_boundary_maps_new_page(self):
        self.assertTrue(self.req_mgr.alloc_req(0, 4))

        with patch.object(
            self.page_mgr,
            "build_token_table",
            side_effect=AssertionError("full token table rebuild"),
        ):
            self.assertTrue(self.req_mgr.extend_req(0, 1))

        self.assertEqual(len(self.req_mgr.req_page_table[0]), 2)
        self.assertEqual(self.req_mgr.b_req_tokens_table[0, 4].item(), 4)

    def test_reserve_req_assigns_and_reuses_request_id(self):
        first = self.req_mgr.reserve_req(3)
        self.assertEqual(first, 0)
        self.assertIsNone(self.req_mgr.reserve_req(1))

        self.req_mgr.free_req(first)
        reused = self.req_mgr.reserve_req(2)
        self.assertEqual(reused, 0)
        self.assertEqual(self.req_mgr.req_token_count[reused], 2)

    def test_batch_metadata_uses_each_request_length_and_last_token(self):
        page_mgr = paged_attention.PagedKVCacheManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            num_pages=8,
            page_size=4,
            dtype=torch.float16,
            device="cpu",
        )
        req_mgr = paged_attention.PagedReqTokensManager(
            max_requests=2,
            max_seq_len=16,
            page_manager=page_mgr,
            device="cpu",
        )
        self.assertTrue(req_mgr.alloc_req(0, 3))
        self.assertTrue(req_mgr.alloc_req(1, 5))

        req_ids, seq_lens, last_indices = req_mgr.batch_metadata([1, 0])

        self.assertEqual(req_ids.tolist(), [1, 0])
        self.assertEqual(seq_lens.tolist(), [5, 3])
        self.assertEqual(
            last_indices.tolist(),
            [
                req_mgr.b_req_tokens_table[1, 4].item(),
                req_mgr.b_req_tokens_table[0, 2].item(),
            ],
        )


class NpuGraphBucketTest(unittest.TestCase):
    def test_deepseek_moe_decode_graph_is_fail_closed(self):
        for model_type in (
            "deepseek_v2",
            "DeepSeek_V2",
            "deepseek_v3",
            "DeepSeek_V3",
        ):
            with self.subTest(model_type=model_type):
                self.assertFalse(npu_graph.supports_decode_graph(model_type))

    def test_qwen3_moe_can_attempt_decode_graph_capture(self):
        self.assertTrue(npu_graph.supports_decode_graph("qwen3"))
        self.assertTrue(npu_graph.supports_decode_graph("qwen3_moe"))

    def test_qwen3_moe_ep_disables_decode_graph_capture(self):
        self.assertFalse(
            npu_graph.supports_decode_graph(
                "qwen3_moe", moe_parallel_mode="ep"
            )
        )

    def test_qwen3_moe_tp_keeps_decode_graph_capture(self):
        self.assertTrue(
            npu_graph.supports_decode_graph(
                "qwen3_moe", moe_parallel_mode="tp"
            )
        )

    def test_sequence_lengths_share_partition_bucket(self):
        self.assertEqual(npu_graph.NpuGraphRunner.sequence_bucket(1), 128)
        self.assertEqual(npu_graph.NpuGraphRunner.sequence_bucket(127), 128)
        self.assertEqual(npu_graph.NpuGraphRunner.sequence_bucket(128), 128)
        self.assertEqual(npu_graph.NpuGraphRunner.sequence_bucket(129), 256)

    def test_graph_key_includes_batch_size_and_bucket(self):
        runner = npu_graph.NpuGraphRunner(model=None)
        input_ids = torch.zeros((4, 1), dtype=torch.long)

        self.assertEqual(runner.graph_key(input_ids, 65), (4, 128))
        self.assertEqual(runner.graph_key(input_ids, 120), (4, 128))
        self.assertEqual(runner.graph_key(input_ids, 129), (4, 256))

    def test_same_bucket_captures_once_and_replays_each_step(self):
        class FakeModel:
            def __init__(self):
                self.forward_count = 0

            def forward(self, input_ids, position_ids, atten_info):
                self.forward_count += 1
                return input_ids.float()

        class FakeGraph:
            def __init__(self):
                self.replay_count = 0

            def capture_begin(self):
                pass

            def capture_end(self):
                pass

            def replay(self):
                self.replay_count += 1

        fake_graph = FakeGraph()

        @contextmanager
        def graph_context(graph, pool=None):
            graph.capture_begin()
            try:
                yield
            finally:
                graph.capture_end()

        fake_npu = SimpleNamespace(
            set_option=lambda options: (_ for _ in ()).throw(
                AssertionError("NPUGraph must not depend on ACL_TASK_SINK")
            ),
            synchronize=lambda: None,
            NPUGraph=lambda: fake_graph,
            graph=graph_context,
            graph_pool_handle=lambda: "shared-pool",
        )
        model = FakeModel()
        runner = npu_graph.NpuGraphRunner(model)
        atten_info = SimpleNamespace(
            max_actual_seq_len=65,
            cur_select_index=torch.tensor([3], dtype=torch.int32),
            b_seq_len=torch.tensor([65], dtype=torch.int32),
            b_req_idx=torch.tensor([0], dtype=torch.int32),
        )

        with (
            patch.object(npu_graph, "_NPU_GRAPH_AVAILABLE", True),
            patch.object(npu_graph.torch, "npu", fake_npu, create=True),
        ):
            runner(
                torch.tensor([[1]]),
                torch.tensor([[64]]),
                atten_info,
            )
            atten_info.max_actual_seq_len = 66
            atten_info.b_seq_len.fill_(66)
            atten_info.cur_select_index.fill_(4)
            runner(
                torch.tensor([[2]]),
                torch.tensor([[65]]),
                atten_info,
            )

        self.assertEqual(runner.capture_count, 1)
        self.assertEqual(runner.replay_count, 2)
        self.assertEqual(model.forward_count, 2)
        self.assertEqual(fake_graph.replay_count, 2)
        captured = runner._graphs[(1, 128)]
        self.assertEqual(captured.b_seq_len.item(), 66)
        self.assertEqual(captured.cur_select_index.item(), 4)

    def test_failed_graph_key_is_not_captured_repeatedly(self):
        class FailingModel:
            def __init__(self):
                self.forward_count = 0

            def forward(self, input_ids, position_ids, atten_info):
                self.forward_count += 1
                raise RuntimeError("unsupported graph operator")

        @contextmanager
        def unused_graph_context(graph, pool=None):
            yield

        fake_npu = SimpleNamespace(
            synchronize=lambda: None,
            NPUGraph=lambda: object(),
            graph=unused_graph_context,
            graph_pool_handle=lambda: "shared-pool",
        )
        model = FailingModel()
        runner = npu_graph.NpuGraphRunner(model, model_type="qwen3_moe")
        atten_info = SimpleNamespace(
            max_actual_seq_len=65,
            cur_select_index=torch.tensor([3], dtype=torch.int32),
            b_seq_len=torch.tensor([65], dtype=torch.int32),
            b_req_idx=torch.tensor([0], dtype=torch.int32),
        )

        with (
            patch.object(npu_graph, "_NPU_GRAPH_AVAILABLE", True),
            patch.object(npu_graph.torch, "npu", fake_npu, create=True),
        ):
            for token in (1, 2):
                with self.assertRaisesRegex(
                    RuntimeError, "unsupported graph operator"
                ):
                    runner(
                        torch.tensor([[token]]),
                        torch.tensor([[64]]),
                        atten_info,
                    )

        self.assertEqual(model.forward_count, 3)
        self.assertEqual(runner.capture_attempt_count, 1)
        self.assertEqual(runner.fallback_count, 2)


class DecodeHostSyncContractTest(unittest.TestCase):
    def test_paged_decode_does_not_convert_request_tensor_to_host_list(self):
        source = (
            ROOT / "lite_llama/executor/model_executor.py"
        ).read_text(encoding="utf-8")
        module = ast.parse(source)
        decode_method = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef)
            and node.name == "decode_alloc_kv_cache"
        )
        method_source = ast.get_source_segment(source, decode_method)

        self.assertNotIn(".tolist(", method_source)
        self.assertIn("_paged_request_ids", method_source)

    def test_stream_decode_reuses_token_copy_for_eos_and_request_cleanup(self):
        source = (
            ROOT / "lite_llama/generate_stream.py"
        ).read_text(encoding="utf-8")
        module = ast.parse(source)
        generate_method = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef)
            and node.name == "generate_stream"
        )
        method_source = ast.get_source_segment(source, generate_method)

        self.assertNotIn("if eos_reached.all()", method_source)
        self.assertNotIn("b_req_idx.tolist()", method_source)
        self.assertIn("release_paged_requests", method_source)


if __name__ == "__main__":
    unittest.main()
