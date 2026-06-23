import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "lite_llama"
    / "executor"
    / "model_executor.py"
)


class ModelExecutorPackedPrefillContractTest(unittest.TestCase):
    def test_model_executor_exposes_packed_prefill_attention_metadata_path(self):
        source = MODULE_PATH.read_text(encoding="utf-8")

        self.assertIn("def activate_paged_packed_prefill_batch", source)
        self.assertIn("self.atten_info.b_start_loc", source)
        self.assertIn("self.atten_info.b_seq_len", source)
        self.assertIn("self.atten_info.cur_select_index", source)
        self.assertIn("sample_indices", source)
        self.assertIn("position_ids", source)

    def test_model_executor_exposes_paged_chunk_prefill_metadata_path(self):
        source = MODULE_PATH.read_text(encoding="utf-8")

        self.assertIn("def activate_paged_chunk_prefill_batch", source)
        self.assertIn("is_paged_chunk_prefill", source)
        self.assertIn("chunk_context_len", source)
        self.assertIn("chunk_q_seq_len", source)
        self.assertIn("max_actual_q_seq_len", source)

    def test_safe_prefill_helper_does_not_split_constructor_initialization(self):
        source = MODULE_PATH.read_text(encoding="utf-8")

        helper_index = source.index("def _infer_safe_prefill_tokens")
        prefix_cache_index = source.index("self._paged_prefix_cache")
        request_manager_index = source.index("self.req_tokens_manager =")

        self.assertGreater(helper_index, prefix_cache_index)
        self.assertGreater(helper_index, request_manager_index)


class PagedChunkFlashAttentionContractTest(unittest.TestCase):
    def test_paged_chunk_flash_attention_uses_910b3_safe_tiles_and_fallback(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "lite_llama"
            / "kernels"
            / "paged_chunk_flashattention.py"
        ).read_text(encoding="utf-8")

        self.assertIn("block_m_size = 16", source)
        self.assertIn("block_n_size = 32", source)
        self.assertIn("_paged_chunk_attention_torch_fallback", source)
        self.assertIn("except Exception as error", source)
        self.assertIn("falling back to torch attention", source)


if __name__ == "__main__":
    unittest.main()
