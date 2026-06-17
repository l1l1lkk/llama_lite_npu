import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "lite_llama"
    / "executor"
    / "paged_attention.py"
)


def load_paged_module():
    spec = importlib.util.spec_from_file_location(
        "lite_llama_paged_attention_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PagedKVRefcountTest(unittest.TestCase):
    def test_shared_page_is_not_returned_to_free_pool_until_last_ref(self):
        module = load_paged_module()
        manager = module.PagedKVCacheManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=8,
            num_pages=4,
            page_size=4,
            device="cpu",
        )

        pages = manager.alloc(4)
        self.assertEqual(manager.num_free_pages, 3)
        manager.add_ref(pages)
        self.assertEqual(int(manager.page_refcount[pages[0]]), 2)

        manager.free(pages)
        self.assertEqual(manager.num_free_pages, 3)
        self.assertFalse(bool(manager.page_free[pages[0]]))

        manager.free(pages)
        self.assertEqual(manager.num_free_pages, 4)
        self.assertTrue(bool(manager.page_free[pages[0]]))

    def test_request_pages_returns_host_page_ids(self):
        module = load_paged_module()
        page_manager = module.PagedKVCacheManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=8,
            num_pages=4,
            page_size=4,
            device="cpu",
        )
        req_manager = module.PagedReqTokensManager(
            max_requests=2,
            max_seq_len=16,
            page_manager=page_manager,
            device="cpu",
        )

        self.assertTrue(req_manager.alloc_req(0, 5))
        pages = req_manager.request_pages(0)

        self.assertEqual(len(pages), 2)
        self.assertTrue(all(isinstance(page, int) for page in pages))


if __name__ == "__main__":
    unittest.main()
