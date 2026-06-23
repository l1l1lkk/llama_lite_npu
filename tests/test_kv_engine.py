import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "lite_llama" / "kv_engine.py"


def load_kv_module():
    spec = importlib.util.spec_from_file_location("lite_llama_kv_engine_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class KVBlockRefCounterTest(unittest.TestCase):
    def test_acquire_and_release_updates_refcounts(self):
        refs = load_kv_module().KVBlockRefCounter(total_blocks=4)

        blocks = refs.acquire(2)
        self.assertEqual(blocks, (0, 1))
        self.assertEqual(refs.refcount(0), 1)
        self.assertEqual(refs.available_count, 2)

        refs.add_ref((0,))
        self.assertEqual(refs.refcount(0), 2)
        refs.release((0, 1))
        self.assertEqual(refs.refcount(0), 1)
        self.assertEqual(refs.refcount(1), 0)
        self.assertEqual(refs.available_count, 3)

    def test_release_rejects_negative_refcount(self):
        refs = load_kv_module().KVBlockRefCounter(total_blocks=1)
        with self.assertRaises(ValueError):
            refs.release((0,))


class PrefixCacheTest(unittest.TestCase):
    def test_matches_longest_block_aligned_prefix(self):
        PrefixCache = load_kv_module().PrefixCache
        cache = PrefixCache(block_size=4)
        cache.put((1, 2, 3, 4, 5, 6, 7, 8), (10, 11))

        match = cache.match((1, 2, 3, 4, 5, 6, 9, 9))

        self.assertEqual(match.matched_tokens, 4)
        self.assertEqual(match.block_ids, (10,))

    def test_ignores_partial_blocks(self):
        PrefixCache = load_kv_module().PrefixCache
        cache = PrefixCache(block_size=4)
        cache.put((1, 2, 3), (10,))

        match = cache.match((1, 2, 3, 4))

        self.assertEqual(match.matched_tokens, 0)
        self.assertEqual(match.block_ids, ())


class ChunkedPrefillPlannerTest(unittest.TestCase):
    def test_splits_prompt_into_fixed_size_chunks(self):
        planner = load_kv_module().ChunkedPrefillPlanner(chunk_size=4)

        chunks = planner.plan(prompt_length=10)

        self.assertEqual([(c.start, c.end) for c in chunks], [(0, 4), (4, 8), (8, 10)])

    def test_rejects_non_positive_chunk_size(self):
        with self.assertRaises(ValueError):
            load_kv_module().ChunkedPrefillPlanner(chunk_size=0)


class MixedLengthPrefillPackerTest(unittest.TestCase):
    def test_packs_mixed_length_prompts_without_padding(self):
        packer = load_kv_module().MixedLengthPrefillPacker()

        plan = packer.pack(((10, 11), (20,), (30, 31, 32)))

        self.assertEqual(plan.flat_token_ids, (10, 11, 20, 30, 31, 32))
        self.assertEqual(plan.flat_position_ids, (0, 1, 0, 0, 1, 2))
        self.assertEqual(
            [(row.request_index, row.start, row.length, row.end) for row in plan.requests],
            [(0, 0, 2, 2), (1, 2, 1, 3), (2, 3, 3, 6)],
        )
        self.assertEqual(plan.total_tokens, 6)


if __name__ == "__main__":
    unittest.main()
