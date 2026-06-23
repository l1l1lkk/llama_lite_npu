import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "benchmark_prefix_cache.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_prefix_cache_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BenchmarkPrefixCacheTest(unittest.TestCase):
    def test_parse_stream_chunks_tracks_first_token_text_and_usage(self):
        module = load_module()
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14}}\n\n',
            b"data: [DONE]\n\n",
        ]

        parsed = list(module.iter_sse_payloads(chunks))

        self.assertEqual(parsed[0]["choices"][0]["delta"]["content"], "Hello")
        self.assertEqual(parsed[2]["usage"]["completion_tokens"], 2)

    def test_percentile_uses_nearest_rank(self):
        module = load_module()

        self.assertEqual(module.percentile([1, 2, 3, 4], 50), 2)
        self.assertEqual(module.percentile([1, 2, 3, 4], 90), 4)

    def test_same_dataset_reuses_exact_prompt(self):
        module = load_module()

        prompts = [
            module.build_prompt(dataset="same", index=i, prompt_len=8, same_prompt="fixed")
            for i in range(3)
        ]

        self.assertEqual(prompts, ["fixed", "fixed", "fixed"])

    def test_random_dataset_changes_prompt(self):
        module = load_module()

        prompts = [
            module.build_prompt(dataset="random", index=i, prompt_len=8, same_prompt="fixed")
            for i in range(3)
        ]

        self.assertEqual(len(set(prompts)), 3)


if __name__ == "__main__":
    unittest.main()
