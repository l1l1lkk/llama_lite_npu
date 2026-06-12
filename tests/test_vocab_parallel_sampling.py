import unittest
import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "lite_llama" / "sampling.py"
)


def load_sampling_module():
    spec = importlib.util.spec_from_file_location(
        "lite_llama_sampling_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalSamplingTest(unittest.TestCase):
    def test_greedy_returns_highest_token(self):
        sample_local_logits = load_sampling_module().sample_local_logits
        logits = torch.tensor([[1.0, 5.0, 3.0]])

        token = sample_local_logits(logits, temperature=0.0, top_p=1.0)

        self.assertEqual(token.tolist(), [1])

    def test_top_p_with_single_surviving_token_is_deterministic(self):
        sample_local_logits = load_sampling_module().sample_local_logits
        logits = torch.tensor([[10.0, 0.0, -1.0]])

        token = sample_local_logits(logits, temperature=1.0, top_p=0.5)

        self.assertEqual(token.tolist(), [0])


class VocabularyParallelHelperTest(unittest.TestCase):
    def test_greedy_selects_global_id_across_vocab_shards(self):
        select_greedy_from_shards = (
            load_sampling_module().select_greedy_from_shards
        )
        gathered_values = torch.tensor(
            [
                [3.0, 9.0],
                [7.0, 4.0],
            ]
        )
        gathered_ids = torch.tensor(
            [
                [2, 5],
                [11, 14],
            ]
        )

        tokens = select_greedy_from_shards(
            gathered_values, gathered_ids
        )

        self.assertEqual(tokens.tolist(), [11, 5])

    def test_candidate_nucleus_is_complete_above_excluded_boundary(self):
        candidate_nucleus_is_complete = (
            load_sampling_module().candidate_nucleus_is_complete
        )
        sorted_logits = torch.tensor([8.0, 7.0, 6.0])
        exact_probabilities = torch.tensor([0.55, 0.30, 0.10])

        complete = candidate_nucleus_is_complete(
            sorted_logits=sorted_logits,
            exact_probabilities=exact_probabilities,
            top_p=0.8,
            max_excluded_logit=torch.tensor(5.5),
        )

        self.assertTrue(complete)

    def test_candidate_nucleus_requires_fallback_when_excluded_token_can_enter(self):
        candidate_nucleus_is_complete = (
            load_sampling_module().candidate_nucleus_is_complete
        )
        sorted_logits = torch.tensor([8.0, 5.0, 4.0])
        exact_probabilities = torch.tensor([0.50, 0.20, 0.15])

        complete = candidate_nucleus_is_complete(
            sorted_logits=sorted_logits,
            exact_probabilities=exact_probabilities,
            top_p=0.7,
            max_excluded_logit=torch.tensor(6.0),
        )

        self.assertFalse(complete)

    def test_candidate_nucleus_requires_fallback_when_mass_is_too_small(self):
        candidate_nucleus_is_complete = (
            load_sampling_module().candidate_nucleus_is_complete
        )
        sorted_logits = torch.tensor([8.0, 7.0])
        exact_probabilities = torch.tensor([0.40, 0.30])

        complete = candidate_nucleus_is_complete(
            sorted_logits=sorted_logits,
            exact_probabilities=exact_probabilities,
            top_p=0.9,
            max_excluded_logit=torch.tensor(1.0),
        )

        self.assertFalse(complete)


if __name__ == "__main__":
    unittest.main()
