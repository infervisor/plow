import unittest
from types import SimpleNamespace

import numpy as np

from vllm_logit_oracle import dense_scores, repeat_metrics, suppression_metadata


class SuppressionTests(unittest.TestCase):
    def row(self, values, ids=None):
        return SimpleNamespace(token_ids=list(range(len(values))) if ids is None else ids, logprobs=values)

    def test_declared_negative_infinity_is_retained(self):
        row = dense_scores(self.row([1.0, -np.inf, 3.0]), 3, suppressed_ids=[1])
        self.assertTrue(np.isneginf(row[1]))
        self.assertEqual(row.shape, (3,))
        metrics = repeat_metrics(row, row, [1])
        self.assertEqual(metrics["full_row_centered_rel_l2"], 0.0)
        self.assertTrue(metrics["same_argmax"])

    def test_other_nonfinite_and_missing_are_rejected(self):
        for values, allowed in [([1.0, -np.inf], []), ([1.0, np.inf], [1]), ([1.0, np.nan], [1]), ([-np.inf, 1.0], [1])]:
            with self.subTest(values=values, allowed=allowed):
                with self.assertRaises(RuntimeError):
                    dense_scores(self.row(values), 2, suppressed_ids=allowed)
        with self.assertRaises(RuntimeError):
            dense_scores(self.row([1.0], [0]), 2, suppressed_ids=[1])

    def test_config_validation_and_finite_default(self):
        for ids in ([True], [-1], [2], [1.0], "1"):
            config = SimpleNamespace(try_get_generation_config=lambda: {"suppress_tokens": ids})
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                suppression_metadata(config, 2)
        config = SimpleNamespace(try_get_generation_config=lambda: {"suppress_tokens": [1, 1]})
        self.assertEqual(suppression_metadata(config, 2)["token_ids"], [1])
        np.testing.assert_array_equal(dense_scores(self.row([1.0, 2.0]), 2), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
