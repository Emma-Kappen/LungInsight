from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

from stage04_ranking import (  # noqa: E402
    PairwiseLinearRanker,
    build_pairwise_training_data,
    pairwise_logistic_loss,
    ranking_metrics,
)


class Stage04RankingTests(unittest.TestCase):
    def test_ties_are_excluded_and_direction_is_explicit(self):
        x = np.asarray([[3.0], [2.0], [1.0], [0.0]])
        y = np.asarray([3.0, 1.0, 1.0, 5.0])
        patients = np.asarray(["p", "p", "p", "p"])
        pairs = build_pairwise_training_data(x, y, patients)
        self.assertEqual(len(pairs.left), 5)
        self.assertTrue(np.all(pairs.target == 1.0))
        self.assertTrue(np.all(pairs.left[:, 0] != pairs.right[:, 0]))

    def test_pairwise_loss_is_lower_for_correct_order(self):
        self.assertLess(pairwise_logistic_loss(np.asarray([2.0, 1.0])), pairwise_logistic_loss(np.asarray([-2.0, -1.0])))

    def test_ranker_is_deterministic_and_keeps_patient_groups_separate(self):
        x = np.asarray([[0.0], [1.0], [2.0], [3.0], [0.0], [1.0]])
        y = np.asarray([1.0, 2.0, 3.0, 4.0, 2.0, 1.0])
        patients = np.asarray(["a", "a", "a", "a", "b", "b"])
        first = PairwiseLinearRanker(epochs=100).fit(x, y, patients)
        second = PairwiseLinearRanker(epochs=100).fit(x, y, patients)
        np.testing.assert_allclose(first.predict(x), second.predict(x))
        metrics = ranking_metrics(y, first.predict(x), patients)
        self.assertEqual(metrics["n_ordered_within_patient_pairs"], 7)
        self.assertGreater(metrics["pairwise_accuracy"], 0.5)

    def test_missing_or_nonfinite_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            build_pairwise_training_data(np.asarray([[1.0]]), np.asarray([np.nan]), np.asarray(["p"]))
        with self.assertRaises(ValueError):
            pairwise_logistic_loss(np.asarray([1.0]), np.asarray([0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
