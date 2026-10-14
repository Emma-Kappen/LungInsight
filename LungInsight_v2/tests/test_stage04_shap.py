from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

IMAGING_ROOT = Path(__file__).resolve().parents[1] / "backend" / "Imaging"
sys.path.insert(0, str(IMAGING_ROOT))
from stage04_shap import explain_regression, load_training_background  # noqa: E402


class Stage04ShapTests(unittest.TestCase):
    def test_background_uses_only_deterministic_training_sample(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            table_path = Path(temporary_directory) / "training.csv"
            with table_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["split", "f1", "f2", "target"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"split": "train", "f1": index, "f2": index + 1, "target": 1}
                        for index in range(8)
                    ]
                    + [{"split": "validation", "f1": 999, "f2": 999, "target": 5}]
                )
            first, metadata = load_training_background(
                table_path,
                ["f1", "f2"],
                split_column="split",
                training_split_value="train",
                sample_size=4,
                random_seed=17,
            )
            second, _ = load_training_background(
                table_path,
                ["f1", "f2"],
                split_column="split",
                training_split_value="train",
                sample_size=4,
                random_seed=17,
            )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(metadata["training_row_count"], 8)
        self.assertEqual(metadata["sample_size"], 4)
        self.assertTrue(np.all(first < 999))

    def test_tree_shap_explains_regressor_raw_predictions_additively(self):
        training_x = np.asarray(
            [[1, 3], [2, 2], [3, 1], [4, 4], [5, 2], [2, 5]],
            dtype=np.float64,
        )
        training_y = np.asarray([2, 2.5, 3, 4, 4.5, 3.5], dtype=np.float64)
        model = ExtraTreesRegressor(n_estimators=8, min_samples_leaf=1, random_state=3)
        model.fit(training_x, training_y)
        explained_x = training_x[:2]
        values, base_values, metadata = explain_regression(
            model,
            training_x,
            explained_x,
            additivity_tolerance=1e-5,
        )
        np.testing.assert_allclose(
            base_values + values.sum(axis=1),
            model.predict(explained_x),
            atol=1e-5,
        )
        self.assertEqual(values.shape, explained_x.shape)
        self.assertEqual(metadata["explainer"], "shap.TreeExplainer")
        self.assertLessEqual(metadata["max_abs_additivity_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
