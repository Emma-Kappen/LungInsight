from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

IMAGING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = IMAGING_DIR.parents[1]
MALIGNANCY_DIR = PROJECT_ROOT / "backend" / "models" / "malignancy"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(IMAGING_DIR))
sys.path.insert(0, str(MALIGNANCY_DIR))

from frozen_attribute_common import CHECKPOINT_GROUPS, FEATURE_NAMES, stage03_prediction_details
from backend.models.nodule_ranker_3d_distributional_v1.spatial_v1.model_objective import (
    MODEL_TARGETS,
    ORDERED_TARGETS,
    RATING_MAX,
    SpatialObjectiveModel,
)

STAGE03_SCRIPT = IMAGING_DIR / "stages" / "03_nodule_classification.py"
STAGE03_SPEC = importlib.util.spec_from_file_location("stage03_prediction_details_test", STAGE03_SCRIPT)
if STAGE03_SPEC is None or STAGE03_SPEC.loader is None:
    raise RuntimeError("Could not import the Stage 03 implementation.")
stage03 = importlib.util.module_from_spec(STAGE03_SPEC)
sys.modules[STAGE03_SPEC.name] = stage03
STAGE03_SPEC.loader.exec_module(stage03)


class StubOrdinalModel(torch.nn.Module):
    objective = "ordinal_reader"

    def forward(self, _values):
        outputs = []
        for name in MODEL_TARGETS:
            if name in ORDERED_TARGETS:
                outputs.append(torch.tensor([[2.0, 1.0, -1.0, -2.0]]))
            else:
                outputs.append(torch.zeros((1, RATING_MAX[name])))
        return tuple(outputs)

    def expected_ratings(self, outputs):
        return SpatialObjectiveModel.expected_ratings(self, outputs)

    def ordered_probabilities(self, logits):
        return SpatialObjectiveModel.ordered_probabilities(self, logits)


class Stage03PredictionDetailsTests(unittest.TestCase):
    def test_ordinal_distribution_and_derived_uncertainty_are_reported_as_uncalibrated(self):
        models = {group: StubOrdinalModel() for group in CHECKPOINT_GROUPS}
        predictions, distributions, attention = stage03_prediction_details(
            np.zeros((64, 64, 64), dtype=np.float32),
            models,
            torch.device("cpu"),
        )
        self.assertEqual(set(predictions), set(FEATURE_NAMES))
        self.assertEqual(set(distributions), set(FEATURE_NAMES))
        self.assertEqual(attention, {})
        for name, result in distributions.items():
            probabilities = np.asarray(result["probabilities"], dtype=np.float64)
            ratings = np.asarray(result["rating_values"], dtype=np.float64)
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
            self.assertAlmostEqual(float(np.dot(probabilities, ratings)), predictions[name], places=5)
            self.assertGreaterEqual(result["entropy_bits"], 0.0)
            self.assertGreaterEqual(result["rating_standard_deviation"], 0.0)
            self.assertIsNone(result["calibrated_confidence"])
            self.assertIn("not_calibrated_confidence", result["semantics"])

    def test_run_patient_writes_distribution_returned_by_prediction_details(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patient_dir = root / "patient-1"
            manifest_path = patient_dir / stage03.STAGE02_SCHEMA["patch_outputs"]["manifest"]
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("{}", encoding="utf-8")
            array_path = manifest_path.parent / "patch.npy"
            np.save(array_path, np.zeros((64, 64, 64), dtype=np.float32), allow_pickle=False)
            patch_record = {
                "patch_id": 0,
                "candidate_id": "stable-1",
                "stable_nodule_id": "stable-1",
                "array_path": str(array_path),
            }
            distributions = {
                name: {
                    "probabilities": [1.0],
                    "rating_values": [3.0],
                    "rating_standard_deviation": 0.0,
                    "entropy_bits": 0.0,
                    "calibrated_confidence": None,
                    "semantics": "not_calibrated_confidence",
                }
                for name in FEATURE_NAMES
            }
            with (
                patch.object(
                    stage03,
                    "load_validated_stage02",
                    return_value=({}, [patch_record], {"stable-1": patch_record}),
                ),
                patch.object(stage03, "check_meta_schema_version"),
                patch.object(
                    stage03,
                    "stage03_prediction_details",
                    return_value=({name: 3.0 for name in FEATURE_NAMES}, distributions, {}),
                ) as predict_details,
            ):
                output = stage03.run_patient(
                    patient_dir,
                    {},
                    {},
                    torch.device("cpu"),
                    output_root=root,
                    collect_attention=False,
                )

            predict_details.assert_called_once()
            self.assertEqual(output["predictions"][0]["attributes"][FEATURE_NAMES[0]]["predictive_distribution"],
                             distributions[FEATURE_NAMES[0]])
            saved = json.loads(
                (patient_dir / stage03.STAGE03_PATHS["directory"] / stage03.STAGE03_OUTPUTS["predictions_json"])
                .read_text(encoding="utf-8")
            )
            self.assertEqual(saved["predictions"][0]["attributes"][FEATURE_NAMES[0]]["uncertainty"]["entropy_bits"], 0.0)


if __name__ == "__main__":
    unittest.main()
