from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGING_ROOT = PROJECT_ROOT / "backend" / "Imaging"
sys.path.insert(0, str(IMAGING_ROOT))


def load_imaging_script(name: str, filename: str):
    subdirectory = {
        "05_gradcam.py": "inference",
        "06_full_ct_gradcam.py": "visualization",
        "07_visualize.py": "visualization",
    }.get(filename)
    script_path = IMAGING_ROOT / subdirectory / filename if subdirectory else IMAGING_ROOT / filename
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Stage06RecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage06 = load_imaging_script("lunginsight_stage06_records_test", "06_full_ct_gradcam.py")
        cls.stage07 = load_imaging_script("lunginsight_stage07_visuals_test", "07_visualize.py")

    def _write_json(self, path: Path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_stage06_record_joins_only_accepted_stable_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            patient_root = Path(temporary_directory) / "patient"
            stage02 = patient_root / "02_detect_and_patch"
            stage03 = patient_root / "03_nodule_classification"
            stage04 = patient_root / "04_malignancy_prediction"
            candidate = {
                "candidate_id": "candidate-abc",
                "patch_id": 7,
                "detection_source": "ViTDet3D",
                "patient_id": "patient",
                "stage04_eligible": True,
                "checkpoint_metadata": {"group": "test"},
                "attributes": {"subtlety": {"expected_rating": 3.0}},
            }
            score = {
                "candidate_id": "candidate-abc",
                "patch_id": 7,
                "predicted_malignancy_rating_continuous": 3.2,
                "predicted_malignancy_rating_rounded": 3,
                "model_raw_prediction_before_rating_clip": 3.2,
                "rating_scale": "1..5",
                "target_semantics": "reader rating",
                "model_type": "ExtraTreesRegressor",
                "model_file": "model.joblib",
                "feature_names_in_order": ["subtlety"],
                "shap": {"model_sha256": "model-hash", "preprocessing": "none", "limitations": []},
                "shap_explanation": {
                    "base_value_raw": 3.0,
                    "feature_contributions": {"subtlety": 0.2},
                    "explained_output": "raw",
                    "model_sha256": "model-hash",
                },
            }
            self._write_json(stage02 / "candidate_filter_report.json", {"gate_status": "passed"})
            self._write_json(stage02 / "candidates.json", [candidate])
            self._write_json(stage03 / "predictions.json", {"schema_version": "3", "checkpoint_metadata": {}, "predictions": [candidate]})
            self._write_json(stage04 / "malignancy_predictions.json", {"predictions": [score]})
            patch = {
                "candidate_id": "candidate-abc",
                "patch_id": 7,
                "detection_source": "ViTDet3D",
                "coordinate_order": ["z", "y", "x"],
                "source_space": "stage01_final_cropped_ct",
                "center_zyx": [10, 20, 30],
                "center_lps_mm": [1.0, 2.0, 3.0],
                "bbox_start_zyx": [9, 19, 29],
                "bbox_end_zyx": [11, 21, 31],
                # prepare_records normalizes this field to a NumPy array for
                # coordinate calculations; the exported record must stay JSON-safe.
                "crop_start_zyx": np.asarray([0, 0, 0], dtype=np.int64),
                "crop_end_zyx": [64, 64, 64],
                "patch_size_vox_zyx": [64, 64, 64],
                "detector_score": 0.8,
            }
            result = self.stage06.build_stage06_nodule_records(
                patient_root,
                {"ct_geometry": {"spacing_zyx_mm": [2.5, 0.7, 0.7]}},
                [patch],
                {7: {"subtlety": {"scaled_path": "cam.npy"}}},
            )
        record = result["records"][0]
        self.assertEqual(record["nodule_id"], "candidate-abc")
        self.assertEqual(record["geometry"]["bbox_start_zyx"], [9, 19, 29])
        self.assertEqual(record["malignancy"]["predicted_malignancy_rating_rounded"], 3)
        self.assertEqual(record["explanations"]["shap"]["feature_contributions"]["subtlety"], 0.2)
        self.assertFalse(record["malignancy"]["is_cancer_probability"])
        self.assertEqual(json.loads(json.dumps(result))["records"][0]["geometry"]["crop_start_zyx"], [0, 0, 0])

    def test_stage06_fails_closed_when_candidate_filter_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            patient_root = Path(temporary_directory) / "patient"
            stage02 = patient_root / "02_detect_and_patch"
            self._write_json(stage02 / "candidate_filter_report.json", {"gate_status": "blocked_heart_mask_review_required"})
            with self.assertRaisesRegex(RuntimeError, "acceptance gate did not pass"):
                self.stage06.build_stage06_nodule_records(
                    patient_root,
                    {},
                    [{"candidate_id": "c", "patch_id": 0, "detection_source": "ViTDet3D"}],
                    {},
                )

    def test_stage07_writes_candidate_bound_ct_cam_and_shap_image(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            patient_root = root / "patient"
            cam_root = patient_root / "05_explainability"
            cam_root.mkdir(parents=True)
            cam_refs = {}
            for attribute in self.stage07.ATTRIBUTES:
                cam_path = cam_root / f"{attribute}.npy"
                np.save(cam_path, np.full((4, 4, 4), 0.5, dtype=np.float32))
                cam_refs[attribute] = {
                    "scaled_path": cam_path.relative_to(patient_root).as_posix(),
                    "raw_signed_path": None,
                }
            record = {
                "nodule_id": "candidate-abc",
                "candidate_id": "candidate-abc",
                "patch_id": 2,
                "detection_source": "ViTDet3D",
                "geometry": {
                    "center_zyx": [2, 4, 4],
                    "bbox_start_zyx": [1, 3, 3],
                    "bbox_end_zyx_exclusive": [4, 6, 6],
                    "crop_start_zyx": [1, 2, 2],
                    "patch_size_vox_zyx": [4, 4, 4],
                },
                "explanations": {
                    "gradcam": cam_refs,
                    "shap": {"feature_contributions": {name: index / 10 for index, name in enumerate(self.stage07.ATTRIBUTES)}},
                },
            }
            row = {"stage06_record": record, "predicted_malignancy_rating_continuous": 3.2}
            paths = self.stage07.render_nodule_visualizations(
                np.zeros((6, 10, 10), dtype=np.float32),
                [row],
                patient_root,
                root / "visualizations",
                "patient",
                -600.0,
                1500.0,
            )
            self.assertEqual(len(paths), 1)
            self.assertTrue(Path(paths[0]).is_file())
            self.assertIn("nodule_002_explanation.png", paths[0])


if __name__ == "__main__":
    unittest.main()
