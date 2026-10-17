from __future__ import annotations

import importlib.util
import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def _write_stage01_gate(self, patient_root: Path, candidate, patch):
        stage02 = patient_root / "02_detect_and_patch"
        stage02.mkdir(parents=True, exist_ok=True)
        shape = (64, 64, 64)
        volume = np.full(shape, -500.0, dtype=np.float32)
        lung = np.ones(shape, dtype=bool)
        lung[0, 0, 0] = False
        heart = np.zeros(shape, dtype=bool)
        heart[0, 0, 0] = True
        np.save(patient_root / "volume_hu_cropped.npy", volume, allow_pickle=False)
        np.save(patient_root / "lung_mask.npy", lung, allow_pickle=False)
        np.save(patient_root / "heart_mask.npy", heart, allow_pickle=False)
        quality = {
            "status": "validated_automated",
            "shape_zyx": list(shape),
            "problems": [],
            "lung_voxels": int(lung.sum()),
            "heart_voxels": int(heart.sum()),
            "lung_heart_overlap_voxels": 0,
        }
        self._write_json(patient_root / "meta.json", {
            "patient_id": patient_root.name,
            "meta_schema_version": self.stage06.SCHEMA["schema_version"],
            "coordinate_system": "DICOM_LPS",
            "array_axes": ["z", "y", "x"],
            "spacing_zyx_mm": [2.5, 0.7, 0.7],
            "origin_mm": [0.0, 0.0, 0.0],
            "direction_lps": np.eye(3).tolist(),
            "mask_quality": quality,
        })
        self._write_json(patient_root / "mask_quality_report.json", quality)
        hashes = {
            "volume_hu_cropped.npy": hashlib.sha256((patient_root / "volume_hu_cropped.npy").read_bytes()).hexdigest(),
            "lung_mask.npy": hashlib.sha256((patient_root / "lung_mask.npy").read_bytes()).hexdigest(),
            "heart_mask.npy": hashlib.sha256((patient_root / "heart_mask.npy").read_bytes()).hexdigest(),
        }
        self._write_json(patient_root / "heart_mask_review_attestation.json", {
            "schema_version": "heart-mask-review-attestation-v1",
            "meta_schema_version": self.stage06.SCHEMA["schema_version"],
            "patient_id": patient_root.name,
            "reviewed": True,
            "reviewer": "test reviewer",
            "clinical_validation": False,
            "artifacts_sha256": hashes,
        })
        self._write_json(stage02 / "candidate_filter_report.json", {
            "gate_status": "passed",
            "review_attestation": {"verified": True, "artifact_hashes": hashes},
        })
        self._write_json(stage02 / "candidates.json", [candidate])
        manifest_path = patient_root / "02_detect_and_patch" / "patches" / "patches_manifest.json"
        manifest_patch = dict(patch)
        manifest_patch["crop_start_zyx"] = np.asarray(patch["crop_start_zyx"]).tolist()
        manifest = {
            "schema_version": self.stage06.STAGE02_SCHEMA["manifest_schema_version"],
            "coordinate_system": self.stage06.SCHEMA["coordinate_system"],
            "coordinate_order": "ZYX",
            "ct_geometry": {
                "shape_zyx": list(shape),
                "spacing_zyx_mm": [2.5, 0.7, 0.7],
                "origin_lps_mm": [0.0, 0.0, 0.0],
                "direction_lps": np.eye(3).tolist(),
            },
            "patches": [manifest_patch],
        }
        self._write_json(manifest_path, manifest)
        return manifest_path, hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def test_stage06_record_joins_only_accepted_stable_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            patient_root = Path(temporary_directory) / "patient"
            stage02 = patient_root / "02_detect_and_patch"
            stage03 = patient_root / "03_nodule_classification"
            stage04 = patient_root / "04_malignancy_prediction"
            candidate = {
                "candidate_id": "candidate-abc",
                "patch_id": 7,
                "stable_nodule_id": "candidate-abc",
                "detection_source": "ViTDet3D",
                "patient_id": "patient",
                "stage04_eligible": True,
                "center_inside_lung_mask": True,
                "checkpoint_metadata": {"group": "test"},
                "center_zyx": [10, 20, 30],
                "center_lps_mm": [25.0, 14.0, 21.0],
                "bbox_start_zyx": [9, 19, 29],
                "bbox_end_zyx": [11, 21, 31],
                "crop_start_zyx": [0, 0, 0],
                "crop_end_zyx": [64, 64, 64],
                "patch_size_vox_zyx": [64, 64, 64],
                "source_space": "stage01_final_cropped_ct",
                "attributes": {
                    name: {"expected_rating": 3.0}
                    for name in self.stage06.ATTRIBUTES
                },
            }
            score = {
                "candidate_id": "candidate-abc",
                "patch_id": 7,
                "patient_id": "patient",
                "stable_nodule_id": "candidate-abc",
                "model_version": "published-coefficients-verified-v1",
                "stage04_eligible": True,
                "brock": {
                    "status": "incomplete",
                    "method": "brock_formula_decomposition",
                    "model_name": "Brock_PanCan_full",
                    "model_version": "published-coefficients-verified-v1",
                    "target": "pulmonary_nodule_malignancy_probability",
                    "probability": None,
                    "features_missing": ["age_years"],
                    "feature_contributions": [],
                    "limitations": ["Required inputs were not supplied."],
                },
                "selected_prediction": {
                    "method": "brock",
                    "model_version": "published-coefficients-verified-v1",
                    "status": "incomplete",
                    "probability": None,
                    "value": None,
                    "missing_inputs": ["age_years"],
                    "limitations": ["Required inputs were not supplied."],
                },
                "target_semantics": "research-only pulmonary nodule malignancy-risk probability",
            }
            support_audit = {
                name: {"status": "supported"}
                for name in self.stage06.ATTRIBUTES
            }
            patch = {
                "candidate_id": "candidate-abc",
                "patch_id": 7,
                "stable_nodule_id": "candidate-abc",
                "detection_source": "ViTDet3D",
                "coordinate_order": ["z", "y", "x"],
                "source_space": "stage01_final_cropped_ct",
                "center_zyx": [10, 20, 30],
                "center_lps_mm": [25.0, 14.0, 21.0],
                "bbox_start_zyx": [9, 19, 29],
                "bbox_end_zyx": [11, 21, 31],
                # prepare_records normalizes this field to a NumPy array for
                # coordinate calculations; the exported record must stay JSON-safe.
                "crop_start_zyx": np.asarray([0, 0, 0], dtype=np.int64),
                "crop_end_zyx": [64, 64, 64],
                "patch_size_vox_zyx": [64, 64, 64],
                "source_space": "stage01_final_cropped_ct",
                "detector_score": 0.8,
            }
            manifest_path, manifest_hash = self._write_stage01_gate(patient_root, candidate, patch)
            self._write_json(stage03 / "predictions.json", {
                "schema_version": "3.0-spatial-hybrid",
                "stage04_eligible": True,
                "patch_manifest_sha256": manifest_hash,
                "checkpoint_metadata": {},
                "attribute_support_audit": support_audit,
                "predictions": [candidate],
            })
            stage03_path = stage03 / "predictions.json"
            self._write_json(stage04 / "malignancy_predictions.json", {
                "stage02_manifest_sha256": manifest_hash,
                "source_predictions_json": str(stage03_path.resolve()),
                "source_predictions_sha256": hashlib.sha256(stage03_path.read_bytes()).hexdigest(),
                "predictions": [score],
            })
            result = self.stage06.build_stage06_nodule_records(
                patient_root,
                {
                    "ct_geometry": {
                        "spacing_zyx_mm": [2.5, 0.7, 0.7],
                        "shape_zyx": [64, 64, 64],
                        "origin_lps_mm": [0.0, 0.0, 0.0],
                        "direction_lps": np.eye(3).tolist(),
                    },
                    "_manifest_path": str(manifest_path),
                },
                [patch],
                {7: {"subtlety": {"scaled_path": "cam.npy"}}},
            )
        record = result["records"][0]
        self.assertEqual(record["nodule_id"], "candidate-abc")
        self.assertEqual(record["geometry"]["bbox_start_zyx"], [9, 19, 29])
        self.assertEqual(record["malignancy"]["selected_prediction"]["method"], "brock")
        self.assertEqual(record["malignancy"]["selected_prediction"]["status"], "incomplete")
        self.assertIsNone(record["malignancy"]["selected_prediction"]["probability"])
        self.assertEqual(record["explanations"]["shap"]["status"], "not_applicable")
        self.assertEqual(json.loads(json.dumps(result))["records"][0]["geometry"]["crop_start_zyx"], [0, 0, 0])

    def test_stage06_fails_closed_when_candidate_filter_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            patient_root = Path(temporary_directory) / "patient"
            stage02 = patient_root / "02_detect_and_patch"
            self._write_json(stage02 / "candidate_filter_report.json", {"gate_status": "blocked_heart_mask_review_required"})
            with self.assertRaisesRegex(RuntimeError, "acceptance gate"):
                self.stage06.build_stage06_nodule_records(
                    patient_root,
                    {},
                    [{"candidate_id": "c", "patch_id": 0, "detection_source": "ViTDet3D"}],
                    {},
                )

    def test_stage07_writes_candidate_bound_ct_cam_and_brock_image(self):
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
                },
                "malignancy": {
                    "brock": {
                        "status": "incomplete",
                        "probability": None,
                        "feature_contributions": [],
                    },
                    "selected_prediction": {
                        "method": "brock",
                        "status": "incomplete",
                        "probability": None,
                    },
                },
            }
            row = {
                "stage06_record": record,
                "brock_status": "incomplete",
                "brock_probability": None,
            }
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

    def test_stage07_slider_draws_native_voxel_bbox_and_stable_id_details(self):
        geometry = {
            "center_zyx": [1.0, 4.0, 4.0],
            "bbox_start_zyx": [1, 3, 3],
            "bbox_end_zyx_exclusive": [3, 6, 6],
            "diameter_mm": 9.0,
            "volume_mm3": 280.0,
        }
        row = {
            "stable_nodule_id": "stable-nodule-123",
            "center_z_voxel": 1.0,
            "center_y_voxel": 4.0,
            "center_x_voxel": 4.0,
            "bbox_start_zyx": geometry["bbox_start_zyx"],
            "bbox_end_zyx_exclusive": geometry["bbox_end_zyx_exclusive"],
            "detection_source": "Merged",
            "brock_status": "incomplete",
            "brock_probability": None,
            "stage06_record": {
                "geometry": geometry,
                "attributes": {"spiculation": {"expected_rating": 4.0}},
            },
        }
        with patch.object(self.stage07.plt, "show"):
            self.stage07.launch_viewer(
                np.zeros((4, 8, 8), dtype=np.float32),
                np.ones((4, 8, 8), dtype=bool),
                np.zeros((4, 8, 8), dtype=bool),
                np.zeros((4, 8, 8), dtype=bool),
                [row],
                "patient",
                -600.0,
                1500.0,
            )
        figure = self.stage07.plt.gcf()
        try:
            axes = figure.axes[0]
            self.assertEqual(len(axes.patches), 1)
            rectangle = axes.patches[0]
            self.assertEqual(rectangle.get_xy(), (3.0, 3.0))
            self.assertEqual((rectangle.get_width(), rectangle.get_height()), (3.0, 3.0))
            annotation = "\n".join(text.get_text() for text in axes.texts)
            self.assertIn("stable-nodule-123", annotation)
            self.assertIn("Diameter 9.0 mm", annotation)
            self.assertIn("volume 280.0 mm3", annotation)
            self.assertIn("spiculation=4.0", annotation)
            self.assertIn("Brock unavailable (incomplete)", annotation)
        finally:
            self.stage07.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
