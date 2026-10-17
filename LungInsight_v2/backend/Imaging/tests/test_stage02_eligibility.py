from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

from stage02_eligibility import load_validated_stage02
from pipeline_schema import load_schema

SCHEMA = load_schema(__file__)


class Stage02EligibilityTests(unittest.TestCase):
    def _write_json(self, path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def _make_patient(self, root: Path) -> tuple[Path, np.ndarray]:
        shape = (4, 4, 4)
        volume = np.full(shape, -500.0, dtype=np.float32)
        lung = np.ones(shape, dtype=bool)
        lung[0, 0, 0] = False
        heart = np.zeros(shape, dtype=bool)
        heart[0, 0, 0] = True
        np.save(root / "volume_hu_cropped.npy", volume, allow_pickle=False)
        np.save(root / "lung_mask.npy", lung, allow_pickle=False)
        np.save(root / "heart_mask.npy", heart, allow_pickle=False)
        quality = {
            "status": "validated_automated",
            "shape_zyx": list(shape),
            "problems": [],
            "lung_voxels": int(lung.sum()),
            "heart_voxels": int(heart.sum()),
            "lung_heart_overlap_voxels": 0,
        }
        self._write_json(root / "meta.json", {
            "patient_id": root.name,
            "meta_schema_version": SCHEMA["schema_version"],
            "coordinate_system": "DICOM_LPS",
            "array_axes": ["z", "y", "x"],
            "spacing_zyx_mm": [1.0, 1.0, 1.0],
            "origin_mm": [0.0, 0.0, 0.0],
            "direction_lps": np.eye(3).tolist(),
            "mask_quality": quality,
        })
        self._write_json(root / "mask_quality_report.json", quality)
        stage02 = root / "02_detect_and_patch"
        self._write_json(stage02 / "candidate_filter_report.json", {
            "gate_status": "passed",
            "review_attestation": {"required": False, "verified": False, "status": "not_performed"},
        })
        candidate = {
            "candidate_id": "stable-candidate",
            "stable_nodule_id": "stable-candidate",
            "center_zyx": [1.0, 1.0, 1.0],
            "center_inside_lung_mask": True,
        }
        self._write_json(stage02 / "candidates.json", [candidate])
        patch = {
            "patch_id": 0,
            "candidate_id": "stable-candidate",
            "stable_nodule_id": "stable-candidate",
            "center_zyx": [1.0, 1.0, 1.0],
            "crop_start_zyx": [0, 0, 0],
            "crop_end_zyx": [4, 4, 4],
            "patch_size_vox_zyx": [4, 4, 4],
            "source_space": SCHEMA["coordinate_format"]["ct_frame_space"],
        }
        manifest = {
            "schema_version": SCHEMA["stage_02_detect_and_patch"]["manifest_schema_version"],
            "coordinate_system": SCHEMA["coordinate_system"],
            "coordinate_order": "ZYX",
            "ct_geometry": {
                "shape_zyx": list(shape),
                "spacing_zyx_mm": [1.0, 1.0, 1.0],
                "origin_lps_mm": [0.0, 0.0, 0.0],
                "direction_lps": np.eye(3).tolist(),
            },
            "patches": [patch],
        }
        manifest_path = root / "02_detect_and_patch" / "patches" / "patches_manifest.json"
        self._write_json(manifest_path, manifest)
        return manifest_path, lung

    def test_returns_selected_manifest_without_human_attestation_after_rechecking_mask_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            patient_root = Path(directory) / "patient"
            patient_root.mkdir()
            manifest_path, _ = self._make_patient(patient_root)
            self.assertFalse((patient_root / "heart_mask_review_attestation.json").exists())
            manifest, patches, accepted = load_validated_stage02(patient_root, manifest_path)
        self.assertEqual(len(patches), 1)
        self.assertEqual(manifest["patch_count"] if "patch_count" in manifest else len(manifest["patches"]), 1)
        self.assertEqual(set(accepted), {"stable-candidate"})

    def test_mask_change_that_moves_candidate_outside_blocks_downstream(self):
        with tempfile.TemporaryDirectory() as directory:
            patient_root = Path(directory) / "patient"
            patient_root.mkdir()
            manifest_path, lung = self._make_patient(patient_root)
            lung[1, 1, 1] = False
            np.save(patient_root / "lung_mask.npy", lung, allow_pickle=False)
            quality = json.loads((patient_root / "meta.json").read_text(encoding="utf-8"))["mask_quality"]
            quality["lung_voxels"] = int(lung.sum())
            self._write_json(patient_root / "meta.json", {
                "patient_id": patient_root.name,
                "meta_schema_version": SCHEMA["schema_version"],
                "coordinate_system": "DICOM_LPS",
                "array_axes": ["z", "y", "x"],
                "spacing_zyx_mm": [1.0, 1.0, 1.0],
                "origin_mm": [0.0, 0.0, 0.0],
                "direction_lps": np.eye(3).tolist(),
                "mask_quality": quality,
            })
            self._write_json(patient_root / "mask_quality_report.json", quality)
            self._write_json(patient_root / "02_detect_and_patch" / "candidate_filter_report.json", {
                "gate_status": "passed",
                "review_attestation": {"required": False, "verified": False, "status": "not_performed"},
            })
            with self.assertRaisesRegex(RuntimeError, "not inside the current validated lung mask"):
                load_validated_stage02(patient_root, manifest_path)

    def test_failed_stage02_gate_still_blocks_downstream(self):
        with tempfile.TemporaryDirectory() as directory:
            patient_root = Path(directory) / "patient"
            patient_root.mkdir()
            manifest_path, _ = self._make_patient(patient_root)
            self._write_json(patient_root / "02_detect_and_patch" / "candidate_filter_report.json", {
                "gate_status": "blocked_empty_heart_mask",
            })
            with self.assertRaisesRegex(RuntimeError, "acceptance gate did not pass"):
                load_validated_stage02(patient_root, manifest_path)


if __name__ == "__main__":
    unittest.main()
