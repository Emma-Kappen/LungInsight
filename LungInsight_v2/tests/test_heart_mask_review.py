from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

IMAGING = Path(__file__).resolve().parents[1] / "backend" / "Imaging"
sys.path.insert(0, str(IMAGING))

from heart_mask_review import (  # noqa: E402
    HeartMaskReviewError,
    create_review_attestation,
    validate_review_attestation,
    write_heart_mask_review_montages,
)


def schema() -> dict:
    return {
        "schema_version": "2.11",
        "stage_01_hu_and_crop": {
            "review_qc": {"slices_per_montage": 2, "columns": 2},
            "filenames": {
                "volume_hu_cropped.npy": "volume_hu_cropped.npy",
                "lung_mask.npy": "lung_mask.npy",
                "heart_mask.npy": "heart_mask.npy",
                "heart_tracking_diagnostics.json": "heart_tracking_diagnostics.json",
                "heart_mask_review_metrics.json": "heart_mask_review_metrics.json",
                "heart_mask_review_montage_pattern": "heart_mask_review_montage_{page:03d}.png",
                "heart_mask_review_attestation.json": "heart_mask_review_attestation.json",
                "meta.json": "meta.json",
            },
        },
    }


class HeartMaskReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patient_dir = Path(self.temp.name) / "patient-1"
        self.patient_dir.mkdir()
        self.schema = schema()
        self.volume = np.full((5, 12, 12), -800.0, dtype=np.float32)
        self.lung = np.zeros(self.volume.shape, dtype=bool)
        self.lung[:, 2:10, 2:5] = True
        self.heart = np.zeros(self.volume.shape, dtype=bool)
        self.heart[1:4, 4:8, 5:8] = True

        np.save(self.patient_dir / "volume_hu_cropped.npy", self.volume, allow_pickle=False)
        np.save(self.patient_dir / "lung_mask.npy", self.lung, allow_pickle=False)
        np.save(self.patient_dir / "heart_mask.npy", self.heart, allow_pickle=False)
        (self.patient_dir / "heart_tracking_diagnostics.json").write_text("{}", encoding="utf-8")
        montage_files, metrics = write_heart_mask_review_montages(
            self.volume, self.lung, self.heart, self.patient_dir, self.schema
        )
        self.metadata = {
            "patient_id": self.patient_dir.name,
            "heart_mask_tracking_successful": True,
            "heart_mask_review_montages_generated": True,
            "heart_mask_review_montage_files": montage_files,
        }
        (self.patient_dir / "meta.json").write_text(
            json.dumps(self.metadata), encoding="utf-8"
        )
        (self.patient_dir / "heart_mask_review_metrics.json").write_text(
            json.dumps({"slice_metrics": metrics, "montage_files": montage_files}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_review_montages_cover_every_slice_and_report_metrics(self):
        paths = sorted(self.patient_dir.glob("heart_mask_review_montage_*.png"))
        self.assertEqual(len(paths), 3)
        payload = json.loads((self.patient_dir / "heart_mask_review_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual([row["z"] for row in payload["slice_metrics"]], list(range(self.volume.shape[0])))
        self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_attestation_binds_review_to_exact_artifacts(self):
        path = create_review_attestation(
            self.patient_dir, self.schema, reviewer="operator-7", confirmed_review=True
        )
        result = validate_review_attestation(
            self.patient_dir, self.schema, self.metadata, self.volume, self.lung, self.heart
        )
        self.assertTrue(result["verified"])
        self.assertEqual(result["reviewer"], "operator-7")
        self.assertEqual(result["heart_mask_voxels"], int(self.heart.sum()))
        self.assertTrue(path.is_file())

    def test_modified_review_overlay_invalidates_attestation(self):
        create_review_attestation(
            self.patient_dir, self.schema, reviewer="operator-7", confirmed_review=True
        )
        montage = self.patient_dir / self.metadata["heart_mask_review_montage_files"][0]
        montage.write_bytes(montage.read_bytes() + b"changed")
        with self.assertRaisesRegex(HeartMaskReviewError, "changed after review"):
            validate_review_attestation(
                self.patient_dir, self.schema, self.metadata, self.volume, self.lung, self.heart
            )

    def test_empty_heart_mask_cannot_be_attested(self):
        np.save(self.patient_dir / "heart_mask.npy", np.zeros_like(self.heart), allow_pickle=False)
        with self.assertRaisesRegex(HeartMaskReviewError, "heart/exclusion mask is empty"):
            create_review_attestation(
                self.patient_dir, self.schema, reviewer="operator-7", confirmed_review=True
            )

    def test_confirmation_and_reviewer_are_required(self):
        with self.assertRaisesRegex(HeartMaskReviewError, "No attestation was written"):
            create_review_attestation(
                self.patient_dir, self.schema, reviewer="operator-7", confirmed_review=False
            )
        with self.assertRaisesRegex(HeartMaskReviewError, "must not be empty"):
            create_review_attestation(
                self.patient_dir, self.schema, reviewer=" ", confirmed_review=True
            )
        with self.assertRaisesRegex(HeartMaskReviewError, "actual operator identity"):
            create_review_attestation(
                self.patient_dir, self.schema, reviewer="YOUR_NAME", confirmed_review=True
            )
        with self.assertRaisesRegex(HeartMaskReviewError, "actual operator identity"):
            create_review_attestation(
                self.patient_dir, self.schema, reviewer="ACTUAL_OPERATOR_ID", confirmed_review=True
            )

    def test_existing_attestation_rejects_reviewer_placeholder(self):
        create_review_attestation(
            self.patient_dir, self.schema, reviewer="operator-7", confirmed_review=True
        )
        path = self.patient_dir / "heart_mask_review_attestation.json"
        attestation = json.loads(path.read_text(encoding="utf-8"))
        attestation["reviewer"] = "YOUR_NAME"
        path.write_text(json.dumps(attestation), encoding="utf-8")
        with self.assertRaisesRegex(HeartMaskReviewError, "actual operator identity"):
            validate_review_attestation(
                self.patient_dir, self.schema, self.metadata, self.volume, self.lung, self.heart
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
