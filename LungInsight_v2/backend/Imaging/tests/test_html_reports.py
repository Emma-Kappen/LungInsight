from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

from html_reports import PAGES, generate_patient_report


class HtmlReportTests(unittest.TestCase):
    def test_writes_all_navigable_pages_and_keeps_stable_id_and_lineage(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            patient_root = root / "patient"
            patient_root.mkdir()
            record = {
                "nodule_id": "nodule_a1",
                "stable_nodule_id": "nodule_a1",
                "detector": {
                    "source": "Merged",
                    "source_members": [{"candidate_id": "v1"}, {"candidate_id": "l1"}],
                    "vitdet_score": 0.8,
                    "log_score": 2.3,
                },
                "malignancy": {"predicted_malignancy_rating_continuous": 3.2},
                "explanations": {
                    "gradcam": {
                        "subtlety": {
                            "patch_cam_path": "05/nodule_a1/attribute_patch_cam.npy",
                            "ct_cam_path": "06/nodule_a1/attribute_ct_cam.npy",
                            "target_output_index": 0,
                        }
                    },
                    "shap": {
                        "feature_contributions": {"subtlety": 0.4},
                        "status": "complete",
                    },
                },
            }
            accepted = [{
                "stable_nodule_id": "nodule_a1",
                "stage06_record": record,
                "center_lps_x_mm": 1.0,
                "center_lps_y_mm": 2.0,
                "center_lps_z_mm": 3.0,
                "stage06_record_path": "06/records.json",
            }]
            pages = generate_patient_report(
                patient_id="patient",
                patient_root=patient_root,
                report_root=patient_root / "reports" / "patient",
                accepted=accepted,
                excluded=[],
                ct=np.zeros((3, 4, 5), dtype=np.float32),
                lung_mask=np.ones((3, 4, 5), dtype=bool),
                heart_mask=np.zeros((3, 4, 5), dtype=bool),
            )
            self.assertEqual({path.name for path in pages}, set(PAGES))
            text = (patient_root / "reports" / "patient" / "lineage.html").read_text(encoding="utf-8")
            self.assertIn("nodule_a1", text)
            self.assertIn("Detection", text)
            gradcam = (patient_root / "reports" / "patient" / "gradcam.html").read_text(encoding="utf-8")
            self.assertIn("attribute_ct_cam.npy", gradcam)
            self.assertTrue((patient_root / "reports" / "patient" / "assets" / "mask_overlay_central_slice.png").is_file())

    def test_missing_masks_are_reported_not_fabricated(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pages = generate_patient_report(
                patient_id="p",
                patient_root=root,
                report_root=root / "reports" / "p",
                accepted=[],
                excluded=[],
            )
            self.assertEqual(len(pages), 6)
            masks = (root / "reports" / "p" / "masks.html").read_text(encoding="utf-8")
            self.assertIn("unavailable", masks)

    def test_renders_patch_and_nan_padded_ct_cam_previews(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            patient_root = root / "patient"
            patient_root.mkdir()
            patch_path = patient_root / "05" / "patch_cam.npy"
            ct_cam_path = patient_root / "06" / "ct_cam.npy"
            patch_path.parent.mkdir()
            ct_cam_path.parent.mkdir()
            patch = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            ct_cam = np.full((3, 4, 5), np.nan, dtype=np.float32)
            ct_cam[0, 1:3, 1:3] = patch[0]
            np.save(patch_path, patch)
            np.save(ct_cam_path, ct_cam)
            record = {
                "nodule_id": "nodule_cam",
                "geometry": {
                    "center_zyx": [0.0, 2.0, 2.0],
                    "bbox_start_zyx": [0, 1, 1],
                    "bbox_end_zyx_exclusive": [1, 3, 3],
                    "crop_start_zyx": [0, 1, 1],
                },
                "detector": {"source": "Merged"},
                "explanations": {
                    "gradcam": {
                        "Subtlety": {
                            "patch_cam_path": "05/patch_cam.npy",
                            "ct_cam_path": "06/ct_cam.npy",
                        }
                    },
                    "shap": {"status": "failed", "failure_reason": "not installed"},
                },
            }
            report_root = patient_root / "reports" / "patient"
            generate_patient_report(
                patient_id="patient",
                patient_root=patient_root,
                report_root=report_root,
                accepted=[{"stable_nodule_id": "nodule_cam", "stage06_record": record}],
                excluded=[],
                ct=np.zeros((3, 4, 5), dtype=np.float32),
            )
            self.assertTrue((report_root / "assets" / "nodule_cam_Subtlety_patch.png").is_file())
            self.assertTrue((report_root / "assets" / "nodule_cam_Subtlety_ct.png").is_file())

    def test_index_includes_interactive_ct_gradcam_and_shap_viewer(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            patient_root = root / "patient"
            patient_root.mkdir()
            patch_path = patient_root / "05" / "patch_cam.npy"
            patch_path.parent.mkdir()
            np.save(patch_path, np.arange(8, dtype=np.float32).reshape(2, 2, 2))
            record = {
                "stable_nodule_id": "nodule_interactive",
                "geometry": {
                    "center_zyx": [1.0, 2.0, 2.0],
                    "bbox_start_zyx": [1, 1, 1],
                    "bbox_end_zyx_exclusive": [2, 3, 3],
                    "crop_start_zyx": [0, 1, 1],
                    "patch_size_vox_zyx": [2, 2, 2],
                },
                "detector": {"source": "LoG"},
                "malignancy": {"predicted_malignancy_rating_continuous": 3.5},
                "explanations": {
                    "gradcam": {
                        "subtlety": {
                            "patch_cam_path": "05/patch_cam.npy",
                            "target_scalar_definition": "expected rating",
                        }
                    },
                    "shap": {
                        "status": "complete",
                        "features": [{
                            "feature_name": "subtlety",
                            "feature_value": 3.2,
                            "shap_value": 0.4,
                        }],
                    },
                },
            }
            report_root = patient_root / "reports" / "patient"
            generate_patient_report(
                patient_id="patient",
                patient_root=patient_root,
                report_root=report_root,
                accepted=[{"stable_nodule_id": "nodule_interactive", "stage06_record": record}],
                excluded=[],
                ct=np.zeros((3, 4, 5), dtype=np.float32),
            )
            index = (report_root / "index.html").read_text(encoding="utf-8")
            self.assertIn("viewer-z", index)
            self.assertIn("viewer-alpha", index)
            self.assertIn("viewer-nodules", index)
            self.assertNotIn("id='viewer-nodule'", index)
            self.assertIn("Last checked nodule SHAP explanation", index)
            self.assertTrue(
                (report_root / "assets" / "viewer" / "nodule_interactive_subtlety_cam_sprite.png").is_file()
            )
            for page in ("masks.html", "nodules.html", "gradcam.html", "shap.html"):
                self.assertIn(
                    "viewer-z",
                    (report_root / page).read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
