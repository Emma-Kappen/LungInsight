from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

from api.contract import build_contract, create_contract_file, validate_contract


def sample_inputs():
    meta = {
        "study_instance_uid": "1.2.3.study",
        "series_instance_uid": "1.2.3.series",
        "origin_mm": [10.0, 20.0, 30.0],
        "spacing_zyx_mm": [2.0, 3.0, 4.0],
        "direction_lps": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }
    stage06 = {
        "ct_volume": "volume_hu_cropped.npy",
        "records": [{
            "stable_nodule_id": "nodule-stable-001",
            "candidate_id": "nodule-stable-001",
            "detector": {
                "source": "Merged",
                "source_members": [
                    {"candidate_id": "v1", "source": "ViTDet3D"},
                ],
                "vitdet_score": 0.83,
            },
            "geometry": {
                "center_zyx": [2.0, 3.0, 4.0],
                "center_lps_mm": [14.0, 29.0, 46.0],
                "bbox_start_zyx": [1, 2, 3],
                "bbox_end_zyx_exclusive": [3, 4, 5],
                "diameter_mm": 8.5,
                "volume_mm3": 250.0,
            },
            "attributes": {
                "subtlety": {"expected_rating": 3.4},
                "sphericity": {"expected_rating": 3.0},
                "margin": {"expected_rating": 3.1},
                "lobulation": {"expected_rating": 2.9},
                "spiculation": {"expected_rating": 2.8},
                "texture": {"expected_rating": 3.2},
            },
            "malignancy": {
                "brock": {
                    "status": "incomplete",
                    "method": "brock_formula_decomposition",
                    "model_name": "Brock_PanCan_full",
                    "model_version": "published-coefficients-verified-v1",
                    "target": "pulmonary_nodule_malignancy_probability",
                    "probability": None,
                    "features_missing": ["age_years"],
                    "feature_contributions": [],
                    "limitations": ["Required input was not supplied."],
                },
                "selected_prediction": {
                    "method": "brock",
                    "model_version": "published-coefficients-verified-v1",
                    "status": "incomplete",
                    "probability": None,
                    "value": None,
                    "missing_inputs": ["age_years"],
                    "limitations": ["Required input was not supplied."],
                },
            },
            "uncertainty_and_limitations": {"status": "model_output"},
            "model_identity": {
                "stage04_model_version": "rating-v1",
                "stage03_attribute_support_audit": {
                    "subtlety": {"status": "supported", "target_index": 0},
                    "internal_structure": {"status": "unsupported_checkpoint_contract", "target_index": 1},
                    "calcification": {"status": "unsupported_checkpoint_contract", "target_index": 2},
                    "sphericity": {"status": "supported", "target_index": 3},
                    "margin": {"status": "supported", "target_index": 4},
                    "lobulation": {"status": "supported", "target_index": 5},
                    "spiculation": {"status": "supported", "target_index": 6},
                    "texture": {"status": "supported", "target_index": 7},
                },
            },
            "explanations": {
                "gradcam": {
                    name: {
                        "patch_cam_path": f"05/{name}_patch.npy",
                        "ct_cam_path": f"06/{name}_ct.npy",
                    }
                    for name in ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture")
                },
                "shap": {
                    "status": "complete",
                    "feature_contributions": {"subtlety": 0.3},
                    "features": [{"feature_name": "subtlety", "feature_value": 3.4, "shap_value": 0.3}],
                },
            },
            "quality_flags": [],
        }],
    }
    return meta, stage06


class ImagingContractTests(unittest.TestCase):
    def test_contract_schema_and_coordinate_transform(self):
        meta, stage06 = sample_inputs()
        contract = build_contract(patient_id="patient-001", meta=meta, stage06=stage06)
        validate_contract(contract)
        nodule = contract["nodules"][0]
        self.assertEqual(nodule["stable_nodule_id"], "nodule-stable-001")
        self.assertEqual(nodule["detector"]["vitdet_score"], 0.83)
        self.assertEqual(nodule["bbox_lps_mm"]["min_lps_mm"], [12.0, 26.0, 42.0])
        self.assertEqual(nodule["bbox_lps_mm"]["max_lps_mm"], [16.0, 32.0, 50.0])
        self.assertEqual(
            nodule["classification"]["attributes"]["internal_structure"]["status"],
            "unsupported_checkpoint_contract",
        )
        self.assertEqual(contract["contract_version"], "2.0.0")
        self.assertEqual(nodule["detection_sources"], ["ViTDet3D"])
        self.assertEqual(contract["study"]["eligible_nodule_count"], 1)

    def test_missing_checkpoint_supported_attribute_is_rejected(self):
        meta, stage06 = sample_inputs()
        stage06["records"][0]["attributes"] = {}
        with self.assertRaisesRegex(ValueError, "missing checkpoint-supported attribute"):
            build_contract(patient_id="patient-001", meta=meta, stage06=stage06)

    def test_placeholder_study_uid_is_rejected(self):
        meta, stage06 = sample_inputs()
        meta["study_instance_uid"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "verified study_instance_uid"):
            build_contract(patient_id="patient-001", meta=meta, stage06=stage06)

    def test_contract_accepts_explicit_shap_failure_artifact(self):
        meta, stage06 = sample_inputs()
        stage06["records"][0]["explanations"]["shap"] = {
            "status": "failed",
            "failure_reason": "SHAP compatibility error",
            "failure_artifact": "04/predictions_shap_failure.json",
        }
        contract = build_contract(patient_id="patient-001", meta=meta, stage06=stage06)
        self.assertEqual(contract["nodules"][0]["shap_refs"]["status"], "failed")

    def test_contract_accepts_selected_brock_result(self):
        meta, stage06 = sample_inputs()
        stage06["records"][0]["malignancy"] = {
            "brock": {
                "status": "incomplete",
                "method": "brock_formula_decomposition",
                "model_name": "Brock_PanCan_full",
                "model_version": "published-coefficients-verified-v1",
                "target": "pulmonary_nodule_malignancy_probability",
                "probability": None,
                "features_missing": ["age_years"],
                "feature_contributions": [],
                "limitations": ["Required input was not supplied."],
            },
            "selected_prediction": {
                "method": "brock",
                "model_version": "published-coefficients-verified-v1",
                "status": "incomplete",
                "probability": None,
                "value": None,
                "missing_inputs": ["age_years"],
                "limitations": ["Required input missing."],
            }
        }
        contract = build_contract(patient_id="patient-001", meta=meta, stage06=stage06)
        self.assertEqual(
            contract["nodules"][0]["malignancy"]["selected_prediction"]["method"],
            "brock",
        )

    def test_contract_preserves_brock_explanation_status_and_probability(self):
        meta, stage06 = sample_inputs()
        stage06["records"][0]["malignancy"]["brock"] = {
            "status": "available",
            "method": "brock_formula_decomposition",
            "model_name": "Brock_PanCan_full",
            "model_version": "published-coefficients-verified-v1",
            "target": "pulmonary_nodule_malignancy_probability",
            "probability": 0.42,
            "features_missing": [],
            "feature_contributions": [{"feature": "intercept", "contribution_logit": -6.7892}],
            "limitations": ["Research risk estimate; not a pathology diagnosis."],
        }
        stage06["records"][0]["malignancy"]["selected_prediction"].update({
            "status": "available",
            "probability": 0.42,
            "value": 0.42,
            "missing_inputs": [],
        })
        contract = build_contract(patient_id="patient-001", meta=meta, stage06=stage06)
        self.assertEqual(contract["nodules"][0]["malignancy"]["brock"]["probability"], 0.42)

    def test_contract_accepts_incomplete_brock_abstention(self):
        meta, stage06 = sample_inputs()
        stage06["records"][0]["malignancy"]["brock"] = {
            "status": "incomplete",
            "method": "brock_formula_decomposition",
            "model_name": "Brock_PanCan_full",
            "model_version": "published-coefficients-verified-v1",
            "target": "pulmonary_nodule_malignancy_probability",
            "probability": None,
            "features_missing": ["spiculation"],
            "feature_contributions": [],
            "limitations": ["Required input was not supplied."],
        }
        contract = build_contract(patient_id="patient-001", meta=meta, stage06=stage06)
        self.assertEqual(contract["nodules"][0]["malignancy"]["brock"]["status"], "incomplete")

    def test_missing_supported_cam_reference_blocks_contract(self):
        meta, stage06 = sample_inputs()
        del stage06["records"][0]["explanations"]["gradcam"]["texture"]
        with self.assertRaisesRegex(ValueError, "missing patch/CT-space Grad-CAM"):
            build_contract(patient_id="patient-001", meta=meta, stage06=stage06)

    def test_contract_file_is_generated_from_stage06_and_has_artifact_references(self):
        meta, stage06 = sample_inputs()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            stage_dir = root / "06"
            stage_dir.mkdir()
            stage06_path = stage_dir / "stage06_records.json"
            stage06_path.write_text(json.dumps(stage06), encoding="utf-8")
            output = create_contract_file(root, "patient-001", stage06_path)
            contract = json.loads(output.read_text(encoding="utf-8"))
            validate_contract(contract)
            refs = contract["nodules"][0]["artifact_refs"]
            self.assertEqual(refs["stage06_record"], "06/stage06_records.json")
            self.assertEqual(refs["ct_volume"], "volume_hu_cropped.npy")


if __name__ == "__main__":
    unittest.main()
