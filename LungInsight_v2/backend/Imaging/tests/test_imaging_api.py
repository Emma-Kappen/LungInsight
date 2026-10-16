from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

from fastapi.testclient import TestClient
from api import app as api_module


def stage06_record():
    return {
        "stable_nodule_id": "nodule-api-001",
        "candidate_id": "nodule-api-001",
        "nodule_id": "nodule-api-001",
        "patient_id": "patient-api",
        "detector": {
            "source": "LoG",
            "source_members": [{"candidate_id": "log-member"}],
            "vitdet_score": None,
            "log_score": 1.4,
        },
        "geometry": {
            "center_zyx": [1.0, 1.0, 1.0],
            "center_lps_mm": [1.0, 1.0, 1.0],
            "bbox_start_zyx": [0, 0, 0],
            "bbox_end_zyx_exclusive": [2, 2, 2],
        },
        "attributes": {
            "subtlety": {"expected_rating": 3.0},
            "sphericity": {"expected_rating": 3.0},
            "margin": {"expected_rating": 3.0},
            "lobulation": {"expected_rating": 3.0},
            "spiculation": {"expected_rating": 3.0},
            "texture": {"expected_rating": 3.0},
        },
        "malignancy": {"target_semantics": "LIDC rating"},
        "uncertainty_and_limitations": {},
        "model_identity": {
            "stage03_attribute_support_audit": {
                "subtlety": {"status": "supported"},
                "internal_structure": {"status": "unsupported_checkpoint_contract"},
                "calcification": {"status": "unsupported_checkpoint_contract"},
                "sphericity": {"status": "supported"},
                "margin": {"status": "supported"},
                "lobulation": {"status": "supported"},
                "spiculation": {"status": "supported"},
                "texture": {"status": "supported"},
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
                "source_prediction_json": "04/predictions.json",
                "features": [{"feature_name": "subtlety", "feature_value": 3.0, "shap_value": 0.2}],
                "feature_contributions": {"subtlety": 0.2},
            },
        },
    }


class ImagingApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.original_root = api_module.OUTPUT_ROOT
        api_module.OUTPUT_ROOT = self.root
        patient = self.root / "patient-api"
        patient.mkdir()
        (patient / "meta.json").write_text(json.dumps({
            "study_instance_uid": "1.2.study.api",
            "series_instance_uid": "1.2.series.api",
            "origin_mm": [0, 0, 0],
            "spacing_zyx_mm": [1, 1, 1],
            "direction_lps": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }), encoding="utf-8")
        stage_dir = patient / str(api_module.STAGE06_PATHS["directory"])
        stage_dir.mkdir(parents=True)
        (stage_dir / api_module.STAGE06_RECORDS).write_text(json.dumps({
            "ct_volume": "volume_hu_cropped.npy",
            "records": [stage06_record()],
        }), encoding="utf-8")
        (patient / "volume_hu_cropped.npy").write_bytes(b"ct-reference")
        self.client = TestClient(api_module.app)

    def tearDown(self):
        api_module.OUTPUT_ROOT = self.original_root
        self.temp.cleanup()

    def test_versioned_study_nodule_artifact_and_openapi_endpoints(self):
        response = self.client.post("/api/v1/imaging/study", json={"patient_id": "patient-api"})
        self.assertEqual(response.status_code, 200, response.text)
        contract = response.json()
        self.assertEqual(contract["study_id"], "1.2.study.api")
        self.assertEqual(len(contract["nodules"]), 1)

        response = self.client.get("/api/v1/imaging/study/1.2.study.api")
        self.assertEqual(response.status_code, 200)
        response = self.client.get("/api/v1/imaging/nodule/nodule-api-001")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["nodule"]["stable_nodule_id"], "nodule-api-001")

        contract_path = self.root / "patient-api" / "imaging_clinical_contract_v1.json"
        from api.app import _flatten_artifacts
        artifact = next(
            item for item in _flatten_artifacts(contract, contract_path.parent)
            if item["path"].endswith("volume_hu_cropped.npy")
        )
        response = self.client.get(f"/api/v1/imaging/artifact/{artifact['artifact_id']}")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["artifact"]["available"])
        openapi = self.client.get("/openapi.json").json()
        for path in (
            "/api/v1/imaging/study",
            "/api/v1/imaging/study/{study_id}",
            "/api/v1/imaging/nodule/{nodule_id}",
            "/api/v1/imaging/artifact/{artifact_id}",
        ):
            self.assertIn(path, openapi["paths"])

    def test_rejects_invalid_patient_path(self):
        response = self.client.post("/api/v1/imaging/study", json={"patient_id": "../outside"})
        self.assertEqual(response.status_code, 422)
        response = self.client.post(
            "/api/v1/imaging/study",
            json={"patient_id": "patient-api", "clinical_history": "must not be accepted"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
