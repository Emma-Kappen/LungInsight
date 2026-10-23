from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

SCRIPT = IMAGING_DIR / "stages" / "04_malignancy_prediction.py"
SPEC = importlib.util.spec_from_file_location("stage04_pipeline_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not import the Stage 04 implementation.")
stage04 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage04
SPEC.loader.exec_module(stage04)


class Stage04PipelineTests(unittest.TestCase):
    def _run_stage04(self, root: Path, row: dict):
        patient_root = root / "patient"
        manifest_path = patient_root / "02_detect_and_patch" / "patches" / "patches_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("selected stage02 manifest", encoding="utf-8")
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        stage03_path = patient_root / "03_nodule_classification" / "predictions.json"
        stage03_path.parent.mkdir(parents=True, exist_ok=True)
        stage03_path.write_text(json.dumps({
            "schema_version": stage04.STAGE03_SCHEMA["schema_version"],
            "meta_schema_version": stage04.SCHEMA["schema_version"],
            "patient_id": "patient",
            "stage04_eligible": True,
            "patch_manifest": str(manifest_path.resolve()),
            "patch_manifest_sha256": manifest_hash,
            "predictions": [row],
        }), encoding="utf-8")
        output_path = patient_root / "04_malignancy_prediction" / "malignancy_predictions.json"
        patch_record = {
            "candidate_id": "stable-1",
            "stable_nodule_id": "stable-1",
            "patch_id": 0,
        }
        with (
            patch.object(stage04, "load_validated_stage02", return_value=({}, [patch_record], {"stable-1": {}})),
            patch.object(sys, "argv", [
                "04_malignancy_prediction.py",
                "--predictions", str(stage03_path),
                "--output", str(output_path),
            ]),
        ):
            stage04.main()
        return json.loads(output_path.read_text(encoding="utf-8"))

    def test_eligible_nodule_keeps_explicit_incomplete_brock_result(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self._run_stage04(Path(directory), {
                "candidate_id": "stable-1",
                "stable_nodule_id": "stable-1",
                "patch_id": 0,
                "stage04_eligible": True,
            })
        prediction = payload["predictions"][0]
        self.assertEqual(payload["status"], "completed")
        self.assertTrue(prediction["stage04_eligible"])
        self.assertEqual(prediction["brock"]["status"], "incomplete")
        self.assertIsNone(prediction["brock"]["probability"])
        self.assertIn("nodule_size_mm", prediction["input_completeness"]["fields_missing"])
        self.assertEqual(prediction["diagnosis"]["status"], "not_predicted")
        self.assertEqual(prediction["pathology"]["status"], "not_provided")

    def test_record_without_explicit_eligibility_is_excluded_not_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self._run_stage04(Path(directory), {
                "candidate_id": "stable-1",
                "stable_nodule_id": "stable-1",
                "patch_id": 0,
            })
        self.assertEqual(payload["predictions"], [])
        self.assertEqual(payload["excluded_candidates"][0]["candidate_id"], "stable-1")


if __name__ == "__main__":
    unittest.main()
