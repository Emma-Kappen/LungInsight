from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE02 = ROOT / "backend" / "Imaging" / "stages" / "02_detect_and_patch.py"


class Stage02GateWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = STAGE02.read_text(encoding="utf-8-sig")
        cls.schema = json.loads((ROOT / "meta_schema.json").read_text(encoding="utf-8-sig"))

    def test_human_review_is_validated_before_detector_execution(self):
        review_check = self.source.index("review_attestation = validate_review_attestation(")
        detector_call = self.source.index("raw_detections = run_vitdet3d_raw(")
        self.assertLess(review_check, detector_call)

    def test_authoritative_filter_precedes_patch_extraction_and_manifest_creation(self):
        filter_call = self.source.index(
            "accepted_candidates, rejected_candidates, candidate_filter_report = filter_detections_by_lung_mask("
        )
        accepted_list = self.source.index("candidate_dicts = accepted_candidates", filter_call)
        extraction = self.source.index("records = extract_all(", accepted_list)
        manifest_write = self.source.index("save_json(patch_manifest_path, manifest)", extraction)
        self.assertLess(filter_call, accepted_list)
        self.assertLess(accepted_list, extraction)
        self.assertLess(extraction, manifest_write)

    def test_rejected_and_accepted_records_have_distinct_schema_paths(self):
        stage2 = self.schema["stage_02_detect_and_patch"]
        filenames = stage2["filenames"]
        self.assertNotEqual(filenames["candidates_json"], filenames["rejected_candidates_audit"])
        self.assertNotEqual(filenames["log_diagnostic"], filenames["log_rejected_candidates_audit"])
        self.assertTrue(stage2["candidate_acceptance_filter"]["require_nonempty_heart_mask"])
        self.assertTrue(stage2["candidate_acceptance_filter"]["require_human_review_attestation"])
        self.assertEqual(self.schema["schema_version"], "2.12")
        self.assertIn("2.11", self.schema["metadata_policy"]["supported_schema_versions"])

    def test_log_gate_precedes_log_patch_extraction(self):
        filter_call = self.source.index(
            "accepted_log_candidates, rejected_log_candidates, log_filter_report = filter_detections_by_lung_mask("
        )
        accepted_selection = self.source.index("selected_log_candidates = log_candidates", filter_call)
        extraction = self.source.index("log_records = extract_all(", accepted_selection)
        manifest_write = self.source.index("save_json(log_patch_manifest_path, log_manifest)", extraction)
        self.assertLess(filter_call, accepted_selection)
        self.assertLess(accepted_selection, extraction)
        self.assertLess(extraction, manifest_write)


if __name__ == "__main__":
    unittest.main(verbosity=2)
