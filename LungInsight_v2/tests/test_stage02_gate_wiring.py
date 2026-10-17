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

    def test_automated_mask_validation_precedes_detector_execution_without_attestation(self):
        automated_check = self.source.index("mask_quality.get(\"problems\") != []")
        detector_call = self.source.index("raw_detections = run_vitdet3d_raw(")
        self.assertLess(automated_check, detector_call)
        self.assertNotIn("validate_review_attestation(", self.source)

    def test_fusion_and_lung_gate_precede_top50_and_patch_extraction(self):
        fusion = self.source.index("fused_candidates, fusion_report = merge_detections(")
        filter_call = self.source.index(
            "accepted_candidates, rejected_candidates, candidate_filter_report = filter_detections_by_lung_mask(",
            fusion,
        )
        selection = self.source.index("selected_candidates = select_fused_candidates(accepted_candidates, limit=50)")
        extraction = self.source.index("records = extract_all(", selection)
        manifest_write = self.source.index("save_json(patch_manifest_path, manifest)", extraction)
        self.assertLess(fusion, filter_call)
        self.assertLess(filter_call, selection)
        self.assertLess(selection, extraction)
        self.assertLess(extraction, manifest_write)

    def test_vitdet_candidates_are_fused_before_the_single_mask_gate(self):
        vit_run = self.source.index("raw_detections = run_vitdet3d_raw(")
        fusion = self.source.index("fused_candidates, fusion_report = merge_detections(")
        gate = self.source.index(
            "accepted_candidates, rejected_candidates, candidate_filter_report = filter_detections_by_lung_mask("
        )
        self.assertLess(vit_run, fusion)
        self.assertLess(fusion, gate)

    def test_rejected_and_accepted_records_have_distinct_schema_paths(self):
        stage2 = self.schema["stage_02_detect_and_patch"]
        filenames = stage2["filenames"]
        self.assertNotEqual(filenames["candidates_json"], filenames["rejected_candidates_audit"])
        self.assertTrue(stage2["candidate_acceptance_filter"]["require_nonempty_heart_mask"])
        self.assertFalse(stage2["candidate_acceptance_filter"]["require_human_review_attestation"])
        self.assertEqual(self.schema["schema_version"], "2.15")
        self.assertIn("2.12", self.schema["metadata_policy"]["supported_schema_versions"])

    def test_log_detector_is_fully_removed_from_stage02(self):
        for token in ("run_log_detector", "LogDiagnosticCandidate", "build_log_patch_candidates", "--skip-log"):
            self.assertNotIn(token, self.source)
        self.assertNotIn("log_diagnostic", self.schema["stage_02_detect_and_patch"]["filenames"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
