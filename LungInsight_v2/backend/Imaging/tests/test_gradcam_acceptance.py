from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

_SPEC = importlib.util.spec_from_file_location("gradcam_stage05", IMAGING_DIR / "inference" / "05_gradcam.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Could not import Stage 05 Grad-CAM module for acceptance tests.")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
filter_accepted_patches = _MODULE.filter_accepted_patches


class GradCamAcceptanceTests(unittest.TestCase):
    def test_only_stage04_accepted_stable_candidates_are_selected(self):
        manifest = [
            {"candidate_id": "accepted", "stable_nodule_id": "accepted", "patch_id": 0},
            {"candidate_id": "rejected", "stable_nodule_id": "rejected", "patch_id": 1},
        ]
        predictions = [{
            "candidate_id": "accepted",
            "stable_nodule_id": "accepted",
            "patch_id": 0,
        }]
        accepted, rejected = filter_accepted_patches(
            manifest,
            predictions,
            [{"candidate_id": "rejected", "reason": "missing_supported_attribute"}],
        )
        self.assertEqual([row["stable_nodule_id"] for row in accepted], ["accepted"])
        self.assertEqual(rejected[0]["reason"], "missing_supported_attribute")

    def test_rejects_identity_mismatch(self):
        with self.assertRaisesRegex(ValueError, "stable identity mismatch"):
            filter_accepted_patches(
                [{"candidate_id": "candidate", "stable_nodule_id": "other", "patch_id": 0}],
                [],
                [],
            )


if __name__ == "__main__":
    unittest.main()
