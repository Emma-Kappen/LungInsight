from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

from api.contract import build_contract

_STAGE06_SPEC = importlib.util.spec_from_file_location(
    "stage06_full_ct_gradcam_refs_test",
    IMAGING_DIR / "visualization" / "06_full_ct_gradcam.py",
)
stage06 = importlib.util.module_from_spec(_STAGE06_SPEC)
_STAGE06_SPEC.loader.exec_module(stage06)

_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "imaging_contract_fixture_for_stage06",
    Path(__file__).with_name("test_imaging_contract.py"),
)
contract_fixture = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(contract_fixture)

SIX_REQUESTED = ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture")


def _write_cam_pair(root: Path, nodule: str, attribute: str, *, with_ct: bool = True) -> dict:
    patch_rel = f"05/{nodule}/{attribute}_patch.npy"
    ct_rel = f"06/{nodule}/{attribute}_ct.npy"
    (root / patch_rel).parent.mkdir(parents=True, exist_ok=True)
    (root / patch_rel).write_bytes(b"patch")
    if with_ct:
        (root / ct_rel).parent.mkdir(parents=True, exist_ok=True)
        (root / ct_rel).write_bytes(b"ct")
    return {"attribute": attribute, "patch_cam_path": patch_rel, "ct_cam_path": ct_rel}


class Stage06GradcamReferenceTests(unittest.TestCase):
    def test_subtlety_refs_registered_for_matching_identity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = _write_cam_pair(root, "nodule-A", "subtlety")
            cams = {("nodule-A", 7): {"subtlety": entry}}
            refs = stage06.gradcam_refs_for_identity(root, cams, "nodule-A", 7)
        self.assertEqual(refs, {"subtlety": entry})

    def test_missing_ct_artifact_is_not_registered(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = _write_cam_pair(root, "nodule-A", "subtlety", with_ct=False)
            cams = {("nodule-A", 7): {"subtlety": entry}}
            refs = stage06.gradcam_refs_for_identity(root, cams, "nodule-A", 7)
        self.assertEqual(refs, {})

    def test_entry_without_ct_path_is_not_registered(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = _write_cam_pair(root, "nodule-A", "subtlety")
            entry["ct_cam_path"] = None
            cams = {("nodule-A", 7): {"subtlety": entry}}
            refs = stage06.gradcam_refs_for_identity(root, cams, "nodule-A", 7)
        self.assertEqual(refs, {})

    def test_refs_do_not_cross_nodule_or_patch_identity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cams = {("nodule-A", 7): {"subtlety": _write_cam_pair(root, "nodule-A", "subtlety")}}
            self.assertEqual(stage06.gradcam_refs_for_identity(root, cams, "nodule-B", 7), {})
            self.assertEqual(stage06.gradcam_refs_for_identity(root, cams, "nodule-A", 8), {})

    def test_contract_fails_without_subtlety_and_passes_with_registered_refs(self):
        meta, stage06_payload = contract_fixture.sample_inputs()
        record = stage06_payload["records"][0]
        nodule_id = record["candidate_id"]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            record["explanations"]["gradcam"] = {}
            with self.assertRaisesRegex(ValueError, "missing patch/CT-space Grad-CAM references for 'subtlety'"):
                build_contract(patient_id="patient-001", meta=meta, stage06=stage06_payload)

            cams = {
                (nodule_id, 7): {name: _write_cam_pair(root, nodule_id, name) for name in SIX_REQUESTED}
            }
            record["explanations"]["gradcam"] = stage06.gradcam_refs_for_identity(root, cams, nodule_id, 7)
            contract = build_contract(patient_id="patient-001", meta=meta, stage06=stage06_payload)
        self.assertEqual(contract["nodules"][0]["stable_nodule_id"], nodule_id)

    def test_stage06_malignancy_block_holds_only_brock_fields_and_completeness_is_top_level(self):
        meta, stage06_payload = contract_fixture.sample_inputs()
        record = stage06_payload["records"][0]
        nodule_id = record["candidate_id"]
        record["malignancy"]["brock"] = {
            "status": "incomplete",
            "method": "brock_formula_decomposition",
            "model_name": "Brock_PanCan_full",
            "model_version": "published-coefficients-verified-v1",
            "target": "pulmonary_nodule_malignancy_probability",
            "probability": None,
            "features_missing": ["family_history_lung_cancer"],
            "feature_contributions": [],
            "limitations": ["Required input was not supplied."],
        }
        record["input_completeness"] = {"fields_missing": ["family_history_lung_cancer"], "status": "incomplete"}
        record["explanations"]["gradcam"] = {}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cams = {
                (nodule_id, 7): {name: _write_cam_pair(root, nodule_id, name) for name in SIX_REQUESTED}
            }
            record["explanations"]["gradcam"] = stage06.gradcam_refs_for_identity(root, cams, nodule_id, 7)
            contract = build_contract(patient_id="patient-001", meta=meta, stage06=stage06_payload)
        nodule = contract["nodules"][0]
        self.assertEqual(set(nodule["malignancy"]), {"brock", "selected_prediction"})
        self.assertEqual(nodule["input_completeness"], record["input_completeness"])

    def test_contract_still_fails_when_subtlety_ct_artifact_is_missing(self):
        meta, stage06_payload = contract_fixture.sample_inputs()
        record = stage06_payload["records"][0]
        nodule_id = record["candidate_id"]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cams = {
                (nodule_id, 7): {
                    name: _write_cam_pair(root, nodule_id, name, with_ct=(name != "subtlety"))
                    for name in SIX_REQUESTED
                }
            }
            record["explanations"]["gradcam"] = stage06.gradcam_refs_for_identity(root, cams, nodule_id, 7)
            self.assertNotIn("subtlety", record["explanations"]["gradcam"])
            with self.assertRaisesRegex(ValueError, "missing patch/CT-space Grad-CAM references for 'subtlety'"):
                build_contract(patient_id="patient-001", meta=meta, stage06=stage06_payload)


if __name__ == "__main__":
    unittest.main()
