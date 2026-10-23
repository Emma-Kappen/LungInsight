from __future__ import annotations

import copy
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

import brock_inputs as bi  # noqa: E402

SCRIPT = IMAGING_DIR / "stages" / "04_malignancy_prediction.py"
SPEC = importlib.util.spec_from_file_location("stage04_inputs_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not import the Stage 04 implementation.")
stage04 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage04
SPEC.loader.exec_module(stage04)

# Inputs chosen so the Brock formula reproduces the BTS calculator reference case
# (62-year-old female, no family history or emphysema, 10 mm solid, lower lobe,
# 4 nodules, not spiculated) that tests/test_brock.py already verifies at 4.49%.
PATCH = {
    "candidate_id": "stable-1",
    "stable_nodule_id": "stable-1",
    "patch_id": 0,
    "bbox_size_mm_zyx": [10.0, 10.0, 10.0],
    "anatomical_location": {"vertical_region": "inferior"},
}
ROW = {
    "candidate_id": "stable-1",
    "stable_nodule_id": "stable-1",
    "patch_id": 0,
    "stage04_eligible": True,
    "attributes": {"spiculation": {"expected_rating": 1.0, "uncertainty": {"entropy_bits": 0.2}}},
}
CLINICAL_INPUTS = {
    "age_at_baseline_years": 62.0,
    "age_at_baseline_years_status": "reported",
    "sex_at_baseline": "female",
    "sex_at_baseline_status": "reported",
    "family_history_lung_cancer": "no",
    "medical_history_status": "reported",
    "medical_history_conditions": [],
}
CLINICAL_RECORD = {
    "schema_version": "clinical-intake-record-v1",
    "patient_id": "patient",
    "saved_at": "2026-10-16T00:00:00Z",
    "clinician_inputs": CLINICAL_INPUTS,
}
COMPLETE_ENVELOPE = {
    "patient": {},
    "nodules": {"stable-1": {"nodule_type": "solid", "nodule_count": 4}},
    "corrections": {"nodules": {"stable-1": {
        "nodule_count": "BTS reference case uses 4 nodules; fixture has one accepted Stage 02 candidate",
    }}},
}


class DerivedFieldTests(unittest.TestCase):
    def test_size_is_longest_in_plane_extent_with_provenance(self):
        fields = bi.derive_nodule_fields(
            {"bbox_size_mm_zyx": [17.5, 9.375, 8.59375], "anatomical_location": {"vertical_region": "superior"}},
            ROW,
            1,
        )
        self.assertEqual(fields["nodule_size_mm"]["value"], 9.375)
        self.assertEqual(fields["nodule_size_mm"]["data_role"], "derived_detector_estimate")

    def test_upper_lobe_maps_only_unambiguous_regions(self):
        superior = bi.derive_nodule_fields({"anatomical_location": {"vertical_region": "superior"}}, ROW, 1)
        middle = bi.derive_nodule_fields({"anatomical_location": {"vertical_region": "middle"}}, ROW, 1)
        self.assertEqual(superior["upper_lobe"]["value"], 1)
        self.assertIsNone(middle["upper_lobe"]["value"])
        self.assertIn("reason", middle["upper_lobe"])

    def test_spiculation_uses_documented_threshold_and_rejects_out_of_scale(self):
        high = bi.derive_nodule_fields({}, {"attributes": {"spiculation": {"expected_rating": 3.2}}}, 1)
        low = bi.derive_nodule_fields({}, {"attributes": {"spiculation": {"expected_rating": 2.9}}}, 1)
        off_scale = bi.derive_nodule_fields({}, {"attributes": {"spiculation": {"expected_rating": 7.0}}}, 1)
        self.assertEqual(high["spiculation"]["value"], 1)
        self.assertEqual(low["spiculation"]["value"], 0)
        self.assertIsNone(off_scale["spiculation"]["value"])
        self.assertIsNone(high["spiculation"]["calibrated_confidence"])

    def test_nodule_count_is_derived_from_eligible_stage02_candidates(self):
        fields = bi.derive_nodule_fields(PATCH, ROW, 3)
        self.assertEqual(fields["nodule_count"]["value"], 3)
        self.assertEqual(fields["nodule_count"]["source"], bi.COUNT_SOURCE)
        self.assertIsNone(bi.derive_nodule_fields(PATCH, ROW, 0)["nodule_count"]["value"])

    def test_nodule_type_maps_predicted_texture_with_provenance(self):
        row = {"attributes": {"texture": {"expected_rating": 4.088, "uncertainty": {"entropy_bits": 0.4}}}}
        fields = bi.derive_nodule_fields(PATCH, row, 1)
        self.assertEqual(fields["nodule_type"]["value"], "solid")
        self.assertEqual(fields["nodule_type"]["source"], bi.TEXTURE_SOURCE)
        self.assertEqual(fields["nodule_type"]["raw_expected_rating"], 4.088)
        self.assertEqual(fields["nodule_type"]["data_role"], "predicted_lidc_style_reader_attribute")
        self.assertIsNone(fields["nodule_type"]["calibrated_confidence"])

    def test_texture_rating_rounds_half_up_into_nodule_type_categories(self):
        expected = {1.0: "pure_ground_glass", 1.4: "pure_ground_glass", 1.5: "part_solid",
                    2.4: "part_solid", 2.5: "part_solid", 3.49: "part_solid", 3.5: "solid", 5.0: "solid"}
        for rating, nodule_type in expected.items():
            row = {"attributes": {"texture": {"expected_rating": rating}}}
            with self.subTest(rating=rating):
                self.assertEqual(bi.derive_nodule_fields(PATCH, row, 1)["nodule_type"]["value"], nodule_type)

    def test_texture_outside_one_to_five_scale_is_missing(self):
        for rating in (0.5, 5.2, float("nan")):
            row = {"attributes": {"texture": {"expected_rating": rating}}}
            with self.subTest(rating=rating):
                self.assertIsNone(bi.derive_nodule_fields(PATCH, row, 1)["nodule_type"]["value"])


class PatientFieldTests(unittest.TestCase):
    def test_unknown_and_unreported_values_are_omitted_not_coerced(self):
        patient = bi.patient_fields_from_clinical({
            "age_at_baseline_years": 62.0,
            "age_at_baseline_years_status": "reported",
            "sex_at_baseline": "unknown",
            "family_history_lung_cancer": "unknown",
            "medical_history_status": "not_reported",
            "medical_history_conditions": ["emphysema"],
        })
        self.assertEqual(patient, {"age_years": 62.0})

    def test_emphysema_is_derived_only_when_history_is_reported(self):
        patient = bi.patient_fields_from_clinical({
            "medical_history_status": "reported",
            "medical_history_conditions": ["emphysema"],
        })
        self.assertIs(patient["emphysema"], True)


class ResolveValuesTests(unittest.TestCase):
    def _sourced(self, value):
        return {"sex": {"value": value, "source": bi.CLINICAL_SOURCE}}

    def test_manual_value_fills_a_missing_source(self):
        values, provenance = bi.resolve_values(
            scope="patient", fields=("sex",), sourced={}, manual={"sex": "male"}, corrections={},
        )
        self.assertEqual(values["sex"], "male")
        self.assertEqual(provenance["sex"]["source"], bi.MANUAL_SOURCE)

    def test_unknown_token_never_overrides_a_known_value(self):
        values, provenance = bi.resolve_values(
            scope="patient", fields=("sex",), sourced=self._sourced("male"),
            manual={"sex": "unknown"}, corrections={},
        )
        self.assertEqual(values["sex"], "male")
        self.assertEqual(provenance["sex"]["manual_entry"], "unknown")

    def test_conflicting_manual_value_requires_a_correction_reason(self):
        with self.assertRaisesRegex(ValueError, "conflicts with clinical_record"):
            bi.resolve_values(
                scope="patient", fields=("sex",), sourced=self._sourced("male"),
                manual={"sex": "female"}, corrections={},
            )

    def test_correction_with_reason_records_the_replaced_source(self):
        values, provenance = bi.resolve_values(
            scope="patient", fields=("sex",), sourced=self._sourced("male"),
            manual={"sex": "female"}, corrections={"sex": "clinician-confirmed transcription error"},
        )
        self.assertEqual(values["sex"], "female")
        self.assertEqual(provenance["sex"]["source"], bi.MANUAL_CORRECTION_SOURCE)
        self.assertEqual(provenance["sex"]["replaced_value"], "male")

    def test_orphan_correction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "corrections without a manual value"):
            bi.resolve_values(
                scope="patient", fields=("sex",), sourced={}, manual={}, corrections={"sex": "reason"},
            )


class ClinicalRecordLoadTests(unittest.TestCase):
    def test_missing_record_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(bi.load_clinical_inputs(Path(directory) / "absent.json", "patient"))

    def test_patient_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text(json.dumps({**CLINICAL_RECORD, "patient_id": "other"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "belongs to 'other'"):
                bi.load_clinical_inputs(path, "patient")


class Stage04JoinTests(unittest.TestCase):
    def _run(self, root: Path, *, clinical=CLINICAL_RECORD, envelope=None, patch_row=PATCH, row=ROW):
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
        clinical_path = root / "clinical.json"
        if clinical is not None:
            clinical_path.write_text(json.dumps(clinical), encoding="utf-8")
        argv = [
            "04_malignancy_prediction.py",
            "--predictions", str(stage03_path),
            "--output", str(root / "out.json"),
            "--clinical-record", str(clinical_path),
        ]
        if envelope is not None:
            envelope_path = root / "envelope.json"
            envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
            argv += ["--brock-inputs", str(envelope_path)]
        with (
            patch.object(stage04, "load_validated_stage02", return_value=({}, [patch_row], {"stable-1": {}})),
            patch.object(sys, "argv", argv),
        ):
            stage04.main()
        return json.loads((root / "out.json").read_text(encoding="utf-8"))

    def _prediction(self, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            payload = self._run(Path(directory), **kwargs)
        return payload, payload["predictions"][0]

    def test_valid_clinical_and_stage03_inputs_join_into_an_available_result(self):
        payload, prediction = self._prediction(envelope=COMPLETE_ENVELOPE)
        self.assertEqual(payload["clinical_record"]["status"], "loaded")
        self.assertEqual(prediction["brock"]["status"], "available")
        self.assertAlmostEqual(prediction["brock"]["probability"] * 100.0, 4.49, delta=0.01)
        self.assertEqual(prediction["input_completeness"]["fields_missing"], [])
        self.assertEqual(prediction["input_provenance"]["patient_fields"]["sex"]["source"], bi.CLINICAL_SOURCE)
        self.assertTrue(prediction["input_provenance"]["stage03_predictions_used_as_brock_predictors"])
        self.assertTrue(prediction["input_provenance"]["stage02_detector_size_used_as_brock_size"])
        self.assertTrue(prediction["input_provenance"]["stage02_location_used_as_brock_upper_lobe"])

    def test_missing_family_history_abstains_without_a_probability(self):
        clinical = copy.deepcopy(CLINICAL_RECORD)
        clinical["clinician_inputs"]["family_history_lung_cancer"] = "unknown"
        _, prediction = self._prediction(clinical=clinical, envelope=COMPLETE_ENVELOPE)
        self.assertEqual(prediction["brock"]["status"], "incomplete")
        self.assertIsNone(prediction["brock"]["probability"])
        self.assertIn("family_history_lung_cancer", prediction["input_completeness"]["fields_missing"])

    def test_missing_stage03_spiculation_abstains(self):
        _, prediction = self._prediction(
            envelope=COMPLETE_ENVELOPE,
            row={**ROW, "attributes": {}},
        )
        self.assertEqual(prediction["brock"]["status"], "incomplete")
        self.assertIn("spiculation", prediction["input_completeness"]["fields_missing"])
        self.assertFalse(prediction["input_provenance"]["stage03_predictions_used_as_brock_predictors"])

    def test_missing_detector_size_abstains(self):
        _, prediction = self._prediction(
            envelope=COMPLETE_ENVELOPE,
            patch_row={**PATCH, "bbox_size_mm_zyx": None},
        )
        self.assertEqual(prediction["brock"]["status"], "incomplete")
        self.assertIn("nodule_size_mm", prediction["input_completeness"]["fields_missing"])

    def test_missing_texture_and_manual_nodule_type_abstains_and_count_is_derived(self):
        _, prediction = self._prediction(
            envelope={"patient": {}, "nodules": {}},
            row={**ROW, "attributes": {"spiculation": {"expected_rating": 1.0}}},
        )
        self.assertEqual(prediction["brock"]["status"], "incomplete")
        self.assertIsNone(prediction["brock"]["probability"])
        self.assertIn("nodule_type", prediction["input_completeness"]["fields_missing"])
        self.assertNotIn("nodule_count", prediction["input_completeness"]["fields_missing"])
        self.assertEqual(prediction["input_provenance"]["nodule_fields"]["nodule_count"]["source"], bi.COUNT_SOURCE)

    def test_unknown_manual_sex_is_not_coerced_into_a_valid_category(self):
        clinical = copy.deepcopy(CLINICAL_RECORD)
        clinical["clinician_inputs"]["sex_at_baseline"] = "unknown"
        envelope = {**COMPLETE_ENVELOPE, "patient": {"sex": "unknown"}}
        _, prediction = self._prediction(clinical=clinical, envelope=envelope)
        self.assertEqual(prediction["brock"]["status"], "incomplete")
        self.assertIn("sex", prediction["input_completeness"]["fields_missing"])
        self.assertEqual(prediction["input_provenance"]["patient_fields"]["sex"]["manual_entry"], "unknown")

    def test_manual_sex_conflicting_with_clinical_record_is_rejected(self):
        envelope = {**COMPLETE_ENVELOPE, "patient": {"sex": "male"}}
        with self.assertRaisesRegex(ValueError, "conflicts with clinical_record"):
            self._prediction(envelope=envelope)

    def test_reasoned_correction_overrides_the_stored_value_with_provenance(self):
        envelope = {
            **COMPLETE_ENVELOPE,
            "patient": {"sex": "male"},
            "corrections": {
                **COMPLETE_ENVELOPE["corrections"],
                "patient": {"sex": "clinician-confirmed transcription error"},
            },
        }
        _, prediction = self._prediction(envelope=envelope)
        sex = prediction["input_provenance"]["patient_fields"]["sex"]
        self.assertEqual(sex["source"], bi.MANUAL_CORRECTION_SOURCE)
        self.assertEqual(sex["replaced_value"], "female")
        self.assertEqual(prediction["brock"]["status"], "available")

    def test_clinical_record_for_another_patient_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "belongs to"):
            self._prediction(clinical={**CLINICAL_RECORD, "patient_id": "other"}, envelope=COMPLETE_ENVELOPE)

    def test_absent_clinical_record_is_reported_and_abstains(self):
        payload, prediction = self._prediction(clinical=None, envelope=COMPLETE_ENVELOPE)
        self.assertEqual(payload["clinical_record"]["status"], "not_found")
        self.assertEqual(prediction["brock"]["status"], "incomplete")
        self.assertIn("age_years", prediction["input_completeness"]["fields_missing"])


if __name__ == "__main__":
    unittest.main()
