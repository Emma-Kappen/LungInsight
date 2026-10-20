from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

from brock import (  # noqa: E402
    BROCK_MODEL_NAME,
    BROCK_REFERENCE_CALCULATOR_URL,
    BrockInputError,
    calculate_brock,
    estimate_brock,
    validate_brock_inputs,
)


class BrockTests(unittest.TestCase):
    def valid(self) -> dict[str, object]:
        return {
            "age_years": 62,
            "sex": "female",
            "family_history_lung_cancer": False,
            "emphysema": False,
            "nodule_size_mm": 10.0,
            "nodule_type": "solid",
            "upper_lobe": False,
            "nodule_count": 4,
            "spiculation": False,
        }

    def test_reference_formula_and_contributions_sum_to_logit(self):
        inputs = validate_brock_inputs(self.valid())
        result = calculate_brock(inputs)
        expected_logit = -6.7892 + 0.6011 + (-5.3854 * (1.0 - 1.58113883))
        self.assertAlmostEqual(result["logit"], expected_logit, places=12)
        self.assertAlmostEqual(
            sum(item["contribution_logit"] for item in result["feature_contributions"]),
            result["logit"],
            places=12,
        )
        self.assertAlmostEqual(
            result["probability"],
            1.0 / (1.0 + math.exp(-expected_logit)),
            places=12,
        )
        self.assertEqual(result["model_name"], BROCK_MODEL_NAME)

    def test_matches_verified_official_bts_calculator_reference_case(self):
        """BTS calculator output retrieved 2026-10-16: 4.49% (rounded)."""
        result = estimate_brock(self.valid())
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["probability"] * 100.0, 4.49, delta=0.01)
        self.assertIn("brit-thoracic.org.uk", BROCK_REFERENCE_CALCULATOR_URL)

    def test_indicators_and_nodule_type_change_the_expected_terms(self):
        base = estimate_brock(self.valid())
        altered = self.valid()
        altered.update({
            "sex": "male",
            "family_history_lung_cancer": True,
            "emphysema": True,
            "nodule_type": "part-solid",
            "upper_lobe": True,
            "nodule_count": 1,
            "spiculation": True,
        })
        changed = estimate_brock(altered)
        self.assertEqual(base["status"], "available")
        self.assertEqual(changed["status"], "available")
        self.assertNotEqual(base["probability"], changed["probability"])

    def test_missing_inputs_abstain_without_probability(self):
        result = estimate_brock({"age_years": 62})
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["probability"])
        self.assertIn("sex", result["features_missing"])

    def test_invalid_values_are_explicit(self):
        for field, value in (
            ("age_years", 17),
            ("nodule_size_mm", 0),
            ("nodule_size_mm", 50),
            ("nodule_count", 0),
            ("spiculation", 2),
        ):
            values = self.valid()
            values[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(BrockInputError):
                    validate_brock_inputs(values)
                self.assertEqual(estimate_brock(values)["status"], "invalid_input")

    def test_unknown_sex_and_nodule_type_are_not_coerced(self):
        values = self.valid()
        values["sex"] = "unknown"
        with self.assertRaises(BrockInputError):
            validate_brock_inputs(values)
        values = self.valid()
        values["nodule_type"] = "mixed"
        with self.assertRaises(BrockInputError):
            validate_brock_inputs(values)


if __name__ == "__main__":
    unittest.main()
