"""Deterministic full Brock/PanCan pulmonary nodule risk calculation.

This implementation is not fitted to LungInsight data. It uses the published
McWilliams et al. coefficients and is the only malignancy-risk estimator
supported by this proof-of-concept pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


BROCK_MODEL_NAME = "Brock_PanCan_full"
BROCK_MODEL_VERSION = "published-coefficients-verified-v1"
BROCK_FORMULA_VERSION = "mcwilliams-ne jmoa1214726-full-v1".replace(" ", "")
BROCK_REFERENCE = (
    "McWilliams et al., NEJM 2013, DOI: 10.1056/NEJMoa1214726"
)
BROCK_REFERENCE_CALCULATOR_URL = (
    "https://www.brit-thoracic.org.uk/clinical-resources/guidelines/"
    "pulmonary-nodules/pn-risk-calculator/"
)
SUPPORTED_NODULE_TYPES = ("solid", "pure_ground_glass", "part_solid")
MIN_NODULE_SIZE_MM = 4.0
MAX_NODULE_SIZE_MM = 30.0
MAX_NODULE_COUNT = 100


class BrockInputError(ValueError):
    """Raised when a complete Brock input set is invalid."""


@dataclass(frozen=True)
class BrockInputs:
    age_years: float
    female: int
    family_history_lung_cancer: int
    emphysema: int
    nodule_size_mm: float
    nodule_type: str
    upper_lobe: int
    nodule_count: int
    spiculation: int


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise BrockInputError(f"{name} must be numeric, not boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise BrockInputError(f"{name} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise BrockInputError(f"{name} must be finite.")
    return parsed


def _binary(value: Any, name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise BrockInputError(f"{name} must be a boolean or integer 0/1.")


def _sex_female(value: Any) -> int:
    if not isinstance(value, str):
        raise BrockInputError("sex must be one of female or male.")
    normalized = value.strip().casefold()
    if normalized in {"female", "f"}:
        return 1
    if normalized in {"male", "m"}:
        return 0
    raise BrockInputError("sex must be one of female or male; other/unknown is unsupported.")


def _nodule_type(value: Any) -> str:
    if not isinstance(value, str):
        raise BrockInputError("nodule_type must be a supported string category.")
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "solid": "solid",
        "pure_ground_glass": "pure_ground_glass",
        "ground_glass": "pure_ground_glass",
        "part_solid": "part_solid",
        "partly_solid": "part_solid",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise BrockInputError(
            f"nodule_type must be one of {list(SUPPORTED_NODULE_TYPES)}."
        ) from exc


def validate_brock_inputs(values: Mapping[str, Any]) -> BrockInputs:
    """Validate and canonicalize the full-model Brock inputs."""
    required = {
        "age_years",
        "sex",
        "family_history_lung_cancer",
        "emphysema",
        "nodule_size_mm",
        "nodule_type",
        "upper_lobe",
        "nodule_count",
        "spiculation",
    }
    missing = sorted(name for name in required if name not in values or values[name] is None)
    if missing:
        raise BrockInputError(f"Missing required Brock inputs: {missing}.")
    age = _finite_number(values["age_years"], "age_years")
    if not 18.0 <= age <= 120.0:
        raise BrockInputError("age_years must be between 18 and 120.")
    size = _finite_number(values["nodule_size_mm"], "nodule_size_mm")
    if not MIN_NODULE_SIZE_MM <= size <= MAX_NODULE_SIZE_MM:
        raise BrockInputError(
            f"nodule_size_mm must be between {MIN_NODULE_SIZE_MM} and "
            f"{MAX_NODULE_SIZE_MM} mm for the validated Brock support range."
        )
    count = values["nodule_count"]
    if isinstance(count, bool) or not isinstance(count, int):
        raise BrockInputError("nodule_count must be a positive integer.")
    if not 1 <= count <= MAX_NODULE_COUNT:
        raise BrockInputError(
            f"nodule_count must be between 1 and {MAX_NODULE_COUNT}."
        )
    return BrockInputs(
        age_years=age,
        female=_sex_female(values["sex"]),
        family_history_lung_cancer=_binary(
            values["family_history_lung_cancer"], "family_history_lung_cancer"
        ),
        emphysema=_binary(values["emphysema"], "emphysema"),
        nodule_size_mm=size,
        nodule_type=_nodule_type(values["nodule_type"]),
        upper_lobe=_binary(values["upper_lobe"], "upper_lobe"),
        nodule_count=count,
        spiculation=_binary(values["spiculation"], "spiculation"),
    )


def _stable_sigmoid(logit: float) -> float:
    if logit >= 0:
        exponent = math.exp(-logit)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(logit)
    return exponent / (1.0 + exponent)


def calculate_brock(inputs: BrockInputs) -> dict[str, Any]:
    """Calculate probability and exact additive logit contributions."""
    size_term = ((inputs.nodule_size_mm / 10.0) ** -0.5) - 1.58113883
    contributions = [
        {"feature": "intercept", "encoded_value": 1.0, "contribution_logit": -6.7892},
        {
            "feature": "age_years",
            "input_value": inputs.age_years,
            "unit": "years",
            "encoded_value": inputs.age_years - 62.0,
            "coefficient": 0.0287,
            "contribution_logit": (inputs.age_years - 62.0) * 0.0287,
        },
        {
            "feature": "female",
            "encoded_value": inputs.female,
            "coefficient": 0.6011,
            "contribution_logit": inputs.female * 0.6011,
        },
        {
            "feature": "family_history_lung_cancer",
            "encoded_value": inputs.family_history_lung_cancer,
            "coefficient": 0.2961,
            "contribution_logit": inputs.family_history_lung_cancer * 0.2961,
        },
        {
            "feature": "emphysema",
            "encoded_value": inputs.emphysema,
            "coefficient": 0.2953,
            "contribution_logit": inputs.emphysema * 0.2953,
        },
        {
            "feature": "nodule_size_mm",
            "input_value": inputs.nodule_size_mm,
            "unit": "mm",
            "encoded_value": size_term,
            "coefficient": -5.3854,
            "contribution_logit": size_term * -5.3854,
        },
        {
            "feature": "pure_ground_glass",
            "encoded_value": int(inputs.nodule_type == "pure_ground_glass"),
            "coefficient": -0.1276,
            "contribution_logit": int(inputs.nodule_type == "pure_ground_glass") * -0.1276,
        },
        {
            "feature": "part_solid",
            "encoded_value": int(inputs.nodule_type == "part_solid"),
            "coefficient": 0.3770,
            "contribution_logit": int(inputs.nodule_type == "part_solid") * 0.3770,
        },
        {
            "feature": "upper_lobe",
            "encoded_value": inputs.upper_lobe,
            "coefficient": 0.6581,
            "contribution_logit": inputs.upper_lobe * 0.6581,
        },
        {
            "feature": "nodule_count",
            "input_value": inputs.nodule_count,
            "encoded_value": inputs.nodule_count - 4,
            "coefficient": -0.0824,
            "contribution_logit": (inputs.nodule_count - 4) * -0.0824,
        },
        {
            "feature": "spiculation",
            "encoded_value": inputs.spiculation,
            "coefficient": 0.7729,
            "contribution_logit": inputs.spiculation * 0.7729,
        },
    ]
    logit = float(sum(item["contribution_logit"] for item in contributions))
    probability = _stable_sigmoid(logit)
    if not math.isfinite(logit) or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise BrockInputError("Brock calculation produced a non-finite or invalid result.")
    return {
        "status": "available",
        "method": "brock_formula_decomposition",
        "model_name": BROCK_MODEL_NAME,
        "model_version": BROCK_MODEL_VERSION,
        "formula_version": BROCK_FORMULA_VERSION,
        "reference": BROCK_REFERENCE,
        "target": "pulmonary_nodule_malignancy_probability",
        "probability": probability,
        "logit": logit,
        "feature_contributions": contributions,
        "features_missing": [],
        "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Risk estimate from the published screening model; not a pathology diagnosis.",
            "Validated input support is restricted to the documented size range and categories.",
            "Population transportability and calibration to the LungInsight population are not established.",
        ],
    }


def estimate_brock(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return an explicit incomplete/invalid status instead of fabricating a risk."""
    missing = sorted(
        name for name in (
            "age_years",
            "sex",
            "family_history_lung_cancer",
            "emphysema",
            "nodule_size_mm",
            "nodule_type",
            "upper_lobe",
            "nodule_count",
            "spiculation",
        )
        if name not in values or values[name] is None
    )
    if missing:
        return {
            "status": "incomplete",
            "method": "brock_formula_decomposition",
            "model_name": BROCK_MODEL_NAME,
            "model_version": BROCK_MODEL_VERSION,
            "target": "pulmonary_nodule_malignancy_probability",
            "probability": None,
            "features_missing": missing,
            "feature_contributions": [],
            "limitations": ["Required clinical or nodule inputs were not supplied."],
        }
    try:
        return calculate_brock(validate_brock_inputs(values))
    except BrockInputError as exc:
        return {
            "status": "invalid_input",
            "method": "brock_formula_decomposition",
            "model_name": BROCK_MODEL_NAME,
            "model_version": BROCK_MODEL_VERSION,
            "target": "pulmonary_nodule_malignancy_probability",
            "probability": None,
            "features_missing": [],
            "feature_contributions": [],
            "failure_reason": str(exc),
            "limitations": ["No probability was generated from invalid inputs."],
        }
