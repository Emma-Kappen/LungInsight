"""Input resolution for the Brock/PanCan formula used by Stage 04.

Patient predictors come from the stored clinical record. Nodule size, upper-lobe
location, spiculation, nodule_type and nodule_count are derived from Stage 02/03 outputs
with the mappings below. A manual value never silently overwrites a known value.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

CLINICAL_RECORD_SCHEMA_VERSION = "clinical-intake-record-v1"
SPICULATION_THRESHOLD = 3.0
UNKNOWN_TOKENS = frozenset({"unknown", "not_reported"})
PATIENT_FIELDS = ("age_years", "sex", "family_history_lung_cancer", "emphysema")
NODULE_FIELDS = ("nodule_size_mm", "upper_lobe", "spiculation", "nodule_type", "nodule_count")

CLINICAL_SOURCE = "clinical_record"
MANUAL_SOURCE = "manual_entry"
MANUAL_CORRECTION_SOURCE = "manual_correction"
SIZE_SOURCE = "stage02_patch_manifest.bbox_size_mm_zyx"
LOCATION_SOURCE = "stage02_patch_manifest.anatomical_location.vertical_region"
SPICULATION_SOURCE = "stage03_predictions.attributes.spiculation"
COUNT_SOURCE = "stage02_patch_manifest.eligible_patch_count"
TEXTURE_SOURCE = "stage03_predictions.attributes.texture"
TEXTURE_MAPPING = "round_to_nearest_lidc_texture: 1=pure_ground_glass, 2-3=part_solid, 4-5=solid"


def patient_fields_from_clinical(clinician_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the Brock patient predictors that are explicitly known."""
    patient: dict[str, Any] = {}
    age = clinician_inputs.get("age_at_baseline_years")
    if age is not None and clinician_inputs.get("age_at_baseline_years_status") == "reported":
        patient["age_years"] = age
    sex = clinician_inputs.get("sex_at_baseline")
    if sex in {"female", "male"}:
        patient["sex"] = sex
    family_history = clinician_inputs.get("family_history_lung_cancer")
    if family_history in {"yes", "no"}:
        patient["family_history_lung_cancer"] = family_history == "yes"
    if clinician_inputs.get("medical_history_status") == "reported":
        conditions = clinician_inputs.get("medical_history_conditions") or []
        patient["emphysema"] = "emphysema" in conditions
    return patient


def load_clinical_inputs(record_path: Path, patient_id: str) -> dict[str, Any] | None:
    """Return stored clinician_inputs, None when no record exists, or raise on an identity mismatch."""
    if not record_path.is_file():
        return None
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed clinical record {record_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != CLINICAL_RECORD_SCHEMA_VERSION:
        raise ValueError(f"Clinical record {record_path} is not {CLINICAL_RECORD_SCHEMA_VERSION}.")
    if payload.get("patient_id") != patient_id:
        raise ValueError(
            f"Clinical record {record_path} belongs to {payload.get('patient_id')!r}, not {patient_id!r}."
        )
    inputs = payload.get("clinician_inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"Clinical record {record_path} lacks object-valued clinician_inputs.")
    return inputs


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _missing(source: str, reason: str) -> dict[str, Any]:
    return {"value": None, "source": source, "reason": reason}


def texture_to_nodule_type(rating: float) -> str:
    """Map a 1-5 LIDC-style texture expected rating to a Brock nodule_type (round half up)."""
    rounded = math.floor(rating + 0.5)
    if rounded <= 1:
        return "pure_ground_glass"
    if rounded <= 3:
        return "part_solid"
    return "solid"


def derive_nodule_fields(
    stage02_patch: Mapping[str, Any],
    stage03_record: Mapping[str, Any],
    eligible_patch_count: int,
) -> dict[str, dict[str, Any]]:
    """Map Stage 02/03 outputs to Brock nodule predictors, each with its provenance."""
    fields: dict[str, dict[str, Any]] = {}

    if isinstance(eligible_patch_count, int) and not isinstance(eligible_patch_count, bool) and eligible_patch_count >= 1:
        fields["nodule_count"] = {
            "value": eligible_patch_count,
            "source": COUNT_SOURCE,
            "derivation": "number of lung-gated Stage 02 candidates for this patient (detector count, not radiologist-confirmed)",
            "data_role": "derived_detector_count",
        }
    else:
        fields["nodule_count"] = _missing(COUNT_SOURCE, "no lung-gated Stage 02 candidates for this patient")

    bbox = stage02_patch.get("bbox_size_mm_zyx")
    extents = [_finite_number(value) for value in bbox] if isinstance(bbox, list) and len(bbox) == 3 else []
    if len(extents) == 3 and None not in extents:
        fields["nodule_size_mm"] = {
            "value": max(extents[1], extents[2]),
            "source": SIZE_SOURCE,
            "derivation": "longest in-plane (y,x) detector bounding-box extent, mm",
            "data_role": "derived_detector_estimate",
        }
    else:
        fields["nodule_size_mm"] = _missing(SIZE_SOURCE, "bbox_size_mm_zyx is missing or non-finite")

    vertical = (stage02_patch.get("anatomical_location") or {}).get("vertical_region")
    upper_lobe = {"superior": 1, "inferior": 0}.get(vertical)
    fields["upper_lobe"] = {
        "value": upper_lobe,
        "source": LOCATION_SOURCE,
        "derivation": (
            f"vertical_region={vertical!r}; superior=upper lobe, inferior=non-upper; "
            "middle and unknown abstain"
        ),
        "data_role": "derived_location_proxy",
    }
    if upper_lobe is None:
        fields["upper_lobe"]["reason"] = f"vertical_region={vertical!r} has no unambiguous upper-lobe mapping"

    attribute = (stage03_record.get("attributes") or {}).get("spiculation")
    rating = _finite_number(attribute.get("expected_rating")) if isinstance(attribute, dict) else None
    if rating is not None and 1.0 <= rating <= 5.0:
        fields["spiculation"] = {
            "value": int(rating >= SPICULATION_THRESHOLD),
            "source": SPICULATION_SOURCE,
            "derivation": f"predicted expected_rating >= {SPICULATION_THRESHOLD} mapped to 1, otherwise 0",
            "data_role": "predicted_lidc_style_reader_attribute",
            "raw_expected_rating": rating,
            "rating_scale": [1, 5],
            "threshold": SPICULATION_THRESHOLD,
            "uncertainty": attribute.get("uncertainty"),
            "calibrated_confidence": None,
        }
    else:
        fields["spiculation"] = _missing(SPICULATION_SOURCE, "spiculation expected_rating is missing or outside 1-5")

    texture = (stage03_record.get("attributes") or {}).get("texture")
    texture_rating = _finite_number(texture.get("expected_rating")) if isinstance(texture, dict) else None
    if texture_rating is not None and 1.0 <= texture_rating <= 5.0:
        fields["nodule_type"] = {
            "value": texture_to_nodule_type(texture_rating),
            "source": TEXTURE_SOURCE,
            "derivation": TEXTURE_MAPPING,
            "data_role": "predicted_lidc_style_reader_attribute",
            "raw_expected_rating": texture_rating,
            "rating_scale": [1, 5],
            "uncertainty": texture.get("uncertainty"),
            "calibrated_confidence": None,
        }
    else:
        fields["nodule_type"] = _missing(TEXTURE_SOURCE, "texture expected_rating is missing or outside 1-5")
    return fields


def source_was_used(entry: Mapping[str, Any], source: str) -> bool:
    """True only when a Brock predictor came from the named derived source, not a correction or manual value."""
    return entry.get("source") == source and entry.get("value") is not None


def resolve_values(
    *,
    scope: str,
    fields: tuple[str, ...],
    sourced: Mapping[str, Mapping[str, Any]],
    manual: Mapping[str, Any],
    corrections: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Combine sourced values with manual entries, refusing silent overwrites.

    A manual value fills a missing or unknown source. A manual value that differs from a
    known source value is accepted only with a non-empty corrections[field] reason.
    Manual unknown tokens are recorded but never override a known value.
    """
    orphan = sorted(set(corrections) - set(manual))
    if orphan:
        raise ValueError(f"{scope}: corrections without a manual value: {orphan}")
    values: dict[str, Any] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for field in fields:
        entry: dict[str, Any] = dict(sourced.get(field) or _missing("none", "no source provided"))
        known = entry.get("value")
        if field in manual:
            manual_value = manual[field]
            if isinstance(manual_value, str) and manual_value in UNKNOWN_TOKENS:
                entry["manual_entry"] = manual_value
            elif known is None:
                entry = {"value": manual_value, "source": MANUAL_SOURCE, "derivation": "manual entry; no known source value"}
            elif manual_value == known:
                entry["manual_confirmed"] = True
            else:
                reason = corrections.get(field)
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(
                        f"{scope}.{field}: manual value {manual_value!r} conflicts with "
                        f"{entry.get('source')} value {known!r}; add a non-empty corrections.{field} reason to override."
                    )
                entry = {
                    "value": manual_value,
                    "source": MANUAL_CORRECTION_SOURCE,
                    "reason": reason.strip(),
                    "replaced_source": entry.get("source"),
                    "replaced_value": known,
                }
        values[field] = entry.get("value")
        provenance[field] = entry
    return values, provenance
