from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from ..Imaging.brock_inputs import bbox_in_plane_diameter_mm

MALIGNANCY_THRESHOLD = 0.10
ATTRIBUTE_NAMES = (
    "subtlety", "internal_structure", "calcification", "sphericity",
    "margin", "lobulation", "spiculation", "texture",
)
ATTRIBUTE_RANGES = {
    "subtlety": (1.0, 5.0),
    "internal_structure": (1.0, 4.0),
    "calcification": (1.0, 6.0),
    "sphericity": (1.0, 5.0),
    "margin": (1.0, 5.0),
    "lobulation": (1.0, 5.0),
    "spiculation": (1.0, 5.0),
    "texture": (1.0, 5.0),
}
V4_CATEGORIES = {
    "sex": {"FEMALE", "MALE"},
    "race": {"AMERICAN INDIAN OR ALASKA NATIVE", "ASIAN", "BLACK OR AFRICAN AMERICAN", "WHITE", "__MISSING__"},
    "ethnicity": {"HISPANIC OR LATINO", "NOT HISPANIC OR LATINO", "__MISSING__"},
    "smoking_history_code": {"1", "2", "3", "4", "5", "__MISSING__"},
    "pulmonary_function_test_performed": {"NO", "YES", "__MISSING__"},
    "lung_location": {"Central Lung", "Peripheral Lung", "__MISSING__"},
    "anatomic_subdivision": {"Bronchial", "L-Lower", "L-Upper", "Other (please specify)", "R-Lower", "R-Middle", "R-Upper", "[Discrepancy]", "__MISSING__"},
    "other_malignancy_history": {"No", "Yes", "Yes, History of Prior Malignancy", "Yes, History of Synchronous/Bilateral Malignancy"},
}


NUMERIC_FEATURES = {
    "age_at_diagnosis", "pack_years", "smoking_year_started", "smoking_year_stopped",
    "dlco_percent_predicted", "fev1_fvc_post_percent", "fev1_fvc_pre_percent",
    "fev1_percent_predicted_post", "fev1_percent_predicted_pre",
    "ecog_preoperative", "karnofsky_preoperative",
}
FEATURE_NAMES = (
    "age_at_diagnosis", "sex", "race", "ethnicity", "smoking_history_code",
    "pack_years", "smoking_year_started", "smoking_year_stopped",
    "dlco_percent_predicted", "fev1_fvc_post_percent", "fev1_fvc_pre_percent",
    "fev1_percent_predicted_post", "fev1_percent_predicted_pre",
    "pulmonary_function_test_performed", "ecog_preoperative",
    "karnofsky_preoperative", "lung_location", "anatomic_subdivision",
    "other_malignancy_history",
)

def validate_histology_v4_inputs(values: Mapping[str, Any] | None) -> dict[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError("histology_v4_inputs must be an object.")
    extra = set(values) - set(FEATURE_NAMES)
    if extra:
        raise ValueError(f"Unsupported histology_v4_inputs fields: {sorted(extra)}")
    result: dict[str, Any] = {}
    for name, value in values.items():
        if value is None:
            result[name] = None
        elif name in NUMERIC_FEATURES:
            number = _finite(value)
            if number is None:
                raise ValueError(f"histology_v4_inputs.{name} must be a finite number or null.")
            result[name] = number
        else:
            if not isinstance(value, str):
                raise ValueError(f"histology_v4_inputs.{name} must be a string or null.")
            if value == "__MISSING__":
                result[name] = None
            elif value not in V4_CATEGORIES.get(name, set()):
                raise ValueError(f"histology_v4_inputs.{name} is not an exact v4 encoder category: {value!r}.")
            else:
                result[name] = value
    return result


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _bbox_diameter_mm(nodule: Mapping[str, Any], spacing: Any) -> float | None:
    bbox = nodule.get("bbox_voxel")
    if not isinstance(bbox, Mapping):
        return None
    start, end = bbox.get("start_zyx"), bbox.get("end_zyx_exclusive")
    if not (isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) == len(end) == 3):
        return None
    sp = spacing if isinstance(spacing, (list, tuple)) and len(spacing) == 3 else None
    if sp is None:
        return None
    extents = []
    for i in range(3):
        a, b, s = _finite(start[i]), _finite(end[i]), _finite(sp[i])
        if a is None or b is None or s is None or s <= 0 or b <= a:
            return None
        extents.append((b - a) * s)
    # Share the exact in-plane size convention used by Brock Stage 04.
    return bbox_in_plane_diameter_mm(extents)


def _selected_brock(nodule: Mapping[str, Any]) -> tuple[float | None, bool]:
    malignancy = nodule.get("malignancy")
    selected = malignancy.get("selected_prediction") if isinstance(malignancy, Mapping) else None
    if not isinstance(selected, Mapping):
        return None, False
    probability = _finite(selected.get("probability"))
    is_malignant = (
        selected.get("method") == "brock"
        and selected.get("status") == "available"
        and probability is not None
        and probability >= MALIGNANCY_THRESHOLD
    )
    return probability, is_malignant


def _map_location(anatomy: Any) -> tuple[str | None, str | None]:
    if not isinstance(anatomy, Mapping):
        return None, "anatomical_location_missing"
    primary = anatomy.get("primary")
    if primary == "bronchus":
        return "Central Lung", None
    if primary == "lung_wall" and anatomy.get("wall_relationship") == "wall_adjacent":
        return "Peripheral Lung", None
    return None, "anatomical_location_not_unambiguously_central_or_peripheral"


def _map_subdivision(anatomy: Any) -> tuple[str | None, str | None]:
    if not isinstance(anatomy, Mapping):
        return None, "anatomical_location_missing"
    if anatomy.get("primary") == "bronchus":
        return "Bronchial", None
    # Stage 02's superior/middle/inferior thirds are not anatomical lobes.
    return None, "upstream_location_does_not_identify_a_TCGA_lobe_or_bronchial_subdivision"


def _mode_with_tie_break(rows: list[dict[str, Any]], field: str) -> str | None:
    values = [row[field] for row in rows if isinstance(row.get(field), str)]
    if not values:
        return None
    counts = Counter(values)
    max_count = max(counts.values())
    tied = {value for value, count in counts.items() if count == max_count}
    # For tied modes: largest in-plane BBox diameter, then highest Brock score,
    # then alphabetical category. Stable nodule ID is the final deterministic fallback.
    candidates = [row for row in rows if row.get(field) in tied]
    candidates.sort(key=lambda row: (
        -(row.get("bbox_diameter_mm") if row.get("bbox_diameter_mm") is not None else -1.0),
        -(row.get("malignancy_score") if row.get("malignancy_score") is not None else -1.0),
        str(row[field]).casefold(),
        str(row.get("stable_nodule_id", "")),
    ))
    return candidates[0][field] if candidates else sorted(tied, key=str.casefold)[0]


def aggregate_malignant_nodules(contract: Mapping[str, Any] | None) -> dict[str, Any]:
    """Pure aggregation over validated Imaging contract records."""
    contract = contract if isinstance(contract, Mapping) else {}
    coordinate = contract.get("coordinate_metadata")
    spacing = coordinate.get("spacing_zyx_mm") if isinstance(coordinate, Mapping) else None
    spacing_available = (
        isinstance(spacing, (list, tuple))
        and len(spacing) == 3
        and all(_finite(value) is not None and _finite(value) > 0 for value in spacing)
    )
    all_nodules = contract.get("nodules")
    all_nodules = all_nodules if isinstance(all_nodules, list) else []
    selected: list[dict[str, Any]] = []
    warnings: list[str] = []
    for nodule in all_nodules:
        if not isinstance(nodule, Mapping):
            continue
        score, malignant = _selected_brock(nodule)
        if not malignant:
            continue
        diameter = _bbox_diameter_mm(nodule, spacing)
        anatomy = nodule.get("anatomical_location")
        location, location_warning = _map_location(anatomy)
        subdivision, subdivision_warning = _map_subdivision(anatomy)
        attributes = nodule.get("classification", {}).get("attributes", {}) if isinstance(nodule.get("classification"), Mapping) else {}
        row = {
            "stable_nodule_id": nodule.get("stable_nodule_id"),
            "malignancy_score": score,
            "bbox_diameter_mm": diameter,
            "lung_location": location,
            "anatomic_subdivision": subdivision,
            "attributes": attributes if isinstance(attributes, Mapping) else {},
        }
        selected.append(row)
        if location_warning:
            warnings.append(f"{row['stable_nodule_id']}: {location_warning}")
        if subdivision_warning:
            warnings.append(f"{row['stable_nodule_id']}: {subdivision_warning}")
        if diameter is None:
            warnings.append(f"{row['stable_nodule_id']}: BBox size unavailable; study spacing or BBox coordinates missing/invalid")

    mean_ranks: dict[str, float | None] = {}
    for attribute in ATTRIBUTE_NAMES:
        values = []
        for row in selected:
            prediction = row["attributes"].get(attribute)
            prediction = prediction if isinstance(prediction, Mapping) else {}
            value = _finite(prediction.get("expected_rating"))
            low, high = ATTRIBUTE_RANGES[attribute]
            if value is not None and low <= value <= high:
                values.append(value)
        mean_ranks[attribute] = sum(values) / len(values) if values else None
        if len(values) != len(selected):
            warnings.append(f"Attribute rank {attribute}: available for {len(values)}/{len(selected)} malignant nodules; mean uses available valid outputs only")

    sizes = [row["bbox_diameter_mm"] for row in selected if row["bbox_diameter_mm"] is not None]
    if selected and not sizes:
        warnings.append("Mean nodule size unavailable because no malignant nodule has a valid BBox diameter in mm")
    elif len(sizes) != len(selected):
        warnings.append(f"Nodule size: available for {len(sizes)}/{len(selected)} malignant nodules; mean uses available valid BBoxes only")

    return {
        "n_malignant_nodules": len(selected),
        "spacing_available": bool(spacing_available),
        "malignant_nodule_ids": [row["stable_nodule_id"] for row in selected],
        "lung_location": _mode_with_tie_break(selected, "lung_location"),
        "anatomic_subdivision": _mode_with_tie_break(selected, "anatomic_subdivision"),
        "mean_nodule_size_mm": sum(sizes) / len(sizes) if sizes else None,
        "mean_attribute_ranks": mean_ranks,
        "warnings": sorted(set(warnings)),
        "_selected_nodules": selected,
    }


def build_histology_features(
    clinician: Mapping[str, Any],
    form_values: Mapping[str, Any] | None,
    imaging_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build unchanged v4 feature row and source provenance; form values take precedence."""
    clinical = clinician if isinstance(clinician, Mapping) else {}
    form = form_values if isinstance(form_values, Mapping) else {}
    aggregated = aggregate_malignant_nodules(imaging_contract)
    imaging_values = {
        "lung_location": aggregated["lung_location"],
        "anatomic_subdivision": aggregated["anatomic_subdivision"],
        # No imaging-only mapping exists for prior/other cancer history.
        "other_malignancy_history": None,
    }
    values = {
        "age_at_diagnosis": clinical.get("age_at_baseline_years"),
        "sex": {"female": "FEMALE", "male": "MALE"}.get(clinical.get("sex_at_baseline")),
        "race": {
            "american_indian_or_alaska_native": "AMERICAN INDIAN OR ALASKA NATIVE",
            "asian": "ASIAN",
            "black_or_african_american": "BLACK OR AFRICAN AMERICAN",
            "white": "WHITE",
        }.get(clinical.get("race_at_baseline")),
        # Ethnicity is intentionally not collected for Histology v4; retain the
        # unchanged model column as missing so fitted training-time imputation applies.
        "ethnicity": None,
        "smoking_history_code": None,
        "pack_years": None,
        "smoking_year_started": None,
        "smoking_year_stopped": None,
        "dlco_percent_predicted": None,
        "fev1_fvc_post_percent": None,
        "fev1_fvc_pre_percent": None,
        "fev1_percent_predicted_post": None,
        "fev1_percent_predicted_pre": None,
        "pulmonary_function_test_performed": None,
        "ecog_preoperative": None,
        "karnofsky_preoperative": None,
        **imaging_values,
    }
    # Existing structured history is a defensible form source for prior cancer.
    if clinical.get("medical_history_status") == "reported":
        conditions = clinical.get("medical_history_conditions") or []
        values["other_malignancy_history"] = (
            "Yes, History of Prior Malignancy" if "prior_cancer" in conditions else "No"
        )
    # Explicit v4 form values override both basic form mappings and imaging-derived fields.
    for key, value in form.items():
        if key in values and value is not None:
            values[key] = value

    warnings = list(aggregated["warnings"])
    warnings.append("other_malignancy_history is not inferred from imaging; multiple nodules, bilateral findings, and Brock scores do not establish prior cancer")
    for field in ("lung_location", "anatomic_subdivision", "other_malignancy_history"):
        form_value = form.get(field)
        imaging_value = imaging_values.get(field)
        if form_value is not None and imaging_value is not None and form_value != imaging_value:
            warnings.append(f"{field}: clinician-entered form value {form_value!r} overrides imaging-derived value {imaging_value!r}")
        value = values.get(field)
        if value is not None and field in V4_CATEGORIES and value not in V4_CATEGORIES[field]:
            warnings.append(f"{field}: value {value!r} is not a fitted v4 encoder category; set to missing for fitted imputation")
            values[field] = None

    provenance = {
        key: ("form" if key in form and form.get(key) is not None else "imaging" if key in imaging_values and imaging_values[key] is not None else "form" if values.get(key) is not None else "imputed")
        for key in values
    }
    # Values that came from legacy clinician fields are still form provenance.
    for key in ("age_at_diagnosis", "sex", "race", "ethnicity", "other_malignancy_history"):
        if values.get(key) is not None and provenance.get(key) == "imputed":
            provenance[key] = "form"
    for key, value in values.items():
        if value is None:
            provenance[key] = "imputed"
    return {
        "values": values,
        "provenance": provenance,
        "imaging_features_available": aggregated["n_malignant_nodules"] > 0,
        "imaging_derived": {key: value for key, value in aggregated.items() if key != "_selected_nodules"},
        "warnings": sorted(set(warnings)),
    }
