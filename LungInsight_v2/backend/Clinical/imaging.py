from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

CLINICAL_DIR = Path(__file__).resolve().parent
IMAGING_SCHEMA_PATH = (
    CLINICAL_DIR.parent / "Imaging" / "schemas" / "imaging_clinical_contract_v1.schema.json"
)


class ImagingEvidenceError(ValueError):
    """Raised when Imaging evidence cannot pass the Clinical input gate."""


def _parse_aware_timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ImagingEvidenceError(f"{name} must be an ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ImagingEvidenceError(f"{name} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ImagingEvidenceError(f"{name} must include a timezone.")
    return parsed


def validate_imaging_evidence(
    evidence: dict[str, Any],
    *,
    patient_id: str,
    baseline_at: str,
) -> dict[str, Any]:
    required = {
        "contract",
        "source_timestamp",
        "baseline_at",
        "gate_status",
        "imaging_pipeline_version",
        "imaging_model_versions",
    }
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise ImagingEvidenceError("Imaging evidence does not match the required Clinical envelope.")
    contract = evidence["contract"]
    if not isinstance(contract, dict):
        raise ImagingEvidenceError("Imaging contract must be a JSON object.")
    if not IMAGING_SCHEMA_PATH.is_file():
        raise ImagingEvidenceError("The versioned Imaging contract schema is unavailable.")
    with IMAGING_SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(contract),
        key=lambda error: list(error.path),
    )
    if errors:
        detail = "; ".join(
            f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors[:5]
        )
        raise ImagingEvidenceError(f"Imaging contract validation failed: {detail}")
    if contract["contract_version"] != "1.0.0":
        raise ImagingEvidenceError("Unsupported Imaging-to-Clinical contract version.")
    if contract["patient_id"] != patient_id:
        raise ImagingEvidenceError("Imaging patient_id does not match the Clinical request.")
    if evidence["gate_status"] != "passed":
        raise ImagingEvidenceError("Imaging gate did not pass.")
    pipeline_version = evidence["imaging_pipeline_version"]
    versions = evidence["imaging_model_versions"]
    if not isinstance(pipeline_version, str) or not pipeline_version.strip():
        raise ImagingEvidenceError("Imaging pipeline version is required.")
    if not isinstance(versions, dict) or not versions:
        raise ImagingEvidenceError("Imaging model version provenance is required.")

    source_time = _parse_aware_timestamp(evidence["source_timestamp"], "source_timestamp")
    evidence_baseline = _parse_aware_timestamp(evidence["baseline_at"], "imaging baseline_at")
    request_baseline = _parse_aware_timestamp(baseline_at, "Clinical baseline_at")
    if request_baseline > datetime.now(timezone.utc):
        raise ImagingEvidenceError("Clinical prediction baseline cannot be in the future.")
    if source_time != evidence_baseline or evidence_baseline != request_baseline:
        raise ImagingEvidenceError(
            "Imaging source time and baseline timestamps must match the Clinical prediction baseline; "
            "no approved image-freshness interval is defined."
        )

    nodules = contract["nodules"]
    diameters = [
        float(nodule["diameter_mm"])
        for nodule in nodules
        if nodule.get("diameter_mm") is not None
    ]
    return {
        "patient_id": contract["patient_id"],
        "study_id": contract["study_id"],
        "study_uid": contract["study_uid"],
        "series_uid": contract["series_uid"],
        "contract_version": contract["contract_version"],
        "source_timestamp": evidence["source_timestamp"],
        "baseline_at": evidence["baseline_at"],
        "gate_status": evidence["gate_status"],
        "imaging_pipeline_version": pipeline_version,
        "imaging_model_versions": versions,
        "imaging_candidate_count": len(nodules),
        "largest_reported_nodule_diameter_mm": max(diameters) if diameters else None,
        "malignant_nodule_count": None,
        "malignant_nodule_count_status": "unavailable_no_validated_malignant_status_field",
        "ranking_scores_used_as_malignancy": False,
        "training_feature_status": "context_only_not_available_in_development_cohorts",
    }
