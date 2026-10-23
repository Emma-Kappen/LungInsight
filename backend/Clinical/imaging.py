from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

CLINICAL_DIR = Path(__file__).resolve().parent
IMAGING_SCHEMA_PATH = (
    CLINICAL_DIR.parent / "Imaging" / "schemas" / "imaging_clinical_contract_v2.schema.json"
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
) -> dict[str, Any]:
    required = {
        "contract",
        "source_timestamp",
        "scan_datetime",
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
    if contract["contract_version"] != "2.0.0":
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

    _parse_aware_timestamp(evidence["source_timestamp"], "source_timestamp")
    scan_datetime = evidence["scan_datetime"]
    if scan_datetime is not None:
        _parse_aware_timestamp(scan_datetime, "scan_datetime")

    nodules = contract["nodules"]
    study = contract["study"]
    if study["processing_status"] != "completed":
        raise ImagingEvidenceError(
            f"Imaging processing is {study['processing_status']}; incomplete or failed studies cannot enter Clinical inference."
        )
    if study["preprocessing_status"] != "completed" or study["mask_validation_status"] not in {
        "validated",
        "validated_automated",
        "validated_visual_review",
    }:
        raise ImagingEvidenceError("Imaging preprocessing and automated mask-quality validation must both be complete.")
    if study["eligible_nodule_count"] != len(nodules):
        raise ImagingEvidenceError("Study eligible_nodule_count does not match the persisted nodule records.")
    brock_counts = study["brock_prediction_counts"]
    if sum(brock_counts.values()) != len(nodules):
        raise ImagingEvidenceError("Study Brock prediction counts do not reconcile with the nodule records.")
    model_versions = study["provenance"]["imaging_model_versions"]
    if not isinstance(model_versions, dict) or not model_versions:
        raise ImagingEvidenceError("Imaging model version provenance is required.")
    if versions != model_versions:
        raise ImagingEvidenceError("Clinical Imaging model versions do not match the validated Imaging contract.")
    malignancy_methods: set[str] = set()
    for nodule in nodules:
        if nodule["eligibility_status"] != "eligible":
            raise ImagingEvidenceError("Only explicitly eligible nodules may enter Clinical inference.")
        malignancy = nodule.get("malignancy", {})
        selected = malignancy.get("selected_prediction", {}) if isinstance(malignancy, dict) else {}
        method = selected.get("method") if isinstance(selected, dict) else None
        if method != "brock":
            raise ImagingEvidenceError(
                "Every nodule must carry a Brock selected_prediction; unsupported malignancy methods are rejected."
            )
        status = selected.get("status")
        probability = selected.get("probability")
        if status == "available" and not isinstance(probability, (int, float)):
            raise ImagingEvidenceError("An available Brock prediction must contain a numeric probability.")
        if status != "available" and probability is not None:
            raise ImagingEvidenceError(
                "Incomplete or invalid Brock predictions must not contain a numeric probability."
            )
        malignancy_methods.add(method)
    diameters = [
        float(nodule["measurements"]["diameter_mm"])
        for nodule in nodules
        if nodule["measurements"].get("diameter_mm") is not None
    ]
    return {
        "patient_id": contract["patient_id"],
        "study_id": contract["study_id"],
        "study_uid": contract["study_uid"],
        "series_uid": contract["series_uid"],
        "contract_version": contract["contract_version"],
        "source_timestamp": evidence["source_timestamp"],
        "scan_datetime": scan_datetime,
        "gate_status": evidence["gate_status"],
        "processing_status": study["processing_status"],
        "preprocessing_status": study["preprocessing_status"],
        "mask_validation_status": study["mask_validation_status"],
        "raw_detection_counts": study["raw_detection_counts"],
        "fused_candidate_count": study["fused_candidate_count"],
        "excluded_candidate_count": study["excluded_candidate_count"],
        "eligible_nodule_count": study["eligible_nodule_count"],
        "brock_prediction_counts": brock_counts,
        "warnings": study["warnings"],
        "errors": study["errors"],
        "imaging_pipeline_version": pipeline_version,
        "imaging_model_versions": versions,
        "imaging_candidate_count": len(nodules),
        "largest_reported_nodule_diameter_mm": max(diameters) if diameters else None,
        "malignant_nodule_count": None,
        "malignant_nodule_count_status": "unavailable_no_validated_malignant_status_field",
        "ranking_scores_used_as_malignancy": False,
        "training_feature_status": "context_only_not_available_in_development_cohorts",
        "malignancy_methods": sorted(malignancy_methods),
    }
