from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

IMAGING_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = IMAGING_DIR / "schemas" / "imaging_clinical_contract_v1.schema.json"
CONTRACT_FILENAME = "imaging_clinical_contract_v1.json"
ATTRIBUTE_KEYS = (
    "subtlety", "internal_structure", "calcification", "sphericity",
    "margin", "lobulation", "spiculation", "texture",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite coordinates.")
    return result


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        return None
    return result


def _bounds_lps(
    start_zyx: np.ndarray,
    end_zyx_exclusive: np.ndarray,
    origin_mm: np.ndarray,
    spacing_zyx_mm: np.ndarray,
    direction_lps: np.ndarray,
) -> dict[str, list[float]]:
    if direction_lps.shape != (3, 3) or not np.isfinite(direction_lps).all():
        raise ValueError("direction_lps must be a finite 3x3 matrix.")
    if np.any(end_zyx_exclusive <= start_zyx):
        raise ValueError("Nodule voxel bbox must have positive extent.")
    corners = np.asarray([
        origin_mm + direction_lps @ (np.asarray([z, y, x]) * spacing_zyx_mm)
        for z in (start_zyx[0], end_zyx_exclusive[0])
        for y in (start_zyx[1], end_zyx_exclusive[1])
        for x in (start_zyx[2], end_zyx_exclusive[2])
    ])
    return {
        "min_lps_mm": corners.min(axis=0).tolist(),
        "max_lps_mm": corners.max(axis=0).tolist(),
    }


def _record_refs(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    explanations = record.get("explanations")
    explanations = explanations if isinstance(explanations, dict) else {}
    gradcam = explanations.get("gradcam")
    gradcam = gradcam if isinstance(gradcam, dict) else {}
    shap = explanations.get("shap")
    shap = shap if isinstance(shap, dict) else {}
    return gradcam, shap


def build_contract(
    *,
    patient_id: str,
    meta: dict[str, Any],
    stage06: dict[str, Any],
) -> dict[str, Any]:
    study_uid = meta.get("study_instance_uid")
    series_uid = meta.get("series_instance_uid")
    placeholders = {"", "UNKNOWN", "N/A", "NA", "NONE", "PLACEHOLDER"}
    if not isinstance(study_uid, str) or study_uid.strip().upper() in placeholders:
        raise ValueError("Stage 01 metadata has no verified study_instance_uid.")
    if not isinstance(series_uid, str) or series_uid.strip().upper() in placeholders:
        raise ValueError("Stage 01 metadata has no verified series_instance_uid.")
    metadata_patient_id = meta.get("patient_id")
    if isinstance(metadata_patient_id, str) and metadata_patient_id not in {"", patient_id}:
        raise ValueError(
            f"Requested patient_id {patient_id!r} does not match Stage 01 metadata patient_id "
            f"{metadata_patient_id!r}."
        )
    origin_mm = _vector(meta.get("origin_mm"), "origin_mm")
    spacing = _vector(meta.get("spacing_zyx_mm"), "spacing_zyx_mm")
    if np.any(spacing <= 0):
        raise ValueError("spacing_zyx_mm must contain three positive values.")
    direction = np.asarray(meta.get("direction_lps"), dtype=np.float64)
    if direction.shape != (3, 3) or not np.isfinite(direction).all():
        raise ValueError("direction_lps must be a finite 3x3 matrix.")
    records = stage06.get("records")
    if not isinstance(records, list):
        raise ValueError("Stage 06 output must contain a records list.")
    nodules = []
    seen_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Stage 06 records must be objects.")
        stable_id = record.get("stable_nodule_id", record.get("nodule_id"))
        if not isinstance(stable_id, str) or not stable_id.strip():
            raise ValueError("Accepted Stage 06 record is missing stable_nodule_id.")
        if stable_id in seen_ids:
            raise ValueError(f"Duplicate stable nodule ID in Stage 06: {stable_id}.")
        seen_ids.add(stable_id)
        geometry = record.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f"Stage 06 record {stable_id} has no geometry.")
        start = _vector(geometry.get("bbox_start_zyx"), f"{stable_id}.bbox_start_zyx")
        end = _vector(geometry.get("bbox_end_zyx_exclusive"), f"{stable_id}.bbox_end_zyx_exclusive")
        centroid = _vector(geometry.get("center_zyx"), f"{stable_id}.center_zyx")
        centroid_lps = _vector(geometry.get("center_lps_mm"), f"{stable_id}.center_lps_mm")
        derived_lps = origin_mm + direction @ (centroid * spacing)
        if not np.allclose(centroid_lps, derived_lps, rtol=0.0, atol=1e-3):
            raise ValueError(
                f"Stage 06 record {stable_id} centroid LPS does not match the verified Stage 01 geometry."
            )
        detector_info = record.get("detector")
        if not isinstance(detector_info, dict):
            raise ValueError(f"Stage 06 record {stable_id} has no detector provenance.")
        attributes = record.get("attributes")
        attributes = attributes if isinstance(attributes, dict) else {}
        model_identity = record.get("model_identity")
        model_identity = model_identity if isinstance(model_identity, dict) else {}
        support_audit = model_identity.get("stage03_attribute_support_audit")
        if not isinstance(support_audit, dict):
            raise ValueError(f"Stage 06 record {stable_id} lacks the verified Stage 03 attribute support audit.")
        shaped_attributes: dict[str, dict[str, Any]] = {}
        for name in ATTRIBUTE_KEYS:
            prediction = attributes.get(name)
            if isinstance(prediction, dict):
                shaped_attributes[name] = prediction
                continue
            audit_entry = support_audit.get(name)
            if isinstance(audit_entry, dict) and audit_entry.get("status") == "unsupported_checkpoint_contract":
                shaped_attributes[name] = {
                    "status": "unsupported_checkpoint_contract",
                    "support_audit": audit_entry,
                }
                continue
            raise ValueError(
                f"Stage 06 record {stable_id} is missing checkpoint-supported attribute output {name!r}."
            )
        gradcam, shap = _record_refs(record)
        for name in ATTRIBUTE_KEYS:
            support = support_audit.get(name, {})
            if not isinstance(support, dict) or support.get("status") != "supported":
                continue
            cam = gradcam.get(name)
            if not isinstance(cam, dict) or not cam.get("patch_cam_path") or not cam.get("ct_cam_path"):
                raise ValueError(
                    f"Stage 06 record {stable_id} is missing patch/CT-space Grad-CAM references for {name!r}."
                )
        if shap.get("status") == "not_applicable":
            pass
        elif shap.get("status") == "complete":
            if not isinstance(shap.get("features"), list) or not isinstance(shap.get("feature_contributions"), dict):
                raise ValueError(f"Stage 06 record {stable_id} has an incomplete successful SHAP explanation.")
        elif shap.get("status") == "failed":
            if not shap.get("failure_reason") or not shap.get("failure_artifact"):
                raise ValueError(f"Stage 06 record {stable_id} has no explicit SHAP failure artifact.")
        else:
            raise ValueError(f"Stage 06 record {stable_id} has no explicit SHAP status.")
        uncertainty = record.get("uncertainty_and_limitations")
        uncertainty = uncertainty if isinstance(uncertainty, dict) else {}
        estimated_uncertainty = record.get("uncertainty")
        if isinstance(estimated_uncertainty, dict) and estimated_uncertainty:
            uncertainty_output = dict(estimated_uncertainty)
            uncertainty_output["status"] = uncertainty_output.get("status", "model_estimated")
            uncertainty_output["limitations"] = uncertainty
        else:
            uncertainty_output = {
                "status": "not_estimated",
                "limitations": uncertainty,
            }
        malignancy = record.get("malignancy")
        malignancy = malignancy if isinstance(malignancy, dict) else {}
        if set(malignancy) - {"brock", "selected_prediction"}:
            raise ValueError(f"Stage 06 record {stable_id} contains unsupported malignancy fields.")
        if not isinstance(malignancy.get("brock"), dict):
            raise ValueError(f"Stage 06 record {stable_id} is missing its Brock result.")
        selected_prediction = malignancy.get("selected_prediction")
        if not isinstance(selected_prediction, dict) or selected_prediction.get("method") != "brock":
            raise ValueError(f"Stage 06 record {stable_id} is missing a Brock selected_prediction.")
        quality_flags = record.get("quality_flags") or []
        if not isinstance(quality_flags, list) or any(not isinstance(flag, str) for flag in quality_flags):
            raise ValueError(f"Stage 06 record {stable_id} quality_flags must be a string list.")
        nodules.append({
            "stable_nodule_id": stable_id,
            "detector": {
                "candidate_id": str(record.get("candidate_id", stable_id)),
                "source": detector_info.get("source"),
                "source_members": detector_info.get("source_members", []),
                "vitdet_score": _finite_or_none(detector_info.get("vitdet_score")),
                "log_score": _finite_or_none(detector_info.get("log_score")),
                "raw_score": _finite_or_none(detector_info.get("score")),
                "raw_log_response": _finite_or_none(detector_info.get("log_response")),
            },
            "bbox_voxel": {
                "start_zyx": start.tolist(),
                "end_zyx_exclusive": end.tolist(),
            },
            "bbox_lps_mm": _bounds_lps(start, end, origin_mm, spacing, direction),
            "centroid_voxel": centroid.tolist(),
            "centroid_lps_mm": centroid_lps.tolist(),
            "diameter_mm": _finite_or_none(geometry.get("diameter_mm")),
            "volume_mm3": _finite_or_none(geometry.get("volume_mm3")),
            "anatomical_location": record.get("anatomical_location") or {},
            "attributes": shaped_attributes,
            "malignancy": malignancy,
            "uncertainty": uncertainty_output,
            "quality_flags": quality_flags,
            "model_versions": model_identity,
            "gradcam_refs": gradcam,
            "shap_refs": shap,
            "artifact_refs": {
                "stage06_record": record.get("record_path"),
                "stage03_prediction": model_identity.get("stage03_prediction_json"),
                "stage04_prediction": shap.get("source_prediction_json"),
            },
        })
    contract = {
        "contract_version": "1.0.0",
        "patient_id": patient_id,
        "study_id": study_uid,
        "study_uid": study_uid,
        "series_uid": series_uid,
        "coordinate_metadata": {
            "array_axes": ["z", "y", "x"],
            "coordinate_system": "DICOM_LPS",
            "origin_mm": origin_mm.tolist(),
            "spacing_zyx_mm": spacing.tolist(),
            "direction_lps": direction.tolist(),
        },
        "nodules": nodules,
    }
    validate_contract(contract)
    return contract


def validate_contract(contract: dict[str, Any]) -> None:
    schema = read_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(contract), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors[:10]
        )
        raise ValueError(f"Imaging contract failed JSON Schema validation: {details}")


def create_contract_file(patient_root: Path, patient_id: str, stage06_path: Path) -> Path:
    meta_path = patient_root / "meta.json"
    contract = build_contract(
        patient_id=patient_id,
        meta=read_json(meta_path),
        stage06=read_json(stage06_path),
    )
    for nodule in contract["nodules"]:
        nodule["artifact_refs"].update({
            "stage06_record": stage06_path.resolve().relative_to(patient_root.resolve()).as_posix(),
            "ct_volume": read_json(stage06_path).get("ct_volume"),
            "html_report": (patient_root.parent / "reports" / patient_id / "index.html").as_posix(),
        })
    output = patient_root / CONTRACT_FILENAME
    output.write_text(json.dumps(contract, indent=2, allow_nan=False), encoding="utf-8")
    return output
