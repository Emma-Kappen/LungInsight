from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

IMAGING_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = IMAGING_DIR / "schemas" / "imaging_clinical_contract_v2.schema.json"
CONTRACT_FILENAME = "imaging_clinical_contract_v2.json"
CONTRACT_VERSION = "2.0.0"
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


def _read_artifact(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        return {"_artifact_read_error": f"{path.name}: {exc}"}


def _relative_refs(patient_root: Path, paths: dict[str, Path]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name, path in paths.items():
        try:
            result[name] = (
                path.resolve().relative_to(patient_root.resolve()).as_posix()
                if path.is_file()
                else None
            )
        except ValueError:
            result[name] = None
    return result


def build_study_summary(
    patient_root: Path,
    *,
    stage06: dict[str, Any] | None = None,
    processing_status: str | None = None,
    run_id: str | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    stage02 = patient_root / "02_detect_and_patch"
    stage03 = patient_root / "03_nodule_classification"
    stage04 = patient_root / "04_malignancy_prediction"
    stage06 = stage06 or _read_artifact(patient_root / "06_full_ct_gradcam" / "full_ct_records.json") or {}
    metadata = _read_artifact(patient_root / "meta.json") or {}
    mask_report = _read_artifact(patient_root / "mask_quality_report.json") or {}
    detector_metadata = _read_artifact(stage02 / "detector_metadata.json") or {}
    filter_report = _read_artifact(stage02 / "candidate_filter_report.json") or {}
    rejected = _read_artifact(stage02 / "candidates_rejected_audit.json")
    selected = _read_artifact(stage02 / "candidates.json")
    vitdet_raw = _read_artifact(stage02 / "vitdet_raw_detections.json")
    stage04_payload = _read_artifact(stage04 / "malignancy_predictions.json") or {}
    stage04_rows = stage04_payload.get("predictions")
    records = stage06.get("records")
    if not isinstance(records, list):
        records = []
    if not isinstance(stage04_rows, list):
        stage04_rows = []

    quality_status = mask_report.get("status") or (
        metadata.get("mask_quality", {}).get("status")
        if isinstance(metadata.get("mask_quality"), dict)
        else None
    )
    preprocessing_paths = (
        "meta.json",
        "volume_hu_cropped.npy",
        "lung_mask.npy",
        "heart_mask.npy",
        "mask_quality_report.json",
    )
    preprocessing_artifacts_present = all(
        (patient_root / name).is_file() for name in preprocessing_paths
    )
    run_summary = _read_artifact(patient_root / "pipeline_run_summary.json")
    preprocessing_status = (
        "blocked" if quality_status == "blocked"
        else "completed" if preprocessing_artifacts_present and quality_status in {"validated_automated", "ready_for_review", "validated"}
        else "failed" if isinstance(run_summary, dict) and run_summary.get("status") == "failed"
        else "incomplete" if any((patient_root / name).exists() for name in preprocessing_paths)
        else "not_started"
    )
    reported_attestation = filter_report.get("review_attestation")
    attestation_verified = (
        isinstance(reported_attestation, dict)
        and reported_attestation.get("verified") is True
    )
    mask_validation_status = (
        "validated_visual_review" if attestation_verified
        else "validated_automated" if quality_status in {"validated_automated", "ready_for_review", "validated"}
        else "blocked" if quality_status == "blocked"
        else "unknown" if metadata or mask_report else "not_run"
    )

    def record_count(value: Any, key: str | None = None) -> int | None:
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict) and key and isinstance(value.get(key), list):
            return len(value[key])
        return None

    overflow_rows = rejected.get("eligible_not_selected") if isinstance(rejected, dict) else None
    rejected_rows = rejected.get("rejected_candidates") if isinstance(rejected, dict) else None
    fused_count = (
        len(selected) + len(rejected_rows) + len(overflow_rows)
        if isinstance(selected, list)
        and isinstance(rejected_rows, list)
        and isinstance(overflow_rows, list)
        else None
    )
    excluded_count = None
    if isinstance(rejected_rows, list) and isinstance(overflow_rows, list):
        excluded_count = len(rejected_rows) + len(overflow_rows)
    stage06_path = patient_root / "06_full_ct_gradcam" / "full_ct_records.json"
    eligible_count = len(records) if stage06_path.is_file() else record_count(selected)
    brock_counts = {"completed": 0, "incomplete": 0, "failed": 0, "not_run": 0}
    for row in stage04_rows:
        if not isinstance(row, dict):
            brock_counts["failed"] += 1
            continue
        brock = row.get("brock")
        status = brock.get("status") if isinstance(brock, dict) else None
        if status == "available":
            brock_counts["completed"] += 1
        elif status == "incomplete":
            brock_counts["incomplete"] += 1
        else:
            brock_counts["failed"] += 1
    brock_counts["not_run"] = max(0, eligible_count - len(stage04_rows)) if eligible_count is not None else 0

    warnings: list[str] = []
    for source in (mask_report, filter_report):
        values = source.get("warnings", source.get("problems", [])) if isinstance(source, dict) else []
        if isinstance(values, list):
            warnings.extend(str(value) for value in values)
    run_failure = run_summary.get("failure") if isinstance(run_summary, dict) else None
    error_list = [str(value) for value in (errors or [])]
    if run_failure and processing_status != "completed":
        error_list.append(str(run_failure))
    for name, payload in (
        ("meta.json", metadata),
        ("mask_quality_report.json", mask_report),
        ("detector_metadata.json", detector_metadata),
        ("candidate_filter_report.json", filter_report),
        ("stage04 predictions", stage04_payload),
        ("rejected candidate audit", rejected),
        ("selected candidates", selected),
        ("raw ViTDet3D detections", vitdet_raw),
    ):
        if isinstance(payload, dict) and payload.get("_artifact_read_error"):
            error_list.append(str(payload["_artifact_read_error"]))
    if processing_status is None:
        processing_status = (
            run_summary.get("status") if isinstance(run_summary, dict) else None
        ) or ("completed" if stage06_path.is_file() else "incomplete")
    status_map = {
        "analysis_queued": "running",
        "stage01_running": "running",
        "stage02_running": "running",
        "stage03_running": "running",
        "stage04_running": "running",
        "stage05_running": "running",
        "stage06_running": "running",
        "clinical_running": "running",
        "imaging_complete": "completed",
        "completed": "completed",
        "failed": "failed",
        "awaiting_imaging_review": "blocked",
        "interrupted": "interrupted",
    }
    processing_status = status_map.get(str(processing_status), str(processing_status))
    if processing_status not in {"not_started", "running", "completed", "incomplete", "blocked", "failed", "interrupted"}:
        processing_status = "incomplete"
    stage03_payload = _read_artifact(stage03 / "predictions.json") or {}
    stage03_metadata = stage03_payload.get("checkpoint_metadata")
    detector = detector_metadata.get("detector")
    detector = detector if isinstance(detector, dict) else {}
    model_versions: dict[str, Any] = {
        "stage02": {
            "name": detector.get("name"),
            "model": detector.get("model"),
            "checkpoint": detector.get("checkpoint"),
            "version": detector.get("model_version"),
        },
    }
    if isinstance(stage03_metadata, dict):
        model_versions["stage03"] = {
            name: value.get("sha256")
            for name, value in stage03_metadata.items()
            if isinstance(value, dict) and value.get("sha256")
        }
    if stage04_payload:
        model_versions["stage04"] = stage04_payload.get("model_version")
    artifact_paths = {
        "stage01_metadata": patient_root / "meta.json",
        "mask_quality_report": patient_root / "mask_quality_report.json",
        "mask_review_attestation": patient_root / "heart_mask_review_attestation.json",
        "vitdet_raw_detections": stage02 / "vitdet_raw_detections.json",
        "stage02_filter_report": stage02 / "candidate_filter_report.json",
        "stage02_selected_candidates": stage02 / "candidates.json",
        "stage02_excluded_candidates": stage02 / "candidates_rejected_audit.json",
        "stage02_patch_manifest": stage02 / "patches" / "patches_manifest.json",
        "stage03_predictions": stage03 / "predictions.json",
        "stage04_predictions": stage04 / "malignancy_predictions.json",
        "stage06_records": stage06_path,
        "pipeline_run_summary": patient_root / "pipeline_run_summary.json",
    }
    return {
        "processing_status": processing_status,
        "preprocessing_status": preprocessing_status,
        "mask_validation_status": mask_validation_status,
        "raw_detection_counts": {
            "ViTDet3D": record_count(vitdet_raw, "detections"),
        },
        "fused_candidate_count": fused_count,
        "excluded_candidate_count": excluded_count,
        "eligible_nodule_count": eligible_count,
        "brock_prediction_counts": brock_counts,
        "warnings": sorted(set(warnings)),
        "errors": error_list,
        "provenance": {
            "run_id": run_id,
            "meta_schema_version": metadata.get("meta_schema_version"),
            "imaging_model_versions": model_versions,
            "review_attestation_verified": attestation_verified,
            "mask_visual_review_status": "verified" if attestation_verified else "not_performed",
        },
        "artifact_refs": _relative_refs(patient_root, artifact_paths),
    }


def build_contract(
    *,
    patient_id: str,
    meta: dict[str, Any],
    stage06: dict[str, Any],
    study_summary: dict[str, Any] | None = None,
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
    if (
        direction.shape != (3, 3)
        or not np.isfinite(direction).all()
        or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4)
        or not np.isclose(abs(np.linalg.det(direction)), 1.0, atol=1e-4)
    ):
        raise ValueError("direction_lps must be a finite orthonormal 3x3 matrix.")
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
        source_values = {
            source
            for source in (
                detector_info.get("source"),
                *(member.get("source") for member in detector_info.get("source_members", []) if isinstance(member, dict)),
            )
            if source in {"ViTDet3D"}
        }
        selected_prediction_status = selected_prediction.get("status")
        if selected_prediction_status != malignancy["brock"].get("status"):
            raise ValueError(f"Stage 06 record {stable_id} has inconsistent Brock status fields.")
        nodules.append({
            "stable_nodule_id": stable_id,
            "eligibility_status": "eligible",
            "quality_status": "passed_with_warnings" if quality_flags else "passed",
            "detector": {
                "candidate_id": str(record.get("candidate_id", stable_id)),
                "source": detector_info.get("source"),
                "source_members": detector_info.get("source_members", []),
                "vitdet_score": _finite_or_none(detector_info.get("vitdet_score")),
                "raw_score": _finite_or_none(detector_info.get("score")),
            },
            "detection_sources": sorted(source_values),
            "bbox_voxel": {
                "start_zyx": start.tolist(),
                "end_zyx_exclusive": end.tolist(),
            },
            "bbox_lps_mm": _bounds_lps(start, end, origin_mm, spacing, direction),
            "centroid_voxel": centroid.tolist(),
            "centroid_lps_mm": centroid_lps.tolist(),
            "measurements": {
                "diameter_mm": _finite_or_none(geometry.get("diameter_mm")),
                "volume_mm3": _finite_or_none(geometry.get("volume_mm3")),
                "source": "stage02_detector_bbox_estimate",
                "status": (
                    "detector_estimate"
                    if geometry.get("diameter_mm") is not None or geometry.get("volume_mm3") is not None
                    else "unavailable"
                ),
            },
            "anatomical_location": record.get("anatomical_location"),
            "classification": {
                "status": (
                    "partial"
                    if any(value.get("status") == "unsupported_checkpoint_contract" for value in shaped_attributes.values())
                    else "available"
                ),
                "source": "stage03_frozen_spatial_hybrid_ordinal_model",
                "attributes": shaped_attributes,
            },
            "malignancy": malignancy,
            "input_completeness": record.get("input_completeness", {}),
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
        "contract_version": CONTRACT_VERSION,
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
        "study": study_summary or {
            "processing_status": "incomplete",
            "preprocessing_status": "unknown",
            "mask_validation_status": "unknown",
            "raw_detection_counts": {"ViTDet3D": None},
            "fused_candidate_count": None,
            "excluded_candidate_count": None,
            "eligible_nodule_count": len(nodules),
            "brock_prediction_counts": {
                "completed": sum(n["malignancy"]["selected_prediction"].get("status") == "available" for n in nodules),
                "incomplete": sum(n["malignancy"]["selected_prediction"].get("status") == "incomplete" for n in nodules),
                "failed": sum(n["malignancy"]["selected_prediction"].get("status") in {"invalid_input", "unsupported", "error"} for n in nodules),
                "not_run": 0,
            },
            "warnings": [],
            "errors": [],
            "provenance": {
                "run_id": None,
                "meta_schema_version": meta.get("meta_schema_version"),
                "imaging_model_versions": {},
                "review_attestation_verified": False,
                "mask_visual_review_status": "not_performed",
            },
            "artifact_refs": {},
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


def create_contract_file(
    patient_root: Path,
    patient_id: str,
    stage06_path: Path,
    *,
    processing_status: str = "incomplete",
    run_id: str | None = None,
    errors: list[str] | None = None,
) -> Path:
    meta_path = patient_root / "meta.json"
    stage06 = read_json(stage06_path)
    contract = build_contract(
        patient_id=patient_id,
        meta=read_json(meta_path),
        stage06=stage06,
        study_summary=build_study_summary(
            patient_root,
            stage06=stage06,
            processing_status=processing_status,
            run_id=run_id,
            errors=errors,
        ),
    )
    for nodule in contract["nodules"]:
        nodule["artifact_refs"].update({
            "stage06_record": stage06_path.resolve().relative_to(patient_root.resolve()).as_posix(),
            "ct_volume": read_json(stage06_path).get("ct_volume"),
            "html_report": None,
        })
    output = patient_root / CONTRACT_FILENAME
    output.write_text(json.dumps(contract, indent=2, allow_nan=False), encoding="utf-8")
    return output
