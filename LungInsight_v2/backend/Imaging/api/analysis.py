from __future__ import annotations

import json
import io
import re
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .contract import (
    CONTRACT_FILENAME,
    build_study_summary,
    create_contract_file,
    read_json,
    validate_contract,
)
from .dicom_upload import get_dicom_job

PATIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
IMAGING_DIR = Path(__file__).resolve().parents[1]
from pipeline_schema import get_required, load_schema

PIPELINE_SCHEMA = load_schema(IMAGING_DIR / "stages" / "01_hu_and_crop.py")
STAGE06_PATHS = get_required(PIPELINE_SCHEMA, "paths.stage_06", expected_type=dict)
STAGE06_SCHEMA = get_required(PIPELINE_SCHEMA, "stage_06_full_ct_gradcam", expected_type=dict)
_RUNS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()
_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _public(run: dict[str, Any]) -> dict[str, Any]:
    result = dict(run)
    result.pop("_clinical_inputs", None)
    result.pop("_process", None)
    result.pop("log_path", None)
    result.pop("contract_path", None)
    result.pop("result_path", None)
    result.pop("_state_path", None)
    return result


def _save_run(run: dict[str, Any]) -> None:
    state_path = run.get("_state_path")
    if not isinstance(state_path, str):
        return
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {key: value for key, value in run.items() if key not in {"_process", "_state_path"}}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(saved, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _set(run_id: str, **values: Any) -> None:
    with _LOCK:
        _RUNS[run_id].update(values)
        _save_run(_RUNS[run_id])


def _stage_from_line(line: str) -> tuple[str, str] | None:
    match = re.search(r"=== (Stage \d{2}[^=]+) ===", line)
    if match:
        stage = match.group(1).strip()
        number = re.search(r"Stage (\d{2})", stage).group(1)
        return f"stage{number}", f"{stage} is running."
    return None


def _record_completed_stage(run_id: str, line: str) -> None:
    match = re.search(r"\[(\d{2})/07\]\s+Stage\s+(\d{2})\s+Complete", line)
    if not match or match.group(1) != match.group(2):
        return
    stage = f"stage{match.group(1)}"
    with _LOCK:
        completed = list(_RUNS[run_id].get("completed_stages", []))
    if stage not in completed:
        completed.append(stage)
        _set(run_id, completed_stages=completed)


def _analysis_brock_inputs(clinical_inputs: dict[str, Any]) -> dict[str, Any]:
    patient: dict[str, Any] = {}
    age = clinical_inputs.get("age_at_baseline_years")
    if age is not None:
        patient["age_years"] = age
    sex = clinical_inputs.get("sex_at_baseline")
    if sex in {"female", "male"}:
        patient["sex"] = sex
    family_history = clinical_inputs.get("family_history_lung_cancer")
    if family_history in {"yes", "no"}:
        patient["family_history_lung_cancer"] = family_history == "yes"
    if clinical_inputs.get("medical_history_status") == "reported":
        conditions = clinical_inputs.get("medical_history_conditions", [])
        patient["emphysema"] = "emphysema" in conditions
    return {"patient": patient, "nodules": {}}


def _stage06_records_path(patient_root: Path) -> Path:
    return patient_root / str(STAGE06_PATHS["directory"]) / str(STAGE06_SCHEMA["record_filename"])


def _write_study_result(
    run: dict[str, Any],
    *,
    run_root: Path,
    clinical_inputs: dict[str, Any],
    status: str,
    clinical_assessment: dict[str, Any] | None = None,
    failure: str | None = None,
) -> Path:
    contract_path = run_root / CONTRACT_FILENAME
    contract = None
    if contract_path.is_file():
        contract = read_json(contract_path)
        validate_contract(contract)
    stage07_path = run_root / str(get_required(PIPELINE_SCHEMA, "paths.stage_07.report_json"))
    stage07 = read_json(stage07_path) if stage07_path.is_file() else None
    nodules = contract.get("nodules", []) if isinstance(contract, dict) else []
    eligible_ids = {row["stable_nodule_id"] for row in nodules}
    if stage07 is not None:
        report_ids = {
            row.get("stable_nodule_id")
            for row in stage07.get("eligible_nodules", [])
            if isinstance(row, dict)
        }
        if report_ids != eligible_ids:
            raise ValueError(
                "Stage 07 report and validated Imaging contract contain different eligible nodule IDs."
            )

    def relative(path: Path) -> str | None:
        try:
            return path.resolve().relative_to(run_root.resolve()).as_posix() if path.is_file() else None
        except ValueError:
            return None

    eligible_nodules = []
    for nodule in nodules:
        risk = nodule["malignancy"]["brock"]
        eligible_nodules.append({
            "stable_nodule_id": nodule["stable_nodule_id"],
            "anatomical_location": nodule.get("anatomical_location"),
            "centroid_voxel": nodule["centroid_voxel"],
            "centroid_lps_mm": nodule["centroid_lps_mm"],
            "bounding_box_voxel": nodule["bbox_voxel"],
            "measurements": nodule["measurements"],
            "detection_sources": nodule["detection_sources"],
            "classification": nodule["classification"],
            "brock_risk": risk,
            "quality_status": nodule["quality_status"],
            "explanation_refs": {
                "gradcam": nodule["gradcam_refs"],
                "shap": nodule["shap_refs"],
            },
            "artifact_refs": nodule["artifact_refs"],
        })
    study = contract.get("study", {}) if isinstance(contract, dict) else {}
    errors = list(study.get("errors", [])) if isinstance(study.get("errors"), list) else []
    if failure and failure not in errors:
        errors.append(failure)
    warnings = list(study.get("warnings", [])) if isinstance(study.get("warnings"), list) else []
    result_path = run_root / f"study_result_{run['run_id']}.json"
    document = {
        "schema_version": "lunginsight-study-result-v1",
        "study_status": status,
        "created_at": _now(),
        "run": {
            "run_id": run["run_id"],
            "patient_id": run["patient_id"],
            "status": status,
            "created_at": run.get("created_at"),
            "started_at": run.get("started_at"),
            "completed_at": run.get("completed_at") or _now(),
            "completed_stages": run.get("completed_stages", []),
            "failure": failure,
        },
        "clinical_inputs": clinical_inputs,
        "study": study,
        "eligible_nodules": eligible_nodules,
        "brock_predictions": [
            {
                "stable_nodule_id": nodule["stable_nodule_id"],
                "status": nodule["brock_risk"].get("status", "unavailable"),
                "probability": nodule["brock_risk"].get("probability"),
                "missing_inputs": nodule["brock_risk"].get(
                    "features_missing",
                    nodule["brock_risk"].get("missing_inputs", []),
                ),
                "model_name": nodule["brock_risk"].get("model_name"),
                "model_version": nodule["brock_risk"].get("model_version"),
            }
            for nodule in eligible_nodules
        ],
        "clinical_assessment": clinical_assessment,
        "visualization": {
            "stage07_report": relative(stage07_path),
            "accepted_nodules_csv": relative(run_root / "07_visualization" / "nodule_report.csv"),
            "excluded_nodules_csv": relative(run_root / "07_visualization" / "excluded_outside_lung.csv"),
            "html_report": f"/api/v1/analysis/runs/{run['run_id']}/report/html",
            "stage07_artifact_refs": stage07.get("artifact_refs", {}) if isinstance(stage07, dict) else {},
        },
        "provenance": study.get("provenance", {}),
        "warnings": warnings,
        "failures": errors,
        "artifact_refs": {
            "imaging_contract": relative(contract_path),
            "stage07_report": relative(stage07_path),
            "clinical_assessment": relative(run_root / f"clinical_prediction_{run['run_id']}.json"),
            "pipeline_summary": relative(run_root / "pipeline_run_summary.json"),
            "pipeline_log": relative(run_root / f"analysis_{run['run_id']}.log"),
        },
    }
    result_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return result_path


def _run_pipeline(
    run_id: str,
    *,
    project_root: Path,
    patient_id: str,
    input_path: Path,
    output_root: Path,
    clinical_inputs: dict[str, Any],
    requested_malignancy_method: str | None,
    resume_from_stage02: bool = False,
) -> None:
    run_root = output_root / patient_id
    log_path = run_root / f"analysis_{run_id}.log"
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(project_root / "run_pipeline.ps1"),
        "-PatientId",
        patient_id,
        "-InputPatientPath",
        str(input_path),
        "-ResumeFromStage02" if resume_from_stage02 else "-RegenerateStage01",
        "-NoGui",
    ]
    try:
        run_root.mkdir(parents=True, exist_ok=True)
        _set(
            run_id,
            status="stage02_running" if resume_from_stage02 else "stage01_running",
            current_stage="stage02" if resume_from_stage02 else "stage01",
            started_at=_now(),
        )
        brock_inputs_path = run_root / f"brock_inputs_{run_id}.json"
        brock_inputs_path.write_text(
            json.dumps(_analysis_brock_inputs(clinical_inputs), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        command.extend(["-BrockInputs", str(brock_inputs_path)])
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in process.stdout or ():
                log.write(line)
                log.flush()
                _record_completed_stage(run_id, line)
                detected = _stage_from_line(line)
                if detected:
                    stage, message = detected
                    _set(
                        run_id,
                        status=f"{stage}_running",
                        current_stage=stage,
                        message=message,
                    )
            exit_code = process.wait()
        if exit_code != 0:
            summary_path = run_root / "pipeline_run_summary.json"
            summary = {}
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
                except json.JSONDecodeError:
                    summary = {}
            failure = summary.get("failure") or "The Imaging runner failed before completion."
            blocked = "blocked" in str(failure).lower()
            failure_message = str(failure)
            contract_path = None
            records_path = _stage06_records_path(run_root)
            if records_path.is_file() and (run_root / "meta.json").is_file():
                try:
                    contract_path = create_contract_file(
                        run_root,
                        patient_id,
                        records_path,
                        processing_status="blocked" if blocked else "failed",
                        run_id=run_id,
                        errors=[failure_message],
                    )
                except (OSError, ValueError, KeyError, TypeError) as contract_error:
                    failure_message = f"{failure_message}; partial Imaging contract could not be validated: {contract_error}"
            result_path = None
            try:
                partial_run = dict(_RUNS[run_id])
                partial_run["completed_at"] = _now()
                result_path = _write_study_result(
                    partial_run,
                    run_root=run_root,
                    clinical_inputs=clinical_inputs,
                    status="blocked" if blocked else "failed",
                    failure=failure_message,
                )
            except (OSError, ValueError, KeyError, TypeError) as result_error:
                failure_message = f"{failure_message}; structured result could not be written: {result_error}"
            _set(
                run_id,
                status="blocked" if blocked else "failed",
                current_stage=None,
                message=failure_message,
                error=failure_message,
                study_summary=build_study_summary(
                    run_root,
                    processing_status="blocked" if blocked else "failed",
                    run_id=run_id,
                    errors=[failure_message],
                ),
                artifact_refs={
                    "pipeline_log": str(log_path.relative_to(project_root).as_posix()),
                    "imaging_contract": str(contract_path.relative_to(run_root).as_posix()) if contract_path else None,
                    "study_result": str(result_path.relative_to(run_root).as_posix()) if result_path else None,
                },
                result_path=str(result_path.relative_to(run_root).as_posix()) if result_path else None,
                blocking_requirements=["Resolve automated Stage 01 mask or geometry validation failures"] if blocked else [],
                completed_at=_now(),
                exit_code=exit_code,
                log_path=str(log_path.relative_to(project_root).as_posix()),
            )
            return

        _set(run_id, status="imaging_complete", current_stage=None, completed_at=_now(), exit_code=0)
        records_path = _stage06_records_path(run_root)
        if not records_path.is_file():
            raise RuntimeError(f"Imaging completed without Stage 06 records at {records_path}.")
        contract_path = create_contract_file(
            run_root,
            patient_id,
            records_path,
            processing_status="completed",
            run_id=run_id,
        )
        contract = read_json(contract_path)
        validate_contract(contract)
        _set(run_id, status="clinical_running", current_stage="clinical", contract_path=str(contract_path.relative_to(project_root).as_posix()))
        from ...Clinical.inference import run_inference

        meta_schema_version = contract["study"]["provenance"]["meta_schema_version"]
        if not isinstance(meta_schema_version, str) or not meta_schema_version:
            raise RuntimeError("The validated Imaging contract is missing its metadata schema provenance.")
        response = run_inference(
            {
                "schema_version": "1.0.0",
                "patient_id": patient_id,
                "baseline_definition": "other_or_unverified",
                "requested_malignancy_method": requested_malignancy_method,
                "clinician_inputs": clinical_inputs,
                "imaging_evidence": {
                    "contract": contract,
                    "source_timestamp": _now(),
                    "scan_datetime": None,
                    "gate_status": "passed",
                    "imaging_pipeline_version": f"meta_schema-{meta_schema_version}",
                    "imaging_model_versions": contract["study"]["provenance"]["imaging_model_versions"],
                },
            }
        )
        result_path = run_root / f"clinical_prediction_{run_id}.json"
        result_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
        current_run = dict(_RUNS[run_id])
        current_run["completed_at"] = _now()
        study_result_path = _write_study_result(
            current_run,
            run_root=run_root,
            clinical_inputs=clinical_inputs,
            status="completed",
            clinical_assessment=response,
        )
        _set(
            run_id,
            status="completed",
            current_stage=None,
            clinical_prediction=response,
            study_summary=contract["study"],
            artifact_refs={
                "imaging_contract": str(contract_path.relative_to(run_root).as_posix()),
                "clinical_prediction": str(result_path.relative_to(run_root).as_posix()),
                "study_result": str(study_result_path.relative_to(run_root).as_posix()),
                "pipeline_log": str(log_path.relative_to(run_root).as_posix()),
            },
            result_path=str(study_result_path.relative_to(run_root).as_posix()),
            completed_at=_now(),
        )
    except (OSError, ValueError, KeyError, RuntimeError, TypeError, subprocess.SubprocessError) as exc:
        message = f"Analysis failed during runner, contract validation, or Clinical inference: {exc}"
        result_path = None
        if run_root.is_dir():
            try:
                partial_run = dict(_RUNS[run_id])
                partial_run["completed_at"] = _now()
                result_path = _write_study_result(
                    partial_run,
                    run_root=run_root,
                    clinical_inputs=clinical_inputs,
                    status="failed",
                    failure=message,
                )
            except (OSError, ValueError, KeyError, TypeError) as result_error:
                message = f"{message}; structured result could not be written: {result_error}"
        _set(
            run_id,
            status="failed",
            message=message,
            error=message,
            study_summary=build_study_summary(
                run_root,
                processing_status="failed",
                run_id=run_id,
                errors=[message],
            ),
            artifact_refs={
                "pipeline_log": str(log_path.relative_to(run_root).as_posix()) if log_path.exists() else None,
                "study_result": str(result_path.relative_to(run_root).as_posix()) if result_path else None,
            },
            result_path=str(result_path.relative_to(run_root).as_posix()) if result_path else None,
            completed_at=_now(),
        )


def start_analysis(
    *,
    project_root: Path,
    output_root: Path,
    patient_id: str,
    upload_job_id: str,
    clinical_inputs: dict[str, Any],
    requested_malignancy_method: str | None,
    data_provenance: str | None,
) -> dict[str, Any]:
    if not PATIENT_ID_PATTERN.fullmatch(patient_id):
        raise ValueError("Invalid patient_id.")
    upload = get_dicom_job(upload_job_id, output_root=output_root)
    if upload.get("patient_id") != patient_id:
        raise ValueError("The upload job does not belong to this patient.")
    if upload.get("status") != "preprocessing_complete":
        raise ValueError("The upload job did not complete DICOM preprocessing.")
    pipeline_input = upload.get("pipeline_input")
    if not isinstance(pipeline_input, str):
        raise ValueError("The upload job has no validated pipeline input.")
    input_path = (output_root / patient_id / pipeline_input).resolve()
    relative_parts = input_path.relative_to((output_root / patient_id).resolve()).parts
    if (
        len(relative_parts) != 4
        or relative_parts[0] != "dicom_uploads"
        or relative_parts[2] != "stage01_input"
        or relative_parts[3] != patient_id
        or not input_path.is_dir()
    ):
        raise ValueError("The upload job input is not a valid server-side artifact.")
    from ...Clinical.harmonization import validate_clinician_inputs

    validate_clinician_inputs(clinical_inputs)
    state_dir = output_root / ".analysis_runs"
    with _LOCK:
        duplicate = next(
            (
                run for run in _RUNS.values()
                if (
                    run.get("upload_job_id") == upload_job_id
                    or run.get("patient_id") == patient_id
                )
                and run.get("status") not in {"failed", "completed"}
            ),
            None,
        )
        if duplicate:
            return _public(duplicate)
        run_id = uuid.uuid4().hex
        state_path = state_dir / f"{run_id}.json"
        run = {
            "run_id": run_id,
            "upload_job_id": upload_job_id,
            "patient_id": patient_id,
            "status": "analysis_queued",
            "message": "Analysis queued for the existing operator-controlled Imaging runner.",
            "current_stage": None,
            "completed_stages": [],
            "blocking_requirements": [],
            "created_at": _now(),
            "data_provenance": data_provenance or "operator_submitted",
            "_clinical_inputs": clinical_inputs,
            "_state_path": str(state_path),
        }
        _RUNS[run_id] = run
        _save_run(run)
    threading.Thread(
        target=_run_pipeline,
        kwargs={
            "run_id": run_id,
            "project_root": project_root,
            "patient_id": patient_id,
            "input_path": input_path,
            "output_root": output_root,
            "clinical_inputs": clinical_inputs,
            "requested_malignancy_method": requested_malignancy_method,
            "resume_from_stage02": False,
        },
        daemon=True,
    ).start()
    return _public(run)


def resume_analysis(run_id: str, *, project_root: Path, output_root: Path) -> dict[str, Any]:
    run = _resolve_run(run_id, output_root)
    with _LOCK:
        if run.get("status") != "awaiting_imaging_review":
            raise ValueError("This analysis is not waiting for Stage 01 review.")
        patient_id = str(run["patient_id"])
        clinical_inputs = dict(run.get("_clinical_inputs", {}))
    patient_root = (output_root / patient_id).resolve()
    _set(run_id, status="stage02_running", current_stage="stage02", message="Resuming from Stage 02 after automated Stage 01 validation.")
    threading.Thread(
        target=_run_pipeline,
        kwargs={
            "run_id": run_id,
            "project_root": project_root,
            "patient_id": patient_id,
            "input_path": patient_root,
            "output_root": output_root,
            "clinical_inputs": clinical_inputs,
            "requested_malignancy_method": "brock",
            "resume_from_stage02": True,
        },
        daemon=True,
    ).start()
    return _public(_RUNS[run_id])


def _resolve_run(run_id: str, output_root: Path) -> dict[str, Any]:
    with _LOCK:
        run = _RUNS.get(run_id)
        if run is not None:
            return run
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise KeyError(run_id)
    path = output_root / ".analysis_runs" / f"{run_id}.json"
    try:
        run = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise KeyError(run_id) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Persisted analysis run {run_id} is invalid.") from exc
    run["_state_path"] = str(path)
    if run.get("status") in {
        "analysis_queued", "stage01_running", "stage02_running", "stage03_running",
        "stage04_running", "stage05_running", "stage06_running", "clinical_running",
    }:
        run.update({
            "status": "interrupted",
            "current_stage": None,
            "message": "The service restarted while this analysis was active; completion was not confirmed.",
            "error": "Analysis process state was lost during service restart.",
            "completed_at": _now(),
        })
        run["study_summary"] = build_study_summary(
            output_root / str(run.get("patient_id", "")),
            processing_status="interrupted",
            run_id=run_id,
            errors=[run["error"]],
        )
        _save_run(run)
    with _LOCK:
        _RUNS[run_id] = run
    return run


def get_analysis(run_id: str, *, output_root: Path) -> dict[str, Any]:
    return _public(_resolve_run(run_id, output_root))


def _run_or_raise(run_id: str, output_root: Path) -> dict[str, Any]:
    return dict(_resolve_run(run_id, output_root))


def _viewer_root(run: dict[str, Any], output_root: Path) -> Path:
    patient_id = str(run["patient_id"])
    root = (output_root / patient_id).resolve()
    if root.parent != output_root.resolve():
        raise ValueError("Invalid analysis output association.")
    return root


def _viewer_arrays(root: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    volume_path = root / "volume_hu_cropped.npy"
    if not volume_path.is_file():
        volume_path = root / "volume_hu.npy"
    if not volume_path.is_file():
        raise FileNotFoundError("The current run has no CT volume available for viewing.")
    volume = np.load(volume_path, mmap_mode="r")
    lung = np.load(root / "lung_mask.npy", mmap_mode="r") if (root / "lung_mask.npy").is_file() else None
    heart = np.load(root / "heart_mask.npy", mmap_mode="r") if (root / "heart_mask.npy").is_file() else None
    if volume.ndim != 3 or not np.issubdtype(volume.dtype, np.number):
        raise ValueError("The current run CT volume is invalid.")
    return volume, lung, heart


def viewer_metadata(run_id: str, *, output_root: Path) -> dict[str, Any]:
    run = _run_or_raise(run_id, output_root)
    root = _viewer_root(run, output_root)
    volume, lung, heart = _viewer_arrays(root)
    nodules: list[dict[str, Any]] = []
    gradcam_options: list[dict[str, str]] = []
    contract_path = root / CONTRACT_FILENAME
    if contract_path.is_file():
        try:
            contract = read_json(contract_path)
            validate_contract(contract)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Stored Imaging contract is invalid for run {run_id}.") from exc
        for nodule in contract.get("nodules", []):
            bbox = nodule.get("bbox_voxel", {})
            nodules.append({
                "stable_nodule_id": nodule.get("stable_nodule_id"),
                "eligibility_status": nodule.get("eligibility_status"),
                "quality_status": nodule.get("quality_status"),
                "geometry": {
                    "bbox_start_zyx": bbox.get("start_zyx"),
                    "bbox_end_zyx_exclusive": bbox.get("end_zyx_exclusive"),
                    "center_zyx": nodule.get("centroid_voxel"),
                    "center_lps_mm": nodule.get("centroid_lps_mm"),
                    "diameter_mm": nodule.get("measurements", {}).get("diameter_mm"),
                },
                "anatomical_location": nodule.get("anatomical_location"),
                "classification": nodule.get("classification"),
                "malignancy": nodule.get("malignancy"),
                "input_completeness": nodule.get("input_completeness"),
                "detection_sources": nodule.get("detection_sources"),
                "artifact_refs": nodule.get("artifact_refs"),
                "gradcam_refs": nodule.get("gradcam_refs"),
            })
            for attribute, reference in (nodule.get("gradcam_refs") or {}).items():
                if isinstance(reference, dict):
                    cam_path = reference.get("ct_cam_path")
                    if isinstance(cam_path, str) and (root / cam_path).resolve().is_file():
                        gradcam_options.append({
                            "stable_nodule_id": str(nodule.get("stable_nodule_id")),
                            "attribute": str(attribute),
                        })
    return {
        "run_id": run_id,
        "patient_id": run["patient_id"],
        "status": run.get("status"),
        "study": contract.get("study") if contract_path.is_file() else run.get("study_summary"),
        "shape_zyx": list(volume.shape),
        "window_hu": {"level": -600, "width": 1500},
        "overlays": {
            "lung_mask": lung is not None and lung.shape == volume.shape,
            "heart_mask": heart is not None and heart.shape == volume.shape,
        },
        "nodules": nodules,
        "gradcam_options": gradcam_options,
        "slice_url_template": f"/api/v1/analysis/runs/{run_id}/viewer/slices/{{index}}",
    }


def render_viewer_slice(
    run_id: str,
    index: int,
    *,
    output_root: Path,
    lung_overlay: bool = False,
    heart_overlay: bool = False,
    gradcam_nodule_id: str | None = None,
    gradcam_attribute: str | None = None,
) -> bytes:
    run = _run_or_raise(run_id, output_root)
    root = _viewer_root(run, output_root)
    volume, lung, heart = _viewer_arrays(root)
    if index < 0 or index >= volume.shape[0]:
        raise IndexError("Slice index is outside the current volume.")
    pixels = np.asarray(volume[index], dtype=np.float32)
    level, width = -600.0, 1500.0
    gray = np.clip((pixels - (level - width / 2)) / width * 255, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    if lung_overlay and lung is not None and lung.shape == volume.shape:
        mask = np.asarray(lung[index], dtype=bool)
        rgb[mask, 1] = np.maximum(rgb[mask, 1], 150)
    if heart_overlay and heart is not None and heart.shape == volume.shape:
        mask = np.asarray(heart[index], dtype=bool)
        rgb[mask, 0] = 220
        rgb[mask, 1] = np.minimum(rgb[mask, 1], 80)
    if gradcam_nodule_id and gradcam_attribute:
        contract_path = root / CONTRACT_FILENAME
        if contract_path.is_file():
            contract = read_json(contract_path)
            validate_contract(contract)
            reference = next(
                (
                    nodule.get("gradcam_refs", {}).get(gradcam_attribute, {})
                    for nodule in contract.get("nodules", [])
                    if nodule.get("stable_nodule_id") == gradcam_nodule_id
                ),
                {},
            )
            cam_path = reference.get("ct_cam_path") if isinstance(reference, dict) else None
            if isinstance(cam_path, str):
                resolved = (root / cam_path).resolve()
                try:
                    resolved.relative_to(root)
                except ValueError:
                    resolved = None
                if resolved is not None and resolved.is_file():
                    cam = np.load(resolved, mmap_mode="r")
                    if cam.ndim == 3 and cam.shape == volume.shape:
                        activation = np.asarray(cam[index], dtype=np.float32)
                        scale = float(np.max(activation))
                        if scale > 0:
                            alpha = np.clip(activation / scale, 0, 1)[..., None]
                            heat = np.zeros_like(rgb)
                            heat[..., 0] = 255
                            rgb = (rgb * (1 - alpha * 0.55) + heat * alpha * 0.55).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
