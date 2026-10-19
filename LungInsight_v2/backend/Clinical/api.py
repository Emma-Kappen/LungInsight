from __future__ import annotations

import logging
import os
import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import File, Form, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from fastapi.staticfiles import StaticFiles

from .inference import (
    ClinicalInferenceError,
    describe_artifacts,
    load_artifacts,
    load_deployed_artifacts,
    run_inference,
    with_standalone_cancer_type,
)
from .harmonization import ClinicalValidationError, validate_clinician_inputs
from ..Imaging.api.contract import CONTRACT_FILENAME, read_json, validate_contract
from ..Imaging.api.app import OUTPUT_ROOT as IMAGING_OUTPUT_ROOT
from ..Imaging.api.dicom_upload import (
    DicomUploadError,
    get_dicom_job,
    process_dicom_upload,
)
from ..Imaging.api.analysis import (
    PATIENT_ID_PATTERN,
    BrockConflictError,
    compute_manual_brock,
    get_analysis,
    render_viewer_slice,
    start_analysis,
    resume_analysis,
    viewer_metadata,
)


logger = logging.getLogger("lunginsight.clinical_api")
_lunginsight_logger = logging.getLogger("lunginsight")
if not _lunginsight_logger.handlers:
    _console = logging.StreamHandler()
    _console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    _lunginsight_logger.addHandler(_console)
    _lunginsight_logger.setLevel(logging.INFO)


class ClinicalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(pattern=r"^1\.0\.0$")
    patient_id: str = Field(min_length=1, max_length=128)
    baseline_definition: Literal["initial_pathologic_diagnosis", "other_or_unverified"]
    requested_malignancy_method: Literal["brock"] | None = None
    clinician_inputs: dict[str, Any]
    imaging_evidence: dict[str, Any]


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(min_length=1, max_length=128)
    upload_job_id: str = Field(min_length=1, max_length=64)
    clinician_inputs: dict[str, Any]
    requested_malignancy_method: Literal["brock"] | None = "brock"
    data_provenance: str | None = None
    baseline_definition: Literal["initial_pathologic_diagnosis", "other_or_unverified"] = "other_or_unverified"


class ClinicalRecordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clinician_inputs: dict[str, Any]


class ManualBrockRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[str, Any] = Field(min_length=1)


def create_app(artifact_path: str | None = None, *, deployed: bool = False) -> FastAPI:
    """`deployed=True` follows the deployed artifact path (re-read when files change), like the Imaging runner."""
    resolved_artifact_path = artifact_path or os.environ.get("LUNGINSIGHT_CLINICAL_ARTIFACT")
    artifacts = (
        with_standalone_cancer_type(load_artifacts(resolved_artifact_path), bundle_configured=True)
        if resolved_artifact_path and not deployed
        else None
    )
    if deployed:
        load_deployed_artifacts()

    def current_artifacts() -> dict[str, Any] | None:
        return load_deployed_artifacts() if deployed else artifacts

    app = FastAPI(
        title="LungInsight Clinical Research API",
        version="0.1.0",
        description=(
            "Research-only clinical inference. Requires validated Imaging evidence and does not "
            "replace pathology, established instruments, or clinician judgment."
        ),
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = secrets.token_hex(6)
        request.state.request_id = request_id
        started = time.perf_counter()
        logger.info("request_received id=%s method=%s path=%s", request_id, request.method, request.url.path)
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "request_completed id=%s status=%s duration_ms=%.1f",
            request_id,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def log_validation_failure(request: Request, exc: RequestValidationError):
        locations = [".".join(str(part) for part in error.get("loc", ())) for error in exc.errors()]
        logger.warning(
            "request_validation_failed id=%s method=%s path=%s fields=%s",
            getattr(request.state, "request_id", "-"),
            request.method,
            request.url.path,
            locations,
        )
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(Exception)
    async def log_unhandled_exception(request: Request, exc: Exception):
        logger.error(
            "unhandled_exception id=%s method=%s path=%s error_type=%s",
            getattr(request.state, "request_id", "-"),
            request.method,
            request.url.path,
            type(exc).__name__,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Unexpected server error. See the server log for the request ID."},
        )

    @app.put("/api/v1/clinical/records/{patient_id}")
    def save_clinical_record(patient_id: str, request: ClinicalRecordRequest) -> dict[str, Any]:
        if not PATIENT_ID_PATTERN.fullmatch(patient_id):
            raise HTTPException(status_code=422, detail="Invalid patient_id.")
        try:
            normalized = validate_clinician_inputs(request.clinician_inputs)
        except ClinicalValidationError as exc:
            logger.warning("clinical_record_rejected patient_id=%s reason=%s", patient_id, exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if normalized.get("date_of_birth"):
            # Store the date of birth, not the age: age is derived from it whenever the record is used.
            normalized.pop("age_at_baseline_years", None)
            normalized.pop("age_at_baseline_years_status", None)
        record_root = IMAGING_OUTPUT_ROOT / ".clinical_records"
        record_root.mkdir(parents=True, exist_ok=True)
        record_path = record_root / f"{patient_id}.json"
        document = {
            "schema_version": "clinical-intake-record-v1",
            "patient_id": patient_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "clinician_inputs": normalized,
        }
        temporary_path = record_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
        temporary_path.replace(record_path)
        logger.info("clinical_record_saved patient_id=%s path=%s", patient_id, record_path)
        return {"status": "saved", "patient_id": patient_id, "saved_at": document["saved_at"]}

    @app.get("/api/v1/clinical/records/{patient_id}")
    def load_clinical_record(patient_id: str) -> dict[str, Any]:
        if not PATIENT_ID_PATTERN.fullmatch(patient_id):
            raise HTTPException(status_code=422, detail="Invalid patient_id.")
        record_path = IMAGING_OUTPUT_ROOT / ".clinical_records" / f"{patient_id}.json"
        try:
            document = json.loads(record_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="No saved clinical record exists for this patient.") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="Saved clinical record is unreadable.") from exc
        if document.get("patient_id") != patient_id or document.get("schema_version") != "clinical-intake-record-v1":
            raise HTTPException(status_code=500, detail="Saved clinical record failed identity validation.")
        return document

    @app.get("/api/v1/clinical/status")
    def status() -> dict[str, Any]:
        return {
            "status": "research_only",
            **describe_artifacts(current_artifacts()),
            "input_schema_version": "1.0.0",
            "imaging_contract_version": "2.0.0",
        }

    @app.get("/api/v1/analysis/runs/{run_id}/viewer")
    def analysis_viewer(run_id: str) -> dict[str, Any]:
        try:
            return viewer_metadata(run_id, output_root=IMAGING_OUTPUT_ROOT)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Viewer artifacts are unavailable for this run.") from exc

    @app.get("/api/v1/analysis/runs/{run_id}/viewer/slices/{index}")
    def analysis_viewer_slice(
        run_id: str,
        index: int,
        lung_overlay: bool = Query(False),
        heart_overlay: bool = Query(False),
        gradcam_nodule_id: list[str] | None = Query(None, max_length=128),
        gradcam_attribute: str | None = Query(None, max_length=64),
        gradcam_alpha: float = Query(0.55, ge=0.0, le=1.0),
    ) -> Response:
        try:
            image = render_viewer_slice(
                run_id,
                index,
                output_root=IMAGING_OUTPUT_ROOT,
                lung_overlay=lung_overlay,
                heart_overlay=heart_overlay,
                gradcam_nodule_ids=gradcam_nodule_id,
                gradcam_attribute=gradcam_attribute,
                gradcam_alpha=gradcam_alpha,
            )
        except (KeyError, FileNotFoundError, ValueError, IndexError) as exc:
            raise HTTPException(status_code=404, detail="Requested CT slice is unavailable.") from exc
        return Response(content=image, media_type="image/png", headers={"Cache-Control": "private, max-age=60"})

    @app.post("/api/v1/clinical/predict")
    def predict(request: ClinicalRequest) -> dict[str, Any]:
        try:
            return run_inference(
                request.model_dump(),
                artifacts=current_artifacts(),
            )
        except ClinicalInferenceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/analysis/runs", status_code=202)
    def create_analysis(request: AnalysisRequest) -> dict[str, Any]:
        logger.info(
            "analysis_requested patient_id=%s upload_job_id=%s",
            request.patient_id,
            request.upload_job_id,
        )
        try:
            validate_clinician_inputs(request.clinician_inputs)
        except ClinicalValidationError as exc:
            logger.warning("analysis_rejected patient_id=%s reason=%s", request.patient_id, exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            return start_analysis(
                project_root=Path(__file__).resolve().parents[2],
                output_root=IMAGING_OUTPUT_ROOT,
                patient_id=request.patient_id,
                upload_job_id=request.upload_job_id,
                clinical_inputs=request.clinician_inputs,
                requested_malignancy_method=request.requested_malignancy_method,
                data_provenance=request.data_provenance,
                baseline_definition=request.baseline_definition,
            )
        except (ValueError, DicomUploadError, ClinicalValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/analysis/runs/{run_id}")
    def analysis_status(run_id: str) -> dict[str, Any]:
        try:
            return get_analysis(run_id, output_root=IMAGING_OUTPUT_ROOT)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Analysis run not found.") from exc

    @app.get("/api/v1/analysis/patients/{patient_id}/studies")
    def patient_studies(patient_id: str) -> dict[str, Any]:
        if not PATIENT_ID_PATTERN.fullmatch(patient_id):
            raise HTTPException(status_code=422, detail="Invalid patient_id.")
        state_root = IMAGING_OUTPUT_ROOT / ".analysis_runs"
        studies = []
        if state_root.is_dir():
            for state_path in sorted(state_root.glob("*.json")):
                try:
                    stored = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise HTTPException(status_code=500, detail=f"Persisted analysis state is unreadable: {state_path.name}.") from exc
                if stored.get("patient_id") == patient_id:
                    run_id = stored.get("run_id")
                    if not isinstance(run_id, str):
                        raise HTTPException(status_code=500, detail="Persisted analysis state has no run identifier.")
                    try:
                        current = get_analysis(run_id, output_root=IMAGING_OUTPUT_ROOT)
                    except (KeyError, RuntimeError) as exc:
                        raise HTTPException(status_code=500, detail="Persisted analysis state could not be loaded.") from exc
                    study = {
                        "run_id": run_id,
                        "status": current.get("status"),
                        "created_at": current.get("created_at"),
                        "completed_at": current.get("completed_at"),
                        "eligible_nodule_count": current.get("study_summary", {}).get("eligible_nodule_count"),
                        "report_available": (IMAGING_OUTPUT_ROOT / patient_id / f"study_result_{run_id}.json").is_file(),
                    }
                    contract_path = IMAGING_OUTPUT_ROOT / patient_id / CONTRACT_FILENAME
                    if contract_path.is_file():
                        try:
                            contract = read_json(contract_path)
                            validate_contract(contract)
                        except (OSError, ValueError, json.JSONDecodeError) as exc:
                            raise HTTPException(status_code=500, detail="Persisted Imaging study contract is invalid.") from exc
                        study["study_id"] = contract["study_id"]
                    studies.append(study)
        return {"patient_id": patient_id, "studies": studies}

    @app.get("/api/v1/analysis/runs/{run_id}/report")
    def analysis_report(run_id: str) -> dict[str, Any]:
        try:
            run = get_analysis(run_id, output_root=IMAGING_OUTPUT_ROOT)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Analysis run not found.") from exc
        report_path = IMAGING_OUTPUT_ROOT / str(run["patient_id"]) / f"study_result_{run_id}.json"
        try:
            document = json.loads(report_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Structured study report is not available for this run.") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="Structured study report is unreadable.") from exc
        if document.get("schema_version") != "lunginsight-study-result-v1" or document.get("run", {}).get("run_id") != run_id:
            raise HTTPException(status_code=500, detail="Structured study report failed identity validation.")
        return document

    @app.get("/api/v1/analysis/runs/{run_id}/report/html", response_class=HTMLResponse)
    def analysis_html_report(run_id: str) -> HTMLResponse:
        try:
            run = get_analysis(run_id, output_root=IMAGING_OUTPUT_ROOT)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Analysis run not found.") from exc
        patient_id = str(run["patient_id"])
        report_root = IMAGING_OUTPUT_ROOT / "reports" / patient_id
        report_path = report_root / "index.html"
        try:
            content = report_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Readable HTML study report is not available.") from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail="HTML study report is unreadable.") from exc
        base_url = f"/api/v1/analysis/runs/{run_id}/report/files/"
        content = content.replace("<meta charset='utf-8'>", f"<meta charset='utf-8'><base href='{base_url}'>", 1)
        content = content.replace('href="../../', 'href="patient/')
        return HTMLResponse(content)

    @app.get("/api/v1/analysis/runs/{run_id}/report/files/{file_path:path}")
    def analysis_report_file(run_id: str, file_path: str) -> FileResponse:
        try:
            run = get_analysis(run_id, output_root=IMAGING_OUTPUT_ROOT)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Analysis run not found.") from exc
        patient_id = str(run["patient_id"])
        pieces = Path(file_path).parts
        if not pieces or any(piece in {"", ".", ".."} for piece in pieces):
            raise HTTPException(status_code=404, detail="Report file not found.")
        if pieces[0] == "patient":
            root = (IMAGING_OUTPUT_ROOT / patient_id).resolve()
            requested = root.joinpath(*pieces[1:]).resolve()
        elif pieces[0] == "assets":
            root = (IMAGING_OUTPUT_ROOT / "reports" / patient_id).resolve()
            requested = root.joinpath(*pieces).resolve()
        elif pieces[0] in {"index.html", "masks.html", "nodules.html", "gradcam.html", "brock.html", "lineage.html"}:
            root = (IMAGING_OUTPUT_ROOT / "reports" / patient_id).resolve()
            requested = root.joinpath(*pieces).resolve()
        else:
            raise HTTPException(status_code=404, detail="Report file not found.")
        try:
            requested.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Report file not found.") from exc
        if not requested.is_file():
            raise HTTPException(status_code=404, detail="Report file not found.")
        return FileResponse(requested)

    @app.get("/api/v1/analysis/runs/{run_id}/artifacts/{artifact_path:path}")
    def analysis_artifact(run_id: str, artifact_path: str) -> FileResponse:
        try:
            run = get_analysis(run_id, output_root=IMAGING_OUTPUT_ROOT)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Analysis run not found.") from exc
        patient_root = (IMAGING_OUTPUT_ROOT / str(run["patient_id"])).resolve()
        requested = (patient_root / artifact_path).resolve()
        try:
            requested.relative_to(patient_root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Artifact not found.") from exc
        contract_path = patient_root / CONTRACT_FILENAME
        if not contract_path.is_file():
            raise HTTPException(status_code=404, detail="Artifact is not referenced by a validated Imaging contract.")
        try:
            contract = read_json(contract_path)
            validate_contract(contract)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="Persisted Imaging contract is invalid.") from exc
        referenced_paths: set[Path] = set()

        def collect_references(value: Any) -> None:
            if isinstance(value, dict):
                for child in value.values():
                    collect_references(child)
            elif isinstance(value, list):
                for child in value:
                    collect_references(child)
            elif isinstance(value, str) and Path(value).suffix.lower() in {
                ".npy", ".npz", ".png", ".jpg", ".jpeg", ".json", ".html", ".nii", ".gz"
            }:
                candidate = Path(value)
                candidate_path = (patient_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
                try:
                    candidate_path.relative_to(patient_root)
                except ValueError:
                    return
                referenced_paths.add(candidate_path)

        collect_references(contract.get("nodules", []))
        collect_references(contract.get("study", {}).get("artifact_refs", {}))
        if requested not in referenced_paths or not requested.is_file():
            raise HTTPException(status_code=404, detail="Artifact is not referenced by this study.")
        return FileResponse(requested)

    @app.get("/api/v1/analysis/runs/{run_id}/study")
    def analysis_study(run_id: str) -> dict[str, Any]:
        try:
            run = get_analysis(run_id, output_root=IMAGING_OUTPUT_ROOT)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Analysis run not found.") from exc
        contract_path = IMAGING_OUTPUT_ROOT / str(run["patient_id"]) / CONTRACT_FILENAME
        if not contract_path.is_file():
            return {
                "run_id": run_id,
                "contract_status": "not_available",
                "processing_status": run.get("status"),
                "study": run.get("study_summary"),
                "artifact_refs": run.get("artifact_refs", {}),
            }
        try:
            contract = read_json(contract_path)
            validate_contract(contract)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail="Persisted Imaging study contract is invalid.") from exc
        return {"run_id": run_id, "contract_status": "validated", "study": contract}

    @app.get("/api/v1/analysis/runs/{run_id}/nodules/{nodule_id}")
    def analysis_nodule(run_id: str, nodule_id: str) -> dict[str, Any]:
        try:
            run = get_analysis(run_id, output_root=IMAGING_OUTPUT_ROOT)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Analysis run not found.") from exc
        contract_path = IMAGING_OUTPUT_ROOT / str(run["patient_id"]) / CONTRACT_FILENAME
        if not contract_path.is_file():
            raise HTTPException(status_code=409, detail="Validated nodule records are not yet available.")
        try:
            contract = read_json(contract_path)
            validate_contract(contract)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail="Persisted Imaging nodule contract is invalid.") from exc
        nodule = next(
            (row for row in contract["nodules"] if row["stable_nodule_id"] == nodule_id),
            None,
        )
        if nodule is None:
            raise HTTPException(status_code=404, detail="Nodule not found in this study.")
        return {"run_id": run_id, "study_id": contract["study_id"], "nodule": nodule}

    @app.post("/api/v1/analysis/runs/{run_id}/nodules/{nodule_id}/brock")
    def analysis_manual_brock(run_id: str, nodule_id: str, request: ManualBrockRequest) -> dict[str, Any]:
        try:
            return compute_manual_brock(run_id, nodule_id, request.values, output_root=IMAGING_OUTPUT_ROOT)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Analysis run or nodule not found.") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except BrockConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/analysis/runs/{run_id}/resume", status_code=202)
    def resume_analysis_run(run_id: str) -> dict[str, Any]:
        try:
            return resume_analysis(
                run_id,
                project_root=Path(__file__).resolve().parents[2],
                output_root=IMAGING_OUTPUT_ROOT,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Analysis run not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/imaging/dicom/upload", status_code=202)
    async def upload_dicom(
        patient_id: str = Form(..., min_length=1, max_length=128),
        files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        logger.info("dicom_upload_received patient_id=%s file_count=%d", patient_id, len(files))
        try:
            result = await process_dicom_upload(
                files,
                patient_id=patient_id,
                output_root=IMAGING_OUTPUT_ROOT,
            )
        except DicomUploadError as exc:
            logger.warning("dicom_upload_rejected patient_id=%s reason=%s", patient_id, exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OSError as exc:
            logger.error("dicom_upload_storage_failed patient_id=%s error=%s", patient_id, exc)
            raise HTTPException(status_code=500, detail="The CT upload could not be stored safely.") from exc
        logger.info("dicom_upload_accepted patient_id=%s job_id=%s", patient_id, result.get("job_id"))
        return result

    @app.get("/api/v1/imaging/dicom/jobs/{job_id}")
    def imaging_dicom_job_status(job_id: str) -> dict[str, Any]:
        try:
            return get_dicom_job(job_id, output_root=IMAGING_OUTPUT_ROOT)
        except DicomUploadError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="clinical-frontend")
    return app


app = create_app(deployed=True)
