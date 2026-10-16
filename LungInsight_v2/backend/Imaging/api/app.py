from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, ConfigDict, Field

IMAGING_DIR = Path(__file__).resolve().parents[1]
if str(IMAGING_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGING_DIR))

from .contract import (
    CONTRACT_FILENAME,
    SCHEMA_PATH,
    create_contract_file,
    read_json,
    validate_contract,
)
from pipeline_schema import get_required, load_schema
from .dicom_upload import DicomUploadError, get_dicom_job, process_dicom_upload

V2_ROOT = IMAGING_DIR.parent.parent
PIPELINE_SCHEMA = load_schema(IMAGING_DIR / "stages" / "01_hu_and_crop.py")
DEFAULT_OUTPUT_ROOT = V2_ROOT / str(get_required(PIPELINE_SCHEMA, "paths.output_root"))
OUTPUT_ROOT = Path(os.environ.get("LUNGINSIGHT_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))).expanduser().resolve()
STAGE06 = get_required(PIPELINE_SCHEMA, "stage_06_full_ct_gradcam", expected_type=dict)
STAGE06_PATHS = get_required(PIPELINE_SCHEMA, "paths.stage_06", expected_type=dict)
STAGE06_RECORDS = str(STAGE06["record_filename"])
PATIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

logger = logging.getLogger("lunginsight.imaging_api.audit")
app = FastAPI(
    title="LungInsight Imaging API",
    version="1.0.0",
    description=(
        "Imaging-only, schema-validated endpoints. Clinical history and clinical inference "
        "are intentionally outside this service."
    ),
)


def _openapi_schema_ref(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _openapi_schema_ref(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_openapi_schema_ref(child) for child in value]
    if isinstance(value, str) and value.startswith("#/$defs/"):
        return value.replace("#/$defs/", "#/components/schemas/ImagingClinicalContractV1_", 1)
    return value


def _custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    contract_schema = read_json(SCHEMA_PATH)
    definitions = contract_schema.pop("$defs", {})
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    components["ImagingClinicalContractV1"] = _openapi_schema_ref(contract_schema)
    for name, definition in definitions.items():
        components[f"ImagingClinicalContractV1_{name}"] = _openapi_schema_ref(definition)
    paths = schema["paths"]
    study_ref = {"$ref": "#/components/schemas/ImagingClinicalContractV1"}
    nodule_ref = {"$ref": "#/components/schemas/ImagingClinicalContractV1_nodule"}
    paths["/api/v1/imaging/study"]["post"]["responses"]["200"]["content"]["application/json"]["schema"] = study_ref
    paths["/api/v1/imaging/study/{study_id}"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] = study_ref
    paths["/api/v1/imaging/nodule/{nodule_id}"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] = {
        "type": "object",
        "required": ["nodule"],
        "properties": {"nodule": nodule_ref},
        "additionalProperties": False,
    }
    paths["/api/v1/imaging/artifact/{artifact_id}"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] = {
        "type": "object",
        "required": ["artifact"],
        "properties": {"artifact": {"type": "object"}},
        "additionalProperties": False,
    }
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi


class StudyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(min_length=1, max_length=128)


@app.post("/api/v1/imaging/dicom/upload", status_code=202)
async def upload_dicom(
    patient_id: str = Form(..., min_length=1, max_length=128),
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """Validate one uploaded CT series and persist a HU volume for pipeline execution."""
    try:
        return await process_dicom_upload(files, patient_id=patient_id, output_root=OUTPUT_ROOT)
    except DicomUploadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="The CT upload could not be stored safely.") from exc


@app.get("/api/v1/imaging/dicom/jobs/{job_id}")
def dicom_job_status(job_id: str) -> dict[str, Any]:
    try:
        return get_dicom_job(job_id)
    except DicomUploadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _patient_root(patient_id: str) -> Path:
    if not PATIENT_ID_PATTERN.fullmatch(patient_id) or patient_id in {".", ".."}:
        raise HTTPException(status_code=422, detail="Invalid patient_id.")
    root = (OUTPUT_ROOT / patient_id).resolve()
    if root.parent != OUTPUT_ROOT:
        raise HTTPException(status_code=422, detail="Invalid patient_id.")
    return root


def _read_validated_contract(path: Path) -> dict[str, Any]:
    try:
        contract = read_json(path)
        validate_contract(contract)
        return contract
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("contract_validation_failed")
        raise HTTPException(status_code=500, detail="Stored imaging contract is invalid.") from exc


def _contracts() -> list[tuple[Path, dict[str, Any]]]:
    found = []
    if not OUTPUT_ROOT.exists():
        return found
    for path in OUTPUT_ROOT.glob(f"*/{CONTRACT_FILENAME}"):
        found.append((path.parent, _read_validated_contract(path)))
    return found


def _one_match(matches: list[dict[str, Any]], description: str) -> dict[str, Any]:
    if not matches:
        raise HTTPException(status_code=404, detail=f"{description} not found.")
    if len(matches) > 1:
        raise HTTPException(status_code=409, detail=f"{description} is ambiguous across studies.")
    return matches[0]


def _flatten_artifacts(
    contract: dict[str, Any],
    patient_root: Path,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    study_id = contract["study_id"]
    output_root = patient_root.parent.resolve()

    def visit(value: Any, *, stable_id: str | None, key: str) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, stable_id=stable_id, key=f"{key}.{child_key}" if key else child_key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, stable_id=stable_id, key=f"{key}[{index}]")
        elif isinstance(value, str) and value.strip():
            candidate = Path(value)
            suffix = candidate.suffix.lower()
            if suffix not in {".npy", ".npz", ".png", ".jpg", ".jpeg", ".json", ".html", ".nii", ".dcm", ".gz"}:
                return
            artifact_id = hashlib.sha256(
                f"{study_id}\0{stable_id or ''}\0{key}\0{value}".encode("utf-8")
            ).hexdigest()[:24]
            resolved = (patient_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
            try:
                relative_path = resolved.relative_to(output_root)
            except ValueError:
                return
            refs.append({
                "artifact_id": artifact_id,
                "study_id": study_id,
                "stable_nodule_id": stable_id,
                "reference_type": key,
                "path": relative_path.as_posix(),
                "available": resolved.is_file(),
            })

    for nodule in contract.get("nodules", []):
        stable_id = nodule["stable_nodule_id"]
        visit(nodule.get("gradcam_refs", {}), stable_id=stable_id, key="gradcam")
        visit(nodule.get("shap_refs", {}), stable_id=stable_id, key="shap")
        visit(nodule.get("artifact_refs", {}), stable_id=stable_id, key="artifact")
    return refs


@app.post("/api/v1/imaging/study")
def create_study(request: StudyRequest) -> dict[str, Any]:
    patient_root = _patient_root(request.patient_id)
    records_path = patient_root / str(STAGE06_PATHS["directory"]) / STAGE06_RECORDS
    if not records_path.is_file():
        raise HTTPException(status_code=404, detail="Validated Stage 06 records are unavailable.")
    try:
        contract_path = create_contract_file(patient_root, request.patient_id, records_path)
        result = _read_validated_contract(contract_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("study_contract_generation_failed")
        raise HTTPException(status_code=422, detail=f"Could not create validated imaging contract: {exc}") from exc
    logger.info("study_contract_created")
    return result


@app.get("/api/v1/imaging/study/{study_id}")
def get_study(study_id: str) -> dict[str, Any]:
    matches = [contract for _, contract in _contracts() if contract["study_id"] == study_id]
    logger.info("study_read")
    return _one_match(matches, "Study")


@app.get("/api/v1/imaging/nodule/{nodule_id}")
def get_nodule(nodule_id: str) -> dict[str, Any]:
    matches = [
        {"nodule": nodule}
        for _, contract in _contracts()
        for nodule in contract["nodules"]
        if nodule["stable_nodule_id"] == nodule_id
    ]
    logger.info("nodule_read")
    return _one_match(matches, "Nodule")


@app.get("/api/v1/imaging/artifact/{artifact_id}")
def get_artifact(artifact_id: str) -> dict[str, Any]:
    for patient_root, contract in _contracts():
        for reference in _flatten_artifacts(contract, patient_root):
            if reference["artifact_id"] == artifact_id:
                logger.info("artifact_read")
                return {"artifact": reference}
    logger.info("artifact_not_found")
    raise HTTPException(status_code=404, detail="Artifact not found.")
