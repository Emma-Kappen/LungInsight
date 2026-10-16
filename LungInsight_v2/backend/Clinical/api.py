from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from fastapi.staticfiles import StaticFiles

from .inference import ClinicalInferenceError, load_artifacts, run_inference


class ClinicalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(pattern=r"^1\.0\.0$")
    patient_id: str = Field(min_length=1, max_length=128)
    baseline_at: str
    baseline_definition: Literal["initial_pathologic_diagnosis", "other_or_unverified"]
    clinician_inputs: dict[str, Any]
    imaging_evidence: dict[str, Any]


def create_app(artifact_path: str | None = None) -> FastAPI:
    resolved_artifact_path = artifact_path or os.environ.get("LUNGINSIGHT_CLINICAL_ARTIFACT")
    artifacts = load_artifacts(resolved_artifact_path) if resolved_artifact_path else None
    app = FastAPI(
        title="LungInsight Clinical Research API",
        version="0.1.0",
        description=(
            "Research-only clinical inference. Requires validated Imaging evidence and does not "
            "replace pathology, established instruments, or clinician judgment."
        ),
    )

    @app.get("/api/v1/clinical/status")
    def status() -> dict[str, Any]:
        available = artifacts or {}
        return {
            "status": "research_only",
            "cancer_type_model": "loaded" if available.get("cancer_type") else "not_trained",
            "survival_model": "loaded" if available.get("survival") else "not_trained",
            "input_schema_version": "1.0.0",
            "imaging_contract_version": "1.0.0",
        }

    @app.post("/api/v1/clinical/predict")
    def predict(request: ClinicalRequest) -> dict[str, Any]:
        try:
            return run_inference(
                request.model_dump(),
                artifacts=artifacts,
            )
        except ClinicalInferenceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="clinical-frontend")
    return app


app = create_app()
