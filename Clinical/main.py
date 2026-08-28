~"""LungInsight backend — FastAPI service.

Run with:  uvicorn main:app --reload --port 8000

See docs/ARCHITECTURE.md and docs/PRD.md for the full design and the
compliance boundary this scaffold intentionally stops at (§7 / §7 resp.).
"""
from __future__ import annotations

import base64
import logging
import time
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from model import MODEL_VERSION, cam_to_overlay_png_bytes, fuse_with_tabular_risk, run_inference
from preprocessing import validate_and_decode
from report import render_report_pdf
from schemas import AnalyzeResponse, FindingOut, PatientMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lunginsight")

app = FastAPI(title="LungInsight API", version="0.1.0")

# Tighten allow_origins to your actual frontend origin(s) in production —
# "*" is left here only for local scaffold testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)

# In-memory case store for the prototype only. Replace with an encrypted,
# access-controlled datastore before any real PHI touches this service —
# see docs/ARCHITECTURE.md §7 and docs/PRD.md §7.
_CASE_STORE: dict[str, dict] = {}

# Append-only audit log placeholder (prototype: in-memory list).
_AUDIT_LOG: list[dict] = []


def require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> str:
    """Placeholder bearer-token auth. Wire this to real SSO/session
    validation before deployment — this only checks a token is present."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    return credentials.credentials


def _audit(event: str, case_id: str, actor_token: str) -> None:
    _AUDIT_LOG.append({
        "event": event,
        "case_id": case_id,
        "actor": actor_token[:8] + "…",  # never log full tokens
        "ts": time.time(),
    })


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    age: int = Form(...),
    sex: Literal["F", "M", "O"] = Form(...),
    smoking_status: Literal["never", "former", "current"] = Form(...),
    pack_years: float = Form(0),
    clinical_notes: str = Form(...),
    image: UploadFile = File(...),
    token: str = Depends(require_auth),
):
    # Server-side validation is authoritative — the client-side checks in
    # frontend/index.html are UX only and are never trusted here.
    try:
        metadata = PatientMetadata(
            age=age, sex=sex, smoking_status=smoking_status,
            pack_years=pack_years, clinical_notes=clinical_notes,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    raw = await image.read()
    pil_image, safe_meta = validate_and_decode(raw, image.filename or "")

    findings, cam, uncertainty = run_inference(pil_image)
    heatmap_png = cam_to_overlay_png_bytes(pil_image, cam)
    heatmap_b64 = base64.b64encode(heatmap_png).decode("ascii")

    composite = fuse_with_tabular_risk(findings, age=metadata.age, pack_years=metadata.pack_years)

    case_id = f"case_{uuid.uuid4().hex[:12]}"
    response = AnalyzeResponse(
        case_id=case_id,
        model_version=MODEL_VERSION,
        findings=[FindingOut(label=f.label, probability=f.probability, flag=f.flag) for f in findings],
        heatmap_png_b64=heatmap_b64,
        composite_risk_score=composite,
        uncertainty=uncertainty,
    )

    # Store only what's needed to regenerate a report; de-identified image
    # metadata only (safe_meta), never the PHI tags stripped in preprocessing.
    _CASE_STORE[case_id] = {
        "metadata": metadata.model_dump(),
        "response": response.model_dump(),
        "image_meta": safe_meta,
        "created_at": time.time(),
    }
    _audit("analyze", case_id, token)
    logger.info("analyze completed case_id=%s uncertainty=%s", case_id, uncertainty)
    return response


@app.get("/api/v1/report/{case_id}")
def get_report(case_id: str, token: str = Depends(require_auth)):
    case = _CASE_STORE.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found or expired.")

    pdf_bytes = render_report_pdf(case)
    _audit("report_generated", case_id, token)

    from fastapi.responses import Response
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="lunginsight_report_{case_id}.pdf"'},
    )
