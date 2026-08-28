"""FastAPI web application for the LungInsight multimodal dashboard."""
from __future__ import annotations
import json, logging, shutil, tempfile, uuid
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from .config import UPLOAD_DIR, DEFAULTS
from .pipeline import ClinicalPipeline
from .pdf_generator import generate_pdf_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)
app = FastAPI(title="LungInsight Multimodal Clinical AI")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
pipeline = ClinicalPipeline()

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "defaults": DEFAULTS})

@app.post("/api/predict")
async def predict(request: Request, ct: UploadFile | None = File(default=None)):
    form = await request.form()
    raw = {k: form.get(k) for k in form.keys() if k != "ct"}
    try:
        raw["patient_id"] = raw.get("patient_id") or "Web Patient"
        ct_path = None
        if ct and ct.filename:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            suffix = Path(ct.filename).suffix
            ct_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
            with ct_path.open("wb") as f: shutil.copyfileobj(ct.file, f)
        result = pipeline.predict(raw, str(ct_path) if ct_path else None)
        result["patient_id"] = raw["patient_id"]
        return JSONResponse(result)
    except Exception as e:
        log.exception("Inference failure")
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/api/pdf")
async def pdf(payload: dict):
    try:
        data = payload.get("patient_data", {})
        results = payload.get("prediction_results", {})
        pdf_bytes = generate_pdf_report(data, results)
        return Response(pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="LungInsight_Clinical_Summary.pdf"'})
    except Exception as e:
        log.exception("PDF generation failure")
        raise HTTPException(status_code=500, detail=str(e))
