# LungInsight — System Architecture & Data Flow

> **Status label:** This document describes an engineering scaffold for a clinical decision-*support* prototype. It is not a description of a validated or FDA-cleared medical device. See §7 for the compliance boundary.

## 1. High-level component map

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              CLIENT (browser)                             │
│  ┌───────────────┐   ┌────────────────────┐   ┌───────────────────────┐   │
│  │ Intake Form   │──▶│ Results / Heatmap  │──▶│ Printable Report View │   │
│  │ (HTML/CSS/JS) │   │ Panel (JS render)  │   │ (@media print / PDF)  │   │
│  └───────────────┘   └────────────────────┘   └───────────────────────┘   │
└───────────────────────────────┬───────────────────────────────────────────┘
                                 │ HTTPS (TLS 1.2+), multipart/form-data
                                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY / REVERSE PROXY                        │
│   - TLS termination         - Rate limiting        - Auth token check     │
└───────────────────────────────┬───────────────────────────────────────────┘
                                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         BACKEND — FastAPI service                         │
│                                                                            │
│  [1] Request validation (Pydantic)                                        │
│        - metadata schema (age, sex, smoking Hx, notes)                    │
│        - file type / size / magic-byte check (DICOM, PNG, JPEG)           │
│                                                                            │
│  [2] Preprocessing pipeline                                                │
│        - DICOM: pydicom → pixel_array → VOI LUT / windowing → 8-bit       │
│        - PNG/JPEG: PIL decode → color-space normalize                     │
│        - Resize/pad to model input size, z-score normalize                │
│        - Strip/quarantine DICOM tags containing PHI before any            │
│          logging or caching (PatientName, PatientID, etc.)                │
│                                                                            │
│  [3] Inference engine                                                     │
│        - CNN feature extractor + classification head                     │
│        - Produces per-class probabilities (multi-label pathology flags)  │
│        - Grad-CAM hook on final conv block → heatmap tensor               │
│        - Tabular clinical-risk adjunct (RF model from training notebook) │
│          combined via late fusion for a composite risk score             │
│                                                                            │
│  [4] Post-processing                                                      │
│        - Heatmap → colormap → alpha-blend over source image → PNG        │
│        - Threshold calibration → discrete flags ("Suspicious nodule")    │
│        - Structured JSON result assembly                                  │
│                                                                            │
│  [5] Persistence (optional, encrypted at rest)                            │
│        - Case record: inputs (de-identified), outputs, model version,    │
│          timestamp, clinician ID                                         │
│                                                                            │
│  [6] Report generator                                                     │
│        - Jinja2 HTML template + case data → WeasyPrint → PDF             │
└───────────────────────────────┬───────────────────────────────────────────┘
                                 ▼
                     ┌───────────────────────┐
                     │ Encrypted object/DB    │
                     │ store (PHI boundary)   │
                     └───────────────────────┘
```

## 2. End-to-end request lifecycle

1. **Form submission.** Clinician fills the intake form (age, sex, smoking history, free-text notes) and attaches an image (`.dcm`, `.png`, `.jpg`). Client-side JS does soft validation (required fields, file-type allow-list, max size) purely for UX — it is never trusted as the security boundary.
2. **Transport.** The browser sends a single `multipart/form-data` POST to `/api/v1/analyze` over TLS, with an auth bearer token (session or SSO-issued) in the header.
3. **Server-side validation.** FastAPI + Pydantic re-validates every field server-side: type, range (e.g., age 0–120), enum membership (sex, smoking status), file magic bytes (not just extension), and file size ceiling.
4. **De-identification gate.** Any DICOM metadata tags are parsed, PHI-bearing tags are stripped into a separate access-controlled field, and only the pixel data plus a generated case ID proceed to the inference path and to logs/metrics.
5. **Preprocessing.** Pixel data is normalized to the model's expected input (windowing for DICOM, resizing, normalization).
6. **Inference.** The imaging model returns class probabilities; Grad-CAM activation is computed in the same forward/backward pass. The tabular risk model (from the training notebook) can optionally combine with structured fields (age, smoking pack-years, etc.) for a fused risk estimate.
7. **Response assembly.** The API returns structured JSON: probabilities per finding, a base64 heatmap-overlay PNG, discrete flags, model version/timestamp, and a confidence/uncertainty indicator.
8. **Client rendering.** The results page renders confidence bars, the heatmap overlay, and flags dynamically from that JSON — no page reload.
9. **Report generation.** On "Generate report," the client either (a) requests a server-rendered PDF (`/api/v1/report/{case_id}`, Jinja2 → WeasyPrint) or (b) triggers `window.print()` against a print-styled version of the same DOM. Either path renders patient details, findings, the heatmap, model version/disclaimer text, and blank physician sign-off lines.
10. **Audit log.** Every analyze/report request is written to an append-only audit log (who, when, case ID, model version) — separate from the PHI store — to support HIPAA accounting-of-disclosures requirements.

## 3. Data contracts

**Request → `/api/v1/analyze`**
```json
{
  "patient": {"age": 64, "sex": "F", "smoking_status": "former", "pack_years": 32},
  "clinical_notes": "Persistent cough x6 weeks, no hemoptysis.",
  "image": "<binary, multipart>"
}
```

**Response**
```json
{
  "case_id": "case_8f2a...",
  "model_version": "lunginsight-imaging-v0.1-PLACEHOLDER",
  "findings": [
    {"label": "Pulmonary nodule", "probability": 0.71, "flag": "suspicious"},
    {"label": "Pleural effusion", "probability": 0.08, "flag": "unlikely"}
  ],
  "heatmap_png_b64": "iVBORw0KGgo...",
  "composite_risk_score": 0.64,
  "uncertainty": "moderate",
  "disclaimer": "Investigational use only. Not a substitute for clinical judgment."
}
```

## 4. Deployment topology (suggested)

- Static frontend served from a CDN or the FastAPI app itself behind the same origin (avoids CORS/PHI-in-transit complications).
- Backend as a containerized FastAPI service (Uvicorn/Gunicorn workers) behind an API gateway that terminates TLS and enforces auth.
- Model weights loaded once at process start; GPU optional (inference works on CPU for a single ResNet-class model, sub-second per image).
- PHI-bearing data (if persisted at all) lives in an encrypted store separate from application logs and metrics, with its own access control list.

## 5. Failure modes to design for

- Corrupt/unsupported DICOM transfer syntax → reject with a clear 422, never silently degrade to a blank image.
- Low-quality/low-resolution image → surface an explicit "image quality insufficient for analysis" flag rather than a falsely confident score.
- Model service timeout → return a typed error, not a partial/garbled JSON.

## 6. Data flow summary (one line)

`Form → validated multipart POST → PHI-strip → preprocess → CNN + Grad-CAM (+ optional tabular fusion) → structured JSON → dynamic UI render → Jinja2/WeasyPrint or @media print → signed physician report`

## 7. Compliance boundary (read before deploying)

This architecture is built so that a real HIPAA-compliant deployment is *possible* on top of it (encryption in transit/at rest, audit logging, de-identification gate, access control hooks), but none of that is turned on by default in the scaffold, and the imaging model shipped with it is an untrained placeholder. Do not point this at real patient data or real clinical workflows until: (a) a validated, clinically evaluated model replaces the placeholder, (b) a BAA-covered infrastructure provider is in place, (c) the security controls in §8 of the PRD are implemented and reviewed, and (d) appropriate regulatory review (e.g., whether the tool constitutes Software as a Medical Device) has been completed for your jurisdiction.
