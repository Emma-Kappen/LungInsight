# LungInsight — prototype scaffold

**Read this before running anything.** This is an engineering prototype, not a validated diagnostic tool.
The imaging model in `backend/model.py` has an ImageNet-pretrained backbone but an **untrained** classification head —
its outputs are for exercising the pipeline (upload → preprocess → infer → Grad-CAM → report), not for interpreting
real images. Do not use this with real patient data or in any real clinical workflow. See `docs/ARCHITECTURE.md` §7
and `docs/PRD.md` §7 for the compliance boundary and what would need to change before that's even worth considering.

## What's here

```
docs/
  ARCHITECTURE.md   — system architecture & end-to-end data flow
  PRD.md            — product requirements document
frontend/
  index.html        — intake form, results panel, printable report (single file, vanilla JS)
backend/
  main.py           — FastAPI app: /api/v1/analyze, /api/v1/report/{case_id}, /healthz
  model.py          — CNN + Grad-CAM (placeholder head — see warning above)
  preprocessing.py  — DICOM/PNG/JPEG decoding, magic-byte validation, PHI-tag stripping
  schemas.py        — Pydantic request/response validation
  report.py         — Jinja2 + WeasyPrint PDF report renderer
  templates/report.html — print-ready report layout
  requirements.txt
```

## Running it locally

```bash
cd backend
pip install -r requirements.txt   # WeasyPrint needs system libs: see weasyprint.org/docs/install
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in a browser (or serve it statically). It talks to
`http://localhost:8000` by default — override with `window.LUNGINSIGHT_API_BASE` if needed.

The `/api/v1/analyze` and `/api/v1/report/{id}` routes require a bearer token
(`Authorization: Bearer <token>`). The scaffold only checks that *a* token is present —
wire `require_auth()` in `main.py` to your real SSO/session validation before this is
used for anything beyond local testing.

## How this connects to your notebook

`train_clinical_3_.ipynb` trains a tabular Random Forest (+ Cox survival model) on structured
clinical features for stage/histology prediction — it doesn't touch images. `model.py`'s
`fuse_with_tabular_risk()` is where that model would plug in as a structured-risk adjunct
to the imaging output (load it with `joblib.load(...)` and feed it the same feature
columns it was trained on); right now that function uses a simple heuristic stand-in
so the fusion step in the architecture has something to build against.

## What was verified

The imaging pipeline (`model.py`), preprocessing/validation (`preprocessing.py`, including a
synthetic DICOM with PHI tags), and the FastAPI routes (`main.py`, including auth rejection
and validation-error paths) were smoke-tested end-to-end. `report.py`/WeasyPrint could not be
executed in this environment (no system Pango/Cairo libraries available here) — verify PDF
rendering in your own environment after `pip install -r requirements.txt`.

## Before this touches anything real

1. Replace the placeholder model with one trained and clinically validated for the specific
   findings you intend to report.
2. Determine whether the intended use triggers Software-as-a-Medical-Device regulatory review.
3. Put real auth, encryption-at-rest, and audit-log persistence behind the placeholders marked
   in `main.py`.
4. Get a BAA in place with any infrastructure vendor before real PHI reaches the system.
5. Have compliance/legal review the disclaimer language in `frontend/index.html` and
   `backend/templates/report.html` — it's a reasonable starting point, not a substitute for
   their sign-off.
