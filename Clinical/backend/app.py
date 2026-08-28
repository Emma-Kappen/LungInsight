from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile, uuid, tarfile, zipfile, base64, io
from pathlib import Path
from typing import Any
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import numpy as np
import pandas as pd
from PIL import Image

from .pipeline import ClinicalPipeline, DEFAULTS
from .pdf_generator import generate_pdf_report

BASE=Path(__file__).resolve().parents[2]
CLINICAL=BASE/'Clinical'; IMAGING=BASE/'Imaging'; MODELS=CLINICAL/'models'; OUTPUT=BASE/'output'; OUTPUT.mkdir(exist_ok=True)
app=FastAPI(title='LungInsight', version='1.0.0')
app.mount('/static',StaticFiles(directory=str(CLINICAL/'static')),name='static')
templates=Jinja2Templates(directory=str(CLINICAL/'templates'))
try: pipeline=ClinicalPipeline(MODELS,IMAGING)
except Exception as e: pipeline=None; MODEL_LOAD_ERROR=str(e)


def _safe_name(s): return ''.join(c if c.isalnum() or c in '._-' else '_' for c in s)

# ---------------------------------------------------------------------
# EXISTING IMAGING PIPELINE BRIDGE
# ---------------------------------------------------------------------
#
# IMPORTANT:
# No CT processing is implemented here.
# The existing Imaging pipeline remains authoritative.
#
# Flow:
#   upload
#     -> staged DICOM directory
#     -> run_pipeline.py
#     -> Imaging/01 ... 08
#     -> Imaging/09_final_presentation.py
#     -> output/<imaging_patient_id>/09_presentation/viewer.html
#
# Stage 09 is the final presentation authority.
# ---------------------------------------------------------------------

def _make_imaging_patient_id(run_id: str) -> str:
    """
    Imaging/run_pipeline.py expects a numeric LIDC-style patient ID.

    The clinical Patient ID is NOT changed. This ID is only the internal
    Imaging pipeline identifier.
    """
    import hashlib

    n = int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16)
    n = 1000 + (n % 8999)
    return f"LIDC-IDRI-{n:04d}"


def _stage_input_dicom(
    source: Path,
    imaging_patient_id: str,
    run_id: str,
) -> Path:
    """
    Convert an uploaded DICOM/archive/folder into:

        <temporary_root>/<imaging_patient_id>/*.dcm

    which is exactly the structure expected by run_pipeline.py.
    """
    input_root = OUTPUT / "_imaging_inputs" / run_id
    patient_dir = input_root / imaging_patient_id

    if patient_dir.exists():
        shutil.rmtree(patient_dir)

    patient_dir.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        for src in source.rglob("*"):
            if not src.is_file():
                continue

            rel = src.relative_to(source)

            # Preserve the folder structure because some DICOM series
            # contain useful hierarchy/metadata.
            dst = patient_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    elif source.is_file():
        name = source.name.lower()

        if name.endswith(".zip"):
            with zipfile.ZipFile(source) as z:
                root = patient_dir.resolve()
                for member in z.infolist():
                    target = (patient_dir / member.filename).resolve()
                    if not str(target).startswith(str(root)):
                        raise HTTPException(
                            400,
                            "Unsafe path detected in ZIP archive."
                        )
                z.extractall(patient_dir)

        elif name.endswith(".tar.gz"):
            # Prevent path traversal.
            with tarfile.open(source, "r:gz") as tar:
                for member in tar.getmembers():
                    target = (patient_dir / member.name).resolve()
                    if not str(target).startswith(str(patient_dir.resolve())):
                        raise HTTPException(
                            400,
                            "Unsafe path detected in TAR.GZ archive."
                        )
                tar.extractall(patient_dir)

        elif name.endswith(".dcm"):
            shutil.copy2(source, patient_dir / source.name)

        else:
            raise HTTPException(
                400,
                f"Cannot convert {source.name} to DICOM input."
            )

    else:
        raise HTTPException(400, "Invalid CT input.")

    if not any(path.is_file() for path in patient_dir.rglob("*")):
        raise HTTPException(
            400,
            "The CT upload contains no readable files."
        )

    return patient_dir


def _run_existing_imaging(
    input_path: Path,
    run_id: str,
) -> dict:
    """
    Execute the EXISTING Imaging pipeline.

    Returns paths/metadata needed by the dashboard.
    """

    imaging_patient_id = _make_imaging_patient_id(run_id)

    dicom_patient_dir = _stage_input_dicom(
        input_path,
        imaging_patient_id,
        run_id,
    )

    imaging_runner = IMAGING / "run_pipeline.py"

    if not imaging_runner.exists():
        raise RuntimeError(
            f"Existing imaging runner not found: {imaging_runner}"
        )

    # run_pipeline.py uses the current Python interpreter, so all Imaging
    # dependencies resolve from the same .venv as FastAPI.
    cmd = [
        sys.executable,
        str(imaging_runner),
        imaging_patient_id,
        "--dicom-root",
        str(dicom_patient_dir.parent),
        "--output-root",
        str(OUTPUT),
    ]

    print("\n" + "=" * 80)
    print("LUNGINSIGHT — RUNNING IMAGING STAGES 01 → 08")
    print("=" * 80)
    print(" ".join(map(str, cmd)))

    proc = subprocess.run(
        cmd,
        cwd=str(BASE),
        capture_output=False,
        text=True,
        timeout=60 * 60 * 4,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Existing Imaging pipeline failed with exit code "
            f"{proc.returncode}."
        )

    patient_output = OUTPUT / imaging_patient_id

    # ---------------------------------------------------------------
    # Stage 09 is NOT included by the current run_pipeline.py.
    # Therefore invoke it explicitly after Stage 08.
    # ---------------------------------------------------------------

    stage09 = IMAGING / "09_final_presentation.py"

    if not stage09.exists():
        raise RuntimeError(
            f"Stage 09 was not found: {stage09}"
        )

    cmd09 = [
        sys.executable,
        str(stage09),
        imaging_patient_id,
        "--output-root",
        str(OUTPUT),
    ]

    print("\n" + "=" * 80)
    print("LUNGINSIGHT — RUNNING STAGE 09 FINAL PRESENTATION")
    print("=" * 80)
    print(" ".join(map(str, cmd09)))

    proc09 = subprocess.run(
        cmd09,
        cwd=str(BASE),
        capture_output=False,
        text=True,
        timeout=60 * 30,
    )

    if proc09.returncode != 0:
        raise RuntimeError(
            f"Stage 09 failed with exit code {proc09.returncode}."
        )

    stage09_dir = patient_output / "09_presentation"
    viewer = stage09_dir / "viewer.html"
    manifest = stage09_dir / "manifest.json"

    if not viewer.exists():
        raise RuntimeError(
            f"Stage 09 completed but viewer.html was not created: {viewer}"
        )

    if not manifest.exists():
        raise RuntimeError(
            f"Stage 09 completed but manifest.json was not created: {manifest}"
        )

    return {
        "imaging_patient_id": imaging_patient_id,
        "output_dir": str(patient_output),
        "stage01_dir": str(patient_output / "01"),
        "stage02_dir": str(patient_output / "02"),
        "stage04_dir": str(patient_output / "04_candidates"),
        "stage05_dir": str(patient_output / "05_classifier_patches"),
        "stage06_dir": str(patient_output / "06_classification"),
        "stage07_dir": str(patient_output / "07_gradcam"),
        "stage08_dir": str(patient_output / "08_visualization"),
        "stage09_dir": str(stage09_dir),
        "stage09_viewer": str(viewer),
        "stage09_manifest": str(manifest),
    }


# ---------------------------------------------------------------------
# STAGE 04 — NODULE COUNT
# ---------------------------------------------------------------------

def _get_stage04_nodule_count(stage04_dir: Path) -> int:
    """
    Read the number of nodules/candidates identified by Imaging Stage 04.

    Stage 04 is authoritative.

    This deliberately checks common Stage 04 manifest/report formats so
    the dashboard is coupled to the existing pipeline output rather than
    implementing another detector.
    """

    if not stage04_dir.exists():
        raise RuntimeError(
            f"Stage 04 output does not exist: {stage04_dir}"
        )

    # ---------------------------------------------------------------
    # JSON reports/manifests
    # ---------------------------------------------------------------

    json_candidates = [
        stage04_dir / "candidates.json",
        stage04_dir / "report.json",
        stage04_dir / "manifest.json",
        stage04_dir / "detections.json",
    ]

    count_keys = (
        "num_nodules",
        "nodule_count",
        "num_candidates",
        "candidate_count",
        "number_of_nodules",
        "number_nodules",
        "n_candidates",
    )

    for p in json_candidates:
        if not p.exists():
            continue

        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Direct count.
        if isinstance(obj, dict):
            for key in count_keys:
                value = obj.get(key)

                if isinstance(value, (int, float)):
                    return max(0, int(value))

        # Candidate list.
        if isinstance(obj, list):
            return len(obj)

        if isinstance(obj, dict):
            for key in (
                "candidates",
                "nodules",
                "detections",
                "predictions",
            ):
                value = obj.get(key)

                if isinstance(value, list):
                    return len(value)

    # ---------------------------------------------------------------
    # CSV / TSV fallback
    # ---------------------------------------------------------------

    for p in sorted(stage04_dir.glob("*.csv")):
        try:
            df = pd.read_csv(p)

            for col in (
                "candidate_id",
                "candidate",
                "nodule_id",
                "nodule",
            ):
                if col in df.columns:
                    return int(df[col].nunique())

            if len(df) > 0:
                return int(len(df))

        except Exception:
            continue

    # ---------------------------------------------------------------
    # Candidate-directory fallback
    # ---------------------------------------------------------------

    candidate_dirs = []

    for p in stage04_dir.iterdir():
        if not p.is_dir():
            continue

        name = p.name.lower()

        if (
            "candidate" in name
            or "nodule" in name
            or name.startswith("cand_")
        ):
            candidate_dirs.append(p)

    if candidate_dirs:
        return len(candidate_dirs)

    raise RuntimeError(
        "Could not determine nodule count from Imaging Stage 04. "
        f"Inspect: {stage04_dir}"
    )

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "defaults": DEFAULTS,
        },
    )

@app.get('/api/health')
async def health(): return {'ok': True, 'clinical_models_loaded': pipeline is not None, 'model_load_error': globals().get('MODEL_LOAD_ERROR')}

@app.get('/api/sample')
async def sample():
    return {'PATIENT_ID':'SAMPLE-0001',**DEFAULTS,'ever_pdl1_positive':'No','eastern_cancer_oncology_group':1.0,'karnofsky_performance_score':80.0}

@app.post('/api/predict')
async def predict(
    background_tasks: BackgroundTasks,
    patient_json: str = Form(...),
    ct_files: list[UploadFile] | None = File(None),
):
    if pipeline is None:
        detail = globals().get('MODEL_LOAD_ERROR', 'Clinical artifacts could not be loaded.')
        raise HTTPException(503, f'Clinical artifacts could not be loaded: {detail}')

    try:
        patient = json.loads(patient_json)
    except Exception as e:
        raise HTTPException(400, f'Invalid patient JSON: {e}')

    # Imaging Stage 04, when CT is uploaded, is the sole authority.
    patient.pop("num_distinct_tumor_sites", None)
    run_id = _safe_name(str(patient.get('PATIENT_ID') or uuid.uuid4().hex[:10]))

    # ---------------------------------------------------------------
    # CT UPLOAD
    # ---------------------------------------------------------------

    imaging_info = None
    ct_input = None

    files = [
        f for f in (ct_files or [])
        if f is not None and f.filename
    ]

    if files:
        work = OUTPUT / run_id / "upload"

        if work.exists():
            shutil.rmtree(work)

        work.mkdir(parents=True, exist_ok=True)

        # ===========================================================
        # CASE A — single ZIP / TAR.GZ / DICOM / NIfTI
        # ===========================================================
        if len(files) == 1:
            f = files[0]
            original_name = Path(f.filename).name
            lower = original_name.lower()

            allowed = (
                lower.endswith(".dcm")
                or lower.endswith(".zip")
                or lower.endswith(".tar.gz")
                or lower.endswith(".nii")
                or lower.endswith(".nii.gz")
            )

            if not allowed:
                raise HTTPException(
                    400,
                    "Unsupported CT format. "
                    "Use ZIP, TAR.GZ, DICOM, NIfTI or NIfTI.GZ."
                )

            saved = work / _safe_name(original_name)
            saved.write_bytes(await f.read())

            # NIfTI remains available for any existing NIfTI-capable
            # Imaging stage. Do NOT invent a second imaging pipeline.
            if lower.endswith(".nii") or lower.endswith(".nii.gz"):
                nifti_dir = work / "nifti"
                nifti_dir.mkdir(exist_ok=True)

                shutil.copy2(saved, nifti_dir / saved.name)

                ct_input = nifti_dir

            else:
                ct_input = saved

        # ===========================================================
        # CASE B — browser directory upload
        # ===========================================================
        else:
            folder = work / "dicom_folder"
            folder.mkdir(exist_ok=True)

            written = 0

            for f in files:

                # Browser supplies webkitRelativePath, but FastAPI exposes it
                # through filename.
                rel = f.filename.replace("\\", "/").lstrip("/")
                parts = [
                    p for p in rel.split("/")
                    if p not in ("", ".", "..")
                ]

                if not parts:
                    continue

                destination = folder.joinpath(*parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(await f.read())
                written += 1

            if written == 0:
                raise HTTPException(
                    400,
                    "No files were received from the selected DICOM folder."
                )

            ct_input = folder

        # ===========================================================
        # EXECUTE EXISTING IMAGING PIPELINE
        # ===========================================================
        try:
            imaging_info = _run_existing_imaging(ct_input, run_id)
            mapping_path = OUTPUT / run_id / 'imaging.json'
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            mapping_path.write_text(
                json.dumps(imaging_info, indent=2),
                encoding='utf-8',
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(
                504,
                "The CT imaging pipeline exceeded its timeout."
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise HTTPException(
                500,
                f"CT imaging pipeline failed: {type(exc).__name__}: {exc}"
            )

    try:
        # ---------------------------------------------------------------
        # Imaging-derived tumor count is authoritative.
        # ---------------------------------------------------------------
        if imaging_info:
            patient["num_distinct_tumor_sites"] = float(
                _get_stage04_nodule_count(Path(imaging_info["stage04_dir"]))
            )

        result = pipeline.predict(
            patient,
            ct_output_dir=(
                imaging_info["output_dir"]
                if imaging_info
                else None
            ),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f'Inference failed: {type(e).__name__}: {e}')

    result["run_id"] = run_id
    result["imaging"] = imaging_info or {
        "used": False
    }
    if imaging_info:
        result["imaging"]["stage04_nodule_count"] = int(
            patient["num_distinct_tumor_sites"]
        )
    result["stage09_viewer_url"] = (
        f"/api/ct/{run_id}/viewer"
        if imaging_info
        else None
    )
    return JSONResponse(result)

@app.get('/api/ct/{run_id}/viewer', response_class=HTMLResponse)
async def ct_viewer(run_id: str):
    mapping_path = OUTPUT / _safe_name(run_id) / 'imaging.json'
    if not mapping_path.exists():
        raise HTTPException(404, 'No imaging result exists for this run.')

    try:
        info = json.loads(mapping_path.read_text(encoding='utf-8'))
        viewer = Path(info['stage09_viewer'])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f'Invalid imaging mapping: {exc}')

    if not viewer.is_file():
        raise HTTPException(404, 'Stage 09 viewer.html does not exist.')

    return HTMLResponse(viewer.read_text(encoding='utf-8'))

@app.get('/api/ct/{run_id}/slice/{z}')
async def ct_slice(run_id: str, z: int, window: str='lung'):
    volume=OUTPUT/_safe_name(run_id)/'01'/'volume_hu.npy'
    if not volume.exists(): raise HTTPException(404,'No Stage 01 CT volume found for this run.')
    arr=np.load(volume,mmap_mode='r')
    if z<0 or z>=arr.shape[0]: raise HTTPException(400,'Slice index out of range.')
    hu=np.asarray(arr[z],dtype=np.float32)
    wl,ww=(-600,1500) if window=='lung' else (40,400)
    lo,hi=wl-ww/2,wl+ww/2
    img=np.clip((hu-lo)/(hi-lo),0,1)*255
    im=Image.fromarray(img.astype(np.uint8),'L')
    buf=io.BytesIO(); im.save(buf,format='PNG')
    return Response(buf.getvalue(),media_type='image/png',headers={'X-Slice-Count':str(arr.shape[0]),'X-Shape':f'{arr.shape[1]}x{arr.shape[2]}'})

@app.get('/api/ct/{run_id}/meta')
async def ct_meta(run_id: str):
    p=OUTPUT/_safe_name(run_id)/'01'/'meta.json'
    v=OUTPUT/_safe_name(run_id)/'01'/'volume_hu.npy'
    if not v.exists(): raise HTTPException(404,'No CT volume found.')
    meta=json.loads(p.read_text()) if p.exists() else {}
    shape=list(np.load(v,mmap_mode='r').shape); return {'shape_zyx':shape,'meta':meta}

@app.post('/api/report')
async def report(payload: dict):
    pdf=generate_pdf_report(payload.get('patient',{}),payload.get('prediction_results',payload))
    return Response(pdf,media_type='application/pdf',headers={'Content-Disposition':'attachment; filename="LungInsight_Report.pdf"'})
