from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pydicom
from fastapi import UploadFile

logger = logging.getLogger("lunginsight.imaging_api.dicom")

MAX_FILE_COUNT = 4096
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
PATIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_JOBS: dict[str, dict[str, Any]] = {}


class DicomUploadError(ValueError):
    """Raised when a raw upload cannot be safely converted to one CT series."""


@dataclass(frozen=True)
class DicomSeries:
    datasets: tuple[Any, ...]
    study_uid: str
    series_uid: str
    patient_key: str
    rows: int
    columns: int
    spacing_xy_mm: tuple[float, float]
    spacing_z_mm: float
    orientation: tuple[float, ...]
    volume_hu: np.ndarray


def _safe_relative_name(name: str | None, index: int) -> Path:
    raw = (name or f"upload_{index}.dcm").replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise DicomUploadError("An uploaded filename contains an unsafe path.")
    parts = [part for part in path.parts if part not in {"", "."}]
    if not parts or any(":" in part for part in parts):
        raise DicomUploadError("An uploaded filename is invalid.")
    return Path(*parts)


def _patient_key(dataset: Any) -> str:
    patient_id = str(getattr(dataset, "PatientID", "") or "").strip()
    patient_name = str(getattr(dataset, "PatientName", "") or "").strip()
    return patient_id or patient_name or "<missing-patient>"


def _float_tuple(value: Any, length: int, field: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise DicomUploadError(f"DICOM metadata field {field} is invalid.") from exc
    if len(result) != length or not all(np.isfinite(result)):
        raise DicomUploadError(f"DICOM metadata field {field} is invalid.")
    return result


def _read_datasets(root: Path) -> list[Any]:
    datasets: list[Any] = []
    file_count = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        try:
            dataset = pydicom.dcmread(path, force=False)
        except (OSError, pydicom.errors.InvalidDicomError):
            raise DicomUploadError(
                "The upload contains a file that is not a readable DICOM object."
            )
        if str(getattr(dataset, "Modality", "")).upper() == "CT" and hasattr(dataset, "PixelData"):
            datasets.append(dataset)
    if not file_count or not datasets:
        raise DicomUploadError("No readable CT DICOM instances were found in the upload.")
    return datasets


def _select_series(datasets: Iterable[Any]) -> tuple[list[Any], str, str, str]:
    groups: dict[tuple[str, str, str], list[Any]] = {}
    for dataset in datasets:
        study = str(getattr(dataset, "StudyInstanceUID", "") or "")
        series = str(getattr(dataset, "SeriesInstanceUID", "") or "")
        if not study or not series:
            continue
        groups.setdefault((study, series, _patient_key(dataset)), []).append(dataset)
    if not groups:
        raise DicomUploadError("The CT files do not contain Study/Series identifiers.")
    if len(groups) != 1:
        raise DicomUploadError(
            "The upload contains multiple CT studies, series, or patients. Select one CT series only."
        )
    (study, series, patient), selected = next(iter(groups.items()))
    return selected, study, series, patient


def _reconstruct(datasets: list[Any], study_uid: str, series_uid: str, patient_key: str) -> DicomSeries:
    first = datasets[0]
    rows = int(getattr(first, "Rows", 0) or 0)
    columns = int(getattr(first, "Columns", 0) or 0)
    if rows <= 0 or columns <= 0:
        raise DicomUploadError("The CT series has invalid image dimensions.")
    spacing_xy = _float_tuple(getattr(first, "PixelSpacing", None), 2, "PixelSpacing")
    orientation = _float_tuple(
        getattr(first, "ImageOrientationPatient", None), 6, "ImageOrientationPatient"
    )
    row_cosine = np.asarray(orientation[:3], dtype=np.float64)
    column_cosine = np.asarray(orientation[3:], dtype=np.float64)
    normal = np.cross(row_cosine, column_cosine)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 0.999 or normal_norm > 1.001:
        raise DicomUploadError("The CT series has invalid orientation vectors.")
    normalized = []
    positions: list[float] = []
    for dataset in datasets:
        if int(getattr(dataset, "Rows", 0) or 0) != rows or int(getattr(dataset, "Columns", 0) or 0) != columns:
            raise DicomUploadError("The CT series contains inconsistent image dimensions.")
        current_orientation = _float_tuple(
            getattr(dataset, "ImageOrientationPatient", None), 6, "ImageOrientationPatient"
        )
        if not np.allclose(current_orientation, orientation, atol=1e-4):
            raise DicomUploadError("The CT series contains inconsistent slice orientation.")
        position = _float_tuple(getattr(dataset, "ImagePositionPatient", None), 3, "ImagePositionPatient")
        positions.append(float(np.dot(np.asarray(position), normal)))
        normalized.append((dataset, position))
    order = np.argsort(positions)
    ordered = [normalized[int(index)][0] for index in order]
    ordered_positions = [positions[int(index)] for index in order]
    if len(ordered) < 2:
        raise DicomUploadError("At least two CT slices are required to reconstruct a volume.")
    deltas = np.diff(ordered_positions)
    spacing_z = float(np.median(np.abs(deltas)))
    if not np.isfinite(spacing_z) or spacing_z <= 0 or not np.allclose(np.abs(deltas), spacing_z, rtol=0.02, atol=0.01):
        raise DicomUploadError("The CT series has inconsistent slice spacing or duplicate positions.")
    arrays = []
    for dataset in ordered:
        try:
            pixels = dataset.pixel_array.astype(np.float32, copy=False)
        except (AttributeError, ValueError, RuntimeError) as exc:
            raise DicomUploadError("A CT slice pixel payload could not be decoded.") from exc
        if pixels.shape != (rows, columns):
            raise DicomUploadError("A CT slice has unexpected pixel dimensions.")
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        if not np.isfinite(slope) or not np.isfinite(intercept):
            raise DicomUploadError("The CT series has invalid intensity calibration metadata.")
        arrays.append(pixels * slope + intercept)
    volume = np.stack(arrays, axis=0).astype(np.float32, copy=False)
    return DicomSeries(
        datasets=tuple(ordered),
        study_uid=study_uid,
        series_uid=series_uid,
        patient_key=patient_key,
        rows=rows,
        columns=columns,
        spacing_xy_mm=(spacing_xy[0], spacing_xy[1]),
        spacing_z_mm=spacing_z,
        orientation=orientation,
        volume_hu=volume,
    )


async def process_dicom_upload(
    files: list[UploadFile],
    *,
    patient_id: str,
    output_root: Path,
) -> dict[str, Any]:
    if not PATIENT_ID_PATTERN.fullmatch(patient_id):
        raise DicomUploadError("Invalid patient identifier.")
    if not files:
        raise DicomUploadError("Select a DICOM folder or one or more DICOM files.")
    if len(files) > MAX_FILE_COUNT:
        raise DicomUploadError(f"Too many files. The maximum is {MAX_FILE_COUNT:,}.")
    job_id = secrets.token_urlsafe(18)
    job_root = (output_root / patient_id / "dicom_uploads" / job_id).resolve()
    if job_root.parent.parent.parent != output_root.resolve():
        raise DicomUploadError("Invalid upload destination.")
    temp_root = Path(tempfile.mkdtemp(prefix="lunginsight-dicom-"))
    _JOBS[job_id] = {"job_id": job_id, "status": "uploading", "patient_id": patient_id}
    total_bytes = 0
    try:
        for index, upload in enumerate(files):
            relative = _safe_relative_name(upload.filename, index)
            destination = temp_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with destination.open("wb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    written += len(chunk)
                    total_bytes += len(chunk)
                    if written > MAX_FILE_BYTES or total_bytes > MAX_TOTAL_BYTES:
                        raise DicomUploadError("The upload exceeds the permitted file or total size.")
                    handle.write(chunk)
        _JOBS[job_id]["status"] = "validating_series"
        series = _reconstruct(*_select_series(_read_datasets(temp_root)))
        job_root.mkdir(parents=True, exist_ok=False)
        stage01_input_root = job_root / "stage01_input" / patient_id
        shutil.copytree(temp_root, stage01_input_root)
        np.save(job_root / "volume_hu.npy", series.volume_hu, allow_pickle=False)
        metadata = {
            "job_id": job_id,
            "patient_id": patient_id,
            "study_instance_uid_sha256": hashlib.sha256(series.study_uid.encode()).hexdigest(),
            "series_instance_uid_sha256": hashlib.sha256(series.series_uid.encode()).hexdigest(),
            "slice_count": len(series.datasets),
            "volume_shape_zyx": list(series.volume_hu.shape),
            "spacing_zyx_mm": [series.spacing_z_mm, *series.spacing_xy_mm],
            "direction_lps": [list(series.orientation[:3]), list(series.orientation[3:])],
            "intensity_calibration": "RescaleSlope and RescaleIntercept applied to produce HU.",
            "stage01_input_relative_path": f"dicom_uploads/{job_id}/stage01_input/{patient_id}",
        }
        (job_root / "dicom_preprocessing.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        result = {
            "job_id": job_id,
            "status": "preprocessing_complete",
            "analysis_status": "blocked",
            "message": (
                "The CT series was validated and reconstructed in HU. The existing multi-stage "
                "Imaging pipeline must be run before an analysis result can be reported."
            ),
            "next_step": (
                "An authorized operator must review the Stage 01 montage, create the hash-bound "
                "heart-mask attestation, then run the supplied runner command. This upload "
                "artifact is not an automatic Stage 01-07 pipeline invocation."
            ),
            "series": {
                "file_count": len(files),
                "slice_count": len(series.datasets),
                "volume_shape_zyx": list(series.volume_hu.shape),
                "spacing_zyx_mm": [series.spacing_z_mm, *series.spacing_xy_mm],
            },
            "pipeline_input": f"dicom_uploads/{job_id}/stage01_input/{patient_id}",
            "runner_command": (
                f"powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\run_pipeline.ps1 "
                f"-PatientId \"{patient_id}\" -InputPatientPath "
                f"\"backend\\output\\{patient_id}\\dicom_uploads\\{job_id}\\stage01_input\\{patient_id}\" -NoGui"
            ),
        }
        _JOBS[job_id] = result
        return result
    except Exception as exc:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "analysis_status": "blocked",
            "message": str(exc) if isinstance(exc, DicomUploadError) else "DICOM preprocessing failed.",
        }
        if not isinstance(exc, DicomUploadError):
            logger.exception("dicom_preprocessing_failed")
        raise
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        for upload in files:
            await upload.close()


def get_dicom_job(job_id: str) -> dict[str, Any]:
    job = _JOBS.get(job_id)
    if job is None:
        raise DicomUploadError("DICOM processing job not found.")
    return dict(job)
