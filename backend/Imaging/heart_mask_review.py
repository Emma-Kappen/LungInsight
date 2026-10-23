"""Human-review attestation and review-montage helpers for Stage 01 masks."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class HeartMaskReviewError(RuntimeError):
    """The Stage 01 heart/lung mask review gate is not satisfied."""


_REVIEWER_PLACEHOLDERS = {
    "your_name",
    "your name",
    "reviewer_name",
    "reviewer name",
    "actual_operator_id",
    "actual operator id",
}


def _require_reviewer_identity(reviewer: Any) -> str:
    value = str(reviewer or "").strip()
    if not value:
        raise HeartMaskReviewError("Reviewer identity must not be empty.")
    if value.casefold() in _REVIEWER_PLACEHOLDERS:
        raise HeartMaskReviewError(
            "Replace the reviewer placeholder with the actual operator identity."
        )
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage01_filenames(schema: dict[str, Any]) -> dict[str, str]:
    try:
        result = schema["stage_01_hu_and_crop"]["filenames"]
    except (KeyError, TypeError) as exc:
        raise HeartMaskReviewError(
            "meta_schema.json is missing stage_01_hu_and_crop.filenames."
        ) from exc
    if not isinstance(result, dict):
        raise HeartMaskReviewError(
            "meta_schema.json stage_01_hu_and_crop.filenames must be an object."
        )
    return {str(key): str(value) for key, value in result.items()}


def _required_file(directory: Path, filename: str, description: str) -> Path:
    path = directory / filename
    if not path.is_file():
        raise HeartMaskReviewError(f"Missing {description}: {path}")
    return path


def write_heart_mask_review_montages(
    volume_hu: np.ndarray,
    lung_mask: np.ndarray,
    heart_mask: np.ndarray,
    output_dir: str | Path,
    schema: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Write paginated axial CT/mask overlays and per-slice geometry metrics."""
    volume = np.asarray(volume_hu)
    lung = np.asarray(lung_mask, dtype=bool)
    heart = np.asarray(heart_mask, dtype=bool)
    if volume.ndim != 3 or lung.shape != volume.shape or heart.shape != volume.shape:
        raise ValueError(
            f"Review montage requires aligned 3-D ZYX arrays; got "
            f"CT={volume.shape}, lung={lung.shape}, heart={heart.shape}."
        )
    if not np.isfinite(volume).all():
        raise ValueError("Review montage CT volume contains NaN or Inf values.")

    try:
        review_schema = schema["stage_01_hu_and_crop"]["review_qc"]
        filenames = _stage01_filenames(schema)
        slices_per_page = int(review_schema["slices_per_montage"])
        columns = int(review_schema["columns"])
        filename_pattern = filenames["heart_mask_review_montage_pattern"]
    except (KeyError, TypeError, ValueError) as exc:
        raise HeartMaskReviewError(
            "meta_schema.json must define Stage 01 review_qc and "
            "heart_mask_review_montage_pattern."
        ) from exc
    if slices_per_page <= 0 or columns <= 0:
        raise HeartMaskReviewError(
            "Stage 01 review_qc slices_per_montage and columns must be positive."
        )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    montage_files: list[str] = []
    slice_metrics: list[dict[str, Any]] = []
    z_size = int(volume.shape[0])
    rows = int(math.ceil(slices_per_page / columns))

    for start in range(0, z_size, slices_per_page):
        page = start // slices_per_page
        stop = min(start + slices_per_page, z_size)
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 3.2), squeeze=False)
        for offset, axis in enumerate(axes.flat):
            z = start + offset
            axis.set_axis_off()
            if z >= stop:
                continue
            ct_slice = volume[z]
            lung_slice = lung[z]
            heart_slice = heart[z]
            axis.imshow(ct_slice, cmap="gray", vmin=-1000, vmax=400, interpolation="nearest")
            if lung_slice.any():
                axis.contour(lung_slice, levels=[0.5], colors=["lime"], linewidths=0.55)
            if heart_slice.any():
                axis.contour(heart_slice, levels=[0.5], colors=["magenta"], linewidths=0.7)
            axis.set_title(
                f"z={z} lung={int(lung_slice.sum())} heart={int(heart_slice.sum())}",
                fontsize=7,
            )
            slice_metrics.append(
                {
                    "z": z,
                    "lung_voxels": int(lung_slice.sum()),
                    "heart_voxels": int(heart_slice.sum()),
                    "heart_lung_overlap_voxels": int(np.count_nonzero(lung_slice & heart_slice)),
                    "heart_fraction_of_slice": float(heart_slice.mean()),
                    "lung_fraction_of_slice": float(lung_slice.mean()),
                }
            )
        fig.suptitle(
            f"Stage 01 heart/lung mask review: axial slices {start}–{stop - 1}\n"
            "Lung contour = green; central heart/exclusion mask = magenta",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        filename = filename_pattern.format(page=page)
        destination = output / filename
        fig.savefig(destination, dpi=120)
        plt.close(fig)
        montage_files.append(filename)

    return montage_files, slice_metrics


def _current_artifact_paths(
    patient_dir: Path,
    schema: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Path]:
    filenames = _stage01_filenames(schema)
    montage_files = metadata.get("heart_mask_review_montage_files")
    if not isinstance(montage_files, list) or not montage_files or not all(
        isinstance(name, str) and name for name in montage_files
    ):
        raise HeartMaskReviewError(
            "Stage 01 metadata has no review montage list; regenerate Stage 01 with the current pipeline."
        )
    keys = (
        "volume_hu_cropped.npy",
        "lung_mask.npy",
        "heart_mask.npy",
        "heart_tracking_diagnostics.json",
        "heart_mask_review_metrics.json",
    )
    paths = {
        key: _required_file(patient_dir, filenames[key], key)
        for key in keys
    }
    for name in montage_files:
        paths[f"montage:{name}"] = _required_file(patient_dir, name, "heart-mask review montage")
    return paths


def _validate_review_inputs(
    patient_dir: Path,
    schema: dict[str, Any],
    metadata: dict[str, Any],
    volume: np.ndarray,
    lung_mask: np.ndarray,
    heart_mask: np.ndarray,
) -> dict[str, Path]:
    volume = np.asarray(volume)
    lung = np.asarray(lung_mask, dtype=bool)
    heart = np.asarray(heart_mask, dtype=bool)
    if volume.ndim != 3 or lung.shape != volume.shape or heart.shape != volume.shape:
        raise HeartMaskReviewError(
            f"Stage 01 review arrays are not aligned: CT={volume.shape}, lung={lung.shape}, heart={heart.shape}."
        )
    if not np.isfinite(volume).all():
        raise HeartMaskReviewError("Stage 01 CT volume contains NaN or Inf values.")
    if not lung.any():
        raise HeartMaskReviewError("Stage 01 lung mask is empty; downstream inference is blocked.")
    if not heart.any():
        raise HeartMaskReviewError("Stage 01 heart/exclusion mask is empty; downstream inference is blocked.")
    if metadata.get("heart_mask_tracking_successful") is not True:
        raise HeartMaskReviewError(
            "Stage 01 heart-mask tracking did not report success; downstream inference is blocked."
        )
    if metadata.get("heart_mask_review_montages_generated") is not True:
        raise HeartMaskReviewError(
            "Stage 01 review overlays are not marked complete; regenerate Stage 01."
        )
    return _current_artifact_paths(patient_dir, schema, metadata)


def create_review_attestation(
    patient_dir: str | Path,
    schema: dict[str, Any],
    *,
    reviewer: str,
    confirmed_review: bool,
) -> Path:
    """Persist an explicit operator assertion tied to hashes of reviewed inputs."""
    directory = Path(patient_dir).resolve()
    reviewer = _require_reviewer_identity(reviewer)
    if not confirmed_review:
        raise HeartMaskReviewError(
            "No attestation was written. Review every Stage 01 montage and pass --confirm-review to attest."
        )
    filenames = _stage01_filenames(schema)
    metadata_path = _required_file(directory, filenames["meta.json"], "Stage 01 metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    volume = np.load(directory / filenames["volume_hu_cropped.npy"], allow_pickle=False)
    lung = np.load(directory / filenames["lung_mask.npy"], allow_pickle=False)
    heart = np.load(directory / filenames["heart_mask.npy"], allow_pickle=False)
    paths = _validate_review_inputs(directory, schema, metadata, volume, lung, heart)
    artifact_hashes = {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
    }
    payload = {
        "schema_version": "heart-mask-review-attestation-v1",
        "meta_schema_version": str(schema["schema_version"]),
        "patient_id": str(metadata.get("patient_id", directory.name)),
        "reviewed": True,
        "reviewer": reviewer,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_scope": "visual inspection of all Stage 01 axial review montages and slice metrics",
        "clinical_validation": False,
        "ct_shape_zyx": list(volume.shape),
        "artifacts_sha256": artifact_hashes,
        "limitations": (
            "Operator review is a processing gate, not clinical validation or proof of anatomical segmentation accuracy."
        ),
    }
    destination = directory / filenames["heart_mask_review_attestation.json"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def validate_review_attestation(
    patient_dir: str | Path,
    schema: dict[str, Any],
    metadata: dict[str, Any],
    volume: np.ndarray,
    lung_mask: np.ndarray,
    heart_mask: np.ndarray,
) -> dict[str, Any]:
    """Verify the operator attestation still matches the exact Stage 01 artifacts."""
    directory = Path(patient_dir).resolve()
    filenames = _stage01_filenames(schema)
    paths = _validate_review_inputs(
        directory, schema, metadata, volume, lung_mask, heart_mask
    )
    attestation_path = _required_file(
        directory, filenames["heart_mask_review_attestation.json"], "heart-mask review attestation"
    )
    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise HeartMaskReviewError(
            f"Malformed heart-mask review attestation {attestation_path}: {exc}"
        ) from exc
    if not isinstance(attestation, dict):
        raise HeartMaskReviewError(f"Review attestation must be a JSON object: {attestation_path}")
    if attestation.get("schema_version") != "heart-mask-review-attestation-v1":
        raise HeartMaskReviewError("Unsupported heart-mask review attestation schema version.")
    if str(attestation.get("meta_schema_version")) != str(schema.get("schema_version")):
        raise HeartMaskReviewError(
            "Heart-mask review attestation schema version does not match the active meta_schema.json."
        )
    if attestation.get("patient_id") != str(metadata.get("patient_id", directory.name)):
        raise HeartMaskReviewError("Heart-mask review attestation belongs to a different patient.")
    if attestation.get("reviewed") is not True:
        raise HeartMaskReviewError("Heart-mask review attestation is not explicitly approved or has no reviewer.")
    reviewer = _require_reviewer_identity(attestation.get("reviewer"))
    if attestation.get("clinical_validation") is not False:
        raise HeartMaskReviewError("Heart-mask attestation must explicitly disclaim clinical validation.")
    if attestation.get("ct_shape_zyx") != list(np.asarray(volume).shape):
        raise HeartMaskReviewError("Heart-mask review attestation CT shape is stale.")
    expected_hashes = {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
    }
    if attestation.get("artifacts_sha256") != expected_hashes:
        raise HeartMaskReviewError(
            "Stage 01 CT, masks, diagnostics, review metrics, or montage files changed after review; "
            "review the current outputs and create a fresh attestation."
        )
    return {
        "verified": True,
        "attestation_path": str(attestation_path),
        "reviewer": reviewer,
        "reviewed_at_utc": str(attestation.get("reviewed_at_utc", "")),
        "heart_mask_voxels": int(np.asarray(heart_mask, dtype=bool).sum()),
        "lung_mask_voxels": int(np.asarray(lung_mask, dtype=bool).sum()),
        "artifact_hashes": expected_hashes,
    }
