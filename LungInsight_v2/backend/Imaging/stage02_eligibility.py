"""Fail-closed validation of the selected, lung-mask-gated Stage 02 candidates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_schema import get_required, load_schema, validate_stage_version

SCHEMA = load_schema(__file__)
STAGE01 = get_required(SCHEMA, "stage_01_hu_and_crop", expected_type=dict)
STAGE02 = get_required(SCHEMA, "stage_02_detect_and_patch", expected_type=dict)
STAGE02_PATHS = get_required(SCHEMA, "paths.stage_02", expected_type=dict)
PATCH_OUTPUTS = get_required(STAGE02, "patch_outputs", expected_type=dict)
STAGE02_FILES = get_required(STAGE02, "filenames", expected_type=dict)
STAGE01_FILES = get_required(STAGE01, "filenames", expected_type=dict)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc


def _finite_vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite values.")
    return result


def load_validated_stage02(
    patient_root: str | Path,
    manifest_path: str | Path | None = None,
    *,
    verify_stage01_volume: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return the primary manifest and its exact selected candidate set after rechecking Stage 01."""
    patient_root = Path(patient_root).resolve()
    stage02_root = patient_root / str(STAGE02_PATHS["directory"])
    if manifest_path is None:
        manifest_path = patient_root / str(PATCH_OUTPUTS["manifest"])
    manifest_path = Path(manifest_path).resolve()
    report_path = stage02_root / str(STAGE02_FILES["candidate_filter_report"])
    candidates_path = stage02_root / str(STAGE02_FILES["candidates_json"])
    stage01_paths = {
        key: patient_root / str(value)
        for key, value in STAGE01_FILES.items()
        if key in {
            "volume_hu_cropped.npy",
            "lung_mask.npy",
            "heart_mask.npy",
            "meta.json",
            "mask_quality_report.json",
        }
    }
    for path in (manifest_path, report_path, candidates_path, *stage01_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Required gated Stage 02/Stage 01 artifact is missing: {path}")

    report = _read_json(report_path)
    metadata = _read_json(stage01_paths["meta.json"])
    if report.get("gate_status") != "passed":
        raise RuntimeError(f"Stage 02 acceptance gate did not pass: {report_path}")

    quality = metadata.get("mask_quality")
    if not isinstance(quality, dict) or quality.get("status") not in {"validated_automated", "ready_for_review"}:
        raise RuntimeError("Stage 01 automated mask-quality validation did not pass.")
    if metadata.get("coordinate_system") != "DICOM_LPS" or metadata.get("array_axes") != ["z", "y", "x"]:
        raise RuntimeError("Stage 01 coordinate system and axes are not the validated DICOM-LPS ZYX contract.")
    volume = np.load(stage01_paths["volume_hu_cropped.npy"], mmap_mode="r", allow_pickle=False)
    lung_values = np.load(stage01_paths["lung_mask.npy"], mmap_mode="r", allow_pickle=False)
    heart_values = np.load(stage01_paths["heart_mask.npy"], mmap_mode="r", allow_pickle=False)
    if volume.ndim != 3:
        raise RuntimeError("Stage 01 CT volume must be a 3D ZYX array.")
    for name, values in (("lung", lung_values), ("heart", heart_values)):
        if values.shape != volume.shape:
            raise RuntimeError(f"Stage 01 {name} mask is invalid or misaligned with the CT volume.")
    lung_count = 0
    heart_count = 0
    overlap = 0
    for z in range(0, volume.shape[0], 16):
        end_z = min(z + 16, volume.shape[0])
        volume_chunk = volume[z:end_z]
        lung_chunk = lung_values[z:end_z]
        heart_chunk = heart_values[z:end_z]
        if verify_stage01_volume and not np.isfinite(volume_chunk).all():
            raise RuntimeError("Stage 01 CT volume contains NaN or infinite values.")
        if (
            lung_chunk.dtype != np.bool_ and not np.isin(lung_chunk, (0, 1)).all()
        ) or (
            heart_chunk.dtype != np.bool_ and not np.isin(heart_chunk, (0, 1)).all()
        ):
            raise RuntimeError("Stage 01 masks must contain only boolean or 0/1 values.")
        lung_bool = lung_chunk.astype(bool, copy=False)
        heart_bool = heart_chunk.astype(bool, copy=False)
        lung_count += int(lung_bool.sum())
        heart_count += int(heart_bool.sum())
        overlap += int(np.count_nonzero(lung_bool & heart_bool))
    if (
        lung_count == 0
        or heart_count == 0
        or overlap
        or quality.get("shape_zyx") != list(volume.shape)
        or quality.get("problems") != []
        or quality.get("lung_voxels") != lung_count
        or quality.get("heart_voxels") != heart_count
        or quality.get("lung_heart_overlap_voxels") != overlap
    ):
        raise RuntimeError("Stage 01 mask-quality diagnostics do not match the saved aligned masks.")

    manifest = _read_json(manifest_path)
    validate_stage_version(
        manifest.get("schema_version"),
        STAGE02["manifest_schema_version"],
        context="Stage 02 primary patch manifest",
    )
    if (
        manifest.get("coordinate_system") != SCHEMA["coordinate_system"]
        or manifest.get("coordinate_order") != "ZYX"
    ):
        raise RuntimeError("Stage 02 manifest coordinate frame or array order is not canonical.")
    geometry = manifest.get("ct_geometry")
    if not isinstance(geometry, dict) or geometry.get("shape_zyx") != list(volume.shape):
        raise RuntimeError("Stage 02 CT geometry does not match the current Stage 01 CT shape.")
    spacing = _finite_vector(metadata.get("spacing_zyx_mm"), "Stage 01 spacing_zyx_mm")
    origin = _finite_vector(metadata.get("origin_mm"), "Stage 01 origin_mm")
    direction = np.asarray(metadata.get("direction_lps"), dtype=np.float64)
    if (
        np.any(spacing <= 0)
        or direction.shape != (3, 3)
        or not np.isfinite(direction).all()
        or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4)
        or not np.isclose(abs(np.linalg.det(direction)), 1.0, atol=1e-4)
        or not np.allclose(np.asarray(geometry.get("spacing_zyx_mm"), dtype=np.float64), spacing, atol=1e-5)
        or not np.allclose(np.asarray(geometry.get("origin_lps_mm"), dtype=np.float64), origin, atol=1e-4)
        or not np.allclose(np.asarray(geometry.get("direction_lps"), dtype=np.float64), direction, atol=1e-5)
    ):
        raise RuntimeError("Stage 02 physical geometry does not match validated Stage 01 geometry.")

    candidate_rows = _read_json(candidates_path)
    if isinstance(candidate_rows, dict):
        candidate_rows = candidate_rows.get("candidates")
    if not isinstance(candidate_rows, list) or any(not isinstance(row, dict) for row in candidate_rows):
        raise ValueError(f"Stage 02 selected candidates are malformed: {candidates_path}")
    accepted: dict[str, dict[str, Any]] = {}
    for candidate in candidate_rows:
        candidate_id = candidate.get("candidate_id")
        stable_id = candidate.get("stable_nodule_id", candidate.get("nodule_id"))
        if not isinstance(candidate_id, str) or not candidate_id or stable_id != candidate_id:
            raise ValueError("Selected Stage 02 candidate lacks a consistent stable identity.")
        if candidate_id in accepted:
            raise ValueError(f"Duplicate selected Stage 02 candidate ID: {candidate_id}")
        center = _finite_vector(candidate.get("center_zyx"), f"{candidate_id}.center_zyx")
        index = np.rint(center).astype(np.int64)
        if (
            np.any(index < 0)
            or np.any(index >= np.asarray(lung_values.shape))
            or not bool(lung_values[tuple(index)])
            or bool(heart_values[tuple(index)])
        ):
            raise RuntimeError(
                f"Selected candidate {candidate_id} is not inside the current validated lung mask "
                "outside the heart exclusion mask."
            )
        if candidate.get("center_inside_lung_mask") is not True:
            raise RuntimeError(f"Selected candidate {candidate_id} lacks positive Stage 02 lung-mask eligibility.")
        accepted[candidate_id] = candidate

    patches = manifest.get("patches")
    if not isinstance(patches, list) or any(not isinstance(patch, dict) for patch in patches):
        raise ValueError(f"{manifest_path}: expected a list of patch objects.")
    patch_ids: set[int] = set()
    patch_candidates: set[str] = set()
    for patch in patches:
        try:
            patch_id = int(patch["patch_id"])
            candidate_id = patch["candidate_id"]
            stable_id = patch["stable_nodule_id"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{manifest_path}: patch identity is incomplete.") from exc
        if patch_id in patch_ids or not isinstance(candidate_id, str) or stable_id != candidate_id:
            raise ValueError(f"{manifest_path}: duplicate or inconsistent patch identity.")
        patch_ids.add(patch_id)
        patch_candidates.add(candidate_id)
        if candidate_id not in accepted:
            raise RuntimeError(f"Manifest candidate {candidate_id} is absent from the selected Stage 02 list.")
        center = _finite_vector(patch.get("center_zyx"), f"patch {patch_id}.center_zyx")
        candidate_center = _finite_vector(accepted[candidate_id].get("center_zyx"), f"{candidate_id}.center_zyx")
        if not np.allclose(center, candidate_center, rtol=0.0, atol=1e-3):
            raise RuntimeError(f"Patch {patch_id} center differs from selected candidate {candidate_id}.")
        index = np.rint(center).astype(np.int64)
        if (
            np.any(index < 0)
            or np.any(index >= np.asarray(lung_values.shape))
            or not bool(lung_values[tuple(index)])
            or bool(heart_values[tuple(index)])
        ):
            raise RuntimeError(
                f"Patch {patch_id} center is not inside the current validated lung mask "
                "outside the heart exclusion mask."
            )
        start = _finite_vector(patch.get("crop_start_zyx"), f"patch {patch_id}.crop_start_zyx")
        end = _finite_vector(patch.get("crop_end_zyx"), f"patch {patch_id}.crop_end_zyx")
        size = _finite_vector(patch.get("patch_size_vox_zyx"), f"patch {patch_id}.patch_size_vox_zyx")
        if not np.array_equal(start, np.rint(start)) or not np.array_equal(end, np.rint(end)) or not np.array_equal(size, np.rint(size)):
            raise ValueError(f"Patch {patch_id} crop geometry must use integer voxel coordinates.")
        if np.any(size <= 0) or not np.array_equal(end - start, size):
            raise ValueError(f"Patch {patch_id} crop extent disagrees with patch_size_vox_zyx.")
        if patch.get("source_space") != SCHEMA["coordinate_format"]["ct_frame_space"]:
            raise ValueError(f"Patch {patch_id} is not in the canonical Stage 01 CT coordinate frame.")
    if patch_candidates != set(accepted):
        raise RuntimeError("Stage 02 selected candidate list and primary patch manifest do not match exactly.")
    return manifest, sorted(patches, key=lambda patch: int(patch["patch_id"])), accepted
