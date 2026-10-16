"""
LungInsight v2 -- Stage 02: candidate detection + nodule patch extraction.

This file consolidates the supplied 04_detect_candidates.py and
05_extract_patches.py implementations into the v2 Stage 02 layout.

Authoritative candidate detectors:
    ViTDet3D (rlsn/DeTr4LungNodule) and LoG

Both detectors contribute to the validated, deduplicated authoritative
candidate set. Detector provenance and source-specific score semantics are
preserved in every fused record.

Input:
    LungInsight_v2/backend/output/<patient_id>/
        volume_hu_cropped.npy
        lung_mask.npy
        meta.json

Output:
    LungInsight_v2/backend/output/<patient_id>/02_detect_and_patch/
        candidates.json
        candidates.csv
        detector_metadata.json
        log_candidates_diagnostic.json
        candidates_diagnostic_agreement.json
        patches/
            arrays/patch_000.npy
            visualizations/patch_000_triplanar.png
            patches_manifest.json

Coordinate contract:
    * Array order is ZYX = [z, y, x].
    * CT physical coordinates are DICOM_LPS millimetres.
    * Stage 01 meta.json defines:
          p_LPS_mm = origin_mm + direction_lps @ ([z,y,x] * spacing_zyx_mm)
      where origin_mm is the physical LPS coordinate of the CENTER of the
      saved CT voxel [0,0,0] in volume_hu_cropped.npy.
    * Detector candidate centers and detector bboxes are in the native
      cropped CT voxel frame.
    * Patches are cropped from that native frame and are not resampled.
    * patch_origin_lps_mm is the LPS coordinate of the CENTER of patch voxel
      [0,0,0], including virtual padded positions outside the CT volume.
    * Patch mapping:
          p_LPS_mm = patch_origin_lps_mm
                      + direction_lps @ ([z,y,x] * spacing_zyx_mm)

The old Stage 04/05 scripts are not imported or dynamically loaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

IMAGING_DIR = Path(__file__).resolve().parent
if IMAGING_DIR.name in {"stages", "inference", "visualization", "validation"}:
    IMAGING_DIR = IMAGING_DIR.parent
if str(IMAGING_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGING_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle
from pipeline_schema import get_required, load_schema, check_meta_schema_version
from pipeline_logging import add_verbose_argument, configure_console, pipeline_print as print, stage_complete
from heart_mask_review import validate_review_attestation
from lunginsight_candidate_fusion import (
    filter_detections_by_lung_mask,
    merge_detections,
    select_fused_candidates,
)

SCHEMA = load_schema(__file__)
STAGE02_SCHEMA = get_required(SCHEMA, "stage_02_detect_and_patch", expected_type=dict)
STAGE01_SCHEMA = get_required(SCHEMA, "stage_01_hu_and_crop", expected_type=dict)
STAGE01_FILES = get_required(STAGE01_SCHEMA, "filenames", expected_type=dict)
STAGE02_DETECTOR = get_required(STAGE02_SCHEMA, "authoritative_detector", expected_type=dict)
STAGE02_ACCEPTANCE_FILTER = get_required(STAGE02_SCHEMA, "candidate_acceptance_filter", expected_type=dict)
STAGE02_LOCALIZATION = get_required(STAGE02_SCHEMA, "anatomical_localization.default_parameters", expected_type=dict)
STAGE02_PATHS = get_required(SCHEMA, "paths.stage_02", expected_type=dict)
STAGE02_FILES = get_required(STAGE02_SCHEMA, "filenames", expected_type=dict)
STAGE02_PATCH_OUTPUTS = get_required(STAGE02_SCHEMA, "patch_outputs", expected_type=dict)
STAGE02_LOG_PATCH_OUTPUTS = get_required(STAGE02_SCHEMA, "diagnostic_patch_outputs", expected_type=dict)
STAGE02_PATCH_ARRAY_SUBDIR = Path(STAGE02_PATCH_OUTPUTS["array_directory"]).relative_to(Path(STAGE02_PATCH_OUTPUTS["root"]))
STAGE02_PATCH_VIZ_SUBDIR = Path(STAGE02_PATCH_OUTPUTS["visualization_directory"]).relative_to(Path(STAGE02_PATCH_OUTPUTS["root"]))
STAGE02_PATCH_ARRAY_PATTERN = str(STAGE02_FILES["patch_array_pattern"])
STAGE02_PATCH_VIZ_PATTERN = str(STAGE02_FILES["patch_visualization_pattern"])
PATCH_ARRAY_DTYPE = np.dtype(STAGE02_PATCH_OUTPUTS["array_dtype"])
if PATCH_ARRAY_DTYPE != np.dtype(np.float32):
    raise RuntimeError(
        f"meta_schema.json Stage 02 patch array dtype must remain float32 for the existing Stage 03 checkpoints, got {PATCH_ARRAY_DTYPE}."
    )
STAGE01_FALLBACK_AIR_THRESHOLD_HU = float(get_required(SCHEMA, "stage_01_hu_and_crop.preprocessing.air_threshold_hu"))
from matplotlib.widgets import Slider
from scipy import ndimage
from scipy.ndimage import gaussian_laplace, maximum_filter

try:
    from transformers import PreTrainedModel, ViTConfig
except ImportError as exc:
    raise ImportError(
        "Stage 02 requires Hugging Face transformers for the existing ViTDet3D detector. "
        "Install the repository-compatible transformers version in the active environment, then retry. "
        "This stage runs on CPU with --device cpu and does not require a GPU."
    ) from exc

try:
    from transformers.utils import ModelOutput  # type: ignore[attr-defined, import-not-found]
except ImportError:  # pragma: no cover - older/newer transformers layout
    from transformers.utils.generic import ModelOutput

try:
    from transformers.activations import ACT2FN
except ImportError:
    ACT2FN = {"gelu": nn.GELU()}


COORDINATE_ORDER = str(get_required(STAGE02_SCHEMA, "coordinate_order"))
CT_SPACE = str(get_required(SCHEMA, "coordinate_format.ct_frame_space"))
PATCH_SPACE = str(get_required(STAGE02_SCHEMA, "patch_space"))

# Heuristic anatomical-localization thresholds. These are intentionally
# separate from detector thresholds: they describe *where* a proposal lies
# using the Stage 01 masks and DICOM geometry; they do not alter detection/NMS.
ANATOMY_WALL_DISTANCE_MM = float(get_required(STAGE02_LOCALIZATION, "wall_distance_mm"))
ANATOMY_AIRWAY_PROXIMITY_MM = float(get_required(STAGE02_LOCALIZATION, "airway_proximity_mm"))
ANATOMY_LATERAL_MIDLINE_TOLERANCE_MM = float(get_required(STAGE02_LOCALIZATION, "midline_tolerance_mm"))
ANATOMY_AIRWAY_CENTRAL_RADIUS_MM = float(get_required(STAGE02_LOCALIZATION, "airway_central_radius_mm"))
ANATOMY_TRACHEA_CENTRAL_RADIUS_MM = float(get_required(STAGE02_LOCALIZATION, "trachea_central_radius_mm"))
ANATOMY_TRACHEA_Z_FRACTION = float(get_required(STAGE02_LOCALIZATION, "trachea_z_fraction"))

VIT_WINDOW = np.asarray(STAGE02_DETECTOR["input_window_zyx_voxels"], dtype=np.int32)
VIT_STRIDE = np.asarray(STAGE02_DETECTOR["sliding_stride_zyx_voxels"], dtype=np.int32)
VIT_MEAN = float(get_required(STAGE02_DETECTOR, "normalization.mean"))
VIT_STD = float(get_required(STAGE02_DETECTOR, "normalization.std"))

DEFAULT_VIT_LOGIT_THRESHOLD = float(STAGE02_SCHEMA["vit_logit_threshold_default"])
DEFAULT_NMS_DISTANCE_MM = float(STAGE02_SCHEMA["nms_distance_mm_default"])
LOG_DIAMETERS_MM = tuple(float(v) for v in get_required(STAGE02_SCHEMA, "diagnostic_log_detector.diameters_mm", expected_type=list))
DEFAULT_LOG_THRESHOLD = float(STAGE02_SCHEMA["log_threshold_default"])
DEFAULT_LOG_MIN_DISTANCE_MM = float(STAGE02_SCHEMA["log_min_distance_mm_default"])
DEFAULT_DIAGNOSTIC_AGREEMENT_DISTANCE_MM = float(STAGE02_SCHEMA["diagnostic_agreement_distance_mm_default"])

AIR_HU = float(get_required(SCHEMA, str(STAGE02_SCHEMA["patch_padding_hu_ref"])))
# Backward-compatible alias retained by the patch-extraction implementation.
DEFAULT_AIR_HU = AIR_HU
DEFAULT_PATCH_SIZE_VOX = int(STAGE02_SCHEMA["patch_size_vox_default"])
DEFAULT_CONTEXT_MARGIN_MM = float(STAGE02_SCHEMA["context_margin_mm_default"])
DEFAULT_WINDOW_LEVEL = float(STAGE02_SCHEMA["ct_display_window_hu"]["level"])
DEFAULT_WINDOW_WIDTH = float(STAGE02_SCHEMA["ct_display_window_hu"]["width"])

SCRIPT_DIR = IMAGING_DIR
V2_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_ROOT = V2_ROOT / get_required(SCHEMA, "paths.output_root")


def json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path: str | Path, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=json_default, allow_nan=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DetectorCandidate:
    """
    The authoritative detector candidate record.

    Field set matches the architecture's detector candidate schema
    exactly (see LungInsight_Imaging_Pipeline_Architecture.pdf,
    "Recommended record" / "B. ViTDet3D Candidate Record"). This is
    ONLY ever populated from ViTDet3D output -- see module docstring.
    """

    candidate_id: int
    source: str                      # always "ViTDet3D"
    detector_score: float
    coordinate_order: str            # always "ZYX"
    center_zyx: List[float]
    bbox_start_zyx: List[int]
    bbox_end_zyx: List[int]
    bbox_size_zyx_voxels: List[int]
    spacing_zyx_mm: List[float]
    space: str                       # always CT_SPACE
    model: Dict[str, Any]            # {"name", "checkpoint", "model_version"}

    # Extra, non-schema-breaking provenance kept alongside (additive
    # fields; every field the architecture requires is still present
    # and unambiguous). Downstream code that only reads the required
    # keys is unaffected.
    raw_logit: Optional[float] = None
    center_lps_mm: Optional[List[float]] = None
    diameter_mm: Optional[float] = None
    anatomical_location: Optional[Dict[str, Any]] = None


@dataclass
class LogDiagnosticCandidate:
    """
    A LoG (Laplacian-of-Gaussian) detection.

    This is the raw detector record. It retains its research-comparison
    purpose in the audit stream; validated records are also fused into the
    authoritative `candidates.json`.
    """

    diagnostic_id: str
    purpose: str                     # raw detector purpose before fusion
    coordinate_order: str
    center_zyx: List[float]
    center_lps_mm: List[float]
    scale_mm: float
    log_response: float
    space: str


# ============================================================================
# ANATOMICAL LOCALIZATION
# ============================================================================


def _ellipsoid_offsets(radius_mm: float, spacing_zyx_mm: np.ndarray) -> np.ndarray:
    """Integer ZYX offsets inside a physical-radius ellipsoid."""
    radius_vox = np.maximum(
        np.ceil(float(radius_mm) / np.asarray(spacing_zyx_mm, dtype=np.float64)).astype(int),
        1,
    )
    zz, yy, xx = np.meshgrid(
        np.arange(-radius_vox[0], radius_vox[0] + 1),
        np.arange(-radius_vox[1], radius_vox[1] + 1),
        np.arange(-radius_vox[2], radius_vox[2] + 1),
        indexing="ij",
    )
    inside = (
        (zz * spacing_zyx_mm[0]) ** 2
        + (yy * spacing_zyx_mm[1]) ** 2
        + (xx * spacing_zyx_mm[2]) ** 2
        <= float(radius_mm) ** 2
    )
    return np.column_stack((zz[inside], yy[inside], xx[inside])).astype(np.int64)


def _local_distance_to_mask(
    mask: np.ndarray,
    center_zyx: np.ndarray,
    spacing_zyx_mm: np.ndarray,
    distance_to_true: bool,
    radius_mm: float = ANATOMY_WALL_DISTANCE_MM + float(STAGE02_LOCALIZATION["local_distance_search_margin_mm"]),
) -> float | None:
    """Distance from a voxel center to the opposite side of a binary mask.

    The computation is deliberately local to avoid allocating a full-volume
    float64 distance-transform array for large CT studies.
    """
    center = np.rint(center_zyx).astype(int)
    radius_vox = np.maximum(
        np.ceil(radius_mm / np.asarray(spacing_zyx_mm, dtype=np.float64)).astype(int),
        2,
    )
    shape = np.asarray(mask.shape, dtype=int)
    lo = np.maximum(center - radius_vox, 0)
    hi = np.minimum(center + radius_vox + 1, shape)

    roi = mask[
        int(lo[0]):int(hi[0]),
        int(lo[1]):int(hi[1]),
        int(lo[2]):int(hi[2]),
    ]
    local_center = center - lo

    if distance_to_true:
        distances = ndimage.distance_transform_edt(~roi, sampling=tuple(spacing_zyx_mm.tolist()))
    else:
        distances = ndimage.distance_transform_edt(roi, sampling=tuple(spacing_zyx_mm.tolist()))

    if np.any(local_center < 0) or np.any(local_center >= np.asarray(roi.shape)):
        return None
    if distances is None or not isinstance(distances, np.ndarray):
        return None
    index: Any = tuple(int(v) for v in local_center)
    distance_array = np.asarray(distances, dtype=np.float64)
    if not all(0 <= idx < size for idx, size in zip(index, distance_array.shape)):
        return None
    value = float(distance_array[index])
    return value if np.isfinite(value) else None


def _build_airway_core(
    lung_mask: np.ndarray,
    air_mask: np.ndarray,
    spacing_zyx_mm: np.ndarray,
    origin_lps_mm: np.ndarray,
    direction_lps: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a conservative airway-lumen proxy from the existing masks.

    This is not a new segmentation algorithm: it uses the Stage 01 physical
    air mask intersected with the Stage 01 lung mask, then retains only 3-D
    connected air components that reach a central/superior seed region. The
    result is used only to label a detector proposal as airway-related.
    """
    empty = np.zeros_like(lung_mask, dtype=bool)
    if not lung_mask.any():
        return empty, {"available": False, "reason": "empty_lung_mask"}

    centroid_zyx = np.asarray(ndimage.center_of_mass(lung_mask), dtype=np.float64)
    centroid_lps = voxel_to_lps_mm(centroid_zyx, spacing_zyx_mm, origin_lps_mm, direction_lps)

    physical_air = air_mask & lung_mask
    if not physical_air.any():
        return empty, {
            "available": False,
            "reason": "no_physical_air_inside_lung_mask",
            "lung_centroid_zyx": centroid_zyx.tolist(),
            "lung_centroid_lps_mm": centroid_lps.tolist(),
        }

    # Use an index-space central/superior seed rather than materializing an
    # argwhere() table for every physical-air voxel. This keeps memory use
    # bounded for large CT volumes while preserving the intended anatomical
    # prior (central airway + superior extent).
    central_x_radius_vox = max(1, int(np.ceil(
        ANATOMY_AIRWAY_CENTRAL_RADIUS_MM / float(spacing_zyx_mm[2])
    )))
    superior_offset_vox = max(1, int(np.ceil(
        float(STAGE02_LOCALIZATION["airway_superior_offset_mm"]) / float(spacing_zyx_mm[0])
    )))
    z_grid = np.arange(physical_air.shape[0], dtype=np.int64)[:, None, None]
    x_grid = np.arange(physical_air.shape[2], dtype=np.int64)[None, None, :]
    central = np.abs(x_grid - np.rint(centroid_zyx[2]).astype(np.int64)) <= central_x_radius_vox
    superior = z_grid >= (np.rint(centroid_zyx[0]).astype(np.int64) + superior_offset_vox)
    seed_mask = physical_air & central & superior

    structure = np.ones((3, 3, 3), dtype=bool)
    label_result = ndimage.label(physical_air, structure=structure)
    labeled = label_result[0]  # type: ignore[index]
    n = int(label_result[1])  # type: ignore[index]
    if n == 0 or not seed_mask.any():
        return empty, {
            "available": False,
            "reason": "no_central_superior_airway_seed",
            "lung_centroid_zyx": centroid_zyx.tolist(),
            "lung_centroid_lps_mm": centroid_lps.tolist(),
        }

    labels = np.unique(labeled[seed_mask])
    labels = labels[labels != 0]
    if labels.size == 0:
        return empty, {
            "available": False,
            "reason": "seed_not_connected_to_air_component",
            "lung_centroid_zyx": centroid_zyx.tolist(),
            "lung_centroid_lps_mm": centroid_lps.tolist(),
        }

    airway_core = np.isin(labeled, labels)
    airway_coords = np.argwhere(airway_core)
    z_min = int(airway_coords[:, 0].min()) if len(airway_coords) else None
    z_max = int(airway_coords[:, 0].max()) if len(airway_coords) else None

    return airway_core, {
        "available": bool(airway_core.any()),
        "seed_voxel_count": int(seed_mask.sum()),
        "component_count_retained": int(labels.size),
        "airway_voxel_count": int(airway_core.sum()),
        "airway_z_min": z_min,
        "airway_z_max": z_max,
        "lung_centroid_zyx": centroid_zyx.tolist(),
        "lung_centroid_lps_mm": centroid_lps.tolist(),
    }


def classify_anatomical_location(
    center_zyx: np.ndarray,
    lung_mask: np.ndarray,
    air_mask: np.ndarray,
    heart_mask: np.ndarray,
    spacing_zyx_mm: np.ndarray,
    origin_lps_mm: np.ndarray,
    direction_lps: np.ndarray,
    *,
    wall_distance_mm: float = ANATOMY_WALL_DISTANCE_MM,
    airway_proximity_mm: float = ANATOMY_AIRWAY_PROXIMITY_MM,
    airway_core: np.ndarray | None = None,
    airway_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assign a conservative anatomical-location label to one detector center.

    The label is derived from existing Stage 01 masks plus DICOM-LPS
    coordinates. It is a localization heuristic, not a clinical anatomical
    segmentation or atlas-based diagnosis.
    """
    center = np.asarray(center_zyx, dtype=np.float64)
    shape = np.asarray(lung_mask.shape, dtype=int)
    center_idx = np.rint(center).astype(int)
    center_idx = np.clip(center_idx, 0, shape - 1)

    center_lps = voxel_to_lps_mm(center, spacing_zyx_mm, origin_lps_mm, direction_lps)

    if lung_mask.any():
        lung_centroid_zyx = np.asarray(ndimage.center_of_mass(lung_mask), dtype=np.float64)
        lung_centroid_lps = voxel_to_lps_mm(
            lung_centroid_zyx, spacing_zyx_mm, origin_lps_mm, direction_lps
        )
    else:
        lung_centroid_zyx = np.full(3, np.nan)
        lung_centroid_lps = np.full(3, np.nan)

    lateral_offset_mm = None
    laterality = "unknown"
    if np.all(np.isfinite(lung_centroid_lps)):
        lateral_offset_mm = float(center_lps[0] - lung_centroid_lps[0])
        if lateral_offset_mm > ANATOMY_LATERAL_MIDLINE_TOLERANCE_MM:
            laterality = "left"
        elif lateral_offset_mm < -ANATOMY_LATERAL_MIDLINE_TOLERANCE_MM:
            laterality = "right"
        else:
            laterality = "midline"

    # Vertical/anatomic level is defined from the axial lung extent. Because
    # the Stage 01 slices were spatially ordered, this provides a stable
    # superior/mid/inferior descriptor without introducing another model.
    vertical_region = "unknown"
    if lung_mask.any():
        lung_z = np.where(lung_mask.any(axis=(1, 2)))[0]
        if len(lung_z):
            frac = float(
                (center[0] - lung_z.min()) / max(float(lung_z.max() - lung_z.min()), 1.0)
            )
            if frac < 1.0 / 3.0:
                vertical_region = "inferior"
            elif frac > 2.0 / 3.0:
                vertical_region = "superior"
            else:
                vertical_region = "middle"

    center_inside_lung = bool(lung_mask[tuple(center_idx)])
    center_in_heart = bool(heart_mask[tuple(center_idx)])

    wall_distance = None
    if center_inside_lung:
        wall_distance = _local_distance_to_mask(
            lung_mask, center, spacing_zyx_mm, distance_to_true=False,
            radius_mm=max(wall_distance_mm + 2.0, 7.0),
        )
    else:
        distance_to_lung = _local_distance_to_mask(
            lung_mask, center, spacing_zyx_mm, distance_to_true=True,
            radius_mm=max(wall_distance_mm + 2.0, 7.0),
        )
        wall_distance = distance_to_lung

    if airway_core is None or airway_info is None:
        airway_core, airway_info = _build_airway_core(
            lung_mask, air_mask, spacing_zyx_mm, origin_lps_mm, direction_lps
        )

    airway_distance = None
    airway_near = False
    if airway_core.any():
        airway_distance = _local_distance_to_mask(
            airway_core, center, spacing_zyx_mm, distance_to_true=True,
            radius_mm=max(airway_proximity_mm + 2.0, 7.0),
        )
        airway_near = airway_distance is not None and airway_distance <= airway_proximity_mm

    # Conservative airway type inference. A proposal is called tracheal only
    # when it is near the retained central airway proxy, remains close to the
    # midline, and lies in the superior portion of that proxy. Otherwise a
    # central airway-related proposal is reported as bronchial.
    airway_type = None
    airway_related = bool(airway_near)
    if airway_related and airway_info.get("available"):
        z_min = airway_info.get("airway_z_min")
        z_max = airway_info.get("airway_z_max")
        z_fraction = None
        if z_min is not None and z_max is not None and z_max > z_min:
            z_fraction = float((center[0] - z_min) / float(z_max - z_min))
        central_mm = abs(float(lateral_offset_mm)) if lateral_offset_mm is not None else float("inf")
        if (
            central_mm <= ANATOMY_TRACHEA_CENTRAL_RADIUS_MM
            and z_fraction is not None
            and z_fraction >= ANATOMY_TRACHEA_Z_FRACTION
        ):
            airway_type = "trachea"
        else:
            airway_type = "bronchus"

    if airway_type == "trachea":
        primary = "trachea"
        region = "trachea"
        laterality_out = "midline"
        compartment = "airway"
        wall_relationship = "airway_not_lung_wall"
    elif airway_type == "bronchus":
        primary = "bronchus"
        region = (
            "left_bronchus" if laterality == "left"
            else "right_bronchus" if laterality == "right"
            else "bronchus"
        )
        laterality_out = laterality if laterality in ("left", "right") else "midline_or_central"
        compartment = "airway"
        wall_relationship = "airway_not_lung_wall"
    elif center_in_heart and not center_inside_lung:
        primary = "mediastinal"
        region = "heart_or_great_vessels"
        laterality_out = laterality
        compartment = "mediastinum"
        wall_relationship = "not_applicable"
    elif center_inside_lung and wall_distance is not None and wall_distance <= wall_distance_mm:
        primary = "lung_wall"
        region = (
            "left_lung_wall" if laterality == "left"
            else "right_lung_wall" if laterality == "right"
            else "lung_wall"
        )
        laterality_out = laterality
        compartment = "lung_wall"
        wall_relationship = "wall_adjacent"
    elif center_inside_lung:
        primary = "lung_parenchyma"
        region = (
            "left_lung_parenchyma" if laterality == "left"
            else "right_lung_parenchyma" if laterality == "right"
            else "lung_parenchyma"
        )
        laterality_out = laterality
        compartment = "parenchyma"
        wall_relationship = "not_wall_adjacent"
    else:
        primary = "outside_lung"
        region = "outside_lung_mask"
        laterality_out = laterality
        compartment = "outside_lung"
        wall_relationship = (
            "near_lung_boundary"
            if wall_distance is not None and wall_distance <= wall_distance_mm
            else "not_near_lung_boundary"
        )

    return {
        "primary": primary,
        "region": region,
        "laterality": laterality_out,
        "compartment": compartment,
        "vertical_region": vertical_region,
        "wall_relationship": wall_relationship,
        "distance_to_lung_mask_mm": wall_distance,
        "airway_type": airway_type,
        "airway_proximity_mm": airway_distance,
        "center_inside_lung_mask": center_inside_lung,
        "center_inside_heart_mask": center_in_heart,
        "center_lps_mm": center_lps.tolist(),
        "method": "heuristic_from_stage01_masks_and_dicom_lps_geometry",
        "method_version": "anatomical_localization_v1",
        "parameters": {
            "wall_distance_mm": float(wall_distance_mm),
            "airway_proximity_mm": float(airway_proximity_mm),
            "midline_tolerance_mm": float(ANATOMY_LATERAL_MIDLINE_TOLERANCE_MM),
            "airway_central_radius_mm": float(ANATOMY_AIRWAY_CENTRAL_RADIUS_MM),
            "trachea_central_radius_mm": float(ANATOMY_TRACHEA_CENTRAL_RADIUS_MM),
            "trachea_z_fraction": float(ANATOMY_TRACHEA_Z_FRACTION),
        },
        "limitations": (
            "Heuristic localization only. No dedicated anatomical atlas or airway segmentation model is used; "
            "airway labels are inferred from the Stage 01 physical-air mask and central 3-D connectivity, "
            "while lung-wall labels use distance to the Stage 01 lung-mask boundary."
        ),
    }


# ============================================================================
# GENERAL HELPERS
# ============================================================================



# ============================================================================
# STAGE 02 INPUT LOADING
# ============================================================================

def _find_first(obj: Dict[str, Any], keys: Sequence[str]):
    """Find the first matching key in a metadata dictionary."""
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def load_stage01(
    stage01_dir: str | Path,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]
    ]:
    """Load the final Z-cropped Stage 01 CT volume, masks, and coordinate metadata."""
    stage01_dir = Path(stage01_dir)

    volume_path = stage01_dir / STAGE01_FILES[STAGE02_SCHEMA["input_volume"]["filename_key"]]
    lung_mask_path = stage01_dir / STAGE01_FILES["lung_mask.npy"]
    meta_path = stage01_dir / STAGE01_FILES["meta.json"]

    if not volume_path.is_file():
        raise FileNotFoundError(f"Missing Stage 01 cropped CT volume:\n{volume_path}")
    if not lung_mask_path.is_file():
        raise FileNotFoundError(f"Missing Stage 01 lung mask:\n{lung_mask_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing Stage 01 metadata:\n{meta_path}")

    air_mask_path = stage01_dir / STAGE01_FILES["air_mask.npy"]
    heart_mask_path = stage01_dir / STAGE01_FILES["heart_mask.npy"]

    volume = np.asarray(
        np.load(volume_path, allow_pickle=False),
        dtype=np.float32,
    )
    lung_mask = np.asarray(
        np.load(lung_mask_path, allow_pickle=False)
    ).astype(bool)
    metadata = load_json(meta_path)
    check_meta_schema_version(metadata, SCHEMA, context=str(meta_path), require=False)

    # Stage 01 writes both masks. Keep safe fallbacks so older Stage 01
    # outputs remain consumable: physical air is reconstructed from the
    # original -950 HU rule, and a missing heart mask is simply empty.
    if air_mask_path.is_file():
        air_mask = np.asarray(
            np.load(air_mask_path, allow_pickle=False)
        ).astype(bool)
    else:
        air_mask = volume <= STAGE01_FALLBACK_AIR_THRESHOLD_HU

    if heart_mask_path.is_file():
        heart_mask = np.asarray(
            np.load(heart_mask_path, allow_pickle=False)
        ).astype(bool)
    else:
        heart_mask = np.zeros_like(lung_mask, dtype=bool)

    if volume.ndim != 3:
        raise RuntimeError(f"Expected volume shape (Z,Y,X), got {volume.shape}")
    if not np.isfinite(volume).all():
        raise RuntimeError(f"Stage 01 CT volume contains NaN or Inf: {volume_path}")
    if lung_mask.shape != volume.shape:
        raise RuntimeError(
            f"Volume/lung-mask shape mismatch:\n"
            f"volume={volume.shape}\nmask={lung_mask.shape}"
        )
    if air_mask.shape != volume.shape:
        raise RuntimeError(
            f"Volume/air-mask shape mismatch:\n"
            f"volume={volume.shape}\nair_mask={air_mask.shape}"
        )
    if heart_mask.shape != volume.shape:
        raise RuntimeError(
            f"Volume/heart-mask shape mismatch:\n"
            f"volume={volume.shape}\nheart_mask={heart_mask.shape}"
        )

    spacing_value = metadata.get("spacing_zyx_mm")
    if spacing_value is None:
        raise RuntimeError("Stage 01 meta.json is missing spacing_zyx_mm.")

    spacing = np.asarray(spacing_value, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise RuntimeError(f"Invalid spacing_zyx_mm: {spacing_value}")

    spacing_order = str(
        metadata.get("spacing_order", "ZYX")
    ).upper().replace(" ", "").replace(",", "")
    if spacing_order == "XYZ":
        spacing = spacing[::-1]
    elif spacing_order != "ZYX":
        raise RuntimeError(f"Unsupported spacing order: {spacing_order}")

    origin = np.asarray(
        metadata.get("origin_mm", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise RuntimeError("Invalid origin_mm in Stage 01 meta.json.")

    direction_lps = np.asarray(
        metadata.get("direction_lps", np.eye(3).tolist()),
        dtype=np.float64,
    )
    if (
        direction_lps.shape != (3, 3)
        or not np.all(np.isfinite(direction_lps))
    ):
        raise RuntimeError("Invalid direction_lps in Stage 01 meta.json.")

    return (
        volume,
        lung_mask,
        air_mask,
        heart_mask,
        spacing,
        origin,
        direction_lps,
        metadata,
    )

# ============================================================================
# GEOMETRY
# ============================================================================

def voxel_to_lps_mm(
    center_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    origin_lps_mm: np.ndarray,
    direction_lps: np.ndarray,
) -> np.ndarray:
    """Map native ZYX voxel coordinates to physical DICOM-LPS mm."""
    return origin_lps_mm + direction_lps @ (
        np.asarray(center_zyx, dtype=np.float64)
        * np.asarray(spacing_zyx, dtype=np.float64)
    )


def bbox_from_center(
    center_zyx: np.ndarray, diameter_mm: float, spacing_zyx: np.ndarray
) -> np.ndarray:
    """Cubic voxel-space bounding box around a center, given a physical diameter."""
    radius_vox = (diameter_mm / 2.0) / spacing_zyx
    low = center_zyx - radius_vox
    high = center_zyx + radius_vox
    return np.concatenate([low, high])


def euclidean_mm(a_zyx: np.ndarray, b_zyx: np.ndarray, spacing_zyx: np.ndarray) -> float:
    """Physical Euclidean distance between two VOXEL-space centers."""
    delta = (a_zyx - b_zyx) * spacing_zyx
    return float(np.linalg.norm(delta))


def euclidean_mm_physical(a_mm: np.ndarray, b_mm: np.ndarray) -> float:
    """Physical Euclidean distance between two already-physical (mm) centers."""
    return float(np.linalg.norm(np.asarray(a_mm) - np.asarray(b_mm)))


# ============================================================================
# VITDET3D WINDOWING
# ============================================================================

def make_vit_windows(
    volume: np.ndarray, window_size: np.ndarray, stride: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate fixed-size ViTDet3D sliding windows.

    Returns
    -------
    offsets : (N, 3) int, Z,Y,X global voxel offsets into the (possibly
              air-padded) volume
    windows : (N, D, H, W) float32

    Air padding (AIR_HU) is used if the Stage 01 volume is smaller than
    the detector window along any axis.
    """
    shape = np.asarray(volume.shape, dtype=np.int32)
    padded_shape = np.maximum(shape, window_size)

    padded = np.full(tuple(padded_shape), AIR_HU, dtype=np.float32)
    padded[: shape[0], : shape[1], : shape[2]] = volume

    offsets_per_axis = []
    for dim in range(3):
        max_start = padded_shape[dim] - window_size[dim]
        if max_start <= 0:
            offsets = np.asarray([0], dtype=np.int32)
        else:
            offsets = np.arange(0, max_start + 1, stride[dim], dtype=np.int32)
            if offsets[-1] != max_start:
                offsets = np.concatenate([offsets, np.asarray([max_start], dtype=np.int32)])
        offsets_per_axis.append(offsets)

    offsets, windows = [], []
    for z in offsets_per_axis[0]:
        for y in offsets_per_axis[1]:
            for x in offsets_per_axis[2]:
                offsets.append([z, y, x])
                windows.append(
                    padded[
                        z : z + window_size[0],
                        y : y + window_size[1],
                        x : x + window_size[2],
                    ]
                )

    return (
        np.asarray(offsets, dtype=np.int32),
        np.asarray(windows, dtype=np.float32),
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


# ============================================================================
# VITDET3D MODEL DEFINITION
#
# The published checkpoint (rlsn/DeTr4LungNodule on the Hugging Face Hub)
# is NOT a class that ships with the `transformers` library -- "VitDet3D"
# has never existed in `transformers`, and the Hub repo contains no
# modeling_*.py / auto_map, so `trust_remote_code=True` cannot load it
# either. It is a custom architecture, defined in the author's own
# `model.py` (github.com/rlsn/LungNoduleDetection), that happens to
# subclass `transformers.PreTrainedModel` with `config_class = ViTConfig`.
# Subclassing PreTrainedModel is exactly what makes
# `VitDet3D.from_pretrained(repo_id)` work -- but only once this exact
# class is defined locally, the same way the author's own training code
# defined it. This is a straight port of that class, kept 1:1 with the
# upstream implementation so the published pytorch_model.bin state_dict
# loads without key mismatches.
#
# Confirmed against the checkpoint's own config.json:
#     "architectures": ["VitDet3D"], "model_type": "vit"
#     "image_size": [40, 128, 128], "patch_size": [4, 16, 16]
# ("image_size"/"patch_size" here are exactly VIT_WINDOW's [D,H,W] shape.)
#
# There is also no preprocessor_config.json in the repo, so
# AutoImageProcessor was never going to work either -- normalization is
# (and always was) handled manually via VIT_MEAN / VIT_STD above.
# ============================================================================

from transformers import PreTrainedModel, ViTConfig  # noqa: E402

try:
    from transformers.utils import ModelOutput  # type: ignore[attr-defined, import-not-found]  # noqa: E402
except ImportError:  # pragma: no cover - extremely old/new transformers
    from transformers.utils.generic import ModelOutput  # noqa: E402

try:
    from transformers.activations import ACT2FN
except ImportError:  # pragma: no cover - extremely old/new transformers
    ACT2FN = {"gelu": nn.GELU()}


# ----------------------------------------------------------------------
# Self-contained ViT transformer block, ported 1:1 (module names and all)
# from transformers==4.34.0's transformers/models/vit/modeling_vit.py --
# the version this checkpoint was trained and saved with.
#
# We do NOT import ViTEncoder/ViTLayer/ViTPooler from the *installed*
# `transformers` package here. Those internals are not a stable public
# API and have already changed shape upstream: `ViTEncoder` has been
# removed entirely in some releases, and `ViTLayer` collapsed its
# `intermediate` + `output` submodules into a single `mlp` submodule in
# others -- either change would silently break loading this checkpoint's
# pytorch_model.bin (wrong or missing state_dict keys) even though the
# import itself might succeed. Pinning our own copy of these classes,
# with the original 4.34.0 module names, keeps this immune to future
# `transformers` refactors.
# ----------------------------------------------------------------------

class _ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


class _ViTSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class _ViTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = _ViTSelfAttention(config)
        self.output = _ViTSelfOutput(config)

    def forward(self, hidden_states):
        self_output = self.attention(hidden_states)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class _ViTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = (
            ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str)
            else config.hidden_act
        )

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class _ViTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class _ViTLayer(nn.Module):
    """Matches transformers==4.34.0's ViTLayer module names exactly."""

    def __init__(self, config):
        super().__init__()
        self.attention = _ViTAttention(config)
        self.intermediate = _ViTIntermediate(config)
        self.output = _ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        attention_output = self.attention(self.layernorm_before(hidden_states))
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)

        return layer_output


class _ViTEncoder3D(nn.Module):
    """Matches transformers==4.34.0's ViTEncoder module names exactly."""

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([_ViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return (hidden_states,)


class _ViTPooler(nn.Module):
    """Matches transformers==4.34.0's ViTPooler module names exactly."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class _ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=(3, 3, 3),
            stride=stride, padding=1, bias=False,
        )
        self.downsample = downsample
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=(3, 3, 3),
            padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class _CNNFeatureExtractor3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = 64
        self.out_size = [3, 8, 8]
        self.conv1 = nn.Conv3d(
            config.num_channels, self.in_channels,
            kernel_size=7, stride=2, padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)

    def _make_layer(self, num_channels, num_layers, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, num_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(num_channels),
            )
        layers = [_ResBlock3D(self.in_channels, num_channels, stride, downsample)]
        self.in_channels = num_channels
        for _ in range(1, num_layers):
            layers.append(_ResBlock3D(self.in_channels, num_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class _PosEmbedding3D(nn.Module):
    def __init__(self, config, in_channels, in_size):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.seq_len = int(np.prod(in_size))
        self.projection = nn.Linear(in_channels, config.hidden_size)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.seq_len + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = x.flatten(2).transpose(1, 2)
        x = self.projection(x)
        embeddings = torch.cat((cls_tokens, x), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class VitDet3D(PreTrainedModel):
    """
    3D CNN-stem + ViT-encoder detector, ported 1:1 from
    rlsn/LungNoduleDetection's model.py so that the published
    rlsn/DeTr4LungNodule checkpoint loads with matching state_dict keys.
    """

    config_class = ViTConfig

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.cnn = _CNNFeatureExtractor3D(config)
        self.embeddings = _PosEmbedding3D(config, self.cnn.in_channels, self.cnn.out_size)
        self.encoder = _ViTEncoder3D(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = _ViTPooler(config) if add_pooling_layer else None
        self.classification_head = _MLP(config.hidden_size, config.num_labels, 3)
        self.bbox_head = _MLP(config.hidden_size, 6, 3)
        self.config = config

        # The original rlsn/LungNoduleDetection model.py never called this.
        # That was harmless under transformers==4.34.0 (the version this
        # checkpoint was trained with), but current transformers releases
        # populate required from_pretrained() bookkeeping here (tied-weight
        # keys, fp32-keep-modules, etc.) and will crash without it.
        self.post_init()

    def forward(self, pixel_values, labels=None, bbox=None):
        feature_maps = self.cnn(pixel_values)
        embeddings = self.embeddings(feature_maps)
        encoder_outputs = self.encoder(embeddings)
        sequence_output = self.layernorm(encoder_outputs[0])
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        logits = self.classification_head(pooled_output)
        bbox_pred = self.bbox_head(pooled_output)

        loss = None
        if labels is not None and bbox is not None:
            loss_bbox_fn = nn.MSELoss(reduction="none")
            if self.config.num_labels == 1:
                loss = nn.BCEWithLogitsLoss()(logits.view(-1), labels.float())
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)
            mask = labels.unsqueeze(-1).bool()
            loss = loss + (loss_bbox_fn(bbox_pred, bbox) * mask).mean()

        return ModelOutput(
            loss=loss,
            logits=logits,
            bbox=bbox_pred,
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


def load_vitdet3d(model_name: str, device: torch.device, revision: Optional[str] = None):
    """
    Load the published ViTDet3D checkpoint.

    "VitDet3D" is not a class in the `transformers` package -- it is
    defined locally above. We instantiate it from the checkpoint's own
    ViTConfig and let `PreTrainedModel.from_pretrained` (inherited, not
    custom) pull down and load the matching weights.

    Returns (model, resolved_commit_hash_or_None). The commit hash, when
    available, is recorded in each candidate's `model.checkpoint` field
    so a run is exactly reproducible (see module docstring point 3/4).
    """

    print()
    print("Loading ViTDet3D:")
    print(f"  {model_name}" + (f" @ {revision}" if revision else ""))

    kwargs: Dict[str, Any] = {"revision": revision} if revision else {}

    try:
        model = VitDet3D.from_pretrained(model_name, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load ViTDet3D checkpoint '{model_name}'.\n"
            "This requires network access to huggingface.co and a "
            "transformers install new enough to provide ViTEncoder/"
            "ViTPooler (transformers.models.vit.modeling_vit)."
        ) from exc

    model_obj = cast(Any, model)
    model_obj = model_obj.to(device)
    model = cast(Any, model_obj)
    model.eval()

    resolved_commit = getattr(model, "config", None)
    resolved_commit = getattr(resolved_commit, "_commit_hash", None)

    print("ViTDet3D loaded.")

    # No processor: the repo ships no preprocessor_config.json.
    # Normalization is handled explicitly via VIT_MEAN / VIT_STD.
    return model, resolved_commit


# ============================================================================
# VITDET3D INFERENCE
# ============================================================================

@torch.no_grad()
def run_vitdet3d_raw(
    volume_hu: np.ndarray,
    device: torch.device,
    model,
    batch_size: int,
    logit_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Run ViTDet3D over sliding windows and return RAW, per-window
    detections (voxel-space, in the Stage 01 global frame). No
    de-duplication and no schema formatting happens here -- that keeps
    this function a pure, auditable mapping from
    "ViTDet3D input space" -> "Stage 01 global CT coordinates"
    (architecture requirement: "map ViTDet3D input space back to Stage
    02 global CT coordinates without spatial shift").

    Each returned dict has:
        center_zyx : np.ndarray (3,) float, GLOBAL voxel coordinates
        low_zyx    : np.ndarray (3,) float, GLOBAL voxel coordinates
        high_zyx   : np.ndarray (3,) float, GLOBAL voxel coordinates
        score      : float, sigmoid(logit)
        logit      : float, raw detector logit
    """

    offsets, windows = make_vit_windows(volume_hu, VIT_WINDOW, VIT_STRIDE)
    print()
    print(f"ViTDet3D windows: {len(windows)}")

    volume_shape = np.asarray(volume_hu.shape)
    raw_detections: List[Dict[str, Any]] = []

    for start in range(0, len(windows), batch_size):
        end = min(start + batch_size, len(windows))
        batch = windows[start:end]
        batch_offsets = offsets[start:end]

        # The published ViTDet3D model was trained using LUNA16
        # mean/std normalization. This is ONLY for ViTDet3D -- see
        # module docstring.
        batch = (batch - VIT_MEAN) / VIT_STD

        tensor = torch.from_numpy(batch).unsqueeze(1).to(device=device, dtype=torch.float32)
        outputs = model(pixel_values=tensor)

        logits = outputs.logits.detach().cpu().numpy().reshape(-1)
        bbox = outputs.bbox.detach().cpu().numpy()

        for j in range(len(logits)):
            logit = float(logits[j])
            if logit <= logit_threshold:
                continue

            # ViTDet3D's bbox head has no explicit sigmoid; the
            # published bbox targets are normalized to local-window
            # [0, 1] coordinates, so clip defensively.
            local_bbox = np.clip(bbox[j], 0.0, 1.0)
            local_low, local_high = local_bbox[:3], local_bbox[3:]

            # Protect against malformed/reversed predictions -- this is
            # sanitizing the detector's own bbox, not re-deriving a
            # center from it (the center below always comes from this
            # same bbox prediction, never a bounding-box "midpoint of a
            # separately-cropped patch").
            low = np.minimum(local_low, local_high)
            high = np.maximum(local_low, local_high)

            global_low = batch_offsets[j] + low * VIT_WINDOW
            global_high = batch_offsets[j] + high * VIT_WINDOW
            center = (global_low + global_high) / 2.0

            # Reject predictions whose center falls entirely outside
            # the real (unpadded) Stage 01 volume.
            if np.any(center < 0) or np.any(center >= volume_shape):
                continue

            raw_detections.append(
                {
                    "center_zyx": center,
                    "low_zyx": global_low,
                    "high_zyx": global_high,
                    "score": float(sigmoid(np.asarray([logit]))[0]),
                    "logit": logit,
                }
            )

    print(f"ViTDet3D raw detections (pre-NMS): {len(raw_detections)}")
    return raw_detections


def nms_by_distance(
    raw_detections: List[Dict[str, Any]],
    spacing_zyx: np.ndarray,
    distance_mm: float,
) -> List[Dict[str, Any]]:
    """
    Greedy non-maximum suppression over the SAME detector's overlapping
    sliding-window detections, by physical distance between centers.

    This exists only because ViTDet3D's 75%-overlap sliding windows
    will detect the same physical lesion from several neighboring
    windows. Per the architecture's "Proposal Center Authority" rule,
    a kept detection's `center_zyx` is returned EXACTLY as the detector
    produced it -- never averaged, never recomputed from a bounding-box
    midpoint of some other structure. All this function does is decide
    which single raw detection to keep from a cluster and discard the
    (lower-scoring, redundant) rest.
    """

    if not raw_detections:
        return []

    ordered = sorted(raw_detections, key=lambda d: d["score"], reverse=True)
    kept: List[Dict[str, Any]] = []

    for det in ordered:
        center = det["center_zyx"]
        suppressed = False
        for kept_det in kept:
            if euclidean_mm(center, kept_det["center_zyx"], spacing_zyx) <= distance_mm:
                suppressed = True
                break
        if not suppressed:
            kept.append(det)

    print(f"Detections after NMS: {len(kept)}")
    return kept


def build_detector_candidates(
    kept_detections: List[Dict[str, Any]],
    spacing_zyx: np.ndarray,
    origin_lps_mm: np.ndarray,
    direction_lps: np.ndarray,
    model_name: str,
    checkpoint_ref: str,
    model_version: str,
) -> List[DetectorCandidate]:
    """
    Format NMS-kept raw ViTDet3D detections into the architecture's
    candidate schema. `candidate_id` is assigned here,
    sequentially, after NMS -- this is the ONLY place candidate
    identity is created; nothing downstream should re-derive it.
    """

    # Highest score first, for a stable/reproducible, score-ordered
    # candidate_id assignment.
    ordered = sorted(kept_detections, key=lambda d: d["score"], reverse=True)

    candidates: List[DetectorCandidate] = []
    for candidate_id, det in enumerate(ordered):
        center = det["center_zyx"]
        low = det["low_zyx"]
        high = det["high_zyx"]

        bbox_start = np.floor(low).astype(int)
        bbox_end = np.ceil(high).astype(int)
        bbox_size = np.maximum(bbox_end - bbox_start, 1)

        center_mm = voxel_to_lps_mm(center, spacing_zyx, origin_lps_mm, direction_lps)
        side_mm = np.maximum((high - low) * spacing_zyx, 0.1)
        diameter_mm = float(np.cbrt(np.prod(side_mm)))

        candidates.append(
            DetectorCandidate(
                candidate_id=candidate_id,
                source=str(STAGE02_DETECTOR["name"]),
                detector_score=float(det["score"]),
                coordinate_order=COORDINATE_ORDER,
                center_zyx=[float(v) for v in center.tolist()],
                bbox_start_zyx=[int(v) for v in bbox_start.tolist()],
                bbox_end_zyx=[int(v) for v in bbox_end.tolist()],
                bbox_size_zyx_voxels=[int(v) for v in bbox_size.tolist()],
                spacing_zyx_mm=[float(v) for v in spacing_zyx.tolist()],
                space=CT_SPACE,
                model={
                    "name": str(STAGE02_DETECTOR["name"]),
                    "checkpoint": checkpoint_ref,
                    "model_version": model_version,
                },
                raw_logit=float(det["logit"]),
                center_lps_mm=[float(v) for v in center_mm.tolist()],
                diameter_mm=diameter_mm,
            )
        )

    return candidates

def validate_detector_candidate_schema(
    candidate: Dict[str, Any],
    volume_shape_zyx: Sequence[int],
    spacing_zyx_mm: Sequence[float],
) -> None:
    """Validate the detector spatial contract before serialization."""

    required = tuple(STAGE02_DETECTOR["required_candidate_fields"])
    missing = [key for key in required if key not in candidate]
    if missing:
        raise RuntimeError(f"detector candidate missing required fields: {missing}")

    if candidate["coordinate_order"] != COORDINATE_ORDER:
        raise RuntimeError(
            f"candidate coordinate_order must be {COORDINATE_ORDER!r}, "
            f"got {candidate['coordinate_order']!r}"
        )
    if candidate["space"] != CT_SPACE:
        raise RuntimeError(
            f"candidate space must be {CT_SPACE!r}, got {candidate['space']!r}"
        )

    center = np.asarray(candidate["center_zyx"], dtype=np.float64)
    shape = np.asarray(volume_shape_zyx, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise RuntimeError(f"Invalid center_zyx: {candidate['center_zyx']}")
    if np.any(center < 0) or np.any(center >= shape):
        raise RuntimeError(
            f"Candidate center outside Stage 01 volume: "
            f"center={center.tolist()}, shape={shape.tolist()}"
        )

    spacing = np.asarray(candidate["spacing_zyx_mm"], dtype=np.float64)
    expected_spacing = np.asarray(spacing_zyx_mm, dtype=np.float64)
    if spacing.shape != (3,) or not np.allclose(
        spacing, expected_spacing, atol=1e-4, rtol=0.0
    ):
        raise RuntimeError(
            "detector candidate spacing does not match detector volume spacing."
        )


# ============================================================================
# LOG DETECTOR (candidate source with optional audit output)
# ============================================================================

def normalize_for_log(volume_hu: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
    """
    Normalize HU values for the LoG response calculation.

    This normalization is used only by the LoG detector candidate generator.
    It is never written into Stage 02 classifier patches, and it never
    touches `candidates.json`.
    """
    valid = volume_hu[lung_mask]
    if valid.size == 0:
        raise RuntimeError("Lung mask contains no voxels.")

    p_low, p_high = np.percentile(valid, [1.0, 99.0])
    if p_high <= p_low:
        p_low, p_high = float(np.min(valid)), float(np.max(valid))
    if p_high <= p_low:
        return np.zeros_like(volume_hu, dtype=np.float32)

    image = (volume_hu - p_low) / (p_high - p_low)
    image = np.clip(image, 0.0, 1.0)
    image[~lung_mask] = 0.0
    return image.astype(np.float32)


def local_maxima(response: np.ndarray, footprint: np.ndarray, threshold: float) -> np.ndarray:
    """Return voxel coordinates of local maxima above threshold."""
    maximum = maximum_filter(response, footprint=footprint, mode="constant", cval=0.0)
    mask = (response == maximum) & (response >= threshold) & np.isfinite(response)
    return np.argwhere(mask)


def run_log_detector(
    volume_hu: np.ndarray,
    lung_mask: np.ndarray,
    spacing_zyx: np.ndarray,
    origin_lps_mm: np.ndarray,
    direction_lps: np.ndarray,
    threshold: float,
    min_distance_mm: float,
) -> List[LogDiagnosticCandidate]:
    """
    Multi-scale Laplacian-of-Gaussian blob detector.

    The raw detector records retain their original detector purpose for audit.
    After validation and fusion, eligible LoG records are represented in the
    authoritative fused candidate set with purpose
    `authoritative_fused_candidate`.
    """

    image = normalize_for_log(volume_hu, lung_mask)
    raw: List[Dict[str, Any]] = []

    print()
    print("Running multi-scale LoG detector:")

    for diameter_mm in LOG_DIAMETERS_MM:
        # For a 3D Gaussian blob, diameter ~= 2 * sqrt(3) * sigma.
        sigma_mm = diameter_mm / (2.0 * math.sqrt(3.0))
        sigma_vox = sigma_mm / spacing_zyx

        response = -gaussian_laplace(image, sigma=tuple(sigma_vox.tolist()), mode="nearest")
        response *= sigma_mm ** 2
        response[~lung_mask] = 0.0

        radius_vox = np.maximum(
            np.ceil(min_distance_mm / spacing_zyx).astype(np.int32), 1
        )
        z = np.arange(-radius_vox[0], radius_vox[0] + 1)
        y = np.arange(-radius_vox[1], radius_vox[1] + 1)
        x = np.arange(-radius_vox[2], radius_vox[2] + 1)
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        footprint = (
            (zz * spacing_zyx[0]) ** 2
            + (yy * spacing_zyx[1]) ** 2
            + (xx * spacing_zyx[2]) ** 2
            <= min_distance_mm ** 2
        )

        coords = local_maxima(response, footprint, threshold)
        print(f"  diameter={diameter_mm:5.1f} mm sigma={sigma_mm:5.2f} mm maxima={len(coords)}")

        for coord in coords:
            center = coord.astype(np.float32)
            raw.append(
                {
                    "center_zyx": center,
                    "score": float(response[tuple(coord)]),
                    "scale_mm": float(diameter_mm),
                }
            )

    print(f"LoG raw detections: {len(raw)}")

    # Same NMS-by-distance approach as ViTDet3D -- keep the strongest
    # raw detection's own center unchanged, discard redundant neighbors.
    kept = nms_by_distance(
        [{"center_zyx": r["center_zyx"], "low_zyx": r["center_zyx"], "high_zyx": r["center_zyx"],
          "score": r["score"], "logit": r["score"]} for r in raw],
        spacing_zyx,
        distance_mm=min_distance_mm,
    )
    # Recover scale_mm for kept detections (matched by identity of the
    # numpy array object, since nms_by_distance only forwards score).
    scale_lookup = {id(r["center_zyx"]): r["scale_mm"] for r in raw}

    diagnostics: List[LogDiagnosticCandidate] = []
    for diagnostic_id, det in enumerate(sorted(kept, key=lambda d: d["score"], reverse=True)):
        center = det["center_zyx"]
        center_mm = voxel_to_lps_mm(center, spacing_zyx, origin_lps_mm, direction_lps)
        diagnostics.append(
            LogDiagnosticCandidate(
                diagnostic_id=f"log_diag_{diagnostic_id:05d}",
                purpose="research_comparison_only",
                coordinate_order=COORDINATE_ORDER,
                center_zyx=[float(v) for v in center.tolist()],
                center_lps_mm=[float(v) for v in center_mm.tolist()],
                scale_mm=float(scale_lookup.get(id(center), float("nan"))),
                log_response=float(det["score"]),
                space=CT_SPACE,
            )
        )

    print(f"LoG candidates after NMS: {len(diagnostics)}")
    return diagnostics


# ============================================================================
# DIAGNOSTIC-ONLY COMPARISON (never fed back into candidates.json)
# ============================================================================

def compute_diagnostic_agreement(
    vit_candidates: List[DetectorCandidate],
    log_candidates: List[LogDiagnosticCandidate],
    agreement_distance_mm: float,
) -> Dict[str, Any]:
    """
    Report-only comparison of ViTDet3D candidates against LoG
    diagnostic detections. This NEVER modifies either candidate list --
    it only annotates, for research purposes, which ViTDet3D candidates
    have a nearby LoG detection and vice versa.
    """

    per_candidate = []
    vit_with_log_support = 0

    for c in vit_candidates:
        c_mm = np.asarray(c.center_lps_mm, dtype=np.float64)
        best_dist = None
        best_log_id = None
        for log_c in log_candidates:
            d = euclidean_mm_physical(c_mm, np.asarray(log_c.center_lps_mm, dtype=np.float64))
            if best_dist is None or d < best_dist:
                best_dist = d
                best_log_id = log_c.diagnostic_id

        has_support = best_dist is not None and best_dist <= agreement_distance_mm
        if has_support:
            vit_with_log_support += 1

        per_candidate.append(
            {
                "candidate_id": c.candidate_id,
                "nearest_log_diagnostic_id": best_log_id,
                "distance_mm": best_dist,
                "log_agrees_within_threshold": bool(has_support),
            }
        )

    log_with_vit_support = 0
    for log_c in log_candidates:
        log_mm = np.asarray(log_c.center_lps_mm, dtype=np.float64)
        nearest = min(
            (euclidean_mm_physical(log_mm, np.asarray(c.center_lps_mm, dtype=np.float64)) for c in vit_candidates),
            default=None,
        )
        if nearest is not None and nearest <= agreement_distance_mm:
            log_with_vit_support += 1

    return {
        "purpose": "research_comparison_only -- does NOT affect candidates.json",
        "agreement_distance_mm": agreement_distance_mm,
        "num_vitdet3d_candidates": len(vit_candidates),
        "num_log_diagnostic_candidates": len(log_candidates),
        "vitdet3d_candidates_with_log_support": vit_with_log_support,
        "log_candidates_with_vitdet3d_support": log_with_vit_support,
        "per_candidate": per_candidate,
    }


# ============================================================================
# OUTPUT
# ============================================================================

def write_candidates_csv(path: str, candidates: List[DetectorCandidate]):
    """Flat, human-readable mirror of the authoritative fused candidates."""

    fields = [
        "candidate_id", "source", "detector_score", "coordinate_order",
        "center_z", "center_y", "center_x",
        "bbox_start_z", "bbox_start_y", "bbox_start_x",
        "bbox_end_z", "bbox_end_y", "bbox_end_x",
        "bbox_size_z", "bbox_size_y", "bbox_size_x",
        "spacing_z_mm", "spacing_y_mm", "spacing_x_mm",
        "space", "model_name", "model_checkpoint", "model_version",
        "diameter_mm", "anatomical_location", "anatomical_region",
        "anatomical_laterality", "anatomical_compartment",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in candidates:
            writer.writerow(
                {
                    "candidate_id": c.candidate_id,
                    "source": c.source,
                    "detector_score": c.detector_score,
                    "coordinate_order": c.coordinate_order,
                    "center_z": c.center_zyx[0], "center_y": c.center_zyx[1], "center_x": c.center_zyx[2],
                    "bbox_start_z": c.bbox_start_zyx[0], "bbox_start_y": c.bbox_start_zyx[1], "bbox_start_x": c.bbox_start_zyx[2],
                    "bbox_end_z": c.bbox_end_zyx[0], "bbox_end_y": c.bbox_end_zyx[1], "bbox_end_x": c.bbox_end_zyx[2],
                    "bbox_size_z": c.bbox_size_zyx_voxels[0], "bbox_size_y": c.bbox_size_zyx_voxels[1], "bbox_size_x": c.bbox_size_zyx_voxels[2],
                    "spacing_z_mm": c.spacing_zyx_mm[0], "spacing_y_mm": c.spacing_zyx_mm[1], "spacing_x_mm": c.spacing_zyx_mm[2],
                    "space": c.space,
                    "model_name": c.model.get("name"),
                    "model_checkpoint": c.model.get("checkpoint"),
                    "model_version": c.model.get("model_version"),
                    "diameter_mm": c.diameter_mm,
                    "anatomical_location": (
                        c.anatomical_location.get("primary")
                        if c.anatomical_location else None
                    ),
                    "anatomical_region": (
                        c.anatomical_location.get("region")
                        if c.anatomical_location else None
                    ),
                    "anatomical_laterality": (
                        c.anatomical_location.get("laterality")
                        if c.anatomical_location else None
                    ),
                    "anatomical_compartment": (
                        c.anatomical_location.get("compartment")
                        if c.anatomical_location else None
                    ),
                }
            )


def write_fused_candidates_csv(path: str | Path, candidates: Sequence[dict[str, Any]]) -> None:
    """Write accepted fused records without flattening detector scores together."""
    fields = (
        "candidate_id", "stable_nodule_id", "source", "source_members",
        "vitdet_score", "log_score", "center_z", "center_y", "center_x",
        "center_lps_x_mm", "center_lps_y_mm", "center_lps_z_mm",
        "bbox_start_z", "bbox_start_y", "bbox_start_x",
        "bbox_end_z", "bbox_end_y", "bbox_end_x",
        "space", "anatomical_region",
    )
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            center = candidate["center_zyx"]
            center_lps = candidate.get("center_lps_mm") or [None, None, None]
            start = candidate["bbox_start_zyx"]
            end = candidate["bbox_end_zyx"]
            anatomy = candidate.get("anatomical_location") or {}
            writer.writerow({
                "candidate_id": candidate["candidate_id"],
                "stable_nodule_id": candidate["stable_nodule_id"],
                "source": candidate["source"],
                "source_members": json.dumps(candidate["source_members"], sort_keys=True),
                "vitdet_score": candidate.get("vitdet_score"),
                "log_score": candidate.get("log_score"),
                "center_z": center[0], "center_y": center[1], "center_x": center[2],
                "center_lps_x_mm": center_lps[0],
                "center_lps_y_mm": center_lps[1],
                "center_lps_z_mm": center_lps[2],
                "bbox_start_z": start[0], "bbox_start_y": start[1], "bbox_start_x": start[2],
                "bbox_end_z": end[0], "bbox_end_y": end[1], "bbox_end_x": end[2],
                "space": candidate.get("space"),
                "anatomical_region": anatomy.get("region"),
            })



def as_zyx(value: Any, name: str, dtype=float) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.shape != (3,):
        raise ValueError(
            f"{name} must contain exactly 3 values in ZYX order; got {value}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values: {value}")
    return arr


def voxel_to_world_mm(
    voxel_zyx: np.ndarray,
    spacing_zyx_mm: np.ndarray,
    origin_lps_mm: np.ndarray,
    direction_lps: np.ndarray,
) -> np.ndarray:
    """Map native ZYX voxel coordinates to physical DICOM-LPS mm."""
    return origin_lps_mm + direction_lps @ (
        np.asarray(voxel_zyx, dtype=np.float64)
        * np.asarray(spacing_zyx_mm, dtype=np.float64)
    )


def window_to_uint8(
    volume_hu: np.ndarray,
    level: float = DEFAULT_WINDOW_LEVEL,
    width: float = DEFAULT_WINDOW_WIDTH,
) -> np.ndarray:
    if width <= 0:
        raise ValueError("Window width must be > 0")
    low = level - width / 2.0
    high = level + width / 2.0
    scaled = (np.asarray(volume_hu, dtype=np.float32) - low) / (high - low)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def load_candidates(
    candidates_path: Path,
    shape: tuple[int, int, int],
    spacing: np.ndarray,
) -> list[dict[str, Any]]:
    if not candidates_path.is_file():
        raise FileNotFoundError(
            f"Missing detector candidates: {candidates_path}"
        )

    data = load_json(candidates_path)

    if isinstance(data, dict):
        for key in ("candidates", "detector_candidates", "instances"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(
            "candidates.json must contain a list or a dict with a candidate list"
        )

    result: list[dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        required = (
            "candidate_id",
            "center_zyx",
            "bbox_start_zyx",
            "bbox_end_zyx",
        )
        missing = [k for k in required if k not in item]

        if missing:
            print(
                f"WARNING: skipping candidate with missing fields: {missing}"
            )
            continue

        try:
            center = as_zyx(
                item["center_zyx"],
                "center_zyx",
                float,
            )
            bbox_start = as_zyx(
                item["bbox_start_zyx"],
                "bbox_start_zyx",
                float,
            )
            bbox_end = as_zyx(
                item["bbox_end_zyx"],
                "bbox_end_zyx",
                float,
            )

            coordinate_order = str(
                item.get("coordinate_order", COORDINATE_ORDER)
            ).upper()

            if coordinate_order != COORDINATE_ORDER:
                raise ValueError("coordinate_order must be ZYX")

            if item.get("space", CT_SPACE) != CT_SPACE:
                raise ValueError(
                    f"space must be {CT_SPACE}"
                )

            if np.any(bbox_end < bbox_start):
                raise ValueError("bbox_end_zyx < bbox_start_zyx")

            if np.any(center < 0) or np.any(
                center >= np.asarray(shape, dtype=float)
            ):
                raise ValueError(
                    f"center {center.tolist()} outside volume {shape}"
                )

            if item.get("spacing_zyx_mm") is not None:
                candidate_spacing = as_zyx(
                    item["spacing_zyx_mm"],
                    "spacing_zyx_mm",
                    float,
                )
                if not np.allclose(
                    candidate_spacing,
                    spacing,
                    atol=1e-4,
                    rtol=0,
                ):
                    raise ValueError(
                        "candidate spacing does not match Stage 02 spacing"
                    )

            result.append(item)

        except (ValueError, TypeError) as exc:
            print(
                f"WARNING: skipping candidate "
                f"{item.get('candidate_id', '<unknown>')}: {exc}"
            )

    return result


def choose_patch_size_vox(
    candidate: dict[str, Any],
    spacing_zyx_mm: np.ndarray,
    fixed_size_vox: int,
    context_margin_mm: float,
    adaptive: bool,
) -> int:
    if fixed_size_vox <= 0:
        raise ValueError("--patch-size-vox must be > 0")

    if context_margin_mm < 0:
        raise ValueError("--context-margin-mm must be >= 0")

    if not adaptive:
        return int(fixed_size_vox)

    diameter = candidate.get("diameter_mm")

    if diameter is None:
        bbox_start = as_zyx(
            candidate["bbox_start_zyx"],
            "bbox_start_zyx",
            float,
        )
        bbox_end = as_zyx(
            candidate["bbox_end_zyx"],
            "bbox_end_zyx",
            float,
        )

        bbox_size_mm = (
            np.maximum(bbox_end - bbox_start, 0.0)
            * spacing_zyx_mm
        )

        diameter = float(np.max(bbox_size_mm))

    try:
        diameter = float(diameter)
    except (TypeError, ValueError):
        diameter = 0.0

    if not np.isfinite(diameter) or diameter <= 0:
        diameter = float(np.max(spacing_zyx_mm))

    requested_physical_side_mm = (
        diameter + 2.0 * context_margin_mm
    )

    # A cubic voxel footprint is used. The smallest voxel spacing guarantees
    # that the requested physical footprint fits along every axis.
    side = int(
        math.ceil(
            requested_physical_side_mm
            / float(np.min(spacing_zyx_mm))
        )
    )

    side = max(8, side)

    if side % 2:
        side += 1

    return side


def compute_crop_bounds(
    center_zyx: np.ndarray,
    patch_size_vox: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return integer [start,end) Stage-02 crop bounds.

    The crop center is the nearest integer voxel to the authoritative
    Stage-04 center. No candidate coordinate itself is changed.
    """
    center_index = np.rint(center_zyx).astype(np.int64)

    half = patch_size_vox // 2

    crop_start = center_index - half
    crop_end = crop_start + int(patch_size_vox)

    return crop_start, crop_end


def extract_padded_patch(
    volume_hu: np.ndarray,
    crop_start_zyx: np.ndarray,
    crop_end_zyx: np.ndarray,
    pad_value_hu: float = DEFAULT_AIR_HU,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract an [start,end) ZYX crop.

    Returns:
        patch:
            float32 patch, always exactly the requested shape.
        valid_mask:
            True where the voxel came from the native Stage-02 volume.
            False where air padding was introduced.
        source_start:
            clipped source start in Stage-02 coordinates.
        source_end:
            clipped source end in Stage-02 coordinates.
    """
    shape = np.asarray(volume_hu.shape, dtype=np.int64)

    start = np.asarray(crop_start_zyx, dtype=np.int64)
    end = np.asarray(crop_end_zyx, dtype=np.int64)

    patch_shape = end - start

    if np.any(patch_shape <= 0):
        raise ValueError(
            f"Invalid crop bounds: {start.tolist()} -> {end.tolist()}"
        )

    patch = np.full(
        tuple(patch_shape.tolist()),
        pad_value_hu,
        dtype=np.float32,
    )

    valid_mask = np.zeros(
        tuple(patch_shape.tolist()),
        dtype=bool,
    )

    source_start = np.maximum(start, 0)
    source_end = np.minimum(end, shape)

    if np.any(source_end <= source_start):
        return patch, valid_mask, source_start, source_end

    destination_start = source_start - start
    destination_end = (
        destination_start
        + (source_end - source_start)
    )

    source_slices = tuple(
        slice(int(source_start[i]), int(source_end[i]))
        for i in range(3)
    )

    destination_slices = tuple(
        slice(int(destination_start[i]), int(destination_end[i]))
        for i in range(3)
    )

    patch[destination_slices] = volume_hu[source_slices]
    valid_mask[destination_slices] = True

    return patch, valid_mask, source_start, source_end


def bbox_in_patch(
    bbox_start_zyx: np.ndarray,
    bbox_end_zyx: np.ndarray,
    crop_start_zyx: np.ndarray,
    patch_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Stage-04 bbox into patch-local ZYX coordinates.

    This only changes coordinate origin for the patch. It does not modify
    the authoritative candidate bbox.
    """
    local_start = bbox_start_zyx - crop_start_zyx
    local_end = bbox_end_zyx - crop_start_zyx

    max_coord = np.asarray(
        patch_shape,
        dtype=np.float64,
    )

    local_start = np.clip(
        local_start,
        0.0,
        max_coord,
    )

    local_end = np.clip(
        local_end,
        0.0,
        max_coord,
    )

    return local_start, local_end


def safe_stats(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "mean_hu": None,
            "std_hu": None,
            "min_hu": None,
            "max_hu": None,
            "median_hu": None,
            "voxel_count": 0,
        }

    return {
        "mean_hu": float(np.mean(values)),
        "std_hu": float(np.std(values)),
        "min_hu": float(np.min(values)),
        "max_hu": float(np.max(values)),
        "median_hu": float(np.median(values)),
        "voxel_count": int(values.size),
    }


def compute_patch_intensity_stats(
    patch: np.ndarray,
    valid_mask: np.ndarray,
    bbox_start_local: np.ndarray,
    bbox_end_local: np.ndarray,
) -> dict[str, Any]:
    """
    Characterize detector-bbox voxels versus surrounding native CT context.

    Out-of-volume padding is excluded from both groups using valid_mask.
    Native CT voxels, including native lung-air voxels, remain eligible context.
    """
    shape = np.asarray(patch.shape, dtype=np.int64)

    start = np.floor(bbox_start_local).astype(np.int64)
    end = np.ceil(bbox_end_local).astype(np.int64)

    start = np.clip(start, 0, shape - 1)
    end = np.maximum(end, start + 1)
    end = np.minimum(end, shape)

    bbox_slices = tuple(
        slice(int(start[i]), int(end[i]))
        for i in range(3)
    )

    nodule_mask = np.zeros(
        patch.shape,
        dtype=bool,
    )
    nodule_mask[bbox_slices] = True

    nodule_values = patch[nodule_mask & valid_mask]
    context_values = patch[(~nodule_mask) & valid_mask]

    nodule_stats = safe_stats(nodule_values)
    context_stats = safe_stats(context_values)

    valid_voxels = int(np.count_nonzero(valid_mask))
    total_voxels = int(valid_mask.size)
    padding_voxels = total_voxels - valid_voxels

    return {
        "nodule_bbox": nodule_stats,
        "surrounding_context": context_stats,
        "valid_native_voxel_count": valid_voxels,
        "padding_voxel_count": padding_voxels,
        "padding_fraction": (
            float(padding_voxels / total_voxels)
            if total_voxels else 0.0
        ),
    }


def draw_plane_bbox(
    ax,
    start_zyx: np.ndarray,
    end_zyx: np.ndarray,
    plane: str,
    current_index: int,
    *,
    color: str = "#00FF00",
    linestyle: str = "-",
    linewidth: float = 1.8,
    label: str | None = None,
) -> None:
    """
    Project a ZYX bbox onto one of three orthogonal planes.

    axial:
        Y-X at fixed Z
    coronal:
        Z-X at fixed Y
    sagittal:
        Z-Y at fixed X
    """
    start = np.asarray(start_zyx, dtype=float)
    end = np.asarray(end_zyx, dtype=float)

    axis = {
        "axial": 0,
        "coronal": 1,
        "sagittal": 2,
    }[plane]

    if (
        current_index < start[axis]
        or current_index > end[axis]
    ):
        return

    if plane == "axial":
        y0, x0 = start[1], start[2]
        y1, x1 = end[1], end[2]
        xy = (x0, y0)
        width = max(0.5, x1 - x0)
        height = max(0.5, y1 - y0)

    elif plane == "coronal":
        z0, x0 = start[0], start[2]
        z1, x1 = end[0], end[2]
        xy = (x0, z0)
        width = max(0.5, x1 - x0)
        height = max(0.5, z1 - z0)

    else:
        z0, y0 = start[0], start[1]
        z1, y1 = end[0], end[1]
        xy = (y0, z0)
        width = max(0.5, y1 - y0)
        height = max(0.5, z1 - z0)

    rect = Rectangle(
        xy,
        width,
        height,
        fill=False,
        edgecolor=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )

    ax.add_patch(rect)

    if label:
        ax.text(
            xy[0],
            max(0.0, xy[1] - 2.0),
            label,
            color=color,
            fontsize=8,
            fontweight="bold",
            bbox=dict(
                facecolor="black",
                alpha=0.65,
                edgecolor="none",
                pad=1.2,
            ),
        )


def draw_crosshair(
    ax,
    y: float,
    x: float,
) -> None:
    ax.axhline(
        y,
        color="white",
        linestyle=":",
        linewidth=0.8,
        alpha=0.75,
    )
    ax.axvline(
        x,
        color="white",
        linestyle=":",
        linewidth=0.8,
        alpha=0.75,
    )


def setup_image_axis(
    ax,
    image: np.ndarray,
    title: str,
) -> None:
    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(
        title,
        fontsize=10,
        fontweight="bold",
    )
    ax.set_axis_off()


def render_triplanar(
    patch: np.ndarray,
    candidate: dict[str, Any],
    patch_id: int,
    bbox_local_start: np.ndarray,
    bbox_local_end: np.ndarray,
    output_path: Path,
) -> None:
    image = window_to_uint8(patch)

    center = (
        np.rint(
            np.asarray(patch.shape, dtype=float) / 2.0 - 0.5
        )
        .astype(int)
    )

    z, y, x = [int(v) for v in center]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 5.2),
    )

    setup_image_axis(
        axes[0],
        image[z, :, :],
        f"Axial  Z={z}",
    )
    draw_plane_bbox(
        axes[0],
        bbox_local_start,
        bbox_local_end,
        "axial",
        z,
        color="#00FF00",
        label="Detector bbox",
    )
    draw_crosshair(
        axes[0],
        y,
        x,
    )

    setup_image_axis(
        axes[1],
        image[:, y, :],
        f"Coronal  Y={y}",
    )
    draw_plane_bbox(
        axes[1],
        bbox_local_start,
        bbox_local_end,
        "coronal",
        y,
        color="#00FF00",
        label="Detector bbox",
    )
    draw_crosshair(
        axes[1],
        z,
        x,
    )

    setup_image_axis(
        axes[2],
        image[:, :, x],
        f"Sagittal  X={x}",
    )
    draw_plane_bbox(
        axes[2],
        bbox_local_start,
        bbox_local_end,
        "sagittal",
        x,
        color="#00FF00",
        label="Detector bbox",
    )
    draw_crosshair(
        axes[2],
        z,
        y,
    )

    candidate_id = candidate.get("candidate_id")
    score = candidate.get("detector_score")
    logit = candidate.get("raw_logit")
    diameter = candidate.get("diameter_mm")

    banner = (
        "LungInsight Stage 02 — "
        f"Patch {patch_id:03d} | "
        f"Candidate {candidate_id} | "
        f"score={score if score is not None else 'NA'} | "
        f"logit={logit if logit is not None else 'NA'} | "
        f"diameter={diameter if diameter is not None else 'NA'} mm | "
        f"shape={patch.shape[0]}³ ZYX"
    )

    fig.suptitle(
        banner,
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


class PatchViewer:
    def __init__(
        self,
        records: list[dict[str, Any]],
        arrays_dir: Path,
    ) -> None:
        if not records:
            raise ValueError(
                "No extracted patches available for interactive viewing"
            )

        self.records = records
        self.arrays_dir = arrays_dir
        self.patch_index = 0
        self.patch = self._load_patch(0)

        self.fig = plt.figure(
            figsize=(10, 8)
        )

        self.ax = self.fig.add_axes((0.08, 0.20, 0.84, 0.70))

        self.patch_slider_ax = self.fig.add_axes((0.15, 0.10, 0.70, 0.035))

        self.z_slider_ax = self.fig.add_axes((0.15, 0.045, 0.70, 0.035))

        self.patch_slider = Slider(
            self.patch_slider_ax,
            "Patch",
            0,
            max(0, len(records) - 1),
            valinit=0,
            valstep=1,
        )

        self.z_slider = Slider(
            self.z_slider_ax,
            "Z",
            0,
            self.patch.shape[0] - 1,
            valinit=self.patch.shape[0] // 2,
            valstep=1,
        )

        self.patch_slider.on_changed(
            self._on_patch_changed
        )
        self.z_slider.on_changed(
            self._on_z_changed
        )

        self.fig.canvas.mpl_connect(
            "scroll_event",
            self._on_scroll,
        )

        self._draw()

    def _load_patch(
        self,
        index: int,
    ) -> np.ndarray:
        record = self.records[index]

        path = Path(
            record["array_path"]
        )

        if not path.is_file():
            path = (
                self.arrays_dir
                / path.name
            )

        if not path.is_file():
            raise FileNotFoundError(
                f"Patch array not found: {path}"
            )

        patch = np.asarray(
            np.load(
                path,
                allow_pickle=False,
            ),
            dtype=np.float32,
        )

        if patch.ndim != 3:
            raise ValueError(
                f"Patch must be 3-D, got {patch.shape}"
            )

        return patch

    def _on_patch_changed(
        self,
        value: float,
    ) -> None:
        self.patch_index = int(round(value))
        self.patch = self._load_patch(
            self.patch_index
        )

        midpoint = (
            self.patch.shape[0] // 2
        )

        # Slider's public limits can be updated between patches.
        self.z_slider.valmin = 0
        self.z_slider.valmax = (
            self.patch.shape[0] - 1
        )
        self.z_slider.set_val(midpoint)

        self._draw()

    def _on_z_changed(
        self,
        _value: float,
    ) -> None:
        self._draw()

    def _on_scroll(
        self,
        event,
    ) -> None:
        if event.inaxes != self.ax:
            return

        current = int(
            round(self.z_slider.val)
        )

        if event.button == "up":
            new_z = current + 1
        elif event.button == "down":
            new_z = current - 1
        else:
            return

        new_z = max(
            0,
            min(
                new_z,
                self.patch.shape[0] - 1,
            ),
        )

        self.z_slider.set_val(new_z)

    def _draw(self) -> None:
        self.ax.clear()

        record = self.records[
            self.patch_index
        ]

        z = int(
            round(self.z_slider.val)
        )
        z = max(
            0,
            min(
                z,
                self.patch.shape[0] - 1,
            ),
        )

        image = window_to_uint8(
            self.patch[z]
        )

        self.ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
        )

        bbox_start = np.asarray(
            record["bbox_local_start_zyx"],
            dtype=float,
        )
        bbox_end = np.asarray(
            record["bbox_local_end_zyx"],
            dtype=float,
        )

        draw_plane_bbox(
            self.ax,
            bbox_start,
            bbox_end,
            "axial",
            z,
            color="#00FF00",
            label="Detector bbox",
        )

        center = (
            np.asarray(self.patch.shape)
            / 2.0
            - 0.5
        )

        draw_crosshair(
            self.ax,
            center[1],
            center[2],
        )

        self.ax.set_title(
            "LungInsight Stage 02 | "
            f"Patch {record['patch_id']:03d} | "
            f"Candidate {record['candidate_id']} | "
            f"Axial Z={z}/{self.patch.shape[0]-1}",
            fontsize=11,
            fontweight="bold",
        )

        self.ax.set_axis_off()
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def extract_one_patch(
    volume: np.ndarray,
    candidate: dict[str, Any],
    patch_id: int,
    spacing: np.ndarray,
    origin_lps_mm: np.ndarray,
    direction_lps: np.ndarray,
    output_root: Path,
    fixed_size_vox: int,
    context_margin_mm: float,
    adaptive: bool,
    render_visualization: bool = True,
    array_subdir: Path | None = None,
    visualization_subdir: Path | None = None,
) -> dict[str, Any]:
    array_subdir = array_subdir if array_subdir is not None else STAGE02_PATCH_ARRAY_SUBDIR
    visualization_subdir = visualization_subdir if visualization_subdir is not None else STAGE02_PATCH_VIZ_SUBDIR
    center = as_zyx(
        candidate["center_zyx"],
        "center_zyx",
        float,
    )

    bbox_start = as_zyx(
        candidate["bbox_start_zyx"],
        "bbox_start_zyx",
        float,
    )

    bbox_end = as_zyx(
        candidate["bbox_end_zyx"],
        "bbox_end_zyx",
        float,
    )

    patch_size = choose_patch_size_vox(
        candidate,
        spacing,
        fixed_size_vox,
        context_margin_mm,
        adaptive,
    )

    crop_start, crop_end = compute_crop_bounds(
        center,
        patch_size,
    )

    (
        patch,
        valid_mask,
        source_start,
        source_end,
    ) = extract_padded_patch(
        volume,
        crop_start,
        crop_end,
        DEFAULT_AIR_HU,
    )

    expected_shape = (
        patch_size,
        patch_size,
        patch_size,
    )

    if patch.shape != expected_shape:
        raise RuntimeError(
            f"Patch {patch_id} has shape {patch.shape}; "
            f"expected {expected_shape}"
        )

    bbox_local_start, bbox_local_end = bbox_in_patch(
        bbox_start,
        bbox_end,
        crop_start,
        patch.shape,
    )

    intensity_stats = compute_patch_intensity_stats(
        patch,
        valid_mask,
        bbox_local_start,
        bbox_local_end,
    )

    center_world = voxel_to_world_mm(
        center,
        spacing,
        origin_lps_mm,
        direction_lps,
    )

    patch_origin_lps_mm = voxel_to_world_mm(
        crop_start.astype(np.float64),
        spacing,
        origin_lps_mm,
        direction_lps,
    )

    bbox_size_vox = np.maximum(
        bbox_end - bbox_start,
        0.0,
    )

    bbox_size_mm_zyx = (
        bbox_size_vox * spacing
    )

    bbox_volume_mm3 = float(
        np.prod(bbox_size_mm_zyx)
    )

    diameter = candidate.get(
        "diameter_mm"
    )

    if diameter is None:
        diameter = float(
            np.cbrt(
                max(
                    bbox_volume_mm3,
                    0.0,
                )
            )
        )

    array_dir = output_root / array_subdir
    viz_dir = output_root / visualization_subdir

    array_dir.mkdir(parents=True, exist_ok=True)
    if render_visualization:
        viz_dir.mkdir(parents=True, exist_ok=True)

    array_path = (
        array_dir
        / STAGE02_PATCH_ARRAY_PATTERN.format(patch_id=patch_id)
    )

    np.save(
        array_path,
        np.asarray(
            patch,
            dtype=PATCH_ARRAY_DTYPE,
        ),
        allow_pickle=False,
    )

    triplanar_path = None
    if render_visualization:
        triplanar_path = viz_dir / STAGE02_PATCH_VIZ_PATTERN.format(patch_id=patch_id)
        render_triplanar(
            patch,
            candidate,
            patch_id,
            bbox_local_start,
            bbox_local_end,
            triplanar_path,
        )

    patch_center_index = (
        np.rint(center)
        .astype(np.int64)
    )

    patch_center_local = (
        patch_center_index
        - crop_start
    )

    patch_center_lps_mm = voxel_to_world_mm(
        patch_center_local.astype(np.float64),
        spacing,
        patch_origin_lps_mm,
        direction_lps,
    )

    detector_score = candidate.get(
        "detector_score"
    )

    raw_logit = candidate.get(
        "raw_logit"
    )

    return {
        "patch_id": int(patch_id),
        "candidate_id": candidate["candidate_id"],
        "stable_nodule_id": candidate.get("stable_nodule_id", candidate.get("nodule_id")),
        "nodule_id": candidate.get("stable_nodule_id", candidate.get("nodule_id")),
        "source": candidate.get("source", candidate.get("detection_source", "ViTDet3D")),
        "source_members": candidate.get("source_members", []),
        "vitdet_score": candidate.get("vitdet_score"),
        "log_score": candidate.get("log_score"),
        "detection_source": candidate.get("detection_source", candidate.get("source", "ViTDet3D")),
        "diagnostic_id": candidate.get("diagnostic_id"),
        "log_response": candidate.get("log_response"),
        "scale_mm": candidate.get("scale_mm"),

        "coordinate_order": COORDINATE_ORDER,
        "space": PATCH_SPACE,
        "source_space": CT_SPACE,

        "center_zyx": center.tolist(),
        "center_lps_mm": center_world.tolist(),

        "crop_start_zyx": crop_start.tolist(),
        "crop_end_zyx": crop_end.tolist(),
        "crop_end_semantics": "exclusive",

        "source_clip_start_zyx": source_start.tolist(),
        "source_clip_end_zyx": source_end.tolist(),
        "source_clip_end_semantics": "exclusive",

        "patch_center_local_zyx": (
            patch_center_local.tolist()
        ),

        "patch_size_vox_zyx": [
            int(v) for v in patch.shape
        ],

        "patch_size_mode": (
            "adaptive"
            if adaptive
            else "fixed"
        ),

        "context_margin_mm": float(
            context_margin_mm
        ),

        "bbox_start_zyx": (
            bbox_start.tolist()
        ),

        "bbox_end_zyx": (
            bbox_end.tolist()
        ),

        "bbox_size_vox_zyx": (
            bbox_size_vox.tolist()
        ),

        "bbox_size_mm_zyx": (
            bbox_size_mm_zyx.tolist()
        ),

        "diameter_mm": (
            float(diameter)
            if diameter is not None
            else None
        ),

        "nodule_volume_estimate_mm3": (
            bbox_volume_mm3
        ),

        "bbox_local_start_zyx": (
            bbox_local_start.tolist()
        ),

        "bbox_local_end_zyx": (
            bbox_local_end.tolist()
        ),

        "spacing_zyx_mm": (
            spacing.tolist()
        ),

        "origin_lps_mm": origin_lps_mm.tolist(),

        "patch_origin_lps_mm": patch_origin_lps_mm.tolist(),

        "patch_center_lps_mm": patch_center_lps_mm.tolist(),

        "direction_lps": direction_lps.tolist(),

        "detector_score": (
            float(detector_score)
            if detector_score is not None
            else None
        ),

        "raw_logit": (
            float(raw_logit)
            if raw_logit is not None
            else None
        ),

        "anatomical_location": candidate.get(
            "anatomical_location"
        ),

        "intensity_statistics": (
            intensity_stats
        ),

        "array_path": str(array_path),

        "triplanar_path": str(triplanar_path) if triplanar_path is not None else None,

        "source_ct_volume": (
            STAGE01_FILES["volume_hu_cropped.npy"]
        ),

        "source_detector_candidates": candidate.get(
            "source_candidates_file", STAGE02_FILES["candidates_json"]
        ),

        "padding_value_hu": (
            DEFAULT_AIR_HU
        ),

        "ct_window": {
            "level_hu": (
                DEFAULT_WINDOW_LEVEL
            ),
            "width_hu": (
                DEFAULT_WINDOW_WIDTH
            ),
        },
    }



def build_log_patch_candidates(
    log_candidates: list[LogDiagnosticCandidate], spacing_zyx_mm: np.ndarray
) -> list[dict[str, Any]]:
    """Adapt LoG records into patch-cropping records without merging candidate authority."""
    converted: list[dict[str, Any]] = []
    spacing = np.asarray(spacing_zyx_mm, dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"Invalid spacing for LoG patch extraction: {spacing.tolist()}")
    source_file = str(STAGE02_FILES["log_diagnostic"])
    for candidate in log_candidates:
        center = np.asarray(candidate.center_zyx, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(f"Invalid center_zyx for LoG candidate {candidate.diagnostic_id!r}")
        diameter_mm = float(candidate.scale_mm)
        if not np.isfinite(diameter_mm) or diameter_mm <= 0:
            raise ValueError(f"Invalid scale_mm for LoG candidate {candidate.diagnostic_id!r}: {diameter_mm}")
        half_extent_vox = np.maximum((diameter_mm / (2.0 * spacing)), 0.5)
        bbox_start = np.floor(center - half_extent_vox).astype(int)
        bbox_end = np.ceil(center + half_extent_vox).astype(int)
        bbox_end = np.maximum(bbox_end, bbox_start + 1)
        response = float(candidate.log_response)
        if not np.isfinite(response):
            raise ValueError(f"Non-finite LoG response for candidate {candidate.diagnostic_id!r}")
        converted.append({
            "candidate_id": str(candidate.diagnostic_id),
            "diagnostic_id": str(candidate.diagnostic_id),
            "detection_source": "LoG",
            "source": "LoG",
            "source_candidates_file": source_file,
            "purpose": str(candidate.purpose),
            "coordinate_order": str(candidate.coordinate_order),
            "center_zyx": center.tolist(),
            "center_lps_mm": list(candidate.center_lps_mm),
            "bbox_start_zyx": bbox_start.tolist(),
            "bbox_end_zyx": bbox_end.tolist(),
            "detector_score": response,
            "log_response": response,
            "scale_mm": diameter_mm,
            "diameter_mm": diameter_mm,
            "anatomical_location": None,
        })
    return converted


def extract_all(
    volume: np.ndarray,
    candidates: list[dict[str, Any]],
    spacing: np.ndarray,
    origin_lps_mm: np.ndarray,
    direction_lps: np.ndarray,
    output_root: Path,
    fixed_size_vox: int,
    context_margin_mm: float,
    adaptive: bool,
    *,
    render_visualization: bool = True,
    progress_label: str = "patch",
    array_subdir: Path | None = None,
    visualization_subdir: Path | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for patch_id, candidate in enumerate(
        candidates
    ):
        record = extract_one_patch(
            volume,
            candidate,
            patch_id,
            spacing,
            origin_lps_mm,
            direction_lps,
            output_root,
            fixed_size_vox,
            context_margin_mm,
            adaptive,
            render_visualization=render_visualization,
            array_subdir=array_subdir,
            visualization_subdir=visualization_subdir,
        )

        records.append(record)
        item_number = patch_id + 1
        should_report = (len(candidates) <= 20) or item_number % 100 == 0 or item_number == len(candidates)
        if should_report:
            print(
                f"Extracted {progress_label}_{patch_id:03d}: "
                f"candidate={candidate['candidate_id']} "
                f"center={record['center_zyx']} "
                f"shape={record['patch_size_vox_zyx']} "
                f"({item_number}/{len(candidates)})",
                flush=True,
            )

    return records



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LungInsight Stage 02: ViTDet3D candidate detection "
            "and 3-D nodule patch extraction."
        )
    )
    parser.add_argument("patient_id", help="Patient identifier, e.g. LIDC-IDRI-0001")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root output directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )

    parser.add_argument(
        "--vitdet-model",
        default=str(STAGE02_DETECTOR["checkpoint_default"]),
        help="Hugging Face ViTDet3D model identifier.",
    )
    parser.add_argument(
        "--vitdet-revision",
        default=None,
        help="Optional pinned Hugging Face revision/commit.",
    )
    parser.add_argument(
        "--model-version",
        default=str(STAGE02_SCHEMA["model_version_default"]),
        help="Model version tag recorded in candidate provenance.",
    )
    parser.add_argument("--device", default=None, help="Device: cuda, cpu, or auto.")
    parser.add_argument("--batch-size", type=int, default=int(STAGE02_SCHEMA["batch_size_default"]))
    parser.add_argument(
        "--vit-logit-threshold",
        type=float,
        default=DEFAULT_VIT_LOGIT_THRESHOLD,
    )
    parser.add_argument(
        "--nms-distance-mm",
        type=float,
        default=DEFAULT_NMS_DISTANCE_MM,
    )

    parser.add_argument(
        "--with-log-diagnostic",
        "--with-log",
        dest="with_log_diagnostic",
        action="store_true",
        default=bool(STAGE02_SCHEMA["default_log_diagnostic_enabled"]),
        help="Run LoG candidate detection and fuse filtered Top-50 proposals (enabled by default).",
    )
    parser.add_argument(
        "--skip-log",
        dest="with_log_diagnostic",
        action="store_false",
    )
    parser.add_argument("--log-threshold", type=float, default=DEFAULT_LOG_THRESHOLD)
    parser.add_argument(
        "--log-min-distance-mm",
        type=float,
        default=DEFAULT_LOG_MIN_DISTANCE_MM,
    )
    log_limit_group = parser.add_mutually_exclusive_group()
    log_limit_group.add_argument(
        "--log-patch-limit",
        type=int,
        default=STAGE02_SCHEMA["diagnostic_log_detector"]["patch_limit_default"],
        help=("Maximum LoG diagnostic patches to extract (default from meta_schema.json; "
              "the strongest post-NMS responses are first)."),
    )
    log_limit_group.add_argument(
        "--all-log-patches",
        dest="all_log_patches",
        action="store_true",
        help="Extract every LoG diagnostic candidate rather than applying the schema's default patch limit.",
    )
    parser.add_argument(
        "--diagnostic-agreement-distance-mm",
        type=float,
        default=DEFAULT_DIAGNOSTIC_AGREEMENT_DISTANCE_MM,
    )

    parser.add_argument(
        "--patch-size-vox",
        type=int,
        default=DEFAULT_PATCH_SIZE_VOX,
        help="Fixed cubic patch side length in voxels. Default: 64.",
    )
    parser.add_argument(
        "--context-margin-mm",
        type=float,
        default=DEFAULT_CONTEXT_MARGIN_MM,
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use diameter_mm + 2*context-margin-mm for patch size.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive patch viewer after extraction.",
    )
    add_verbose_argument(parser)
    return parser


def run_pipeline(args: argparse.Namespace) -> Path:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.nms_distance_mm < 0:
        raise ValueError("--nms-distance-mm must be >= 0")
    if args.patch_size_vox <= 0:
        raise ValueError("--patch-size-vox must be > 0")
    if args.context_margin_mm < 0:
        raise ValueError("--context-margin-mm must be >= 0")
    if args.log_patch_limit is not None and args.log_patch_limit <= 0:
        raise ValueError("--log-patch-limit must be a positive integer when supplied")

    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    patient_root = Path(args.output_root) / args.patient_id
    output_stage = patient_root / STAGE02_PATHS["directory"]
    declared_patch_root = Path(str(STAGE02_PATCH_OUTPUTS["root"]))
    expected_patch_root = Path(STAGE02_PATHS["directory"]) / str(STAGE02_PATHS["patch_directory"])
    if declared_patch_root.as_posix().rstrip("/") != expected_patch_root.as_posix():
        raise RuntimeError(
            "Inconsistent Stage 02 patch output paths in meta_schema.json: "
            f"stage_02_detect_and_patch.patch_outputs.root={declared_patch_root.as_posix()!r}, "
            f"paths.stage_02 implies {expected_patch_root.as_posix()!r}. Correct the schema so downstream stages find the manifest."
        )
    output_stage.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("LUNGINSIGHT STAGE 02")
    print("VITDET3D CANDIDATE DETECTION + NODULE PATCH EXTRACTION")
    print("=" * 78)
    print(f"Patient : {args.patient_id}")
    print(f"Device  : {device}")
    print(f"Input   : {patient_root}")
    print(f"Output  : {output_stage}")
    print()

    (
        volume_hu,
        lung_mask,
        air_mask,
        heart_mask,
        spacing_zyx,
        origin_lps_mm,
        direction_lps,
        stage01_meta,
    ) = load_stage01(patient_root)

    review_attestation = validate_review_attestation(
        patient_root,
        SCHEMA,
        stage01_meta,
        volume_hu,
        lung_mask,
        heart_mask,
    )

    print(f"CT volume: {volume_hu.shape} (Z, Y, X)")
    print(f"Spacing : {spacing_zyx.tolist()} mm")
    print(f"Origin  : {origin_lps_mm.tolist()} LPS mm")
    print(f"Lung voxels: {int(lung_mask.sum()):,}")

    # ------------------------------------------------------------------
    # Authoritative ViTDet3D detector
    # ------------------------------------------------------------------
    model, resolved_commit = load_vitdet3d(
        args.vitdet_model,
        device,
        revision=args.vitdet_revision,
    )
    checkpoint_ref = args.vitdet_model
    if args.vitdet_revision:
        checkpoint_ref += f"@{args.vitdet_revision}"
    elif resolved_commit:
        checkpoint_ref += f"@{resolved_commit}"

    raw_detections = run_vitdet3d_raw(
        volume_hu=volume_hu,
        device=device,
        model=model,
        batch_size=args.batch_size,
        logit_threshold=args.vit_logit_threshold,
    )

    kept_detections = nms_by_distance(
        raw_detections,
        spacing_zyx,
        distance_mm=args.nms_distance_mm,
    )
    vit_candidates = build_detector_candidates(
        kept_detections,
        spacing_zyx=spacing_zyx,
        origin_lps_mm=origin_lps_mm,
        direction_lps=direction_lps,
        model_name=str(STAGE02_DETECTOR["name"]),
        checkpoint_ref=checkpoint_ref,
        model_version=args.model_version,
    )

    candidate_dicts = [asdict(c) for c in vit_candidates]

    # Build the central airway proxy once and reuse it for all detections. This
    # avoids repeated full-volume connected-component labeling for every LoG point.
    airway_core_cache, airway_info_cache = _build_airway_core(
        lung_mask, air_mask, spacing_zyx, origin_lps_mm, direction_lps
    )

    for candidate in candidate_dicts:
        validate_detector_candidate_schema(
            candidate,
            volume_shape_zyx=volume_hu.shape,
            spacing_zyx_mm=tuple(float(v) for v in spacing_zyx),
        )
        candidate["anatomical_location"] = classify_anatomical_location(
            np.asarray(candidate["center_zyx"], dtype=np.float64),
            lung_mask,
            air_mask,
            heart_mask,
            spacing_zyx,
            origin_lps_mm,
            direction_lps,
            wall_distance_mm=ANATOMY_WALL_DISTANCE_MM,
            airway_proximity_mm=ANATOMY_AIRWAY_PROXIMITY_MM,
            airway_core=airway_core_cache,
            airway_info=airway_info_cache,
        )

    # Keep the typed candidate list in sync with the serialized records so
    # patch extraction can propagate exactly the same anatomical label.
    for c, candidate in zip(vit_candidates, candidate_dicts):
        c.anatomical_location = candidate["anatomical_location"]

    raw_authoritative_count = len(candidate_dicts)
    accepted_candidates, rejected_candidates, candidate_filter_report = filter_detections_by_lung_mask(
        candidate_dicts,
        lung_mask,
        heart_mask,
        min_bbox_lung_occupancy=float(STAGE02_ACCEPTANCE_FILTER["minimum_bbox_lung_occupancy"]),
        spacing_zyx_mm=spacing_zyx,
        expected_space=CT_SPACE,
        require_nonempty_heart_mask=bool(STAGE02_ACCEPTANCE_FILTER["require_nonempty_heart_mask"]),
        require_heart_mask_reviewed=bool(STAGE02_ACCEPTANCE_FILTER["require_human_review_attestation"]),
        heart_mask_reviewed=bool(review_attestation["verified"]),
    )
    if candidate_filter_report["gate_status"] != "passed":
        raise RuntimeError(
            "Stage 02 candidate acceptance gate blocked patch extraction: "
            f"{candidate_filter_report['gate_status']}."
        )
    accepted_ids = {str(row["candidate_id"]) for row in accepted_candidates}
    vit_candidates = [
        candidate for candidate in vit_candidates
        if str(candidate.candidate_id) in accepted_ids
    ]
    candidate_dicts = accepted_candidates
    if len(candidate_dicts) + len(rejected_candidates) != raw_authoritative_count:
        raise RuntimeError("Stage 02 candidate acceptance counts do not reconcile.")
    candidate_filter_report.update(
        {
            "meta_schema_version": SCHEMA["schema_version"],
            "stage": STAGE02_SCHEMA["stage"],
            "patient_id": args.patient_id,
            "detector": str(STAGE02_DETECTOR["name"]),
            "review_attestation": review_attestation,
        }
    )
    save_json(output_stage / STAGE02_FILES["candidate_filter_report"], candidate_filter_report)
    save_json(
        output_stage / STAGE02_FILES["rejected_candidates_audit"],
        {
            "meta_schema_version": SCHEMA["schema_version"],
            "patient_id": args.patient_id,
            "role": "audit_only_not_for_patch_extraction_or_inference",
            "rejected_count": len(rejected_candidates),
            "rejected_candidates": rejected_candidates,
        },
    )
    save_json(output_stage / STAGE02_FILES["candidates_json"], candidate_dicts)
    write_candidates_csv(output_stage / STAGE02_FILES["candidates_csv"], vit_candidates)

    # ------------------------------------------------------------------
    # Detect and quality-filter LoG candidates before ranking and fusion.
    # ------------------------------------------------------------------
    log_candidates: List[LogDiagnosticCandidate] = []
    if args.with_log_diagnostic:
        log_candidates = run_log_detector(
            volume_hu=volume_hu,
            lung_mask=lung_mask,
            spacing_zyx=spacing_zyx,
            origin_lps_mm=origin_lps_mm,
            direction_lps=direction_lps,
            threshold=args.log_threshold,
            min_distance_mm=args.log_min_distance_mm,
        )
    else:
        print("LoG candidate detector: SKIPPED")

    raw_log_candidate_dicts = [asdict(candidate) for candidate in log_candidates]
    accepted_log_candidates, rejected_log_candidates, log_filter_report = filter_detections_by_lung_mask(
        raw_log_candidate_dicts,
        lung_mask,
        heart_mask,
        min_bbox_lung_occupancy=float(STAGE02_ACCEPTANCE_FILTER["minimum_bbox_lung_occupancy"]),
        spacing_zyx_mm=spacing_zyx,
        expected_space=CT_SPACE,
        require_nonempty_heart_mask=bool(STAGE02_ACCEPTANCE_FILTER["require_nonempty_heart_mask"]),
        require_heart_mask_reviewed=bool(STAGE02_ACCEPTANCE_FILTER["require_human_review_attestation"]),
        heart_mask_reviewed=bool(review_attestation["verified"]),
    )
    if log_filter_report["gate_status"] != "passed":
        raise RuntimeError(
            "Stage 02 LoG diagnostic acceptance gate blocked patch extraction: "
            f"{log_filter_report['gate_status']}."
        )
    accepted_log_ids = {
        str(row.get("diagnostic_id", row.get("candidate_id")))
        for row in accepted_log_candidates
    }
    log_candidates = [
        candidate for candidate in log_candidates
        if str(candidate.diagnostic_id) in accepted_log_ids
    ]
    if len(accepted_log_candidates) + len(rejected_log_candidates) != len(raw_log_candidate_dicts):
        raise RuntimeError("Stage 02 LoG candidate acceptance counts do not reconcile.")

    patient_id = str(stage01_meta.get("patient_id", args.patient_id))
    study_uid = str(stage01_meta.get("study_instance_uid", ""))
    series_uid = str(stage01_meta.get("series_instance_uid", ""))
    if not study_uid or study_uid == "UNKNOWN" or not series_uid or series_uid == "UNKNOWN":
        raise RuntimeError(
            "Stable nodule identity requires DICOM StudyInstanceUID and SeriesInstanceUID in Stage 01 meta.json. "
            "Rerun Stage 01 with the updated UID-preserving metadata writer; no placeholder UID will be used."
        )

    fused_candidates, fusion_report = merge_detections(
        accepted_log_candidates,
        accepted_candidates,
        spacing_zyx_mm=spacing_zyx,
        expected_space=CT_SPACE,
        patient_id=patient_id,
        study_uid=study_uid,
        series_uid=series_uid,
        origin_lps_mm=origin_lps_mm,
        direction_lps=direction_lps,
    )
    if fusion_report["deduplicated_count"] > len(accepted_candidates) + len(accepted_log_candidates):
        raise RuntimeError("Stage 02 fusion output count exceeds its accepted input union.")
    for candidate in fused_candidates:
        candidate["patient_id"] = patient_id
        candidate["study_uid"] = study_uid
        candidate["series_uid"] = series_uid
        candidate["nodule_id"] = candidate["stable_nodule_id"]
        candidate["purpose"] = "authoritative_fused_candidate"
        candidate["anatomical_location"] = candidate.get("anatomical_location") or classify_anatomical_location(
            np.asarray(candidate["center_zyx"], dtype=np.float64),
            lung_mask,
            air_mask,
            heart_mask,
            spacing_zyx,
            origin_lps_mm,
            direction_lps,
            wall_distance_mm=ANATOMY_WALL_DISTANCE_MM,
            airway_proximity_mm=ANATOMY_AIRWAY_PROXIMITY_MM,
            airway_core=airway_core_cache,
            airway_info=airway_info_cache,
        )
        candidate["spacing_zyx_mm"] = spacing_zyx.tolist()
    candidate_dicts = select_fused_candidates(fused_candidates, limit=50)
    selected_ids = {str(row["stable_nodule_id"]) for row in candidate_dicts}
    for row in fused_candidates:
        row["selection_status"] = "selected" if str(row["stable_nodule_id"]) in selected_ids else "eligible_not_selected"
    candidate_dicts = [row for row in fused_candidates if str(row["stable_nodule_id"]) in selected_ids]
    fusion_report.update({
        "raw_log_count": len(raw_log_candidate_dicts),
        "filtered_log_count": len(accepted_log_candidates),
        "selected_count": len(candidate_dicts),
        "vitdet_accepted_count": len(accepted_candidates),
        "accepted_candidate_count": len(fused_candidates),
        "eligible_fused_count": len(fused_candidates),
        "log_rejection_counts": log_filter_report.get("rejection_counts", {}),
        "vitdet_rejection_counts": candidate_filter_report.get("rejection_counts", {}),
    })
    fusion_report_path = output_stage / "DETECTION_FUSION_AUDIT.md"
    rejection_lines = [
        f"- LoG: `{reason}` — {count}"
        for reason, count in sorted(fusion_report["log_rejection_counts"].items())
    ] + [
        f"- ViTDet3D: `{reason}` — {count}"
        for reason, count in sorted(fusion_report["vitdet_rejection_counts"].items())
    ]
    fusion_report_path.write_text(
        "\n".join([
            "# Detection Fusion Audit",
            "",
            f"- Patient: `{patient_id}`",
            f"- Study UID: `{study_uid}`",
            f"- Series UID: `{series_uid}`",
            f"- ViTDet3D accepted: {fusion_report['vitdet_accepted_count']}",
            f"- LoG raw: {fusion_report['raw_log_count']}",
            f"- LoG filtered: {fusion_report['filtered_log_count']}",
            f"- Eligible fused candidates: {fusion_report['eligible_fused_count']}",
            f"- Selected fused candidates (maximum 50): {fusion_report['selected_count']}",
            f"- Deduplicated accepted candidates: {fusion_report['accepted_candidate_count']}",
            f"- Union before deduplication: {fusion_report['union_count_before_deduplication']}",
            f"- Duplicates removed: {fusion_report['duplicates_removed']}",
            "",
            "## Rejection reasons",
            "",
            *(rejection_lines or ["- None"]),
            "",
            "Detector scores remain separate by source and are never averaged or normalized.",
        ]) + "\n",
        encoding="utf-8",
    )

    candidate_filter_report["fusion"] = fusion_report
    candidate_filter_report["stable_identity"] = {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "id_field": "stable_nodule_id",
    }
    save_json(output_stage / STAGE02_FILES["candidate_filter_report"], candidate_filter_report)
    save_json(output_stage / STAGE02_FILES["candidates_json"], candidate_dicts)
    write_fused_candidates_csv(output_stage / STAGE02_FILES["candidates_csv"], candidate_dicts)

    log_filter_report.update(
        {
            "meta_schema_version": SCHEMA["schema_version"],
            "stage": STAGE02_SCHEMA["stage"],
            "patient_id": args.patient_id,
            "detector": "LoG",
            "policy": "quality_filtered_before_top50_fusion; source manifest retained for audit only",
            "review_attestation": review_attestation,
        }
    )
    save_json(output_stage / STAGE02_FILES["log_candidate_filter_report"], log_filter_report)
    save_json(
        output_stage / STAGE02_FILES["log_rejected_candidates_audit"],
        {
            "meta_schema_version": SCHEMA["schema_version"],
            "patient_id": args.patient_id,
            "role": "audit_only_not_for_patch_extraction_or_inference",
            "rejected_count": len(rejected_log_candidates),
            "rejected_candidates": rejected_log_candidates,
        },
    )
    save_json(
        output_stage / STAGE02_FILES["log_diagnostic"],
        accepted_log_candidates,
    )
    agreement_report = compute_diagnostic_agreement(
        vit_candidates,
        log_candidates,
        agreement_distance_mm=args.diagnostic_agreement_distance_mm,
    )
    save_json(
        output_stage / STAGE02_FILES["diagnostic_agreement"],
        agreement_report,
    )

    # ------------------------------------------------------------------
    # Authoritative candidate patch extraction
    # ------------------------------------------------------------------
    # The schema stores patch root relative to the patient directory.
    patch_root = patient_root / STAGE02_PATCH_OUTPUTS["root"]
    records = extract_all(
        volume_hu,
        candidate_dicts,
        spacing_zyx,
        origin_lps_mm,
        direction_lps,
        patch_root,
        args.patch_size_vox,
        args.context_margin_mm,
        args.adaptive,
    )

    # Resolve both manifest paths before constructing metadata that references
    # them. The LoG extraction records are filled in below, so the authoritative
    # manifest itself is written only after both extraction branches complete.
    patch_manifest_path = patient_root / STAGE02_PATCH_OUTPUTS["manifest"]
    log_patch_manifest_path = patient_root / STAGE02_LOG_PATCH_OUTPUTS["manifest"]

    manifest = {
        "schema_version": STAGE02_SCHEMA["manifest_schema_version"],
        "meta_schema_version": SCHEMA["schema_version"],
        "stage": STAGE02_SCHEMA["stage"],
        "module": "Validated LoG + ViTDet3D Candidate Fusion + 3-D Nodule Patch Extraction",
        "patient_id": args.patient_id,
        "study_uid": study_uid,
        "series_uid": series_uid,
        "candidate_identity_field": "stable_nodule_id",
        "candidate_fusion_audit": str(fusion_report_path),
        "candidate_fusion": fusion_report,
        "coordinate_order": COORDINATE_ORDER,
        "coordinate_system": SCHEMA["coordinate_system"],
        "source_space": CT_SPACE,
        "patch_space": PATCH_SPACE,
        "inputs": {
            "stage01_directory": str(patient_root),
            "ct_volume": STAGE01_FILES["volume_hu_cropped.npy"],
            "lung_mask": STAGE01_FILES["lung_mask.npy"],
            "stage01_metadata": STAGE01_FILES["meta.json"],
            "authoritative_candidates": STAGE02_FILES["candidates_json"],
        },
        "ct_geometry": {
            "shape_zyx": [int(v) for v in volume_hu.shape],
            "spacing_zyx_mm": spacing_zyx.tolist(),
            "origin_lps_mm": origin_lps_mm.tolist(),
            "direction_lps": direction_lps.tolist(),
            "index_to_physical_formula": STAGE02_SCHEMA["output_coordinate_formulas"]["ct_index_to_physical"],
        },
        "detector": {
            "name": str(STAGE02_DETECTOR["name"]),
            "model": args.vitdet_model,
            "checkpoint": checkpoint_ref,
            "model_version": args.model_version,
            "window_zyx": VIT_WINDOW.tolist(),
            "stride_zyx": VIT_STRIDE.tolist(),
            "normalization_mean": VIT_MEAN,
            "normalization_std": VIT_STD,
            "logit_threshold": args.vit_logit_threshold,
            "nms_distance_mm": args.nms_distance_mm,
            "num_raw_detections": len(raw_detections),
            "num_authoritative_candidates": len(vit_candidates),
            "num_fused_accepted_candidates": len(candidate_dicts),
        },
        "diagnostic_log_detector": {
            "enabled": bool(args.with_log_diagnostic),
            "note": "Filtered LoG candidates are fused into the authoritative candidate set. The separate LoG manifest is audit-only and is not another inference input.",
            "diameters_mm": list(LOG_DIAMETERS_MM),
            "threshold": args.log_threshold,
            "min_distance_mm": args.log_min_distance_mm,
            "num_candidates": len(log_candidates),
            "stage03_attribute_inference_enabled": bool(STAGE02_SCHEMA["diagnostic_log_detector"]["stage03_attribute_inference_enabled_by_default"]),
            "separate_patch_manifest": str(log_patch_manifest_path),
            # Updated with the actual extraction count before the manifest is saved.
            "num_log_patches_extracted": 0,
            "candidate_limit": args.log_patch_limit,
        },
        "patch_geometry": {
            "space": PATCH_SPACE,
            "array_order": ["z", "y", "x"],
            "physical_units": "mm",
            "patch_spacing_zyx_mm": spacing_zyx.tolist(),
            "origin_definition": (
                "patch_origin_lps_mm is the physical DICOM-LPS coordinate "
                "of the CENTER of saved patch voxel [0,0,0], including "
                "virtual air-padded coordinates outside the source CT."
            ),
            "index_to_physical_formula": STAGE02_SCHEMA["patch_coordinate_conventions"]["index_to_physical_formula"],
            "spacing_definition": (
                "Patch spacing equals the source CT spacing because "
                "patches are cropped, not resampled."
            ),
            "crop_bounds_semantics": STAGE02_SCHEMA["output_coordinate_formulas"]["crop_bounds_semantics"],
        },
        "anatomical_localization": {
            "enabled": True,
            "method": "heuristic_from_stage01_masks_and_dicom_lps_geometry",
            "wall_distance_mm": ANATOMY_WALL_DISTANCE_MM,
            "airway_proximity_mm": ANATOMY_AIRWAY_PROXIMITY_MM,
            "midline_tolerance_mm": ANATOMY_LATERAL_MIDLINE_TOLERANCE_MM,
            "airway_central_radius_mm": ANATOMY_AIRWAY_CENTRAL_RADIUS_MM,
            "trachea_central_radius_mm": ANATOMY_TRACHEA_CENTRAL_RADIUS_MM,
            "trachea_z_fraction": ANATOMY_TRACHEA_Z_FRACTION,
            "outputs": [
                "candidates.json anatomical_location",
                "candidates.csv anatomical location columns",
                "patches_manifest.json patches[].anatomical_location",
            ],
            "limitations": (
                "Heuristic only; no anatomical atlas or dedicated airway segmentation model is used."
            ),
        },
        "patch_configuration": {
            "patch_size_vox": int(args.patch_size_vox),
            "context_margin_mm": float(args.context_margin_mm),
            "adaptive": bool(args.adaptive),
            "padding_value_hu": AIR_HU,
            "ct_window_level_hu": DEFAULT_WINDOW_LEVEL,
            "ct_window_width_hu": DEFAULT_WINDOW_WIDTH,
        },
        "patch_count": len(records),
        "patches": records,
    }
    # Filtered Top-50 LoG proposals are already in the primary fused manifest.
    # Do not create a second inference stream for duplicate nodule identities.
    selected_log_candidates: list[LogDiagnosticCandidate] = []
    log_patch_candidates: list[dict[str, Any]] = []
    log_patch_root = patient_root / STAGE02_LOG_PATCH_OUTPUTS["root"]
    bytes_per_patch = int(args.patch_size_vox) ** 3 * int(PATCH_ARRAY_DTYPE.itemsize)
    estimated_gib = len(log_patch_candidates) * bytes_per_patch / (1024 ** 3)
    print(
        f"Preparing {len(log_patch_candidates)} LoG diagnostic patches for Stage 03 "
        f"(detected={len(log_candidates)}; estimated array payload={estimated_gib:.2f} GiB; "
        "visualizations disabled by default).",
        flush=True,
    )
    log_records = extract_all(
        volume_hu,
        log_patch_candidates,
        spacing_zyx,
        origin_lps_mm,
        direction_lps,
        log_patch_root,
        args.patch_size_vox,
        args.context_margin_mm,
        args.adaptive,
        render_visualization=bool(STAGE02_LOG_PATCH_OUTPUTS["visualizations_enabled"]),
        progress_label="log_patch",
        array_subdir=Path(STAGE02_LOG_PATCH_OUTPUTS["array_directory"]).relative_to(Path(STAGE02_LOG_PATCH_OUTPUTS["root"])),
        visualization_subdir=Path(STAGE02_LOG_PATCH_OUTPUTS["visualization_directory"]).relative_to(Path(STAGE02_LOG_PATCH_OUTPUTS["root"])),
    )
    log_manifest = {
        "schema_version": STAGE02_SCHEMA["manifest_schema_version"],
        "meta_schema_version": SCHEMA["schema_version"],
        "stage": STAGE02_SCHEMA["stage"],
        "module": "Legacy LoG Diagnostic Patch Manifest (empty; fused candidates are in primary manifest)",
        "purpose": "research_comparison_only",
        "patient_id": args.patient_id,
        "coordinate_order": COORDINATE_ORDER,
        "coordinate_system": SCHEMA["coordinate_system"],
        "source_space": CT_SPACE,
        "patch_space": PATCH_SPACE,
        "source_detector_candidates": STAGE02_FILES["log_diagnostic"],
        "diagnostic_policy": STAGE02_SCHEMA["diagnostic_log_detector"]["stage03_inference_note"],
        "detected_candidate_count": len(log_candidates),
        "selected_candidate_count": len(selected_log_candidates),
        "candidate_limit": args.log_patch_limit,
        "ct_geometry": manifest["ct_geometry"],
        "patch_geometry": manifest["patch_geometry"],
        "patch_configuration": manifest["patch_configuration"],
        "patch_count": len(log_records),
        "patches": log_records,
    }
    log_patch_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(log_patch_manifest_path, log_manifest)

    # The optional legacy diagnostic manifest is deliberately empty to prevent
    # downstream duplicate inference for LoG records already fused above.
    manifest["diagnostic_log_detector"]["num_log_patches_extracted"] = len(log_records)
    patch_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(patch_manifest_path, manifest)

    detector_metadata = {
        "patient_id": args.patient_id,
        "stage": int(STAGE02_SCHEMA["stage"]),
        "schema_version": STAGE02_SCHEMA["detector_metadata_schema_version"],
        "meta_schema_version": SCHEMA["schema_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_order": COORDINATE_ORDER,
        "coordinate_system": SCHEMA["coordinate_system"],
        "space": CT_SPACE,
        "volume_shape_zyx": list(volume_hu.shape),
        "spacing_zyx_mm": spacing_zyx.tolist(),
        "origin_lps_mm": origin_lps_mm.tolist(),
        "direction_lps": direction_lps.tolist(),
        "stage01_crop_offset_zyx": stage01_meta.get("crop_offset_zyx", [0, 0, 0]),
        "stage01_z_crop_range": stage01_meta.get("z_crop_range"),
        "authoritative_detector": {
            "name": str(STAGE02_DETECTOR["name"]),
            "model": args.vitdet_model,
            "checkpoint": checkpoint_ref,
            "model_version": args.model_version,
            "window_zyx": VIT_WINDOW.tolist(),
            "stride_zyx": VIT_STRIDE.tolist(),
            "normalization_mean": VIT_MEAN,
            "normalization_std": VIT_STD,
            "logit_threshold": args.vit_logit_threshold,
            "nms_distance_mm": args.nms_distance_mm,
            "num_raw_detections": len(raw_detections),
            "num_candidates": len(vit_candidates),
        },
        "anatomical_localization": {
            "enabled": True,
            "method": "heuristic_from_stage01_masks_and_dicom_lps_geometry",
            "wall_distance_mm": ANATOMY_WALL_DISTANCE_MM,
            "airway_proximity_mm": ANATOMY_AIRWAY_PROXIMITY_MM,
            "midline_tolerance_mm": ANATOMY_LATERAL_MIDLINE_TOLERANCE_MM,
            "airway_central_radius_mm": ANATOMY_AIRWAY_CENTRAL_RADIUS_MM,
            "trachea_central_radius_mm": ANATOMY_TRACHEA_CENTRAL_RADIUS_MM,
            "trachea_z_fraction": ANATOMY_TRACHEA_Z_FRACTION,
        },
        "patch_extraction": {
            "patch_size_vox": int(args.patch_size_vox),
            "context_margin_mm": float(args.context_margin_mm),
            "adaptive": bool(args.adaptive),
            "padding_value_hu": AIR_HU,
            "patch_space": PATCH_SPACE,
            "num_patches": len(records),
            "manifest": str(patch_manifest_path),
        },
        "output_files": {
            "candidates.json": "AUTHORITATIVE - validated LoG + ViTDet3D union after deduplication.",
            "candidates.csv": "Flat mirror of candidates.json.",
            "detector_metadata.json": "Detector and coordinate provenance.",
            "log_candidates_diagnostic.json": "Diagnostic only.",
            "candidates_diagnostic_agreement.json": "Comparison report only.",
            "patches/patches_manifest.json": "Authoritative ViTDet3D patch geometry and localization.",
            "log_diagnostic_patches/log_patches_manifest.json": "Separate diagnostic LoG patch geometry for Stage 03 inference.",
        },
    }
    save_json(output_stage / STAGE02_FILES["detector_metadata"], detector_metadata)

    print()
    print("=" * 78)
    print("STAGE 02 COMPLETE")
    print("=" * 78)
    print(f"ViTDet3D detector candidates        : {len(vit_candidates)}")
    print(f"LoG detector candidates             : {len(log_candidates)}")
    print(f"Extracted fused authoritative patches: {len(records)}")
    print(f"Extracted LoG diagnostic patches    : {len(log_records)}")
    print(f"Output: {output_stage}")
    print("Authoritative candidate file: candidates.json")
    print("Authoritative patch manifest: patches/patches_manifest.json")
    print(f"LoG diagnostic patch manifest: {log_patch_manifest_path}")

    if args.interactive and records:
        viewer = PatchViewer(records, patch_root / Path(STAGE02_PATCH_OUTPUTS["array_directory"]).name)
        viewer.show()

    return output_stage


def main() -> None:
    args = build_parser().parse_args()
    configure_console(args.verbose)
    if getattr(args, "all_log_patches", False):
        args.log_patch_limit = None
    run_pipeline(args)
    stage_complete(2, "02")


if __name__ == "__main__":
    main()
