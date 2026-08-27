"""
08_full_ct_gradcam.py

LungInsight -- Stage 08
========================

Local-to-Global Grad-CAM Projection and Visualization,
with full-CT interactive viewer + animated GIF export.

Stage 08 is the SOLE projection authority in the LungInsight pipeline.
It takes the local (64, 64, 64) candidate-patch Grad-CAM arrays produced
by Stage 07 and projects them back onto the Stage 02 native CT volume
that Stage 04 originally ran candidate detection against.

Coordinate convention
----------------------
ALL spatial coordinates are Z, Y, X.

Inputs (mandatory)
-------------------
1. Stage 02 CT       : output/<patient>/02/volume_hu.npy, lung_mask.npy, meta.json
2. Stage 05 geometry  : output/<patient>/05_classifier_patches/patches.json
                        (Patch Geometry Record -- one entry per candidate)
3. Stage 07 local CAM : output/<patient>/07_gradcam/candidate_<id>/gradcam/<head>.npy
                        + candidate_<id>/metadata.json (carries the Stage 05
                        geometry forward under metadata["geometry"])

Mapping rules (see LungInsight Imaging Pipeline Architecture spec)
--------------------------------------------------------------------
Physical mapping:

    physical_zyx_mm = patch_center_physical_zyx_mm
                       + (local_zyx - local_patch_center_zyx) * patch_spacing_zyx_mm

Native voxel mapping:

    native_zyx = candidate_center_zyx
                 + (local_zyx - local_patch_center_zyx) * patch_spacing_zyx_mm
                   / native_spacing_zyx_mm

Note this native-voxel formula is deliberately expressed *relative* to
`candidate_center_zyx` (which is already a native Stage 02 voxel index)
rather than by round-tripping through absolute physical millimetres and
dividing by spacing. That absolute-physical route requires subtracting
the Stage 02 physical origin before dividing by spacing -- forgetting the
origin term silently corrupts every projection whenever the CT's physical
origin isn't exactly (0, 0, 0). The relative formula above sidesteps that
failure mode entirely because the origin cancels out algebraically. See
"Audit notes" below for the bug this replaces.

Explicit crop offsets:

    stage01_voxel_zyx = crop_offset_zyx + stage02_voxel_zyx

Stage 02 in this pipeline is not itself a crop of a larger volume (Stage 04
detects candidates directly against it), so this only matters if a future
Stage 02 records a `crop_offset_zyx` in its meta.json. It is applied only
when converting Stage 02 coordinates back to the original Stage 01 frame.

Audit notes (deviations found in the prior script, `08_full_ct_gradcam.py`)
-----------------------------------------------------------------------------
1. **Dropped origin term.** `get_native_crop_geometry()` and the peak-voxel
   calculation both computed `native = physical_center / native_spacing`,
   omitting subtraction of the Stage 02 physical origin. Stage 05 explicitly
   folds the origin into `patch_center_physical_zyx_mm`
   (`origin + candidate_center_zyx * native_spacing_zyx_mm`), so dividing
   that value straight through by spacing reintroduces an origin-sized
   translation error whenever origin != (0, 0, 0). Fixed by adopting the
   spec's relative formula, which never needs the origin.
2. **Wrong source CT volume.** `find_full_volume()` preferred
   `<patient>/01/volume_hu.npy` (the pre-crop/pre-mask Stage 01 volume) over
   Stage 02. Stage 04's `candidate_center_zyx` -- and therefore every
   downstream geometry record -- is only meaningful in the Stage 02 voxel
   grid; projecting onto Stage 01 silently misaligns every heatmap whenever
   Stage 02 crops, resamples, or otherwise re-origins relative to Stage 01.
   Fixed by loading `output/<patient>/02/{volume_hu.npy,lung_mask.npy,meta.json}`
   unconditionally, matching Stage 04/05's own contract.
3. **No explicit crop-offset support.** The architecture requires
    Stage 02 carries crop bounds. The offset is applied only for explicit
    Stage 02-to-Stage 01 provenance conversion, never for Stage 02 indexing.
4. **Output layout / report schema mismatch.** The prior script wrote to
   `08_full_ct_gradcam/heads/<head>/...` with a `stage08_summary.json`.
   The architecture spec mandates `08_visualization/{overlays,projections}/`
   and `08_visualization/report.json` in the schema documented below.
   Fixed by conforming the output tree and report exactly to spec.

Interactive viewer & GIF export
--------------------------------
On top of projection, Stage 08 now also produces, per (candidate, head):

1. A self-contained HTML volumetric viewer
   (`08_visualization/viewer/candidate_<id>_viewer.html`, one per candidate,
   covering all of its projected heads). It renders the *full* Stage 02 CT
   frame (not just the candidate crop) with the projected heatmap overlaid,
   and gives the user:
     - a Z-slice slider that scrubs through the native CT volume, with a
       real-time "Slice z / N" readout,
     - a head selector (dropdown) to switch which classifier head's
       heatmap is shown,
     - a colormap selector (jet / inferno),
     - an alpha/transparency slider.
   This is implemented as plain HTML canvas/`<img>` layers driven by JS, so
   it needs no Jupyter/ipywidgets/Plotly runtime -- it opens in any browser.
   The CT base layer and the colored heatmap layer are pre-rendered as
   separate PNG frames (base64-embedded) so that both the colormap and the
   alpha blend can be changed live, client-side, without recomputation.

2. An animated GIF per (candidate, head)
   (`08_visualization/animations/candidate_<id>_<head>_full_ct.gif`) that
   cycles through every native Z-slice whose projected heatmap has at least
   one voxel above the display threshold, each frame showing the *full* CT
   slice (native Y, X extent) with the heatmap alpha-blended in place.

Both stages share the same full-frame compositing utilities so the GIF and
the viewer show pixel-identical projections of the overlay PNGs already
produced above.

Outputs
-------
output/<patient>/08_visualization/
    overlays/
        candidate_<id>_<head>_overlay.png        (axial/coronal/sagittal slices)
    projections/
        candidate_<id>_<head>_projection.png     (axial/coronal/sagittal MIPs)
    animations/
        candidate_<id>_<head>_full_ct.gif         (full-CT Z-scroll animation)
    viewer/
        candidate_<id>_viewer.html                (interactive full-CT viewer)
    report.json
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import map_coordinates


# ============================================================================
# CONSTANTS
# ============================================================================

HU_MIN = -1000.0
HU_MAX = 400.0

DEFAULT_PATCH_SHAPE_ZYX = (64, 64, 64)
DEFAULT_LOCAL_CENTER_ZYX = np.array([31.5, 31.5, 31.5], dtype=np.float64)

EXPECTED_HEADS = [
    "calcification",
    "lobulation",
    "malignancy",
    "margin",
    "sphericity",
    "spiculation",
    "subtlety",
    "texture",
]

COORDINATE_ORDER = "ZYX"
SOURCE_SPACE = "stage02_native_ct"
PROJECTION_AUTHORITY = "stage08"

DEFAULT_COLORMAPS = ("jet", "inferno")
DEFAULT_VIEWER_MAX_DIM = 320   # downsample cap (px) for embedded viewer frames
DEFAULT_GIF_MAX_FRAMES = 120   # cap animated-GIF frame count for large lesions
DEFAULT_GIF_FPS = 6


def _get_cmap(name: str):
    """Resolve a matplotlib colormap by name across matplotlib versions."""

    try:
        return matplotlib.colormaps[name]
    except Exception:  # noqa: BLE001 -- fall back for older matplotlib
        return plt.get_cmap(name)


# ============================================================================
# SMALL UTILITIES
# ============================================================================


def _zyx(
    mapping: Dict[str, Any],
    keys: Tuple[str, ...],
    default: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pull the first present (3,) ZYX field out of `mapping`."""

    for key in keys:
        if key in mapping and mapping[key] is not None:
            value = np.asarray(mapping[key], dtype=np.float64)
            if value.shape == (3,) and np.all(np.isfinite(value)):
                return value

    if default is not None:
        return np.array(default, dtype=np.float64)

    raise RuntimeError(
        f"None of the expected keys {keys} were found (or valid) in: "
        f"{list(mapping.keys())}"
    )


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return str(value)


def normalize_head_name(stem: str) -> str:
    name = stem.lower().strip()
    for prefix in ("gradcam_", "cam_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


# ============================================================================
# STAGE 02 -- NATIVE CT
# ============================================================================


# ============================================================================
# STAGE 02 -- NATIVE CT / CROPPED FRAME
# ============================================================================

STAGE02_SPACE = "stage02_native_ct"
STAGE01_SPACE = "stage01_original_ct"


def resolve_crop_offset_zyx(meta: Dict[str, Any]) -> np.ndarray:
    """
    Return the offset of Stage 02 voxel [0,0,0] inside the original
    Stage 01 volume.

    IMPORTANT:
        This offset is NOT added when indexing Stage 02 itself.

        Stage 02 array coordinates:
            stage02_zyx

        Original Stage 01 coordinates:
            stage01_zyx = crop_offset_zyx + stage02_zyx
    """

    for key in (
        "crop_offset_zyx",
        "crop_offset_zyx",
        "crop_origin_voxel_zyx",
    ):
        if key not in meta or meta[key] is None:
            continue

        value = np.asarray(meta[key], dtype=np.float64)

        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(
                f"Invalid {key}: expected finite 3-element ZYX vector, "
                f"got {meta[key]!r}"
            )

        return value

    return np.zeros(3, dtype=np.float64)


def load_stage02(stage02_dir: Path) -> Dict[str, Any]:
    """
    Load the exact Stage 02 CT volume used by Stage 04/05.

    Stage 08 projection coordinates are expressed in this array's
    native ZYX voxel frame.

    The Stage 02 -> Stage 01 crop offset is retained separately and
    NEVER silently added to Stage 02 array indices.
    """

    volume_path = stage02_dir / "volume_hu.npy"
    mask_path = stage02_dir / "lung_mask.npy"
    meta_path = stage02_dir / "meta.json"

    for path, label in (
        (volume_path, "Stage 02 CT volume"),
        (mask_path, "Stage 02 lung mask"),
        (meta_path, "Stage 02 metadata"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    ct = np.asarray(
        np.load(volume_path, allow_pickle=False),
        dtype=np.float32,
    )

    lung_mask = np.asarray(
        np.load(mask_path, allow_pickle=False)
    ).astype(bool)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if ct.ndim != 3:
        raise ValueError(
            f"Expected Stage 02 CT shape (Z,Y,X), got {ct.shape}"
        )

    if lung_mask.shape != ct.shape:
        raise ValueError(
            f"Stage 02 lung mask shape {lung_mask.shape} does not "
            f"match CT shape {ct.shape}"
        )

    spacing = _zyx(
        meta,
        (
            "spacing_zyx_mm",
            "spacing_zyx",
            "spacing",
            "voxel_spacing",
        ),
    )

    origin = _zyx(
        meta,
        (
            "origin_zyx_mm",
            "origin_zyx",
            "origin_mm",
            "origin",
        ),
        default=np.zeros(3),
    )

    crop_offset = resolve_crop_offset_zyx(meta)

    return {
        "ct": ct,
        "lung_mask": lung_mask,

        # Stage 02 array geometry.
        "spacing_zyx_mm": spacing,
        "origin_zyx_mm": origin,
        "volume_shape_zyx": tuple(ct.shape),

        # Stage 02 -> original Stage 01 coordinate conversion.
        "crop_offset_zyx": crop_offset,

        "meta": meta,
    }


def stage02_to_stage01_voxel(
    stage02_zyx: np.ndarray,
    crop_offset_zyx: np.ndarray,
) -> np.ndarray:
    """
    Convert Stage 02 voxel coordinates to original Stage 01 coordinates.

        stage01 = crop_offset + stage02
    """

    stage02_zyx = np.asarray(stage02_zyx, dtype=np.float64)
    crop_offset_zyx = np.asarray(crop_offset_zyx, dtype=np.float64)

    if stage02_zyx.shape != (3,):
        raise ValueError(
            f"stage02_zyx must have shape (3,), got {stage02_zyx.shape}"
        )

    if crop_offset_zyx.shape != (3,):
        raise ValueError(
            "crop_offset_zyx must have shape (3,)"
        )

    return stage02_zyx + crop_offset_zyx


def add_stage01_coordinates_to_report(
    report: Dict[str, Any],
    candidate_center_stage02_zyx: np.ndarray,
    crop_offset_zyx: np.ndarray,
    native_min_stage02_zyx: np.ndarray,
    native_max_stage02_zyx: np.ndarray,
) -> Dict[str, Any]:
    """
    Add original Stage 01 coordinates for provenance.

    Projection remains entirely in Stage 02 coordinates.
    """

    candidate_center_stage01 = stage02_to_stage01_voxel(
        candidate_center_stage02_zyx,
        crop_offset_zyx,
    )

    native_min_stage01 = stage02_to_stage01_voxel(
        native_min_stage02_zyx,
        crop_offset_zyx,
    )

    native_max_stage01 = stage02_to_stage01_voxel(
        native_max_stage02_zyx,
        crop_offset_zyx,
    )

    report["coordinate_frames"] = {
        "projection_space": STAGE02_SPACE,
        "original_space": STAGE01_SPACE,

        "candidate_center_stage02_zyx":
            candidate_center_stage02_zyx.tolist(),

        "candidate_center_stage01_zyx":
            candidate_center_stage01.tolist(),

        "projection_bounds_stage02_zyx": {
            "min": native_min_stage02_zyx.tolist(),
            "max": native_max_stage02_zyx.tolist(),
        },

        "projection_bounds_stage01_zyx": {
            "min": native_min_stage01.tolist(),
            "max": native_max_stage01.tolist(),
        },

        "crop_offset_stage02_to_stage01_zyx":
            crop_offset_zyx.tolist(),
    }

    return report

# ============================================================================
# STAGE 05 -- PATCH GEOMETRY RECORD
# ============================================================================


def load_stage05_manifest(stage05_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load the Stage 05 patch geometry manifest (`patches.json`) and index it
    by candidate id, so we have a fallback source of truth for geometry if
    Stage 07's metadata.json is ever incomplete.
    """

    manifest_path = stage05_dir / "patches.json"

    if not manifest_path.is_file():
        return {}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    records = manifest.get("candidates", manifest.get("patches", manifest))
    if isinstance(records, dict):
        records = records.get("records", [])

    indexed: Dict[str, Dict[str, Any]] = {}
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            candidate_id = record.get("candidate_id", record.get("candidate_index"))
            if candidate_id is not None:
                indexed[str(candidate_id)] = record

    return indexed


# ============================================================================
# STAGE 07 -- LOCAL GRAD-CAM DISCOVERY
# ============================================================================


def discover_candidate_directories(stage07_dir: Path) -> List[Path]:
    candidates = [
        path
        for path in sorted(stage07_dir.glob("candidate_*"))
        if path.is_dir() and (path / "metadata.json").is_file()
    ]

    def sort_key(path: Path) -> int:
        digits = "".join(ch for ch in path.name if ch.isdigit())
        return int(digits) if digits else 10**9

    candidates.sort(key=sort_key)
    return candidates


def discover_head_files(candidate_dir: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for root in (candidate_dir / "gradcam", candidate_dir):
        if not root.is_dir():
            continue
        for path in root.glob("*.npy"):
            name = normalize_head_name(path.stem)
            if name in EXPECTED_HEADS:
                result.setdefault(name, path)
    return result


def extract_candidate_id(candidate_dir: Path) -> int:
    digits = "".join(ch for ch in candidate_dir.name if ch.isdigit())
    return int(digits) if digits else -1


# ============================================================================
# GEOMETRY RESOLUTION
# ============================================================================


def _vectors_close(
    a: Any,
    b: Any,
    atol: float = 1e-4,
) -> bool:
    try:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)

        return (
            aa.shape == bb.shape
            and np.allclose(aa, bb, atol=atol, rtol=0.0)
        )
    except Exception:
        return False


def resolve_candidate_geometry(
    metadata: Dict[str, Any],
    stage05_record: Optional[Dict[str, Any]],
    native_spacing_zyx_mm: np.ndarray,
) -> Dict[str, Any]:
    """
    Stage 05 is the authoritative geometry source.

    Stage 07 is only a carrier of that geometry.

    If Stage 07 and Stage 05 disagree on spatially authoritative
    fields, fail loudly instead of allowing Stage 07 metadata to
    move the Grad-CAM.
    """

    if not stage05_record:
        raise RuntimeError(
            "Stage 05 patch geometry record is missing. "
            "Stage 08 refuses to project without authoritative "
            "Stage 05 geometry."
        )

    stage07_geometry = metadata.get("geometry")

    if not isinstance(stage07_geometry, dict):
        raise RuntimeError(
            "Stage 07 metadata.json does not contain a valid "
            "'geometry' object. Stage 08 refuses to infer geometry."
        )

    # ------------------------------------------------------------------
    # Stage 05 is authoritative.
    # ------------------------------------------------------------------

    geometry = dict(stage05_record)

    authoritative_fields = (
        "candidate_id",
        "candidate_center_zyx",
        "coordinate_order",
        "source_space",
        "native_volume_shape_zyx",
        "native_spacing_zyx_mm",
        "local_patch_center_zyx",
        "patch_center_physical_zyx_mm",
        "patch_spacing_zyx_mm",
        "patch_shape_zyx",
    )

    # ------------------------------------------------------------------
    # Validate Stage 07's carried-forward geometry.
    # ------------------------------------------------------------------

    for key in authoritative_fields:
        if key not in stage07_geometry:
            continue

        if key not in stage05_record:
            raise RuntimeError(
                f"Stage 07 contains authoritative field '{key}', "
                "but Stage 05 does not. Refusing ambiguous geometry."
            )

        a = stage07_geometry[key]
        b = stage05_record[key]

        if isinstance(a, (list, tuple, np.ndarray)):
            if not _vectors_close(a, b):
                raise RuntimeError(
                    f"GEOMETRY MISMATCH for candidate "
                    f"{stage05_record.get('candidate_id')}: "
                    f"Stage 07 '{key}'={a} != "
                    f"Stage 05 '{key}'={b}"
                )
        else:
            if a != b:
                raise RuntimeError(
                    f"GEOMETRY MISMATCH for candidate "
                    f"{stage05_record.get('candidate_id')}: "
                    f"Stage 07 '{key}'={a!r} != "
                    f"Stage 05 '{key}'={b!r}"
                )

    # ------------------------------------------------------------------
    # Validate required fields.
    # ------------------------------------------------------------------

    candidate_center_zyx = _zyx(
        geometry,
        ("candidate_center_zyx", "center_zyx"),
    )

    local_patch_center_zyx = _zyx(
        geometry,
        (
            "local_patch_center_zyx",
            "patch_center_local_zyx",
            "local_center_zyx",
        ),
        default=DEFAULT_LOCAL_CENTER_ZYX,
    )

    patch_spacing_zyx_mm = _zyx(
        geometry,
        ("patch_spacing_zyx_mm",),
    )

    recorded_native_spacing = _zyx(
        geometry,
        ("native_spacing_zyx_mm",),
        default=native_spacing_zyx_mm,
    )

    if not np.allclose(
        recorded_native_spacing,
        native_spacing_zyx_mm,
        atol=1e-4,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Stage 05 native spacing does not match Stage 02 spacing:\n"
            f"  Stage 05: {recorded_native_spacing.tolist()}\n"
            f"  Stage 02: {native_spacing_zyx_mm.tolist()}"
        )

    coordinate_order = geometry.get(
        "coordinate_order",
        COORDINATE_ORDER,
    )

    if coordinate_order != "ZYX":
        raise RuntimeError(
            f"Unsupported coordinate order: {coordinate_order!r}. "
            "Stage 08 requires ZYX."
        )

    source_space = geometry.get(
        "source_space",
        geometry.get("space"),
    )

    if source_space != SOURCE_SPACE:
        raise RuntimeError(
            f"Invalid geometry source_space={source_space!r}; "
            f"expected {SOURCE_SPACE!r}"
        )

    geometry["candidate_center_zyx"] = candidate_center_zyx.tolist()
    geometry["local_patch_center_zyx"] = local_patch_center_zyx.tolist()
    geometry["patch_spacing_zyx_mm"] = patch_spacing_zyx_mm.tolist()
    geometry["native_spacing_zyx_mm"] = recorded_native_spacing.tolist()
    geometry["coordinate_order"] = "ZYX"
    geometry["source_space"] = SOURCE_SPACE
    geometry["geometry_authority"] = "stage05"

    return geometry

# ============================================================================
# LOCAL <-> NATIVE PROJECTION (the spec's mapping rules)
# ============================================================================


def local_to_native(
    local_zyx: np.ndarray,
    candidate_center_zyx: np.ndarray,
    local_patch_center_zyx: np.ndarray,
    patch_spacing_zyx_mm: np.ndarray,
    native_spacing_zyx_mm: np.ndarray,
) -> np.ndarray:
    """
    native_zyx = candidate_center_zyx
                 + (local_zyx - local_patch_center_zyx) * patch_spacing_zyx_mm
                   / native_spacing_zyx_mm
    """

    offset_mm = (local_zyx - local_patch_center_zyx) * patch_spacing_zyx_mm
    return candidate_center_zyx + offset_mm / native_spacing_zyx_mm


def local_cam_to_stage02_native(
    cam: np.ndarray,
    candidate_center_zyx: np.ndarray,
    local_patch_center_zyx: np.ndarray,
    patch_spacing_zyx_mm: np.ndarray,
    native_spacing_zyx_mm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project a local 64^3 Grad-CAM into the Stage 02 native CT grid.

    ALL coordinates are Z,Y,X.

    Formula:

        native_zyx =
            candidate_center_zyx
            +
            (local_zyx - local_patch_center_zyx)
            * patch_spacing_zyx_mm
            / native_spacing_zyx_mm

    Returns:
        source_coords_zyx : (3,Z,Y,X)
        native_min_zyx
        native_max_zyx
    """

    cam = np.asarray(cam)

    if cam.ndim != 3:
        raise ValueError(
            f"Grad-CAM must be 3D (Z,Y,X), got {cam.shape}"
        )

    candidate_center_zyx = np.asarray(
        candidate_center_zyx,
        dtype=np.float64,
    )

    local_patch_center_zyx = np.asarray(
        local_patch_center_zyx,
        dtype=np.float64,
    )

    patch_spacing_zyx_mm = np.asarray(
        patch_spacing_zyx_mm,
        dtype=np.float64,
    )

    native_spacing_zyx_mm = np.asarray(
        native_spacing_zyx_mm,
        dtype=np.float64,
    )

    for name, value in (
        ("candidate_center_zyx", candidate_center_zyx),
        ("local_patch_center_zyx", local_patch_center_zyx),
        ("patch_spacing_zyx_mm", patch_spacing_zyx_mm),
        ("native_spacing_zyx_mm", native_spacing_zyx_mm),
    ):
        if value.shape != (3,):
            raise ValueError(
                f"{name} must have shape (3,), got {value.shape}"
            )

    if np.any(native_spacing_zyx_mm <= 0):
        raise ValueError(
            "native_spacing_zyx_mm must be strictly positive"
        )

    if np.any(patch_spacing_zyx_mm <= 0):
        raise ValueError(
            "patch_spacing_zyx_mm must be strictly positive"
        )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # np.indices returns Z,Y,X because the array itself is Z,Y,X.
    # There is deliberately NO transpose here.
    # ---------------------------------------------------------------

    local_z, local_y, local_x = np.indices(
        cam.shape,
        dtype=np.float64,
    )

    local_zyx = np.stack(
        [local_z, local_y, local_x],
        axis=0,
    )

    delta_local_zyx = (
        local_zyx
        - local_patch_center_zyx[:, None, None, None]
    )

    native_zyx = (
        candidate_center_zyx[:, None, None, None]
        +
        delta_local_zyx
        * patch_spacing_zyx_mm[:, None, None, None]
        / native_spacing_zyx_mm[:, None, None, None]
    )

    native_min_zyx = np.min(
        native_zyx.reshape(3, -1),
        axis=1,
    )

    native_max_zyx = np.max(
        native_zyx.reshape(3, -1),
        axis=1,
    )

    return native_zyx, native_min_zyx, native_max_zyx


def sample_projected_cam(
    cam: np.ndarray,
    native_coords_zyx: np.ndarray,
    volume_shape_zyx: Tuple[int, int, int],
) -> np.ndarray:
    """
    Resample local CAM directly into the Stage 02 native voxel grid.

    The returned array has shape (Z,Y,X).

    No axis transpose occurs.
    """

    projected = np.zeros(
        volume_shape_zyx,
        dtype=np.float32,
    )

    coords = native_coords_zyx

    sampled = map_coordinates(
        np.asarray(cam, dtype=np.float32),
        coords,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    valid = (
        (coords[0] >= 0)
        & (coords[0] < volume_shape_zyx[0])
        & (coords[1] >= 0)
        & (coords[1] < volume_shape_zyx[1])
        & (coords[2] >= 0)
        & (coords[2] < volume_shape_zyx[2])
    )

    # Native voxel indices corresponding to projected coordinates.
    iz = np.rint(coords[0][valid]).astype(np.int64)
    iy = np.rint(coords[1][valid]).astype(np.int64)
    ix = np.rint(coords[2][valid]).astype(np.int64)

    np.maximum(iz, 0, out=iz)
    np.maximum(iy, 0, out=iy)
    np.maximum(ix, 0, out=ix)

    np.minimum(iz, volume_shape_zyx[0] - 1, out=iz)
    np.minimum(iy, volume_shape_zyx[1] - 1, out=iy)
    np.minimum(ix, volume_shape_zyx[2] - 1, out=ix)

    np.maximum.at(
        projected,
        (iz, iy, ix),
        sampled[valid],
    )

    return projected


def native_to_local(
    native_zyx: np.ndarray,
    candidate_center_zyx: np.ndarray,
    local_patch_center_zyx: np.ndarray,
    patch_spacing_zyx_mm: np.ndarray,
    native_spacing_zyx_mm: np.ndarray,
) -> np.ndarray:
    """Inverse of `local_to_native` -- used to pull samples for every native
    voxel out of the local Grad-CAM array."""

    offset_native = (native_zyx - candidate_center_zyx) * native_spacing_zyx_mm
    return local_patch_center_zyx + offset_native / patch_spacing_zyx_mm


def physical_of_local(
    local_zyx: np.ndarray,
    patch_center_physical_zyx_mm: np.ndarray,
    local_patch_center_zyx: np.ndarray,
    patch_spacing_zyx_mm: np.ndarray,
) -> np.ndarray:
    """physical_zyx_mm = patch_center_physical_zyx_mm
    + (local_zyx - local_patch_center_zyx) * patch_spacing_zyx_mm"""

    return patch_center_physical_zyx_mm + (
        local_zyx - local_patch_center_zyx
    ) * patch_spacing_zyx_mm


def project_cam_to_native(
    cam: np.ndarray,
    geometry: Dict[str, Any],
    volume_shape_zyx: Tuple[int, int, int],
    margin_vox: int = 4,
) -> Optional[Dict[str, Any]]:
    """
    Project a local (Z, Y, X) Grad-CAM array into the Stage 02 native CT
    grid, restricted to the (small) bounding box the patch actually covers,
    padded by `margin_vox` for context and clamped to the volume bounds.

    Returns a dict with the sampled heatmap plus the native bounding box it
    occupies (`native_start_zyx` inclusive, `native_end_zyx` exclusive), or
    None if the patch falls entirely outside the CT volume.
    """

    candidate_center_zyx = geometry["candidate_center_zyx"]
    local_patch_center_zyx = geometry["local_patch_center_zyx"]
    patch_spacing_zyx_mm = geometry["patch_spacing_zyx_mm"]
    native_spacing_zyx_mm = geometry["native_spacing_zyx_mm"]

    patch_shape = np.asarray(cam.shape, dtype=np.float64)

    corners_local = np.array(
        list(
            itertools.product(
                [0.0, patch_shape[0] - 1.0],
                [0.0, patch_shape[1] - 1.0],
                [0.0, patch_shape[2] - 1.0],
            )
        )
    )

    corners_native = np.array(
        [
            local_to_native(
                corner,
                candidate_center_zyx,
                local_patch_center_zyx,
                patch_spacing_zyx_mm,
                native_spacing_zyx_mm,
            )
            for corner in corners_local
        ]
    )

    lo = np.floor(corners_native.min(axis=0)).astype(np.int64) - margin_vox
    hi = np.ceil(corners_native.max(axis=0)).astype(np.int64) + margin_vox + 1

    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(volume_shape_zyx, dtype=np.int64))

    if np.any(hi <= lo):
        return None

    zz, yy, xx = np.meshgrid(
        np.arange(lo[0], hi[0], dtype=np.float64),
        np.arange(lo[1], hi[1], dtype=np.float64),
        np.arange(lo[2], hi[2], dtype=np.float64),
        indexing="ij",
    )
    native_pts = np.stack([zz, yy, xx], axis=-1)

    local_pts = native_to_local(
        native_pts,
        candidate_center_zyx,
        local_patch_center_zyx,
        patch_spacing_zyx_mm,
        native_spacing_zyx_mm,
    )

    coords = [local_pts[..., axis] for axis in range(3)]

    sampled = map_coordinates(cam, coords, order=1, mode="constant", cval=0.0)
    sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
    sampled = np.clip(sampled, 0.0, 1.0).astype(np.float32)

    return {
        "heatmap": sampled,
        "native_start_zyx": lo,
        "native_end_zyx": hi,
    }

def validate_stage02_projection(
    native_coords_zyx: np.ndarray,
    volume_shape_zyx: Tuple[int, int, int],
    candidate_center_zyx: np.ndarray,
) -> None:
    """
    Validate that the projected Grad-CAM is spatially consistent
    with the Stage 02 volume.
    """

    if native_coords_zyx.shape[0] != 3:
        raise ValueError(
            "Projection coordinates must have shape (3,Z,Y,X)"
        )

    mins = np.min(
        native_coords_zyx.reshape(3, -1),
        axis=1,
    )

    maxs = np.max(
        native_coords_zyx.reshape(3, -1),
        axis=1,
    )

    shape = np.asarray(volume_shape_zyx, dtype=np.float64)

    # Some out-of-bounds voxels are allowed because map_coordinates
    # can sample outside the volume. But the candidate center itself
    # must be inside the Stage 02 CT.
    center = np.asarray(
        candidate_center_zyx,
        dtype=np.float64,
    )

    if np.any(center < 0) or np.any(center >= shape):
        raise RuntimeError(
            "Candidate center lies outside Stage 02 volume:\n"
            f"  center={center.tolist()}\n"
            f"  shape={volume_shape_zyx}"
        )

    print(
        "Projection bounds (Stage 02 ZYX): "
        f"min={mins.tolist()} max={maxs.tolist()}"
    )

def normalize_heatmap(raw: np.ndarray) -> np.ndarray:
    heatmap = np.squeeze(np.asarray(raw, dtype=np.float32))
    if heatmap.ndim != 3:
        raise RuntimeError(f"Grad-CAM must be 3D, got shape {heatmap.shape}")

    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = np.maximum(heatmap, 0.0)  # positive evidence only

    peak = float(np.max(heatmap))
    if peak <= 0.0:
        return np.zeros_like(heatmap, dtype=np.float32)

    return np.clip(heatmap / peak, 0.0, 1.0).astype(np.float32)


# ============================================================================
# VISUALIZATION
# ============================================================================


def _window_ct(slice_hu: np.ndarray) -> np.ndarray:
    windowed = np.clip(slice_hu, HU_MIN, HU_MAX)
    return (windowed - HU_MIN) / (HU_MAX - HU_MIN)


def save_overlay(
    ct: np.ndarray,
    heat: np.ndarray,
    lo: np.ndarray,
    candidate_center_zyx: np.ndarray,
    threshold: float,
    alpha: float,
    title: str,
    out_path: Path,
) -> None:
    """2D slice overlay: axial / coronal / sagittal through the candidate
    center, each showing the CT windowed to lung/soft-tissue HU with the
    projected heatmap alpha-blended on top."""

    center_local = np.clip(
        np.round(candidate_center_zyx - lo).astype(int),
        [0, 0, 0],
        np.array(heat.shape) - 1,
    )
    z, y, x = center_local

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    views = [
        ("Axial", ct[z, :, :], heat[z, :, :]),
        ("Coronal", ct[:, y, :], heat[:, y, :]),
        ("Sagittal", ct[:, :, x], heat[:, :, x]),
    ]

    for ax, (name, ct_slice, heat_slice) in zip(axes, views):
        ax.imshow(_window_ct(ct_slice), cmap="gray", interpolation="nearest")
        masked = np.ma.masked_less(heat_slice, threshold)
        ax.imshow(masked, cmap="jet", alpha=alpha, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_projection(
    ct: np.ndarray,
    heat: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """3D projection: maximum-intensity projections of the heatmap (and its
    matching CT window) along each anatomical axis, within the candidate's
    native bounding box."""

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axis_names = ["Axial (Z-MIP)", "Coronal (Y-MIP)", "Sagittal (X-MIP)"]

    for ax, name, axis in zip(axes, axis_names, (0, 1, 2)):
        ct_mip = _window_ct(ct.max(axis=axis))
        heat_mip = heat.max(axis=axis)
        ax.imshow(ct_mip, cmap="gray", interpolation="nearest")
        ax.imshow(heat_mip, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ============================================================================
# FULL-CT COMPOSITING (shared by GIF export and the interactive viewer)
# ============================================================================


def nonzero_native_z_slices(heat: np.ndarray, lo: np.ndarray, threshold: float) -> Tuple[List[int], List[int]]:
    """
    Return (global_z_indices, local_z_indices) -- sorted, paired -- for every
    Z-slice of the projected heatmap that has at least one voxel at or above
    `threshold`. `global_z_indices` are native Stage 02 volume Z indices
    (`local_z + lo[0]`); `local_z_indices` index directly into `heat`.
    """

    local_zs = np.where(np.any(heat >= threshold, axis=(1, 2)))[0]
    global_zs = local_zs + int(lo[0])
    return global_zs.tolist(), local_zs.tolist()


def window_ct_uint8_rgb(slice_hu: np.ndarray) -> np.ndarray:
    """Full CT slice (native Y, X extent), lung/soft-tissue windowed, as
    uint8 grayscale-in-RGB (H, W, 3)."""

    gray = (_window_ct(slice_hu) * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def colorize_heatmap_rgba(
    heat_slice_local: np.ndarray,
    cmap_name: str,
    threshold: float,
) -> np.ndarray:
    """Color a *local* (candidate-bbox-sized) 2D heatmap slice with `cmap_name`,
    fully transparent below `threshold`. Returns (h, w, 4) uint8 RGBA."""

    cmap = _get_cmap(cmap_name)
    colored = (cmap(np.clip(heat_slice_local, 0.0, 1.0))[..., :3] * 255.0).astype(np.uint8)
    alpha = np.where(heat_slice_local >= threshold, 255, 0).astype(np.uint8)
    return np.dstack([colored, alpha])


def full_frame_overlay_rgba(
    heat_slice_local: np.ndarray,
    lo_y: int,
    lo_x: int,
    full_shape_yx: Tuple[int, int],
    cmap_name: str,
    threshold: float,
) -> np.ndarray:
    """Place a colorized local heatmap slice into a full-CT-sized (H, W, 4)
    transparent RGBA canvas at its native (Y, X) offset."""

    height, width = full_shape_yx
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    colored = colorize_heatmap_rgba(heat_slice_local, cmap_name, threshold)
    hh, ww = heat_slice_local.shape
    y0, x0 = max(lo_y, 0), max(lo_x, 0)
    y1, x1 = min(y0 + hh, height), min(x0 + ww, width)
    if y1 > y0 and x1 > x0:
        canvas[y0:y1, x0:x1, :] = colored[: y1 - y0, : x1 - x0, :]
    return canvas


def composite_full_frame_rgb(
    ct_slice_hu: np.ndarray,
    heat_slice_local: Optional[np.ndarray],
    lo_y: int,
    lo_x: int,
    cmap_name: str,
    threshold: float,
    alpha: float,
) -> np.ndarray:
    """Flat (non-transparent) full-CT frame with the heatmap alpha-blended
    in place -- used for GIF frames, which have no separate alpha layer."""

    base = window_ct_uint8_rgb(ct_slice_hu).astype(np.float32)
    if heat_slice_local is not None and heat_slice_local.size:
        overlay_rgba = full_frame_overlay_rgba(
            heat_slice_local, lo_y, lo_x, base.shape[:2], cmap_name, threshold
        ).astype(np.float32)
        mask = overlay_rgba[..., 3:4] / 255.0  # 0 or 1 per the threshold mask
        blend = mask * alpha
        base = base * (1.0 - blend) + overlay_rgba[..., :3] * blend
    return np.clip(base, 0, 255).astype(np.uint8)


def export_animated_gif(
    ct_volume: np.ndarray,
    heat: np.ndarray,
    lo: np.ndarray,
    threshold: float,
    alpha: float,
    cmap_name: str,
    out_path: Path,
    fps: int = DEFAULT_GIF_FPS,
    max_frames: int = DEFAULT_GIF_MAX_FRAMES,
) -> Optional[Dict[str, Any]]:
    """
    Render and save an animated GIF cycling through every native Z-slice
    with non-zero (>= threshold) projected heatmap, each frame showing the
    FULL native CT slice with the heatmap overlaid at its projected
    position. Returns metadata about the export, or None if there is
    nothing to animate (no voxel ever crosses `threshold`).
    """

    global_zs, local_zs = nonzero_native_z_slices(heat, lo, threshold)
    if not global_zs:
        return None

    if len(global_zs) > max_frames:
        sample_idx = np.linspace(0, len(global_zs) - 1, max_frames).round().astype(int)
        sample_idx = sorted(set(sample_idx.tolist()))
        global_zs = [global_zs[i] for i in sample_idx]
        local_zs = [local_zs[i] for i in sample_idx]

    frames = []
    for gz, lz in zip(global_zs, local_zs):
        frame = composite_full_frame_rgb(
            ct_volume[gz, :, :],
            heat[lz, :, :],
            lo_y=int(lo[1]),
            lo_x=int(lo[2]),
            cmap_name=cmap_name,
            threshold=threshold,
            alpha=alpha,
        )
        frames.append(Image.fromarray(frame, mode="RGB"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(fps, 1))))
    frames[0].save(
        out_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )

    return {
        "path": str(out_path),
        "num_frames": len(frames),
        "fps": fps,
        "z_slices_native": global_zs,
    }


def _encode_png_b64(rgb_or_rgba: np.ndarray, max_dim: Optional[int]) -> str:
    mode = "RGBA" if rgb_or_rgba.shape[-1] == 4 else "RGB"
    img = Image.fromarray(rgb_or_rgba, mode=mode)
    if max_dim and max(img.size) > max_dim:
        scale = max_dim / float(max(img.size))
        new_size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
        resample = Image.NEAREST if mode == "RGBA" else Image.BILINEAR
        img = img.resize(new_size, resample)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_VIEWER_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 08 -- {patient_id} -- candidate {candidate_id} -- interactive viewer</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ background:#111318; color:#e8e8ec; font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; padding:24px; }}
  h1 {{ font-size:16px; font-weight:600; margin:0 0 4px; }}
  .sub {{ color:#9aa0ab; font-size:12px; margin-bottom:16px; }}
  .layout {{ display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap; }}
  .frame-wrap {{ position:relative; background:#000; border-radius:8px; overflow:hidden; line-height:0; }}
  .frame-wrap img {{ display:block; position:absolute; top:0; left:0; width:100%; height:auto; image-rendering:pixelated; }}
  .frame-wrap img.base {{ position:static; }}
  .controls {{ min-width:260px; display:flex; flex-direction:column; gap:14px; }}
  .control label {{ display:block; font-size:12px; color:#9aa0ab; margin-bottom:4px; }}
  select, input[type=range] {{ width:100%; }}
  .readout {{ font-variant-numeric: tabular-nums; font-size:13px; color:#e8e8ec; }}
  .badge {{ display:inline-block; background:#1f2430; border-radius:4px; padding:2px 8px; font-size:11px; color:#8fb3ff; margin-right:6px; }}
</style>
</head>
<body>
<h1>{patient_id} &mdash; candidate {candidate_id} &mdash; full-CT Grad-CAM viewer</h1>
<div class="sub">
  <span class="badge">coordinate_order: ZYX</span>
  <span class="badge">source_space: stage02_native_ct</span>
  <span class="badge">projection_authority: stage08</span>
</div>
<div class="layout">
  <div class="frame-wrap" id="frameWrap">
    <img class="base" id="baseImg" alt="CT slice">
    <img id="overlayImg" alt="Grad-CAM overlay" style="opacity:0.45;">
  </div>
  <div class="controls">
    <div class="control">
      <label for="headSelect">Classifier head</label>
      <select id="headSelect"></select>
    </div>
    <div class="control">
      <label for="cmapSelect">Colormap</label>
      <select id="cmapSelect"></select>
    </div>
    <div class="control">
      <label for="alphaRange">Overlay transparency (alpha)</label>
      <input type="range" id="alphaRange" min="0" max="1" step="0.01" value="0.45">
    </div>
    <div class="control">
      <label for="zRange">Z-slice</label>
      <input type="range" id="zRange" min="0" max="0" step="1" value="0">
      <div class="readout" id="zReadout">Slice: -- / --</div>
    </div>
  </div>
</div>
<script>
const VIEWER_DATA = {viewer_data_json};

const headSelect = document.getElementById('headSelect');
const cmapSelect = document.getElementById('cmapSelect');
const alphaRange = document.getElementById('alphaRange');
const zRange = document.getElementById('zRange');
const zReadout = document.getElementById('zReadout');
const baseImg = document.getElementById('baseImg');
const overlayImg = document.getElementById('overlayImg');

const heads = Object.keys(VIEWER_DATA.heads);
for (const h of heads) {{
  const opt = document.createElement('option');
  opt.value = h; opt.textContent = h;
  headSelect.appendChild(opt);
}}
for (const c of VIEWER_DATA.colormaps) {{
  const opt = document.createElement('option');
  opt.value = c; opt.textContent = c;
  cmapSelect.appendChild(opt);
}}

function currentHeadData() {{
  return VIEWER_DATA.heads[headSelect.value];
}}

function render() {{
  const headData = currentHeadData();
  if (!headData) return;
  const z = parseInt(zRange.value, 10);
  const zKey = String(z);
  baseImg.src = 'data:image/png;base64,' + (VIEWER_DATA.base_frames[zKey] || '');
  const cmap = cmapSelect.value;
  const overlayB64 = (headData.overlays[cmap] && headData.overlays[cmap][zKey]) || null;
  overlayImg.style.display = overlayB64 ? 'block' : 'none';
  if (overlayB64) overlayImg.src = 'data:image/png;base64,' + overlayB64;
  overlayImg.style.opacity = alphaRange.value;
  zReadout.textContent = 'Slice: ' + z + ' / ' + (VIEWER_DATA.z_max) +
    '  (native ZYX, head range ' + headData.z_start + '-' + headData.z_end + ')';
}}

function setHead() {{
  const headData = currentHeadData();
  zRange.min = VIEWER_DATA.z_min;
  zRange.max = VIEWER_DATA.z_max;
  zRange.value = headData.z_start;
  render();
}}

headSelect.addEventListener('change', setHead);
cmapSelect.addEventListener('change', render);
alphaRange.addEventListener('input', render);
zRange.addEventListener('input', render);

if (heads.length) {{
  headSelect.value = heads[0];
  setHead();
}}
</script>
</body>
</html>
"""


def export_interactive_viewer(
    ct_volume: np.ndarray,
    per_head_results: Dict[str, Dict[str, Any]],
    patient_id: str,
    candidate_id: int,
    threshold: float,
    default_alpha: float,
    cmap_names: Tuple[str, ...],
    out_path: Path,
    max_dim: int = DEFAULT_VIEWER_MAX_DIM,
) -> Optional[Dict[str, Any]]:
    """
    Build a single self-contained HTML viewer (no server, no Jupyter
    required) for one candidate, covering every projected head. Ships a
    Z-slider (real-time slice readout), a head selector, a colormap
    selector, and an alpha slider -- all live/client-side, backed by
    pre-rendered base64 PNG frames.
    """

    heads_with_signal = {
        head: res for head, res in per_head_results.items()
        if nonzero_native_z_slices(res["heatmap"], res["native_start_zyx"], threshold)[0]
    }
    if not heads_with_signal:
        return None

    global_z0 = min(int(res["native_start_zyx"][0]) for res in heads_with_signal.values())
    global_z1 = max(int(res["native_end_zyx"][0]) - 1 for res in heads_with_signal.values())

    base_frames: Dict[str, str] = {}
    for z in range(global_z0, global_z1 + 1):
        rgb = window_ct_uint8_rgb(ct_volume[z, :, :])
        base_frames[str(z)] = _encode_png_b64(rgb, max_dim)

    full_shape_yx = ct_volume.shape[1:]
    heads_json: Dict[str, Any] = {}
    for head, res in heads_with_signal.items():
        heat = res["heatmap"]
        lo = res["native_start_zyx"]
        global_zs, local_zs = nonzero_native_z_slices(heat, lo, threshold)
        overlays: Dict[str, Dict[str, str]] = {c: {} for c in cmap_names}
        for gz, lz in zip(global_zs, local_zs):
            for cmap_name in cmap_names:
                rgba = full_frame_overlay_rgba(
                    heat[lz, :, :], int(lo[1]), int(lo[2]), full_shape_yx, cmap_name, threshold
                )
                overlays[cmap_name][str(gz)] = _encode_png_b64(rgba, max_dim)
        heads_json[head] = {
            "z_start": int(min(global_zs)),
            "z_end": int(max(global_zs)),
            "overlays": overlays,
        }

    viewer_data = {
        "patient_id": patient_id,
        "candidate_id": candidate_id,
        "z_min": global_z0,
        "z_max": global_z1,
        "colormaps": list(cmap_names),
        "base_frames": base_frames,
        "heads": heads_json,
    }

    html = _VIEWER_HTML_TEMPLATE.format(
        patient_id=patient_id,
        candidate_id=candidate_id,
        viewer_data_json=json.dumps(viewer_data),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    return {
        "path": str(out_path),
        "heads": sorted(heads_json.keys()),
        "z_min": global_z0,
        "z_max": global_z1,
    }


# ============================================================================
# CANDIDATE PROCESSING
# ============================================================================


def process_candidate(
    candidate_dir: Path,
    stage02: Dict[str, Any],
    stage05_manifest: Dict[str, Dict[str, Any]],
    patient_id: str,
    heads_filter: Optional[List[str]],
    threshold: float,
    alpha: float,
    overlays_dir: Path,
    projections_dir: Path,
    animations_dir: Path,
    viewer_dir: Path,
    colormaps: Tuple[str, ...] = DEFAULT_COLORMAPS,
    gif_fps: int = DEFAULT_GIF_FPS,
    gif_max_frames: int = DEFAULT_GIF_MAX_FRAMES,
    viewer_max_dim: int = DEFAULT_VIEWER_MAX_DIM,
    make_gif: bool = True,
    make_viewer: bool = True,
) -> Dict[str, Any]:

    candidate_id = extract_candidate_id(candidate_dir)
    metadata = json.loads((candidate_dir / "metadata.json").read_text(encoding="utf-8"))
    stage05_record = stage05_manifest.get(str(candidate_id))

    geometry = resolve_candidate_geometry(
        metadata, stage05_record, stage02["spacing_zyx_mm"]
    )

    head_files = discover_head_files(candidate_dir)
    if heads_filter:
        head_files = {h: p for h, p in head_files.items() if h in heads_filter}

    gif_cmap = colormaps[0] if colormaps else "jet"

    if not head_files:
        return {
            "candidate_id": candidate_id,
            "patient_id": patient_id,
            "visualizations": {
                "overlay_dir": "08_visualization/overlays/",
                "projection_dir": "08_visualization/projections/",
                "animation_dir": "08_visualization/animations/",
                "interactive_viewer_status": "NOT_APPLICABLE",
                "interactive_viewer_path": None,
            },
            "status": "NO_HEAD_FILES",
        }

    projected_heads: List[str] = []
    heads_report: Dict[str, Any] = {}
    per_head_results: Dict[str, Dict[str, Any]] = {}  # feeds the interactive viewer
    projection_bounds_stage02 = None

    for head, path in sorted(head_files.items()):
        raw_cam = np.load(path, allow_pickle=False)
        cam = normalize_heatmap(raw_cam)

        result = project_cam_to_native(cam, geometry, stage02["volume_shape_zyx"])
        if result is None:
            continue

        native_coords, _, _ = local_cam_to_stage02_native(
            cam,
            np.asarray(geometry["candidate_center_zyx"], dtype=np.float64),
            np.asarray(geometry["local_patch_center_zyx"], dtype=np.float64),
            np.asarray(geometry["patch_spacing_zyx_mm"], dtype=np.float64),
            np.asarray(geometry["native_spacing_zyx_mm"], dtype=np.float64),
        )
        validate_stage02_projection(
            native_coords,
            stage02["volume_shape_zyx"],
            np.asarray(geometry["candidate_center_zyx"], dtype=np.float64),
        )
        if projection_bounds_stage02 is None:
            _, native_min_stage02, native_max_stage02 = local_cam_to_stage02_native(
                cam,
                geometry["candidate_center_zyx"],
                geometry["local_patch_center_zyx"],
                geometry["patch_spacing_zyx_mm"],
                geometry["native_spacing_zyx_mm"],
            )
            projection_bounds_stage02 = (native_min_stage02, native_max_stage02)

        heat = result["heatmap"]
        lo = result["native_start_zyx"]
        hi = result["native_end_zyx"]
        per_head_results[head] = result

        ct_crop = stage02["ct"][lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]

        overlay_path = overlays_dir / f"candidate_{candidate_id}_{head}_overlay.png"
        save_overlay(
            ct_crop,
            heat,
            lo,
            geometry["candidate_center_zyx"],
            threshold,
            alpha,
            title=f"{patient_id} | candidate {candidate_id} | {head}",
            out_path=overlay_path,
        )

        projection_path = projections_dir / f"candidate_{candidate_id}_{head}_projection.png"
        save_projection(
            ct_crop,
            heat,
            title=f"{patient_id} | candidate {candidate_id} | {head} (MIP)",
            out_path=projection_path,
        )

        gif_info = None
        if make_gif:
            gif_path = animations_dir / f"candidate_{candidate_id}_{head}_full_ct.gif"
            gif_info = export_animated_gif(
                stage02["ct"],
                heat,
                lo,
                threshold=threshold,
                alpha=alpha,
                cmap_name=gif_cmap,
                out_path=gif_path,
                fps=gif_fps,
                max_frames=gif_max_frames,
            )

        peak_local = np.asarray(np.unravel_index(int(np.argmax(cam)), cam.shape), dtype=np.float64)
        peak_native_zyx = local_to_native(
            peak_local,
            geometry["candidate_center_zyx"],
            geometry["local_patch_center_zyx"],
            geometry["patch_spacing_zyx_mm"],
            geometry["native_spacing_zyx_mm"],
        )
        peak_global_zyx = stage02_to_stage01_voxel(
            peak_native_zyx,
            stage02["crop_offset_zyx"],
        )

        projected_heads.append(head)
        heads_report[head] = {
            "native_start_zyx": lo.tolist(),
            "native_end_zyx": hi.tolist(),
            "peak_local_zyx": peak_local.tolist(),
            "peak_native_zyx": peak_native_zyx.tolist(),
            "peak_global_zyx": peak_global_zyx.tolist(),
            "active_voxels_native": int(np.count_nonzero(heat >= threshold)),
            "overlay_path": str(overlay_path),
            "projection_path": str(projection_path),
            "animation_path": gif_info["path"] if gif_info else None,
            "animation_num_frames": gif_info["num_frames"] if gif_info else 0,
            "animation_status": "EXPORTED" if gif_info else ("SKIPPED" if not make_gif else "NO_SIGNAL_ABOVE_THRESHOLD"),
        }

    viewer_info = None
    viewer_status = "SKIPPED"
    if projected_heads:
        if make_viewer:
            try:
                viewer_path = viewer_dir / f"candidate_{candidate_id}_viewer.html"
                viewer_info = export_interactive_viewer(
                    stage02["ct"],
                    per_head_results,
                    patient_id,
                    candidate_id,
                    threshold=threshold,
                    default_alpha=alpha,
                    cmap_names=colormaps,
                    out_path=viewer_path,
                    max_dim=viewer_max_dim,
                )
                viewer_status = "initialized" if viewer_info else "NO_SIGNAL_ABOVE_THRESHOLD"
            except Exception as exc:  # noqa: BLE001 -- never let viewer export sink a candidate
                viewer_status = f"ERROR: {exc}"
    else:
        viewer_status = "NOT_APPLICABLE"

    report = {
        "candidate_id": candidate_id,
        "patient_id": patient_id,
        "coordinate_order": COORDINATE_ORDER,
        "source_space": SOURCE_SPACE,
        "projection_authority": PROJECTION_AUTHORITY,
        "candidate_center_zyx": np.asarray(
            geometry["candidate_center_zyx"], dtype=np.float64
        ).tolist(),
        "patch_geometry": {
            "patch_shape_zyx": list(geometry["patch_shape_zyx"]),
            "local_center_zyx": list(geometry["local_patch_center_zyx"]),
            "patch_spacing_zyx_mm": list(geometry["patch_spacing_zyx_mm"]),
        },
        "projected_heads": projected_heads,
        "heads": heads_report,
        "visualizations": {
            "overlay_dir": "08_visualization/overlays/",
            "projection_dir": "08_visualization/projections/",
            "animation_dir": "08_visualization/animations/",
            "interactive_viewer_status": viewer_status,
            "interactive_viewer_path": viewer_info["path"] if viewer_info else None,
        },
        "status": "PROJECTED" if projected_heads else "OUT_OF_BOUNDS",
    }

    if projection_bounds_stage02 is not None:
        report = add_stage01_coordinates_to_report(
            report,
            np.asarray(geometry["candidate_center_zyx"], dtype=np.float64),
            stage02["crop_offset_zyx"],
            projection_bounds_stage02[0],
            projection_bounds_stage02[1],
        )

    return report


# ============================================================================
# CLI / MAIN
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 08 -- project local Grad-CAM heatmaps into native CT space."
    )
    parser.add_argument("patient_id", help="Patient/output identifier, e.g. LIDC-IDRI-0141")
    parser.add_argument("--output-root", default="output", help="Root output directory. Default: output")
    parser.add_argument("--stage02-dir", default=None, help="Override Stage 02 directory.")
    parser.add_argument("--stage05-dir", default=None, help="Override Stage 05 directory.")
    parser.add_argument("--stage07-dir", default=None, help="Override Stage 07 directory.")
    parser.add_argument(
        "--heads",
        nargs="*",
        default=None,
        help="Restrict projection to these classifier heads. Default: all discovered.",
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        type=int,
        default=None,
        help="Restrict projection to these candidate ids. Default: all discovered.",
    )
    parser.add_argument("--threshold", type=float, default=0.4, help="Heatmap display threshold in [0,1].")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay blend alpha.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    patient_dir = Path(args.output_root) / args.patient_id
    stage02_dir = Path(args.stage02_dir) if args.stage02_dir else patient_dir / "02"
    stage05_dir = Path(args.stage05_dir) if args.stage05_dir else patient_dir / "05_classifier_patches"
    stage07_dir = Path(args.stage07_dir) if args.stage07_dir else patient_dir / "07_gradcam"

    if not stage07_dir.is_dir():
        raise FileNotFoundError(f"Stage 07 directory not found: {stage07_dir}")

    output_dir = patient_dir / "08_visualization"
    overlays_dir = output_dir / "overlays"
    projections_dir = output_dir / "projections"
    animations_dir = output_dir / "animations"
    viewer_dir = output_dir / "viewer"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    projections_dir.mkdir(parents=True, exist_ok=True)
    animations_dir.mkdir(parents=True, exist_ok=True)
    viewer_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stage 02 (CT)     : {stage02_dir}")
    print(f"Stage 05 (geom)   : {stage05_dir}")
    print(f"Stage 07 (CAM)    : {stage07_dir}")
    print(f"Output            : {output_dir}")

    stage02 = load_stage02(stage02_dir)
    print(f"CT volume shape   : {stage02['volume_shape_zyx']}")
    print(f"Native spacing    : {stage02['spacing_zyx_mm'].tolist()} mm")
    print(f"Native origin     : {stage02['origin_zyx_mm'].tolist()} mm")
    print(f"Crop offset (ZYX) : {stage02['crop_offset_zyx'].tolist()}")

    stage05_manifest = load_stage05_manifest(stage05_dir)

    candidate_dirs = discover_candidate_directories(stage07_dir)
    if args.candidates is not None:
        wanted = set(args.candidates)
        candidate_dirs = [d for d in candidate_dirs if extract_candidate_id(d) in wanted]

    print(f"Candidates found  : {len(candidate_dirs)}")

    candidate_reports = []
    for candidate_dir in candidate_dirs:
        try:
            report = process_candidate(
                candidate_dir=candidate_dir,
                stage02=stage02,
                stage05_manifest=stage05_manifest,
                patient_id=args.patient_id,
                heads_filter=args.heads,
                threshold=args.threshold,
                alpha=args.alpha,
                overlays_dir=overlays_dir,
                projections_dir=projections_dir,
                animations_dir=animations_dir,
                viewer_dir=viewer_dir,
            )
        except Exception as exc:  # noqa: BLE001 -- keep batch alive on a bad candidate
            report = {
                "candidate_id": extract_candidate_id(candidate_dir),
                "patient_id": args.patient_id,
                "status": "ERROR",
                "error": str(exc),
            }
            print(f"  [ERROR] {candidate_dir.name}: {exc}")

        candidate_reports.append(report)
        print(f"  {candidate_dir.name}: {report.get('status')}")

    summary = {
        "stage": 8,
        "patient_id": args.patient_id,
        "coordinate_order": COORDINATE_ORDER,
        "source_space": SOURCE_SPACE,
        "projection_authority": PROJECTION_AUTHORITY,
        "native_volume_shape_zyx": list(stage02["volume_shape_zyx"]),
        "native_spacing_zyx_mm": stage02["spacing_zyx_mm"].tolist(),
        "native_origin_zyx_mm": stage02["origin_zyx_mm"].tolist(),
        "crop_offset_stage02_to_stage01_zyx": stage02[
            "crop_offset_zyx"
        ].tolist(),
        "num_candidates": len(candidate_reports),
        "visualizations": {
            "overlay_dir": "08_visualization/overlays/",
            "projection_dir": "08_visualization/projections/",
            "animation_dir": "08_visualization/animations/",
            "viewer_dir": "08_visualization/viewer/",
        },
        "candidates": candidate_reports,
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()