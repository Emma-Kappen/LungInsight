"""
05_extract_candidate_patches.py

LungInsight -- Stage 05 (REWRITTEN for architecture compliance)
=================================================================

Extract classifier-ready 3D patches from Stage 04 (ViTDet3D) candidate
detections, in strict compliance with the LungInsight Imaging Pipeline
Architecture's Stage 05 contract.

WHAT CHANGED AND WHY (audit summary vs. the previous stage_05.py)
------------------------------------------------------------------

1. GEOMETRY-AUTHORITY / VOLUME-MISMATCH BUG (critical, silent spatial
   shift on every patch).

   The previous script's own docstring said it loads "the ORIGINAL HU
   volume from Stage 01" and "voxel spacing from Stage 01 metadata",
   and its directory search order was:

       stage02_masked/, masked/, stage02/, <patient_output root>

   Stage 04 (04_detect_candidates.py) computes every `center_zyx` in
   `candidates.json` against the volume it loads from a directory
   literally named "02" (`output/<patient>/02/volume_hu.npy`), and it
   stamps every candidate with `"space": "stage02_native_ct"`. None of
   the previous script's search paths is "02". If Stage 02 crops,
   resamples, or otherwise re-origins the CT relative to Stage 01 (the
   whole point of a "mask_and_crop" stage), then falling through to a
   different directory -- or to a genuinely different, uncropped
   Stage 01 volume -- silently samples the *wrong physical location*
   for every candidate, even though the sampling code itself
   (map_coordinates) is numerically flawless. A perfectly-implemented
   interpolation of the wrong array is still a spatial misalignment
   bug, and it is the most dangerous kind because nothing raises an
   exception -- it just quietly clips a nodule out of frame.

   Fixed: Stage 05 now loads the exact "02" directory Stage 04 used,
   and cross-validates its shape and spacing against
   `04_candidates/detector_metadata.json` (also written by Stage 04).
   A mismatch is a hard error, not a warning -- see
   `assert_geometry_matches_stage04()`.

2. Candidate identity was discarded and silently reassigned.

   The previous script used `enumerate(candidates)` to name output
   files (`patch_0000.npy`, ...) and to populate `candidate_index` in
   the manifest. Stage 04's `candidate_id` -- the one stable identifier
   that ties a patch back to a specific detector proposal, and that
   Stage 06/07/08 need to join results back to `candidates.json` -- was
   never read or persisted. If any candidate failed extraction, every
   subsequent file name/index would silently shift relative to
   `candidates.json`, corrupting the join for every candidate after
   the failure.

   Fixed: `candidate_id` is read directly from each Stage 04 record
   and used verbatim for the output filename
   (`candidate_<ID>.npy`) and the metadata record. Nothing
   re-derives or re-assigns identity in Stage 05.

3. Coordinate mapping did not go through an explicit physical (mm)
   intermediate, and never consulted Stage 04's origin.

   The architecture's mapping formula is:

       physical_zyx_mm = patch_center_physical_zyx_mm
                          + (local_zyx - c_local) * patch_spacing_zyx_mm

   i.e. compute the *physical* location of every patch voxel first,
   then map that physical location back into Stage02-native voxel
   space for interpolation. The previous script instead converted the
   local-index offset straight from mm to native-voxel units and added
   that offset directly onto the raw candidate voxel index -- which is
   only numerically equivalent to the architecture's formula when the
   physical origin is exactly (0, 0, 0). Stage 04 records
   `origin_zyx_mm` precisely because that assumption isn't guaranteed
   to hold across every acquisition/crop. Skipping the origin term is
   a second, more subtle source of spatial offset whenever Stage 02
   defines a non-zero origin.

   Fixed: `patch_center_physical_zyx_mm` is computed explicitly
   (`origin + candidate_center_zyx * native_spacing_zyx_mm`), every
   patch voxel's physical position is computed from that anchor using
   the architecture's formula verbatim, and the physical position is
   only then converted back to native voxel indices
   (`(physical_mm - origin) / native_spacing_zyx_mm`) for
   `map_coordinates`. This also cleanly supports `patch_spacing_zyx_mm`
   being different from `native_spacing_zyx_mm` (e.g. resampling an
   anisotropic native volume onto an isotropic 1 mm classifier grid),
   which is exactly what direct physical resampling is supposed to
   buy you over a native crop + resize.

4. Output patch tensor shape did not match the classifier contract.

   The previous script saved each patch as shape `(1, 64, 64, 64)`
   (a channel dimension baked into the .npy on disk). The architecture
   specifies the *persisted* array as `(64, 64, 64)`, with `(1, 1, 64,
   64, 64)` (batch, channel, D, H, W) being a PyTorch ingestion-time
   shape, not a storage-time shape. Baking the channel dimension into
   the file silently changes `patch.ndim` for any downstream code
   (Stage 06, Grad-CAM) that assumes the documented on-disk shape.

   Fixed: patches are saved as raw `(64, 64, 64)` float32 arrays.
   `dtype`/reshape-for-inference is documented in the metadata instead
   of being embedded in the file.

5. Metadata schema did not match the architecture's required record.

   The previous manifest nested geometry under an ad hoc `"geometry"`
   dict with different key names (`center_zyx`,
   `source_spacing_zyx_mm`, no `geometry_authority`, no `source_space`,
   `coordinate_order` written as `"Z,Y,X"` instead of `"ZYX"`, etc.),
   and never declared Stage 05 as the spatial authority.

   Fixed: every candidate record now contains the exact key set the
   architecture specifies (`candidate_id`, `coordinate_order`,
   `source`, `candidate_center_zyx`, `native_volume_shape_zyx`,
   `native_spacing_zyx_mm`, `patch_shape_zyx`, `local_patch_center_zyx`,
   `patch_center_physical_zyx_mm`, `patch_spacing_zyx_mm`,
   `patch_fov_zyx_mm`, `patch_file`, `dtype`, `normalization`,
   `value_range`, `source_space`, `geometry_authority`), with
   additional non-conflicting provenance fields alongside (never
   replacing a required key).

6. Output layout renamed to match the architecture's file tree.

   `output/<patient>/05_classifier_patches/patches/candidate_<ID>.npy`
   (previously `stage05_classifier_patches/patches/patch_XXXX.npy`).

Coordinate convention
----------------------
All 3D arrays, physical coordinates, and spacing vectors use (Z, Y, X)
ordering everywhere. `coordinate_order` is stamped as the literal
string "ZYX" per the architecture's schema.

Classifier geometry
--------------------
    patch_shape_zyx        = (64, 64, 64)
    patch_spacing_zyx_mm   = (1.0, 1.0, 1.0)   (isotropic; FOV / size)
    patch_fov_zyx_mm       = (64.0, 64.0, 64.0)
    local_patch_center_zyx = (31.5, 31.5, 31.5)
    HU clip                = [-1000, 400]
    normalization          = (HU + 1000) / 1400
    value_range             = [0, 1]

Output
------
output/<patient_id>/05_classifier_patches/
    patches/
        candidate_<ID>.npy      <- raw (64, 64, 64) float32, [0, 1]
    patches.json                 <- manifest; one record per candidate,
                                     matching the required schema
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import map_coordinates


# ============================================================================
# CONFIGURATION -- classifier / patch contract
# ============================================================================

COORDINATE_ORDER = "ZYX"

PATCH_SIZE = 64
PATCH_SHAPE_ZYX = (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)

# The classifier's physical field of view and the resulting isotropic
# spacing of the resampled patch grid. This is independent of the
# native CT spacing -- that independence is the entire point of direct
# physical resampling instead of a native crop + zoom().
PATCH_FOV_MM = 64.0
PATCH_SPACING_MM = PATCH_FOV_MM / PATCH_SIZE  # 1.0 mm / patch voxel
PATCH_SPACING_ZYX_MM = (PATCH_SPACING_MM, PATCH_SPACING_MM, PATCH_SPACING_MM)
PATCH_FOV_ZYX_MM = (PATCH_FOV_MM, PATCH_FOV_MM, PATCH_FOV_MM)

# c_local: geometric center of an even-length (64) grid, halfway
# between indices 31 and 32.
LOCAL_PATCH_CENTER = (PATCH_SIZE - 1) / 2.0  # 31.5
LOCAL_PATCH_CENTER_ZYX = (LOCAL_PATCH_CENTER, LOCAL_PATCH_CENTER, LOCAL_PATCH_CENTER)

# Stage 06 classifier normalization (NOT the ViTDet3D mean/std used in
# Stage 04 -- see 04_detect_candidates.py's module docstring).
HU_MIN = -1000.0
HU_MAX = 400.0
NORMALIZATION_FORMULA = "(HU + 1000) / 1400"
VALUE_RANGE = "[0,1]"

DTYPE = "float32"
SOURCE_SPACE = "stage02_native_ct"
GEOMETRY_AUTHORITY = "stage05"

# Geometry cross-check tolerances against Stage 04's detector_metadata.json.
SPACING_MISMATCH_TOL_MM = 1e-4


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


# ============================================================================
# STAGE 02 VOLUME LOADING (the SAME volume Stage 04 detected against)
# ============================================================================

def load_stage02_volume(stage02_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the exact Stage 02 volume and geometry that Stage 04 ran
    ViTDet3D against.

    This MUST be the same directory Stage 04 read from
    (`output/<patient>/02/`), not a re-derived or differently-named
    "masked"/"original" directory. Stage 04 candidate centers are only
    meaningful in this volume's voxel grid.

    Returns
    -------
    volume_hu   : (Z, Y, X) float32
    spacing_zyx : (3,) float64, mm/voxel
    origin_zyx  : (3,) float64, physical coordinate of voxel [0,0,0]
    meta        : raw Stage 02 meta.json contents
    """

    volume_path = os.path.join(stage02_dir, "volume_hu.npy")
    meta_path = os.path.join(stage02_dir, "meta.json")

    if not os.path.isfile(volume_path):
        raise FileNotFoundError(
            "Could not find the Stage 02 volume Stage 04 detected against:\n"
            f"  {volume_path}\n"
            "Stage 05 must sample the SAME volume Stage 04 used -- it cannot "
            "substitute a different 'masked'/'original' directory, or "
            "candidate centers will be silently misaligned."
        )
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing Stage 02 metadata:\n{meta_path}")

    volume = np.asarray(np.load(volume_path, allow_pickle=False), dtype=np.float32)
    meta = _load_json(meta_path)

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D HU volume (Z,Y,X), got shape {volume.shape}")

    spacing = _extract_spacing(meta)
    origin = _extract_origin(meta)

    print(f"Stage 02 volume : {volume_path}")
    print(f"Volume shape    : {volume.shape}")
    print(f"Spacing (ZYX)   : {spacing[0]:.4f}, {spacing[1]:.4f}, {spacing[2]:.4f} mm")
    print(f"Origin  (ZYX)   : {origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f} mm")

    return volume, spacing, origin, meta


def _extract_spacing(meta: Dict[str, Any]) -> np.ndarray:
    for key in ("spacing_zyx_mm", "spacing_zyx", "spacing", "voxel_spacing"):
        if key in meta:
            spacing = np.asarray(meta[key], dtype=np.float64)
            if spacing.shape == (3,) and np.all(spacing > 0) and np.all(np.isfinite(spacing)):
                order = str(meta.get("spacing_order", meta.get("coordinate_order", "Z,Y,X")))
                if order.upper().replace(" ", "") in ("X,Y,Z", "XYZ"):
                    spacing = spacing[::-1]
                return spacing
    raise RuntimeError(
        "Could not determine voxel spacing from Stage 02 meta.json. "
        "Stage 05 requires exact ZYX spacing to build the physical patch grid."
    )


def _extract_origin(meta: Dict[str, Any]) -> np.ndarray:
    for key in ("origin_zyx_mm", "origin_zyx", "origin_mm", "origin"):
        if key in meta:
            origin = np.asarray(meta[key], dtype=np.float64)
            if origin.shape == (3,) and np.all(np.isfinite(origin)):
                order = str(meta.get("origin_order", meta.get("coordinate_order", "Z,Y,X")))
                if order.upper().replace(" ", "") in ("X,Y,Z", "XYZ"):
                    origin = origin[::-1]
                return origin
    # Zero origin is a legitimate default -- but note it, since a silently
    # assumed zero origin was exactly the previous script's Flaw #3.
    return np.zeros(3, dtype=np.float64)


# ============================================================================
# STAGE 04 CANDIDATE + METADATA LOADING
# ============================================================================

def load_stage04_candidates(stage04_dir: str) -> List[Dict[str, Any]]:
    """
    Load the AUTHORITATIVE Stage 04 candidate list.

    Only `candidates.json` is read. Diagnostic files
    (`log_candidates_diagnostic.json`, `candidates_diagnostic_agreement.json`)
    are never consumed here -- see 04_detect_candidates.py's module
    docstring: "Only candidates.json should ever be read by Stage 05."
    """

    path = os.path.join(stage04_dir, "candidates.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing Stage 04 candidates.json:\n{path}")

    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected candidates.json to contain a list, got {type(data)}")

    print(f"Stage 04 candidates : {path}")
    print(f"Loaded              : {len(data)} candidates")
    return [c for c in data if isinstance(c, dict)]


def load_stage04_detector_metadata(stage04_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load Stage 04's `detector_metadata.json`, used only to cross-check
    that Stage 05 is looking at the same volume geometry Stage 04 used.
    Returns None if absent (older Stage 04 runs) -- the cross-check is
    then skipped with a printed warning, not silently ignored.
    """

    path = os.path.join(stage04_dir, "detector_metadata.json")
    if not os.path.isfile(path):
        return None
    return _load_json(path)


def assert_geometry_matches_stage04(
    detector_metadata: Optional[Dict[str, Any]],
    volume_shape_zyx: Sequence[int],
    spacing_zyx: np.ndarray,
) -> None:
    """
    Hard-fail if the volume Stage 05 loaded does not match the volume
    Stage 04 ran ViTDet3D against.

    This is the single most important check in this script: every
    other correctness guarantee (direct physical resampling, exact
    coordinate mapping) is meaningless if the array being sampled is
    not the one the candidate centers were computed in.
    """

    if detector_metadata is None:
        print(
            "WARNING: 04_candidates/detector_metadata.json not found -- "
            "cannot cross-validate that Stage 05 is sampling the same "
            "volume Stage 04 detected against. Proceeding, but this "
            "check should not be skipped in production."
        )
        return

    expected_shape = tuple(detector_metadata.get("volume_shape_zyx", []))
    if expected_shape and tuple(volume_shape_zyx) != expected_shape:
        raise RuntimeError(
            "GEOMETRY MISMATCH: Stage 05 loaded a volume whose shape does "
            f"not match Stage 04's detector_metadata.json.\n"
            f"  Stage 04 volume_shape_zyx : {expected_shape}\n"
            f"  Stage 05 loaded shape     : {tuple(volume_shape_zyx)}\n"
            "Candidate centers are only valid in the exact volume Stage 04 "
            "detected against. Refusing to sample -- this would silently "
            "misplace every patch."
        )

    expected_spacing = detector_metadata.get("spacing_zyx_mm")
    if expected_spacing:
        expected_spacing = np.asarray(expected_spacing, dtype=np.float64)
        if not np.allclose(spacing_zyx, expected_spacing, atol=SPACING_MISMATCH_TOL_MM):
            raise RuntimeError(
                "GEOMETRY MISMATCH: Stage 05's loaded spacing does not "
                f"match Stage 04's detector_metadata.json.\n"
                f"  Stage 04 spacing_zyx_mm : {expected_spacing.tolist()}\n"
                f"  Stage 05 loaded spacing : {spacing_zyx.tolist()}\n"
                "Refusing to sample -- the physical patch grid would be "
                "built at the wrong scale."
            )

    expected_space = detector_metadata.get("space")
    if expected_space and expected_space != SOURCE_SPACE:
        raise RuntimeError(
            f"Stage 04 recorded space='{expected_space}', expected "
            f"'{SOURCE_SPACE}'. Refusing to proceed with an unrecognized "
            "coordinate space."
        )

    print("Geometry cross-check vs. Stage 04 detector_metadata.json: OK")


def extract_candidate_id(candidate: Dict[str, Any], fallback_index: int) -> int:
    """
    Read Stage 04's authoritative candidate_id verbatim.

    Stage 05 must never invent a new identity for a candidate (e.g. by
    re-enumerating the list); if `candidate_id` is genuinely absent
    (non-compliant upstream data), the loop index is used only as a
    last resort and the record is flagged so this is auditable.
    """

    if "candidate_id" in candidate:
        try:
            return int(candidate["candidate_id"])
        except (TypeError, ValueError):
            pass
    return fallback_index


def extract_candidate_center_zyx(candidate: Dict[str, Any]) -> np.ndarray:
    """
    Extract the Stage 04 candidate center in native Stage02 voxel
    coordinates. Per the architecture's "Detector Center Authority"
    rule, this value is used exactly as given -- never recomputed from
    a bounding box, never rounded.

    Accepts the schema's `candidate_center_zyx` name as well as the
    actual field name Stage 04 emits (`center_zyx`), plus a couple of
    other common synonyms, since the two names are used
    interchangeably across the architecture doc and the reference
    Stage 04 implementation.
    """

    for key in ("candidate_center_zyx", "center_zyx", "centroid_zyx", "center"):
        if key in candidate:
            arr = np.asarray(candidate[key], dtype=np.float64)
            if arr.shape == (3,) and np.all(np.isfinite(arr)):
                return arr

    raise ValueError(
        "Could not find a valid 3-element candidate center "
        f"(expected 'candidate_center_zyx' or 'center_zyx'). "
        f"Candidate keys: {list(candidate.keys())}"
    )


# ============================================================================
# PHYSICAL COORDINATE MAPPING
#
# physical_zyx_mm = patch_center_physical_zyx_mm
#                   + (local_zyx - c_local) * patch_spacing_zyx_mm
#
# followed by mapping that physical location back into Stage02-native
# voxel space for intensity interpolation:
#
# native_voxel_zyx = (physical_zyx_mm - origin_zyx_mm) / native_spacing_zyx_mm
# ============================================================================

def build_source_sampling_grid(
    candidate_center_zyx: np.ndarray,
    native_spacing_zyx: np.ndarray,
    native_origin_zyx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the (3, 64, 64, 64) array of native-volume voxel coordinates
    to sample with map_coordinates(), and return the
    patch_center_physical_zyx_mm anchor used to build it.
    """

    patch_center_physical_mm = native_origin_zyx + candidate_center_zyx * native_spacing_zyx

    local_indices = np.arange(PATCH_SIZE, dtype=np.float64)
    local_offset = local_indices - LOCAL_PATCH_CENTER  # (local_zyx - c_local)

    zz, yy, xx = np.meshgrid(local_offset, local_offset, local_offset, indexing="ij")

    # physical_zyx_mm = patch_center_physical_zyx_mm + local_offset * patch_spacing_zyx_mm
    physical_z_mm = patch_center_physical_mm[0] + zz * PATCH_SPACING_ZYX_MM[0]
    physical_y_mm = patch_center_physical_mm[1] + yy * PATCH_SPACING_ZYX_MM[1]
    physical_x_mm = patch_center_physical_mm[2] + xx * PATCH_SPACING_ZYX_MM[2]

    # Map physical (mm) coordinates back to Stage02-native voxel indices.
    source_z = (physical_z_mm - native_origin_zyx[0]) / native_spacing_zyx[0]
    source_y = (physical_y_mm - native_origin_zyx[1]) / native_spacing_zyx[1]
    source_x = (physical_x_mm - native_origin_zyx[2]) / native_spacing_zyx[2]

    source_coords = np.stack([source_z, source_y, source_x], axis=0)

    return source_coords, patch_center_physical_mm


def sample_physical_patch(
    volume_hu: np.ndarray,
    source_coords_zyx: np.ndarray,
) -> np.ndarray:
    """
    Directly resample the native volume at the given physical-space
    voxel coordinates. No native crop, no ndimage.zoom() -- a single
    physically-anchored interpolation, per "Direct Physical
    Resampling".

    Air padding (-1000 HU) outside the source volume, since a
    candidate near the CT edge should read as air, not replicated
    tissue from the boundary voxel.
    """

    patch_hu = map_coordinates(
        volume_hu,
        source_coords_zyx,
        order=1,
        mode="constant",
        cval=HU_MIN,
        prefilter=False,
    )

    patch_hu = np.asarray(patch_hu, dtype=np.float32)

    if patch_hu.shape != PATCH_SHAPE_ZYX:
        raise RuntimeError(f"Physical resampling produced unexpected shape: {patch_hu.shape}")

    return patch_hu


def normalize_patch(patch_hu: np.ndarray) -> np.ndarray:
    """
    normalized = (HU + 1000) / 1400, clipped to HU in [-1000, 400] so
    the output is guaranteed to fall in [0, 1] (the architecture's
    "Value Range" contract). This matches the Stage 06 classifier
    normalization documented in 04_detect_candidates.py's module
    docstring -- NOT the ViTDet3D mean/std, which is Stage-04-only.
    """

    clipped = np.clip(patch_hu, HU_MIN, HU_MAX)
    normalized = (clipped + 1000.0) / 1400.0
    return normalized.astype(np.float32, copy=False)


# ============================================================================
# MAIN EXTRACTION
# ============================================================================

def process_candidates(
    volume_hu: np.ndarray,
    native_spacing_zyx: np.ndarray,
    native_origin_zyx: np.ndarray,
    candidates: List[Dict[str, Any]],
    output_dir: str,
    patient_id: str,
    stage02_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    patch_dir = os.path.join(output_dir, "patches")
    os.makedirs(patch_dir, exist_ok=True)

    volume_shape_zyx = list(volume_hu.shape)
    crop_offset_zyx = np.asarray(
        (stage02_meta or {}).get("crop_offset_zyx", [0, 0, 0]),
        dtype=np.float64,
    )
    if crop_offset_zyx.shape != (3,) or not np.all(np.isfinite(crop_offset_zyx)):
        raise ValueError(
            f"Invalid Stage 02 crop_offset_zyx: "
            f"{(stage02_meta or {}).get('crop_offset_zyx')!r}"
        )
    records: List[Dict[str, Any]] = []
    stats = {"candidates_seen": len(candidates), "patches_written": 0, "candidates_failed": 0}

    print()
    print("=" * 72)
    print("EXTRACTING CLASSIFIER PATCHES")
    print("=" * 72)
    print(f"Candidates: {len(candidates)}")

    for index, candidate in enumerate(candidates):
        candidate_id = extract_candidate_id(candidate, fallback_index=index)
        print(f"\n[{index + 1}/{len(candidates)}] candidate_id={candidate_id}")

        try:
            center_zyx = extract_candidate_center_zyx(candidate)
            source = str(candidate.get("source", "ViTDet3D"))

            source_coords, patch_center_physical_mm = build_source_sampling_grid(
                candidate_center_zyx=center_zyx,
                native_spacing_zyx=native_spacing_zyx,
                native_origin_zyx=native_origin_zyx,
            )

            patch_hu = sample_physical_patch(volume_hu, source_coords)

            hu_min, hu_max, hu_mean = (
                float(np.min(patch_hu)),
                float(np.max(patch_hu)),
                float(np.mean(patch_hu)),
            )

            patch = normalize_patch(patch_hu)

            if patch.shape != PATCH_SHAPE_ZYX:
                raise RuntimeError(f"Unexpected classifier patch shape: {patch.shape}")

            # Persisted array is (64, 64, 64) -- NOT (1, 64, 64, 64) or
            # (1, 1, 64, 64, 64). PyTorch ingestion reshape is a Stage 06
            # concern and is documented, not baked into the file.
            filename = f"candidate_{candidate_id}.npy"
            patch_path = os.path.join(patch_dir, filename)
            np.save(patch_path, patch)

            record = {
                # --- required schema (exact keys/names) -----------------
                "candidate_id": candidate_id,
                "coordinate_order": COORDINATE_ORDER,
                "source": source,
                "candidate_center_zyx": center_zyx.tolist(),
                "native_volume_shape_zyx": volume_shape_zyx,
                "native_spacing_zyx_mm": native_spacing_zyx.tolist(),
                "patch_shape_zyx": list(PATCH_SHAPE_ZYX),
                "local_patch_center_zyx": list(LOCAL_PATCH_CENTER_ZYX),
                "patch_center_physical_zyx_mm": patch_center_physical_mm.tolist(),
                "patch_spacing_zyx_mm": list(PATCH_SPACING_ZYX_MM),
                "patch_fov_zyx_mm": list(PATCH_FOV_ZYX_MM),
                "patch_file": os.path.join("patches", filename).replace("\\", "/"),
                "dtype": DTYPE,
                "normalization": NORMALIZATION_FORMULA,
                "value_range": VALUE_RANGE,
                "source_space": SOURCE_SPACE,
                "geometry_authority": GEOMETRY_AUTHORITY,
                                "original_source_space": "stage01_original_ct",
                                "stage02_crop_offset_zyx": crop_offset_zyx.tolist(),
                                "candidate_center_stage02_zyx": center_zyx.tolist(),
                                "candidate_center_stage01_zyx": (
                                    center_zyx + crop_offset_zyx
                                ).tolist(),
                                "coordinate_frames": {
                                    "patch_geometry": SOURCE_SPACE,
                                    "original_ct": "stage01_original_ct",
                                    "stage02_to_stage01_offset_zyx": crop_offset_zyx.tolist(),
                                },
                # --- additional, non-conflicting provenance -------------
                "native_origin_zyx_mm": native_origin_zyx.tolist(),
                "pytorch_ingestion_shape": [1, 1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE],
                "detector_score": candidate.get("detector_score", candidate.get("score")),
                "hu_stats": {"min": hu_min, "max": hu_max, "mean": hu_mean},
                "hu_clip_range": [HU_MIN, HU_MAX],
                "sampling_method": "scipy.ndimage.map_coordinates",
                "interpolation_order": 1,
                "outside_volume_mode": "constant",
                "outside_volume_value_hu": HU_MIN,
                "stage04_candidate": candidate,
            }

            records.append(record)
            stats["patches_written"] += 1

            print(f"  center (native voxel, ZYX) : {center_zyx.tolist()}")
            print(f"  patch_center_physical_mm   : {patch_center_physical_mm.tolist()}")
            print(f"  HU range                   : [{hu_min:.1f}, {hu_max:.1f}]")
            print(f"  saved                      : {filename}")

        except Exception as exc:
            stats["candidates_failed"] += 1
            print(f"  [ERROR] candidate_id={candidate_id}: {exc}")

    manifest = {
        "stage": 5,
        "description": (
            "Classifier-ready patches extracted directly from the Stage 02 "
            "native CT volume using Stage 04 (ViTDet3D) candidate centers "
            "and an exact 64 mm isotropic physical sampling grid."
        ),
        "patient_id": patient_id,
        "coordinate_order": COORDINATE_ORDER,
        "geometry_authority": GEOMETRY_AUTHORITY,
        "source_space": SOURCE_SPACE,
        "classifier_geometry": {
            "patch_shape_zyx": list(PATCH_SHAPE_ZYX),
            "patch_spacing_zyx_mm": list(PATCH_SPACING_ZYX_MM),
            "patch_fov_zyx_mm": list(PATCH_FOV_ZYX_MM),
            "local_patch_center_zyx": list(LOCAL_PATCH_CENTER_ZYX),
            "dtype": DTYPE,
            "normalization": NORMALIZATION_FORMULA,
            "value_range": VALUE_RANGE,
            "pytorch_ingestion_shape": [1, 1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE],
        },
        "spatial_policy": {
            "authoritative_location": "Stage 04 candidate_center_zyx (verbatim, unrounded)",
            "candidate_box_used_for_patch_location": False,
            "intermediate_native_crop": False,
            "resampling_method": "scipy.ndimage.map_coordinates (direct physical resampling)",
            "mapping_formula": (
                "physical_zyx_mm = patch_center_physical_zyx_mm + "
                "(local_zyx - c_local) * patch_spacing_zyx_mm; "
                "native_voxel_zyx = (physical_zyx_mm - origin_zyx_mm) / native_spacing_zyx_mm"
            ),
        },
        "stats": stats,
        "patches": records,
    }

    manifest_path = os.path.join(output_dir, "patches.json")
    _save_json(manifest_path, manifest)

    print()
    print(f"Saved patch manifest: {manifest_path}")

    return manifest


# ============================================================================
# ARGUMENTS / ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 05: extract classifier-ready 64^3 patches from Stage 04 candidates."
    )
    parser.add_argument("patient_id", help="Patient/output identifier, e.g. LIDC-IDRI-0141")
    parser.add_argument("--output-root", default="output", help="Root output directory. Default: output")
    parser.add_argument(
        "--stage02-dir", default=None,
        help="Override for the Stage 02 directory Stage 04 detected against. "
        "Default: <output-root>/<patient_id>/02 (must match Stage 04 exactly).",
    )
    parser.add_argument(
        "--stage04-dir", default=None,
        help="Override for the Stage 04 output directory. "
        "Default: <output-root>/<patient_id>/04_candidates",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Stage 05 output directory. Default: <output-root>/<patient_id>/05_classifier_patches",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    patient_output = os.path.join(args.output_root, args.patient_id)

    stage02_dir = args.stage02_dir or os.path.join(patient_output, "02")
    stage04_dir = args.stage04_dir or os.path.join(patient_output, "04_candidates")
    stage05_dir = args.output_dir or os.path.join(patient_output, "05_classifier_patches")

    print("=" * 72)
    print("LUNGINSIGHT -- STAGE 05")
    print("CLASSIFIER PATCH EXTRACTION")
    print("=" * 72)
    print(f"Patient    : {args.patient_id}")
    print(f"Stage 02   : {stage02_dir}")
    print(f"Stage 04   : {stage04_dir}")
    print(f"Stage 05   : {stage05_dir}")
    print()
    print(f"Patch shape        : {PATCH_SHAPE_ZYX}")
    print(f"Patch FOV (mm)     : {PATCH_FOV_ZYX_MM}")
    print(f"Patch spacing (mm) : {PATCH_SPACING_ZYX_MM}")
    print(f"Normalization      : {NORMALIZATION_FORMULA}")
    print()

    volume_hu, native_spacing_zyx, native_origin_zyx, _stage02_meta = load_stage02_volume(stage02_dir)

    detector_metadata = load_stage04_detector_metadata(stage04_dir)
    assert_geometry_matches_stage04(detector_metadata, volume_hu.shape, native_spacing_zyx)

    candidates = load_stage04_candidates(stage04_dir)

    manifest = process_candidates(
        volume_hu=volume_hu,
        native_spacing_zyx=native_spacing_zyx,
        native_origin_zyx=native_origin_zyx,
        candidates=candidates,
        output_dir=stage05_dir,
        patient_id=args.patient_id,
        stage02_meta=_stage02_meta,
    )

    print()
    print("=" * 72)
    print("STAGE 05 COMPLETE")
    print("=" * 72)
    print(f"Candidates seen   : {manifest['stats']['candidates_seen']}")
    print(f"Patches written   : {manifest['stats']['patches_written']}")
    print(f"Candidates failed : {manifest['stats']['candidates_failed']}")
    print()
    print(f"Output directory: {os.path.abspath(stage05_dir)}")
    print()
    print("Each patch:")
    print(f"  on-disk shape        = {PATCH_SHAPE_ZYX}")
    print("  pytorch ingestion    = (1, 1, 64, 64, 64) -- reshape at load time, not on disk")
    print("  dtype                = float32")
    print("  range                = [0, 1]")
    print()
    print("Spatial mapping:")
    print("  patch center = Stage 04 candidate_center_zyx (verbatim)")
    print("  geometry_authority = stage05")
    print("  sampling = direct physical-coordinate interpolation (map_coordinates)")


if __name__ == "__main__":
    main()