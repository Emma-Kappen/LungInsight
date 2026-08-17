"""
04_detect_and_patch.py

STEP 4, in the pipeline:
    01_dicom_to_hu.py           -> DICOM -> HU volume
    02_mask_and_crop.py         -> lung segmentation + non-lung blanking + Z-crop
    03_visualize.py             -> viewing
    04_detect_and_patch.py      <- this file: characteristic-based candidate
                                    detection + FIXED-size patch extraction
    05_shape_filter_and_grow.py -> reject tubular candidates, grow +
                                    tightly crop each real nodule
    06_run_inference_xai.py     -> run the trained classifier + Grad-CAM,
                                    rank candidates by malignancy score

Loads volume_hu.npy / lung_mask.npy / meta.json from 02_mask_and_crop.py's
output directory (masked_dir).

=== WHAT THIS STAGE IS ===

This stage is pure CANDIDATE GENERATION. For every region it finds, it
answers exactly one question:

    "Is this a region of the lung that looks abnormal enough, relative
     to normal aerated lung, that it deserves to be handed to the
     shape filter (05) and eventually the classifier (06)?"

=== WHAT THIS STAGE IS NOT ===

This stage does NOT decide malignant vs. benign, does NOT reject
tubular/vessel-like shapes (that is 05's job -- this stage would
rather over-generate and let 05 filter than risk missing something
here), and does NOT assign a diagnosis of any kind. "candidate_score"
below is a TRIAGE PRIORITY used only to decide which candidates to
keep when there are far more than --max-raw-candidates -- it is not a
malignancy score.

=== WHY CHARACTERISTIC-BASED DETECTION, NOT LoG ===

An earlier version of this stage used multi-scale 3D Laplacian-of-
Gaussian (LoG) blob detection. LoG is fast, but it is built around a
spherical response model, so it systematically under-detects:

  * spiculated / lobulated masses
  * elongated or irregularly-shaped lesions
  * ground-glass opacities with soft, diffuse edges
  * lesions abutting the pleura or vessels (non-isolated blobs)

This version instead detects candidates using a set of independent,
non-parametric CHARACTERISTICS of suspicious pulmonary tissue, none of
which assume a spherical shape:

    1. Local attenuation abnormality (absolute HU)
    2. Local contrast against the surrounding lung background
    3. Heterogeneous internal attenuation
    4. Boundary irregularity / fill ratio
    5. Non-spherical / elongated morphology (allowed, not penalized)
    6. Physical size plausibility
    7. Multi-slice persistence
    8. Boundary contrast ("interface strength" with surrounding lung)

This is a DIFFERENT false-positive/recall trade-off than per-scale
LoG thresholding, not a strict improvement on every axis -- it is
deliberately high-recall (see docstring note above), and 05's Hessian
eigenvalue shape filter is what actually rejects the vessel/airway
cross-sections this approach will happily also flag.

=== INTERFACE CONTRACT WITH 05 (read this before changing field names) ===

05_shape_filter_and_grow.py loads this stage's candidates.json and
requires each candidate dict to contain, at minimum:

    "voxel_z", "voxel_y", "voxel_x"  (int)
        Seed voxel in volume_hu's own index space. This is the point
        05's Hessian shape analysis and region growing are centered on.

    "sigma_mm"  (float)
        A physical blob-scale estimate for this candidate, used by 05
        as the Gaussian scale for its Hessian eigenvalue computation
        ("reusing sigma_mm already stored in candidates.csv/json by
        04 -- no need to re-derive it", per 05's own docstring). Since
        this stage does not do multi-scale LoG, sigma_mm here is a
        derived heuristic (see estimate_sigma_mm()), not a true
        matched-filter scale -- but the field still needs to be
        present and reasonable, since 05 reads it unconditionally.

    "diameter_mm"  (float)
        This stage's own size estimate, used by 05 to size a FALLBACK
        box when region growing fails or leaks
        ("fallback_bbox_global_zyx(... diameter_mm ...)").

05 indexes candidates POSITIONALLY: it does `for candidate_id, cand in
enumerate(candidates)`, and 06 later does `candidates[candidate_id]`
to look coordinates back up. That means the ORDER candidates appear
in candidates.json is itself part of the contract -- do not silently
re-sort or de-duplicate candidates.json after writing it, or 05's
"candidate_id" numbering (baked into nodules.csv/nodules.json and the
patch filenames 06 reads) will point at the wrong candidate.

All other fields on each candidate dict (candidate_score,
mean_contrast_hu, irregularity, ...) are this stage's own diagnostic
output -- useful for auditing/tuning, not required by 05 or 06.

=== Correctness notes (mirroring the conventions established in 01-03) ===

  * All arrays and coordinate tuples are (Z, Y, X).
    meta["pixel_spacing_mm"] is DICOM PixelSpacing = [Y spacing,
    X spacing]; meta["slice_spacing_mm"] is the Z spacing. Assembled
    into (z, y, x) order in exactly one place (get_spacing_zyx) --
    05_shape_filter_and_grow.py copies this same helper verbatim, so
    if this assembly ever changes it must change in both files.
  * HU is read from volume_hu.npy AS-IS (already rescaled once, in
    step 1) -- never re-rescaled here.
  * Detection runs on a BLANKED copy of the volume (non-lung voxels
    set to AIR_HU) so background/vessel/chest-wall anatomy outside
    the lung can't seed a candidate. Patch EXTRACTION, by contrast,
    reads from the real, unblanked volume_hu -- the classifier
    downstream should see true local anatomy, not an artificial air
    cliff at the lung boundary (same rationale 05 documents for why
    ITS growth step also reads real HU, never the blanked volume).
  * extract_patch()/resample_patch() here are the versions
    05_shape_filter_and_grow.py's own docstring refers to when it
    says its own crop_native_window()/resample_to_shape() are
    "behavior-identical" reimplementations (05 can't import this file
    directly -- filenames starting with a digit aren't valid Python
    module names without importlib gymnastics).

Usage:
    python 04_detect_and_patch.py output/LIDC-IDRI-0001_masked \
        --out-dir output/LIDC-IDRI-0001_candidates \
        --patch-shape 64

Outputs (written to --out-dir):
    candidates.csv          -> one row per candidate: seed voxel,
                                diameter_mm, sigma_mm, candidate_score,
                                and all characteristic features.
    candidates.json          -> {"params": {...}, "candidates": [...]}
                                same rows as candidates.csv, in full
                                (list ORDER is part of the contract --
                                see interface note above)
    patches/candidate_XXXX.npy -> float32 HU array, shape
                                (patch_shape, patch_shape, patch_shape),
                                one per SURVIVING candidate (post-NMS),
                                index XXXX matches its position in
                                candidates.json. Convenience output for
                                direct inspection -- 05 recomputes its
                                own adaptive crop and does not read
                                these files.
    meta.json                -> copy of input meta.json + these params
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

try:
    from scipy import ndimage
except ImportError:
    print(
        "scipy is required. Install it with:\n"
        "    pip install scipy --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(1)


AIR_HU = -1000.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_masked_output(masked_dir: str):
    """Load volume_hu.npy (real, unblanked HU), lung_mask.npy, and
    meta.json written by 02_mask_and_crop.py."""
    paths = {
        "volume_hu": os.path.join(masked_dir, "volume_hu.npy"),
        "lung_mask": os.path.join(masked_dir, "lung_mask.npy"),
        "meta": os.path.join(masked_dir, "meta.json"),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. Run 02_mask_and_crop.py first, and "
                f"pass its --out-dir as masked_dir here."
            )
    volume_hu = np.load(paths["volume_hu"]).astype(np.float32, copy=False)
    lung_mask = np.load(paths["lung_mask"]).astype(bool, copy=False)
    with open(paths["meta"]) as f:
        meta = json.load(f)
    if volume_hu.shape != lung_mask.shape:
        raise ValueError(
            f"Shape mismatch in '{masked_dir}': volume_hu {volume_hu.shape} "
            f"vs lung_mask {lung_mask.shape}. Re-run 02_mask_and_crop.py."
        )
    return volume_hu, lung_mask, meta


def get_spacing_zyx(meta: dict):
    """(slice_spacing_mm, y_spacing_mm, x_spacing_mm). See module
    docstring -- 05_shape_filter_and_grow.py copies this helper
    verbatim, so keep the two in sync if this ever changes."""
    slice_spacing_mm = float(meta.get("slice_spacing_mm", 1.0) or 1.0)
    pixel_spacing_mm = meta.get("pixel_spacing_mm", [1.0, 1.0])
    y_spacing_mm = float(pixel_spacing_mm[0])
    x_spacing_mm = float(pixel_spacing_mm[1])
    return slice_spacing_mm, y_spacing_mm, x_spacing_mm


# ---------------------------------------------------------------------------
# Shared low-level helpers (native-spacing crop + fixed-shape resample)
# ---------------------------------------------------------------------------
#
# 05_shape_filter_and_grow.py's own crop_native_window()/
# resample_to_shape() are documented there as "behavior-identical"
# reimplementations of these two functions.

def extract_patch(volume: np.ndarray, center_zyx, half_extent_vox_zyx, pad_value: float):
    """
    Cut a (2*hz, 2*hy, 2*hx)-voxel box centered on center_zyx out of
    `volume` at native spacing, padding with pad_value outside the
    volume bounds. Also returns the GLOBAL voxel offset (z_lo, y_lo,
    x_lo) of the window's origin.
    """
    cz, cy, cx = center_zyx
    hz, hy, hx = half_extent_vox_zyx
    out_shape = (2 * hz, 2 * hy, 2 * hx)
    window = np.full(out_shape, pad_value, dtype=np.float32)

    z_lo, z_hi = cz - hz, cz + hz
    y_lo, y_hi = cy - hy, cy + hy
    x_lo, x_hi = cx - hx, cx + hx

    src_z_lo, src_z_hi = max(0, z_lo), min(volume.shape[0], z_hi)
    src_y_lo, src_y_hi = max(0, y_lo), min(volume.shape[1], y_hi)
    src_x_lo, src_x_hi = max(0, x_lo), min(volume.shape[2], x_hi)

    if src_z_lo >= src_z_hi or src_y_lo >= src_y_hi or src_x_lo >= src_x_hi:
        return window, (z_lo, y_lo, x_lo)

    dst_z_lo, dst_y_lo, dst_x_lo = src_z_lo - z_lo, src_y_lo - y_lo, src_x_lo - x_lo
    dst_z_hi = dst_z_lo + (src_z_hi - src_z_lo)
    dst_y_hi = dst_y_lo + (src_y_hi - src_y_lo)
    dst_x_hi = dst_x_lo + (src_x_hi - src_x_lo)

    window[dst_z_lo:dst_z_hi, dst_y_lo:dst_y_hi, dst_x_lo:dst_x_hi] = volume[
        src_z_lo:src_z_hi, src_y_lo:src_y_hi, src_x_lo:src_x_hi
    ]
    return window, (z_lo, y_lo, x_lo)


def resample_patch(patch: np.ndarray, target_shape_zyx, pad_value: float):
    """Resample `patch` (any shape) to exactly target_shape_zyx voxels
    via linear interpolation."""
    zoom_factors = [t / s for t, s in zip(target_shape_zyx, patch.shape)]
    resampled = ndimage.zoom(patch, zoom_factors, order=1, mode="nearest")
    out = np.full(target_shape_zyx, pad_value, dtype=np.float32)
    slices = tuple(slice(0, min(s, t)) for s, t in zip(resampled.shape, target_shape_zyx))
    out[slices] = resampled[slices]
    return out


# ---------------------------------------------------------------------------
# 1. Candidate detection
# ---------------------------------------------------------------------------

def characteristic_candidate_detect(
    volume_hu_masked: np.ndarray,
    lung_mask: np.ndarray,
    spacing_zyx,
    min_diameter_mm: float = 4.0,
    max_diameter_mm: float = 40.0,
    background_sigma_mm: float = 5.0,
    min_contrast_hu: float = 40.0,
    min_component_voxels: int = 20,
    max_raw_candidates: int = 8000,
):
    """
    Generate high-recall pulmonary lesion candidate regions using
    suspicious morphological/intensity characteristics rather than
    spherical LoG blob detection. See module docstring.

    `volume_hu_masked` should already have non-lung voxels blanked
    (e.g. set to AIR_HU) by the caller -- see module docstring's
    "Correctness notes" for why detection and patch extraction
    deliberately use different volumes.

    Returns
    -------
    list[dict]
        Raw candidate dictionaries, each containing the fields 05
        requires (voxel_z/y/x, sigma_mm, diameter_mm) plus diagnostic
        characteristic features and an axis-aligned bounding box.
    """

    if volume_hu_masked.shape != lung_mask.shape:
        raise ValueError(
            f"Shape mismatch: volume {volume_hu_masked.shape} "
            f"vs lung_mask {lung_mask.shape}"
        )

    sz, sy, sx = spacing_zyx

    # ------------------------------------------------------------------
    # Local-background contrast map.
    #
    # Instead of "is this voxel bright?", ask "is this voxel denser
    # than its local lung background?" -- this matters for ground-
    # glass / low-density lesions that never cross a hard HU threshold.
    # ------------------------------------------------------------------

    sigma_vox = (
        max(background_sigma_mm / sz, 0.5),
        max(background_sigma_mm / sy, 0.5),
        max(background_sigma_mm / sx, 0.5),
    )

    lung_values = volume_hu_masked[lung_mask]
    if lung_values.size == 0:
        return []

    background = ndimage.gaussian_filter(
        volume_hu_masked,
        sigma=sigma_vox,
        mode="nearest",
    )

    contrast = volume_hu_masked - background
    contrast[~lung_mask] = -np.inf

    # ------------------------------------------------------------------
    # Two complementary foreground mechanisms, unioned together:
    #   A. Absolute soft-tissue component (solid nodules/masses)
    #   B. Local-contrast component (ground-glass / low-density)
    # ------------------------------------------------------------------

    soft_tissue_mask = (volume_hu_masked > -650.0) & lung_mask
    contrast_mask = (contrast > min_contrast_hu) & lung_mask
    candidate_foreground = soft_tissue_mask | contrast_mask

    # ------------------------------------------------------------------
    # Denoise without enforcing spherical geometry.
    # ------------------------------------------------------------------

    structure = ndimage.generate_binary_structure(3, 1)

    candidate_foreground = ndimage.binary_opening(
        candidate_foreground, structure=structure, iterations=1,
    )
    candidate_foreground = ndimage.binary_closing(
        candidate_foreground, structure=structure, iterations=1,
    )

    # ------------------------------------------------------------------
    # Connected components.
    # ------------------------------------------------------------------

    labels, num_labels = ndimage.label(
        candidate_foreground,
        structure=ndimage.generate_binary_structure(3, 3),
    )

    if num_labels == 0:
        return []

    voxel_volume_mm3 = sz * sy * sx
    max_volume_mm3 = (np.pi / 6.0) * (max_diameter_mm ** 3)

    candidates = []

    for label_id in range(1, num_labels + 1):

        component = labels == label_id
        voxel_count = int(component.sum())

        if voxel_count < min_component_voxels:
            continue

        volume_mm3 = voxel_count * voxel_volume_mm3

        # Sanity bound only -- rejects e.g. large connected vessels /
        # hilar structures. Not a malignancy criterion.
        if volume_mm3 > max_volume_mm3 * 8.0:
            continue

        coords = np.argwhere(component)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        extent_z_mm = (z_max - z_min + 1) * sz
        extent_y_mm = (y_max - y_min + 1) * sy
        extent_x_mm = (x_max - x_min + 1) * sx
        max_extent_mm = max(extent_z_mm, extent_y_mm, extent_x_mm)

        if max_extent_mm < min_diameter_mm:
            continue

        # ---- internal intensity characteristics ----

        lesion_values = volume_hu_masked[component]
        lesion_contrast = contrast[component]

        mean_hu = float(np.mean(lesion_values))
        std_hu = float(np.std(lesion_values))
        mean_contrast = float(np.mean(lesion_contrast))
        max_contrast = float(np.max(lesion_contrast))

        heterogeneity = float(np.clip(std_hu / 200.0, 0.0, 1.0))
        contrast_score = float(np.clip(mean_contrast / 200.0, 0.0, 1.0))

        # ---- shape characteristics (irregularity is a feature, not a rejection) ----

        bbox_volume_mm3 = extent_z_mm * extent_y_mm * extent_x_mm
        fill_ratio = volume_mm3 / bbox_volume_mm3 if bbox_volume_mm3 > 0 else 0.0
        irregularity = 1.0 - np.clip(fill_ratio, 0.0, 1.0)

        min_extent = max(min(extent_z_mm, extent_y_mm, extent_x_mm), 1e-6)
        elongation = max_extent_mm / min_extent

        # ---- boundary / interface-strength proxy ----

        boundary_shell = (
            ndimage.binary_dilation(
                component,
                structure=ndimage.generate_binary_structure(3, 1),
                iterations=2,
            )
            & lung_mask
            & ~component
        )

        if boundary_shell.any():
            boundary_contrast = contrast[boundary_shell]
            positive_boundary_fraction = float(np.mean(boundary_contrast > min_contrast_hu))
            boundary_mean_contrast = float(np.mean(boundary_contrast))
        else:
            positive_boundary_fraction = 0.0
            boundary_mean_contrast = 0.0

        # ---- multi-slice persistence ----

        slice_presence = component.any(axis=(1, 2))
        occupied_slices = int(slice_presence.sum())
        persistence = float(np.clip(
            occupied_slices * sz / max(min_diameter_mm, 1.0),
            0.0, 1.0,
        ))

        # ---- seed selection: strongest-contrast voxel, not centroid ----
        #
        # For an irregular mass the centroid can land in a low-density
        # region or entirely outside the visually strongest part, so
        # we anchor downstream patch extraction / 05's shape analysis
        # on peak local contrast instead.

        component_indices = np.argwhere(component)
        component_contrasts = contrast[component]
        best_idx = int(np.argmax(component_contrasts))
        seed_z, seed_y, seed_x = component_indices[best_idx]

        # ---- size estimates consumed by 05 ----
        #
        # diameter_mm: equivalent diameter of a sphere with this
        # component's volume -- 05 uses this to size its FALLBACK box
        # when region growing fails or leaks.
        diameter_mm = float((6.0 * volume_mm3 / np.pi) ** (1.0 / 3.0))

        # sigma_mm: a physically reasonable Gaussian scale for 05's
        # Hessian eigenvalue computation at this candidate. This stage
        # does not do multi-scale LoG, so there is no true matched-
        # filter sigma to hand off -- this is a deliberate heuristic
        # (roughly diameter/4, clamped to a sane range) rather than a
        # derived optimum. See module docstring's interface contract.
        sigma_mm = float(np.clip(diameter_mm / 4.0, 1.0, 6.0))

        # ---- triage priority (NOT a cancer score) ----

        size_score = float(np.clip(
            (np.log1p(max_extent_mm) - np.log1p(min_diameter_mm))
            / (np.log1p(max_diameter_mm) - np.log1p(min_diameter_mm) + 1e-8),
            0.0, 1.0,
        ))

        boundary_score = float(np.clip(
            0.6 * positive_boundary_fraction
            + 0.4 * np.clip(boundary_mean_contrast / 200.0, 0.0, 1.0),
            0.0, 1.0,
        ))

        candidate_score = (
            0.30 * contrast_score
            + 0.20 * heterogeneity
            + 0.20 * boundary_score
            + 0.15 * irregularity
            + 0.10 * persistence
            + 0.05 * size_score
        )

        candidates.append({
            # --- required by 05_shape_filter_and_grow.py ---
            "voxel_z": int(seed_z), "voxel_y": int(seed_y), "voxel_x": int(seed_x),
            "sigma_mm": sigma_mm,
            "diameter_mm": diameter_mm,

            # --- axis-aligned bounding box (used by this stage's own NMS) ---
            "z_min": int(z_min), "y_min": int(y_min), "x_min": int(x_min),
            "z_max": int(z_max), "y_max": int(y_max), "x_max": int(x_max),

            # --- triage priority only, NOT a malignancy score ---
            "candidate_score": float(candidate_score),

            # --- diagnostic / auditing features ---
            "volume_mm3": float(volume_mm3),
            "extent_z_mm": float(extent_z_mm),
            "extent_y_mm": float(extent_y_mm),
            "extent_x_mm": float(extent_x_mm),
            "mean_hu": mean_hu,
            "std_hu": std_hu,
            "mean_contrast_hu": mean_contrast,
            "max_contrast_hu": max_contrast,
            "fill_ratio": float(fill_ratio),
            "irregularity": float(irregularity),
            "elongation": float(elongation),
            "boundary_contrast_hu": boundary_mean_contrast,
            "boundary_positive_fraction": positive_boundary_fraction,
            "occupied_slices": occupied_slices,
            "persistence": persistence,
        })

    # ---- cap raw candidate count, keeping highest-priority first ----

    candidates.sort(key=lambda c: c["candidate_score"], reverse=True)

    if len(candidates) > max_raw_candidates:
        print(
            f"[warn] {len(candidates)} characteristic candidates "
            f"exceed --max-raw-candidates ({max_raw_candidates}); "
            f"keeping the highest-priority {max_raw_candidates}."
        )
        candidates = candidates[:max_raw_candidates]

    return candidates


# ---------------------------------------------------------------------------
# 2. Non-maximum suppression
# ---------------------------------------------------------------------------

def _bbox_iou(a, b):
    """Axis-aligned bounding-box IoU in voxel space (unitless ratio)."""

    z1, z2 = max(a["z_min"], b["z_min"]), min(a["z_max"], b["z_max"])
    y1, y2 = max(a["y_min"], b["y_min"]), min(a["y_max"], b["y_max"])
    x1, x2 = max(a["x_min"], b["x_min"]), min(a["x_max"], b["x_max"])

    if z2 < z1 or y2 < y1 or x2 < x1:
        return 0.0

    inter = (z2 - z1 + 1) * (y2 - y1 + 1) * (x2 - x1 + 1)
    vol_a = (
        (a["z_max"] - a["z_min"] + 1)
        * (a["y_max"] - a["y_min"] + 1)
        * (a["x_max"] - a["x_min"] + 1)
    )
    vol_b = (
        (b["z_max"] - b["z_min"] + 1)
        * (b["y_max"] - b["y_min"] + 1)
        * (b["x_max"] - b["x_min"] + 1)
    )
    union = vol_a + vol_b - inter

    return inter / union if union > 0 else 0.0


def non_max_suppress_candidates(candidates, iou_threshold: float = 0.15):
    """
    Collapse heavily-overlapping candidate regions before they are
    written out. Because candidates here are irregular / non-
    spherical, this uses axis-aligned bounding-box IoU rather than a
    spherical-radius distance rule. For each cluster of overlapping
    candidates, the single highest candidate_score survives.

    IMPORTANT: this is the ONLY de-duplication step in the 04->05->06
    chain -- 05 does not re-run NMS, and positionally assigns
    candidate_id from whatever order candidates.json is written in.
    Run this before writing output, not after.
    """

    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: c["candidate_score"], reverse=True)
    kept = []

    for cand in ordered:
        if all(_bbox_iou(cand, k) <= iou_threshold for k in kept):
            kept.append(cand)

    return kept


# ---------------------------------------------------------------------------
# 3. Fixed-size patch extraction (convenience output, not required by 05)
# ---------------------------------------------------------------------------

def extract_candidate_patches(
    volume_hu: np.ndarray,
    candidates,
    spacing_zyx,
    out_dir: str,
    patch_size_mm: float = 50.0,
    patch_shape: int = 64,
    pad_value: float = AIR_HU,
):
    """
    Crop a FIXED-size patch around each candidate's seed voxel from
    the real (unblanked) volume, resample it to a
    (patch_shape, patch_shape, patch_shape) voxel cube, and save it to
    <out_dir>/patches/candidate_XXXX.npy.

    This is a convenience output for direct inspection/debugging. 05
    replaces this fixed box with an adaptive, region-grown crop sized
    to each nodule's real physical extent and does NOT read these
    files -- only candidates.json/csv matter for the 04->05 handoff.

    Returns
    -------
    list[str]
        Relative path (from out_dir) of each saved patch file, in the
        same order as `candidates`, for attaching as "patch_file".
    """

    patches_dir = os.path.join(out_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    sz, sy, sx = spacing_zyx
    half_extent_vox_zyx = (
        max(1, int(round((patch_size_mm / 2.0) / sz))),
        max(1, int(round((patch_size_mm / 2.0) / sy))),
        max(1, int(round((patch_size_mm / 2.0) / sx))),
    )
    target_shape_zyx = (patch_shape, patch_shape, patch_shape)

    patch_files = []

    for idx, cand in enumerate(candidates):
        center_zyx = (cand["voxel_z"], cand["voxel_y"], cand["voxel_x"])
        raw_patch, _offset = extract_patch(
            volume_hu, center_zyx, half_extent_vox_zyx, pad_value,
        )
        patch = resample_patch(raw_patch, target_shape_zyx, pad_value)

        filename = f"candidate_{idx:04d}.npy"
        np.save(os.path.join(patches_dir, filename), patch)
        patch_files.append(os.path.join("patches", filename))

    return patch_files


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def detect_and_patch(
    masked_dir: str,
    out_dir: str,
    min_diameter_mm: float = 4.0,
    max_diameter_mm: float = 40.0,
    background_sigma_mm: float = 5.0,
    min_contrast_hu: float = 40.0,
    min_component_voxels: int = 20,
    max_raw_candidates: int = 8000,
    nms_iou_threshold: float = 0.15,
    patch_size_mm: float = 50.0,
    patch_shape: int = 64,
):
    print(f"[info] Loading volume/lung_mask from '{masked_dir}'...")
    volume_hu, lung_mask, meta = load_masked_output(masked_dir)
    spacing_zyx = get_spacing_zyx(meta)

    # Detection runs on a blanked copy so anatomy outside the lung
    # can't seed a candidate; patches are cropped from real volume_hu
    # below. See module docstring's "Correctness notes".
    volume_hu_masked = np.where(lung_mask, volume_hu, AIR_HU).astype(np.float32)

    candidates = characteristic_candidate_detect(
        volume_hu_masked, lung_mask, spacing_zyx,
        min_diameter_mm=min_diameter_mm,
        max_diameter_mm=max_diameter_mm,
        background_sigma_mm=background_sigma_mm,
        min_contrast_hu=min_contrast_hu,
        min_component_voxels=min_component_voxels,
        max_raw_candidates=max_raw_candidates,
    )
    num_raw = len(candidates)
    print(f"[info] {num_raw} raw candidates detected.")

    candidates = non_max_suppress_candidates(candidates, iou_threshold=nms_iou_threshold)
    print(f"[info] {len(candidates)} candidates survive NMS "
          f"(iou_threshold={nms_iou_threshold}).")

    os.makedirs(out_dir, exist_ok=True)

    if candidates:
        patch_files = extract_candidate_patches(
            volume_hu, candidates, spacing_zyx, out_dir,
            patch_size_mm=patch_size_mm, patch_shape=patch_shape,
        )
        for cand, patch_file in zip(candidates, patch_files):
            cand["patch_file"] = patch_file
    else:
        patch_files = []

    params = {
        "min_diameter_mm": min_diameter_mm,
        "max_diameter_mm": max_diameter_mm,
        "background_sigma_mm": background_sigma_mm,
        "min_contrast_hu": min_contrast_hu,
        "min_component_voxels": min_component_voxels,
        "max_raw_candidates": max_raw_candidates,
        "nms_iou_threshold": nms_iou_threshold,
        "patch_size_mm": patch_size_mm,
        "patch_shape": patch_shape,
    }

    csv_path = os.path.join(out_dir, "candidates.csv")
    fieldnames = [
        "voxel_z", "voxel_y", "voxel_x", "sigma_mm", "diameter_mm",
        "candidate_score",
        "z_min", "y_min", "x_min", "z_max", "y_max", "x_max",
        "volume_mm3", "extent_z_mm", "extent_y_mm", "extent_x_mm",
        "mean_hu", "std_hu", "mean_contrast_hu", "max_contrast_hu",
        "fill_ratio", "irregularity", "elongation",
        "boundary_contrast_hu", "boundary_positive_fraction",
        "occupied_slices", "persistence", "patch_file",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)

    with open(os.path.join(out_dir, "candidates.json"), "w") as f:
        json.dump({"params": params, "candidates": candidates}, f, indent=2)

    out_meta = dict(meta)
    out_meta["detect_and_patch_params"] = params
    out_meta["num_raw_candidates"] = num_raw
    out_meta["num_candidates_after_nms"] = len(candidates)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(out_meta, f, indent=2)

    print(f"[done] Wrote candidates.csv, candidates.json, meta.json, and "
          f"{len(patch_files)} patch(es) -> '{out_dir}'")

    return candidates


def parse_args():
    parser = argparse.ArgumentParser(
        description="STEP 4: characteristic-based candidate detection "
        "(not spherical LoG) + fixed-size patch extraction."
    )
    parser.add_argument(
        "masked_dir",
        help="Directory containing volume_hu.npy / lung_mask.npy / "
        "meta.json (the --out-dir from 02_mask_and_crop.py).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to write candidates.csv/json and patches/ "
        "(default: '<masked_dir>_candidates').",
    )
    parser.add_argument(
        "--min-diameter-mm", type=float, default=4.0,
        help="Reject candidate regions smaller than this in every "
        "axis (default: 4.0).",
    )
    parser.add_argument(
        "--max-diameter-mm", type=float, default=40.0,
        help="Used only for candidate_score normalization and the "
        "8x sanity-bound reject on component volume, not a hard cap "
        "on candidate size (default: 40.0).",
    )
    parser.add_argument(
        "--background-sigma-mm", type=float, default=5.0,
        help="Gaussian smoothing radius (mm) used to estimate local "
        "lung background for the contrast map (default: 5.0).",
    )
    parser.add_argument(
        "--min-contrast-hu", type=float, default=40.0,
        help="Minimum local contrast (HU above smoothed background) "
        "to be treated as foreground via the contrast mechanism "
        "(default: 40.0).",
    )
    parser.add_argument(
        "--min-component-voxels", type=int, default=20,
        help="Reject connected components smaller than this many "
        "voxels (default: 20).",
    )
    parser.add_argument(
        "--max-raw-candidates", type=int, default=8000,
        help="Cap on raw candidates kept before NMS, highest "
        "candidate_score first (default: 8000).",
    )
    parser.add_argument(
        "--nms-iou-threshold", type=float, default=0.15,
        help="Bounding-box IoU above which overlapping candidates "
        "are treated as duplicates and collapsed to the highest-"
        "scoring one (default: 0.15).",
    )
    parser.add_argument(
        "--patch-size-mm", type=float, default=50.0,
        help="Physical side length (mm) of the FIXED-size window "
        "cropped around each candidate before resampling (default: "
        "50.0).",
    )
    parser.add_argument(
        "--patch-shape", type=int, default=64,
        help="Output patch side length in VOXELS after resampling "
        "(patch is patch_shape^3) (default: 64, matching "
        "cir_multihead_pipeline.PATCH_SIZE used downstream in 06 -- "
        "keep 05's --patch-shape consistent with whatever you use "
        "here if you rely on these patches directly).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.masked_dir.rstrip("/\\") + "_candidates")
    detect_and_patch(
        args.masked_dir, out_dir,
        min_diameter_mm=args.min_diameter_mm,
        max_diameter_mm=args.max_diameter_mm,
        background_sigma_mm=args.background_sigma_mm,
        min_contrast_hu=args.min_contrast_hu,
        min_component_voxels=args.min_component_voxels,
        max_raw_candidates=args.max_raw_candidates,
        nms_iou_threshold=args.nms_iou_threshold,
        patch_size_mm=args.patch_size_mm,
        patch_shape=args.patch_shape,
    )


if __name__ == "__main__":
    main()