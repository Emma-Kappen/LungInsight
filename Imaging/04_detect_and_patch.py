"""
04_detect_and_patch.py

STEP 4 (candidate detection + patch extraction), extending the pipeline:
    01_dicom_to_hu.py       -> DICOM -> HU volume
    02_mask_and_crop.py     -> lung segmentation + non-lung blanking + Z-crop
    03_visualize.py         -> viewing
    04_detect_and_patch.py  <- this file: nodule CANDIDATE detection
                               (multi-scale 3D LoG) + fixed-size patch
                               extraction for a downstream classifier

Loads volume_hu.npy / volume_hu_masked.npy / lung_mask.npy / meta.json
produced by 02_mask_and_crop.py, finds rounded-blob candidates via a
multi-scale 3D Laplacian-of-Gaussian (LoG), suppresses duplicate/
overlapping detections, and cuts a fixed-size patch around each
surviving candidate for a downstream classifier (e.g. the SE-ResNet3D
head).

=== Why LoG, and why on volume_hu_masked rather than volume_hu ===

Pulmonary nodules are, physically, roughly round, soft-tissue-density
blobs (rest of typical nodule density: roughly -700 to +200 HU)
embedded in much less dense aerated lung parenchyma (roughly -950 to
-700 HU). That density step is exactly the "bright blob on a dark
background" case Laplacian-of-Gaussian blob detection is built for: a
LoG response is strongly NEGATIVE at the center of a bright blob whose
physical size matches the filter's scale.

Detection runs on volume_hu_masked.npy (chest wall / ribs / mediastinum
already blanked to air HU by step 2) rather than raw volume_hu.npy, so
the rib cage and mediastinal soft tissue -- which are also "bright
blobs relative to their surroundings" -- can't generate candidates in
the first place. lung_mask.npy is used as a second, independent gate
on top of that (candidates must fall inside the mask) as a belt-and-
suspenders check, since the masked volume alone doesn't stop a
candidate centered exactly on the mask boundary.

Patch EXTRACTION, however, reads from volume_hu.npy (real, unblanked
HU), not volume_hu_masked.npy. A classifier benefits from seeing a
nodule's true local context (vessels, pleura, wall) -- the blanking in
step 2 exists to keep candidate DETECTION from tripping on chest-wall
anatomy, not to hide real anatomy from the classifier that looks at a
confirmed candidate afterward.

=== Physical (mm) scale-space, not voxel scale-space ===

CT voxels are usually anisotropic (slice spacing != in-plane pixel
spacing). A LoG filter built directly in voxel units would detect
differently-shaped-in-mm blobs depending on scan protocol. Instead,
every scale is defined in physical mm (a target nodule DIAMETER) and
converted to a per-axis voxel sigma using the actual mm spacing from
meta.json -- the same anisotropic-structuring-element idea
02_mask_and_crop.py already uses for its closing/dilation steps.

=== Why a single fixed --response-threshold doesn't work (fixed here) ===

An earlier version of this script used one hand-picked absolute
`--response-threshold` (e.g. -6.0) applied uniformly at every scale.
That fails for a reason that has NOTHING to do with unit/normalization
bugs (verified directly: switching the sigma^2 normalization term
between physical-mm and voxel units barely moved the numbers below) --
it's simpler and more fundamental than that. A finely-scaled LoG
(matched to a small nodule diameter) applies only light Gaussian
smoothing, so it still resolves ordinary parenchymal texture, small
vessel cross-sections, and scanner noise -- all of which produce
strong local minima in the response, in huge numbers. A coarsely-
scaled LoG (matched to a large nodule diameter) smooths heavily enough
that essentially all of that fine texture is averaged away, leaving
only genuinely large, coherent structures -- so it naturally produces
far fewer candidates. On a real scan this showed up as almost 300x
more raw candidates at the finest scale (4mm) than the coarsest (30mm)
-- confirmed independently on synthetic pure-noise lung tissue with NO
embedded nodules at all, which alone produced thousands of "candidates"
under the old fixed threshold. No single magnitude can be simultaneously
correct for both regimes, because the *noise floor itself* is scale-
dependent, not just the signal.

The fix: `--response-threshold` now defaults to None, meaning the
threshold is derived AUTOMATICALLY, SEPARATELY AT EACH SCALE, as the
`--auto-threshold-percentile`-th percentile of that scale's own in-
lung response distribution (default: the most extreme 0.5%). This
adapts to each scale's own noise floor instead of comparing every
scale to one shared number that's only meaningful for one of them.
You can still pass `--response-threshold` explicitly to force one
absolute value everywhere (kept for backward compatibility / manual
tuning), but per-scale response statistics are now always printed so
that number, if used, is chosen with the actual distribution in view
rather than blind. A hard `--max-raw-candidates` cap (new) also
protects downstream NMS from ever having to process an unbounded list,
regardless of how the threshold is chosen.

=== Correctness notes (mirroring the things that bit this pipeline before) ===

  * HU is read AS-IS from volume_hu.npy / volume_hu_masked.npy. Both
    already had RescaleSlope/RescaleIntercept applied exactly once, in
    step 1. This script never re-applies any rescale -- doing so here
    would silently double-apply it.
  * All arrays are consistently indexed (Z, Y, X), matching every
    other script in this pipeline. Scale/sigma tuples, patch half-
    extents, and spacing tuples are ALWAYS built and consumed in that
    same (z, y, x) order -- mixing that order with meta.json's
    pixel_spacing_mm (which is [row/Y spacing, column/X spacing], the
    DICOM PixelSpacing convention) is exactly the kind of bug that's
    easy to introduce silently, so every place that assembles a
    spacing tuple is commented with which axis is which.
  * Non-max suppression is done with a k-d tree (scipy.spatial.cKDTree)
    instead of an all-pairs distance matrix. A LoG response volume can
    easily produce several thousand raw local-minima candidates before
    suppression; an O(n^2) all-pairs NMS is fine at a few hundred
    candidates and silently becomes the runtime bottleneck once you're
    at a few thousand. The k-d tree approach below is O(n log n)
    average case.
  * Every tunable default below is a real, used default -- each one is
    read from argparse's `default=` and flows through function
    parameters (never hardcoded again deeper in the call chain, and
    never silently shadowed by an inner function re-defining it).

Usage:
    python 04_detect_and_patch.py output/LIDC-IDRI-0001_masked \
        --out-dir output/LIDC-IDRI-0001_candidates \
        --min-diameter-mm 4 --max-diameter-mm 30 --num-scales 8 \
        --nms-radius-mm 6 --patch-size-mm 32 --patch-shape 32

    # Force one fixed absolute threshold everywhere instead of the
    # per-scale auto-calibration (check the printed per-scale response
    # percentiles first so this number is informed, not a guess):
    python 04_detect_and_patch.py output/LIDC-IDRI-0001_masked \
        --response-threshold -30.0 ...

Outputs (written to --out-dir):
    candidates.csv        -> one row per surviving candidate: voxel
                              (z,y,x), approx world mm (z,y,x),
                              diameter_mm, sigma_mm, response,
                              scale_index
    candidates.json        -> same candidates + the detection params
                              used (so results are reproducible)
    patches/patch_XXXX.npy -> float32 HU array, shape
                              (patch_shape, patch_shape, patch_shape),
                              one per candidate in candidates.csv/json
                              (patch_XXXX index matches candidate id)
    meta.json               -> copy of the input meta.json plus
                              detection parameters and candidate count
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

try:
    from scipy import ndimage
    from scipy.spatial import cKDTree
except ImportError:
    print(
        "scipy is required for candidate detection. Install it with:\n"
        "    pip install scipy --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(1)


AIR_HU = -1000.0

# Same simplification 02_mask_and_crop.py already bakes into origin_mm
# when it Z-crops (it defaults meta["_z_step_sign"] to -1.0, since that
# key never survives being written to meta.json): world Z is assumed
# to DECREASE as voxel Z-index increases (the common superior->inferior
# scan order). This only affects the informational world-mm columns in
# candidates.csv -- voxel-space columns (used for patch extraction) are
# unaffected by this assumption.
Z_WORLD_STEP_SIGN = -1.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_masked_output(masked_dir: str):
    """Load volume_hu.npy, volume_hu_masked.npy, lung_mask.npy, meta.json
    written by 02_mask_and_crop.py."""
    paths = {
        "volume_hu": os.path.join(masked_dir, "volume_hu.npy"),
        "volume_hu_masked": os.path.join(masked_dir, "volume_hu_masked.npy"),
        "lung_mask": os.path.join(masked_dir, "lung_mask.npy"),
        "meta": os.path.join(masked_dir, "meta.json"),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. Run 02_mask_and_crop.py first, and "
                f"pass its --out-dir here."
            )

    volume_hu = np.load(paths["volume_hu"]).astype(np.float32, copy=False)
    volume_hu_masked = np.load(paths["volume_hu_masked"]).astype(np.float32, copy=False)
    lung_mask = np.load(paths["lung_mask"]).astype(bool, copy=False)
    with open(paths["meta"]) as f:
        meta = json.load(f)

    if volume_hu.shape != volume_hu_masked.shape or volume_hu.shape != lung_mask.shape:
        raise ValueError(
            f"Shape mismatch in '{masked_dir}': volume_hu {volume_hu.shape}, "
            f"volume_hu_masked {volume_hu_masked.shape}, lung_mask "
            f"{lung_mask.shape}. These should all match -- re-run "
            f"02_mask_and_crop.py."
        )

    return volume_hu, volume_hu_masked, lung_mask, meta


def get_spacing_zyx(meta: dict):
    """
    Returns (slice_spacing_mm, y_spacing_mm, x_spacing_mm), i.e. mm
    spacing in the SAME (Z, Y, X) axis order every array in this
    pipeline uses.

    meta["pixel_spacing_mm"] is DICOM PixelSpacing = [row spacing,
    column spacing] = [Y spacing, X spacing] -- NOT (Z, Y, X). Only
    meta["slice_spacing_mm"] is the Z spacing. Mixing these up (e.g.
    treating pixel_spacing_mm[0] as a Z value) silently produces
    correctly-shaped but wrongly-scaled sigmas and patches, so this is
    the single place that assembly happens.
    """
    slice_spacing_mm = float(meta.get("slice_spacing_mm", 1.0) or 1.0)
    pixel_spacing_mm = meta.get("pixel_spacing_mm", [1.0, 1.0])
    y_spacing_mm = float(pixel_spacing_mm[0])
    x_spacing_mm = float(pixel_spacing_mm[1])
    return slice_spacing_mm, y_spacing_mm, x_spacing_mm


# ---------------------------------------------------------------------------
# Multi-scale 3D LoG candidate detection
# ---------------------------------------------------------------------------


def compute_log_response(volume_hu: np.ndarray, sigma_vox_zyx, sigma_mm: float) -> np.ndarray:
    """
    Scale-normalized 3D Laplacian-of-Gaussian response.

    scipy.ndimage.gaussian_laplace smooths with an anisotropic Gaussian
    (sigma given per axis, in VOXELS) and then applies the Laplacian.
    A bright blob (nodule) on a dark background (aerated lung) produces
    a strongly NEGATIVE response at the blob center when the filter
    scale matches the blob's physical size.

    Raw LoG amplitude shrinks as sigma grows (the Laplacian scales
    roughly as 1/sigma^2), so responses at different scales aren't
    directly comparable unless normalized. Multiplying by sigma_mm**2
    (the standard "gamma-normalized derivative", gamma=2 for the
    Laplacian) puts every scale's response back on a comparable
    footing, which is what lets us pick a single --response-threshold
    that works across all scales at once instead of one per scale.
    """
    response = ndimage.gaussian_laplace(volume_hu, sigma=sigma_vox_zyx, mode="nearest")
    return response * (sigma_mm ** 2)


def find_local_minima(response: np.ndarray, lung_mask: np.ndarray,
                       size_zyx, threshold: float):
    """
    Local minima of `response` (strong negative = blob-like), gated to
    lung_mask and to values at or below `threshold`.

    Uses ndimage.minimum_filter's separable `size=` window (an axis-
    aligned box) rather than an arbitrary ellipsoidal `footprint=`.
    scipy's arbitrary-footprint path uses a much more expensive
    generic rank-filter algorithm that can blow up in memory once the
    window gets large (this bit large-diameter scales here -- a 30mm-
    radius ellipsoid footprint is roughly 25x43x43 voxels, which
    reliably triggered a MemoryError). A box window is a coarser
    approximation of "roughly this blob's own size" than an ellipsoid,
    but that's fine: this filter only exists to avoid reporting a
    cluster of adjacent near-duplicate peaks for one blob AT THIS
    SINGLE SCALE -- the real, precise, physically-correct deduplication
    (including across scales) is the KD-tree NMS pass afterward, which
    already operates in real mm space regardless of this window's shape.

    Returns an (N, 3) int array of (z, y, x) voxel coordinates and an
    (N,) float array of response values at those coordinates.
    """
    min_filtered = ndimage.minimum_filter(response, size=size_zyx, mode="nearest")
    is_local_min = (response == min_filtered) & lung_mask & (response <= threshold)
    coords = np.argwhere(is_local_min)
    values = response[is_local_min]
    return coords, values


def multi_scale_detect(volume_hu_masked: np.ndarray, lung_mask: np.ndarray,
                        spacing_zyx, min_diameter_mm: float, max_diameter_mm: float,
                        num_scales: int, response_threshold=None,
                        auto_threshold_percentile: float = 0.5,
                        min_sigma_voxels: float = 0.8,
                        max_raw_candidates: int = 8000):
    """
    Runs LoG detection at `num_scales` log-spaced diameters between
    min_diameter_mm and max_diameter_mm (inclusive), pooling raw
    (pre-NMS) candidates from every scale into one list.

    === Threshold calibration (this is the part that used to break) ===

    If `response_threshold` is given explicitly, it's applied as one
    fixed absolute value at every scale. DON'T reach for this as the
    first fix if you're getting too many/too few candidates -- see the
    module docstring for why a single fixed threshold structurally
    can't work well across scales (finer scales have a much noisier
    response floor than coarser ones purely from smoothing less, not
    from any unit/normalization bug). It's kept only for reproducing
    an exact prior run or for deliberate manual tuning once you've
    looked at the printed per-scale response percentiles.

    The default (`response_threshold=None`) instead derives a
    threshold SEPARATELY AT EACH SCALE: the `auto_threshold_percentile`
    -th percentile of that scale's own in-lung response values (default
    0.5, i.e. the most extreme 0.5% of that scale's in-lung responses
    qualify). This adapts to each scale's own noise floor automatically
    instead of comparing every scale against one shared number.

    `max_raw_candidates` is a hard safety cap on the TOTAL pooled
    candidate count across all scales (checked once at the end, after
    every scale has run) -- if exceeded, only the strongest
    `max_raw_candidates` (by response) are kept, and a warning is
    printed. This protects the NMS step from ever having to process an
    unbounded list, regardless of how permissive the threshold (auto
    or manual) turns out to be for a given scan.

    Returns a list of dicts, one per raw candidate:
        {z, y, x, response, diameter_mm, sigma_mm, scale_index}
    """
    slice_spacing_mm, y_spacing_mm, x_spacing_mm = spacing_zyx

    if num_scales < 1:
        raise ValueError("--num-scales must be >= 1")
    diameters_mm = (
        [min_diameter_mm] if num_scales == 1
        else np.geomspace(min_diameter_mm, max_diameter_mm, num_scales)
    )

    raw_candidates = []
    for scale_index, diameter_mm in enumerate(diameters_mm):
        diameter_mm = float(diameter_mm)
        radius_mm = diameter_mm / 2.0
        # Standard 3D LoG blob<->sigma relation: a blob of physical
        # radius r is detected most strongly at sigma = r / sqrt(3).
        sigma_mm = radius_mm / np.sqrt(3.0)

        # Convert the single physical sigma_mm into a per-axis VOXEL
        # sigma using this scan's actual (z, y, x) spacing -- this is
        # what makes the filter isotropic in mm despite anisotropic
        # voxels. Floored at min_sigma_voxels so a small nominal
        # diameter on coarse slice spacing can't produce a sub-voxel
        # kernel that degenerates into a noise amplifier (see module
        # docstring's aliasing note).
        sigma_vox_zyx = (
            max(sigma_mm / slice_spacing_mm, min_sigma_voxels),
            max(sigma_mm / y_spacing_mm, min_sigma_voxels),
            max(sigma_mm / x_spacing_mm, min_sigma_voxels),
        )

        response = compute_log_response(volume_hu_masked, sigma_vox_zyx, sigma_mm)

        # Non-max window sized to roughly the blob's own physical
        # diameter (odd voxel counts, box-shaped -- see
        # find_local_minima's docstring for why box instead of
        # ellipsoid), so we don't report multiple adjacent local
        # minima for what's really one blob at this scale (cross-scale
        # duplicates are handled separately by the NMS pass below).
        size_zyx = tuple(
            max(1, int(round(diameter_mm / spacing)) | 1)  # force odd
            for spacing in (slice_spacing_mm, y_spacing_mm, x_spacing_mm)
        )

        in_lung_response = response[lung_mask]
        response_percentiles = np.percentile(in_lung_response, [0.1, 0.5, 1.0, 5.0])

        if response_threshold is None:
            scale_threshold = float(np.percentile(in_lung_response, auto_threshold_percentile))
        else:
            scale_threshold = response_threshold

        coords, values = find_local_minima(response, lung_mask, size_zyx, scale_threshold)

        print(
            f"[info] scale {scale_index + 1}/{len(diameters_mm)}: "
            f"diameter={diameter_mm:.1f}mm sigma={sigma_mm:.2f}mm "
            f"threshold={scale_threshold:.2f}"
            f"{' (auto)' if response_threshold is None else ' (manual)'} "
            f"-> {len(coords)} raw candidates  "
            f"[in-lung response percentiles 0.1/0.5/1/5%%: "
            f"{response_percentiles[0]:.1f}/{response_percentiles[1]:.1f}/"
            f"{response_percentiles[2]:.1f}/{response_percentiles[3]:.1f}]"
        )

        for (z, y, x), val in zip(coords, values):
            raw_candidates.append({
                "z": int(z), "y": int(y), "x": int(x),
                "response": float(val),
                "diameter_mm": diameter_mm,
                "sigma_mm": sigma_mm,
                "scale_index": scale_index,
            })

    if len(raw_candidates) > max_raw_candidates:
        print(f"[warn] {len(raw_candidates)} pooled raw candidates exceed "
              f"--max-raw-candidates ({max_raw_candidates}); keeping only the "
              f"{max_raw_candidates} strongest (by response) before NMS. If "
              f"this triggers often, tighten --auto-threshold-percentile.")
        raw_candidates.sort(key=lambda c: c["response"])  # most negative first
        raw_candidates = raw_candidates[:max_raw_candidates]

    return raw_candidates


# ---------------------------------------------------------------------------
# Non-max suppression (k-d tree based -- O(n log n) average, not O(n^2))
# ---------------------------------------------------------------------------

def nms_candidates(raw_candidates, spacing_zyx, nms_radius_mm: float):
    """
    Greedy non-max suppression across all scales at once: strongest
    (most negative) response wins, then anything within nms_radius_mm
    of an accepted candidate is suppressed, regardless of which scale
    produced it.

    Uses a k-d tree over physical-mm coordinates (voxel coords scaled
    by this scan's actual spacing) so each suppression step is a
    radius query instead of a scan over every remaining candidate --
    the fix for the quadratic-runtime version of this that doesn't
    scale past a few hundred raw candidates.
    """
    if not raw_candidates:
        return []

    slice_spacing_mm, y_spacing_mm, x_spacing_mm = spacing_zyx
    zyx_vox = np.array([[c["z"], c["y"], c["x"]] for c in raw_candidates], dtype=np.float64)
    # Physical (relative) mm coordinates -- only used for distances
    # within this function, so an absolute world origin isn't needed.
    zyx_mm = zyx_vox * np.array([slice_spacing_mm, y_spacing_mm, x_spacing_mm])
    responses = np.array([c["response"] for c in raw_candidates], dtype=np.float64)

    tree = cKDTree(zyx_mm)
    order = np.argsort(responses)  # ascending: most negative (strongest) first

    suppressed = np.zeros(len(raw_candidates), dtype=bool)
    kept_indices = []
    for idx in order:
        if suppressed[idx]:
            continue
        kept_indices.append(idx)
        neighbor_indices = tree.query_ball_point(zyx_mm[idx], r=nms_radius_mm)
        suppressed[neighbor_indices] = True

    return [raw_candidates[i] for i in kept_indices]


# ---------------------------------------------------------------------------
# World-mm coordinates (informational -- see Z_WORLD_STEP_SIGN note above)
# ---------------------------------------------------------------------------

def voxel_to_world_mm(z, y, x, meta, spacing_zyx):
    slice_spacing_mm, y_spacing_mm, x_spacing_mm = spacing_zyx
    origin = meta.get("origin_mm", [0.0, 0.0, 0.0])  # DICOM order: [x, y, z]
    origin_x, origin_y, origin_z = origin[0], origin[1], origin[2]
    world_x = origin_x + x * x_spacing_mm
    world_y = origin_y + y * y_spacing_mm
    world_z = origin_z + z * Z_WORLD_STEP_SIGN * slice_spacing_mm
    return world_z, world_y, world_x


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patch(volume_hu: np.ndarray, center_zyx, half_extent_vox_zyx,
                   pad_value: float = AIR_HU):
    """
    Cuts a (2*hz, 2*hy, 2*hx)-voxel box centered on center_zyx out of
    volume_hu, padding with `pad_value` (physical air HU) wherever the
    box would run outside the volume -- e.g. a candidate near the top/
    bottom of a Z-cropped volume.
    """
    cz, cy, cx = center_zyx
    hz, hy, hx = half_extent_vox_zyx
    out_shape = (2 * hz, 2 * hy, 2 * hx)
    patch = np.full(out_shape, pad_value, dtype=np.float32)

    z_lo, z_hi = cz - hz, cz + hz
    y_lo, y_hi = cy - hy, cy + hy
    x_lo, x_hi = cx - hx, cx + hx

    src_z_lo, src_z_hi = max(0, z_lo), min(volume_hu.shape[0], z_hi)
    src_y_lo, src_y_hi = max(0, y_lo), min(volume_hu.shape[1], y_hi)
    src_x_lo, src_x_hi = max(0, x_lo), min(volume_hu.shape[2], x_hi)

    if src_z_lo >= src_z_hi or src_y_lo >= src_y_hi or src_x_lo >= src_x_hi:
        # Candidate center is entirely outside the volume -- shouldn't
        # happen since candidates come from inside the volume, but
        # guard against it rather than raising on a broadcast error.
        return patch

    dst_z_lo, dst_y_lo, dst_x_lo = src_z_lo - z_lo, src_y_lo - y_lo, src_x_lo - x_lo
    dst_z_hi = dst_z_lo + (src_z_hi - src_z_lo)
    dst_y_hi = dst_y_lo + (src_y_hi - src_y_lo)
    dst_x_hi = dst_x_lo + (src_x_hi - src_x_lo)

    patch[dst_z_lo:dst_z_hi, dst_y_lo:dst_y_hi, dst_x_lo:dst_x_hi] = volume_hu[
        src_z_lo:src_z_hi, src_y_lo:src_y_hi, src_x_lo:src_x_hi
    ]
    return patch


def resample_patch(patch: np.ndarray, target_shape_zyx, pad_value: float = AIR_HU):
    """
    Resample a (native-spacing) patch to a fixed isotropic voxel grid
    of shape target_shape_zyx, using linear interpolation. This is
    what lets patches cut from scans with different slice spacing all
    come out the same voxel shape for a downstream fixed-input-size
    classifier.
    """
    zoom_factors = [t / s for t, s in zip(target_shape_zyx, patch.shape)]
    resampled = ndimage.zoom(patch, zoom_factors, order=1, mode="nearest")
    # zoom() output size can be off by one voxel from rounding; pad/crop
    # to the exact requested shape so every saved patch is identical.
    out = np.full(target_shape_zyx, pad_value, dtype=np.float32)
    slices_src = tuple(slice(0, min(s, t)) for s, t in zip(resampled.shape, target_shape_zyx))
    slices_dst = tuple(slice(0, min(s, t)) for s, t in zip(resampled.shape, target_shape_zyx))
    out[slices_dst] = resampled[slices_src]
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def detect_and_patch(
    masked_dir: str,
    out_dir: str,
    min_diameter_mm: float = 4.0,
    max_diameter_mm: float = 30.0,
    num_scales: int = 8,
    response_threshold=None,
    auto_threshold_percentile: float = 0.5,
    min_sigma_voxels: float = 0.8,
    max_raw_candidates: int = 8000,
    nms_radius_mm: float = 6.0,
    patch_size_mm: float = 32.0,
    patch_shape: int = 32,
    resample: bool = True,
):
    print(f"[info] Loading masked/cropped volume from '{masked_dir}'...")
    volume_hu, volume_hu_masked, lung_mask, meta = load_masked_output(masked_dir)
    spacing_zyx = get_spacing_zyx(meta)

    if not lung_mask.any():
        print("[warn] lung_mask is empty -- no candidates are possible. "
              "Check that 02_mask_and_crop.py segmented this scan correctly.")

    threshold_desc = (
        f"fixed threshold {response_threshold:.2f} (manual override)" if response_threshold is not None
        else f"auto-calibrated per scale (top {auto_threshold_percentile:.2g}% of each "
             f"scale's own in-lung response)"
    )
    print(
        f"[info] Detecting candidates: diameters "
        f"[{min_diameter_mm:.1f}, {max_diameter_mm:.1f}]mm across "
        f"{num_scales} scales, {threshold_desc}..."
    )
    raw_candidates = multi_scale_detect(
        volume_hu_masked, lung_mask, spacing_zyx,
        min_diameter_mm=min_diameter_mm,
        max_diameter_mm=max_diameter_mm,
        num_scales=num_scales,
        response_threshold=response_threshold,
        auto_threshold_percentile=auto_threshold_percentile,
        min_sigma_voxels=min_sigma_voxels,
        max_raw_candidates=max_raw_candidates,
    )
    print(f"[info] {len(raw_candidates)} raw candidates before NMS.")

    print(f"[info] Running k-d-tree NMS (radius {nms_radius_mm:.1f}mm)...")
    candidates = nms_candidates(raw_candidates, spacing_zyx, nms_radius_mm)
    # Strongest (most negative response) first in the saved output.
    candidates.sort(key=lambda c: c["response"])
    print(f"[info] {len(candidates)} candidates survive NMS.")

    os.makedirs(out_dir, exist_ok=True)
    patches_dir = os.path.join(out_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    slice_spacing_mm, y_spacing_mm, x_spacing_mm = spacing_zyx
    half_extent_vox_zyx = (
        max(1, int(round((patch_size_mm / 2.0) / slice_spacing_mm))),
        max(1, int(round((patch_size_mm / 2.0) / y_spacing_mm))),
        max(1, int(round((patch_size_mm / 2.0) / x_spacing_mm))),
    )
    target_shape_zyx = (patch_shape, patch_shape, patch_shape)

    rows = []
    for candidate_id, cand in enumerate(candidates):
        center_zyx = (cand["z"], cand["y"], cand["x"])
        patch = extract_patch(volume_hu, center_zyx, half_extent_vox_zyx)
        if resample:
            patch = resample_patch(patch, target_shape_zyx)
        patch_filename = f"patch_{candidate_id:04d}.npy"
        np.save(os.path.join(patches_dir, patch_filename), patch.astype(np.float32))

        world_z, world_y, world_x = voxel_to_world_mm(
            cand["z"], cand["y"], cand["x"], meta, spacing_zyx
        )
        rows.append({
            "candidate_id": candidate_id,
            "voxel_z": cand["z"], "voxel_y": cand["y"], "voxel_x": cand["x"],
            "world_z_mm": round(world_z, 2),
            "world_y_mm": round(world_y, 2),
            "world_x_mm": round(world_x, 2),
            "diameter_mm": round(cand["diameter_mm"], 2),
            "sigma_mm": round(cand["sigma_mm"], 3),
            "response": round(cand["response"], 4),
            "scale_index": cand["scale_index"],
            "patch_file": os.path.join("patches", patch_filename),
        })

    csv_path = os.path.join(out_dir, "candidates.csv")
    with open(csv_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write(
                "candidate_id,voxel_z,voxel_y,voxel_x,world_z_mm,world_y_mm,"
                "world_x_mm,diameter_mm,sigma_mm,response,scale_index,patch_file\n"
            )

    detection_params = {
        "min_diameter_mm": min_diameter_mm,
        "max_diameter_mm": max_diameter_mm,
        "num_scales": num_scales,
        "response_threshold": response_threshold,
        "auto_threshold_percentile": auto_threshold_percentile if response_threshold is None else None,
        "min_sigma_voxels": min_sigma_voxels,
        "max_raw_candidates": max_raw_candidates,
        "nms_radius_mm": nms_radius_mm,
        "patch_size_mm": patch_size_mm,
        "patch_shape": patch_shape,
        "resampled_to_isotropic": resample,
        "z_world_step_sign_assumed": Z_WORLD_STEP_SIGN,
    }
    with open(os.path.join(out_dir, "candidates.json"), "w") as f:
        json.dump({"detection_params": detection_params, "candidates": rows}, f, indent=2)

    out_meta = dict(meta)
    out_meta["detect_and_patch_params"] = detection_params
    out_meta["num_raw_candidates"] = len(raw_candidates)
    out_meta["num_candidates_after_nms"] = len(candidates)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(out_meta, f, indent=2)

    print(f"[done] Wrote candidates.csv, candidates.json, meta.json, "
          f"and {len(candidates)} patch(es) -> '{patches_dir}'")
    print(f"[done] Next: feed patches/*.npy into the classifier "
          f"(patch shape: {target_shape_zyx if resample else 'native spacing, per-candidate'}).")

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="STEP 4: Detect nodule candidates (multi-scale 3D LoG) "
        "and extract fixed-size patches, using output from "
        "02_mask_and_crop.py."
    )
    parser.add_argument(
        "masked_dir",
        help="Directory containing volume_hu.npy / volume_hu_masked.npy / "
        "lung_mask.npy / meta.json (the --out-dir from 02_mask_and_crop.py).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write candidates.csv/json and patches/ "
        "(default: '<masked_dir>_candidates').",
    )
    parser.add_argument(
        "--min-diameter-mm", type=float, default=4.0,
        help="Smallest nodule diameter (mm) to search for (default: 4.0).",
    )
    parser.add_argument(
        "--max-diameter-mm", type=float, default=30.0,
        help="Largest nodule diameter (mm) to search for (default: 30.0).",
    )
    parser.add_argument(
        "--num-scales", type=int, default=8,
        help="Number of log-spaced diameters between --min-diameter-mm and "
        "--max-diameter-mm to run LoG at (default: 8).",
    )
    parser.add_argument(
        "--response-threshold", type=float, default=None,
        help="Force ONE fixed absolute LoG response threshold at every "
        "scale (more negative = stronger, more blob-like required). "
        "Default: None, meaning the threshold is derived AUTOMATICALLY "
        "AND SEPARATELY AT EACH SCALE from that scale's own in-lung "
        "response distribution (see --auto-threshold-percentile). A "
        "single fixed value here is scan- AND scale-dependent -- read "
        "the per-scale response percentiles this script prints before "
        "picking one, or you'll likely reproduce the runaway-candidate-"
        "count problem this default was changed to fix.",
    )
    parser.add_argument(
        "--auto-threshold-percentile", type=float, default=0.5,
        help="When --response-threshold is not given, keep the most "
        "extreme this-percent of each scale's own in-lung LoG response "
        "values as raw candidates (default: 0.5, i.e. the strongest "
        "0.5%% at each scale). Lower = stricter = fewer raw candidates.",
    )
    parser.add_argument(
        "--min-sigma-voxels", type=float, default=0.8,
        help="Floor applied to each axis's voxel sigma before filtering, "
        "so a small --min-diameter-mm combined with coarse slice "
        "spacing can't produce a sub-voxel LoG kernel that degenerates "
        "into a noise amplifier (default: 0.8).",
    )
    parser.add_argument(
        "--max-raw-candidates", type=int, default=8000,
        help="Hard cap on the total pooled raw candidate count (across "
        "all scales, combined) allowed into NMS -- keeps the strongest "
        "this-many by response if exceeded, regardless of threshold "
        "(default: 8000).",
    )
    parser.add_argument(
        "--nms-radius-mm", type=float, default=6.0,
        help="Suppress weaker candidates within this physical radius (mm) "
        "of a stronger one, across all scales (default: 6.0).",
    )
    parser.add_argument(
        "--patch-size-mm", type=float, default=32.0,
        help="Physical side length (mm) of the cube cut out of volume_hu.npy "
        "around each surviving candidate, before any resampling "
        "(default: 32.0).",
    )
    parser.add_argument(
        "--patch-shape", type=int, default=32,
        help="Output patch side length in VOXELS (patch is "
        "patch_shape^3). Only used when resampling is enabled "
        "(default: 32).",
    )
    parser.add_argument(
        "--no-resample", action="store_true",
        help="Skip isotropic resampling; save each patch at this scan's "
        "native voxel spacing instead (patch voxel shape will then vary "
        "with --patch-size-mm and this scan's spacing, not --patch-shape).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.masked_dir.rstrip("/\\") + "_candidates")
    detect_and_patch(
        args.masked_dir, out_dir,
        min_diameter_mm=args.min_diameter_mm,
        max_diameter_mm=args.max_diameter_mm,
        num_scales=args.num_scales,
        response_threshold=args.response_threshold,
        auto_threshold_percentile=args.auto_threshold_percentile,
        min_sigma_voxels=args.min_sigma_voxels,
        max_raw_candidates=args.max_raw_candidates,
        nms_radius_mm=args.nms_radius_mm,
        patch_size_mm=args.patch_size_mm,
        patch_shape=args.patch_shape,
        resample=not args.no_resample,
    )


if __name__ == "__main__":
    main()