"""
05_shape_filter_and_grow.py

STEP 5, extending the pipeline:
    01_dicom_to_hu.py           -> DICOM -> HU volume
    02_mask_and_crop.py         -> lung segmentation + non-lung blanking + Z-crop
    03_visualize.py             -> viewing
    04_detect_and_patch.py      -> multi-scale 3D LoG candidate detection +
                                    FIXED-size patch extraction
    05_shape_filter_and_grow.py <- this file: (a) reject candidates whose
                                    local shape is TUBULAR (vessel /
                                    bronchus / trachea cross-section)
                                    rather than round, and (b) replace
                                    04's fixed-size box with a region-
                                    GROWN, tightly-fitted crop that
                                    adapts to each surviving nodule's
                                    real physical size.

Loads candidates.csv/json produced by 04_detect_and_patch.py, plus
volume_hu.npy / lung_mask.npy / meta.json from 02_mask_and_crop.py's
output directory.

=== Why 04's LoG detector alone isn't enough ===

Laplacian-of-Gaussian responds to ANY locally round, dense-relative-to-
background blob. A nodule is one instance of that. So is a cross-
section through a pulmonary vessel or a small bronchus/bronchiole wall
-- CT slices cut tubular structures at an angle constantly, and any
near-perpendicular cut through a vessel looks exactly like a round
blob to a filter that only ever looks at a small local neighborhood.
04's own docstring documents the false-positive-rate problem this
causes; this script is the fix for the SHAPE-DISCRIMINATION half of
it. (04's per-scale auto-thresholding already fixed the separate
NOISE-FLOOR half of the false-positive problem -- these are two
independent causes of false positives, not the same bug twice.)

=== Part 1: Hessian eigenvalue shape filter (reject tubes) ===

At a true blob center, the local Hessian matrix (3x3 matrix of second
spatial derivatives) has three eigenvalues of comparable magnitude,
because the surface curves away similarly in every direction. At a
point on a tube's centerline, curvature is strong in the two
directions PERPENDICULAR to the tube's long axis, but nearly flat
ALONG the axis -- so one eigenvalue is close to zero while the other
two are large. This eigenvalue-ratio idea is the same principle behind
vesselness filters (e.g. Frangi/Sato) used elsewhere in CT vessel
segmentation; here it's used in the opposite direction, to REJECT
vessel/airway-like shapes rather than enhance them.

Concretely, for each candidate:
  1. Cut a small local patch around it from volume_hu, resample to an
     ISOTROPIC voxel grid (equal mm spacing on every axis). This is
     required, not cosmetic -- CT voxels are anisotropic, and Hessian
     eigenvalues are only comparable across axes if those axes are
     physically equal-sized. Skipping this would make an ordinary
     round nodule look artificially elongated along whichever axis
     has coarser native spacing (almost always Z).
  2. Compute the Hessian at that candidate's own matched sigma (reusing
     sigma_mm already stored in candidates.csv/json by 04 -- no need
     to re-derive it).
  3. Sort eigenvalues by ABSOLUTE magnitude: |lam1| <= |lam2| <= |lam3|.
  4. sphericity = |lam1| / |lam3|  (near 1.0 = isotropic/blob-like,
     near 0.0 = one flat axis = tube-like).
  5. Reject if sphericity < --min-sphericity, OR if fewer than 2 of
     the 3 eigenvalues are negative (negative curvature in every
     direction is what "locally brighter than its surroundings, in
     every direction" means -- a candidate that fails this isn't
     sitting on a real local blob-like maximum at all, regardless of
     its sphericity ratio).

This is a heuristic, not a certainty -- a nodule attached to a vessel
("juxtavascular") can score less spherical than an isolated one, and
--min-sphericity may need tuning once you have real LIDC-IDRI XML
malignancy/shape labels to check against. Treat --min-sphericity as a
starting point, not a validated constant.

=== Part 2: Region-growing GROWTH FUNCTION (adaptive crop) ===

04's patch extraction cuts a FIXED --patch-size-mm box around every
candidate regardless of the nodule's real size. That's a mismatch in
both directions: a 4mm micro-nodule occupies a tiny fraction of a
32mm box (mostly background after resampling), while a 30mm mass can
be clipped by the same fixed box. This part fixes that by actually
SEGMENTING each candidate before cropping:

  1. Cut a generous native-spacing search window around the candidate
     (--growth-search-radius-mm -- deliberately larger than any
     expected nodule, so growth has room to find the true boundary
     without hitting the window edge under normal conditions).
  2. Threshold that window with Otsu's method (bimodal: aerated lung
     parenchyma/air vs. denser soft-tissue blob), gated to lung_mask
     so growth can't leak into chest wall or mediastinum through a
     spot where the lung boundary is nearby.
  3. Label connected components (26-connectivity) and keep ONLY the
     component containing the candidate's own seed voxel -- this is
     what makes the crop specific to THIS candidate and not just
     "everything above threshold in the window."
  4. LEAK GUARD: if EITHER that component's equivalent diameter OR
     its longest bounding-box edge exceeds --max-grown-diameter-mm,
     the region growing almost certainly leaked into an attached
     vessel or the mediastinum through a thin bridge (a real nodule
     should not grow this large in any sense). Both checks are needed:
     volume/equivalent-diameter alone misses an elongated leak (a
     thin, long vessel segment can have the same VOLUME as a compact
     nodule despite being obviously not one), so the longest bounding-
     box edge is checked independently. When either check fires,
     growth is DISCARDED for that candidate and a safe
     fallback box (sized from the candidate's own detected
     diameter_mm from 04, plus --fallback-margin-mm) is used instead,
     and the candidate is flagged `leaked=True` in the output so you
     can audit how often this triggers.
  5. Take the grown component's tight bounding box, add
     --growth-margin-mm of context on every side, clip to the volume,
     and cut a NATIVE-spacing patch of that (variable, per-candidate)
     size from volume_hu -- real HU, unblanked, same rationale 04
     documents for why the classifier should see true local anatomy.
  6. Resample that variable-size patch to the fixed --patch-shape
     voxel cube every downstream classifier input needs (same
     resample_patch() approach as 04) -- so nodules of any real size
     still produce a uniformly-shaped tensor, just tightly zoomed to
     the nodule's own true extent instead of one shared physical box
     size for every candidate.

A useful side effect: the grown component's voxel count directly
gives an estimated volume_mm3 and equivalent_diameter_mm per
candidate (written to nodules.csv). This is a first building block
toward the volume-doubling-time (VDT) tracking on the project roadmap
-- it is NOT itself a longitudinal measurement (that still requires
matching this same nodule across two dated studies from a dataset
that actually has genuine time gaps, e.g. NLST), but the per-study
volume estimate this produces is exactly the quantity VDT is computed
from.

=== Correctness notes (mirroring the conventions established in 01-04) ===

  * All arrays and coordinate tuples are (Z, Y, X), matching 01-04.
    meta["pixel_spacing_mm"] is DICOM PixelSpacing = [Y spacing,
    X spacing]; meta["slice_spacing_mm"] is the Z spacing. Assembled
    into (z, y, x) order in exactly one place (get_spacing_zyx),
    copied verbatim from 04's own helper of the same name.
  * HU is read from volume_hu.npy AS-IS (already rescaled once, in
    step 1) -- never re-rescaled here.
  * This script does NOT import 04_detect_and_patch.py as a module
    (filenames starting with a digit aren't valid Python module
    names without importlib gymnastics) -- extract_patch/
    resample_patch equivalents are reimplemented here, deliberately
    kept behavior-identical to 04's versions.
  * Growth and shape analysis both read from volume_hu (real HU) and
    gate against lung_mask, never from volume_hu_masked -- the
    blanked volume exists to keep 04's DETECTION step from tripping
    on chest-wall anatomy; it's not needed here since candidates are
    already known, and using real HU avoids growth seeing an
    artificial air-HU cliff at the lung_mask boundary.

Usage:
    python 05_shape_filter_and_grow.py output/LIDC-IDRI-0001_masked \
        output/LIDC-IDRI-0001_candidates \
        --out-dir output/LIDC-IDRI-0001_nodules \
        --min-sphericity 0.35 --patch-shape 32

Outputs (written to --out-dir):
    nodules.csv          -> one row per candidate: shape-filter result
                             (sphericity, kept/rejected), growth result
                             (grown_ok, leaked, volume_mm3,
                             equivalent_diameter_mm, bbox), patch_file
    nodules.json          -> same rows + all params used (reproducible)
    patches/nodule_XXXX.npy -> float32 HU array, shape
                             (patch_shape, patch_shape, patch_shape),
                             one per candidate that SURVIVES the shape
                             filter (nodule_XXXX index matches
                             candidate_id, so rejected ids are simply
                             absent from patches/)
    masks/nodule_XXXX_mask.npy -> optional (--save-masks), bool grown
                             segmentation mask cropped to the same
                             pre-resample extent as the patch
    meta.json              -> copy of input meta.json + these params
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

try:
    from skimage.filters import threshold_otsu
except ImportError:
    print(
        "scikit-image is required. Install it with:\n"
        "    pip install scikit-image --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(1)


AIR_HU = -1000.0
# Safe fallback split between aerated lung parenchyma and soft-tissue-
# density structures (nodules, vessels, airway walls), used only when
# Otsu's method can't be computed (e.g. a degenerate, near-uniform
# local window) -- see grow_nodule_region().
FALLBACK_HU_THRESHOLD = -400.0

# 26-connectivity structuring element for 3D connected components.
STRUCT_26 = np.ones((3, 3, 3), dtype=bool)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_masked_output(masked_dir: str):
    """Load volume_hu.npy, lung_mask.npy, meta.json written by
    02_mask_and_crop.py."""
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


def load_candidates(candidates_dir: str):
    """Load candidates.json written by 04_detect_and_patch.py (has the
    same rows as candidates.csv but avoids re-parsing CSV strings)."""
    path = os.path.join(candidates_dir, "candidates.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run 04_detect_and_patch.py first, and "
            f"pass its --out-dir as candidates_dir here."
        )
    with open(path) as f:
        data = json.load(f)
    return data["candidates"]


def get_spacing_zyx(meta: dict):
    """(slice_spacing_mm, y_spacing_mm, x_spacing_mm) -- identical
    convention/derivation to 04_detect_and_patch.py's helper of the
    same name. See that file's docstring for why this assembly has to
    happen in exactly one place."""
    slice_spacing_mm = float(meta.get("slice_spacing_mm", 1.0) or 1.0)
    pixel_spacing_mm = meta.get("pixel_spacing_mm", [1.0, 1.0])
    y_spacing_mm = float(pixel_spacing_mm[0])
    x_spacing_mm = float(pixel_spacing_mm[1])
    return slice_spacing_mm, y_spacing_mm, x_spacing_mm


# ---------------------------------------------------------------------------
# Shared low-level helpers (native-spacing crop + isotropic resample)
# ---------------------------------------------------------------------------

def crop_native_window(volume: np.ndarray, center_zyx, half_extent_vox_zyx, pad_value: float):
    """
    Cut a (2*hz, 2*hy, 2*hx)-voxel box centered on center_zyx out of
    `volume` at native spacing, padding with pad_value outside the
    volume bounds. Also returns the GLOBAL voxel offset (z_lo, y_lo,
    x_lo) of the window's origin, needed to translate any coordinate
    found inside the window back to global volume coordinates.

    Behavior-identical to 04_detect_and_patch.py's extract_patch(),
    reimplemented here (see module docstring for why this isn't a
    cross-file import) with the offset also returned.
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


def resample_to_shape(patch: np.ndarray, target_shape_zyx, pad_value: float):
    """Resample `patch` (any shape) to exactly target_shape_zyx voxels
    via linear interpolation. Behavior-identical to 04's
    resample_patch()."""
    zoom_factors = [t / s for t, s in zip(target_shape_zyx, patch.shape)]
    resampled = ndimage.zoom(patch, zoom_factors, order=1, mode="nearest")
    out = np.full(target_shape_zyx, pad_value, dtype=np.float32)
    slices = tuple(slice(0, min(s, t)) for s, t in zip(resampled.shape, target_shape_zyx))
    out[slices] = resampled[slices]
    return out


def resample_to_isotropic(window: np.ndarray, native_spacing_zyx, iso_spacing_mm: float,
                           pad_value: float):
    """Resample a native-spacing window to an ISOTROPIC voxel grid
    (iso_spacing_mm on every axis). Output shape varies (not fixed),
    unlike resample_to_shape() -- this is used for Hessian shape
    analysis, where equal physical voxel size on every axis is what
    makes eigenvalues comparable across axes, not a fixed tensor
    shape for a classifier."""
    zoom_factors = [s / iso_spacing_mm for s in native_spacing_zyx]
    resampled = ndimage.zoom(window, zoom_factors, order=1, mode="nearest")
    return resampled.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Part 1: Hessian eigenvalue shape filter
# ---------------------------------------------------------------------------

def hessian_sphericity(volume_hu: np.ndarray, spacing_zyx, center_zyx, sigma_mm: float,
                        analysis_size_mm: float, iso_spacing_mm: float):
    """
    Returns (sphericity, num_negative_eigs, ok) for the candidate at
    center_zyx.

    sphericity = |lam1| / |lam3| where |lam1| <= |lam2| <= |lam3| are
    the Hessian eigenvalue magnitudes at the candidate's own
    LoG-matched scale, computed on an ISOTROPICALLY resampled local
    patch (see module docstring, Part 1). num_negative_eigs counts how
    many of the 3 eigenvalues are negative (3 = "locally brighter than
    surroundings in every direction", the expected signature of a real
    blob center). `ok` is False if the local window was degenerate
    (e.g. right at the volume edge) and no reliable Hessian could be
    computed -- callers should treat that as "cannot classify" rather
    than "rejected".
    """
    slice_spacing_mm, y_spacing_mm, x_spacing_mm = spacing_zyx
    half_extent_vox_zyx = (
        max(2, int(round((analysis_size_mm / 2.0) / slice_spacing_mm))),
        max(2, int(round((analysis_size_mm / 2.0) / y_spacing_mm))),
        max(2, int(round((analysis_size_mm / 2.0) / x_spacing_mm))),
    )
    window, _offset = crop_native_window(volume_hu, center_zyx, half_extent_vox_zyx, AIR_HU)

    iso = resample_to_isotropic(window, spacing_zyx, iso_spacing_mm, AIR_HU)
    if min(iso.shape) < 5:
        # Window too small after resampling to compute a stable
        # second derivative -- can happen right at a volume edge.
        return 0.0, 0, False

    sigma_iso_vox = max(sigma_mm / iso_spacing_mm, 0.75)

    # Six second-derivative volumes making up the symmetric 3x3
    # Hessian (Hzy == Hyz etc., so only the upper triangle is needed).
    Hzz = ndimage.gaussian_filter(iso, sigma=sigma_iso_vox, order=(2, 0, 0), mode="nearest")
    Hyy = ndimage.gaussian_filter(iso, sigma=sigma_iso_vox, order=(0, 2, 0), mode="nearest")
    Hxx = ndimage.gaussian_filter(iso, sigma=sigma_iso_vox, order=(0, 0, 2), mode="nearest")
    Hzy = ndimage.gaussian_filter(iso, sigma=sigma_iso_vox, order=(1, 1, 0), mode="nearest")
    Hzx = ndimage.gaussian_filter(iso, sigma=sigma_iso_vox, order=(1, 0, 1), mode="nearest")
    Hyx = ndimage.gaussian_filter(iso, sigma=sigma_iso_vox, order=(0, 1, 1), mode="nearest")

    cz, cy, cx = (s // 2 for s in iso.shape)
    # Average over a small central neighborhood (not just the single
    # center voxel) for stability against discretization/noise -- the
    # window was built centered on the candidate, so its geometric
    # center IS the candidate location.
    r = 1
    z0, z1 = max(0, cz - r), min(iso.shape[0], cz + r + 1)
    y0, y1 = max(0, cy - r), min(iso.shape[1], cy + r + 1)
    x0, x1 = max(0, cx - r), min(iso.shape[2], cx + r + 1)

    H = np.array([
        [Hzz[z0:z1, y0:y1, x0:x1].mean(), Hzy[z0:z1, y0:y1, x0:x1].mean(), Hzx[z0:z1, y0:y1, x0:x1].mean()],
        [Hzy[z0:z1, y0:y1, x0:x1].mean(), Hyy[z0:z1, y0:y1, x0:x1].mean(), Hyx[z0:z1, y0:y1, x0:x1].mean()],
        [Hzx[z0:z1, y0:y1, x0:x1].mean(), Hyx[z0:z1, y0:y1, x0:x1].mean(), Hxx[z0:z1, y0:y1, x0:x1].mean()],
    ], dtype=np.float64)

    eigs = np.linalg.eigvalsh(H)  # symmetric matrix -> real eigenvalues
    num_negative = int(np.sum(eigs < 0))
    abs_sorted = np.sort(np.abs(eigs))  # lam1 <= lam2 <= lam3, by magnitude
    lam1, lam3 = abs_sorted[0], abs_sorted[2]
    sphericity = float(lam1 / lam3) if lam3 > 1e-8 else 0.0
    return sphericity, num_negative, True


# ---------------------------------------------------------------------------
# Part 2: Region-growing adaptive crop
# ---------------------------------------------------------------------------

def build_anisotropic_ball(radius_mm: float, spacing_zyx):
    """
    Boolean structuring element that is a SPHERE in physical mm space,
    even though the underlying voxel grid is anisotropic (native
    spacing, not resampled -- resampling the whole search window to
    isotropic just for this would be expensive and unnecessary here).

    Built directly from physical distance per axis rather than a
    fixed voxel-count radius, so e.g. a 1.5mm radius produces a squat
    ellipsoid in voxel-count terms on an anisotropic grid (few Z
    voxels, more Y/X voxels) that is still a true 1.5mm sphere
    physically. Used to erode away thin vessel necks (~1-3mm across)
    without needing an isotropic resample of the whole growth window.
    """
    slice_spacing_mm, y_spacing_mm, x_spacing_mm = spacing_zyx
    rz = max(1, int(np.ceil(radius_mm / slice_spacing_mm)))
    ry = max(1, int(np.ceil(radius_mm / y_spacing_mm)))
    rx = max(1, int(np.ceil(radius_mm / x_spacing_mm)))
    zz, yy, xx = np.ogrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    dist2 = (
        (zz * slice_spacing_mm) ** 2
        + (yy * y_spacing_mm) ** 2
        + (xx * x_spacing_mm) ** 2
    )
    ball = dist2 <= radius_mm ** 2
    # A structuring element that ends up all-False (radius smaller
    # than a single voxel on every axis) would make binary_erosion
    # erase everything; guard by always keeping at least the center
    # voxel plus its 6-neighbors.
    if not ball.any():
        ball[rz, ry, rx] = True
    return ball


def _attempt_opening_at_radius(foreground, seed_zyx, spacing_zyx, radius_mm,
                                slice_spacing_mm, y_spacing_mm, x_spacing_mm,
                                voxel_volume_mm3):
    """
    One erode -> label -> dilate (morphological opening) attempt at a
    single radius. Returns None if the candidate's own core doesn't
    survive erosion at this radius (even after a small neighborhood
    search), signaling the caller to stop trying LARGER radii (they'd
    only erase more). Otherwise returns a dict with the grown
    component and its measurements, leak-checked or not -- the caller
    decides what counts as a leak.
    """
    sz, sy, sx = seed_zyx
    struct_ball = build_anisotropic_ball(radius_mm, spacing_zyx)
    eroded_foreground = ndimage.binary_erosion(foreground, structure=struct_ball)

    labeled_eroded, _num = ndimage.label(eroded_foreground, structure=STRUCT_26)
    seed_label = labeled_eroded[sz, sy, sx]

    if seed_label == 0:
        for r in (1, 2, 3):
            z0, z1 = max(0, sz - r), min(eroded_foreground.shape[0], sz + r + 1)
            y0, y1 = max(0, sy - r), min(eroded_foreground.shape[1], sy + r + 1)
            x0, x1 = max(0, sx - r), min(eroded_foreground.shape[2], sx + r + 1)
            local = labeled_eroded[z0:z1, y0:y1, x0:x1]
            if local.any():
                seed_label = local[local > 0][0]
                break

    if seed_label == 0:
        return None

    eroded_seed_component = labeled_eroded == seed_label
    # Bounded dilation by the SAME radius as the erosion (classic
    # opening), not geodesic reconstruction -- see grow_nodule_region
    # docstring for why reconstruction would silently undo the erosion.
    component = ndimage.binary_dilation(
        eroded_seed_component, structure=struct_ball
    ) & foreground

    voxel_count = int(component.sum())
    volume_mm3 = voxel_count * voxel_volume_mm3
    equivalent_diameter_mm = (6.0 * volume_mm3 / np.pi) ** (1.0 / 3.0)

    zs, ys, xs = np.where(component)
    # Longest bounding-box edge, in mm. Volume/equivalent-diameter
    # alone is NOT a sufficient leak guard: a long, thin vessel
    # segment (e.g. 3mm radius x 80mm length) has roughly the same
    # VOLUME as a compact ~16mm nodule despite being an obvious leak,
    # so an elongation check is needed alongside the volume check.
    bbox_extent_mm = max(
        (zs.max() - zs.min() + 1) * slice_spacing_mm,
        (ys.max() - ys.min() + 1) * y_spacing_mm,
        (xs.max() - xs.min() + 1) * x_spacing_mm,
    )

    return {
        "zs": zs, "ys": ys, "xs": xs,
        "volume_mm3": volume_mm3,
        "equivalent_diameter_mm": equivalent_diameter_mm,
        "bbox_extent_mm": bbox_extent_mm,
    }


def grow_nodule_region(volume_hu: np.ndarray, lung_mask: np.ndarray, spacing_zyx,
                        center_zyx, search_radius_mm: float, max_grown_diameter_mm: float,
                        erosion_radius_mm: float = 1.5, max_erosion_radius_mm: float = 4.0,
                        erosion_radius_step_mm: float = 0.5):
    """
    Region-grow from the candidate seed voxel within a native-spacing
    search window, gated to lung_mask. Returns a dict:
        {
            "grown_ok": bool,       # False only if the seed itself
                                     # never cleared threshold even
                                     # after a small neighborhood search
            "leaked": bool,         # True if every erosion radius
                                     # tried still produced a region
                                     # exceeding max_grown_diameter_mm
            "volume_mm3": float,    # 0.0 if not grown_ok
            "equivalent_diameter_mm": float,
            "erosion_radius_used_mm": float or None,
                                     # which radius in the sweep
                                     # actually worked (None if leaked
                                     # or not grown)
            "bbox_global_zyx": ((z0,z1),(y0,y1),(x0,x1)) or None,
                                     # tight bbox of the grown region,
                                     # in GLOBAL volume voxel indices,
                                     # half-open (z1/y1/x1 exclusive)
        }

    See module docstring, Part 2, for the full rationale (Otsu
    threshold -> 26-connected component containing the seed -> leak
    guard on equivalent diameter -> tight bounding box).

    === Why a plain threshold+connected-component leaks 100% of the
    time, and what the erosion sweep fixes ===

    A nodule and the pulmonary vessels around it are BOTH soft-tissue
    density (roughly 0 to +100 HU) -- there is no HU threshold that
    separates "nodule" from "vessel", only "soft tissue" from
    "aerated lung". Otsu's method finds exactly that split. The
    pulmonary vasculature is a single, physically continuous branching
    tree that reaches through the entire lung, so ANY candidate that
    touches or comes within one voxel of a vessel wall -- which is
    nearly all of them, since vessels are everywhere -- has its
    "foreground" connected-component swell to include a chunk of that
    tree.

    The fix is morphological opening (erode with a small sphere to
    sever thin necks, then dilate the surviving core back out by the
    SAME radius, intersected with the original foreground). But there
    is no single erosion radius that works for every candidate: a
    thin distal vessel neck (~1mm radius) is severed by a 1.5mm ball,
    but a nodule sitting against a thicker proximal vessel (2-3mm
    radius) needs a bigger ball to sever THAT neck -- and vessel
    caliber varies candidate to candidate, patient to patient. Using
    one fixed radius for every candidate is why some still leaked
    (2/16 grown, 14/16 leaked at a flat 1.5mm).

    So this sweeps radii from erosion_radius_mm up to
    max_erosion_radius_mm in erosion_radius_step_mm steps, and uses
    the SMALLEST radius that produces a compact (non-leaking) result
    for THIS candidate -- smallest-that-works, because a bigger ball
    than necessary erodes more of the true nodule surface than needed,
    distorting its recovered shape and volume. If the candidate's own
    core doesn't survive a given radius at all (nodule smaller than
    the ball), larger radii are skipped, since they'd only erase more.
    If NO radius in the sweep produces a compact result, the smallest
    (least-bad) leaked attempt is reported, same "leaked=True ->
    caller falls back to a fixed box" contract as before -- this
    function never silently returns a bad crop as if it were good.
    """
    slice_spacing_mm, y_spacing_mm, x_spacing_mm = spacing_zyx
    voxel_volume_mm3 = slice_spacing_mm * y_spacing_mm * x_spacing_mm

    half_extent_vox_zyx = (
        max(3, int(round(search_radius_mm / slice_spacing_mm))),
        max(3, int(round(search_radius_mm / y_spacing_mm))),
        max(3, int(round(search_radius_mm / x_spacing_mm))),
    )
    window, (z_off, y_off, x_off) = crop_native_window(
        volume_hu, center_zyx, half_extent_vox_zyx, AIR_HU
    )
    lung_window, _ = crop_native_window(
        lung_mask.astype(np.float32), center_zyx, half_extent_vox_zyx, 0.0
    )
    lung_window = lung_window > 0.5

    seed_local = tuple(h for h in half_extent_vox_zyx)  # window is centered on the seed

    # Otsu threshold on the in-lung portion of the window only, so a
    # window that happens to include some out-of-lung padding/anatomy
    # doesn't skew the bimodal split. Falls back to a fixed HU split
    # if Otsu can't be computed (e.g. a near-uniform window).
    in_lung_values = window[lung_window]
    try:
        if in_lung_values.size < 50 or np.ptp(in_lung_values) < 1.0:
            raise ValueError("degenerate window")
        threshold = float(threshold_otsu(in_lung_values))
    except Exception:
        threshold = FALLBACK_HU_THRESHOLD

    foreground = (window > threshold) & lung_window

    # If the seed voxel itself isn't foreground (e.g. sits exactly on
    # a partial-volume edge voxel), search a small neighborhood for
    # the nearest foreground voxel before giving up.
    sz, sy, sx = seed_local
    if not foreground[sz, sy, sx]:
        found = False
        for r in (1, 2, 3):
            z0, z1 = max(0, sz - r), min(foreground.shape[0], sz + r + 1)
            y0, y1 = max(0, sy - r), min(foreground.shape[1], sy + r + 1)
            x0, x1 = max(0, sx - r), min(foreground.shape[2], sx + r + 1)
            local = foreground[z0:z1, y0:y1, x0:x1]
            if local.any():
                lz, ly, lx = np.argwhere(local)[0]
                sz, sy, sx = z0 + lz, y0 + ly, x0 + lx
                found = True
                break
        if not found:
            return {
                "grown_ok": False, "leaked": False, "volume_mm3": 0.0,
                "equivalent_diameter_mm": 0.0, "bbox_global_zyx": None,
            }

    # --- Sever thin vessel/airway necks before labeling, sweeping the
    # erosion radius upward since vessel caliber varies per candidate
    # (see docstring) ---
    radii_to_try = []
    r = erosion_radius_mm
    while r <= max_erosion_radius_mm + 1e-9:
        radii_to_try.append(round(r, 4))
        r += erosion_radius_step_mm

    best_leaked_attempt = None       # smallest-diameter leak, for reporting if nothing succeeds

    for radius_mm in radii_to_try:
        attempt = _attempt_opening_at_radius(
            foreground, (sz, sy, sx), spacing_zyx, radius_mm,
            slice_spacing_mm, y_spacing_mm, x_spacing_mm, voxel_volume_mm3,
        )
        if attempt is None:
            # This candidate's own core didn't survive this radius --
            # larger radii will only erase more of it, so stop here.
            break

        leaked = (
            attempt["equivalent_diameter_mm"] > max_grown_diameter_mm
            or attempt["bbox_extent_mm"] > max_grown_diameter_mm
        )
        if not leaked:
            # Smallest radius that worked -- accept immediately rather
            # than continuing to sweep, since a bigger ball than
            # necessary erodes more of the true nodule surface than
            # needed.
            zs, ys, xs = attempt["zs"], attempt["ys"], attempt["xs"]
            bbox_global_zyx = (
                (int(zs.min()) + z_off, int(zs.max()) + 1 + z_off),
                (int(ys.min()) + y_off, int(ys.max()) + 1 + y_off),
                (int(xs.min()) + x_off, int(xs.max()) + 1 + x_off),
            )
            return {
                "grown_ok": True, "leaked": False,
                "volume_mm3": attempt["volume_mm3"],
                "equivalent_diameter_mm": attempt["equivalent_diameter_mm"],
                "erosion_radius_used_mm": radius_mm,
                "bbox_global_zyx": bbox_global_zyx,
            }

        if (best_leaked_attempt is None
                or attempt["equivalent_diameter_mm"] < best_leaked_attempt["equivalent_diameter_mm"]):
            best_leaked_attempt = attempt

    if best_leaked_attempt is not None:
        # Every radius we tried still leaked -- report the least-bad
        # attempt so the CSV shows the closest measured diameter, but
        # still flagged leaked=True so the caller falls back to a
        # fixed box rather than trusting a bad crop.
        return {
            "grown_ok": True, "leaked": True,
            "volume_mm3": best_leaked_attempt["volume_mm3"],
            "equivalent_diameter_mm": best_leaked_attempt["equivalent_diameter_mm"],
            "erosion_radius_used_mm": None,
            "bbox_global_zyx": None,
        }

    # Every radius in the sweep fully erased the candidate's own core
    # (a genuinely tiny nodule, smaller than even the smallest radius
    # tried) -- fall back to labeling the raw, un-eroded foreground
    # directly. This candidate loses the bridge-severing protection,
    # but the existing max_grown_diameter_mm leak guard below still
    # catches it if it swells into a vessel, so nothing is silently
    # wrong.
    labeled_raw, _num = ndimage.label(foreground, structure=STRUCT_26)
    seed_label_raw = labeled_raw[sz, sy, sx]
    if seed_label_raw == 0:
        return {
            "grown_ok": False, "leaked": False, "volume_mm3": 0.0,
            "equivalent_diameter_mm": 0.0, "erosion_radius_used_mm": None,
            "bbox_global_zyx": None,
        }
    component = labeled_raw == seed_label_raw
    voxel_count = int(component.sum())
    volume_mm3 = voxel_count * voxel_volume_mm3
    equivalent_diameter_mm = (6.0 * volume_mm3 / np.pi) ** (1.0 / 3.0)

    zs, ys, xs = np.where(component)
    bbox_extent_mm = max(
        (zs.max() - zs.min() + 1) * slice_spacing_mm,
        (ys.max() - ys.min() + 1) * y_spacing_mm,
        (xs.max() - xs.min() + 1) * x_spacing_mm,
    )

    if equivalent_diameter_mm > max_grown_diameter_mm or bbox_extent_mm > max_grown_diameter_mm:
        return {
            "grown_ok": True, "leaked": True, "volume_mm3": volume_mm3,
            "equivalent_diameter_mm": equivalent_diameter_mm,
            "erosion_radius_used_mm": None, "bbox_global_zyx": None,
        }

    bbox_global_zyx = (
        (int(zs.min()) + z_off, int(zs.max()) + 1 + z_off),
        (int(ys.min()) + y_off, int(ys.max()) + 1 + y_off),
        (int(xs.min()) + x_off, int(xs.max()) + 1 + x_off),
    )
    return {
        "grown_ok": True, "leaked": False, "volume_mm3": volume_mm3,
        "equivalent_diameter_mm": equivalent_diameter_mm,
        "erosion_radius_used_mm": 0.0, "bbox_global_zyx": bbox_global_zyx,
    }


def fallback_bbox_global_zyx(center_zyx, spacing_zyx, diameter_mm: float, margin_mm: float,
                              volume_shape):
    """Fixed-size bbox around center_zyx, sized from the candidate's
    own 04-detected diameter_mm + margin -- used when growth fails or
    leaks, so every candidate still gets SOME reasonable crop rather
    than silently vanishing from the output."""
    slice_spacing_mm, y_spacing_mm, x_spacing_mm = spacing_zyx
    half_mm = diameter_mm / 2.0 + margin_mm
    half_vox = (
        max(1, int(round(half_mm / slice_spacing_mm))),
        max(1, int(round(half_mm / y_spacing_mm))),
        max(1, int(round(half_mm / x_spacing_mm))),
    )
    cz, cy, cx = center_zyx
    bbox = (
        (max(0, cz - half_vox[0]), min(volume_shape[0], cz + half_vox[0])),
        (max(0, cy - half_vox[1]), min(volume_shape[1], cy + half_vox[1])),
        (max(0, cx - half_vox[2]), min(volume_shape[2], cx + half_vox[2])),
    )
    return bbox


def add_margin_and_clip(bbox_zyx, spacing_zyx, margin_mm: float, volume_shape):
    """Expand a (z_range, y_range, x_range) bbox by margin_mm of
    physical context on every side, clipped to the volume bounds."""
    slice_spacing_mm, y_spacing_mm, x_spacing_mm = spacing_zyx
    margins_vox = (
        max(1, int(round(margin_mm / slice_spacing_mm))),
        max(1, int(round(margin_mm / y_spacing_mm))),
        max(1, int(round(margin_mm / x_spacing_mm))),
    )
    out = []
    for (lo, hi), m, size in zip(bbox_zyx, margins_vox, volume_shape):
        out.append((max(0, lo - m), min(size, hi + m)))
    return tuple(out)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def shape_filter_and_grow(
    masked_dir: str,
    candidates_dir: str,
    out_dir: str,
    min_sphericity: float = 0.35,
    shape_analysis_size_mm: float = 16.0,
    iso_spacing_mm: float = 0.75,
    growth_search_radius_mm: float = 40.0,
    growth_margin_mm: float = 2.0,
    max_grown_diameter_mm: float = 45.0,
    erosion_radius_mm: float = 1.5,
    max_erosion_radius_mm: float = 4.0,
    erosion_radius_step_mm: float = 0.5,
    fallback_margin_mm: float = 4.0,
    patch_shape: int = 32,
    save_masks: bool = False,
):
    print(f"[info] Loading volume/lung_mask from '{masked_dir}'...")
    volume_hu, lung_mask, meta = load_masked_output(masked_dir)
    spacing_zyx = get_spacing_zyx(meta)

    print(f"[info] Loading candidates from '{candidates_dir}'...")
    candidates = load_candidates(candidates_dir)
    print(f"[info] {len(candidates)} candidates to evaluate.")

    os.makedirs(out_dir, exist_ok=True)
    patches_dir = os.path.join(out_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    masks_dir = os.path.join(out_dir, "masks")
    if save_masks:
        os.makedirs(masks_dir, exist_ok=True)

    target_shape_zyx = (patch_shape, patch_shape, patch_shape)

    rows = []
    num_rejected_shape = 0
    num_grown_ok = 0
    num_leaked = 0
    num_grow_failed = 0

    for candidate_id, cand in enumerate(candidates):
        center_zyx = (int(cand["voxel_z"]), int(cand["voxel_y"]), int(cand["voxel_x"]))
        sigma_mm = float(cand["sigma_mm"])
        detected_diameter_mm = float(cand["diameter_mm"])

        sphericity, num_negative, shape_ok = hessian_sphericity(
            volume_hu, spacing_zyx, center_zyx, sigma_mm,
            analysis_size_mm=shape_analysis_size_mm, iso_spacing_mm=iso_spacing_mm,
        )
        # A degenerate window (shape_ok=False, e.g. right at the
        # volume edge) is treated as "cannot classify -> keep", not
        # as evidence of a tube -- absence of a measurement isn't
        # evidence against the candidate.
        is_tubular = shape_ok and (sphericity < min_sphericity or num_negative < 2)

        if is_tubular:
            num_rejected_shape += 1
            rows.append({
                "candidate_id": candidate_id,
                "kept": False,
                "sphericity": round(sphericity, 3),
                "num_negative_eigs": num_negative,
                "grown_ok": None, "leaked": None,
                "volume_mm3": None, "equivalent_diameter_mm": None,
                "erosion_radius_used_mm": None,
                "bbox_used": "rejected_tubular",
                "bbox_z0": None, "bbox_z1": None,
                "bbox_y0": None, "bbox_y1": None,
                "bbox_x0": None, "bbox_x1": None,
                "patch_file": None,
            })
            continue

        growth = grow_nodule_region(
            volume_hu, lung_mask, spacing_zyx, center_zyx,
            search_radius_mm=growth_search_radius_mm,
            max_grown_diameter_mm=max_grown_diameter_mm,
            erosion_radius_mm=erosion_radius_mm,
            max_erosion_radius_mm=max_erosion_radius_mm,
            erosion_radius_step_mm=erosion_radius_step_mm,
        )

        if growth["grown_ok"] and not growth["leaked"]:
            bbox = add_margin_and_clip(
                growth["bbox_global_zyx"], spacing_zyx, growth_margin_mm, volume_hu.shape
            )
            bbox_used = "grown"
            num_grown_ok += 1
        else:
            bbox = fallback_bbox_global_zyx(
                center_zyx, spacing_zyx, detected_diameter_mm, fallback_margin_mm, volume_hu.shape
            )
            bbox_used = "fallback_leaked" if growth["leaked"] else "fallback_grow_failed"
            if growth["leaked"]:
                num_leaked += 1
            else:
                num_grow_failed += 1

        (z0, z1), (y0, y1), (x0, x1) = bbox
        raw_patch = volume_hu[z0:z1, y0:y1, x0:x1].astype(np.float32, copy=True)
        if raw_patch.size == 0:
            # Degenerate bbox (shouldn't happen given the clipping
            # above, but guard rather than let resample() divide by
            # zero on an empty axis).
            raw_patch = np.full((1, 1, 1), AIR_HU, dtype=np.float32)
        patch = resample_to_shape(raw_patch, target_shape_zyx, AIR_HU)

        patch_filename = f"nodule_{candidate_id:04d}.npy"
        np.save(os.path.join(patches_dir, patch_filename), patch)

        if save_masks and bbox_used == "grown":
            mask_crop = np.zeros(raw_patch.shape, dtype=bool)
            # Recompute which voxels of the crop belong to the grown
            # component by re-deriving it from the same bbox region --
            # cheaper than threading the full-resolution mask through
            # grow_nodule_region()'s window coordinates, since we
            # already trust this bbox came from that component.
            zs, ys, xs = np.where(
                (volume_hu[z0:z1, y0:y1, x0:x1] > FALLBACK_HU_THRESHOLD) & lung_mask[z0:z1, y0:y1, x0:x1]
            )
            mask_crop[zs, ys, xs] = True
            np.save(os.path.join(masks_dir, f"nodule_{candidate_id:04d}_mask.npy"), mask_crop)

        rows.append({
            "candidate_id": candidate_id,
            "kept": True,
            "sphericity": round(sphericity, 3),
            "num_negative_eigs": num_negative,
            "grown_ok": growth["grown_ok"],
            "leaked": growth["leaked"],
            "volume_mm3": round(growth["volume_mm3"], 1) if growth["grown_ok"] else None,
            "equivalent_diameter_mm": round(growth["equivalent_diameter_mm"], 2) if growth["grown_ok"] else None,
            "erosion_radius_used_mm": growth.get("erosion_radius_used_mm"),
            "bbox_used": bbox_used,
            # The ACTUAL voxel region (in volume_hu's own index space)
            # this patch was cropped from -- WITHOUT this, nothing
            # downstream can place this patch's Grad-CAM heatmap back
            # into the full volume correctly. It was previously
            # computed (see `bbox` above) and used to crop, but never
            # saved -- 07_visualize_gradcam.py's own docstring already
            # anticipates a "bbox_zyx" field under exactly this name
            # and falls back to an approximation (centered on the
            # ORIGINAL LoG seed voxel, not the true grown-region
            # center) whenever it's absent, which was always, until
            # now. IMPORTANT: (z0,z1,y0,y1,x0,x1) above are Python
            # slice bounds (z1 EXCLUSIVE, from `volume_hu[z0:z1, ...]`
            # a few lines up) -- 07's own placement code
            # (build_full_volume_heatmap/compute_fallback_bbox) uses
            # INCLUSIVE bounds (`full_heatmap[z0:z1+1, ...]`), so the
            # upper bound of each axis is written as z1-1 (etc.) here,
            # not the raw exclusive z1 -- writing the exclusive value
            # directly would silently shrink every placed region by
            # one voxel on the far edge of each axis.
            "bbox_zyx": [int(z0), int(z1 - 1), int(y0), int(y1 - 1), int(x0), int(x1 - 1)],
            "bbox_z0": int(z0), "bbox_z1": int(z1 - 1),
            "bbox_y0": int(y0), "bbox_y1": int(y1 - 1),
            "bbox_x0": int(x0), "bbox_x1": int(x1 - 1),
            "patch_file": os.path.join("patches", patch_filename),
        })

        if (candidate_id + 1) % 25 == 0 or (candidate_id + 1) == len(candidates):
            print(f"[info] Processed {candidate_id + 1}/{len(candidates)} candidates...")

    csv_path = os.path.join(out_dir, "nodules.csv")
    fieldnames = [
        "candidate_id", "kept", "sphericity", "num_negative_eigs",
        "grown_ok", "leaked", "volume_mm3", "equivalent_diameter_mm",
        "erosion_radius_used_mm", "bbox_used",
        "bbox_z0", "bbox_z1", "bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1",
        "patch_file",
    ]
    with open(csv_path, "w", newline="") as f:
        # extrasaction="ignore": rows also carry a nested "bbox_zyx"
        # list (same 6 numbers, for nodules.json/07's consumption --
        # see the rows.append comment above) that has no sane flat-CSV
        # column; the 6 bbox_z0..bbox_x1 columns above already cover
        # the same information for nodules.csv.
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    params = {
        "min_sphericity": min_sphericity,
        "shape_analysis_size_mm": shape_analysis_size_mm,
        "iso_spacing_mm": iso_spacing_mm,
        "growth_search_radius_mm": growth_search_radius_mm,
        "growth_margin_mm": growth_margin_mm,
        "max_grown_diameter_mm": max_grown_diameter_mm,
        "erosion_radius_mm": erosion_radius_mm,
        "max_erosion_radius_mm": max_erosion_radius_mm,
        "erosion_radius_step_mm": erosion_radius_step_mm,
        "fallback_margin_mm": fallback_margin_mm,
        "patch_shape": patch_shape,
    }
    with open(os.path.join(out_dir, "nodules.json"), "w") as f:
        json.dump({"params": params, "nodules": rows}, f, indent=2)

    out_meta = dict(meta)
    out_meta["shape_filter_and_grow_params"] = params
    out_meta["num_candidates_in"] = len(candidates)
    out_meta["num_rejected_shape"] = num_rejected_shape
    out_meta["num_grown_ok"] = num_grown_ok
    out_meta["num_leaked"] = num_leaked
    out_meta["num_grow_failed"] = num_grow_failed
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(out_meta, f, indent=2)

    kept = len(candidates) - num_rejected_shape
    print(f"\n[done] {len(candidates)} candidates in -> {num_rejected_shape} rejected as "
          f"tubular -> {kept} kept.")
    print(f"[done] Of {kept} kept: {num_grown_ok} grown successfully, "
          f"{num_leaked} leaked (fell back to fixed box), "
          f"{num_grow_failed} failed to grow at all (fell back to fixed box).")
    print(f"[done] Wrote nodules.csv, nodules.json, meta.json, and {kept} "
          f"patch(es) -> '{patches_dir}'")

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="STEP 5: Reject tubular (vessel/airway) candidates via "
        "Hessian eigenvalue shape analysis, then replace 04's fixed-size "
        "patch with a region-grown, size-adaptive crop."
    )
    parser.add_argument(
        "masked_dir",
        help="Directory containing volume_hu.npy / lung_mask.npy / "
        "meta.json (the --out-dir from 02_mask_and_crop.py).",
    )
    parser.add_argument(
        "candidates_dir",
        help="Directory containing candidates.json (the --out-dir from "
        "04_detect_and_patch.py).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to write nodules.csv/json and patches/ (default: "
        "'<candidates_dir>_nodules').",
    )
    parser.add_argument(
        "--min-sphericity", type=float, default=0.35,
        help="Reject candidates below this Hessian eigenvalue sphericity "
        "ratio (|lam1|/|lam3|, 0-1, higher=more sphere-like) as tubular "
        "(default: 0.35). Tune this against known nodule/vessel examples "
        "before trusting it on new scans.",
    )
    parser.add_argument(
        "--shape-analysis-size-mm", type=float, default=16.0,
        help="Physical side length (mm) of the local window used for "
        "Hessian shape analysis (default: 16.0).",
    )
    parser.add_argument(
        "--iso-spacing-mm", type=float, default=0.75,
        help="Isotropic voxel spacing (mm) the shape-analysis window is "
        "resampled to before computing Hessian eigenvalues (default: 0.75).",
    )
    parser.add_argument(
        "--growth-search-radius-mm", type=float, default=40.0,
        help="Half-width (mm) of the native-spacing search window region "
        "growing is allowed to explore from each candidate seed "
        "(default: 40.0). Should comfortably exceed any expected nodule "
        "radius.",
    )
    parser.add_argument(
        "--growth-margin-mm", type=float, default=2.0,
        help="Physical context margin (mm) added around a successfully "
        "grown region's tight bounding box before cropping (default: 2.0).",
    )
    parser.add_argument(
        "--max-grown-diameter-mm", type=float, default=45.0,
        help="Leak guard: if the grown region's equivalent diameter "
        "exceeds this (mm), discard the growth and fall back to a fixed "
        "box instead (default: 45.0, i.e. larger than any plausible "
        "single nodule -- a grown region this big almost certainly leaked "
        "into an attached vessel or the mediastinum).",
    )
    parser.add_argument(
        "--erosion-radius-mm", type=float, default=1.5,
        help="Starting radius (mm) of the spherical structuring element "
        "used to erode the thresholded foreground BEFORE connected-"
        "component labeling, severing thin vessel/airway necks so growth "
        "can't leak through them (default: 1.5). This is what actually "
        "fixes the leak failure mode -- a nodule and a vessel share the "
        "same HU range, so only geometry (a vessel neck is thin, a "
        "nodule core isn't) can tell them apart, not intensity alone. "
        "The radius is swept upward per-candidate up to "
        "--max-erosion-radius-mm, since vessel caliber varies candidate "
        "to candidate -- see that flag.",
    )
    parser.add_argument(
        "--max-erosion-radius-mm", type=float, default=4.0,
        help="Cap (mm) on the erosion-radius sweep (default: 4.0). For "
        "each kept candidate, the erosion radius starts at "
        "--erosion-radius-mm and increases by --erosion-radius-step-mm "
        "until either a non-leaking result is found (the SMALLEST "
        "working radius is used, to avoid over-eroding the true nodule "
        "shape) or this cap is reached. Raise this if candidates near "
        "thick proximal vessels still leak; a higher cap risks eroding "
        "away genuinely small nodules before it ever helps, though the "
        "sweep already stops increasing the moment a candidate's own "
        "core stops surviving erosion.",
    )
    parser.add_argument(
        "--erosion-radius-step-mm", type=float, default=0.5,
        help="Step size (mm) between successive radii in the erosion "
        "sweep (default: 0.5). Smaller steps find the minimal working "
        "radius more precisely at the cost of more attempts per "
        "candidate (cheap here -- there are only ever a handful of kept, "
        "post-shape-filter candidates per scan).",
    )
    parser.add_argument(
        "--fallback-margin-mm", type=float, default=4.0,
        help="Margin (mm) added around the candidate's own 04-detected "
        "diameter_mm when building the fallback box (used when growth "
        "fails or leaks) (default: 4.0).",
    )
    parser.add_argument(
        "--patch-shape", type=int, default=32,
        help="Output patch side length in VOXELS (patch is patch_shape^3), "
        "matching 04's convention (default: 32).",
    )
    parser.add_argument(
        "--save-masks", action="store_true",
        help="Also save each grown region's boolean segmentation mask "
        "(native spacing, pre-resample) to masks/ -- useful for later "
        "volume/VDT work.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.candidates_dir.rstrip("/\\") + "_nodules")
    shape_filter_and_grow(
        args.masked_dir, args.candidates_dir, out_dir,
        min_sphericity=args.min_sphericity,
        shape_analysis_size_mm=args.shape_analysis_size_mm,
        iso_spacing_mm=args.iso_spacing_mm,
        growth_search_radius_mm=args.growth_search_radius_mm,
        growth_margin_mm=args.growth_margin_mm,
        max_grown_diameter_mm=args.max_grown_diameter_mm,
        erosion_radius_mm=args.erosion_radius_mm,
        max_erosion_radius_mm=args.max_erosion_radius_mm,
        erosion_radius_step_mm=args.erosion_radius_step_mm,
        fallback_margin_mm=args.fallback_margin_mm,
        patch_shape=args.patch_shape,
        save_masks=args.save_masks,
    )


if __name__ == "__main__":
    main()