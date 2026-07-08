"""
detect_candidates_cpu.py  (v3 -- LoG blob detector, RECALL-FIRST TUNING)

Local CPU-only step of the annotation-free inference pipeline.

Detects pulmonary nodule candidates using a multi-scale 3D
Laplacian-of-Gaussian (LoG) blob detector, then matches candidates
against pylidc ground-truth annotations to produce validation labels.
Saves 64x64x64 HU patches + a manifest CSV ready for Colab GPU inference.

Why LoG instead of threshold + connected components:
    Solid nodules attach to vessel branches and the pleura, so simple
    thresholding produces one giant connected component per lung that
    swallows every nodule. LoG responds to LOCAL spherical contrast at
    the correct spatial scale and is unaffected by attachment geometry.

--------------------------------------------------------------------------
v3 CHANGES -- this pass is tuned to maximize sensitivity (find every
nodule), accepting more false positives as the cost. Precision is cheap
to recover downstream (the Colab classifier/fusion head filters FPs);
a missed nodule at this stage can never be recovered later, so nothing
here should silently throw candidates away.

1) FIXED: juxtapleural / chest-wall-attached nodule blind spot.
   segment_lungs() only keeps AIR voxels (HU < -400). A nodule is soft
   tissue and is NOT air, so a nodule sitting at the pleural surface (a
   very common, clinically important case) has its center sitting right
   on -- or just outside -- the mask boundary. binary_fill_holes() only
   rescues nodules fully *surrounded* by air; it does nothing for a
   nodule that opens onto the exterior/mediastinal side. This silently
   dropped an entire class of nodules regardless of detector threshold.
   Fix: dilate_lung_mask() adds an isotropic ~18 mm morphological margin
   (mm-aware per axis, since slice spacing != in-plane spacing) around
   the air mask before detection, so pleural-attached and vessel-attached
   nodules fall inside the searched region. The bounding-box padding in
   detect_nodules_log() was already generous enough to accommodate this
   (verified against the new margin), so no other geometry changes needed.

2) Lowered LOG_THRESHOLD (0.015 -> 0.007): the previous threshold was
   cutting real, lower-contrast (GGO / part-solid) nodules below the
   response cutoff. Halving it roughly doubles candidate recall on
   subtle nodules at the cost of more FPs, which is the intended tradeoff.

3) Widened diameter range (3-35mm -> 2-40mm): 2mm floor catches small
   solid nodules and micronodules; 40mm ceiling catches large masses
   that would otherwise fall outside the largest LoG scale entirely
   (previously such a mass would only be *partially* matched by the
   sigma=35mm scale, biasing its measured diameter/response low enough
   to sometimes miss the local-maxima threshold).

4) More scales (12 -> 18) for finer sigma sampling across the wider
   range, so no nodule size falls into a gap between scales.

5) Reduced NMS_MIN_DIST_MM (5.0 -> 3.5mm): with more scales/lower
   threshold, closely-spaced distinct nodules are less likely to be
   collapsed into one detection.

6) Raised MAX_CANDS_PER_SCAN (400 -> 1200): the NMS list is already
   sorted by response descending before truncation, so this cap was
   silently discarding low-response *true* nodules whenever a scan had
   many stronger FPs (busy/vascular lungs). Raising it makes truncation
   effectively a non-issue; downstream classifier handles the volume.

7) Local-maxima footprint no longer keyed off MIN_DIAM_MM (which is now
   2mm and would force a footprint of ~1 voxel, i.e. no non-max
   suppression at the per-voxel level and huge duplicate counts).
   It's now keyed off a fixed small floor (LOCAL_MAX_FLOOR_MM) so tiny
   nodules aren't over-merged at the raw local-maxima stage; the mm-space
   NMS pass afterwards is what actually enforces separation.

8) segment_lungs()'s per-slice component floor (`comp_size < 30`) is
   left in place (it protects against noise specks, not real anatomy)
   but is now also computed relative to a mm-aware minimum footprint so
   it doesn't accidentally discard genuine lung cross-sections on scans
   with unusually fine in-plane spacing.

Usage:
    Single scan:
        python detect_candidates_cpu.py ^
            --patient-id LIDC-IDRI-0001 ^
            --output-dir candidate_patches

    Specific scans:
        python detect_candidates_cpu.py ^
            --patient-ids LIDC-IDRI-0001 LIDC-IDRI-0007 LIDC-IDRI-0141 ^
            --output-dir candidate_patches

    Full dataset (run overnight -- more candidates now, so expect this to
    run longer than v2; recall-first config trades runtime for coverage):
        python detect_candidates_cpu.py ^
            --all-scans ^
            --output-dir candidate_patches

    Resume an interrupted run (skips scans already in the manifest):
        python detect_candidates_cpu.py ^
            --all-scans --resume ^
            --output-dir candidate_patches

Outputs written to --output-dir:
    candidates_manifest.csv     one row per candidate (TP/FP) + FN rows
    <candidate_id>.npy          64x64x64 HU patch per detected candidate
    detection_summary.txt       per-scan TP/FP/FN counts + totals
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.ndimage import (
    gaussian_laplace,
    maximum_filter,
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    generate_binary_structure,
    label as ndlabel,
)

# -- pylidc compatibility patches (Python 3.12 + NumPy 1.24+) -----------------
import configparser
if not hasattr(configparser, 'SafeConfigParser'):
    configparser.SafeConfigParser = configparser.ConfigParser
import numpy as _np
if not hasattr(_np, 'int'):
    _np.int = int
if not hasattr(_np, 'float'):
    _np.float = float
if not hasattr(_np, 'bool'):
    _np.bool = bool

import pylidc as pl
# -----------------------------------------------------------------------------

from cir_multihead_pipeline import (
    FEATURE_NAMES,
    PATCH_SIZE,
    _normalize_feature,
    _get_feature_value,
)

# -- Detection hyperparameters (RECALL-FIRST) ---------------------------------
MIN_DIAM_MM         = 2.0    # smallest nodule to detect (mm)   [was 3.0]
MAX_DIAM_MM         = 40.0   # largest nodule to detect (mm)    [was 35.0]
N_SCALES            = 18     # number of log-spaced sigma values [was 12]
LOG_THRESHOLD       = 0.007  # min normalised LoG response       [was 0.015]
NMS_MIN_DIST_MM     = 3.5    # NMS min centre separation (mm)    [was 5.0]
MAX_CANDS_PER_SCAN  = 1200   # safety cap on candidates per scan [was 400]
MATCH_THRESHOLD_MM  = 15.0   # max centroid distance (mm) for TP match (evaluation only, unchanged)
LUNG_MASK_MARGIN_MM = 18.0   # NEW: dilate lung mask by this much to catch
                              #      pleural/vessel-attached nodules
LOCAL_MAX_FLOOR_MM  = 3.0    # NEW: floor for local-maxima footprint sizing,
                              #      decoupled from MIN_DIAM_MM


# -----------------------------------------------------------------------------
# Volume loading
# -----------------------------------------------------------------------------

def get_spacing_mm(scan):
    """Return (z_mm, y_mm, x_mm) voxel spacing."""
    try:
        ps = scan.pixel_spacing
        y_mm, x_mm = (float(ps[0]), float(ps[1])) if hasattr(ps, '__len__') \
                      else (float(ps), float(ps))
    except Exception:
        y_mm = x_mm = 0.75
    try:
        z_mm = float(scan.slice_thickness)
    except Exception:
        z_mm = 1.5
    return (z_mm, y_mm, x_mm)


def load_volume_hu(scan):
    """
    Load CT as float32 HU array in (Z, Y, X) = (slices, rows, cols) order.

    pylidc's to_volume() returns (rows, cols, slices) = (Y, X, Z).
    We transpose to (Z, Y, X) so that:
      - dim 0 matches get_spacing_mm()[0] = slice_thickness (z_mm)
      - dim 1 matches get_spacing_mm()[1] = pixel_spacing   (y_mm)
      - dim 2 matches get_spacing_mm()[2] = pixel_spacing   (x_mm)
    and the annotation centroid reorder (c[2], c[1], c[0]) = (k, i, j)
    = (slice, row, col) maps correctly into this (Z, Y, X) space.
    Note:
    pylidc's Scan.to_volume() already applies RescaleSlope/RescaleIntercept
    internally when it builds the volume from the DICOM slices -- it returns
    Hounsfield Units, not raw stored pixel values. Re-applying slope/intercept
    here was double-counting the intercept (confirmed via diag_hu.py: pipeline
    output was a constant `intercept` below the correctly-rescaled value, e.g.
    -1024 HU off on scans with RescaleIntercept=-1024), which pushed soft
    tissue down into air-threshold range and broke lung segmentation.
    """
    vol = scan.to_volume(verbose=False).astype(np.float32)
    # pylidc returns (rows, cols, slices) -- transpose to (slices, rows, cols)
    return np.transpose(vol, (2, 0, 1))


# -----------------------------------------------------------------------------
# Lung segmentation
# -----------------------------------------------------------------------------

def segment_lungs(volume_hu, spacing_mm):
    """Binary lung-field mask via per-slice air segmentation + 3D morphology.

    Whole-volume 3D border-touch rejection is fragile: if lung air merges
    with exterior air through even a single thin bridge (trachea/mouth,
    partial-volume pixels at the body surface, FOV edge artifacts),
    connected-component labeling fuses them into ONE component that
    necessarily touches the scan border -- so a 3D border check throws out
    the entire lung+exterior blob together, on every scan, regardless of
    which axis is checked.

    Instead we clear border-touching air PER AXIAL SLICE (2D), which is the
    standard approach (matches the LUNA16 preprocessing tutorials): a bridge
    on a handful of slices near the trachea can't poison slices where the
    lungs are still cleanly separated from exterior air. On the vast
    majority of slices, clearing 2D border-touching components correctly
    drops exterior air while keeping the two lung cross-sections intact.

    NOTE: this returns the raw AIR-based mask only. It intentionally does
    NOT try to include nodules yet -- see dilate_lung_mask(), which is
    applied on top of this and is what actually makes pleural/vessel
    attached nodules detectable.
    """
    air = volume_hu < -400
    n_slices = air.shape[0]
    lung_air = np.zeros_like(air, dtype=bool)

    # mm-aware minimum component size floor (~ a few mm^2 of in-plane area)
    in_plane_area_mm2 = spacing_mm[1] * spacing_mm[2]
    min_component_px = max(10, int(round(20.0 / in_plane_area_mm2)))

    for z in range(n_slices):
        slice_air = air[z]
        labeled2d, n2d = ndlabel(slice_air)
        if n2d == 0:
            continue

        slice_size = float(slice_air.size)
        kept_labels = []
        for lbl in range(1, n2d + 1):
            comp = labeled2d == lbl
            comp_size = comp.sum()
            if comp_size < min_component_px:  # too small to be a lung cross-section
                continue
            if comp_size > 0.9 * slice_size:  # whole-slice air, e.g. empty scan margin
                continue
            border_touch = (
                comp[0, :].any() or comp[-1, :].any() or
                comp[:, 0].any() or comp[:, -1].any()
            )
            if border_touch:
                continue
            kept_labels.append((lbl, comp_size))

        if not kept_labels:
            continue

        # Keep up to the two largest surviving components (left + right lung
        # cross-section on this slice).
        kept_labels.sort(key=lambda t: t[1], reverse=True)
        for lbl, _ in kept_labels[:2]:
            lung_air[z] |= (labeled2d == lbl)

    if not lung_air.any():
        return np.zeros_like(air, dtype=bool)

    struct3d = generate_binary_structure(3, 2)
    lung_closed = binary_closing(lung_air, structure=struct3d, iterations=4)
    lung_filled = np.zeros_like(lung_closed)
    for z in range(lung_closed.shape[0]):
        lung_filled[z] = binary_fill_holes(lung_closed[z])
    return lung_filled.astype(bool)


def dilate_lung_mask(lung_mask, spacing_mm, margin_mm=LUNG_MASK_MARGIN_MM):
    """
    Grow the air-based lung mask by margin_mm in real-world space.

    This is the key recall fix: a nodule is soft tissue, not air, so
    pleural-attached, chest-wall-attached, and vessel-attached nodules
    can sit centered just outside the raw air mask (segment_lungs()
    above). binary_fill_holes() only rescues nodules fully enclosed by
    air on all sides -- it cannot rescue anything opening onto the
    exterior or mediastinum. Dilating by a physiologically generous
    margin (default 18mm, i.e. bigger than all but the largest nodules)
    ensures the searched region extends past the pleural surface.

    Dilation iterations are computed per-axis in voxels since slice
    thickness and in-plane spacing usually differ substantially.
    """
    if not lung_mask.any():
        return lung_mask
    iters = tuple(max(1, int(round(margin_mm / s))) for s in spacing_mm)
    # binary_dilation's `iterations` is a single int, not per-axis, so we
    # dilate axis-by-axis with a 1D structuring element to respect
    # anisotropic voxel spacing.
    dilated = lung_mask
    for axis, n_iter in enumerate(iters):
        if n_iter <= 0:
            continue
        struct = np.zeros((3, 3, 3), dtype=bool)
        idx = [1, 1, 1]
        idx[axis] = slice(0, 3)
        struct[tuple(idx)] = True
        dilated = binary_dilation(dilated, structure=struct, iterations=n_iter)
    return dilated


def _resolve_detection_mask(mask, volume_shape):
    """Use the lung mask when it contains enough support; otherwise fall back to full volume."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != tuple(volume_shape):
        mask = np.ones(tuple(volume_shape), dtype=bool)
    if mask.sum() < 2000:
        return np.ones(tuple(volume_shape), dtype=bool)
    return mask


# -----------------------------------------------------------------------------
# LoG blob detector
# -----------------------------------------------------------------------------

def _sigma_range(min_diam_mm, max_diam_mm, n_scales):
    """Log-spaced sigmas (mm). For 3-D sphere: sigma = diameter / (2 * sqrt(3))."""
    sigma_min = min_diam_mm / (2.0 * np.sqrt(3.0))
    sigma_max = max_diam_mm / (2.0 * np.sqrt(3.0))
    return np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_scales)


def _nms(candidates, spacing_mm, min_dist_mm):
    """Greedy distance-based non-maximum suppression (stronger response wins)."""
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda c: c['response'], reverse=True)
    kept = []
    for c in ranked:
        cz, cy, cx = c['centroid_zyx']
        too_close = False
        for k in kept:
            kz, ky, kx = k['centroid_zyx']
            dist = np.sqrt(
                ((cz - kz) * spacing_mm[0]) ** 2 +
                ((cy - ky) * spacing_mm[1]) ** 2 +
                ((cx - kx) * spacing_mm[2]) ** 2
            )
            if dist < min_dist_mm:
                too_close = True
                break
        if not too_close:
            kept.append(c)
    return kept


def detect_nodules_log(
    volume_hu,
    lung_mask,
    spacing_mm,
    min_diam_mm    = MIN_DIAM_MM,
    max_diam_mm    = MAX_DIAM_MM,
    n_scales       = N_SCALES,
    threshold      = LOG_THRESHOLD,
    nms_dist_mm    = NMS_MIN_DIST_MM,
    max_candidates = MAX_CANDS_PER_SCAN,
    verbose        = True,
):
    """
    Multi-scale 3D LoG nodule detector.

    Steps:
      1. Clip HU to [-1000, 400] and normalise to [0, 1].
      2. Crop computation to (dilated) lung bounding box + padding.
      3. For each log-spaced sigma, compute scale-normalised LoG response:
            R = -sigma_mm^2 * gaussian_laplace(vol_norm, sigma_vox)
         (positive for bright-on-dark blobs; nodules denser than lung).
         Accumulate element-wise maximum across scales.
      4. Find 3-D local maxima above threshold.
      5. Non-maximum suppression in mm space.
      6. Cap at max_candidates (candidates are sorted by response before
         truncation, so this only ever drops the weakest candidates and
         the cap is set high enough that it should rarely bind).

    `lung_mask` is expected to already be the DILATED mask (see
    dilate_lung_mask) so pleural/vessel-attached nodules aren't excluded
    before the detector even runs.

    Returns list of dicts: centroid_zyx (full-volume voxel space),
    diameter_mm, response, hu_mean.
    """
    t0 = time.time()

    # Step 1: normalise
    HU_MIN, HU_MAX = -1000.0, 400.0
    vol_norm = np.clip(volume_hu, HU_MIN, HU_MAX).astype(np.float32)
    vol_norm = (vol_norm - HU_MIN) / (HU_MAX - HU_MIN)

    # Step 2: lung bounding box + padding
    lung_mask = _resolve_detection_mask(lung_mask, volume_hu.shape)
    coords = np.argwhere(lung_mask)
    if coords.size == 0:
        return []

    sigma_max_mm = max_diam_mm / (2.0 * np.sqrt(3.0))
    pad_vox = min(int(np.ceil(3.0 * sigma_max_mm / min(spacing_mm))), 80)

    bb_min = np.maximum(coords.min(axis=0) - pad_vox, 0)
    bb_max = np.minimum(coords.max(axis=0) + pad_vox + 1,
                        np.array(vol_norm.shape))
    z0, y0, x0 = bb_min
    z1, y1, x1 = bb_max

    vol_crop  = vol_norm[z0:z1, y0:y1, x0:x1]
    mask_crop = lung_mask[z0:z1, y0:y1, x0:x1]

    if verbose:
        print(f'      Lung crop: {vol_crop.shape}  '
              f'(full vol: {vol_norm.shape})', flush=True)

    # Step 3: incremental scale-space maximum
    sigmas_mm  = _sigma_range(min_diam_mm, max_diam_mm, n_scales)
    max_resp   = np.full(vol_crop.shape, -np.inf, dtype=np.float32)
    best_scale = np.zeros(vol_crop.shape, dtype=np.uint8)

    for i, sigma_mm in enumerate(sigmas_mm):
        sigma_vox = tuple(sigma_mm / s for s in spacing_mm)
        resp = (-sigma_mm ** 2) * gaussian_laplace(
            vol_crop, sigma=sigma_vox, mode='reflect'
        )
        resp[~mask_crop] = -np.inf

        improved = resp > max_resp
        max_resp[improved]   = resp[improved]
        best_scale[improved] = i

        if verbose:
            n_above = int((resp[mask_crop] > threshold).sum())
            print(f'      scale {i+1:2d}/{n_scales}  '
                  f'sigma={sigma_mm:.2f} mm  '
                  f'diam~{2*np.sqrt(3)*sigma_mm:.1f} mm  '
                  f'responses>{threshold}: {n_above}', flush=True)

    max_resp = np.where(mask_crop, np.maximum(max_resp, 0.0), 0.0)

    # Step 4: local maxima
    # Footprint floor decoupled from MIN_DIAM_MM (which can now be as low
    # as 2mm -- using it directly here would force a ~1-voxel footprint,
    # i.e. essentially no local suppression and a flood of duplicates).
    fp_radius = max(1, int(np.ceil(LOCAL_MAX_FLOOR_MM / (2.0 * max(spacing_mm)))))
    fp_size   = 2 * fp_radius + 1

    local_max  = (max_resp == maximum_filter(max_resp, size=fp_size))
    local_max &= (max_resp > threshold)
    local_max &= mask_crop

    positions = np.argwhere(local_max)
    if verbose:
        print(f'      {len(positions)} local maxima above threshold '
              f'(footprint {fp_size})', flush=True)

    # Step 5: build candidate list
    candidates = []
    for zc, yc, xc in positions:
        sigma_mm = float(sigmas_mm[best_scale[zc, yc, xc]])
        candidates.append({
            'centroid_zyx': (float(zc + z0), float(yc + y0), float(xc + x0)),
            'diameter_mm':  2.0 * np.sqrt(3.0) * sigma_mm,
            'response':     float(max_resp[zc, yc, xc]),
            'hu_mean':      float(volume_hu[zc + z0, yc + y0, xc + x0]),
        })

    # Step 6: NMS + cap (sorted by response descending inside _nms, so the
    # cap below only discards the weakest candidates once the true count
    # exceeds max_candidates)
    candidates = _nms(candidates, spacing_mm, nms_dist_mm)
    if len(candidates) > max_candidates and verbose:
        print(f'      WARNING: {len(candidates)} candidates exceed cap '
              f'({max_candidates}); truncating weakest responses. '
              f'Consider raising --max-cands if this happens often.',
              flush=True)
    candidates = candidates[:max_candidates]

    if verbose:
        print(f'      {len(candidates)} candidates after NMS '
              f'({time.time() - t0:.1f} s)', flush=True)

    return candidates


# -----------------------------------------------------------------------------
# Patch extraction
# -----------------------------------------------------------------------------

def extract_patch(volume_hu, center_zyx, patch_size=PATCH_SIZE):
    """
    Extract a cubic patch centred on center_zyx.

    If the patch extends outside the volume (boundary nodule), the volume
    is padded with edge-value replication before cropping. This avoids
    discarding near-boundary detections, which are common for nodules in
    the upper or lower lung fields when slice thickness is large.
    """
    cz, cy, cx = (int(round(c)) for c in center_zyx)
    half = patch_size // 2
    Z, Y, X = volume_hu.shape

    # Compute how much padding is needed on each face
    pz0 = max(0, half - cz);         pz1 = max(0, cz + half - Z)
    py0 = max(0, half - cy);         py1 = max(0, cy + half - Y)
    px0 = max(0, half - cx);         px1 = max(0, cx + half - X)

    if pz0 or pz1 or py0 or py1 or px0 or px1:
        vol = np.pad(volume_hu,
                     ((pz0, pz1), (py0, py1), (px0, px1)),
                     mode="edge")
        cz += pz0; cy += py0; cx += px0
    else:
        vol = volume_hu

    patch = vol[cz - half:cz + half,
                cy - half:cy + half,
                cx - half:cx + half]
    return patch.astype(np.float32) if patch.shape == (patch_size,) * 3 else None


# -----------------------------------------------------------------------------
# Annotation loading
# -----------------------------------------------------------------------------

def get_annotation_data(scan):
    annotations = []
    for cluster in scan.cluster_annotations():
        if not cluster:
            continue
        centroids = []
        for ann in cluster:
            try:
                c = ann.centroid
                if callable(c):
                    c = c()
                centroids.append((float(c[2]), float(c[1]), float(c[0])))
            except Exception:
                continue
        if not centroids:
            continue
        centroid_zyx = tuple(np.mean(centroids, axis=0).tolist())
        feat_scores = {}
        for feat in FEATURE_NAMES:
            vals = [float(v) for ann in cluster
                    for v in [_get_feature_value(ann, feat)]
                    if v is not None]
            feat_scores[feat] = _normalize_feature(feat, float(np.mean(vals))) \
                                 if vals else np.nan
        annotations.append({
            'centroid_zyx':  centroid_zyx,
            'scores':        feat_scores,
            'n_annotators':  len(cluster),
        })
    return annotations


# -----------------------------------------------------------------------------
# Matching
# -----------------------------------------------------------------------------

def _dist_mm(a, b, spacing):
    return float(np.sqrt(sum(((a[i] - b[i]) * spacing[i]) ** 2
                             for i in range(3))))


def match_detections(detections, annotations, spacing_mm):
    if not detections or not annotations:
        return {
            'tp_pairs': [],
            'fp_idxs':  list(range(len(detections))),
            'fn_idxs':  list(range(len(annotations))),
        }
    dist = np.full((len(detections), len(annotations)), np.inf)
    for di, det in enumerate(detections):
        for ai, ann in enumerate(annotations):
            dist[di, ai] = _dist_mm(det['centroid_zyx'],
                                     ann['centroid_zyx'], spacing_mm)
    matched_d, matched_a, tp_pairs = set(), set(), []
    for di, ai in np.dstack(
        np.unravel_index(np.argsort(dist, axis=None), dist.shape)
    )[0]:
        if dist[di, ai] > MATCH_THRESHOLD_MM:
            break
        if di in matched_d or ai in matched_a:
            continue
        tp_pairs.append((int(di), int(ai), float(dist[di, ai])))
        matched_d.add(di); matched_a.add(ai)
    return {
        'tp_pairs': tp_pairs,
        'fp_idxs':  [i for i in range(len(detections)) if i not in matched_d],
        'fn_idxs':  [i for i in range(len(annotations)) if i not in matched_a],
    }


# -----------------------------------------------------------------------------
# Per-scan pipeline
# -----------------------------------------------------------------------------

def process_scan(scan, output_dir, verbose=True):
    pid   = scan.patient_id
    t_scan = time.time()

    print(f'\n[{pid}] Loading volume ...', flush=True)
    volume_hu  = load_volume_hu(scan)
    spacing_mm = get_spacing_mm(scan)
    print(f'[{pid}] {volume_hu.shape}  '
          f'spacing=({spacing_mm[0]:.2f},{spacing_mm[1]:.2f},'
          f'{spacing_mm[2]:.2f}) mm', flush=True)

    print(f'[{pid}] Segmenting lungs ...', flush=True)
    lung_mask_air = segment_lungs(volume_hu, spacing_mm)
    lung_mask = dilate_lung_mask(lung_mask_air, spacing_mm,
                                  margin_mm=LUNG_MASK_MARGIN_MM)
    mask_fraction = float(lung_mask.sum()) / float(lung_mask.size)
    print(f'[{pid}] lung mask fraction (dilated +{LUNG_MASK_MARGIN_MM:.0f}mm): '
          f'{mask_fraction:.3f} ({lung_mask.sum()} / {lung_mask.size})',
          flush=True)

    print(f'[{pid}] Running LoG detector ...', flush=True)
    detections = detect_nodules_log(volume_hu, lung_mask, spacing_mm,
                                    min_diam_mm=MIN_DIAM_MM,
                                    max_diam_mm=MAX_DIAM_MM,
                                    n_scales=N_SCALES,
                                    threshold=LOG_THRESHOLD,
                                    nms_dist_mm=NMS_MIN_DIST_MM,
                                    max_candidates=MAX_CANDS_PER_SCAN,
                                    verbose=verbose)
    print(f'[{pid}] {len(detections)} candidates after NMS', flush=True)

    print(f'[{pid}] Loading annotations ...', flush=True)
    annotations = get_annotation_data(scan)
    print(f'[{pid}] {len(annotations)} annotated nodules', flush=True)

    # Padded extraction: every candidate yields a valid patch now
    os.makedirs(output_dir, exist_ok=True)
    valid_dets, patch_paths = [], []
    for det in detections:
        patch = extract_patch(volume_hu, det['centroid_zyx'])
        if patch is None:          # should never happen with padding
            continue
        cand_id  = f'{pid}_cand{len(valid_dets):03d}'
        npy_path = os.path.join(output_dir, f'{cand_id}.npy')
        np.save(npy_path, patch)
        valid_dets.append(det)
        patch_paths.append(os.path.abspath(npy_path))
    print(f'[{pid}] {len(valid_dets)} patches saved', flush=True)

    match_result = match_detections(valid_dets, annotations, spacing_mm)
    tp_set    = {di: (ai, d) for di, ai, d in match_result['tp_pairs']}
    ann_by_di = {di: annotations[ai]
                 for di, ai, _ in match_result['tp_pairs']}

    if verbose and annotations:
        print(f'[{pid}] Diagnostic: annotation vs nearest candidate', flush=True)
        for ai, ann in enumerate(annotations):
            best_di = None
            best_dist = np.inf
            ann_z, ann_y, ann_x = ann['centroid_zyx']
            for di, det in enumerate(valid_dets):
                dist = _dist_mm(det['centroid_zyx'], ann['centroid_zyx'], spacing_mm)
                if dist < best_dist:
                    best_dist = dist
                    best_di = di
            if best_di is None:
                print(
                    f'  ann {ai:02d}: ann=({ann_z:.2f},{ann_y:.2f},{ann_x:.2f})  no candidates',
                    flush=True,
                )
            else:
                det = valid_dets[best_di]
                det_z, det_y, det_x = det['centroid_zyx']
                print(
                    f'  ann {ai:02d}: ann=({ann_z:.2f},{ann_y:.2f},{ann_x:.2f})  '
                    f'nearest cand{best_di}=({det_z:.2f},{det_y:.2f},{det_x:.2f})  '
                    f'dist={best_dist:.2f} mm',
                    flush=True,
                )

    if verbose and match_result['tp_pairs']:
        print(f'[{pid}] Matched pairs:', flush=True)
        for di, ai, dist in match_result['tp_pairs']:
            det = valid_dets[di]
            ann = annotations[ai]
            det_z, det_y, det_x = det['centroid_zyx']
            ann_z, ann_y, ann_x = ann['centroid_zyx']
            print(
                f'  TP cand{di}: ann{ai}  dist={dist:.2f} mm  '
                f'det=({det_z:.2f},{det_y:.2f},{det_x:.2f})  '
                f'ann=({ann_z:.2f},{ann_y:.2f},{ann_x:.2f})',
                flush=True,
            )

    rows = []
    for di, det in enumerate(valid_dets):
        is_tp   = di in tp_set
        ann     = ann_by_di.get(di)
        dist_mm = tp_set[di][1] if is_tp else np.nan
        row = {
            'candidate_id':      f'{pid}_cand{di:03d}',
            'patient_id':        pid,
            'file_path':         patch_paths[di],
            'status':            'TP' if is_tp else 'FP',
            'centroid_z':        round(det['centroid_zyx'][0], 1),
            'centroid_y':        round(det['centroid_zyx'][1], 1),
            'centroid_x':        round(det['centroid_zyx'][2], 1),
            'diameter_mm':       round(det['diameter_mm'], 1),
            'log_response':      round(det['response'], 4),
            'hu_mean':           round(det['hu_mean'], 1),
            'match_distance_mm': round(dist_mm, 1) if not np.isnan(dist_mm)
                                 else np.nan,
        }
        for feat in FEATURE_NAMES:
            row[f'{feat}_gt'] = round(ann['scores'][feat], 4) if ann else np.nan
        rows.append(row)

    for ai in match_result['fn_idxs']:
        ann = annotations[ai]
        row = {
            'candidate_id':      f'{pid}_fn{ai:03d}',
            'patient_id':        pid,
            'file_path':         '',
            'status':            'FN',
            'centroid_z':        round(ann['centroid_zyx'][0], 1),
            'centroid_y':        round(ann['centroid_zyx'][1], 1),
            'centroid_x':        round(ann['centroid_zyx'][2], 1),
            'diameter_mm':       np.nan,
            'log_response':      np.nan,
            'hu_mean':           np.nan,
            'match_distance_mm': np.nan,
        }
        for feat in FEATURE_NAMES:
            row[f'{feat}_gt'] = round(ann['scores'][feat], 4)
        rows.append(row)

    tp = len(match_result['tp_pairs'])
    fp = len(match_result['fp_idxs'])
    fn = len(match_result['fn_idxs'])
    n_ann = tp + fn; n_det = tp + fp
    print(f'[{pid}] TP={tp}  FP={fp}  FN={fn}  '
          f'Sensitivity={tp/n_ann*100:.0f}%  '
          f'Precision={tp/n_det*100:.0f}%  '
          f'({time.time()-t_scan:.0f} s)', flush=True)

    return pd.DataFrame(rows), (tp, fp, fn)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='LoG nodule detection + annotation matching for LIDC-IDRI '
                    '(recall-first tuning: prioritizes finding every nodule '
                    'over minimizing false positives).'
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--patient-id',  type=str)
    g.add_argument('--patient-ids', nargs='+')
    g.add_argument('--all-scans',   action='store_true')
    p.add_argument('--output-dir',  default='candidate_patches')
    p.add_argument('--resume',      action='store_true',
                   help='Skip patients already in candidates_manifest.csv')
    p.add_argument('--threshold',    type=float, default=LOG_THRESHOLD)
    p.add_argument('--n-scales',     type=int,   default=N_SCALES)
    p.add_argument('--max-cands',    type=int,   default=MAX_CANDS_PER_SCAN)
    p.add_argument('--mask-margin-mm', type=float, default=LUNG_MASK_MARGIN_MM,
                   help='Lung mask dilation margin in mm (catches pleural/'
                        'vessel-attached nodules; larger = safer but slower)')
    p.add_argument('--quiet',       action='store_true',
                   help='Suppress per-scale progress lines')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    global LOG_THRESHOLD, N_SCALES, MAX_CANDS_PER_SCAN, LUNG_MASK_MARGIN_MM
    LOG_THRESHOLD       = args.threshold
    N_SCALES            = args.n_scales
    MAX_CANDS_PER_SCAN  = args.max_cands
    LUNG_MASK_MARGIN_MM = args.mask_margin_mm

    if args.patient_id:
        patient_ids = [args.patient_id]
    elif args.patient_ids:
        patient_ids = args.patient_ids
    else:
        patient_ids = None

    scans = (pl.query(pl.Scan).all() if patient_ids is None else
             [s for pid in patient_ids
              for s in [pl.query(pl.Scan)
                        .filter(pl.Scan.patient_id == pid).first()]
              if s is not None])

    manifest_path = os.path.join(args.output_dir, 'candidates_manifest.csv')

    already_done = set()
    if args.resume and os.path.isfile(manifest_path):
        existing = pd.read_csv(manifest_path)
        already_done = set(existing['patient_id'].unique())
        print(f'Resume: {len(already_done)} patients already done, skipping.',
              flush=True)

    scans_to_run = [s for s in scans if s.patient_id not in already_done]
    print(f'{len(scans_to_run)} scan(s) to process.', flush=True)

    all_frames    = []
    summary_lines = []
    total_tp = total_fp = total_fn = 0

    if args.resume and os.path.isfile(manifest_path):
        all_frames.append(pd.read_csv(manifest_path))

    t_total = time.time()
    for i, scan in enumerate(scans_to_run, 1):
        try:
            df, (tp, fp, fn) = process_scan(
                scan, args.output_dir, verbose=not args.quiet
            )
            all_frames.append(df)
            total_tp += tp; total_fp += fp; total_fn += fn
            summary_lines.append(
                f'{scan.patient_id}: TP={tp} FP={fp} FN={fn}'
            )
            # Write after every scan so interruption loses at most one scan
            pd.concat(all_frames, ignore_index=True).to_csv(
                manifest_path, index=False
            )
            elapsed = time.time() - t_total
            eta_h   = (elapsed / i) * (len(scans_to_run) - i) / 3600
            print(f'  Progress: [{i}/{len(scans_to_run)}]  ETA {eta_h:.1f} h',
                  flush=True)
        except Exception as e:
            print(f'ERROR on {scan.patient_id}: {e}', flush=True)
            import traceback; traceback.print_exc()

    n_ann = total_tp + total_fn
    n_det = total_tp + total_fp
    summary_lines += [
        '',
        f'TOTAL  TP={total_tp}  FP={total_fp}  FN={total_fn}',
        f'Sensitivity : {total_tp/n_ann*100:.1f}%' if n_ann else 'Sensitivity: N/A',
        f'Precision   : {total_tp/n_det*100:.1f}%' if n_det else 'Precision: N/A',
    ]
    summary_txt = os.path.join(args.output_dir, 'detection_summary.txt')
    with open(summary_txt, 'w') as f:
        f.write('\n'.join(summary_lines))

    print('\n' + '\n'.join(summary_lines))
    print(f'\nManifest : {manifest_path}')
    print(f'Summary  : {summary_txt}')
    print('\nNext: upload candidate_patches/ to Drive, '
          'then run detect_validate_colab.ipynb on Colab.')


if __name__ == '__main__':
    main()