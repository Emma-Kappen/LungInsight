"""
02_mask_and_crop.py

STEP 2 of 3 in the pipeline:
    01_dicom_to_hu.py       -> DICOM -> HU volume
    02_mask_and_crop.py     <- this file: lung segmentation + non-lung
                               blanking + Z-crop
    03_visualize.py         -> viewing

Loads volume_hu.npy / meta.json produced by 01_dicom_to_hu.py, computes
two masks, blanks out non-lung anatomy, and crops the volume down to
just the lung-containing slices:

  * air_mask.npy  -> PHYSICAL air only (HU <= -950): background air
    outside the patient + trachea/large airways. Does NOT include
    normal lung parenchyma, which is denser than true air.
  * lung_mask.npy -> the full anatomical LUNG REGION -- both lungs'
    aerated parenchyma, the vessels/airway walls/nodules inside it,
    the trachea and main bronchi, AND a physical margin covering the
    lung wall (pleura). Explicitly excludes the heart/great vessels
    (see heart_mask.npy below).
  * heart_mask.npy -> dedicated heart + great-vessel mask, built from
    the mediastinal region that the lung silhouette rings around at
    cardiac-level slices (see compute_heart_mask()). Empty if no
    qualifying candidate was found.
  * volume_hu_masked.npy -> volume_hu with every voxel OUTSIDE
    lung_mask blanked to physical-air HU. This is the "apply the mask"
    deliverable: whatever organs survive the Z-crop margin (liver,
    stomach, bowel, chest wall, etc.) are erased at the voxel level,
    not just left in as un-highlighted background.

By default the volume/masks are also cropped along Z to just the
slices that contain lung (plus a margin) -- neck, shoulder, and
abdomen slices are dropped. Pass --no-z-crop to keep every slice.

=== Why the Z-crop logic changed from a per-slice "2 components" test ===

A previous version of this pipeline decided whether an axial slice
"contains lung" by checking, independently per 2D slice, whether
lung_mask had >= 2 separate connected components of meaningful size --
the idea being "two side-by-side air cavities = left lung + right
lung". That test is NOT anatomically specific, and fails in both
directions:

  * FALSE POSITIVES (abdominal slices let back in): a gas-filled
    stomach bubble plus a bowel-gas loop are two SEPARATE 2D
    components in a single abdominal slice. Per-slice, that looks
    identical to "left lung + right lung" -- the test has no way to
    know these two blobs aren't lungs, because it only ever looks
    within one slice at a time.
  * FALSE NEGATIVES (real expiration lung bases dropped): on
    expiration, the diaphragm domes rise unevenly, so a real lung-base
    slice can have aerated tissue on only ONE side at that exact Z
    level even though it's genuinely still lung. A "must have 2
    components in THIS slice" test throws that slice away.

The root problem is that "2D component count, checked independently
per slice" is a symmetry heuristic, not an anatomical one. It has no
concept of which blob is actually contiguous with the lungs in 3D.

The fix here uses actual 3D connectivity instead, in two stages that
do two different jobs:

  1. ORGAN IDENTITY (3D): after thresholding + per-slice border
     removal + a 3D morphological closing (which physically bridges
     the trachea to both lungs into ONE connected 3D component), we
     label connected components in full 3D and keep only the
     component(s) whose voxel count is within `core_relative_size_
     threshold` of the single largest component. A stomach bubble or
     bowel-gas loop is a SEPARATE 3D component from the bridged
     lung+airway tree (the diaphragm physically separates them), and
     is essentially always much smaller than the full bilateral lung
     volume, so it's excluded here regardless of how many 2D
     components it happens to form in any one slice. This is the part
     that actually distinguishes "lung tissue" from "abdominal gas" --
     and it's inherently immune to the single-slice symmetry trap,
     because it never looks at one slice in isolation.

  2. Z-EXTENT (per-slice area, on the ALREADY-VERIFIED lung
     component only): the bridged component still reaches up into the
     neck as a thin trachea tube, which would push the Z-crop too far
     superior if we just took the component's full Z range. So, on
     that same verified-lung component (never on the raw lung_mask,
     and never comparing against abdominal blobs -- those are already
     gone), we do a simple per-slice AREA-FRACTION test: does this
     slice's cross-section of the verified lung component cover a
     lung-sized area, or just a thin tracheal lumen? This needs no
     bilateral symmetry at all, so a one-sided expiration lung base
     still passes (it's real lung-sized area, just on one side), while
     a thin trachea-only neck slice correctly fails.

Net effect: organ identity comes from full 3D connectivity (fixes the
stomach/bowel false-positive), and the Z boundary comes from area on
that verified structure rather than a bilateral count (fixes the
uneven-diaphragm expiration false-negative).

=== Follow-up fix: identity must run BEFORE bridging, not after ===

Even with full 3D connectivity, an earlier version of this file still
let gastric/bowel gas leak into lung_mask, because it ran the
trachea-bridging 3D closing step FIRST and only labeled/size-filtered
components afterward. Closing bridges gaps up to roughly twice
`closing_radius_mm` (default 5mm, so up to ~10mm) -- but the diaphragm
separating the lung base from the stomach is often only 2-5mm thick,
especially right under a domed hemidiaphragm at end-expiration. That
closing step could physically weld a stomach/bowel gas pocket onto the
real lung base into a single 3D component before the size-relative
"is this really lung-sized" test ever saw them as separate -- and once
welded, the combined component's size was dominated by real lung
tissue, so it sailed through the size check with the gas pocket riding
along for free.

The fix is to swap the order: run the 3D organ-identity/size check on
the un-bridged mask first (where the diaphragm has not been
artificially erased, so real anatomy alone -- not a closing artifact
-- decides what counts as "lung"), and only run the trachea-bridging
closing step afterward, on that already-verified lung-only core. The
core by construction contains no stomach/bowel voxels for the closing
step to weld onto, so it can still bridge trachea/bronchial-wall gaps
within real lung tissue without ever being able to reach abdominal gas.

Usage:
    python 02_mask_and_crop.py output/LIDC-IDRI-0001_processed \
        --out-dir output/LIDC-IDRI-0001_masked \
        --air-threshold -950 --lung-threshold -320 --z-crop-margin-mm 10

Outputs (written to --out-dir):
    volume_hu.npy        -> float32 (Z, Y, X), Z-cropped, UNMASKED HU
                            (full tissue values inside the crop range --
                            still useful for soft-tissue/bone windows)
    volume_hu_masked.npy -> float32 (Z, Y, X), Z-cropped, with every
                            voxel outside lung_mask blanked to air HU
    air_mask.npy          -> bool, same shape, True = physical air
    lung_mask.npy         -> bool, same shape, True = lung region
    meta.json              -> updated with mask stats + crop range
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    from scipy import ndimage
except ImportError:
    print(
        "scipy is required for lung segmentation. Install it with:\n"
        "    pip install scipy --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(1)


AIR_REPLACEMENT_HU = -1000.0


def load_hu_volume(processed_dir: str):
    """Load volume_hu.npy + meta.json written by 01_dicom_to_hu.py."""
    vol_path = os.path.join(processed_dir, "volume_hu.npy")
    meta_path = os.path.join(processed_dir, "meta.json")

    if not os.path.exists(vol_path):
        raise FileNotFoundError(
            f"'{vol_path}' not found. Run 01_dicom_to_hu.py first, and "
            f"pass its --out-dir here."
        )

    volume_hu = np.load(vol_path)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return volume_hu, meta


def compute_air_mask(volume_hu: np.ndarray, air_threshold: float = -950.0) -> np.ndarray:
    """
    Boolean mask, True where a voxel is PHYSICAL air (near-vacuum
    density): water = 0 HU by definition, true air = -1000 HU by
    definition, so a -950 HU cutoff (with a little slack for scanner
    noise) catches background air outside the patient and the
    trachea / large airways.

    IMPORTANT: this does NOT capture normal lung parenchyma. Aerated
    lung tissue is a mix of air and tissue and typically reads around
    -950 to -500 HU -- entirely above this threshold. If you want the
    lung fields themselves, use compute_lung_mask() instead.
    """
    return volume_hu <= air_threshold


def _ellipsoid_structure(radii_vox):
    """
    Build a boolean ellipsoidal structuring element sized in VOXELS
    along each axis. Needed because CT voxels are usually anisotropic
    (slice spacing != in-plane pixel spacing), so a physically round
    (in mm) dilation/closing footprint is NOT a round footprint in
    voxel space -- a plain cube/sphere structuring element would
    over-grow in one axis and under-grow in another.
    """
    rz, ry, rx = [max(1, int(round(r))) for r in radii_vox]
    zz, yy, xx = np.ogrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    ellipsoid = (zz / rz) ** 2 + (yy / ry) ** 2 + (xx / rx) ** 2 <= 1.0
    return ellipsoid


def _fill_small_holes_2d(mask_2d: np.ndarray, max_hole_area_voxels: float, structure_2d):
    """
    Fill enclosed background ("hole") regions in a 2D mask, but ONLY
    holes up to `max_hole_area_voxels` in size. Plain
    scipy.ndimage.binary_fill_holes fills every enclosed hole
    regardless of size -- which is exactly wrong at a cardiac-level
    slice, where the two lungs can wrap around and meet both in front
    of and behind the mediastinum, turning the pair into a single ring
    shape. A plain fill_holes on a ring fills its ENTIRE interior,
    silently absorbing the heart, great vessels, and esophagus into
    the "lung" mask.

    Real intrapulmonary holes (vessels, airway walls, nodules, even
    fairly large masses) are essentially always far smaller in
    per-slice cross-section than the whole mediastinal compartment, so
    capping the fillable hole size gives normal intra-lung filling
    while leaving a heart-sized (or larger) enclosed region alone.

    A "hole" here means a background component that does NOT touch the
    slice's outer border -- border-touching background is true outside
    air, never a hole, and is never filled regardless of size.

    Returns (filled_mask, large_holes_mask): `large_holes_mask` is
    True where an enclosed hole existed but was too big to fill --
    exactly the mediastinal candidate region compute_heart_mask() uses
    to build a dedicated heart mask.
    """
    if not mask_2d.any():
        return mask_2d, np.zeros_like(mask_2d)

    background = ~mask_2d
    labeled_bg, n_bg = ndimage.label(background, structure=structure_2d)
    if n_bg == 0:
        return mask_2d, np.zeros_like(mask_2d)

    border_labels = set(labeled_bg[0, :].tolist())
    border_labels |= set(labeled_bg[-1, :].tolist())
    border_labels |= set(labeled_bg[:, 0].tolist())
    border_labels |= set(labeled_bg[:, -1].tolist())
    border_labels.discard(0)

    sizes = ndimage.sum(background, labeled_bg, index=np.arange(1, n_bg + 1))
    fill_labels = []
    large_labels = []
    for lbl in range(1, n_bg + 1):
        if lbl in border_labels:
            continue
        if sizes[lbl - 1] <= max_hole_area_voxels:
            fill_labels.append(lbl)
        else:
            large_labels.append(lbl)

    filled = mask_2d
    if fill_labels:
        filled = mask_2d | np.isin(labeled_bg, fill_labels)

    large_holes = np.isin(labeled_bg, large_labels) if large_labels else np.zeros_like(mask_2d)
    return filled, large_holes


def compute_heart_mask(
    large_holes_stack: np.ndarray,
    pixel_spacing_mm=(1.0, 1.0),
    slice_spacing_mm: float = 1.0,
    min_span_mm: float = 25.0,
    min_component_fraction: float = 0.25,
):
    """
    Build a dedicated heart mask from the "too big to fill" enclosed
    regions gathered while segmenting the lungs (see
    _fill_small_holes_2d). At cardiac-level slices, the lung silhouette
    rings around the mediastinum, leaving exactly one large enclosed
    hole per slice there -- the heart + great vessels + esophagus. That
    structure is large, roughly central, and (unlike a one-off
    unfillable cavity elsewhere) spans many contiguous slices, so it's
    identified the same way the lung's own airway tree is identified
    in compute_lung_mask: by 3D connectivity and relative size, not by
    any single-slice heuristic.

    Steps:
      1. Label `large_holes_stack` in full 3D.
      2. Keep only the single largest connected component (the heart
         is essentially always the biggest mediastinal structure that
         gets excluded from lung filling), provided it's also at least
         `min_component_fraction` of the total "large hole" volume
         (guards against a degenerate case where several similar-sized
         candidates exist) and spans at least `min_span_mm` of Z
         (guards against a one-off unfillable single-slice blob, e.g.
         an unusually large intrapulmonary cavity, being mistaken for
         the heart).
      3. Fill any small internal holes per slice (plain fill_holes is
         fine here -- the heart itself isn't a ring, so this step just
         smooths minor internal gaps rather than risking swallowing
         another organ).

    Returns heart_mask (bool array, same shape as large_holes_stack).
    Empty (all False) if no qualifying candidate is found.

    Caveat: a true intrapulmonary mass large enough to exceed
    max_hole_area_mm2 in compute_lung_mask AND coincidentally spanning
    many slices could in principle be mistaken for the heart by size
    and shape alone -- this function has no independent way to confirm
    cardiac anatomy (e.g. no atlas/position prior). Sanity-check the
    output where large intrapulmonary masses are suspected, and adjust
    max_hole_area_mm2 upward if needed.
    """
    empty = np.zeros_like(large_holes_stack, dtype=bool)
    if not large_holes_stack.any():
        return empty

    structure_3d = np.ones((3, 3, 3), dtype=bool)
    labeled, num_features = ndimage.label(large_holes_stack, structure=structure_3d)
    if num_features == 0:
        return empty

    sizes = ndimage.sum(large_holes_stack, labeled, index=np.arange(1, num_features + 1))
    total_size = sizes.sum()
    largest_label = int(np.argmax(sizes)) + 1
    largest_size = sizes[largest_label - 1]

    if largest_size < min_component_fraction * total_size:
        return empty

    candidate = labeled == largest_label
    z_indices = np.where(candidate.any(axis=(1, 2)))[0]
    if len(z_indices) == 0:
        return empty

    span_mm = (z_indices.max() - z_indices.min()) * slice_spacing_mm
    if span_mm < min_span_mm:
        return empty

    heart_mask = candidate.copy()
    for z in range(heart_mask.shape[0]):
        if heart_mask[z].any():
            heart_mask[z] = ndimage.binary_fill_holes(heart_mask[z])

    return heart_mask


def compute_lung_mask(
    volume_hu: np.ndarray,
    lung_threshold: float = -320.0,
    min_component_fraction: float = 0.0008,
    core_relative_size_threshold: float = 0.12,
    max_hole_area_mm2: float = 2500.0,
    pixel_spacing_mm=(1.0, 1.0),
    slice_spacing_mm: float = 1.0,
    closing_radius_mm: float = 5.0,
    wall_margin_mm: float = 3.0,
):
    """
    Segment the full lung REGION -- aerated parenchyma, the vessels/
    airway walls/nodules inside it, the trachea and main bronchi, and
    a margin covering the lung wall (pleura) -- as a boolean mask,
    True = lung region.

    Steps:
      1. Threshold at `lung_threshold` (default -320 HU).
      2. Remove whichever connected component(s) touch each slice's
         outer border -- that's always background air surrounding the
         patient, done per-slice so a lung apex/base legitimately
         reaching the top/bottom of the stack isn't mistaken for it.
      3. *3D organ identity, done BEFORE any bridging*: label connected
         components in full 3D on the un-bridged mask and keep the
         component(s) whose size is within `core_relative_size_
         threshold` of the single largest component (plus an absolute
         noise floor, `min_component_fraction` of the volume). Doing
         this before closing is deliberate: the diaphragm is often only
         2-5mm thick, thinner than the closing step's own bridging
         radius, so running closing first can physically weld a
         stomach bubble or bowel-gas loop onto the lung base into one
         3D component -- and once fused, its size is dominated by the
         real lung tissue, so it passes the size check riding along
         with it. Checking identity on the un-bridged mask means a gas
         pocket has to be disconnected in true anatomy, not just after
         an artificial bridge, to be excluded -- and it always is,
         since the diaphragm genuinely separates them and it is
         essentially always much smaller than the full bilateral lung
         volume, however many components it forms within any single 2D
         slice. This step also records `core_z_indices`: the actual Z
         indices (in this verified, un-bridged component) that contain
         lung tissue, which the caller uses for Z-cropping (see module
         docstring) instead of any per-slice symmetry test.
      4. 3D binary closing, applied only to the just-verified lung
         core, with an mm-sized ellipsoidal structuring element. This
         bridges the small physical gaps between the trachea, the main
         bronchi, and the two lung fields so they render as ONE
         connected structure. Because it runs on the verified core --
         which by construction contains no stomach/bowel voxels -- it
         has nothing non-pulmonary nearby left to weld onto.
      5. Fill holes per axial slice, so vessels/airway walls/nodules
         inside the lung silhouette are included rather than excluded
         -- but ONLY up to `max_hole_area_mm2` in cross-sectional
         area. At a cardiac-level slice, the two lungs can wrap around
         and meet both anteriorly and posteriorly around the
         mediastinum, forming a single ring shape; a plain,
         size-unaware hole fill would then fill the ENTIRE ring
         interior, silently absorbing the heart, great vessels, and
         esophagus into "lung". Real intrapulmonary holes (vessels,
         nodules, even sizeable masses) are essentially always far
         smaller in cross-section than the whole mediastinal
         compartment, so capping the fillable hole size preserves
         normal intra-lung filling while leaving heart-sized enclosed
         regions alone (see _fill_small_holes_2d). The regions that
         were too big to fill are collected across all slices and
         handed to compute_heart_mask() to build a dedicated heart
         mask, which is then explicitly subtracted from the lung mask
         as a belt-and-suspenders step (so a later dilation can't
         re-absorb any of it at the boundary).
      6. Dilate outward by `wall_margin_mm` (mm-aware, anisotropic) to
         pull in a rim of the lung wall itself -- the pleura and the
         walls of the trachea/bronchi, which are denser than
         `lung_threshold` and would otherwise be excluded even though
         they're literally the boundary of the organ. A final
         size-limited hole-fill cleans up anything the dilation
         reopened, and the heart mask is subtracted again afterward.

    Returns (lung_mask, core_z_indices, heart_mask):
      lung_mask      -- bool array, same shape as volume_hu
      core_z_indices -- 1D int array of Z indices where the verified
                        (pre-dilation) 3D lung component has any
                        cross-sectional area at all. Empty if no lung
                        tissue was found.
      heart_mask     -- bool array, same shape as volume_hu; the
                        dedicated heart/great-vessel mask (see
                        compute_heart_mask()). Empty if no qualifying
                        candidate was found.

    Note: this is a solid general-purpose segmentation but not a
    clinical-grade one -- pathology that makes lung tissue much denser
    (e.g. large consolidations, pleural effusion) can locally break
    the threshold assumption, and `wall_margin_mm` is a fixed
    approximation of the pleura, not a true tissue-boundary detector.
    """
    binary = volume_hu < lung_threshold

    structure_2d = np.ones((3, 3), dtype=bool)
    interior = np.zeros_like(binary)
    for z in range(binary.shape[0]):
        slice_binary = binary[z]
        if not slice_binary.any():
            continue
        labeled_slice, _ = ndimage.label(slice_binary, structure=structure_2d)
        border_labels = set(labeled_slice[0, :].tolist())
        border_labels |= set(labeled_slice[-1, :].tolist())
        border_labels |= set(labeled_slice[:, 0].tolist())
        border_labels |= set(labeled_slice[:, -1].tolist())
        border_labels.discard(0)
        interior[z] = slice_binary & ~np.isin(labeled_slice, list(border_labels))

    empty_mask = np.zeros_like(volume_hu, dtype=bool)
    empty_z = np.array([], dtype=int)
    if not interior.any():
        return empty_mask, empty_z, empty_mask.copy()

    # Physical-to-voxel radii for the mm-sized structuring elements
    # (z spacing is almost always coarser than in-plane spacing, so
    # this must be anisotropic to behave correctly in physical space).
    py, px = pixel_spacing_mm
    closing_radii_vox = (closing_radius_mm / slice_spacing_mm,
                         closing_radius_mm / py,
                         closing_radius_mm / px)
    closing_structure = _ellipsoid_structure(closing_radii_vox)

    # --- 3D organ identity FIRST, before any bridging. ---
    #
    # BUG FIXED HERE: the previous version ran the 3D closing (bridging
    # step) *before* labeling/size-filtering components. That closing
    # uses an isotropic-in-mm structuring element sized by
    # `closing_radius_mm` (default 5mm, i.e. it can bridge gaps up to
    # ~10mm). The diaphragm separating the lung base from the stomach/
    # bowel is frequently only 2-5mm thick, especially right under a
    # domed hemidiaphragm at end-expiration -- exactly the geometry in
    # the screenshot that motivated this fix. So the closing step could
    # physically weld a gastric/bowel gas pocket onto the lung base
    # into ONE 3D component *before* the size-relative "is this really
    # lung-sized" test ever ran. Once fused, that component's total
    # size was dominated by the real lung tissue, so it sailed through
    # `core_relative_size_threshold` with the gas pocket riding along --
    # defeating the very check that was supposed to reject it.
    #
    # The fix is to swap the order: label and size-filter components on
    # the un-bridged `interior` mask, where the diaphragm has not been
    # artificially erased. Real anatomy alone (no closing artifact) is
    # enough to identify lung tissue here -- the left and right lungs
    # are each independently large (comfortably >= 12% of the larger
    # one), while a stomach/bowel gas pocket is a genuinely separate,
    # much smaller 3D component with no path to the lungs at all. Only
    # *after* this verified lung-only core is established do we run the
    # closing step, and we run it on the core alone -- so it can still
    # bridge trachea/bronchial-wall gaps within real lung tissue, but it
    # no longer has any stomach/bowel voxels sitting nearby to weld onto.
    structure_3d = np.ones((3, 3, 3), dtype=bool)
    labeled, num_features = ndimage.label(interior, structure=structure_3d)
    if num_features == 0:
        return empty_mask, empty_z, empty_mask.copy()

    sizes = ndimage.sum(interior, labeled, index=np.arange(1, num_features + 1))
    largest_size = sizes.max()
    abs_floor = min_component_fraction * volume_hu.size

    # Keep components close in size to the largest one (handles left
    # lung + right lung as two separate components -- they no longer
    # need pre-bridging to both pass, since each is independently
    # lung-sized), but reject anything much smaller -- which is exactly
    # what an abdominal gas pocket is relative to the full bilateral
    # lung volume, regardless of how many 2D components it forms within
    # a single slice, and regardless of how close it sits to the
    # diaphragm.
    keep_mask = (sizes >= abs_floor) & (sizes >= core_relative_size_threshold * largest_size)
    keep_labels = list(np.nonzero(keep_mask)[0] + 1)  # ndimage labels are 1-indexed

    core_mask_preclosing = np.isin(labeled, keep_labels)
    core_z_indices = np.where(core_mask_preclosing.any(axis=(1, 2)))[0]

    # Bridge the trachea/bronchi to the already-verified lung fields so
    # the whole airway tree renders as one connected structure. Applied
    # to the verified core only, this can no longer create a false path
    # from stomach/bowel gas into "lung" -- there is no stomach/bowel
    # tissue left in the mask for it to reach.
    lung_mask = ndimage.binary_closing(core_mask_preclosing, structure=closing_structure)

    # Physical area-per-voxel in-plane, used to convert max_hole_area_mm2
    # into a voxel count for the size-limited hole fill.
    max_hole_area_voxels = max_hole_area_mm2 / (py * px)

    # Fill holes per axial slice (small ones only -- see docstring and
    # _fill_small_holes_2d) so vessels/nodules inside the lung
    # silhouette are included rather than excluded as "not air", while
    # a heart-sized enclosed region at a cardiac-level slice is left
    # alone. Gather the too-big-to-fill regions to build a dedicated
    # heart mask afterward.
    large_holes_stack = np.zeros_like(lung_mask)
    for z in range(lung_mask.shape[0]):
        if lung_mask[z].any():
            filled, large_holes = _fill_small_holes_2d(lung_mask[z], max_hole_area_voxels, structure_2d)
            lung_mask[z] = filled
            large_holes_stack[z] = large_holes

    heart_mask = compute_heart_mask(
        large_holes_stack, pixel_spacing_mm=pixel_spacing_mm, slice_spacing_mm=slice_spacing_mm,
    )
    # Belt-and-suspenders: make sure the heart never ends up in the
    # lung mask even before the wall-margin dilation below.
    lung_mask &= ~heart_mask

    # Grow outward by a physical margin to pull in the lung wall
    # (pleura) and the walls of the trachea/main bronchi -- these are
    # denser than lung_threshold so thresholding alone always excludes
    # them, even though they're literally the organ's own boundary.
    if wall_margin_mm > 0:
        wall_radii_vox = (wall_margin_mm / slice_spacing_mm,
                          wall_margin_mm / py,
                          wall_margin_mm / px)
        wall_structure = _ellipsoid_structure(wall_radii_vox)
        lung_mask = ndimage.binary_dilation(lung_mask, structure=wall_structure)
        for z in range(lung_mask.shape[0]):
            if lung_mask[z].any():
                lung_mask[z], _ = _fill_small_holes_2d(lung_mask[z], max_hole_area_voxels, structure_2d)
        # The wall-margin dilation can grow the lung mask right back
        # onto/into the heart boundary (they're anatomically adjacent,
        # with no real gap) -- subtract the heart mask again so this
        # dilation can't silently re-absorb any of it.
        lung_mask &= ~heart_mask

    return lung_mask, core_z_indices, heart_mask


def blank_non_lung(volume_hu: np.ndarray, lung_mask: np.ndarray,
                    blank_value: float = AIR_REPLACEMENT_HU) -> np.ndarray:
    """
    Return a copy of volume_hu with every voxel OUTSIDE lung_mask set
    to `blank_value` (physical-air HU by default). This is the actual
    "apply the mask, blank out non-lung organs" step: whatever slices
    survive the Z-crop margin around the lungs (a rim of chest wall,
    partial abdominal organs near the diaphragm, etc.) get erased at
    the voxel level rather than left in as unmarked background.
    """
    masked = volume_hu.copy()
    masked[~lung_mask] = blank_value
    return masked


def crop_to_lung_slices(
    volume_hu, air_mask, lung_mask, heart_mask, meta, core_z_indices,
    margin_mm: float = 10.0,
    z_crop_min_area_frac: float = 0.008,
):
    """
    Trim the Z axis down to only the slices that actually contain lung
    (plus a physical margin), dropping neck/shoulder/abdomen slices.
    Crops volume_hu, air_mask, lung_mask, AND heart_mask together so
    they all stay aligned.

    Only crops Z (not Y/X) -- "slices" means axial slices here.

    `core_z_indices` comes from compute_lung_mask()'s 3D-verified lung
    component (see that function's docstring and the module docstring
    for why this replaced a per-slice "2 components" test). Given
    those indices, the actual Z boundary is refined with a per-slice
    AREA-FRACTION test computed on that SAME verified component (never
    on the final dilated lung_mask, and never compared against
    anything that could be an abdominal organ, since those are already
    excluded by 3D component selection): a slice only counts as
    "lung-containing" if the verified component covers at least
    `z_crop_min_area_frac` of that slice's area. This has no bilateral
    /symmetry requirement, so a one-sided expiration lung-base slice
    still passes, while a thin trachea-only neck slice (small area,
    even though it's part of the same verified 3D component) does not.

    If the area-fraction test rejects every slice (e.g. very limited
    lung capture), this falls back to "any voxel of the verified
    component" rather than ever discarding the whole volume.

    Returns (cropped_volume_hu, cropped_air_mask, cropped_lung_mask,
    cropped_heart_mask, updated_meta). If core_z_indices is empty,
    returns the inputs unchanged and adds a warning to meta instead of
    raising.
    """
    if core_z_indices is None or len(core_z_indices) == 0:
        meta["z_crop_applied"] = False
        meta["z_crop_warning"] = (
            "No 3D-verified lung component was found; skipped Z-cropping "
            "to avoid discarding the whole volume."
        )
        return volume_hu, air_mask, lung_mask, heart_mask, meta

    slice_spacing_mm = meta.get("slice_spacing_mm", 1.0) or 1.0
    margin_slices = int(round(margin_mm / slice_spacing_mm))

    # Per-slice area fraction of lung_mask restricted to the Z range
    # where the verified 3D component exists at all -- this keeps the
    # area test anchored to the already-identified lung structure
    # instead of ever re-deriving organ identity from scratch per
    # slice.
    z_lo_candidates = int(core_z_indices.min())
    z_hi_candidates = int(core_z_indices.max())
    candidate_slice_area_frac = lung_mask.mean(axis=(1, 2))

    zs_in_range = np.arange(z_lo_candidates, z_hi_candidates + 1)
    passing = zs_in_range[
        candidate_slice_area_frac[z_lo_candidates:z_hi_candidates + 1] >= z_crop_min_area_frac
    ]

    if len(passing) == 0:
        # Fall back to the full verified-component Z range rather than
        # discarding everything.
        passing = core_z_indices
        z_selection_method = "core_component_any_voxel_fallback"
    else:
        z_selection_method = "core_component_area_fraction"

    meta["z_crop_selection_method"] = z_selection_method
    z_lo = max(0, int(passing.min()) - margin_slices)
    z_hi = min(volume_hu.shape[0] - 1, int(passing.max()) + margin_slices)

    original_num_slices = volume_hu.shape[0]
    cropped_volume_hu = volume_hu[z_lo:z_hi + 1]
    cropped_air_mask = air_mask[z_lo:z_hi + 1]
    cropped_lung_mask = lung_mask[z_lo:z_hi + 1]
    cropped_heart_mask = heart_mask[z_lo:z_hi + 1]

    # Shift the z-origin so downstream code that maps voxel index -> mm
    # position still lands in the right place after cropping.
    origin = meta.get("origin_mm", [0.0, 0.0, 0.0])
    z_sign = meta.get("_z_step_sign", -1.0)
    origin_shifted = list(origin)
    origin_shifted[2] = origin[2] + z_sign * z_lo * slice_spacing_mm

    meta["z_crop_applied"] = True
    meta["original_num_slices"] = original_num_slices
    meta["z_crop_range"] = [z_lo, z_hi]
    meta["z_crop_margin_mm"] = margin_mm
    meta["z_crop_min_area_frac"] = z_crop_min_area_frac
    meta["origin_mm"] = origin_shifted
    meta["num_slices"] = cropped_volume_hu.shape[0]

    print(
        f"[info] Cropped Z from {original_num_slices} slices to "
        f"{cropped_volume_hu.shape[0]} slices (kept [{z_lo}, {z_hi}], "
        f"+/-{margin_mm:.0f}mm margin, selection method: "
        f"{z_selection_method})."
    )

    return cropped_volume_hu, cropped_air_mask, cropped_lung_mask, cropped_heart_mask, meta


def process_masks(
    processed_dir: str,
    out_dir: str,
    air_threshold: float = -950.0,
    lung_threshold: float = -320.0,
    z_crop_margin_mm: float = 10.0,
    z_crop_min_area_frac: float = 0.008,
    core_relative_size_threshold: float = 0.12,
    max_hole_area_mm2: float = 2500.0,
    apply_z_crop: bool = True,
    blank_value: float = AIR_REPLACEMENT_HU,
):
    print(f"[info] Loading HU volume from '{processed_dir}'...")
    volume_hu, meta = load_hu_volume(processed_dir)
    pixel_spacing_mm = meta.get("pixel_spacing_mm", [1.0, 1.0])
    slice_spacing_mm = meta.get("slice_spacing_mm", 1.0) or 1.0

    print(f"[info] Computing physical air mask (threshold <= {air_threshold} HU)...")
    air_mask = compute_air_mask(volume_hu, air_threshold=air_threshold)

    print(f"[info] Segmenting lung region (threshold < {lung_threshold} HU + "
          f"3D component identity + airway bridging + size-limited hole "
          f"filling + dedicated heart mask + wall margin)...")
    lung_mask, core_z_indices, heart_mask = compute_lung_mask(
        volume_hu,
        lung_threshold=lung_threshold,
        core_relative_size_threshold=core_relative_size_threshold,
        max_hole_area_mm2=max_hole_area_mm2,
        pixel_spacing_mm=pixel_spacing_mm,
        slice_spacing_mm=slice_spacing_mm,
    )

    meta["air_threshold_hu"] = air_threshold
    meta["air_voxel_fraction"] = float(air_mask.mean())
    meta["lung_threshold_hu"] = lung_threshold
    meta["lung_voxel_fraction"] = float(lung_mask.mean())
    meta["core_relative_size_threshold"] = core_relative_size_threshold
    meta["max_hole_area_mm2"] = max_hole_area_mm2
    meta["heart_voxel_fraction"] = float(heart_mask.mean())
    meta["heart_mask_found"] = bool(heart_mask.any())

    if lung_mask.any():
        zs, ys, xs = np.where(lung_mask)
        meta["lung_bounding_box_zyx"] = {
            "z": [int(zs.min()), int(zs.max())],
            "y": [int(ys.min()), int(ys.max())],
            "x": [int(xs.min()), int(xs.max())],
        }
    else:
        meta["lung_bounding_box_zyx"] = None
        print(
            "[warn] Lung mask is empty -- lung_threshold may need tuning for "
            "this scan, or the series may not be a chest CT."
        )

    if apply_z_crop:
        volume_hu, air_mask, lung_mask, heart_mask, meta = crop_to_lung_slices(
            volume_hu, air_mask, lung_mask, heart_mask, meta, core_z_indices,
            margin_mm=z_crop_margin_mm,
            z_crop_min_area_frac=z_crop_min_area_frac,
        )
    else:
        meta["z_crop_applied"] = False

    print("[info] Blanking non-lung voxels (chest wall, abdominal organs, "
          "background)...")
    volume_hu_masked = blank_non_lung(volume_hu, lung_mask, blank_value=blank_value)
    meta["non_lung_blank_value_hu"] = blank_value

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "volume_hu.npy"), volume_hu)
    np.save(os.path.join(out_dir, "volume_hu_masked.npy"), volume_hu_masked)
    np.save(os.path.join(out_dir, "air_mask.npy"), air_mask)
    np.save(os.path.join(out_dir, "lung_mask.npy"), lung_mask)
    np.save(os.path.join(out_dir, "heart_mask.npy"), heart_mask)

    meta_to_write = {k: v for k, v in meta.items() if not k.startswith("_")}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta_to_write, f, indent=2)

    print(f"[done] Volume shape: {volume_hu.shape} (Z, Y, X)")
    print(f"[done] Physical air voxels: {meta['air_voxel_fraction'] * 100:.1f}% of volume")
    print(f"[done] Lung region voxels: {meta['lung_voxel_fraction'] * 100:.1f}% of volume")
    if meta["heart_mask_found"]:
        print(f"[done] Heart mask voxels: {meta['heart_voxel_fraction'] * 100:.1f}% of volume")
    else:
        print("[info] No dedicated heart mask found (scan may not include a "
              "cardiac-level slice, or --max-hole-area-mm2 needs tuning).")
    if meta.get("z_crop_applied"):
        print(f"[done] Z-cropped to lung-containing slices: "
              f"{meta['original_num_slices']} -> {meta['num_slices']} slices")
    print(f"[done] Wrote volume_hu.npy, volume_hu_masked.npy, air_mask.npy, "
          f"lung_mask.npy, heart_mask.npy, meta.json -> '{out_dir}'")

    return volume_hu, volume_hu_masked, air_mask, lung_mask, heart_mask, meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="STEP 2/3: Segment lungs, blank non-lung anatomy, and "
        "Z-crop to lung-containing slices, using output from "
        "01_dicom_to_hu.py."
    )
    parser.add_argument(
        "processed_dir",
        help="Directory containing volume_hu.npy / meta.json (the "
        "--out-dir from 01_dicom_to_hu.py).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write masked/cropped outputs (default: "
        "'<processed_dir>_masked').",
    )
    parser.add_argument(
        "--air-threshold",
        type=float,
        default=-950.0,
        help="HU cutoff at/below which a voxel is classified as PHYSICAL "
        "air (default: -950.0).",
    )
    parser.add_argument(
        "--lung-threshold",
        type=float,
        default=-320.0,
        help="HU cutoff used to segment the lung region's aerated core "
        "(default: -320.0).",
    )
    parser.add_argument(
        "--core-relative-size-threshold",
        type=float,
        default=0.12,
        help="3D organ-identity test: a connected component (after "
        "airway bridging) is kept as 'lung tissue' only if its voxel "
        "count is at least this fraction of the single largest "
        "component's voxel count (default: 0.12). This is what "
        "distinguishes the bridged lung+airway tree from a same-sized-"
        "looking-per-slice but much-smaller-in-3D abdominal gas pocket.",
    )
    parser.add_argument(
        "--max-hole-area-mm2",
        type=float,
        default=2500.0,
        help="Largest per-slice enclosed-hole area (mm^2) that will be "
        "filled inside the lung silhouette (default: 2500.0, i.e. 25 "
        "cm^2). Keeps normal filling of vessels/airway walls/nodules "
        "while preventing a heart-sized enclosed region -- formed when "
        "the two lungs' silhouette rings around the mediastinum at a "
        "cardiac-level slice -- from being filled in as 'lung'. The "
        "excluded region is also used to build a dedicated heart_mask.npy.",
    )
    parser.add_argument(
        "--z-crop-margin-mm",
        type=float,
        default=10.0,
        help="Physical margin (mm) kept above/below the identified lung "
        "extent when cropping out non-lung slices (default: 10.0).",
    )
    parser.add_argument(
        "--z-crop-min-area-frac",
        type=float,
        default=0.008,
        help="Per-slice area-fraction threshold (of the already-3D-"
        "verified lung component) used to trim the thin trachea-in-the-"
        "neck extent from the Z-crop range (default: 0.008, i.e. 0.8%%). "
        "No bilateral/symmetry requirement, so single-sided expiration "
        "lung-base slices still pass.",
    )
    parser.add_argument(
        "--blank-value",
        type=float,
        default=AIR_REPLACEMENT_HU,
        help="HU value written into voxels outside lung_mask in "
        "volume_hu_masked.npy (default: -1000.0, physical air).",
    )
    parser.add_argument(
        "--no-z-crop",
        action="store_true",
        help="Disable Z-cropping; keep every slice (neck/abdomen "
        "included) in the saved volume/masks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.processed_dir.rstrip("/\\") + "_masked")
    process_masks(
        args.processed_dir, out_dir,
        air_threshold=args.air_threshold,
        lung_threshold=args.lung_threshold,
        z_crop_margin_mm=args.z_crop_margin_mm,
        z_crop_min_area_frac=args.z_crop_min_area_frac,
        core_relative_size_threshold=args.core_relative_size_threshold,
        max_hole_area_mm2=args.max_hole_area_mm2,
        apply_z_crop=not args.no_z_crop,
        blank_value=args.blank_value,
    )


if __name__ == "__main__":
    main()