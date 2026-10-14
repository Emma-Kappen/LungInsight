r"""
LungInsight v2 -- standalone CT preprocessing pipeline.

This file consolidates the supplied 01_dicom_to_hu.py and 02_mask_and_crop.py.
It intentionally does not import or dynamically load either legacy script.
The original DICOM discovery/series selection/slice ordering/HU conversion/
windowing functions and the original lung/heart segmentation, blanking, and
Z-cropping functions are retained in this file.

Input:
    python backend\\Imaging\\01_hu_and_crop.py LIDC-IDRI-0001

LIDC input layout:
    <project_root>\\datasets\\LIDC\\lidc_idri\\<patient_id>\\<study>\\<series>\\...

For example:
    datasets\\LIDC\\lidc_idri\\LIDC-IDRI-0001\\30178\\03192\\<DICOM files>

IMPORTANT: 30178 and 03192 are study/series directories, NOT the patient ID.
The script resolves the patient directory first and then recursively searches
beneath it for DICOM files.

Output:
    <project_root>\\LungInsight_v2\\backend\\output\\<patient_id>\\

Saved arrays:
    volume_hu.npy
        float32 (Z,Y,X), complete HU volume before Z-cropping.

    volume_hu_cropped.npy
        float32 (Z,Y,X), final result volume after the original Z-crop.

    volume_hu_masked.npy
        float32 (Z,Y,X), final cropped volume with voxels outside lung_mask
        blanked to the physical-air replacement HU (-1000 by default).

    volume_windowed.npy
        uint8 (Z,Y,X,3), the original lung/soft-tissue/bone window channels,
        cropped along Z so they remain aligned with the final result volume.

    air_mask.npy / lung_mask.npy / heart_mask.npy
        bool arrays aligned to volume_hu_cropped.npy.

    meta.json
        Per-patient metadata and the final voxel-coordinate convention.

Coordinate convention:
    Array order is [z,y,x].
    Coordinate system is DICOM_LPS and units are millimetres.
    origin_mm is the physical DICOM-LPS coordinate of the CENTER of saved
    voxel [0,0,0] in volume_hu_cropped.npy.

    p_LPS_mm = origin_mm + direction_lps @ ([z,y,x] * spacing_zyx_mm)

Dependencies:
    numpy
    scipy
    pydicom
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, List, Tuple, cast

IMAGING_DIR = Path(__file__).resolve().parent
if IMAGING_DIR.name in {"stages", "inference", "visualization", "validation"}:
    IMAGING_DIR = IMAGING_DIR.parent
if str(IMAGING_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGING_DIR))

import numpy as np

from pipeline_schema import get_required, load_schema, resolve_template
from heart_mask_review import write_heart_mask_review_montages

SCHEMA = load_schema(__file__)
STAGE01_SCHEMA = get_required(SCHEMA, "stage_01_hu_and_crop", expected_type=dict)
STAGE01_PREPROCESS = get_required(STAGE01_SCHEMA, "preprocessing", expected_type=dict)
OUTPUT_SCHEMA = get_required(SCHEMA, "outputs", expected_type=dict)
STAGE01_FILES = get_required(STAGE01_SCHEMA, "filenames", expected_type=dict)
STAGE01_PATHS = get_required(SCHEMA, "paths.stage_01", expected_type=dict)
get_required(SCHEMA, str(STAGE01_SCHEMA["coordinate_format_ref"]), expected_type=dict)

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    print(
        "pydicom is required. Install it with:\n"
        "    pip install pydicom",
        file=sys.stderr,
    )
    raise

try:
    from scipy import ndimage
except ImportError:
    print(
        "scipy is required for lung segmentation. Install it with:\n"
        "    pip install scipy",
        file=sys.stderr,
    )
    raise

try:
    from heart_rod_tracking import compute_heart_mask_rod, associate_large_hole_components, build_heart_excluded_volume
except ImportError as exc:
    raise ImportError(
        "Stage 01 requires backend/Imaging/heart_rod_tracking.py for the expanding 3D central-rod heart tracker. "
        "Restore the replacement bundle as a unit; the helper is part of this stage's implementation."
    ) from exc

# Retained exactly from the supplied 01_dicom_to_hu.py.
OUT_OF_SCAN_PIXEL_HU_FLOOR = int(get_required(STAGE01_PREPROCESS, "out_of_scan_pixel_hu_floor"))
AIR_REPLACEMENT_HU = float(get_required(STAGE01_PREPROCESS, "air_replacement_hu"))
WINDOWS_HU = get_required(STAGE01_PREPROCESS, "windows_hu", expected_type=dict)
LUNG_WINDOW_HU = tuple(WINDOWS_HU["lung"])
SOFT_TISSUE_WINDOW_HU = tuple(WINDOWS_HU["soft_tissue"])
BONE_WINDOW_HU = tuple(WINDOWS_HU["bone"])
HEART_TRACKING_SCHEMA = get_required(STAGE01_PREPROCESS, "heart_tracking", expected_type=dict)

SCRIPT_DIR = IMAGING_DIR
V2_ROOT = SCRIPT_DIR.parent.parent
PROJECT_ROOT = V2_ROOT.parent
DEFAULT_LIDC_DATA_ROOT = PROJECT_ROOT / get_required(STAGE01_SCHEMA, "input_root_from_project_root")
# Backward-compatible archive parent for layouts that place patient folders
# directly under datasets/LIDC. Derive it from the schema-defined data root.
DEFAULT_LIDC_ROOT = DEFAULT_LIDC_DATA_ROOT.parent
DEFAULT_OUTPUT_ROOT = V2_ROOT / get_required(SCHEMA, "paths.output_root")

@dataclass
class SeriesSlices:
    series_uid: str
    datasets: List[Any] = field(default_factory=list)


def find_dicom_files(patient_dir: str) -> List[str]:
    """Recursively collect every candidate DICOM file under patient_dir.

    Resolves ~, tries the provided path, a script-relative path, and
    the absolute path, in that order.
    """
    # Expand ~ to the user home directory first
    expanded = os.path.expanduser(patient_dir)

    # Candidate paths to try (as absolute paths for clear error messages)
    script_dir = str(SCRIPT_DIR)
    script_relative = os.path.join(script_dir, patient_dir)
    abspath_candidate = os.path.abspath(patient_dir)

    attempted = [
        os.path.abspath(expanded),
        os.path.abspath(script_relative),
        os.path.abspath(abspath_candidate),
    ]

    # Pick the first that exists
    if os.path.exists(expanded):
        resolved_dir = expanded
    elif os.path.exists(script_relative):
        resolved_dir = script_relative
    elif os.path.exists(abspath_candidate):
        resolved_dir = abspath_candidate
    else:
        raise FileNotFoundError(
            "Patient directory not found; attempted the following paths:\n"
            f"  {attempted[0]}\n  {attempted[1]}\n  {attempted[2]}"
        )

    dicom_paths = []
    for root, _dirs, files in os.walk(resolved_dir):
        for fname in files:
            # LIDC-IDRI slice files usually have no extension or .dcm;
            # don't filter by extension, just try to read everything.
            if fname.startswith("."):
                continue
            dicom_paths.append(os.path.join(root, fname))
    if not dicom_paths:
        raise FileNotFoundError(f"No files found under '{resolved_dir}'.")
    return dicom_paths


def load_ct_series(patient_dir: str) -> SeriesSlices:
    """
    Read every DICOM file under patient_dir, keep only CT image slices
    that carry pixel data, group them by SeriesInstanceUID, and return
    the series with the most slices (the actual volumetric CT scan,
    as opposed to scout/localizer images or RTSTRUCT/SEG annotation
    objects also present in LIDC-IDRI).
    """
    candidate_paths = find_dicom_files(patient_dir)

    series_map = {}
    skipped = 0
    for path in candidate_paths:
        try:
            ds = pydicom.dcmread(path, force=False)
        except (InvalidDicomError, Exception):
            skipped += 1
            continue

        # Skip non-CT objects (RTSTRUCT, SEG, SR, scout/localizer, etc.)
        # and anything without actual pixel data.
        if getattr(ds, "Modality", None) != "CT":
            skipped += 1
            continue
        if "PixelData" not in ds:
            skipped += 1
            continue

        series_uid = getattr(ds, "SeriesInstanceUID", "UNKNOWN_SERIES")
        series_map.setdefault(series_uid, SeriesSlices(series_uid)).datasets.append(ds)

    if not series_map:
        raise ValueError(
            f"No readable CT image slices found under '{patient_dir}' "
            f"({skipped} files skipped as non-CT/no-pixel-data)."
        )

    # Pick the series with the most slices -> the real volumetric scan.
    best_series = max(series_map.values(), key=lambda s: len(s.datasets))

    if len(series_map) > 1:
        print(
            f"[info] Found {len(series_map)} series under '{patient_dir}'; "
            f"selected series {best_series.series_uid} with "
            f"{len(best_series.datasets)} slices (largest)."
        )

    return best_series


def order_slices(datasets: List[Any]):
    """
    Sort slices into correct spatial (axial) order using the z
    component of ImagePositionPatient. Falls back to InstanceNumber,
    and finally to SliceLocation, if position data is missing.
    Returns (ordered_datasets, slice_spacing_mm).
    """

    def sort_key(ds):
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        if hasattr(ds, "SliceLocation"):
            return float(ds.SliceLocation)
        if hasattr(ds, "InstanceNumber"):
            return float(ds.InstanceNumber)
        return 0.0

    ordered = sorted(datasets, key=sort_key)

    # Estimate slice spacing from consecutive z positions (more
    # reliable than trusting a single SliceThickness tag, which can
    # disagree with the actual reconstruction increment).
    zs = []
    for ds in ordered:
        if hasattr(ds, "ImagePositionPatient"):
            zs.append(float(ds.ImagePositionPatient[2]))
    if len(zs) >= 2:
        diffs = np.diff(zs)
        slice_spacing = float(np.median(np.abs(diffs)))
    else:
        slice_spacing = float(getattr(ordered[0], "SliceThickness", 1.0))

    return ordered, slice_spacing


def slices_to_hu_volume(ordered_datasets):
    """
    Stack ordered slices into a 3D array (Z, Y, X) and convert to HU.

    HU = pixel_value * RescaleSlope + RescaleIntercept

    Applied exactly once, per-slice, using each slice's own
    RescaleSlope/RescaleIntercept (defaults to slope=1, intercept=0
    if absent, matching the DICOM standard's fallback behavior).
    """
    first = ordered_datasets[0]
    rows, cols = int(first.Rows), int(first.Columns)
    n_slices = len(ordered_datasets)

    volume_hu = np.zeros((n_slices, rows, cols), dtype=np.float32)

    for i, ds in enumerate(ordered_datasets):
        pixel_array = ds.pixel_array.astype(np.float32)

        if pixel_array.shape != (rows, cols):
            raise ValueError(
                f"Slice {i} has shape {pixel_array.shape}, expected "
                f"({rows}, {cols}). Mixed-geometry series should not "
                f"be combined into one volume."
            )

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))

        hu_slice = pixel_array * slope + intercept

        # Remap scanner "outside field of view" padding to air so it
        # doesn't masquerade as bone/metal or skew HU statistics.
        hu_slice[hu_slice < OUT_OF_SCAN_PIXEL_HU_FLOOR] = AIR_REPLACEMENT_HU

        volume_hu[i] = hu_slice

    return volume_hu


def apply_ct_window(volume_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Linearly stretch a single CT window to uint8 [0, 255].

    Clips to [center - width/2, center + width/2] first, THEN stretches
    -- clipping before stretching is what actually creates the
    contrast gain: everything outside the window is thrown away, so
    the full 0-255 range is spent entirely on the HU band you care
    about, instead of being diluted across the whole -1000..+2000+ HU
    range of the raw scan.
    """
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(volume_hu, low, high)
    stretched = (clipped - low) / (high - low)  # -> [0, 1]
    return (stretched * 255.0).astype(np.uint8)


def build_multi_window_volume(
    volume_hu: np.ndarray,
    lung_window=LUNG_WINDOW_HU,
    soft_tissue_window=SOFT_TISSUE_WINDOW_HU,
    bone_window=BONE_WINDOW_HU,
) -> np.ndarray:
    """
    Build a 3-channel contrast-enhanced volume, shape (Z, Y, X, 3),
    dtype uint8, channels = [lung window, soft-tissue window, bone
    window].

    Why three separate windows instead of one: air (~-1000 HU), soft
    tissue (~-100 to +80 HU), and bone (~+300 to +1900 HU) occupy
    wildly different, mostly non-overlapping HU bands. Any single
    linear window wide enough to show bone detail compresses the
    entire soft-tissue range into just a few gray levels (and vice
    versa for a window tuned to soft tissue). Windowing each tissue
    class separately -- each stretched to use the FULL 0-255 range on
    just its own band -- maximizes contrast within each tissue type
    simultaneously. Stacked as channels, this is also a useful 3-band
    input for a CNN.
    """
    lung_channel = apply_ct_window(volume_hu, *lung_window)
    soft_tissue_channel = apply_ct_window(volume_hu, *soft_tissue_window)
    bone_channel = apply_ct_window(volume_hu, *bone_window)
    return np.stack([lung_channel, soft_tissue_channel, bone_channel], axis=-1)


def extract_metadata(ordered_datasets, slice_spacing_mm: float) -> dict:
    first = ordered_datasets[0]
    pixel_spacing = [float(x) for x in getattr(first, "PixelSpacing", [1.0, 1.0])]
    origin = [float(x) for x in getattr(first, "ImagePositionPatient", [0.0, 0.0, 0.0])]

    orientation = [float(x) for x in getattr(
        first, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )]
    row_direction = np.asarray(orientation[:3], dtype=np.float64)
    column_direction = np.asarray(orientation[3:], dtype=np.float64)
    slice_direction = np.cross(row_direction, column_direction)
    direction_lps = np.column_stack((slice_direction, row_direction, column_direction))

    # Direction z moves in as slice index increases, so a later Z-crop
    # (in step 2) can shift origin_mm by the right sign (superior->
    # inferior scans usually DECREASE z with index, but this isn't
    # universal).
    z_step_sign = -1.0
    if len(ordered_datasets) >= 2:
        z0 = getattr(ordered_datasets[0], "ImagePositionPatient", None)
        z1 = getattr(ordered_datasets[1], "ImagePositionPatient", None)
        if z0 is not None and z1 is not None:
            diff = float(z1[2]) - float(z0[2])
            if diff != 0:
                z_step_sign = 1.0 if diff > 0 else -1.0

    return {
        "_z_step_sign": z_step_sign,
        "patient_id": str(getattr(first, "PatientID", "UNKNOWN")),
        "series_instance_uid": str(getattr(first, "SeriesInstanceUID", "UNKNOWN")),
        "num_slices": len(ordered_datasets),
        "rows": int(first.Rows),
        "columns": int(first.Columns),
        # (x, y) mm spacing within a slice, then z spacing between slices
        "pixel_spacing_mm": pixel_spacing,
        "spacing_zyx_mm": [slice_spacing_mm, pixel_spacing[0], pixel_spacing[1]],
        "slice_spacing_mm": slice_spacing_mm,
        "origin_mm": origin,
        "direction_lps": direction_lps.tolist(),
        "array_axes": ["z", "y", "x"],
        "coordinate_system": "DICOM_LPS",
        "manufacturer": str(getattr(first, "Manufacturer", "UNKNOWN")),
        "kvp": getattr(first, "KVP", None),
        "convolution_kernel": str(getattr(first, "ConvolutionKernel", "UNKNOWN")),
    }


def compute_air_mask(volume_hu: np.ndarray, air_threshold: float = float(STAGE01_PREPROCESS["air_threshold_hu"])) -> np.ndarray:
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
    label_result = cast(tuple[np.ndarray, int], ndimage.label(background, structure=structure_2d))
    labeled_bg = label_result[0]  # type: ignore[index]
    n_bg = int(label_result[1])  # type: ignore[index]
    if n_bg == 0:
        return mask_2d, np.zeros_like(mask_2d)

    border_labels = set(labeled_bg[0, :].tolist())
    border_labels |= set(labeled_bg[-1, :].tolist())
    border_labels |= set(labeled_bg[:, 0].tolist())
    border_labels |= set(labeled_bg[:, -1].tolist())
    border_labels.discard(0)

    n_bg_int = int(n_bg)
    sizes = ndimage.sum(background, labeled_bg, index=np.arange(1, n_bg_int + 1))
    fill_labels: List[int] = []
    large_labels: List[int] = []
    for lbl in range(1, n_bg_int + 1):
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
    min_span_mm: float = float(STAGE01_PREPROCESS["heart_mask_min_span_mm"]),
    min_component_fraction: float = float(STAGE01_PREPROCESS["heart_mask_min_component_fraction"]),
):
    """
    LEGACY compatibility helper. The production Stage 01 path no longer calls
    this hole-only heart detector; it calls compute_heart_mask_rod() using the
    canonical expanding 3D central-rod contract.

    This legacy helper is retained only for external imports from older scripts.
    It infers the mask from the "too big to fill" holes gathered by
    _fill_small_holes_2d. Production heart tracking no longer depends on it.

    At cardiac-level slices, the lung silhouette
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
    label_result = cast(tuple[np.ndarray, int], ndimage.label(large_holes_stack, structure=structure_3d))
    labeled = label_result[0]  # type: ignore[index]
    num_features = int(label_result[1])  # type: ignore[index]
    if num_features == 0:
        return empty

    num_features_int = int(num_features)
    sizes = ndimage.sum(large_holes_stack, labeled, index=np.arange(1, num_features_int + 1))
    total_size = sizes.sum()
    largest_label = int(np.argmax(sizes)) + 1
    largest_size = sizes[largest_label - 1]

    if largest_size < min_component_fraction * total_size:
        return empty

    candidate = labeled == largest_label
    z_indices = np.asarray(np.where(candidate.any(axis=(1, 2)))[0], dtype=np.intp)
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
    lung_threshold: float = float(STAGE01_PREPROCESS["lung_threshold_hu"]),
    min_component_fraction: float = float(STAGE01_PREPROCESS["min_component_fraction"]),
    core_relative_size_threshold: float = float(STAGE01_PREPROCESS["core_relative_size_threshold"]),
    max_hole_area_mm2: float = float(STAGE01_PREPROCESS["max_hole_area_mm2"]),
    pixel_spacing_mm=(1.0, 1.0),
    slice_spacing_mm: float = 1.0,
    closing_radius_mm: float = float(STAGE01_PREPROCESS["closing_radius_mm"]),
    wall_margin_mm: float = float(STAGE01_PREPROCESS["wall_margin_mm"]),
    trachea_max_area_mm2: float = 700.0,
    return_diagnostics: bool = False,
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
         regions alone (see _fill_small_holes_2d). Large holes are collected for
         mass recovery, but heart identity is NOT inferred from those 2D holes.
         The expanding central-rod 3D tracker finds central soft tissue directly
         from the unmasked HU volume; 3D overlap is then used to associate any
         matching large-hole components with the tracked structure.
      6. Any "too big to fill" region that ISN'T associated with the tracked heart
         is, by elimination, dense intra-lung TISSUE rather than
         mediastinal cavity -- almost always a solid mass/
         consolidation large enough to fail the air threshold in
         cross-section. That is real anatomy 04_detect_and_patch.py
         needs to see, so it is added back into lung_mask here (never
         into heart_mask, which stays excluded). Without this step, a
         large tumor is indistinguishable from mediastinum to this
         function and gets silently blanked to air HU downstream --
         The 3D rod method has diagnostic evidence (selected radius, axial
         span, rod contact and component size), but it remains a heuristic and
         can confuse contiguous central lung lesions with adjacent mediastinal
         tissue. An empty/non-qualifying result must remain a downstream block.
      7. Dilate outward by `wall_margin_mm` (mm-aware, anisotropic) to
         pull in a rim of the lung wall itself -- the pleura and the
         walls of the trachea/bronchi, which are denser than
         `lung_threshold` and would otherwise be excluded even though
         they're literally the boundary of the organ. A final
         size-limited hole-fill cleans up anything the dilation
         reopened, and the heart mask is subtracted again afterward.

    Returns (lung_mask, core_z_indices, heart_mask) by default. With
    ``return_diagnostics=True``, a fourth value contains the expanding-rod
    tracking report. The rod result is central cardiac/mediastinal tissue for
    exclusion; it is not a pathology-grade cardiac segmentation.

    Note: this is a solid general-purpose segmentation but not a
    clinical-grade one -- `wall_margin_mm` is a fixed approximation of
    the pleura, not a true tissue-boundary detector, and distinguishing
    a large intrapulmonary mass from the heart (step 6 above) is a
    size/connectivity heuristic with no independent anatomical prior --
    see compute_heart_mask's docstring for when that heuristic can
    still misclassify a large mass as heart.
    """
    binary = volume_hu < lung_threshold

    structure_2d = np.ones((3, 3), dtype=bool)
    interior = np.zeros_like(binary)
    for z in range(binary.shape[0]):
        slice_binary = binary[z]
        if not slice_binary.any():
            continue
        label_result = cast(tuple[np.ndarray, int], ndimage.label(slice_binary, structure=structure_2d))
        labeled_slice = label_result[0]  # type: ignore[index]
        border_labels = set(labeled_slice[0, :].tolist())
        border_labels |= set(labeled_slice[-1, :].tolist())
        border_labels |= set(labeled_slice[:, 0].tolist())
        border_labels |= set(labeled_slice[:, -1].tolist())
        border_labels.discard(0)
        interior[z] = slice_binary & ~np.isin(labeled_slice, list(border_labels))

    empty_mask = np.zeros_like(volume_hu, dtype=bool)
    empty_z = np.array([], dtype=int)
    if not interior.any():
        no_lung_diag = {
            "method": str(HEART_TRACKING_SCHEMA["method"]),
            "status": "not_run_no_lung_core",
            "failure_reason": "no_lung_core_for_thoracic_slice_selection",
            "heart_mask_voxels": 0,
        }
        if return_diagnostics:
            return empty_mask, empty_z, empty_mask.copy(), no_lung_diag
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
    label_result = cast(tuple[np.ndarray, int], ndimage.label(interior, structure=structure_3d))
    labeled = label_result[0]  # type: ignore[index]
    num_features = int(label_result[1])  # type: ignore[index]
    if num_features == 0:
        no_lung_diag = {
            "method": str(HEART_TRACKING_SCHEMA["method"]),
            "status": "not_run_no_lung_core",
            "failure_reason": "no_3d_lung_components",
            "heart_mask_voxels": 0,
        }
        if return_diagnostics:
            return empty_mask, empty_z, empty_mask.copy(), no_lung_diag
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
    keep_labels = [int(v) + 1 for v in np.asarray(np.nonzero(keep_mask)[0], dtype=np.intp)]  # ndimage labels are 1-indexed

    core_mask_preclosing = np.isin(labeled, keep_labels)

    # Drop small, midline, per-slice components above the carina (trachea).
    midline_half_width = 0.12 * volume_hu.shape[2]
    center_x = (volume_hu.shape[2] - 1) / 2.0
    for z in range(core_mask_preclosing.shape[0]):
        if not core_mask_preclosing[z].any():
            continue
        slice_result = cast(tuple[np.ndarray, int], ndimage.label(core_mask_preclosing[z], structure=structure_2d))
        slice_labels, slice_count = slice_result[0], int(slice_result[1])  # type: ignore[index]
        for lbl in range(1, slice_count + 1):
            comp = slice_labels == lbl
            if comp.sum() * py * px > trachea_max_area_mm2:
                continue
            if abs(float(np.nonzero(comp)[1].mean()) - center_x) <= midline_half_width:
                core_mask_preclosing[z][comp] = False
    core_z_indices = np.asarray(np.where(core_mask_preclosing.any(axis=(1, 2)))[0], dtype=np.intp)

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

    # Track central cardiac/mediastinal tissue with an expanding 3D rod
    # through the YX center of the CT, then follow the seed component's
    # shape across Z. This deliberately does not write out a filled cylinder.
    rod_cfg = HEART_TRACKING_SCHEMA
    heart_mask, heart_tracking_diag = compute_heart_mask_rod(
        volume_hu,
        lung_mask,
        pixel_spacing_mm=pixel_spacing_mm,
        slice_spacing_mm=slice_spacing_mm,
        min_span_mm=float(STAGE01_PREPROCESS["heart_mask_min_span_mm"]),
        seed_radius_mm=float(rod_cfg["seed_radius_mm"]),
        radius_step_mm=float(rod_cfg["radius_step_mm"]),
        max_radius_mm=float(rod_cfg["max_radius_mm"]),
        max_contour_growth_mm=float(rod_cfg.get("max_contour_growth_mm", 8.0)),
        max_slice_area_ratio=float(rod_cfg.get("max_slice_area_ratio", 1.8)),
        soft_tissue_hu_min=float(rod_cfg["soft_tissue_hu_min"]),
        soft_tissue_hu_max=float(rod_cfg["soft_tissue_hu_max"]),
        exclusion_mask_hu_min=float(rod_cfg.get("exclusion_mask_hu_min", STAGE01_PREPROCESS["air_threshold_hu"])),
        min_lung_slice_fraction=float(rod_cfg["min_lung_slice_fraction"]),
        min_support_voxels_per_slice=int(rod_cfg.get("min_support_voxels_per_slice", 8)),
        axial_margin_mm=float(rod_cfg["axial_margin_mm"]),
        smoothing_closing_mm=float(rod_cfg.get("smoothing_closing_mm", 6.0)),
        max_heart_span_mm=float(rod_cfg.get("max_heart_span_mm", 110.0)),
    )
    # Associate an entire large-hole component with the tracked structure only
    # if it intersects the central 3D component. This avoids restoring the
    # heart cavity as a "large lung mass" while preserving unrelated masses.
    associated_heart_holes, hole_association_diag = associate_large_hole_components(
        large_holes_stack, heart_mask
    )
    heart_mask |= associated_heart_holes
    heart_tracking_diag.update(hole_association_diag)
    heart_tracking_diag["heart_mask_voxels_after_hole_association"] = int(heart_mask.sum())
    heart_tracking_diag["algorithm_config"] = {
        "method": str(rod_cfg["method"]),
        "seed_radius_mm": float(rod_cfg["seed_radius_mm"]),
        "radius_step_mm": float(rod_cfg["radius_step_mm"]),
        "max_radius_mm": float(rod_cfg["max_radius_mm"]),
        "max_contour_growth_mm": float(rod_cfg.get("max_contour_growth_mm", 8.0)),
        "max_slice_area_ratio": float(rod_cfg.get("max_slice_area_ratio", 1.8)),
        "soft_tissue_hu_range": [float(rod_cfg["soft_tissue_hu_min"]), float(rod_cfg["soft_tissue_hu_max"])],
        "exclusion_mask_hu_min": float(rod_cfg.get("exclusion_mask_hu_min", STAGE01_PREPROCESS["air_threshold_hu"])),
        "min_support_voxels_per_slice": int(rod_cfg.get("min_support_voxels_per_slice", 8)),
        "min_lung_slice_fraction": float(rod_cfg["min_lung_slice_fraction"]),
        "axial_margin_mm": float(rod_cfg["axial_margin_mm"]),
        "minimum_component_span_mm": float(STAGE01_PREPROCESS["heart_mask_min_span_mm"]),
    }
    if not heart_mask.any():
        print(
            "[warn] Expanding 3D central-rod tracker could not identify a qualifying "
            "cardiac/mediastinal component. heart_mask remains empty; downstream "
            "candidate gating must remain blocked."
        )
    else:
        print(
            "[info] Expanding 3D central-rod + shape tracker: "
            f"status={heart_tracking_diag.get('status')}, "
            f"rod_radius={heart_tracking_diag.get('selected_radius_mm')} mm, "
            f"tracked_Z={heart_tracking_diag.get('tracked_z_range')}, "
            f"slices={heart_tracking_diag.get('tracked_slice_count')}, "
            f"voxels={int(heart_mask.sum())}."
        )

    # Exclude the tracked central component before large-hole restoration and
    # again after wall-margin dilation.
    lung_mask &= ~heart_mask

    # Recover non-heart large holes into lung_mask.
    #
    # BUG FIXED HERE: `filled` (now `lung_mask`) never contained large
    # holes to begin with -- _fill_small_holes_2d excludes anything
    # over max_hole_area_mm2 by construction, before heart_mask is even
    # computed a few lines above. heart_mask was only ever used to
    # SUBTRACT from lung_mask (a no-op for voxels that were never in
    # lung_mask), never to add anything back -- so EVERY "too big to
    # fill" region was unconditionally discarded, whether it was the
    # heart, the esophagus/great vessels, or a real intrapulmonary
    # tumor/consolidation large enough to exceed the area cap in
    # cross-section. A large solid mass got treated exactly like
    # mediastinum: silently blanked to air HU downstream in
    # blank_non_lung(), so 04_detect_and_patch.py's candidate detector
    # -- which only ever looks inside lung_mask -- could never generate
    # a candidate there, however good its thresholds are.
    #
    # By elimination, a large-hole voxel that is NOT part of the
    # identified heart component is dense TISSUE, not mediastinal
    # cavity -- most often a mass/consolidation that failed the air
    # threshold for a pathological reason. That is real lung-interior
    # anatomy 04 needs to see, so it belongs in lung_mask.
    #
    # Caveat (see compute_heart_mask's own docstring): if a tumor is
    # itself large/central enough to win the "largest 3D large-hole
    # component" contest, or is contiguous with the true heart
    # component, it will be classified as heart_mask and still get
    # excluded here. This fix recovers the common case (a mass smaller
    # than, or spatially separate from, the heart) but is not a
    # substitute for visually QA-ing the mask (03_visualize.py) on any
    # scan with a suspected large intrapulmonary mass, and adjusting
    # max_hole_area_mm2 / compute_heart_mask's own thresholds if the
    # mass and heart are still being confused for one another.
    non_heart_large_mass = large_holes_stack & ~associated_heart_holes & ~heart_mask
    lung_mask |= non_heart_large_mass

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

    heart_tracking_diag["heart_mask_voxels_final_pre_crop"] = int(heart_mask.sum())
    heart_tracking_diag["lung_mask_voxels_final_pre_crop"] = int(lung_mask.sum())
    heart_tracking_diag["heart_mask_overlap_lung_voxels_after_exclusion"] = int(np.count_nonzero(heart_mask & lung_mask))
    if return_diagnostics:
        return lung_mask, core_z_indices, heart_mask, heart_tracking_diag
    return lung_mask, core_z_indices, heart_mask


def blank_non_lung(volume_hu: np.ndarray, lung_mask: np.ndarray,
                    blank_value: float = float(STAGE01_PREPROCESS["default_blank_value_hu"])) -> np.ndarray:
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
    margin_mm: float = float(STAGE01_PREPROCESS["z_crop_margin_mm"]),
    z_crop_min_area_frac: float = float(STAGE01_PREPROCESS["z_crop_min_area_fraction"]),
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
        meta["crop_offset_zyx"] = [0, 0, 0]
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
    origin = np.asarray(meta.get("origin_mm", [0.0, 0.0, 0.0]), dtype=float)
    direction = np.asarray(meta.get("direction_lps", np.eye(3).tolist()), dtype=float)
    if direction.shape != (3, 3):
        raise ValueError("meta.json contains an invalid direction_lps matrix")
    spacing_zyx = np.asarray(meta.get(
        "spacing_zyx_mm",
        [slice_spacing_mm, *meta.get("pixel_spacing_mm", [1.0, 1.0])],
    ), dtype=float)
    origin_shifted = origin + direction @ (np.asarray([z_lo, 0, 0], dtype=float) * spacing_zyx)

    meta["z_crop_applied"] = True
    meta["original_num_slices"] = original_num_slices
    meta["z_crop_range"] = [z_lo, z_hi]
    meta["z_crop_margin_mm"] = margin_mm
    meta["z_crop_min_area_frac"] = z_crop_min_area_frac
    meta["origin_mm"] = origin_shifted.tolist()
    meta["crop_offset_zyx"] = [z_lo, 0, 0]
    meta["spacing_zyx_mm"] = spacing_zyx.tolist()
    meta["num_slices"] = cropped_volume_hu.shape[0]

    print(
        f"[info] Cropped Z from {original_num_slices} slices to "
        f"{cropped_volume_hu.shape[0]} slices (kept [{z_lo}, {z_hi}], "
        f"+/-{margin_mm:.0f}mm margin, selection method: "
        f"{z_selection_method})."
    )

    return cropped_volume_hu, cropped_air_mask, cropped_lung_mask, cropped_heart_mask, meta



def _extract_lidc_patient_id(path: Path) -> str:
    """Return the LIDC patient folder from anywhere below a patient tree.

    LIDC-IDRI data are commonly arranged as:
        datasets/LIDC/lidc_idri/<PatientID>/<StudyUID>/<SeriesUID>/<DICOM files>

    The StudyUID/SeriesUID directories (for example ``30178/03192``) are
    NOT patient IDs.  The patient ID is the ancestor matching LIDC-IDRI-<n>.
    """
    for part in reversed(path.parts):
        if part.upper().startswith("LIDC-IDRI-"):
            return part
    raise ValueError(
        f"Could not determine an LIDC patient ID from path: {path}. "
        "Expected an ancestor directory named like 'LIDC-IDRI-0001'."
    )


def resolve_patient_dir(patient_arg: str):
    """Resolve a patient ID or explicit path using the actual LIDC layout.

    For a patient ID such as ``LIDC-IDRI-0001``, the preferred location is:
        datasets/LIDC/lidc_idri/LIDC-IDRI-0001

    An explicit path may point either to the patient directory itself or to
    any directory below it (for example a StudyUID/SeriesUID directory).
    In the latter case the patient ID is recovered from the ancestor path,
    while the returned scan root is still the patient directory so the DICOM
    search remains recursive across all study/series subfolders.
    """
    expanded = Path(os.path.expanduser(patient_arg))

    # Direct path candidates first. This preserves support for callers that
    # already pass an absolute/relative patient directory.
    direct_candidates = [
        expanded,
        Path(patient_arg),
        SCRIPT_DIR / patient_arg,
        DEFAULT_LIDC_DATA_ROOT / patient_arg,
        DEFAULT_LIDC_ROOT / patient_arg,
    ]

    for candidate in direct_candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate.absolute()
        if resolved.exists() and resolved.is_dir():
            try:
                patient_id = _extract_lidc_patient_id(resolved)
            except ValueError:
                # If this is a bare patient directory with a non-standard
                # name, retain the historical behavior and use its basename.
                patient_id = resolved.name

            # Always return the actual LIDC-IDRI patient directory when the
            # supplied path is somewhere underneath it.
            parts = list(resolved.parts)
            for i, part in enumerate(parts):
                if part.upper().startswith("LIDC-IDRI-"):
                    patient_root = Path(*parts[: i + 1])
                    return patient_root, part
            return resolved, patient_id

    # If only the patient ID was supplied, search the actual archive root.
    # This also handles case differences in the ID on Windows.
    wanted = Path(patient_arg).name.upper()
    if wanted.startswith("LIDC-IDRI-") and DEFAULT_LIDC_DATA_ROOT.is_dir():
        for candidate in DEFAULT_LIDC_DATA_ROOT.iterdir():
            if candidate.is_dir() and candidate.name.upper() == wanted:
                return candidate, candidate.name

    raise FileNotFoundError(
        "Patient directory not found. Tried:\n"
        + "\n".join(f"  {p}" for p in direct_candidates)
        + f"\n\nExpected LIDC layout: "
        f"{DEFAULT_LIDC_DATA_ROOT / patient_arg}"
    )


def run_pipeline(
    patient_arg,
    air_threshold=float(STAGE01_PREPROCESS["air_threshold_hu"]),
    lung_threshold=float(STAGE01_PREPROCESS["lung_threshold_hu"]),
    z_crop_margin_mm=float(STAGE01_PREPROCESS["z_crop_margin_mm"]),
    z_crop_min_area_frac=float(STAGE01_PREPROCESS["z_crop_min_area_fraction"]),
    core_relative_size_threshold=float(STAGE01_PREPROCESS["core_relative_size_threshold"]),
    max_hole_area_mm2=float(STAGE01_PREPROCESS["max_hole_area_mm2"]),
    apply_z_crop=bool(STAGE01_PREPROCESS["z_crop_enabled_by_default"]),
    blank_value=float(STAGE01_PREPROCESS["default_blank_value_hu"]),
):
    patient_dir, patient_id = resolve_patient_dir(patient_arg)

    out_dir = resolve_template(str(STAGE01_PATHS["patient_output"]), root=DEFAULT_OUTPUT_ROOT, patient_id=patient_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Patient: {patient_id}")
    print(f"[info] Input:   {patient_dir}")
    print(f"[info] Output:  {out_dir}")

    # ------------------------------------------------------------------
    # 01_dicom_to_hu.py functionality -- retained in this standalone file
    # ------------------------------------------------------------------
    print("[info] Scanning recursively for DICOM CT images...")
    series = load_ct_series(str(patient_dir))

    print(f"[info] Ordering {len(series.datasets)} CT slices...")
    ordered_datasets, slice_spacing_mm = order_slices(series.datasets)

    print("[info] Converting pixel data to Hounsfield Units...")
    volume_hu = slices_to_hu_volume(ordered_datasets)
    original_volume_hu = volume_hu.copy()

    meta = extract_metadata(ordered_datasets, slice_spacing_mm)
    meta["meta_schema_version"] = SCHEMA["schema_version"]
    meta["stage_schema_version"] = STAGE01_SCHEMA["metadata_schema_version"]
    meta["source_patient_dir"] = str(patient_dir)
    meta["source_shape_zyx"] = list(volume_hu.shape)
    meta["source_num_slices"] = int(volume_hu.shape[0])
    meta["hu_min"] = float(volume_hu.min())
    meta["hu_max"] = float(volume_hu.max())
    meta["hu_mean"] = float(volume_hu.mean())

    # Preserve the original multi-window representation, but crop it along
    # Z after segmentation so it stays aligned with the final saved volume.
    print("[info] Building original 3-channel CT window volume...")
    volume_windowed = build_multi_window_volume(volume_hu)
    meta["multi_window_saved"] = True
    meta["multi_window_channels"] = list(WINDOWS_HU.keys())
    meta["multi_window_hu"] = {
        "lung": list(LUNG_WINDOW_HU),
        "soft_tissue": list(SOFT_TISSUE_WINDOW_HU),
        "bone": list(BONE_WINDOW_HU),
    }

    # ------------------------------------------------------------------
    # 02_mask_and_crop.py functionality -- retained in this standalone file
    # ------------------------------------------------------------------
    print(
        f"[info] Computing physical air mask "
        f"(threshold <= {air_threshold} HU)..."
    )
    air_mask = compute_air_mask(volume_hu, air_threshold=air_threshold)

    print(
        f"[info] Segmenting lung region "
        f"(threshold < {lung_threshold} HU + 3D component identity + "
        f"airway bridging + an expanding 3D central-rod cardiac/mediastinal "
        f"tracker + wall margin)..."
    )
    lung_mask, core_z_indices, heart_mask, heart_tracking_diag = compute_lung_mask(
        volume_hu,
        lung_threshold=lung_threshold,
        core_relative_size_threshold=core_relative_size_threshold,
        max_hole_area_mm2=max_hole_area_mm2,
        pixel_spacing_mm=meta.get("pixel_spacing_mm", [1.0, 1.0]),
        slice_spacing_mm=meta.get("slice_spacing_mm", 1.0) or 1.0,
        return_diagnostics=True,
    )

    meta["air_threshold_hu"] = air_threshold
    meta["air_voxel_fraction"] = float(air_mask.mean())
    meta["lung_threshold_hu"] = lung_threshold
    meta["lung_voxel_fraction"] = float(lung_mask.mean())
    meta["core_relative_size_threshold"] = core_relative_size_threshold
    meta["max_hole_area_mm2"] = max_hole_area_mm2
    meta["heart_voxel_fraction"] = float(heart_mask.mean())
    meta["heart_mask_found"] = bool(heart_mask.any())
    meta["heart_mask_method"] = str(HEART_TRACKING_SCHEMA["method"])
    meta["heart_mask_tracking_diagnostics"] = heart_tracking_diag
    meta["heart_mask_tracking_successful"] = bool(
        heart_mask.any() and heart_tracking_diag.get("status") == "tracked_central_3d_component"
    )
    # A nonempty heuristic mask is not, by itself, proof of anatomical quality.
    # Keep the review gate closed until a person checks the generated QC montage.
    meta["heart_mask_exclusion_verified"] = False
    meta["heart_mask_review_status"] = "pending_visual_review" if heart_mask.any() else "blocked_tracker_failed"

    if lung_mask.any():
        indices = np.where(lung_mask)
        assert len(indices) == 3, "Expected a 3D lung mask when computing bounding box."
        zs = np.asarray(indices[0], dtype=np.intp)
        ys = np.asarray(indices[1], dtype=np.intp)
        xs = np.asarray(indices[2], dtype=np.intp)
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
        (
            volume_hu_cropped,
            air_mask,
            lung_mask,
            heart_mask,
            meta,
        ) = crop_to_lung_slices(
            volume_hu,
            air_mask,
            lung_mask,
            heart_mask,
            meta,
            core_z_indices,
            margin_mm=z_crop_margin_mm,
            z_crop_min_area_frac=z_crop_min_area_frac,
        )

        if meta.get("z_crop_applied"):
            z0, z1 = meta["z_crop_range"]
            volume_windowed = volume_windowed[z0:z1 + 1]
        else:
            meta["z_crop_range"] = [0, int(volume_hu_cropped.shape[0] - 1)]
            meta["crop_offset_zyx"] = [0, 0, 0]
    else:
        volume_hu_cropped = volume_hu
        meta["z_crop_applied"] = False
        meta["z_crop_range"] = [0, int(volume_hu.shape[0] - 1)]
        meta["crop_offset_zyx"] = [0, 0, 0]
        meta["z_crop_margin_mm"] = z_crop_margin_mm
        meta["z_crop_min_area_frac"] = z_crop_min_area_frac
        meta["num_slices"] = int(volume_hu.shape[0])

    print("[info] Blanking non-lung voxels...")
    volume_hu_masked = blank_non_lung(
        volume_hu_cropped,
        lung_mask,
        blank_value=blank_value,
    )
    # Additional aligned numeric NPY output requested for direct inspection:
    # heart/central-mediastinal mask voxels are replaced with the configured
    # physical-air HU. The raw detection input volume_hu_cropped.npy is kept
    # unchanged for compatibility with the existing detector contract.
    volume_hu_heart_excluded = build_heart_excluded_volume(
        volume_hu_cropped,
        heart_mask,
        replacement_hu=AIR_REPLACEMENT_HU,
    )
    meta["non_lung_blank_value_hu"] = blank_value
    meta["heart_excluded_volume_file"] = STAGE01_FILES["volume_hu_heart_excluded.npy"]
    meta["heart_excluded_volume_shape_zyx"] = list(volume_hu_heart_excluded.shape)
    meta["heart_excluded_volume_replacement_hu"] = float(AIR_REPLACEMENT_HU)
    meta["heart_excluded_volume_is_detector_input"] = False

    # Final-volume coordinate metadata. No resampling is performed here, so
    # spacing and direction are unchanged by this standalone consolidation.
    direction = np.asarray(meta.get("direction_lps", np.eye(3).tolist()), dtype=np.float64)
    if direction.shape != (3, 3):
        raise ValueError("metadata contains an invalid direction_lps matrix")
    origin = np.asarray(meta.get("origin_mm", [0.0, 0.0, 0.0]), dtype=np.float64)
    spacing = np.asarray(meta.get("spacing_zyx_mm", [slice_spacing_mm, 1.0, 1.0]), dtype=np.float64)
    final_shape = tuple(int(v) for v in volume_hu_cropped.shape)
    last_index = np.asarray(final_shape, dtype=np.float64) - 1.0

    meta["final_shape_zyx"] = list(final_shape)
    meta["num_slices"] = int(final_shape[0])
    # The coordinate contract is written directly from the canonical schema.
    meta["coordinate_format"] = dict(get_required(SCHEMA, "coordinate_format", expected_type=dict))
    meta["voxel_000_lps_mm"] = origin.tolist()
    meta["last_voxel_index_zyx"] = [int(v) for v in last_index]
    meta["last_voxel_lps_mm"] = (
        origin + direction @ (last_index * spacing)
    ).tolist()
    meta["final_outputs"] = dict(OUTPUT_SCHEMA)

    np.save(out_dir / STAGE01_FILES["volume_hu.npy"], original_volume_hu.astype(np.float32))
    np.save(out_dir / STAGE01_FILES["volume_hu_cropped.npy"], np.asarray(volume_hu_cropped, dtype=np.float32))
    np.save(out_dir / STAGE01_FILES["volume_hu_masked.npy"], volume_hu_masked.astype(np.float32))
    np.save(out_dir / STAGE01_FILES["volume_hu_heart_excluded.npy"], volume_hu_heart_excluded.astype(np.float32))
    np.save(out_dir / STAGE01_FILES["volume_windowed.npy"], volume_windowed.astype(np.uint8))
    np.save(out_dir / STAGE01_FILES["air_mask.npy"], air_mask.astype(bool))
    np.save(out_dir / STAGE01_FILES["lung_mask.npy"], lung_mask.astype(bool))
    np.save(out_dir / STAGE01_FILES["heart_mask.npy"], heart_mask.astype(bool))
    review_montages, review_slice_metrics = write_heart_mask_review_montages(
        volume_hu_cropped,
        lung_mask,
        heart_mask,
        out_dir,
        SCHEMA,
    )
    meta["heart_mask_review_montages_generated"] = True
    meta["heart_mask_review_montage_files"] = review_montages
    meta["heart_mask_review_metrics_file"] = STAGE01_FILES["heart_mask_review_metrics.json"]
    meta["heart_mask_review_slice_count"] = len(review_slice_metrics)
    meta["heart_mask_review_status"] = (
        "pending_visual_review" if heart_mask.any() and lung_mask.any() else "blocked_empty_mask"
    )
    with open(out_dir / STAGE01_FILES["heart_mask_review_metrics.json"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "heart-mask-review-metrics-v1",
                "meta_schema_version": SCHEMA["schema_version"],
                "patient_id": patient_id,
                "coordinate_order": "ZYX",
                "ct_shape_zyx": list(volume_hu_cropped.shape),
                "slice_metrics": review_slice_metrics,
                "montage_files": review_montages,
            },
            f,
            indent=2,
            allow_nan=False,
        )
    with open(out_dir / STAGE01_FILES["heart_tracking_diagnostics.json"], "w", encoding="utf-8") as f:
        json.dump({
            "patient_id": patient_id,
            "meta_schema_version": SCHEMA["schema_version"],
            "coordinate_system": "DICOM_LPS",
            "array_axes": ["z", "y", "x"],
            "source_volume_shape_zyx": list(volume_hu.shape),
            "final_volume_shape_zyx": list(volume_hu_cropped.shape),
            "crop_offset_zyx": list(meta.get("crop_offset_zyx", [0, 0, 0])),
            "heart_mask_tracking": heart_tracking_diag,
            "heart_mask_found": bool(heart_mask.any()),
            "heart_mask_tracking_successful": bool(meta["heart_mask_tracking_successful"]),
            "heart_mask_exclusion_verified": False,
            "heart_mask_review_status": str(meta["heart_mask_review_status"]),
            "heart_mask_review_montage_files": review_montages,
            "heart_mask_review_slice_count": len(review_slice_metrics),
            "note": "Heuristic 3D central-rod tracking diagnostic; image-level review is required before candidate filtering is approved.",
        }, f, indent=2, allow_nan=False)

    with open(out_dir / STAGE01_FILES["meta.json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, allow_nan=False)

    print(f"[done] Full HU volume shape: {original_volume_hu.shape} (Z, Y, X)")
    print(f"[done] Final HU volume shape: {final_shape} (Z, Y, X)")
    print(f"[done] Final spacing: {meta['spacing_zyx_mm']} mm")
    print(f"[done] Output directory: {out_dir}")
    print(
        "[done] Wrote volume_hu.npy, volume_hu_cropped.npy, "
        "volume_hu_masked.npy, volume_hu_heart_excluded.npy, volume_windowed.npy, "
        "air_mask.npy, lung_mask.npy, heart_mask.npy, heart_tracking_diagnostics.json, meta.json"
    )

    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone LungInsight v2 preprocessing: "
            "DICOM -> HU -> masks -> Z-crop."
        )
    )
    parser.add_argument(
        "patient",
        help=(
            "Patient ID such as LIDC-IDRI-0001, resolved from "
            "datasets/LIDC/lidc_idri, or an explicit path to a patient "
            "directory (or a directory below one)."
        ),
    )
    parser.add_argument(
        "--air-threshold", type=float, default=float(STAGE01_PREPROCESS["air_threshold_hu"]),
        help="HU cutoff at/below which a voxel is classified as physical air. Default: -950.",
    )
    parser.add_argument(
        "--lung-threshold", type=float, default=float(STAGE01_PREPROCESS["lung_threshold_hu"]),
        help="HU cutoff used to segment the lung region's aerated core. Default: -320.",
    )
    parser.add_argument(
        "--core-relative-size-threshold", type=float, default=float(STAGE01_PREPROCESS["core_relative_size_threshold"]),
        help="3D organ-identity relative component size threshold. Default: 0.12.",
    )
    parser.add_argument(
        "--max-hole-area-mm2", type=float, default=float(STAGE01_PREPROCESS["max_hole_area_mm2"]),
        help="Largest per-slice enclosed-hole area filled inside the lung silhouette. Default: 2500 mm2.",
    )
    parser.add_argument(
        "--z-crop-margin-mm", type=float, default=float(STAGE01_PREPROCESS["z_crop_margin_mm"]),
        help="Physical margin kept above/below the identified lung extent. Default: 10 mm.",
    )
    parser.add_argument(
        "--z-crop-min-area-frac", type=float, default=float(STAGE01_PREPROCESS["z_crop_min_area_fraction"]),
        help="Per-slice verified-lung area-fraction threshold for Z selection. Default: 0.008.",
    )
    parser.add_argument(
        "--blank-value", type=float, default=float(STAGE01_PREPROCESS["default_blank_value_hu"]),
        help="HU written into voxels outside lung_mask. Default: -1000.",
    )
    parser.add_argument(
        "--no-z-crop", action="store_true",
        help="Disable Z-cropping and keep every source slice.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        args.patient,
        air_threshold=args.air_threshold,
        lung_threshold=args.lung_threshold,
        z_crop_margin_mm=args.z_crop_margin_mm,
        z_crop_min_area_frac=args.z_crop_min_area_frac,
        core_relative_size_threshold=args.core_relative_size_threshold,
        max_hole_area_mm2=args.max_hole_area_mm2,
        apply_z_crop=(not args.no_z_crop) and bool(STAGE01_PREPROCESS["z_crop_enabled_by_default"]),
        blank_value=args.blank_value,
    )


if __name__ == "__main__":
    main()
