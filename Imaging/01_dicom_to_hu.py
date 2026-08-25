"""
01_dicom_to_hu.py

STEP 1 of 3 in the pipeline:
    01_dicom_to_hu.py       <- this file: DICOM -> HU volume (this file)
    02_mask_and_crop.py     -> lung segmentation + non-lung blanking + Z-crop
    03_visualize.py         -> viewing

Reads a stack of CT DICOM (.dcm) slices for a single LIDC-IDRI patient,
recursively searching sub-folders (LIDC-IDRI patients are typically
structured as <PatientID>/<StudyUID>/<SeriesUID>/*.dcm), assembles them
into a properly ordered 3D volume, and converts raw pixel values to
Hounsfield Units (HU).

This script does NOT do any lung segmentation, masking, or slice
cropping -- that's step 2 (02_mask_and_crop.py). This keeps "get the
physics right" (HU conversion, slice ordering, series selection)
cleanly separated from "anatomical segmentation" (which is a much
more involved, tunable step and easy to get subtly wrong).

Usage:
    python 01_dicom_to_hu.py "Imaging/LIDC/lidc_idri/LIDC-IDRI-0001" \
        --out-dir output/LIDC-IDRI-0001

Outputs (written to --out-dir):
    volume_hu.npy       -> float32 array, shape (Z, Y, X), values in HU
    volume_windowed.npy -> uint8 array, shape (Z, Y, X, 3), contrast-
                            enhanced channels = [lung window, soft-
                            tissue window, bone window] (see below).
                            Pass --no-multi-window to skip this.
    meta.json            -> spacing, origin, HU stats, window params, etc.

Why volume_windowed.npy exists: raw HU spans roughly -1000 (air) to
+2000 (dense bone) -- far too wide a range for any single linear
0-255 stretch to show good contrast in air, soft tissue, AND bone at
once (whichever one the stretch is tuned for, the other two get
crushed into a handful of gray levels). volume_windowed.npy instead
applies three separate clinical CT windows (lung / soft-tissue / bone)
and stacks them as channels, each stretched to use the full 0-255
range on just its own tissue band. Useful both for visual inspection
and as ready-made 3-channel input to a CNN.

Notes on correctness (things that commonly go wrong with LIDC-IDRI):
  * A patient folder can contain MULTIPLE series (e.g. a localizer/
    scout series plus the real axial CT series). We group slices by
    SeriesInstanceUID and pick the series with the most axial slices,
    since that's virtually always the diagnostic CT volume.
  * Non-image DICOM files (e.g. RTSTRUCT / SEG annotation files that
    LIDC-IDRI also ships) must be skipped -- they have no PixelData.
  * Slices must be ordered by their actual spatial position
    (ImagePositionPatient z-coordinate), NOT by filename or
    InstanceNumber, since these are not guaranteed to match spatial
    order.
  * HU conversion must be applied EXACTLY ONCE:
        HU = pixel_array * RescaleSlope + RescaleIntercept
    Applying it twice (e.g. once during loading and again later in a
    pipeline) silently produces wrong intensities that still "look"
    plausible -- a classic hard-to-catch bug.
  * Some scanners pad the outside of the reconstruction circle with a
    fixed value (often -2000) to mark "outside scan field of view".
    We clip those to -1000 (air) before saving so they don't get
    treated as a separate exotic material downstream.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    print(
        "pydicom is required. Install it with:\n"
        "    pip install pydicom --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(1)


# Value scanners commonly use to pad pixels outside the reconstructed
# field of view. We treat anything at or below this as "out of scan"
# and remap it to air (-1000 HU) rather than letting it corrupt HU
# stats / masks as a fake ultra-dense or ultra-negative material.
OUT_OF_SCAN_PIXEL_HU_FLOOR = -2000
AIR_REPLACEMENT_HU = -1000

# CT windows (center, width) in HU, used to build a contrast-enhanced
# multi-window volume. A single linear stretch of raw HU can't show
# good contrast for air, soft tissue, AND bone at once -- they span
# roughly -1000 to +2000+ HU, so any one stretch that's wide enough to
# reach bone crushes the soft-tissue range into a handful of gray
# levels, and vice versa. Windowing each tissue class separately (then
# treating them as channels) is the standard fix radiologists and CT
# preprocessing pipelines both use.
LUNG_WINDOW_HU = (-600, 1500)      # center, width -- air vs aerated parenchyma
SOFT_TISSUE_WINDOW_HU = (40, 400)  # center, width -- mediastinum/soft tissue detail
BONE_WINDOW_HU = (400, 1800)       # center, width -- trabecular vs cortical bone


@dataclass
class SeriesSlices:
    series_uid: str
    datasets: List["pydicom.dataset.FileDataset"] = field(default_factory=list)


def find_dicom_files(patient_dir: str) -> List[str]:
    """Recursively collect every candidate DICOM file under patient_dir.

    Resolves ~, tries the provided path, a script-relative path, and
    the absolute path, in that order.
    """
    # Expand ~ to the user home directory first
    expanded = os.path.expanduser(patient_dir)

    # Candidate paths to try (as absolute paths for clear error messages)
    script_dir = os.path.dirname(__file__)
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


def order_slices(datasets: List["pydicom.dataset.FileDataset"]):
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


def convert_patient_to_hu(
    patient_dir: str,
    out_dir: str,
    save_multi_window: bool = True,
):
    """
    STEP 1: DICOM -> ordered, HU-converted volume. No segmentation, no
    masking, no cropping -- see 02_mask_and_crop.py for that.
    """
    print(f"[info] Scanning '{patient_dir}' for DICOM slices (recursive)...")
    series = load_ct_series(patient_dir)

    print(f"[info] Ordering {len(series.datasets)} slices by spatial position...")
    ordered_datasets, slice_spacing_mm = order_slices(series.datasets)

    print("[info] Converting pixel data to Hounsfield Units...")
    volume_hu = slices_to_hu_volume(ordered_datasets)

    meta = extract_metadata(ordered_datasets, slice_spacing_mm)
    meta["hu_min"] = float(volume_hu.min())
    meta["hu_max"] = float(volume_hu.max())
    meta["hu_mean"] = float(volume_hu.mean())

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "volume_hu.npy"), volume_hu)

    if save_multi_window:
        print("[info] Building contrast-enhanced multi-window volume "
              "(lung / soft-tissue / bone channels)...")
        volume_windowed = build_multi_window_volume(volume_hu)
        np.save(os.path.join(out_dir, "volume_windowed.npy"), volume_windowed)
        meta["multi_window_saved"] = True
        meta["multi_window_channels"] = ["lung", "soft_tissue", "bone"]
        meta["multi_window_hu"] = {
            "lung": list(LUNG_WINDOW_HU),
            "soft_tissue": list(SOFT_TISSUE_WINDOW_HU),
            "bone": list(BONE_WINDOW_HU),
        }
    else:
        meta["multi_window_saved"] = False

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] Volume shape: {volume_hu.shape} (Z, Y, X)")
    print(f"[done] HU range: [{meta['hu_min']:.1f}, {meta['hu_max']:.1f}]")
    outputs_written = "volume_hu.npy"
    if save_multi_window:
        outputs_written += ", volume_windowed.npy"
    outputs_written += ", meta.json"
    print(f"[done] Wrote {outputs_written} -> '{out_dir}'")
    print("[done] Next: run 02_mask_and_crop.py on this output directory "
          "to segment the lungs and crop to lung-containing slices.")

    return volume_hu, meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="STEP 1/3: Convert a LIDC-IDRI patient's DICOM CT "
        "slices to an HU volume (no masking/cropping)."
    )
    parser.add_argument(
        "patient_dir",
        help="Path to the patient folder, e.g. "
        "'Imaging/LIDC/lidc_idri/LIDC-IDRI-0001'. May contain subfolders.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write volume_hu.npy / meta.json "
        "(default: '<patient_dir>_processed').",
    )
    parser.add_argument(
        "--no-multi-window",
        action="store_true",
        help="Skip building volume_windowed.npy (the contrast-enhanced "
        "lung/soft-tissue/bone 3-channel volume); only save raw HU.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or (r"output/" + args.patient_dir.rstrip("/\\") + "_processed")
    convert_patient_to_hu(
        args.patient_dir, out_dir,
        save_multi_window=not args.no_multi_window,
    )


if __name__ == "__main__":
    main()