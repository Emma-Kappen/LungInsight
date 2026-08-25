"""
build_lidc_h5.py

Build a compact HDF5 training dataset of annotated LIDC-IDRI nodule
patches for LungInsight classifier training.

INPUT
-----
Imaging/LIDC/
    LIDC-IDRI-XXXX/
        ... DICOM files ...

OUTPUT
------
Imaging/LIDC/lung_nodule_patches.h5

Each H5 sample contains:

    patch:
        Raw HU patch, shape (64, 64, 64), float32

    labels:
        8 LIDC radiological characteristics

    patient_id:
        LIDC-IDRI patient identifier

    scan_id:
        pylidc scan identifier

    annotation_index:
        Index of the annotation within the scan

    centroid:
        Annotation centroid in voxel coordinates (Z, Y, X)

The eight classifier targets are:

    calcification
    lobulation
    malignancy
    margin
    sphericity
    spiculation
    subtlety
    texture

The following pylidc fields are intentionally NOT used:

    density
    internalStructure

IMPORTANT
---------
This script does NOT use stages 01, 02, or 03.

It reads the original LIDC DICOM scans directly and creates the
training patches used by the classifier.

The patches are extracted in the native LIDC voxel coordinate system
and then resampled to a fixed 64 x 64 x 64 tensor.

The resulting H5 contains RAW HU values. Any classifier-specific
normalization should be performed in the training dataset/loader so
that training and inference use exactly the same normalization.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pylidc as pl
from scipy.ndimage import zoom


# =====================================================================
# CONFIGURATION
# =====================================================================

PATCH_SIZE = 64

# Physical side length of the patch in millimetres.
#
# The classifier receives 64^3 voxels, but we first define the patch
# physically so that scans with different voxel spacing are treated
# consistently.
#
# 64 mm is deliberately used here because it gives the classifier
# substantial surrounding lung context around a nodule.
PATCH_MM = 64.0

# HU values outside this range are not useful for the classifier and
# can create unnecessarily large numerical ranges.
#
# NOTE:
# This is clipping, NOT normalization.
HU_MIN = -1000.0
HU_MAX = 400.0

FEATURE_NAMES = [
    "calcification",
    "lobulation",
    "malignancy",
    "margin",
    "sphericity",
    "spiculation",
    "subtlety",
    "texture",
]


# =====================================================================
# pylidc FIELD ACCESS
# =====================================================================

def get_annotation_value(annotation, feature_name: str) -> float:
    """
    Read one LIDC annotation characteristic.

    pylidc stores these characteristics as integer ratings.

    If the characteristic is unavailable, NaN is returned.
    """

    value = getattr(annotation, feature_name, None)

    if value is None:
        return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def get_labels(annotation) -> np.ndarray:
    """
    Return the eight classifier targets in the canonical order.
    """

    return np.asarray(
        [
            get_annotation_value(annotation, name)
            for name in FEATURE_NAMES
        ],
        dtype=np.float32,
    )


# =====================================================================
# DICOM / VOLUME HELPERS
# =====================================================================

def load_scan_volume(scan) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a pylidc scan into a numpy HU volume.

    Returns
    -------
    volume:
        3D HU volume.

    spacing:
        Voxel spacing in mm, ordered as (Z, Y, X).
    """

    volume = scan.to_volume()

    volume = np.asarray(volume, dtype=np.float32)

    if volume.ndim != 3:
        raise RuntimeError(
            f"Expected 3D scan volume, got shape {volume.shape}"
        )

    # pylidc's scan.to_volume() is indexed as Z,Y,X.
    #
    # Get the physical spacing from the DICOM metadata.
    #
    # pylidc exposes slice thickness and pixel spacing through
    # the scan's DICOM objects.
    try:
        first_slice = scan.load_all_dicom_images()[0]

        pixel_spacing = np.asarray(
            first_slice.PixelSpacing,
            dtype=np.float32,
        )

        slice_thickness = float(
            getattr(
                first_slice,
                "SliceThickness",
                1.0,
            )
        )

        spacing = np.asarray(
            [
                slice_thickness,
                pixel_spacing[0],
                pixel_spacing[1],
            ],
            dtype=np.float32,
        )

    except Exception:
        #
        # pylidc normally provides sufficient metadata. If metadata
        # cannot be read, fall back to 1 mm isotropic spacing rather
        # than silently producing a malformed patch.
        #
        raise RuntimeError(
            "Could not determine DICOM voxel spacing for scan "
            f"{getattr(scan, 'id', '<unknown>')}."
        )

    return volume, spacing


# =====================================================================
# CENTROID / PATCH EXTRACTION
# =====================================================================

def annotation_centroid_voxel(annotation) -> np.ndarray:
    """
    Return annotation centroid as (Z, Y, X) voxel coordinates.

    pylidc Annotation.centroid is already expressed in voxel
    coordinates for the scan.
    """

    centroid = np.asarray(
        annotation.centroid,
        dtype=np.float32,
    )

    if centroid.shape != (3,):
        raise RuntimeError(
            f"Unexpected annotation centroid shape: {centroid.shape}"
        )

    return centroid


def extract_physical_patch(
    volume: np.ndarray,
    spacing: np.ndarray,
    center_zyx: np.ndarray,
    patch_mm: float = PATCH_MM,
    output_size: int = PATCH_SIZE,
) -> Optional[np.ndarray]:
    """
    Extract a fixed physical-size patch around a nodule centroid.

    The patch is extracted from the ORIGINAL HU volume.

    Input coordinate convention:
        (Z, Y, X)

    Output:
        (64, 64, 64)

    Procedure
    ---------
    1. Convert the requested physical patch size into native voxels.
    2. Extract the surrounding native HU region.
    3. Pad with air (-1000 HU) when the nodule is near a scan boundary.
    4. Resample to exactly 64^3.

    This avoids silently losing nodules close to the lung/scan boundary.
    """

    if volume.ndim != 3:
        raise ValueError(
            f"Volume must be 3D, got {volume.shape}"
        )

    spacing = np.asarray(spacing, dtype=np.float32)

    if spacing.shape != (3,):
        raise ValueError(
            f"Spacing must have shape (3,), got {spacing.shape}"
        )

    # Number of native voxels corresponding to PATCH_MM.
    native_size = np.maximum(
        np.ceil(patch_mm / spacing).astype(np.int32),
        3,
    )

    # Make the native extraction dimensions odd where possible so the
    # centroid remains approximately at the center voxel.
    native_size += (native_size % 2 == 0).astype(np.int32)

    half = native_size // 2

    center = np.rint(center_zyx).astype(np.int32)

    start = center - half
    end = start + native_size

    # ---------------------------------------------------------------
    # Clip source coordinates to the available volume.
    # ---------------------------------------------------------------

    src_start = np.maximum(start, 0)
    src_end = np.minimum(
        end,
        np.asarray(volume.shape, dtype=np.int32),
    )

    if np.any(src_end <= src_start):
        return None

    # ---------------------------------------------------------------
    # Create an air-filled destination patch.
    #
    # This is important for peripheral nodules. We do not discard
    # them merely because the requested context extends outside the
    # CT volume.
    # ---------------------------------------------------------------

    patch = np.full(
        tuple(native_size),
        HU_MIN,
        dtype=np.float32,
    )

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    patch[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = volume[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]

    # ---------------------------------------------------------------
    # Clip HU range.
    # ---------------------------------------------------------------

    patch = np.clip(
        patch,
        HU_MIN,
        HU_MAX,
    )

    # ---------------------------------------------------------------
    # Resample to classifier size.
    #
    # scipy zoom operates in the same Z,Y,X ordering.
    # ---------------------------------------------------------------

    zoom_factor = (
        output_size / patch.shape[0],
        output_size / patch.shape[1],
        output_size / patch.shape[2],
    )

    patch = zoom(
        patch,
        zoom=zoom_factor,
        order=1,
        mode="nearest",
        prefilter=False,
    )

    # Floating-point rounding can occasionally result in a dimension
    # being one voxel off. Force exact geometry if necessary.
    if patch.shape != (
        output_size,
        output_size,
        output_size,
    ):
        patch = _center_crop_or_pad(
            patch,
            output_size,
        )

    return patch.astype(np.float32, copy=False)


def _center_crop_or_pad(
    volume: np.ndarray,
    size: int,
) -> np.ndarray:
    """
    Force a 3D volume to exactly (size,size,size).

    Cropping/padding is centered independently on each axis.
    Padding uses air HU.
    """

    result = np.full(
        (size, size, size),
        HU_MIN,
        dtype=np.float32,
    )

    src_slices = []
    dst_slices = []

    for current in volume.shape:

        if current >= size:
            src_start = (current - size) // 2
            src_end = src_start + size

            dst_start = 0
            dst_end = size

        else:
            src_start = 0
            src_end = current

            dst_start = (size - current) // 2
            dst_end = dst_start + current

        src_slices.append(
            slice(src_start, src_end)
        )

        dst_slices.append(
            slice(dst_start, dst_end)
        )

    result[
        dst_slices[0],
        dst_slices[1],
        dst_slices[2],
    ] = volume[
        src_slices[0],
        src_slices[1],
        src_slices[2],
    ]

    return result


# =====================================================================
# H5 DATASET WRITER
# =====================================================================

class H5PatchWriter:
    """
    Incrementally writes patches to an HDF5 file.

    This avoids keeping the complete LIDC dataset in RAM.
    """

    def __init__(
        self,
        path: str,
        initial_capacity: int = 256,
    ):

        self.path = path

        self.file = h5py.File(
            path,
            "w",
        )

        self.capacity = initial_capacity
        self.count = 0

        # -------------------------------------------------------------
        # Main patch tensor.
        # -------------------------------------------------------------

        self.patches = self.file.create_dataset(
            "patches",
            shape=(
                initial_capacity,
                PATCH_SIZE,
                PATCH_SIZE,
                PATCH_SIZE,
            ),
            maxshape=(
                None,
                PATCH_SIZE,
                PATCH_SIZE,
                PATCH_SIZE,
            ),
            dtype=np.float32,
            chunks=(
                1,
                PATCH_SIZE,
                PATCH_SIZE,
                PATCH_SIZE,
            ),
            compression="gzip",
            compression_opts=4,
        )

        # -------------------------------------------------------------
        # Eight regression targets.
        # -------------------------------------------------------------

        self.labels = self.file.create_dataset(
            "labels",
            shape=(initial_capacity, len(FEATURE_NAMES)),
            maxshape=(None, len(FEATURE_NAMES)),
            dtype=np.float32,
            chunks=(min(256, initial_capacity), len(FEATURE_NAMES)),
            compression="gzip",
            compression_opts=4,
        )

        # -------------------------------------------------------------
        # Patient / annotation metadata.
        #
        # Store strings as fixed-length UTF-8-compatible byte arrays.
        # -------------------------------------------------------------

        self.patient_ids = self.file.create_dataset(
            "patient_id",
            shape=(initial_capacity,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

        self.scan_ids = self.file.create_dataset(
            "scan_id",
            shape=(initial_capacity,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

        self.annotation_indices = self.file.create_dataset(
            "annotation_index",
            shape=(initial_capacity,),
            maxshape=(None,),
            dtype=np.int32,
        )

        self.centroids = self.file.create_dataset(
            "centroid_zyx",
            shape=(initial_capacity, 3),
            maxshape=(None, 3),
            dtype=np.float32,
        )

        self.feature_names = self.file.create_dataset(
            "feature_names",
            data=np.asarray(
                FEATURE_NAMES,
                dtype=h5py.string_dtype(encoding="utf-8"),
            ),
        )

        # -------------------------------------------------------------
        # File-level metadata.
        # -------------------------------------------------------------

        self.file.attrs["patch_size"] = PATCH_SIZE
        self.file.attrs["patch_mm"] = PATCH_MM
        self.file.attrs["hu_min"] = HU_MIN
        self.file.attrs["hu_max"] = HU_MAX
        self.file.attrs["coordinate_order"] = "Z,Y,X"
        self.file.attrs["source"] = "LIDC-IDRI"
        self.file.attrs["description"] = (
            "Annotated LIDC nodule patches for LungInsight "
            "3D multi-head classifier training."
        )

    def _grow(self):

        new_capacity = self.capacity * 2

        self.patches.resize(
            (new_capacity, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
        )

        self.labels.resize(
            (new_capacity, len(FEATURE_NAMES))
        )

        self.patient_ids.resize(
            (new_capacity,)
        )

        self.scan_ids.resize(
            (new_capacity,)
        )

        self.annotation_indices.resize(
            (new_capacity,)
        )

        self.centroids.resize(
            (new_capacity, 3)
        )

        self.capacity = new_capacity

    def append(
        self,
        patch: np.ndarray,
        labels: np.ndarray,
        patient_id: str,
        scan_id: str,
        annotation_index: int,
        centroid: np.ndarray,
    ):

        if self.count >= self.capacity:
            self._grow()

        i = self.count

        self.patches[i] = patch
        self.labels[i] = labels
        self.patient_ids[i] = patient_id
        self.scan_ids[i] = scan_id
        self.annotation_indices[i] = annotation_index
        self.centroids[i] = centroid

        self.count += 1

    def close(self):

        # Shrink datasets to their actual size.
        self.patches.resize(
            (self.count, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
        )

        self.labels.resize(
            (self.count, len(FEATURE_NAMES))
        )

        self.patient_ids.resize(
            (self.count,)
        )

        self.scan_ids.resize(
            (self.count,)
        )

        self.annotation_indices.resize(
            (self.count,)
        )

        self.centroids.resize(
            (self.count, 3)
        )

        self.file.attrs["num_patches"] = self.count
        self.file.attrs["num_features"] = len(FEATURE_NAMES)

        self.file.flush()
        self.file.close()


# =====================================================================
# LIDC PROCESSING
# =====================================================================

def find_lidc_scans(lidc_root: str):
    """
    Find all pylidc Scan objects underneath the supplied LIDC root.

    pylidc itself determines the actual DICOM structure.
    """

    print()
    print("=" * 72)
    print("QUERYING LIDC DATABASE")
    print("=" * 72)

    # pylidc uses its configured database/cache to discover scans.
    #
    # We still verify that the requested directory exists because the
    # user should explicitly point this script at Imaging/LIDC.
    if not os.path.isdir(lidc_root):
        raise FileNotFoundError(
            f"LIDC directory does not exist:\n{lidc_root}"
        )

    scans = pl.query(pl.Scan).all()

    print(f"pylidc returned {len(scans)} scans.")

    return scans


def process_scan(
    scan,
    writer: H5PatchWriter,
    stats: Dict[str, int],
):

    patient_id = str(
        getattr(
            scan,
            "patient_id",
            f"scan_{scan.id}",
        )
    )

    scan_id = str(
        getattr(
            scan,
            "id",
            "",
        )
    )

    print(
        f"\nProcessing {patient_id} "
        f"(scan id {scan_id})"
    )

    # ---------------------------------------------------------------
    # Load volume once per scan.
    # ---------------------------------------------------------------

    try:

        volume, spacing = load_scan_volume(scan)

    except Exception as exc:

        print(
            f"  [ERROR] Could not load volume: {exc}"
        )

        stats["scans_failed"] += 1

        return

    print(
        f"  Volume: {volume.shape}"
    )

    print(
        f"  Spacing: "
        f"{spacing[0]:.3f}, "
        f"{spacing[1]:.3f}, "
        f"{spacing[2]:.3f} mm"
    )

    # ---------------------------------------------------------------
    # Get individual radiologist annotations.
    #
    # We intentionally keep annotations separate.
    #
    # The later train/validation split MUST be patient-aware.
    # ---------------------------------------------------------------

    try:

        annotations = scan.annotations

    except Exception as exc:

        print(
            f"  [WARN] Could not retrieve annotations: {exc}"
        )

        stats["scans_failed"] += 1

        return

    if not annotations:

        print("  No annotations.")
        stats["scans_without_annotations"] += 1
        return

    print(
        f"  Annotations: {len(annotations)}"
    )

    for annotation_index, annotation in enumerate(
        annotations
    ):

        stats["annotations_seen"] += 1

        try:

            # -------------------------------------------------------
            # Get target labels.
            # -------------------------------------------------------

            labels = get_labels(annotation)

            # An annotation is only useful if at least one of the
            # classifier targets exists.
            if np.isnan(labels).all():

                stats["annotations_without_labels"] += 1

                print(
                    f"    Annotation {annotation_index}: "
                    "no usable labels -> skipped"
                )

                continue

            # -------------------------------------------------------
            # Centroid.
            # -------------------------------------------------------

            centroid = annotation_centroid_voxel(
                annotation
            )

            # -------------------------------------------------------
            # Extract fixed physical-size HU patch.
            # -------------------------------------------------------

            patch = extract_physical_patch(
                volume=volume,
                spacing=spacing,
                center_zyx=centroid,
                patch_mm=PATCH_MM,
                output_size=PATCH_SIZE,
            )

            if patch is None:

                stats["patches_failed"] += 1

                print(
                    f"    Annotation {annotation_index}: "
                    "patch extraction failed"
                )

                continue

            # -------------------------------------------------------
            # Write to H5.
            # -------------------------------------------------------

            writer.append(
                patch=patch,
                labels=labels,
                patient_id=patient_id,
                scan_id=scan_id,
                annotation_index=annotation_index,
                centroid=centroid,
            )

            stats["patches_written"] += 1

            valid_labels = [
                FEATURE_NAMES[i]
                for i, value in enumerate(labels)
                if np.isfinite(value)
            ]

            print(
                f"    Annotation {annotation_index}: "
                f"patch={patch.shape}, "
                f"labels={valid_labels}"
            )

        except Exception as exc:

            stats["annotations_failed"] += 1

            print(
                f"    [ERROR] Annotation "
                f"{annotation_index}: {exc}"
            )


# =====================================================================
# MAIN
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Convert LIDC-IDRI annotated nodules into a compact "
            "64x64x64 HDF5 training dataset."
        )
    )

    parser.add_argument(
        "--lidc-root",
        default=os.path.join(
            "Imaging",
            "LIDC",
        ),
        help=(
            "Root LIDC directory. Default: Imaging/LIDC"
        ),
    )

    parser.add_argument(
        "--output",
        default=os.path.join(
            "Imaging",
            "LIDC",
            "lung_nodule_patches.h5",
        ),
        help=(
            "Output H5 path."
        ),
    )

    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help=(
            "Optional maximum number of scans to process. "
            "Useful for testing."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Reserved for deterministic future dataset operations."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    np.random.seed(args.seed)

    print("=" * 72)
    print("LUNGINSIGHT LIDC -> HDF5 DATASET BUILDER")
    print("=" * 72)

    print()
    print(f"LIDC root : {os.path.abspath(args.lidc_root)}")
    print(f"Output    : {os.path.abspath(args.output)}")
    print(f"Patch     : {PATCH_SIZE} x {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Patch FOV : {PATCH_MM:.1f} mm")
    print(f"HU range  : [{HU_MIN:.1f}, {HU_MAX:.1f}]")
    print()
    print("Features:")
    for feature in FEATURE_NAMES:
        print(f"  - {feature}")

    print()
    print(
        "WARNING: pylidc must be configured to point at the "
        "same LIDC dataset."
    )

    # ---------------------------------------------------------------
    # Make output directory.
    # ---------------------------------------------------------------

    output_dir = os.path.dirname(
        os.path.abspath(args.output)
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # Prevent accidental overwrite.
    if os.path.exists(args.output):

        raise FileExistsError(
            f"\nOutput file already exists:\n"
            f"{args.output}\n\n"
            "Delete it manually or choose another --output path."
        )

    # ---------------------------------------------------------------
    # Find scans.
    # ---------------------------------------------------------------

    scans = find_lidc_scans(
        args.lidc_root
    )

    if args.max_scans is not None:

        scans = scans[:args.max_scans]

        print(
            f"TEST MODE: processing only "
            f"{len(scans)} scans."
        )

    if not scans:

        raise RuntimeError(
            "No LIDC scans were found."
        )

    # ---------------------------------------------------------------
    # H5 writer.
    # ---------------------------------------------------------------

    writer = H5PatchWriter(
        args.output
    )

    stats = {
        "scans_seen": 0,
        "scans_failed": 0,
        "scans_without_annotations": 0,
        "annotations_seen": 0,
        "annotations_failed": 0,
        "annotations_without_labels": 0,
        "patches_failed": 0,
        "patches_written": 0,
    }

    try:

        for scan_number, scan in enumerate(
            scans,
            start=1,
        ):

            stats["scans_seen"] += 1

            print()
            print(
                f"[{scan_number}/{len(scans)}]"
            )

            process_scan(
                scan=scan,
                writer=writer,
                stats=stats,
            )

            # Flush periodically so a long run has data on disk.
            if (
                stats["patches_written"] > 0
                and stats["patches_written"] % 100 == 0
            ):

                writer.file.flush()

                print(
                    f"  H5 flushed: "
                    f"{stats['patches_written']} patches"
                )

    finally:

        writer.close()

    # ---------------------------------------------------------------
    # Final report.
    # ---------------------------------------------------------------

    print()
    print("=" * 72)
    print("DATASET BUILD COMPLETE")
    print("=" * 72)

    for key, value in stats.items():
        print(
            f"{key:32s}: {value}"
        )

    print()
    print(
        f"H5 file:\n"
        f"  {os.path.abspath(args.output)}"
    )

    print()
    print(
        "The H5 file contains RAW HU patches with shape "
        "(N, 64, 64, 64)."
    )


if __name__ == "__main__":
    main()