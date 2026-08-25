"""
05_extract_candidate_patches.py

STEP 5: Extract fixed-size classifier patches.

Pipeline:
    01_dicom_to_hu.py
        ->
    02_mask_and_crop.py
        ->
    04_detect_candidates.py
        ->
    05_extract_candidate_patches.py
        ->
    06 inference_cpu.py

For every Stage-04 candidate:

    candidate center in Stage-02 voxel coordinates
                    |
                    v
            64 x 64 x 64 crop
                    |
                    v
          original HU volume

The default source is:

    volume_hu.npy

rather than:

    volume_hu_masked.npy

This is intentional. The classifier should see real surrounding
anatomical context rather than having the entire non-lung region replaced
with artificial -1000 HU air.

The candidate coordinates remain in exactly the same ZYX voxel grid as
the Stage-02 volume.

IMPORTANT:
    This script DOES NOT resize or resample the patch.

The classifier receives the exact 64^3 voxel patch expected by
inference_cpu.py.

If a candidate is near a volume boundary, the crop is padded with
-1000 HU while preserving the candidate's original coordinate within
the 64^3 patch. The crop is never shifted inward merely to avoid padding.

Outputs:
    <candidate_id>_patch.npy
    patch_manifest.csv
    meta.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


AIR_HU = -1000.0


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def load_source_volume(
    volume_dir: str,
    source_filename: str,
):
    vol_path = os.path.join(
        volume_dir,
        source_filename,
    )

    meta_path = os.path.join(
        volume_dir,
        "meta.json",
    )

    if not os.path.exists(vol_path):
        raise FileNotFoundError(
            f"Source volume not found:\n"
            f"    {vol_path}\n\n"
            f"Run 02_mask_and_crop.py first."
        )

    volume = np.load(vol_path)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume, got shape "
            f"{volume.shape}."
        )

    meta = {}

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    return volume, meta


def load_candidates(
    candidates_dir: str,
):
    candidates_path = os.path.join(
        candidates_dir,
        "candidates.csv",
    )

    if not os.path.exists(candidates_path):
        raise FileNotFoundError(
            f"Candidates file not found:\n"
            f"    {candidates_path}\n\n"
            f"Run 04_detect_candidates.py first."
        )

    df = pd.read_csv(
        candidates_path
    )

    required = {
        "candidate_id",
        "center_z",
        "center_y",
        "center_x",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "candidates.csv is missing required "
            f"columns: {sorted(missing)}"
        )

    return df


# ---------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------

def crop_cubic_patch(
    volume: np.ndarray,
    center_zyx,
    patch_size: int,
    pad_value: float = AIR_HU,
):
    """
    Extract a fixed-size cubic patch centered on center_zyx.

    Coordinates:
        volume: Z,Y,X
        center_zyx: float voxel coordinates

    The center is rounded to the nearest voxel.

    If the requested crop extends outside the volume, it is padded
    with AIR_HU.

    Crucially, the crop is NOT shifted inward at the boundary.
    """

    if patch_size <= 0:
        raise ValueError(
            f"patch_size must be positive, got {patch_size}"
        )

    if patch_size % 2 != 0:
        raise ValueError(
            "patch_size must be even so the candidate can remain "
            "symmetrically centered. "
            f"Got {patch_size}."
        )

    center_float = np.asarray(
        center_zyx,
        dtype=np.float64,
    )

    if center_float.shape != (3,):
        raise ValueError(
            f"center_zyx must have shape (3), "
            f"got {center_float.shape}"
        )

    if not np.all(
        np.isfinite(center_float)
    ):
        raise ValueError(
            f"Non-finite candidate center: "
            f"{center_float}"
        )

    center = np.rint(
        center_float
    ).astype(np.int64)

    half = patch_size // 2

    lo = center - half
    hi = lo + patch_size

    volume_shape = np.asarray(
        volume.shape,
        dtype=np.int64,
    )

    pad_before = np.maximum(
        -lo,
        0,
    )

    pad_after = np.maximum(
        hi - volume_shape,
        0,
    )

    was_padded = bool(
        np.any(pad_before)
        or np.any(pad_after)
    )

    if was_padded:

        padded = np.pad(
            volume,
            [
                (
                    int(pad_before[i]),
                    int(pad_after[i]),
                )
                for i in range(3)
            ],
            mode="constant",
            constant_values=pad_value,
        )

        lo_padded = (
            lo + pad_before
        )

        hi_padded = (
            lo_padded + patch_size
        )

        patch = padded[
            lo_padded[0]:hi_padded[0],
            lo_padded[1]:hi_padded[1],
            lo_padded[2]:hi_padded[2],
        ]

    else:

        patch = volume[
            lo[0]:hi[0],
            lo[1]:hi[1],
            lo[2]:hi[2],
        ]

    expected_shape = (
        patch_size,
        patch_size,
        patch_size,
    )

    if patch.shape != expected_shape:
        raise RuntimeError(
            f"Internal crop error: expected "
            f"{expected_shape}, got {patch.shape}. "
            f"center={center}, lo={lo}, hi={hi}, "
            f"volume_shape={volume.shape}"
        )

    # Location of candidate center INSIDE patch coordinates.
    #
    # For a normal interior crop this is exactly [32,32,32].
    # Near a boundary, padding moves the volume rather than the candidate.
    center_in_patch = (
        center - lo + pad_before
    ).astype(np.int64)

    return (
        patch.astype(np.float32),
        was_padded,
        center,
        lo,
        hi,
        pad_before,
        pad_after,
        center_in_patch,
    )


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------

def extract_candidate_patches(
    candidates_dir: str,
    volume_dir: str,
    out_dir: str,
    patch_size: int = 64,
    source_filename: str = "volume_hu.npy",
):
    candidates_df = load_candidates(
        candidates_dir
    )

    print(
        f"[info] Loading '{source_filename}' "
        f"from '{volume_dir}'..."
    )

    volume, volume_meta = load_source_volume(
        volume_dir,
        source_filename,
    )

    print(
        f"[info] Source volume shape: "
        f"{volume.shape}"
    )

    # -------------------------------------------------------------
    # Coordinate integrity check
    # -------------------------------------------------------------

    detector_meta_path = os.path.join(
        candidates_dir,
        "meta.json",
    )

    detector_meta = {}

    if os.path.exists(
        detector_meta_path
    ):
        with open(
            detector_meta_path,
            "r",
        ) as f:
            detector_meta = json.load(f)

    detector_shape = detector_meta.get(
        "volume_shape_zyx"
    )

    if detector_shape is not None:

        detector_shape = tuple(
            int(v)
            for v in detector_shape
        )

        if detector_shape != tuple(
            volume.shape
        ):
            raise ValueError(
                "\nStage-04 / Stage-05 coordinate mismatch.\n\n"
                f"Detector volume shape: {detector_shape}\n"
                f"Patch source shape:   {volume.shape}\n\n"
                "The candidate voxel coordinates cannot safely be "
                "used with this source volume.\n\n"
                "Use the same Stage-02 volume/grid for both stages."
            )

    # -------------------------------------------------------------
    # Spacing check
    # -------------------------------------------------------------

    spacing = volume_meta.get(
        "spacing_zyx_mm",
        detector_meta.get(
            "spacing_zyx_mm",
            [1.0, 1.0, 1.0],
        ),
    )

    spacing = np.asarray(
        spacing,
        dtype=np.float32,
    )

    if spacing.shape != (3,):
        raise ValueError(
            f"Invalid spacing_zyx_mm: {spacing}"
        )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    manifest_rows = []

    # -------------------------------------------------------------
    # Candidate loop
    # -------------------------------------------------------------

    for row_index, row in candidates_df.iterrows():

        candidate_id = str(
            row["candidate_id"]
        )

        center_zyx = np.asarray(
            [
                float(row["center_z"]),
                float(row["center_y"]),
                float(row["center_x"]),
            ],
            dtype=np.float64,
        )

        # ---------------------------------------------------------
        # Validate center against source volume.
        # ---------------------------------------------------------

        if not np.all(
            np.isfinite(center_zyx)
        ):
            raise ValueError(
                f"{candidate_id}: invalid center "
                f"{center_zyx}"
            )

        # Candidate may be slightly outside due to detector padding
        # or floating-point box decoding. A truly outside center is
        # almost certainly a Stage-04 bug.
        if np.any(
            center_zyx < -1.0
        ) or np.any(
            center_zyx > (
                np.asarray(volume.shape)
                + 1.0
            )
        ):
            raise ValueError(
                f"{candidate_id}: candidate center "
                f"{center_zyx} lies outside source volume "
                f"shape {volume.shape}."
            )

        (
            patch,
            was_padded,
            center_vox,
            crop_lo,
            crop_hi,
            pad_before,
            pad_after,
            center_in_patch,
        ) = crop_cubic_patch(
            volume=volume,
            center_zyx=center_zyx,
            patch_size=patch_size,
            pad_value=AIR_HU,
        )

        # ---------------------------------------------------------
        # Save patch.
        # ---------------------------------------------------------

        patch_filename = (
            f"{candidate_id}_patch.npy"
        )

        patch_path = os.path.join(
            out_dir,
            patch_filename,
        )

        np.save(
            patch_path,
            patch,
        )

        # ---------------------------------------------------------
        # Useful physical metadata.
        # ---------------------------------------------------------

        crop_size_mm = (
            np.asarray(
                [patch_size] * 3,
                dtype=np.float32,
            )
            * spacing
        )

        center_mm_relative = (
            center_zyx * spacing
        )

        crop_lo_clipped = np.maximum(
            crop_lo,
            0,
        )

        crop_hi_clipped = np.minimum(
            crop_hi,
            np.asarray(
                volume.shape
            ),
        )

        actual_voxel_extent = (
            crop_hi_clipped
            - crop_lo_clipped
        )

        # ---------------------------------------------------------
        # Manifest row.
        # ---------------------------------------------------------

        manifest_row = row.to_dict()

        manifest_row.update(
            {
                "patch_path": os.path.abspath(
                    patch_path
                ),

                "patch_size": int(
                    patch_size
                ),

                "source_volume": source_filename,

                "border_padded": bool(
                    was_padded
                ),

                "center_z_rounded": int(
                    center_vox[0]
                ),
                "center_y_rounded": int(
                    center_vox[1]
                ),
                "center_x_rounded": int(
                    center_vox[2]
                ),

                "center_in_patch_z": int(
                    center_in_patch[0]
                ),
                "center_in_patch_y": int(
                    center_in_patch[1]
                ),
                "center_in_patch_x": int(
                    center_in_patch[2]
                ),

                "crop_z_lo": int(
                    crop_lo[0]
                ),
                "crop_z_hi_exclusive": int(
                    crop_hi[0]
                ),

                "crop_y_lo": int(
                    crop_lo[1]
                ),
                "crop_y_hi_exclusive": int(
                    crop_hi[1]
                ),

                "crop_x_lo": int(
                    crop_lo[2]
                ),
                "crop_x_hi_exclusive": int(
                    crop_hi[2]
                ),

                "pad_z_before": int(
                    pad_before[0]
                ),
                "pad_y_before": int(
                    pad_before[1]
                ),
                "pad_x_before": int(
                    pad_before[2]
                ),

                "pad_z_after": int(
                    pad_after[0]
                ),
                "pad_y_after": int(
                    pad_after[1]
                ),
                "pad_x_after": int(
                    pad_after[2]
                ),

                "actual_z_voxels": int(
                    actual_voxel_extent[0]
                ),
                "actual_y_voxels": int(
                    actual_voxel_extent[1]
                ),
                "actual_x_voxels": int(
                    actual_voxel_extent[2]
                ),

                "center_z_mm_relative": float(
                    center_mm_relative[0]
                ),
                "center_y_mm_relative": float(
                    center_mm_relative[1]
                ),
                "center_x_mm_relative": float(
                    center_mm_relative[2]
                ),

                "patch_z_mm": float(
                    crop_size_mm[0]
                ),
                "patch_y_mm": float(
                    crop_size_mm[1]
                ),
                "patch_x_mm": float(
                    crop_size_mm[2]
                ),
            }
        )

        manifest_rows.append(
            manifest_row
        )

        if was_padded:
            print(
                f"[warn] {candidate_id}: "
                f"border padding required. "
                f"pad_before={pad_before.tolist()}, "
                f"pad_after={pad_after.tolist()}, "
                f"center_in_patch={center_in_patch.tolist()}"
            )

    # -------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    manifest_path = os.path.join(
        out_dir,
        "patch_manifest.csv",
    )

    manifest_df.to_csv(
        manifest_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    patch_meta = {
        "source_volume_dir": os.path.abspath(
            volume_dir
        ),
        "source_volume": source_filename,

        "candidates_dir": os.path.abspath(
            candidates_dir
        ),

        "patch_size_zyx": [
            patch_size,
            patch_size,
            patch_size,
        ],

        "padding_value_hu": AIR_HU,

        "volume_shape_zyx": list(
            volume.shape
        ),

        "spacing_zyx_mm": spacing.tolist(),

        "num_candidates": int(
            len(manifest_df)
        ),
    }

    with open(
        os.path.join(
            out_dir,
            "meta.json",
        ),
        "w",
    ) as f:
        json.dump(
            patch_meta,
            f,
            indent=2,
        )

    print(
        f"[done] Wrote {len(manifest_df)} "
        f"patch(es) -> '{out_dir}'"
    )

    print(
        f"[done] Manifest -> '{manifest_path}'"
    )

    print(
        "[done] Every patch is float32 "
        f"{patch_size}x{patch_size}x{patch_size} HU."
    )

    return manifest_df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "STEP 5: Extract fixed-size classifier "
            "patches around Stage-04 candidates."
        )
    )

    parser.add_argument(
        "candidates_dir",
        help=(
            "Stage-04 output directory containing "
            "candidates.csv."
        ),
    )

    parser.add_argument(
        "--volume-dir",
        required=True,
        help=(
            "Stage-02 output directory containing "
            "the source HU volume."
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. Default: "
            "'<candidates_dir>_patches'."
        ),
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help=(
            "Classifier patch side length. "
            "Default 64."
        ),
    )

    parser.add_argument(
        "--source",
        default="volume_hu.npy",
        help=(
            "Source HU filename. Default "
            "'volume_hu.npy'. Use "
            "'volume_hu_masked.npy' only if you "
            "explicitly want lung-blanked classifier "
            "inputs."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    out_dir = (
        args.out_dir
        or (
            args.candidates_dir.rstrip("/\\")
            + "_patches"
        )
    )

    extract_candidate_patches(
        candidates_dir=args.candidates_dir,
        volume_dir=args.volume_dir,
        out_dir=out_dir,
        patch_size=args.patch_size,
        source_filename=args.source,
    )


if __name__ == "__main__":
    main()