"""
04_detect_candidates.py

STEP 4: High-recall nodule candidate detection.

Pipeline:
    01_dicom_to_hu.py
        ->
    02_mask_and_crop.py
        ->
    03_visualize.py
        ->
    04_detect_candidates.py
        ->
    05_extract_candidate_patches.py
        ->
    06 inference_cpu.py

Detector:
    rlsn/LungNoduleDetection-derived 3D CNN + ViT.

IMPORTANT:
    The upstream detector was trained on LUNA16 CT volumes using
    native [40, 128, 128] voxel crops and the following normalization:

        mean = -775.657161489884
        std  =  962.3208802005623

    The upstream evaluation:
        - uses the detector logits directly
        - accepts candidates with logit > -5
        - converts predicted normalized boxes directly to voxel coordinates
        - merges nearby detections using physical center distance (~10 mm)

    This implementation follows that behavior rather than using an
    aggressive sigmoid > 0.5 threshold or 3D IoU NMS.

INPUT:
    --volume-dir should normally point at the output directory of
    02_mask_and_crop.py.

    By default the detector reads:

        volume_hu.npy

    NOT:

        volume_hu_masked.npy

    This is intentional. The original rlsn detector was trained on
    ordinary CT volumes, not lung-blanked volumes. The lung mask can
    still be used later for visualization/filtering.

OUTPUT:
    candidates.csv
    meta.json

Candidate coordinates are always expressed in the ZYX voxel coordinate
system of volume_hu.npy / volume_hu_masked.npy from Stage 02.
"""

import argparse
import json
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from vitdet3d import (
    DEFAULT_CROP_SIZE,
    LUNA16_MEAN,
    LUNA16_STD,
    load_vitdet3d_checkpoint,
)


AIR_HU = -1000.0

# Exact upstream detector behavior.
DEFAULT_LOGIT_THRESHOLD = -5.0

# Upstream eval merges candidates within approximately 10 mm.
DEFAULT_MERGE_DISTANCE_MM = 10.0


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def load_volume(volume_dir: str, filename: str):
    vol_path = os.path.join(volume_dir, filename)
    meta_path = os.path.join(volume_dir, "meta.json")

    if not os.path.exists(vol_path):
        raise FileNotFoundError(
            f"Volume not found:\n"
            f"    {vol_path}\n\n"
            f"Run 02_mask_and_crop.py first."
        )

    volume = np.load(vol_path)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume, got shape {volume.shape} "
            f"from '{vol_path}'."
        )

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    return volume, meta


def get_spacing_zyx_mm(meta) -> np.ndarray:
    spacing = meta.get("spacing_zyx_mm", [1.0, 1.0, 1.0])

    spacing = np.asarray(spacing, dtype=np.float32)

    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)):
        raise ValueError(
            f"Invalid spacing_zyx_mm in meta.json: {spacing}"
        )

    if np.any(spacing <= 0):
        raise ValueError(
            f"spacing_zyx_mm must be positive, got {spacing}"
        )

    return spacing


# ---------------------------------------------------------------------
# Sliding-window utilities
# ---------------------------------------------------------------------

def sliding_window_offsets(
    volume_shape: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
    """
    Enumerate sliding-window starts in Z,Y,X.

    Matches the upstream rlsn behavior:
        np.arange(size-window)[::stride]
        + final flush-to-edge position

    If an axis is smaller than the detector window, start=0 is used.
    The caller pads the volume before extraction.
    """

    offsets_per_axis = []

    for axis_len, win, st in zip(volume_shape, window_size, stride):

        if axis_len <= win:
            offsets_per_axis.append([0])
            continue

        # Same intent as upstream:
        # range up to size-win, then explicit final edge window.
        starts = list(range(0, axis_len - win, st))
        starts.append(axis_len - win)

        # Deduplicate.
        starts = list(dict.fromkeys(starts))

        offsets_per_axis.append(starts)

    return [
        (z, y, x)
        for z in offsets_per_axis[0]
        for y in offsets_per_axis[1]
        for x in offsets_per_axis[2]
    ]


def pad_to_window(
    volume: np.ndarray,
    window_size: Tuple[int, int, int],
    pad_value: float = AIR_HU,
):
    """
    Pad only when the volume is smaller than the detector window.

    Padding is appended to the high side of each axis, preserving the
    original voxel coordinate system.
    """

    pads = [
        max(0, int(w - s))
        for s, w in zip(volume.shape, window_size)
    ]

    if not any(pads):
        return volume, np.zeros(3, dtype=np.int32)

    pad_width = [(0, p) for p in pads]

    padded = np.pad(
        volume,
        pad_width,
        mode="constant",
        constant_values=pad_value,
    )

    return padded, np.asarray(pads, dtype=np.int32)


# ---------------------------------------------------------------------
# Detector output conversion
# ---------------------------------------------------------------------

def decode_detector_box(
    bbox_frac: np.ndarray,
    window_origin_zyx: np.ndarray,
    crop_size_zyx: np.ndarray,
    original_shape_zyx: np.ndarray,
):
    """
    Convert detector output from normalized crop coordinates into
    absolute voxel coordinates in the ORIGINAL volume.

    Detector output:
        [z_lo, y_lo, x_lo, z_hi, y_hi, x_hi]

    is expressed as a fraction of the [40,128,128] detector window.
    """

    bbox_frac = np.asarray(bbox_frac, dtype=np.float32)

    if bbox_frac.shape != (6,):
        raise ValueError(
            f"Expected bbox shape (6,), got {bbox_frac.shape}"
        )

    if not np.all(np.isfinite(bbox_frac)):
        return None

    lo_frac = bbox_frac[:3]
    hi_frac = bbox_frac[3:]

    # Convert normalized coordinates to detector-window voxels.
    lo_local = lo_frac * crop_size_zyx
    hi_local = hi_frac * crop_size_zyx

    # The training target is low/high normalized coordinates.
    # Be defensive about pathological predictions.
    lo_local, hi_local = np.minimum(lo_local, hi_local), np.maximum(
        lo_local, hi_local
    )

    # Convert to absolute voxel coordinates.
    abs_lo = window_origin_zyx + lo_local
    abs_hi = window_origin_zyx + hi_local

    # Clip to the ORIGINAL, unpadded volume.
    abs_lo = np.maximum(abs_lo, 0.0)
    abs_hi = np.minimum(abs_hi, original_shape_zyx.astype(np.float32))

    # Reject boxes with no physical intersection with the real volume.
    if np.any(abs_hi <= abs_lo):
        return None

    # Ensure finite values.
    if not (
        np.all(np.isfinite(abs_lo))
        and np.all(np.isfinite(abs_hi))
    ):
        return None

    return abs_lo, abs_hi


# ---------------------------------------------------------------------
# Sliding-window detector
# ---------------------------------------------------------------------

def run_sliding_window_detection(
    volume_hu: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    crop_size: Tuple[int, int, int],
    stride_fraction: float,
    batch_size: int,
    logit_threshold: float,
):
    """
    Run the detector over the complete volume.

    Returns raw detections before physical-distance merging.

    Each detection contains:
        confidence_logit
        probability
        bbox_lo
        bbox_hi
        center_zyx
        diameter_vox_approx
    """

    if stride_fraction <= 0 or stride_fraction > 1:
        raise ValueError(
            f"stride_fraction must be in (0,1], got {stride_fraction}"
        )

    if batch_size < 1:
        raise ValueError(
            f"batch_size must be >= 1, got {batch_size}"
        )

    original_shape = np.asarray(volume_hu.shape, dtype=np.int32)
    crop_size_arr = np.asarray(crop_size, dtype=np.float32)

    stride = tuple(
        max(1, int(round(c * stride_fraction)))
        for c in crop_size
    )

    padded, _ = pad_to_window(
        volume_hu,
        crop_size,
        pad_value=AIR_HU,
    )

    offsets = sliding_window_offsets(
        padded.shape,
        crop_size,
        stride,
    )

    print(
        f"[info] detector window={crop_size}, "
        f"stride={stride}, "
        f"windows={len(offsets)}"
    )

    # Exact upstream LUNA16 normalization.
    normalized = (
        padded.astype(np.float32) - LUNA16_MEAN
    ) / LUNA16_STD

    detections = []

    model.eval()

    with torch.no_grad():

        for batch_start in range(
            0,
            len(offsets),
            batch_size,
        ):

            batch_offsets = offsets[
                batch_start:batch_start + batch_size
            ]

            patches = np.stack(
                [
                    normalized[
                        z:z + crop_size[0],
                        y:y + crop_size[1],
                        x:x + crop_size[2],
                    ]
                    for z, y, x in batch_offsets
                ],
                axis=0,
            )

            pixel_values = (
                torch.from_numpy(patches)
                .unsqueeze(1)
                .to(device)
            )

            output = model(pixel_values)

            logits = (
                output.logits
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )

            bbox_frac = (
                output.bbox
                .detach()
                .cpu()
                .numpy()
            )

            for i, (z, y, x) in enumerate(batch_offsets):

                logit = float(logits[i])

                # IMPORTANT:
                # Do NOT use sigmoid(logit) >= 0.5 here.
                #
                # Upstream evaluation keeps logit > -5, corresponding
                # to approximately 0.0067 probability.
                if not np.isfinite(logit):
                    continue

                if logit <= logit_threshold:
                    continue

                origin = np.asarray(
                    [z, y, x],
                    dtype=np.float32,
                )

                decoded = decode_detector_box(
                    bbox_frac[i],
                    origin,
                    crop_size_arr,
                    original_shape,
                )

                if decoded is None:
                    continue

                abs_lo, abs_hi = decoded

                center = (abs_lo + abs_hi) / 2.0

                size_vox = np.maximum(
                    abs_hi - abs_lo,
                    1e-3,
                )

                probability = float(
                    1.0 / (1.0 + math.exp(-np.clip(logit, -60, 60)))
                )

                detections.append(
                    {
                        "confidence_logit": logit,
                        "confidence": probability,
                        "bbox_lo": abs_lo,
                        "bbox_hi": abs_hi,
                        "center_zyx": center,
                        "size_zyx_vox": size_vox,
                    }
                )

    return detections


# ---------------------------------------------------------------------
# Physical-distance merging
# ---------------------------------------------------------------------

def merge_detections_by_distance(
    detections: List[dict],
    spacing_zyx_mm: np.ndarray,
    merge_distance_mm: float,
) -> List[dict]:
    """
    Merge detections that represent the same physical nodule.

    This follows the strategy used by the original rlsn evaluation:
    candidate centers within approximately 10 mm are grouped.

    Instead of IoU NMS, which is unreliable for small nodules whose
    predicted boxes may vary significantly between overlapping windows,
    center-distance merging is used.
    """

    if not detections:
        return []

    spacing = np.asarray(
        spacing_zyx_mm,
        dtype=np.float32,
    )

    # Sort highest-confidence first.
    detections = sorted(
        detections,
        key=lambda d: d["confidence_logit"],
        reverse=True,
    )

    clusters = []

    for det in detections:

        center_mm = (
            det["center_zyx"] * spacing
        )

        assigned = False

        for cluster in clusters:

            cluster_center_mm = np.mean(
                [
                    d["center_zyx"] * spacing
                    for d in cluster
                ],
                axis=0,
            )

            distance = float(
                np.linalg.norm(
                    center_mm - cluster_center_mm
                )
            )

            if distance <= merge_distance_mm:
                cluster.append(det)
                assigned = True
                break

        if not assigned:
            clusters.append([det])

    merged = []

    for cluster in clusters:

        # Highest-confidence detection controls the confidence and
        # representative box.
        best = max(
            cluster,
            key=lambda d: d["confidence_logit"],
        )

        # Mean center follows the upstream merge behavior and is more
        # stable than choosing an arbitrary sliding-window center.
        center = np.mean(
            [
                d["center_zyx"]
                for d in cluster
            ],
            axis=0,
        )

        # Estimate diameter from the cluster's predicted box sizes.
        sizes_mm = np.asarray(
            [
                d["size_zyx_vox"] * spacing
                for d in cluster
            ],
            dtype=np.float32,
        )

        # Upstream uses the equivalent of:
        # norm(size_mm) / sqrt(3)
        diameter_mm = float(
            np.mean(
                np.linalg.norm(
                    sizes_mm,
                    axis=1,
                ) / math.sqrt(3.0)
            )
        )

        # Recenter the representative box around the merged center.
        best_size_vox = best["size_zyx_vox"]

        bbox_lo = center - best_size_vox / 2.0
        bbox_hi = center + best_size_vox / 2.0

        merged.append(
            {
                "confidence_logit": float(
                    best["confidence_logit"]
                ),
                "confidence": float(
                    best["confidence"]
                ),
                "center_zyx": center.astype(np.float32),
                "bbox_lo": bbox_lo.astype(np.float32),
                "bbox_hi": bbox_hi.astype(np.float32),
                "diameter_mm": diameter_mm,
                "num_merged_windows": len(cluster),
            }
        )

    merged.sort(
        key=lambda d: d["confidence_logit"],
        reverse=True,
    )

    return merged


# ---------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------

def detect_candidates(
    volume_dir: str,
    checkpoint_path: str,
    out_dir: str,
    source_filename: str = "volume_hu.npy",
    crop_size: Tuple[int, int, int] = DEFAULT_CROP_SIZE,
    stride_fraction: float = 0.75,
    logit_threshold: float = DEFAULT_LOGIT_THRESHOLD,
    merge_distance_mm: float = DEFAULT_MERGE_DISTANCE_MM,
    batch_size: int = 4,
    device_str: str = "cpu",
):
    if tuple(crop_size) != (40, 128, 128):
        print(
            "[warn] This detector checkpoint was trained for "
            f"[40,128,128]. Requested crop_size={crop_size}."
        )

    device = torch.device(device_str)

    print(
        f"[info] Loading detector source volume "
        f"'{source_filename}'..."
    )

    volume_hu, meta = load_volume(
        volume_dir,
        source_filename,
    )

    spacing_zyx_mm = get_spacing_zyx_mm(meta)

    print(
        f"[info] Volume shape: {volume_hu.shape}"
    )

    print(
        f"[info] Spacing ZYX: "
        f"{spacing_zyx_mm.tolist()} mm"
    )

    print(
        f"[info] Detector normalization: "
        f"mean={LUNA16_MEAN}, std={LUNA16_STD}"
    )

    print(
        f"[info] Detector threshold: "
        f"logit > {logit_threshold:.4f} "
        f"(probability > "
        f"{1.0 / (1.0 + math.exp(-logit_threshold)):.6f})"
    )

    print(
        f"[info] Candidate merge distance: "
        f"{merge_distance_mm:.2f} mm"
    )

    print(
        f"[info] Loading detector checkpoint "
        f"'{checkpoint_path}'..."
    )

    model = load_vitdet3d_checkpoint(
        checkpoint_path,
        device,
    )

    raw_detections = run_sliding_window_detection(
        volume_hu=volume_hu,
        model=model,
        device=device,
        crop_size=crop_size,
        stride_fraction=stride_fraction,
        batch_size=batch_size,
        logit_threshold=logit_threshold,
    )

    print(
        f"[info] Raw detector hits: "
        f"{len(raw_detections)}"
    )

    final_detections = merge_detections_by_distance(
        detections=raw_detections,
        spacing_zyx_mm=spacing_zyx_mm,
        merge_distance_mm=merge_distance_mm,
    )

    print(
        f"[info] Candidates after physical-distance "
        f"merging: {len(final_detections)}"
    )

    volume_shape = np.asarray(
        volume_hu.shape,
        dtype=np.float32,
    )

    rows = []

    for idx, det in enumerate(final_detections):

        candidate_id = f"cand{idx:03d}"

        center = np.asarray(
            det["center_zyx"],
            dtype=np.float32,
        )

        bbox_lo = np.maximum(
            det["bbox_lo"],
            0.0,
        )

        bbox_hi = np.minimum(
            det["bbox_hi"],
            volume_shape,
        )

        center_mm_relative = (
            center * spacing_zyx_mm
        )

        size_mm = (
            np.maximum(
                bbox_hi - bbox_lo,
                0.0,
            )
            * spacing_zyx_mm
        )

        diameter_mm = float(
            det["diameter_mm"]
        )

        rows.append(
            {
                "candidate_id": candidate_id,

                "confidence": float(
                    det["confidence"]
                ),

                "confidence_logit": float(
                    det["confidence_logit"]
                ),

                "center_z": float(center[0]),
                "center_y": float(center[1]),
                "center_x": float(center[2]),

                # These are relative physical coordinates in the
                # Stage-02 ZYX volume grid.
                "center_z_mm": float(
                    center_mm_relative[0]
                ),
                "center_y_mm": float(
                    center_mm_relative[1]
                ),
                "center_x_mm": float(
                    center_mm_relative[2]
                ),

                "bbox_z_lo": float(bbox_lo[0]),
                "bbox_z_hi": float(bbox_hi[0]),

                "bbox_y_lo": float(bbox_lo[1]),
                "bbox_y_hi": float(bbox_hi[1]),

                "bbox_x_lo": float(bbox_lo[2]),
                "bbox_x_hi": float(bbox_hi[2]),

                "bbox_z_size_mm": float(size_mm[0]),
                "bbox_y_size_mm": float(size_mm[1]),
                "bbox_x_size_mm": float(size_mm[2]),

                "diameter_mm": diameter_mm,

                "num_merged_windows": int(
                    det["num_merged_windows"]
                ),
            }
        )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    columns = [
        "candidate_id",
        "confidence",
        "confidence_logit",

        "center_z",
        "center_y",
        "center_x",

        "center_z_mm",
        "center_y_mm",
        "center_x_mm",

        "bbox_z_lo",
        "bbox_z_hi",
        "bbox_y_lo",
        "bbox_y_hi",
        "bbox_x_lo",
        "bbox_x_hi",

        "bbox_z_size_mm",
        "bbox_y_size_mm",
        "bbox_x_size_mm",

        "diameter_mm",
        "num_merged_windows",
    ]

    candidates_df = pd.DataFrame(
        rows,
        columns=columns,
    )

    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(
            "confidence_logit",
            ascending=False,
        ).reset_index(drop=True)

        # Reassign IDs after final sorting.
        candidates_df["candidate_id"] = [
            f"cand{i:03d}"
            for i in range(len(candidates_df))
        ]

    candidates_path = os.path.join(
        out_dir,
        "candidates.csv",
    )

    candidates_df.to_csv(
        candidates_path,
        index=False,
    )

    run_meta = {
        "source_volume": os.path.abspath(
            os.path.join(
                volume_dir,
                source_filename,
            )
        ),
        "source_volume_filename": source_filename,
        "checkpoint_path": os.path.abspath(
            checkpoint_path
        ),

        "crop_size_zyx": list(crop_size),
        "stride_fraction": float(stride_fraction),

        "logit_threshold": float(
            logit_threshold
        ),

        "probability_threshold": float(
            1.0 / (
                1.0 + math.exp(-logit_threshold)
            )
        ),

        "merge_distance_mm": float(
            merge_distance_mm
        ),

        "batch_size": int(batch_size),
        "device": device_str,

        "num_raw_window_hits": len(
            raw_detections
        ),

        "num_candidates": len(
            final_detections
        ),

        "volume_shape_zyx": list(
            volume_hu.shape
        ),

        "spacing_zyx_mm": spacing_zyx_mm.tolist(),

        "normalization_mean": float(
            LUNA16_MEAN
        ),

        "normalization_std": float(
            LUNA16_STD
        ),
    }

    with open(
        os.path.join(out_dir, "meta.json"),
        "w",
    ) as f:
        json.dump(
            run_meta,
            f,
            indent=2,
        )

    print(
        f"[done] Wrote {len(candidates_df)} "
        f"candidates -> '{candidates_path}'"
    )

    print(
        "[done] Next: run "
        "05_extract_candidate_patches.py"
    )

    return candidates_df, run_meta


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "STEP 4: High-recall lung-nodule candidate "
            "detection using the rlsn-derived 3D "
            "CNN+ViT detector."
        )
    )

    parser.add_argument(
        "volume_dir",
        help=(
            "Stage-02 output directory containing "
            "volume_hu.npy and meta.json."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to the DETECTOR checkpoint. "
            "This is NOT best_model_gpu_v2.pth."
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. Default: "
            "'<volume_dir>_candidates'."
        ),
    )

    parser.add_argument(
        "--source",
        default="volume_hu.npy",
        help=(
            "Source volume filename. Default is "
            "'volume_hu.npy' because the upstream "
            "detector was trained on ordinary CT, "
            "not lung-blanked CT."
        ),
    )

    parser.add_argument(
        "--stride-fraction",
        type=float,
        default=0.75,
        help=(
            "Sliding-window stride as fraction of "
            "detector crop size. Default 0.75."
        ),
    )

    parser.add_argument(
        "--logit-threshold",
        type=float,
        default=DEFAULT_LOGIT_THRESHOLD,
        help=(
            "Minimum detector logit. Default -5, "
            "matching the upstream evaluation. "
            "This corresponds to probability "
            "~0.0067."
        ),
    )

    parser.add_argument(
        "--merge-distance-mm",
        type=float,
        default=DEFAULT_MERGE_DISTANCE_MM,
        help=(
            "Physical center distance used to merge "
            "overlapping sliding-window detections. "
            "Default 10 mm."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Detector batch size. Default 4.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device. Default cpu.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    out_dir = (
        args.out_dir
        or (
            args.volume_dir.rstrip("/\\")
            + "_candidates"
        )
    )

    detect_candidates(
        volume_dir=args.volume_dir,
        checkpoint_path=args.checkpoint,
        out_dir=out_dir,
        source_filename=args.source,
        stride_fraction=args.stride_fraction,
        logit_threshold=args.logit_threshold,
        merge_distance_mm=args.merge_distance_mm,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()