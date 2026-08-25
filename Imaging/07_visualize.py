"""
07_visualize_gradcam.py

STAGE 07 -- GEOMETRY-SAFE GRAD-CAM++ DIAGNOSTIC

Purpose
-------
Visualize per-candidate, per-head Grad-CAM++ heatmaps on the exact 64^3
classifier patch produced by Stage 05.

IMPORTANT GEOMETRY CONTRACT
----------------------------
All Stage 07 geometry is expressed in PATCH coordinates.

The Stage 05 patch is created by:

    center = round(center_zyx)
    lo = center - patch_size // 2
    patch = volume[lo : lo + patch_size]

Therefore, for a candidate with volume-space center:

    candidate_local = candidate_volume - patch_origin

For a 64^3 patch, an integer-rounded candidate center is normally near
(32, 32, 32), NOT necessarily (31.5, 31.5, 31.5).

The geometric center of the array is:

    (31.5, 31.5, 31.5)

but the candidate center is determined by the actual Stage 05 crop.

This distinction is critical.

Stage 07 NEVER compares a patch-space Grad-CAM peak directly with
patient/volume coordinates.

AXIAL CONVENTION
----------------
All arrays are treated as:

    (Z, Y, X)

For an axial slice:

    image = patch[z, :, :]
    heatmap = cam[z, :, :]

Matplotlib therefore receives:

    Y rows
    X columns

No transpose or axis permutation is performed.

INPUT
-----
patch_manifest.csv produced by:

    05_extract_candidate_patches.py

Required columns:

    candidate_id
    patch_path
    center_z
    center_y
    center_x

Optional bbox columns:

    bbox_z_lo
    bbox_z_hi
    bbox_y_lo
    bbox_y_hi
    bbox_x_lo
    bbox_x_hi

Optional:

    diameter_mm
    confidence
    border_padded
    source_volume

OUTPUT
------
output-dir/
    figures/
        cand000_gradcam_diagnostic.png
        cand001_gradcam_diagnostic.png
        ...
        all_candidates_malignancy_summary.png

    gradcam_geometry_diagnostics.csv

Each candidate figure contains:

    - CT patch slice
    - per-head Grad-CAM++ heatmap
    - CT + heatmap overlay
    - candidate center
    - optional bbox
    - CAM peak
    - candidate-to-peak displacement
    - geometry warning/status

Usage
-----
python Imaging/07_visualize_gradcam.py ^
    --patch-manifest "output/LIDC-IDRI-0141/05_patches/patch_manifest.csv" ^
    --checkpoint "Imaging/checkpoints/best_model_gpu_v2.pth" ^
    --output-dir "output/LIDC-IDRI-0141/07_gradcam" ^
    --target-layer "backbone.layer2" ^
    --slice-mode "bbox-center"

"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cir_multihead_pipeline import (
    FEATURE_NAMES,
    PATCH_SIZE,
    create_multihead_model,
    generate_characteristic_heatmaps,
)
from inference_cpu import load_checkpoint


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PATCH_SIZE = 64

# Axial CT display window.
# The patch values are HU because Stage 05 saves volume_hu.npy.
CT_WINDOW_MIN = -1000.0
CT_WINDOW_MAX = 400.0

# CAM diagnostic thresholds in patch voxels.
#
# These are deliberately diagnostic rather than filtering thresholds.
CAM_NEAR_CENTER_VOXELS = 8.0
CAM_ACCEPTABLE_CENTER_VOXELS = 16.0

# CAM visualization.
CAM_ALPHA = 0.45

# Small numerical epsilon.
EPS = 1e-8


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def safe_float(value, default=np.nan) -> float:
    """Convert a manifest value to float without crashing on NaN/None."""
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def require_columns(df: pd.DataFrame, columns) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            "patch_manifest.csv is missing required columns: "
            + ", ".join(missing)
        )


def clamp_slice_index(index: int, size: int) -> int:
    return max(0, min(int(index), size - 1))


# ---------------------------------------------------------------------------
# Patch geometry
# ---------------------------------------------------------------------------

def compute_stage05_patch_origin(
    center_zyx: np.ndarray,
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct the exact Stage 05 crop geometry.

    Stage 05 does:

        center = np.round(center_zyx).astype(int)
        lo = center - patch_size // 2

    For a non-border-padded crop, this is the patch origin.

    Returns
    -------
    rounded_center_zyx
        Integer voxel center used by Stage 05.

    patch_origin_zyx
        Volume-space coordinate corresponding to patch[0,0,0].
    """
    rounded_center = np.round(center_zyx).astype(int)
    half = patch_size // 2
    patch_origin = rounded_center - half

    return rounded_center, patch_origin


def volume_to_patch(
    volume_zyx: np.ndarray,
    patch_origin_zyx: np.ndarray,
) -> np.ndarray:
    """
    Convert volume-space ZYX coordinates into patch-local ZYX coordinates.
    """
    return np.asarray(volume_zyx, dtype=np.float64) - np.asarray(
        patch_origin_zyx, dtype=np.float64
    )


def patch_to_volume(
    patch_zyx: np.ndarray,
    patch_origin_zyx: np.ndarray,
) -> np.ndarray:
    """Convert patch-local ZYX coordinates back to volume-space ZYX."""
    return np.asarray(patch_zyx, dtype=np.float64) + np.asarray(
        patch_origin_zyx, dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Bounding-box geometry
# ---------------------------------------------------------------------------

def manifest_bbox_available(row: pd.Series) -> bool:
    bbox_columns = [
        "bbox_z_lo",
        "bbox_z_hi",
        "bbox_y_lo",
        "bbox_y_hi",
        "bbox_x_lo",
        "bbox_x_hi",
    ]

    return all(
        c in row.index and np.isfinite(safe_float(row[c]))
        for c in bbox_columns
    )


def get_bbox_volume_coordinates(
    row: pd.Series,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Return bbox lower/upper coordinates in volume ZYX space.

    The manifest stores:

        bbox_z_lo / bbox_z_hi
        bbox_y_lo / bbox_y_hi
        bbox_x_lo / bbox_x_hi
    """
    if not manifest_bbox_available(row):
        return None

    lo = np.array(
        [
            safe_float(row["bbox_z_lo"]),
            safe_float(row["bbox_y_lo"]),
            safe_float(row["bbox_x_lo"]),
        ],
        dtype=np.float64,
    )

    hi = np.array(
        [
            safe_float(row["bbox_z_hi"]),
            safe_float(row["bbox_y_hi"]),
            safe_float(row["bbox_x_hi"]),
        ],
        dtype=np.float64,
    )

    return lo, hi


def bbox_volume_to_patch(
    bbox_volume: Tuple[np.ndarray, np.ndarray],
    patch_origin_zyx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert volume-space bbox into patch-local coordinates."""
    lo_volume, hi_volume = bbox_volume

    lo_patch = volume_to_patch(lo_volume, patch_origin_zyx)
    hi_patch = volume_to_patch(hi_volume, patch_origin_zyx)

    return lo_patch, hi_patch


def bbox_center_patch(
    bbox_patch: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Return bbox center in patch-local ZYX coordinates."""
    lo, hi = bbox_patch
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Slice selection
# ---------------------------------------------------------------------------

def choose_slice_z(
    slice_mode: str,
    patch_size: int,
    candidate_local_zyx: np.ndarray,
    bbox_patch: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[int, str]:
    """
    Select axial Z slice in PATCH coordinates.

    Modes
    -----
    candidate-center
        Slice nearest candidate center.

    bbox-center
        Slice at bbox center when bbox exists.
        Falls back to candidate-center otherwise.

    patch-center
        Geometric center of the 64^3 array.
    """
    if slice_mode == "candidate-center":
        z = int(round(candidate_local_zyx[0]))
        return clamp_slice_index(z, patch_size), "candidate-center"

    if slice_mode == "bbox-center":
        if bbox_patch is not None:
            center = bbox_center_patch(bbox_patch)
            z = int(round(center[0]))
            return clamp_slice_index(z, patch_size), "bbox-center"
        z = int(round(candidate_local_zyx[0]))
        return clamp_slice_index(z, patch_size), "candidate-center-fallback"

    if slice_mode == "patch-center":
        z = patch_size // 2
        return clamp_slice_index(z, patch_size), "patch-center"

    raise ValueError(f"Unknown slice mode: {slice_mode}")


# ---------------------------------------------------------------------------
# Grad-CAM geometry
# ---------------------------------------------------------------------------

def find_heatmap_peak(cam: np.ndarray) -> Tuple[int, int, int]:
    """
    Return peak coordinate as Z,Y,X.

    IMPORTANT:
    np.argmax() preserves the actual array ordering.
    No axis conversion occurs here.
    """
    if cam.ndim != 3:
        raise ValueError(
            f"Expected a 3D Grad-CAM array, got shape {cam.shape}"
        )

    flat_index = int(np.nanargmax(cam))
    return tuple(int(v) for v in np.unravel_index(flat_index, cam.shape))


def calculate_geometry(
    cam: np.ndarray,
    candidate_local_zyx: np.ndarray,
) -> Dict[str, float]:
    """
    Compare CAM peak against the ACTUAL candidate position in patch space.

    This is the key correction over the previous diagnostic.

    The old diagnostic effectively compared:

        patch peak  <->  volume coordinate

    which produces meaningless distances around 200-300 voxels.

    This diagnostic compares:

        patch peak  <->  candidate_local_zyx
    """
    peak = np.asarray(find_heatmap_peak(cam), dtype=np.float64)

    delta = peak - candidate_local_zyx

    euclidean = float(np.linalg.norm(delta))

    return {
        "peak_z": float(peak[0]),
        "peak_y": float(peak[1]),
        "peak_x": float(peak[2]),
        "delta_z": float(delta[0]),
        "delta_y": float(delta[1]),
        "delta_x": float(delta[2]),
        "peak_distance_voxels": euclidean,
    }


def geometry_status(distance: float) -> str:
    """
    Diagnostic-only classification.

    This does NOT reject candidates or alter model output.
    """
    if not np.isfinite(distance):
        return "invalid"

    if distance <= CAM_NEAR_CENTER_VOXELS:
        return "near_candidate"

    if distance <= CAM_ACCEPTABLE_CENTER_VOXELS:
        return "moderately_offset"

    return "far_from_candidate"


# ---------------------------------------------------------------------------
# CT display
# ---------------------------------------------------------------------------

def normalize_ct_for_display(slice_hu: np.ndarray) -> np.ndarray:
    """
    Window HU values for display.

    No normalization is applied to the underlying classifier input.
    """
    clipped = np.clip(
        slice_hu,
        CT_WINDOW_MIN,
        CT_WINDOW_MAX,
    )

    return (
        (clipped - CT_WINDOW_MIN)
        / (CT_WINDOW_MAX - CT_WINDOW_MIN)
    )


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def run_candidate_inference(
    patch: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    target_layer: str,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """
    Run one candidate through the canonical classifier.

    The heatmap generator performs one independent backward pass per head,
    producing a separate CAM for each candidate/head.

    No CAMs are accumulated across candidates.
    """
    if patch.ndim != 3:
        raise ValueError(
            f"Patch must be 3D, got {patch.shape}"
        )

    if patch.shape != (
        PATCH_SIZE,
        PATCH_SIZE,
        PATCH_SIZE,
    ):
        raise ValueError(
            f"Expected patch shape "
            f"({PATCH_SIZE},{PATCH_SIZE},{PATCH_SIZE}), "
            f"got {patch.shape}"
        )

    patch_tensor = torch.from_numpy(
        patch.astype(np.float32, copy=False)
    ).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(patch_tensor)

    probs: Dict[str, float] = {}

    for head in FEATURE_NAMES:
        if head not in outputs:
            raise KeyError(
                f"Model output does not contain head '{head}'. "
                f"Available heads: {list(outputs.keys())}"
            )

        score = outputs[head]

        if score.dim() > 1 and score.size(1) == 1:
            score = score.squeeze(1)

        probs[head] = float(
            torch.sigmoid(score).detach().cpu().item()
        )

    heatmaps, gradient_diagnostics = generate_characteristic_heatmaps(
        model,
        patch_tensor,
        device=device,
        target_layer=target_layer,
        return_diagnostics=True,
    )

    return probs, heatmaps, gradient_diagnostics


# ---------------------------------------------------------------------------
# Diagnostic figure
# ---------------------------------------------------------------------------

def draw_candidate_marker(
    ax,
    candidate_local_zyx: np.ndarray,
):
    """
    Draw candidate position on an axial Y/X image.

    candidate_local_zyx = Z,Y,X.
    """
    ax.plot(
        candidate_local_zyx[2],
        candidate_local_zyx[1],
        marker="+",
        markersize=12,
        markeredgewidth=2.0,
        linestyle="None",
        color="cyan",
    )


def draw_bbox(
    ax,
    bbox_patch: Optional[Tuple[np.ndarray, np.ndarray]],
    slice_z: int,
):
    """
    Draw the candidate bbox on an axial Y/X slice if the selected slice
    intersects the bbox.
    """
    if bbox_patch is None:
        return

    lo, hi = bbox_patch

    if not (lo[0] <= slice_z <= hi[0]):
        return

    x0 = lo[2]
    x1 = hi[2]
    y0 = lo[1]
    y1 = hi[1]

    width = x1 - x0
    height = y1 - y0

    if width <= 0 or height <= 0:
        return

    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x0, y0),
        width,
        height,
        fill=False,
        edgecolor="yellow",
        linewidth=1.5,
        linestyle="--",
    )

    ax.add_patch(rect)


def plot_head_panel(
    ax_ct,
    ax_cam,
    ax_overlay,
    patch: np.ndarray,
    cam: np.ndarray,
    slice_z: int,
    candidate_local_zyx: np.ndarray,
    bbox_patch: Optional[Tuple[np.ndarray, np.ndarray]],
    head: str,
    score: float,
    geometry: Dict[str, float],
):
    """
    Plot one characteristic's three-panel diagnostic:

        CT
        CAM
        CT + CAM

    Every panel uses identical Y/X geometry.
    """
    ct_slice = patch[slice_z, :, :]
    cam_slice = cam[slice_z, :, :]

    ct_display = normalize_ct_for_display(ct_slice)

    # ------------------------------------------------------------------
    # CT
    # ------------------------------------------------------------------

    ax_ct.imshow(
        ct_display,
        cmap="gray",
        origin="upper",
        interpolation="nearest",
        aspect="equal",
    )

    draw_candidate_marker(
        ax_ct,
        candidate_local_zyx,
    )

    draw_bbox(
        ax_ct,
        bbox_patch,
        slice_z,
    )

    ax_ct.set_title(
        f"{head}\nCT | score={score:.3f}",
        fontsize=9,
    )

    ax_ct.set_xlabel("X")
    ax_ct.set_ylabel("Y")

    # ------------------------------------------------------------------
    # CAM
    # ------------------------------------------------------------------

    ax_cam.imshow(
        cam_slice,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
    )

    draw_candidate_marker(
        ax_cam,
        candidate_local_zyx,
    )

    draw_bbox(
        ax_cam,
        bbox_patch,
        slice_z,
    )

    ax_cam.set_title(
        "Grad-CAM++",
        fontsize=9,
    )

    ax_cam.set_xlabel("X")
    ax_cam.set_ylabel("Y")

    # ------------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------------

    ax_overlay.imshow(
        ct_display,
        cmap="gray",
        origin="upper",
        interpolation="nearest",
        aspect="equal",
    )

    # Suppress tiny CAM values so the overlay does not wash out the CT.
    overlay = np.ma.masked_where(
        cam_slice <= 0.05,
        cam_slice,
    )

    ax_overlay.imshow(
        overlay,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        alpha=CAM_ALPHA,
    )

    draw_candidate_marker(
        ax_overlay,
        candidate_local_zyx,
    )

    draw_bbox(
        ax_overlay,
        bbox_patch,
        slice_z,
    )

    status = geometry_status(
        geometry["peak_distance_voxels"]
    )

    ax_overlay.set_title(
        f"Overlay\n"
        f"peak Δ={geometry['peak_distance_voxels']:.1f} vox "
        f"[{status}]",
        fontsize=9,
    )

    ax_overlay.set_xlabel("X")
    ax_overlay.set_ylabel("Y")


def make_candidate_figure(
    candidate_id: str,
    patch: np.ndarray,
    probs: Dict[str, float],
    heatmaps: Dict[str, np.ndarray],
    candidate_local_zyx: np.ndarray,
    bbox_patch: Optional[Tuple[np.ndarray, np.ndarray]],
    slice_z: int,
    slice_mode_used: str,
    output_path: str,
):
    """
    Generate a complete per-candidate diagnostic figure.

    8 heads x 3 panels = 24 panels.
    """
    heads = [
        h for h in FEATURE_NAMES
        if h in heatmaps
    ]

    if not heads:
        raise RuntimeError(
            f"No heatmaps available for candidate {candidate_id}"
        )

    n_heads = len(heads)

    fig, axes = plt.subplots(
        n_heads,
        3,
        figsize=(12, 3.8 * n_heads),
        squeeze=False,
    )

    fig.suptitle(
        f"Candidate {candidate_id} -- Geometry-Safe Grad-CAM++\n"
        f"Axial Z={slice_z} | slice mode={slice_mode_used} | "
        f"candidate local ZYX="
        f"({candidate_local_zyx[0]:.2f}, "
        f"{candidate_local_zyx[1]:.2f}, "
        f"{candidate_local_zyx[2]:.2f})",
        fontsize=14,
    )

    for row_index, head in enumerate(heads):
        cam = np.asarray(
            heatmaps[head],
            dtype=np.float32,
        )

        if cam.ndim != 3:
            raise ValueError(
                f"Heatmap for {candidate_id}/{head} must be 3D, "
                f"got {cam.shape}"
            )

        if cam.shape != patch.shape:
            raise ValueError(
                f"Geometry mismatch for {candidate_id}/{head}: "
                f"patch={patch.shape}, heatmap={cam.shape}"
            )

        geometry = calculate_geometry(
            cam,
            candidate_local_zyx,
        )

        plot_head_panel(
            axes[row_index, 0],
            axes[row_index, 1],
            axes[row_index, 2],
            patch,
            cam,
            slice_z,
            candidate_local_zyx,
            bbox_patch,
            head,
            probs[head],
            geometry,
        )

    plt.tight_layout(
        rect=(0, 0, 1, 0.985)
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# Geometry diagnostics
# ---------------------------------------------------------------------------

def build_candidate_geometry_rows(
    row: pd.Series,
    patch: np.ndarray,
    heatmaps: Dict[str, np.ndarray],
    candidate_local_zyx: np.ndarray,
    patch_origin_zyx: np.ndarray,
    bbox_patch: Optional[Tuple[np.ndarray, np.ndarray]],
    slice_z: int,
    gradient_diagnostics: Optional[Dict[str, Dict[str, float]]] = None,
) -> list:
    """
    Produce one diagnostics row per head.

    This CSV is intentionally explicit about coordinate systems.
    """
    candidate_id = str(row["candidate_id"])

    rows = []

    for head, cam in heatmaps.items():
        cam = np.asarray(cam)

        geometry = calculate_geometry(
            cam,
            candidate_local_zyx,
        )

        peak_patch = np.array(
            [
                geometry["peak_z"],
                geometry["peak_y"],
                geometry["peak_x"],
            ],
            dtype=np.float64,
        )

        peak_volume = patch_to_volume(
            peak_patch,
            patch_origin_zyx,
        )

        if bbox_patch is not None:
            bbox_lo_patch, bbox_hi_patch = bbox_patch
            bbox_center = bbox_center_patch(bbox_patch)

            bbox_center_distance = float(
                np.linalg.norm(
                    peak_patch - bbox_center
                )
            )

            bbox_z_lo = float(bbox_lo_patch[0])
            bbox_z_hi = float(bbox_hi_patch[0])
            bbox_y_lo = float(bbox_lo_patch[1])
            bbox_y_hi = float(bbox_hi_patch[1])
            bbox_x_lo = float(bbox_lo_patch[2])
            bbox_x_hi = float(bbox_hi_patch[2])
        else:
            bbox_center_distance = np.nan
            bbox_z_lo = np.nan
            bbox_z_hi = np.nan
            bbox_y_lo = np.nan
            bbox_y_hi = np.nan
            bbox_x_lo = np.nan
            bbox_x_hi = np.nan

        status = geometry_status(
            geometry["peak_distance_voxels"]
        )

        grad_diag = (gradient_diagnostics or {}).get(head, {})
        gradient_reliable = grad_diag.get("reliable", None)
        active_cell_fraction = grad_diag.get("active_cell_fraction", np.nan)

        if gradient_reliable is False:
            # Gradient is too spatially concentrated to trust the CAM
            # argmax -- this is a different failure mode from genuine
            # off-center model attention, and must not be conflated with
            # "far_from_candidate" in downstream review.
            status = "gradient_unreliable"

        rows.append(
            {
                "candidate_id": candidate_id,
                "head": head,
                "gradient_reliable": gradient_reliable,
                "gradient_active_cell_fraction": active_cell_fraction,

                # Classifier score.
                "score": np.nan,

                # Original volume-space candidate.
                "candidate_volume_z": safe_float(row["center_z"]),
                "candidate_volume_y": safe_float(row["center_y"]),
                "candidate_volume_x": safe_float(row["center_x"]),

                # Exact Stage 05 rounded center.
                "stage05_center_volume_z": float(
                    np.round(safe_float(row["center_z"]))
                ),
                "stage05_center_volume_y": float(
                    np.round(safe_float(row["center_y"]))
                ),
                "stage05_center_volume_x": float(
                    np.round(safe_float(row["center_x"]))
                ),

                # Patch origin.
                "patch_origin_volume_z": float(
                    patch_origin_zyx[0]
                ),
                "patch_origin_volume_y": float(
                    patch_origin_zyx[1]
                ),
                "patch_origin_volume_x": float(
                    patch_origin_zyx[2]
                ),

                # Actual candidate position inside 64^3 patch.
                "candidate_patch_z": float(
                    candidate_local_zyx[0]
                ),
                "candidate_patch_y": float(
                    candidate_local_zyx[1]
                ),
                "candidate_patch_x": float(
                    candidate_local_zyx[2]
                ),

                # Geometric array center.
                "patch_geometric_center_z": (
                    PATCH_SIZE - 1
                ) / 2.0,
                "patch_geometric_center_y": (
                    PATCH_SIZE - 1
                ) / 2.0,
                "patch_geometric_center_x": (
                    PATCH_SIZE - 1
                ) / 2.0,

                # CAM peak in patch coordinates.
                "cam_peak_patch_z": geometry["peak_z"],
                "cam_peak_patch_y": geometry["peak_y"],
                "cam_peak_patch_x": geometry["peak_x"],

                # CAM peak converted back to volume coordinates.
                "cam_peak_volume_z": float(
                    peak_volume[0]
                ),
                "cam_peak_volume_y": float(
                    peak_volume[1]
                ),
                "cam_peak_volume_x": float(
                    peak_volume[2]
                ),

                # Peak - candidate.
                "delta_z": geometry["delta_z"],
                "delta_y": geometry["delta_y"],
                "delta_x": geometry["delta_x"],
                "peak_distance_voxels": (
                    geometry["peak_distance_voxels"]
                ),

                # Optional bbox in patch coordinates.
                "bbox_patch_z_lo": bbox_z_lo,
                "bbox_patch_z_hi": bbox_z_hi,
                "bbox_patch_y_lo": bbox_y_lo,
                "bbox_patch_y_hi": bbox_y_hi,
                "bbox_patch_x_lo": bbox_x_lo,
                "bbox_patch_x_hi": bbox_x_hi,

                "peak_distance_to_bbox_center": (
                    bbox_center_distance
                ),

                "slice_z": int(slice_z),

                "geometry_status": status,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def make_malignancy_summary(
    summary_rows: list,
    output_path: str,
):
    """
    Generate one compact candidate malignancy summary.

    This is intentionally independent of Grad-CAM geometry.
    """
    if not summary_rows:
        return

    df = pd.DataFrame(summary_rows)

    if "malignancy" not in df.columns:
        return

    df = df.sort_values(
        "malignancy",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, max(4, 0.55 * len(df)))
    )

    ax.barh(
        df["candidate_id"].astype(str),
        df["malignancy"].astype(float),
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Malignancy probability")
    ax.set_ylabel("Candidate")
    ax.set_title(
        "Stage 07 -- Candidate malignancy scores"
    )

    for index, value in enumerate(
        df["malignancy"].astype(float)
    ):
        ax.text(
            min(value + 0.015, 0.98),
            index,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main candidate processing
# ---------------------------------------------------------------------------

def process_candidate(
    row: pd.Series,
    model: torch.nn.Module,
    device: torch.device,
    target_layer: str,
    slice_mode: str,
    figures_dir: str,
):
    candidate_id = str(row["candidate_id"])
    patch_path = str(row["patch_path"])

    if not os.path.isfile(patch_path):
        raise FileNotFoundError(
            f"{candidate_id}: patch file does not exist:\n"
            f"{patch_path}"
        )

    patch = np.load(patch_path)

    if patch.ndim != 3:
        raise ValueError(
            f"{candidate_id}: patch must be 3D, "
            f"got {patch.shape}"
        )

    if patch.shape != (
        PATCH_SIZE,
        PATCH_SIZE,
        PATCH_SIZE,
    ):
        raise ValueError(
            f"{candidate_id}: expected "
            f"{PATCH_SIZE}^3 patch, got {patch.shape}"
        )

    # ---------------------------------------------------------------
    # Original candidate coordinate in Stage 05 volume.
    # ---------------------------------------------------------------

    center_volume = np.array(
        [
            safe_float(row["center_z"]),
            safe_float(row["center_y"]),
            safe_float(row["center_x"]),
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(center_volume)):
        raise ValueError(
            f"{candidate_id}: invalid candidate center "
            f"{center_volume}"
        )

    # ---------------------------------------------------------------
    # Reconstruct exact Stage 05 crop geometry.
    # ---------------------------------------------------------------

    stage05_center, patch_origin = (
        compute_stage05_patch_origin(
            center_volume,
            PATCH_SIZE,
        )
    )

    candidate_local = volume_to_patch(
        center_volume,
        patch_origin,
    )

    # ---------------------------------------------------------------
    # Convert bbox to patch coordinates.
    # ---------------------------------------------------------------

    bbox_volume = get_bbox_volume_coordinates(row)

    if bbox_volume is not None:
        bbox_patch = bbox_volume_to_patch(
            bbox_volume,
            patch_origin,
        )
    else:
        bbox_patch = None

    # ---------------------------------------------------------------
    # Sanity check candidate location.
    # ---------------------------------------------------------------

    if np.any(candidate_local < -EPS) or np.any(
        candidate_local > (PATCH_SIZE - 1) + EPS
    ):
        warnings.warn(
            f"{candidate_id}: candidate local coordinate "
            f"{candidate_local} lies outside the patch. "
            f"Check Stage 05 crop geometry."
        )

    # ---------------------------------------------------------------
    # Run canonical model + per-head Grad-CAM++.
    # ---------------------------------------------------------------

    probs, heatmaps, gradient_diagnostics = run_candidate_inference(
        patch,
        model,
        device,
        target_layer,
    )

    for head_name, diag in gradient_diagnostics.items():
        if not diag["reliable"]:
            print(
                f"[warn] {candidate_id}/{head_name}: Grad-CAM gradient is "
                f"concentrated in only {diag['active_cell_fraction'] * 100:.1f}% "
                "of the native activation grid. Its CAM peak/geometry_status "
                "below is NOT reliable evidence of model attention -- treat "
                "it as 'gradient_unreliable', not as a real off-center finding."
            )

    # ---------------------------------------------------------------
    # Validate heatmap geometry.
    # ---------------------------------------------------------------

    for head, cam in heatmaps.items():
        cam = np.asarray(cam)

        if cam.shape != patch.shape:
            raise RuntimeError(
                f"{candidate_id}/{head}: CAM shape {cam.shape} "
                f"does not match patch shape {patch.shape}. "
                f"Stage 07 refuses to overlay mismatched geometry."
            )

    # ---------------------------------------------------------------
    # Choose axial visualization slice.
    # ---------------------------------------------------------------

    slice_z, slice_mode_used = choose_slice_z(
        slice_mode,
        PATCH_SIZE,
        candidate_local,
        bbox_patch,
    )

    # ---------------------------------------------------------------
    # Print diagnostic information.
    # ---------------------------------------------------------------

    print(
        f"Candidate: {candidate_id}"
    )

    print(
        f"Patch    : {patch_path}"
    )

    print(
        "Candidate volume ZYX: "
        f"({center_volume[0]:.3f}, "
        f"{center_volume[1]:.3f}, "
        f"{center_volume[2]:.3f})"
    )

    print(
        "Stage 05 rounded ZYX: "
        f"({stage05_center[0]}, "
        f"{stage05_center[1]}, "
        f"{stage05_center[2]})"
    )

    print(
        "Patch origin volume ZYX: "
        f"({patch_origin[0]}, "
        f"{patch_origin[1]}, "
        f"{patch_origin[2]})"
    )

    print(
        "Candidate patch ZYX: "
        f"({candidate_local[0]:.3f}, "
        f"{candidate_local[1]:.3f}, "
        f"{candidate_local[2]:.3f})"
    )

    if bbox_patch is not None:
        bbox_center = bbox_center_patch(
            bbox_patch
        )

        print(
            "BBox patch ZYX: "
            f"lo=({bbox_patch[0][0]:.3f}, "
            f"{bbox_patch[0][1]:.3f}, "
            f"{bbox_patch[0][2]:.3f}) "
            f"hi=({bbox_patch[1][0]:.3f}, "
            f"{bbox_patch[1][1]:.3f}, "
            f"{bbox_patch[1][2]:.3f})"
        )

        print(
            "BBox center patch ZYX: "
            f"({bbox_center[0]:.3f}, "
            f"{bbox_center[1]:.3f}, "
            f"{bbox_center[2]:.3f})"
        )
    else:
        print("Candidate bbox: unavailable")

    print(
        f"Slice Z     : {slice_z} "
        f"({slice_mode_used})"
    )

    # ---------------------------------------------------------------
    # Geometry diagnostics.
    # ---------------------------------------------------------------

    diagnostic_rows = build_candidate_geometry_rows(
        row,
        patch,
        heatmaps,
        candidate_local,
        patch_origin,
        bbox_patch,
        slice_z,
        gradient_diagnostics=gradient_diagnostics,
    )

    for diag in diagnostic_rows:
        head = diag["head"]

        diag["score"] = probs.get(
            head,
            np.nan,
        )

        print(
            f"  {head:<14} "
            f"score={probs[head]:.3f} "
            f"peak=("
            f"{diag['cam_peak_patch_z']:.0f},"
            f"{diag['cam_peak_patch_y']:.0f},"
            f"{diag['cam_peak_patch_x']:.0f}) "
            f"Δ={diag['peak_distance_voxels']:.1f} "
            f"[{diag['geometry_status']}]"
        )

    # ---------------------------------------------------------------
    # Figure.
    # ---------------------------------------------------------------

    figure_path = os.path.join(
        figures_dir,
        f"{candidate_id}_gradcam_diagnostic.png",
    )

    make_candidate_figure(
        candidate_id,
        patch,
        probs,
        heatmaps,
        candidate_local,
        bbox_patch,
        slice_z,
        slice_mode_used,
        figure_path,
    )

    print(
        f"Figure   -> {figure_path}"
    )

    return probs, diagnostic_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "STAGE 07: Geometry-safe per-candidate Grad-CAM++ "
            "diagnostic visualization."
        )
    )

    parser.add_argument(
        "--patch-manifest",
        required=True,
        help=(
            "Stage 05 patch_manifest.csv containing patch_path and "
            "candidate volume coordinates."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best_model_gpu_v2.pth.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for Stage 07 diagnostic outputs.",
    )

    parser.add_argument(
        "--target-layer",
        default="backbone.layer2",
        help=(
            "Grad-CAM++ target layer. Recommended for 64^3 patches: "
            "backbone.layer2 (8^3 activation map). "
            "backbone.layer3 is coarser."
        ),
    )

    parser.add_argument(
        "--slice-mode",
        choices=[
            "candidate-center",
            "bbox-center",
            "patch-center",
        ],
        default="bbox-center",
        help=(
            "Axial slice selection. bbox-center uses the Stage 05 "
            "candidate bbox converted into patch coordinates."
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu"],
        help="Stage 07 currently runs on CPU.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of candidates to process.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print()
    print("=" * 78)
    print("STAGE 07 -- GRAD-CAM++ GEOMETRY-SAFE DIAGNOSTIC")
    print("=" * 78)

    # ---------------------------------------------------------------
    # Manifest.
    # ---------------------------------------------------------------

    manifest_path = args.patch_manifest

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Patch manifest not found:\n{manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    require_columns(
        manifest,
        [
            "candidate_id",
            "patch_path",
            "center_z",
            "center_y",
            "center_x",
        ],
    )

    if args.limit is not None:
        manifest = manifest.head(
            max(0, args.limit)
        )

    # ---------------------------------------------------------------
    # Output directories.
    # ---------------------------------------------------------------

    output_dir = os.path.abspath(
        args.output_dir
    )

    figures_dir = os.path.join(
        output_dir,
        "figures",
    )

    os.makedirs(
        figures_dir,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Device.
    # ---------------------------------------------------------------

    device = torch.device(
        args.device
    )

    # ---------------------------------------------------------------
    # Model.
    # ---------------------------------------------------------------

    print(
        f"Candidates       : {len(manifest)}"
    )

    print(
        f"Target layer     : {args.target_layer}"
    )

    print(
        f"Slice mode       : {args.slice_mode}"
    )

    print(
        f"Patch size       : {PATCH_SIZE}^3"
    )

    print(
        "Patch array center: "
        f"[{(PATCH_SIZE - 1) / 2:.1f} "
        f"{(PATCH_SIZE - 1) / 2:.1f} "
        f"{(PATCH_SIZE - 1) / 2:.1f}]"
    )

    print()

    model = create_multihead_model(
        device=device
    )

    model = load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    model.eval()

    print(
        f"Checkpoint       : {args.checkpoint}"
    )

    print(
        f"Model heads      : {', '.join(FEATURE_NAMES)}"
    )

    print()

    # ---------------------------------------------------------------
    # Process candidates.
    # ---------------------------------------------------------------

    all_geometry_rows = []
    malignancy_rows = []

    for index, (_, row) in enumerate(
        manifest.iterrows()
    ):
        print("-" * 78)

        try:
            probs, diagnostic_rows = process_candidate(
                row=row,
                model=model,
                device=device,
                target_layer=args.target_layer,
                slice_mode=args.slice_mode,
                figures_dir=figures_dir,
            )

            all_geometry_rows.extend(
                diagnostic_rows
            )

            malignancy_rows.append(
                {
                    "candidate_id": str(
                        row["candidate_id"]
                    ),
                    "malignancy": probs.get(
                        "malignancy",
                        np.nan,
                    ),
                    "confidence": safe_float(
                        row.get(
                            "confidence",
                            np.nan,
                        )
                    ),
                    "diameter_mm": safe_float(
                        row.get(
                            "diameter_mm",
                            np.nan,
                        )
                    ),
                }
            )

        except Exception as exc:
            print(
                f"[ERROR] Candidate "
                f"{row['candidate_id']} failed: "
                f"{exc}"
            )

            # Do not silently continue with corrupt geometry.
            raise

    # ---------------------------------------------------------------
    # Geometry diagnostics CSV.
    # ---------------------------------------------------------------

    diagnostics_path = os.path.join(
        output_dir,
        "gradcam_geometry_diagnostics.csv",
    )

    diagnostics_df = pd.DataFrame(
        all_geometry_rows
    )

    diagnostics_df.to_csv(
        diagnostics_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Malignancy summary.
    # ---------------------------------------------------------------

    summary_path = os.path.join(
        figures_dir,
        "all_candidates_malignancy_summary.png",
    )

    make_malignancy_summary(
        malignancy_rows,
        summary_path,
    )

    print()
    print(
        f"Geometry diagnostics -> "
        f"{diagnostics_path}"
    )

    print(
        f"Patient summary       -> "
        f"{summary_path}"
    )

    print()
    print("=" * 78)
    print(
        f"DONE -- {len(manifest)} candidate(s) processed."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()