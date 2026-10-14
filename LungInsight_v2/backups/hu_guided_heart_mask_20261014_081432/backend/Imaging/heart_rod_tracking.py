"""Expanding Z-rod tracker for central cardiac/mediastinal exclusion.

The rod is centered on the CT's YX field and expands in physical millimetres.
Instead of requiring one HU-thresholded 3D connected component to touch every
qualifying rod slice (an unstable assumption in real CT), the tracker evaluates
central tissue support at increasing radii, selects a persistent contiguous Z
run, and forms a conservative central *non-lung* soft-tissue exclusion mask in
an expanded cylinder. The mask is a heuristic mediastinal exclusion region,
not a clinical heart segmentation. Always review its QC overlay.

All arrays use the schema's canonical ZYX order. The helper also creates the
optional aligned HU NPY volume with mask voxels replaced by physical air HU.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage


def _validate_spacing(pixel_spacing_mm: tuple[float, float], slice_spacing_mm: float) -> tuple[float, float, float]:
    try:
        py, px = float(pixel_spacing_mm[0]), float(pixel_spacing_mm[1])
        pz = float(slice_spacing_mm)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("Expected pixel_spacing_mm=(y_mm, x_mm) and positive slice_spacing_mm") from exc
    if not all(math.isfinite(v) and v > 0 for v in (pz, py, px)):
        raise ValueError("All voxel spacings must be finite and positive")
    return pz, py, px


def _true_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive start/end pairs for contiguous True indices."""
    values = np.asarray(indices, dtype=bool)
    positions = np.flatnonzero(values)
    if positions.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(positions) > 1)
    starts = np.r_[positions[0], positions[cuts + 1]]
    ends = np.r_[positions[cuts], positions[-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def build_heart_excluded_volume(
    volume_hu: np.ndarray,
    heart_mask: np.ndarray,
    *,
    replacement_hu: float = -1000.0,
) -> np.ndarray:
    """Return an aligned float32 HU volume with heart-mask voxels blanked.

    The source array is not mutated. A copy is always returned, even when the
    mask is empty, so output shape/dtype are deterministic for pipeline use.
    """
    volume = np.asarray(volume_hu)
    mask = np.asarray(heart_mask, dtype=bool)
    if volume.ndim != 3:
        raise ValueError(f"volume_hu must be a 3D ZYX array, got {volume.shape}")
    if mask.shape != volume.shape:
        raise ValueError(f"heart_mask shape {mask.shape} does not match CT shape {volume.shape}")
    if not math.isfinite(float(replacement_hu)):
        raise ValueError("replacement_hu must be finite")
    out = np.asarray(volume, dtype=np.float32).copy()
    out[mask] = np.float32(replacement_hu)
    return out


def compute_heart_mask_rod(
    volume_hu: np.ndarray,
    lung_mask: np.ndarray,
    *,
    pixel_spacing_mm: tuple[float, float] = (1.0, 1.0),
    slice_spacing_mm: float = 1.0,
    min_span_mm: float = 25.0,
    seed_radius_mm: float = 2.0,
    radius_step_mm: float = 4.0,
    max_radius_mm: float = 85.0,
    minimum_mask_radius_mm: float = 65.0,
    mask_growth_mm: float = 30.0,
    soft_tissue_hu_min: float = -350.0,
    soft_tissue_hu_max: float = 500.0,
    exclusion_mask_hu_min: float = -950.0,
    min_lung_slice_fraction: float = 0.08,
    min_support_voxels_per_slice: int = 8,
    axial_margin_mm: float = 10.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Track persistent central tissue with a growing cylinder parallel to Z.

    The rod is tested at radii ``seed_radius_mm, seed_radius_mm + n*step``.
    At each radius the helper measures central soft-tissue support per axial
    slice, chooses the longest qualifying contiguous Z run, and stops at the
    first radius producing at least ``min_span_mm`` of support. The final mask
    expands that rod footprint by ``mask_growth_mm`` (with a ``minimum_mask_radius_mm``
    floor) and includes soft tissue outside the lung mask inside the supported
    thoracic range. The outside-lung restriction is intentional: the mask is an
    exclusion region and must not erase normal lung parenchyma.

    If no qualifying run is found, the function returns an all-False mask with
    an explicit failure reason; callers must keep downstream gating closed.
    """
    hu = np.asarray(volume_hu)
    lung = np.asarray(lung_mask, dtype=bool)
    if hu.ndim != 3:
        raise ValueError(f"volume_hu must be a 3D ZYX array, got shape={hu.shape}")
    if lung.shape != hu.shape:
        raise ValueError(f"lung_mask shape {lung.shape} does not match CT shape {hu.shape}")
    pz, py, px = _validate_spacing(pixel_spacing_mm, slice_spacing_mm)
    numeric = {
        "min_span_mm": min_span_mm,
        "seed_radius_mm": seed_radius_mm,
        "radius_step_mm": radius_step_mm,
        "max_radius_mm": max_radius_mm,
        "minimum_mask_radius_mm": minimum_mask_radius_mm,
        "mask_growth_mm": mask_growth_mm,
        "soft_tissue_hu_min": soft_tissue_hu_min,
        "soft_tissue_hu_max": soft_tissue_hu_max,
        "exclusion_mask_hu_min": exclusion_mask_hu_min,
        "min_lung_slice_fraction": min_lung_slice_fraction,
        "axial_margin_mm": axial_margin_mm,
    }
    for name, value in numeric.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if min_span_mm <= 0 or seed_radius_mm < 0 or radius_step_mm <= 0 or max_radius_mm < seed_radius_mm:
        raise ValueError("Invalid rod radii or minimum span")
    if minimum_mask_radius_mm < 0 or mask_growth_mm < 0:
        raise ValueError("Mask growth/radius must be non-negative")
    if soft_tissue_hu_min >= soft_tissue_hu_max:
        raise ValueError("soft_tissue_hu_min must be less than soft_tissue_hu_max")
    if exclusion_mask_hu_min >= soft_tissue_hu_max:
        raise ValueError("exclusion_mask_hu_min must be less than soft_tissue_hu_max")
    if not 0.0 < min_lung_slice_fraction <= 1.0 or axial_margin_mm < 0:
        raise ValueError("min_lung_slice_fraction must be in (0, 1] and axial_margin_mm non-negative")
    if type(min_support_voxels_per_slice) is not int or min_support_voxels_per_slice < 1:
        raise ValueError("min_support_voxels_per_slice must be a positive integer")

    empty = np.zeros(hu.shape, dtype=bool)
    nz, ny, nx = hu.shape
    cy, cx = (ny - 1.0) / 2.0, (nx - 1.0) / 2.0
    lung_fraction = lung.mean(axis=(1, 2), dtype=np.float64)
    thoracic = np.flatnonzero(lung_fraction >= float(min_lung_slice_fraction))
    diag: dict[str, Any] = {
        "method": "expanding_central_rod_3d_persistent_support",
        "status": "not_found",
        "selection_mode": "growing_cylindrical_rod_with_Z_persistence",
        "center_yx_voxels": [float(cy), float(cx)],
        "spacing_zyx_mm": [float(pz), float(py), float(px)],
        "soft_tissue_hu_range": [float(soft_tissue_hu_min), float(soft_tissue_hu_max)],
        "seed_radius_mm": float(seed_radius_mm),
        "radius_step_mm": float(radius_step_mm),
        "max_radius_mm": float(max_radius_mm),
        "minimum_mask_radius_mm": float(minimum_mask_radius_mm),
        "mask_growth_mm": float(mask_growth_mm),
        "min_span_mm": float(min_span_mm),
        "min_support_voxels_per_slice": int(min_support_voxels_per_slice),
        "min_lung_slice_fraction": float(min_lung_slice_fraction),
        "axial_margin_mm": float(axial_margin_mm),
        "lung_slice_fraction_min": float(lung_fraction.min()),
        "lung_slice_fraction_max": float(lung_fraction.max()),
        "thoracic_slice_count": int(thoracic.size),
        "selected_radius_mm": None,
        "expanded_mask_radius_mm": None,
        "selected_component_label": None,
        "selected_component_voxels": 0,
        "selected_z_range": None,
        "selected_component_span_mm": 0.0,
        "rod_touch_z_range": None,
        "rod_touch_span_mm": 0.0,
        "rod_touch_slice_count": 0,
        "candidate_tissue_voxels": 0,
        "output_mask_voxels": 0,
        "output_mask_lung_overlap_voxels": 0,
        "failure_reason": None,
    }
    if thoracic.size == 0:
        diag["failure_reason"] = "no_thoracic_slices_from_lung_mask"
        return empty, diag

    z_margin = int(math.ceil(float(axial_margin_mm) / pz))
    z_start = max(0, int(thoracic[0]) - z_margin)
    z_end = min(nz - 1, int(thoracic[-1]) + z_margin)
    cy_roi, cx_roi = cy, cx
    y0 = max(0, int(math.floor(cy_roi - max_radius_mm / py)))
    y1 = min(ny, int(math.ceil(cy_roi + max_radius_mm / py)) + 1)
    x0 = max(0, int(math.floor(cx_roi - max_radius_mm / px)))
    x1 = min(nx, int(math.ceil(cx_roi + max_radius_mm / px)) + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    radial_mm = np.sqrt(((yy - cy) * py) ** 2 + ((xx - cx) * px) ** 2)
    max_rod = radial_mm <= float(max_radius_mm)

    hu_roi = hu[z_start:z_end + 1, y0:y1, x0:x1]
    lung_roi = lung[z_start:z_end + 1, y0:y1, x0:x1]
    tissue = np.isfinite(hu_roi) & (hu_roi >= soft_tissue_hu_min) & (hu_roi <= soft_tissue_hu_max)
    tissue &= max_rod[None, :, :]
    support_by_radius: list[dict[str, Any]] = []
    radii: list[float] = []
    radius = float(seed_radius_mm)
    while radius < float(max_radius_mm):
        radii.append(radius)
        radius += float(radius_step_mm)
    if not radii or radii[-1] < float(max_radius_mm):
        radii.append(float(max_radius_mm))

    required_slices = max(2, int(math.ceil(float(min_span_mm) / pz)) + 1)
    selected: tuple[float, int, int, np.ndarray] | None = None
    for radius in radii:
        rod_xy = radial_mm <= radius
        counts = np.count_nonzero(tissue & rod_xy[None, :, :], axis=(1, 2))
        support = counts >= int(min_support_voxels_per_slice)
        runs = _true_runs(support)
        qualifying = [r for r in runs if (r[1] - r[0] + 1) >= required_slices]
        if qualifying:
            qualifying.sort(key=lambda r: (r[1] - r[0] + 1, int(counts[r[0]:r[1] + 1].sum()), -r[0]), reverse=True)
            run_start, run_end = qualifying[0]
            selected = (float(radius), int(run_start), int(run_end), counts)
            support_by_radius.append({"radius_mm": float(radius), "supported_slices": int(support.sum()), "longest_run_slices": int(max((b-a+1 for a,b in runs), default=0)), "selected": True})
            break
        support_by_radius.append({"radius_mm": float(radius), "supported_slices": int(support.sum()), "longest_run_slices": int(max((b-a+1 for a,b in runs), default=0)), "selected": False})

    diag["radius_search"] = support_by_radius
    diag["thoracic_z_range"] = [int(z_start), int(z_end)]
    diag["search_roi_yx"] = [[int(y0), int(y1)], [int(x0), int(x1)]]
    diag["candidate_tissue_voxels"] = int(tissue.sum())
    diag["required_rod_touch_slices"] = int(required_slices)
    if selected is None:
        diag["failure_reason"] = "no_persistent_central_tissue_track_within_max_radius"
        return empty, diag

    selected_radius, run_start, run_end, selected_counts = selected
    global_z_start = z_start + run_start
    global_z_end = z_start + run_end
    # Expand in Z beyond the support interval to cover the connected cardiac
    # structure at the top/base of its axial extent, bounded by the thoracic ROI.
    z_growth = int(math.ceil(float(axial_margin_mm) / pz))
    out_z_start = max(z_start, global_z_start - z_growth)
    out_z_end = min(z_end, global_z_end + z_growth)
    expanded_radius = min(float(max_radius_mm), max(float(minimum_mask_radius_mm), selected_radius + float(mask_growth_mm)))
    expanded_xy = radial_mm <= expanded_radius
    local_range = slice(out_z_start - z_start, out_z_end - z_start + 1)

    # The exclusion mask follows the expanding rod through central thoracic
    # soft tissue, but cannot swallow true lung-mask voxels. The subsequent
    # large-hole association in Stage 01 can fill an enclosed mediastinal
    # cavity when it touches this tracked central component.
    non_lung = ~lung_roi
    # The expanding rod selects the axial interval; the expanded cylinder then
    # includes non-air voxels outside the lung mask, not just HU-thresholded
    # myocardium. This conservatively covers blood pools, fat and connected
    # central mediastinal tissue rather than leaving holes inside the exclusion
    # footprint. Lung voxels are excluded explicitly before and after closing.
    non_air_tissue = np.isfinite(hu_roi) & (hu_roi > float(exclusion_mask_hu_min))
    raw_mask = non_air_tissue & non_lung & expanded_xy[None, :, :]
    z_allowed = np.zeros((hu_roi.shape[0],), dtype=bool)
    z_allowed[local_range] = True
    raw_mask &= z_allowed[:, None, None]

    # Light 3D closing only (not 2D hole-based detection) smooths small
    # discontinuities across adjacent slices. Reapply every spatial constraint
    # afterward so the operation cannot grow into lung or outside the rod.
    structure = np.ones((3, 3, 3), dtype=bool)
    closed = ndimage.binary_closing(raw_mask, structure=structure)
    mask_roi = (raw_mask | closed) & non_lung & expanded_xy[None, :, :] & z_allowed[:, None, None]
    heart_mask = empty.copy()
    heart_mask[z_start:z_end + 1, y0:y1, x0:x1] = mask_roi

    touch_global = np.arange(global_z_start, global_z_end + 1, dtype=np.int64)
    selected_span = float((global_z_end - global_z_start) * pz)
    diag.update({
        "status": "tracked_central_3d_component" if heart_mask.any() else "not_found",
        "selected_radius_mm": float(selected_radius),
        "expanded_mask_radius_mm": float(expanded_radius),
        "selected_component_label": "rod_support_run",
        "selected_component_voxels": int(heart_mask.sum()),
        "selected_z_range": [int(global_z_start), int(global_z_end)],
        "selected_component_span_mm": selected_span,
        "rod_touch_z_range": [int(global_z_start), int(global_z_end)],
        "rod_touch_span_mm": selected_span,
        "rod_touch_slice_count": int(touch_global.size),
        "output_mask_voxels": int(heart_mask.sum()),
        "output_mask_lung_overlap_voxels": int(np.count_nonzero(heart_mask & lung)),
        "selected_radius_support_voxel_counts": [int(v) for v in selected_counts[run_start:run_end + 1]],
        "mask_description": "expanded central Z-cylinder intersected with non-air voxels outside lung_mask; associated large-hole components are added by Stage 01. Conservative mediastinal exclusion region, not a clinical heart contour.",
        "failure_reason": None if heart_mask.any() else "persistent_track_found_but_no_nonlung_soft_tissue_in_expanded_mask",
    })
    if not heart_mask.any():
        diag["status"] = "not_found"
    return heart_mask, diag


def associate_large_hole_components(
    large_holes_stack: np.ndarray,
    tracked_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Associate whole large-hole components intersecting the rod mask in 3D."""
    holes = np.asarray(large_holes_stack, dtype=bool)
    tracked = np.asarray(tracked_mask, dtype=bool)
    if holes.ndim != 3 or holes.shape != tracked.shape:
        raise ValueError("large_holes_stack and tracked_mask must have the same 3D ZYX shape")
    associated = np.zeros_like(holes, dtype=bool)
    labels, count = ndimage.label(holes, structure=np.ones((3, 3, 3), dtype=bool))
    matched: list[int] = []
    if count:
        touched = np.unique(labels[tracked & holes])
        for label_value in touched.tolist():
            label_id = int(label_value)
            if label_id == 0:
                continue
            associated |= labels == label_id
            matched.append(label_id)
    return associated, {
        "large_hole_component_count": int(count),
        "associated_large_hole_component_labels": matched,
        "associated_large_hole_voxels": int(associated.sum()),
    }
