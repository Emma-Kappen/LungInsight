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


def _regularize_tracked_roi(
    tracked_roi: np.ndarray,
    py: float,
    px: float,
    pz: float,
    closing_mm: float,
    max_span_mm: float,
) -> np.ndarray:
    """Close, fill and keep one blob per slice, cap Z span, keep the largest 3D component."""
    ry = max(1, int(round(closing_mm / py)))
    rx = max(1, int(round(closing_mm / px)))
    yy, xx = np.ogrid[-ry:ry + 1, -rx:rx + 1]
    disk = (yy / ry) ** 2 + (xx / rx) ** 2 <= 1.0
    out = np.zeros_like(tracked_roi, dtype=bool)
    for z in range(tracked_roi.shape[0]):
        if not tracked_roi[z].any():
            continue
        padded = np.pad(tracked_roi[z], ((ry, ry), (rx, rx)))
        closed = ndimage.binary_closing(padded, structure=disk)[ry:-ry, rx:-rx]
        filled = ndimage.binary_fill_holes(closed)
        labels, count = ndimage.label(filled)
        if count > 1:
            sizes = ndimage.sum(filled, labels, index=np.arange(1, count + 1))
            filled = labels == (int(np.argmax(sizes)) + 1)
        out[z] = filled
    zs = np.flatnonzero(out.any(axis=(1, 2)))
    if zs.size and max_span_mm > 0:
        out[int(zs[0]) + int(math.ceil(max_span_mm / pz)) + 1:] = False
    labels, count = ndimage.label(out, structure=np.ones((3, 3, 3), dtype=bool))
    if count > 1:
        sizes = ndimage.sum(out, labels, index=np.arange(1, count + 1))
        out = labels == (int(np.argmax(sizes)) + 1)
    return out


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
    max_radius_mm: float = 75.0,
    minimum_mask_radius_mm: float = 65.0,
    mask_growth_mm: float = 30.0,
    max_contour_growth_mm: float = 5.0,
    max_slice_area_ratio: float = 1.4,
    soft_tissue_hu_min: float = -80.0,
    soft_tissue_hu_max: float = 500.0,
    exclusion_mask_hu_min: float = -950.0,
    min_lung_slice_fraction: float = 0.14,
    min_support_voxels_per_slice: int = 8,
    axial_margin_mm: float = 10.0,
    smoothing_closing_mm: float = 6.0,
    max_heart_span_mm: float = 110.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Track central soft tissue with a growing rod, then follow its shape.

    The Z-parallel expanding rod determines a persistent axial support run,
    but HU is used as an actual tissue discriminator: only voxels in the
    configured soft-tissue HU interval can seed or grow the mask. The default
    lower bound is -80 HU (not -350 HU), excluding most fat and aerated lung.
    Candidate components may overlap the provisional lung mask so that dense
    cardiac voxels mistakenly admitted by lung segmentation can be removed by
    Stage 01's subsequent ``lung_mask &= ~heart_mask`` operation. The tracker
    is restricted to slices with sufficient bilateral thoracic lung area; this
    prevents an inferior liver/upper-abdominal component being selected merely
    because it has similar HU to myocardium. Each propagated slice is further
    constrained by physical contour motion and a maximum area-growth ratio.

    The output is a conservative central cardiac/mediastinal exclusion region,
    not a clinical heart segmentation. If the rod cannot find a persistent
    central support track, the function returns an empty mask and diagnostics;
    downstream gating must remain closed.

    ``minimum_mask_radius_mm`` and ``mask_growth_mm`` are accepted for backward
    call compatibility with the v2.8 helper, but they no longer define a solid
    cylindrical output mask. The maximum contour motion is controlled by
    ``max_contour_growth_mm`` instead.
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
        "max_contour_growth_mm": max_contour_growth_mm,
        "max_slice_area_ratio": max_slice_area_ratio,
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
    if minimum_mask_radius_mm < 0 or mask_growth_mm < 0 or max_contour_growth_mm <= 0:
        raise ValueError("Mask radii/growth must be non-negative and contour growth must be positive")
    if max_slice_area_ratio <= 1.0:
        raise ValueError("max_slice_area_ratio must be greater than 1.0")
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
    thoracic_slice_eligible = lung_fraction >= float(min_lung_slice_fraction)
    thoracic = np.flatnonzero(thoracic_slice_eligible)
    diag: dict[str, Any] = {
        "method": "hu_guided_expanding_central_rod_with_thoracic_slice_gate",
        "status": "not_found",
        "selection_mode": "growing_zyx_center_rod_plus_slice_component_tracking_not_cylinder_fill",
        "center_yx_voxels": [float(cy), float(cx)],
        "spacing_zyx_mm": [float(pz), float(py), float(px)],
        "soft_tissue_hu_range": [float(soft_tissue_hu_min), float(soft_tissue_hu_max)],
        "hu_threshold_is_growth_constraint": True,
        "thoracic_slice_gate_is_growth_constraint": True,
        "allows_seed_mask_overlap_with_provisional_lung": True,
        "seed_radius_mm": float(seed_radius_mm),
        "radius_step_mm": float(radius_step_mm),
        "max_radius_mm": float(max_radius_mm),
        "legacy_minimum_mask_radius_mm_ignored": float(minimum_mask_radius_mm),
        "legacy_cylindrical_mask_growth_mm_ignored": float(mask_growth_mm),
        "max_contour_growth_mm": float(max_contour_growth_mm),
        "max_slice_area_ratio": float(max_slice_area_ratio),
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
        "tracked_slice_count": 0,
        "component_area_by_slice": [],
        "failure_reason": None,
    }
    if thoracic.size == 0:
        diag["failure_reason"] = "no_thoracic_slices_from_lung_mask"
        return empty, diag

    z_margin = int(math.ceil(float(axial_margin_mm) / pz))
    z_start = max(0, int(thoracic[0]) - z_margin)
    z_end = min(nz - 1, int(thoracic[-1]) + z_margin)
    y0 = max(0, int(math.floor(cy - max_radius_mm / py)))
    y1 = min(ny, int(math.ceil(cy + max_radius_mm / py)) + 1)
    x0 = max(0, int(math.floor(cx - max_radius_mm / px)))
    x1 = min(nx, int(math.ceil(cx + max_radius_mm / px)) + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    radial_mm = np.sqrt(((yy - cy) * py) ** 2 + ((xx - cx) * px) ** 2)
    max_rod = radial_mm <= float(max_radius_mm)

    hu_roi = hu[z_start:z_end + 1, y0:y1, x0:x1]
    lung_roi = lung[z_start:z_end + 1, y0:y1, x0:x1]
    support_tissue = np.isfinite(hu_roi) & (hu_roi >= soft_tissue_hu_min) & (hu_roi <= soft_tissue_hu_max)
    support_tissue &= max_rod[None, :, :]
    # Do not let axial padding around the thorax seed a liver, shoulder, or
    # neck component. The same slice gate is applied to both rod support and
    # propagation domain; margins are for search bookkeeping only.
    thoracic_local = thoracic_slice_eligible[z_start:z_end + 1]
    support_tissue &= thoracic_local[:, None, None]
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
        counts = np.count_nonzero(support_tissue & rod_xy[None, :, :], axis=(1, 2))
        support = counts >= int(min_support_voxels_per_slice)
        runs = _true_runs(support)
        qualifying = [r for r in runs if (r[1] - r[0] + 1) >= required_slices]
        if qualifying:
            qualifying.sort(key=lambda r: (r[1] - r[0] + 1, int(counts[r[0]:r[1] + 1].sum()), -r[0]), reverse=True)
            run_start, run_end = qualifying[0]
            selected = (float(radius), int(run_start), int(run_end), counts)
            support_by_radius.append({
                "radius_mm": float(radius),
                "supported_slices": int(support.sum()),
                "longest_run_slices": int(max((b - a + 1 for a, b in runs), default=0)),
                "selected": True,
            })
            break
        support_by_radius.append({
            "radius_mm": float(radius),
            "supported_slices": int(support.sum()),
            "longest_run_slices": int(max((b - a + 1 for a, b in runs), default=0)),
            "selected": False,
        })

    diag["radius_search"] = support_by_radius
    diag["thoracic_z_range"] = [int(z_start), int(z_end)]
    diag["thoracic_eligible_z_range"] = [int(thoracic[0]), int(thoracic[-1])]
    diag["thoracic_eligible_slice_count"] = int(thoracic.size)
    diag["search_roi_yx"] = [[int(y0), int(y1)], [int(x0), int(x1)]]
    diag["candidate_tissue_voxels"] = int(support_tissue.sum())
    diag["required_rod_touch_slices"] = int(required_slices)
    if selected is None:
        diag["failure_reason"] = "no_persistent_central_tissue_track_within_max_radius"
        return empty, diag

    selected_radius, run_start, run_end, selected_counts = selected
    global_z_start = z_start + run_start
    global_z_end = z_start + run_end
    center_local_z = (run_start + run_end) // 2
    rod_core = radial_mm <= selected_radius

    # Use HU to select tissue; do not subtract lung_roi here. That subtraction
    # used to make it mathematically impossible for the heart mask to correct
    # voxels where provisional lung segmentation had leaked into the heart.
    # Thoracic-slice and physical-radius gates above keep this from becoming a
    # whole-body soft-tissue mask.
    component_domain = support_tissue.copy()
    structure_2d = np.ones((3, 3), dtype=bool)
    labels_by_slice: list[np.ndarray] = []
    component_counts_by_slice: list[int] = []
    for z_local in range(component_domain.shape[0]):
        labels, number = ndimage.label(component_domain[z_local], structure=structure_2d)
        labels_by_slice.append(labels.astype(np.int32, copy=False))
        component_counts_by_slice.append(int(number))

    # Select an anchor slice with a connected component actually touching the
    # rod. Prefer the strongest overlap, then proximity to the run midpoint;
    # this tolerates short spans where an imperfect lung mask has swallowed
    # part of the tracked central tissue.
    supported_locals = np.arange(run_start, run_end + 1, dtype=int)
    anchor_options: list[tuple[int, int, int, float, int, int]] = []
    for candidate_z in supported_locals.tolist():
        candidate_labels = labels_by_slice[candidate_z]
        intersection = candidate_labels[rod_core]
        ids, id_counts = np.unique(intersection[intersection > 0], return_counts=True)
        for label_id, overlap_count in zip(ids.tolist(), id_counts.tolist()):
            comp = candidate_labels == int(label_id)
            area = int(comp.sum())
            ys, xs = np.where(comp)
            centroid_r = float(np.mean(np.sqrt(((ys + y0 - cy) * py) ** 2 + ((xs + x0 - cx) * px) ** 2)))
            anchor_options.append((int(overlap_count), -abs(int(candidate_z) - center_local_z), -area,
                                   -centroid_r, int(candidate_z), int(label_id)))
    if not anchor_options:
        diag["failure_reason"] = "rod_support_run_has_no_nonlung_soft_tissue_component_seed"
        return empty, diag
    best_anchor = max(anchor_options)
    anchor_local_z, anchor_label = int(best_anchor[4]), int(best_anchor[5])
    anchor_labels = labels_by_slice[anchor_local_z]
    anchor_mask = anchor_labels == anchor_label
    if not anchor_mask.any():
        diag["failure_reason"] = "empty_seed_component_on_anchor_slice"
        return empty, diag

    # Track the component outwards from the anchor slice. The physical-distance
    # envelope lets the component move/expand naturally, while preventing a
    # single connected liver/body-wall component from flooding the whole ROI.
    z_growth = int(math.ceil(float(axial_margin_mm) / pz))
    track_low = max(z_start, global_z_start - z_growth)
    track_high = min(z_end, global_z_end + z_growth)
    tracked_roi = np.zeros(component_domain.shape, dtype=bool)
    tracked_roi[anchor_local_z] = anchor_mask
    per_slice_area: dict[int, int] = {int(z_start + anchor_local_z): int(anchor_mask.sum())}
    stop_reasons: dict[str, list[int]] = {"no_candidate_component": [], "no_nearby_component": [], "area_growth_capped": []}

    def _propagate(step: int) -> None:
        prev_local = anchor_local_z
        prev_mask = anchor_mask.copy()
        z_local = anchor_local_z + step
        while 0 <= z_local < component_domain.shape[0]:
            global_z = z_start + z_local
            if global_z < track_low or global_z > track_high:
                break
            labels = labels_by_slice[z_local]
            if component_counts_by_slice[z_local] == 0:
                stop_reasons["no_candidate_component"].append(int(global_z))
                break
            distance = ndimage.distance_transform_edt(~prev_mask, sampling=(py, px))
            envelope = distance <= float(max_contour_growth_mm)
            overlap_labels = labels[envelope]
            label_values, overlap_counts = np.unique(overlap_labels[overlap_labels > 0], return_counts=True)
            if label_values.size == 0:
                stop_reasons["no_nearby_component"].append(int(global_z))
                break
            # Highest overlap wins; deterministic tie-breakers prefer component
            # centroid nearest the previous mask and then smallest area.
            prev_ys, prev_xs = np.where(prev_mask)
            prev_cy = float(prev_ys.mean()) if prev_ys.size else float((ny - 1) / 2 - y0)
            prev_cx = float(prev_xs.mean()) if prev_xs.size else float((nx - 1) / 2 - x0)
            choices: list[tuple[int, float, int, int]] = []
            for label_id, overlap_count in zip(label_values.tolist(), overlap_counts.tolist()):
                comp = labels == int(label_id)
                cyx = np.argwhere(comp)
                if cyx.size == 0:
                    continue
                d = float(np.sqrt((((cyx[:, 0].mean() - prev_cy) * py) ** 2) + (((cyx[:, 1].mean() - prev_cx) * px) ** 2)))
                choices.append((int(overlap_count), -d, -int(comp.sum()), -int(label_id)))
            if not choices:
                stop_reasons["no_nearby_component"].append(int(global_z))
                break
            best_choice = max(choices)
            chosen_label = -best_choice[3]
            component = labels == chosen_label
            proposed = component & envelope
            prev_area = max(1, int(prev_mask.sum()))
            area_cap = max(1, int(math.ceil(prev_area * float(max_slice_area_ratio))))
            if int(proposed.sum()) > area_cap:
                stop_reasons["area_growth_capped"].append(int(global_z))
                # Reduce allowed motion radius by bisection so component pixels
                # nearest to the prior contour are retained first.
                lo, hi = 0.0, float(max_contour_growth_mm)
                for _ in range(18):
                    mid = (lo + hi) / 2.0
                    local = component & (distance <= mid)
                    if int(local.sum()) > area_cap:
                        hi = mid
                    else:
                        lo = mid
                proposed = component & (distance <= lo)
            if not proposed.any():
                stop_reasons["no_nearby_component"].append(int(global_z))
                break
            tracked_roi[z_local] = proposed
            per_slice_area[int(global_z)] = int(proposed.sum())
            prev_mask = proposed.copy()
            prev_local = z_local
            z_local += step

    _propagate(-1)
    _propagate(+1)

    tracked_roi = _regularize_tracked_roi(
        tracked_roi, py, px, pz, float(smoothing_closing_mm), float(max_heart_span_mm)
    )
    per_slice_area = {
        int(z_start + z): int(tracked_roi[z].sum())
        for z in np.flatnonzero(tracked_roi.any(axis=(1, 2))).tolist()
    }

    heart_mask = empty.copy()
    heart_mask[z_start:z_end + 1, y0:y1, x0:x1] = tracked_roi
    # Intentional: allow overlap with provisional lung_mask here. Stage 01
    # performs the explicit subtraction AFTER tracking, which is the point of
    # this mask. Forcing disjoint masks here would prevent correction of a
    # heart region accidentally swallowed by the provisional lung segmentation.

    selected_span = float((global_z_end - global_z_start) * pz)
    tracked_slices = sorted(per_slice_area)
    diag.update({
        "status": "tracked_central_3d_component" if heart_mask.any() else "not_found",
        "selected_radius_mm": float(selected_radius),
        "expanded_mask_radius_mm": float(min(float(max_radius_mm), selected_radius + float(max_contour_growth_mm))),
        "selected_component_label": int(anchor_label),
        "selected_component_voxels": int(heart_mask.sum()),
        "selected_z_range": [int(global_z_start), int(global_z_end)],
        "selected_component_span_mm": selected_span,
        "rod_touch_z_range": [int(global_z_start), int(global_z_end)],
        "rod_touch_span_mm": selected_span,
        "rod_touch_slice_count": int(global_z_end - global_z_start + 1),
        "tracked_z_range": [int(tracked_slices[0]), int(tracked_slices[-1])] if tracked_slices else None,
        "tracked_slice_count": int(len(tracked_slices)),
        "anchor_z": int(z_start + anchor_local_z),
        "anchor_component_area_voxels": int(anchor_mask.sum()),
        "candidate_component_counts_by_z": [int(v) for v in component_counts_by_slice],
        "component_area_by_slice": [{"z": int(z), "area_voxels": int(a)} for z, a in sorted(per_slice_area.items())],
        "propagation_stop_reasons": {k: sorted(set(v)) for k, v in stop_reasons.items()},
        "output_mask_voxels": int(heart_mask.sum()),
        "output_mask_lung_overlap_voxels": int(np.count_nonzero(heart_mask & lung)),
        "output_mask_lung_overlap_semantics": "pre-subtraction overlap; Stage 01 must apply lung_mask &= ~heart_mask",
        "mask_description": (
            "Rod-seeded, slice-propagated HU-qualified soft-tissue component constrained "
            "to thoracic slices and a physical radial envelope. It may overlap provisional "
            "lung_mask so Stage 01 can subtract it. Each slice is bounded by physical contour "
            "motion and maximum area growth. Heuristic cardiac/mediastinal exclusion, not "
            "clinical heart segmentation."
        ),
        "failure_reason": None if heart_mask.any() else "rod_seed_component_propagation_produced_empty_mask",
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
