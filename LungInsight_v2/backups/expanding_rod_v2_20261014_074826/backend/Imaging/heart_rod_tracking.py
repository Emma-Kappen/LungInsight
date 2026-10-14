"""3D expanding-central-rod tracker for cardiac/mediastinal CT tissue.

This is a heuristic exclusion-mask helper, not a clinically validated cardiac
segmentation model. Input arrays use the canonical ZYX voxel order. All search
radii/distances are specified in millimetres and respect anisotropic spacing.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage


def _validate_spacing(pixel_spacing_mm: tuple[float, float], slice_spacing_mm: float) -> tuple[float, float, float]:
    try:
        py, px = (float(pixel_spacing_mm[0]), float(pixel_spacing_mm[1]))
        pz = float(slice_spacing_mm)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("Expected pixel_spacing_mm=(y_mm, x_mm) and positive slice_spacing_mm") from exc
    if not all(math.isfinite(v) and v > 0 for v in (pz, py, px)):
        raise ValueError("All voxel spacings must be finite and positive")
    return pz, py, px


def _span_mm(indices: np.ndarray, spacing_mm: float) -> float:
    """Physical distance between first/last occupied voxel centers."""
    if indices.size == 0:
        return 0.0
    return float((int(indices[-1]) - int(indices[0])) * spacing_mm)


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
    soft_tissue_hu_min: float = -250.0,
    soft_tissue_hu_max: float = 500.0,
    min_lung_slice_fraction: float = 0.08,
    axial_margin_mm: float = 5.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Track a central soft-tissue component using an expanding 3D rod.

    A cylindrical rod runs parallel to array Z at the center of the CT's YX
    field. Its physical radius expands from ``seed_radius_mm`` to
    ``max_radius_mm`` in ``radius_step_mm`` increments. At each radius we look
    for a 3D-connected soft-tissue component intersecting the rod over a
    sufficiently long Z span. The first radius with a qualifying component is
    selected; among ties, longer axial rod contact, longer component span, and
    larger component size win, deterministically.

    The search is constrained to a thoracic Z interval inferred from slices
    where the current lung mask occupies at least ``min_lung_slice_fraction``
    of the image. HU filtering is independent of lung_mask so that a heart
    accidentally swallowed by the current lung mask can still be detected.

    Returns (mask, diagnostics). An ambiguous/no-match result is an all-False
    mask, which callers must treat as an unverified exclusion gate.
    """
    hu = np.asarray(volume_hu)
    lung = np.asarray(lung_mask, dtype=bool)
    if hu.ndim != 3:
        raise ValueError(f"volume_hu must be a 3D ZYX array, got shape={hu.shape}")
    if lung.shape != hu.shape:
        raise ValueError(f"lung_mask shape {lung.shape} does not match CT shape {hu.shape}")
    pz, py, px = _validate_spacing(pixel_spacing_mm, slice_spacing_mm)
    values = {
        "min_span_mm": min_span_mm,
        "seed_radius_mm": seed_radius_mm,
        "radius_step_mm": radius_step_mm,
        "max_radius_mm": max_radius_mm,
        "soft_tissue_hu_min": soft_tissue_hu_min,
        "soft_tissue_hu_max": soft_tissue_hu_max,
        "min_lung_slice_fraction": min_lung_slice_fraction,
        "axial_margin_mm": axial_margin_mm,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if min_span_mm <= 0 or seed_radius_mm < 0 or radius_step_mm <= 0 or max_radius_mm < seed_radius_mm:
        raise ValueError("Invalid rod radii or minimum span")
    if soft_tissue_hu_min >= soft_tissue_hu_max:
        raise ValueError("soft_tissue_hu_min must be less than soft_tissue_hu_max")
    if not 0.0 < min_lung_slice_fraction <= 1.0 or axial_margin_mm < 0:
        raise ValueError("min_lung_slice_fraction must be in (0, 1] and axial_margin_mm non-negative")

    empty = np.zeros(hu.shape, dtype=bool)
    nz, ny, nx = hu.shape
    cy, cx = (ny - 1.0) / 2.0, (nx - 1.0) / 2.0
    # Limit computation to the bounding box of the largest possible rod. This
    # avoids allocating a full-volume int32 label map for a mostly empty ROI.
    y0 = max(0, int(math.floor(cy - float(max_radius_mm) / py)))
    y1 = min(ny, int(math.ceil(cy + float(max_radius_mm) / py)) + 1)
    x0 = max(0, int(math.floor(cx - float(max_radius_mm) / px)))
    x1 = min(nx, int(math.ceil(cx + float(max_radius_mm) / px)) + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    radius_y_mm = (yy - cy) * py
    radius_x_mm = (xx - cx) * px
    radial_distance_mm = np.sqrt(radius_y_mm * radius_y_mm + radius_x_mm * radius_x_mm)
    max_cylinder = radial_distance_mm <= float(max_radius_mm)

    lung_fraction_by_z = lung.mean(axis=(1, 2), dtype=np.float64)
    thoracic_indices = np.flatnonzero(lung_fraction_by_z >= float(min_lung_slice_fraction))
    base_diag: dict[str, Any] = {
        "method": "expanding_central_rod_3d",
        "status": "not_found",
        "center_yx_voxels": [float(cy), float(cx)],
        "spacing_zyx_mm": [float(pz), float(py), float(px)],
        "soft_tissue_hu_range": [float(soft_tissue_hu_min), float(soft_tissue_hu_max)],
        "seed_radius_mm": float(seed_radius_mm),
        "radius_step_mm": float(radius_step_mm),
        "max_radius_mm": float(max_radius_mm),
        "min_span_mm": float(min_span_mm),
        "min_lung_slice_fraction": float(min_lung_slice_fraction),
        "axial_margin_mm": float(axial_margin_mm),
        "lung_slice_fraction_min": float(lung_fraction_by_z.min()),
        "lung_slice_fraction_max": float(lung_fraction_by_z.max()),
        "thoracic_slice_count": int(thoracic_indices.size),
        "selected_radius_mm": None,
        "selected_component_label": None,
        "selected_component_voxels": 0,
        "selected_z_range": None,
        "selected_component_span_mm": 0.0,
        "rod_touch_z_range": None,
        "rod_touch_span_mm": 0.0,
        "rod_touch_slice_count": 0,
        "candidate_tissue_voxels": 0,
        "failure_reason": None,
    }
    if thoracic_indices.size == 0:
        base_diag["failure_reason"] = "no_thoracic_slices_from_lung_mask"
        return empty, base_diag

    margin_slices = int(math.ceil(float(axial_margin_mm) / pz))
    z0 = max(0, int(thoracic_indices[0]) - margin_slices)
    z1 = min(nz - 1, int(thoracic_indices[-1]) + margin_slices)
    hu_roi = hu[z0:z1 + 1, y0:y1, x0:x1]
    search_region = max_cylinder[None, :, :]

    tissue = np.isfinite(hu_roi) & (hu_roi >= soft_tissue_hu_min) & (hu_roi <= soft_tissue_hu_max)
    candidate = tissue & search_region
    base_diag["thoracic_z_range"] = [int(z0), int(z1)]
    base_diag["search_roi_yx"] = [[int(y0), int(y1)], [int(x0), int(x1)]]
    base_diag["candidate_tissue_voxels"] = int(candidate.sum())
    if not candidate.any():
        base_diag["failure_reason"] = "no_soft_tissue_in_expanding_rod_search_region"
        return empty, base_diag

    # 3D 26-connectivity is intentional: unlike the retired hole-only method,
    # component identity is tracked across the volume, not inferred slice-by-slice.
    labels, component_count = ndimage.label(candidate, structure=np.ones((3, 3, 3), dtype=bool))
    if component_count <= 0:
        base_diag["failure_reason"] = "no_3d_soft_tissue_components"
        return empty, base_diag

    component_slices: dict[int, np.ndarray] = {}
    component_sizes_array = np.bincount(labels.ravel(), minlength=int(component_count) + 1)
    component_sizes: dict[int, int] = {}
    component_spans: dict[int, float] = {}
    # A 3D connected component touches every Z slice between its first and
    # last occupied planes because we use 26-connectivity.
    objects = ndimage.find_objects(labels, max_label=int(component_count))
    for label_id, obj in enumerate(objects, start=1):
        if obj is None:
            continue
        z_slice = obj[0]
        z_indices = np.arange(int(z_slice.start), int(z_slice.stop), dtype=np.int64)
        component_slices[label_id] = z_indices
        component_sizes[label_id] = int(component_sizes_array[label_id])
        component_spans[label_id] = _span_mm(z_indices, pz)

    required_touch_slices = max(2, int(math.ceil(float(min_span_mm) / pz)) + 1)
    radii = []
    r = float(seed_radius_mm)
    while r < max_radius_mm:
        radii.append(r)
        r += float(radius_step_mm)
    if not radii or radii[-1] < max_radius_mm:
        radii.append(float(max_radius_mm))

    chosen_label: int | None = None
    chosen_radius: float | None = None
    chosen_touch_indices = np.array([], dtype=np.int64)
    chosen_score: tuple[float, ...] | None = None

    for radius in radii:
        rod_xy = radial_distance_mm <= radius
        labels_on_rod = labels[:, rod_xy]
        touch_z_by_label: dict[int, list[int]] = {}
        for local_z_index in range(labels.shape[0]):
            row_labels = np.unique(labels_on_rod[local_z_index])
            for label_value in row_labels[row_labels != 0].tolist():
                touch_z_by_label.setdefault(int(label_value), []).append(local_z_index)
        eligible: list[tuple[tuple[float, ...], int, np.ndarray]] = []
        for label_id, touched_z_list in touch_z_by_label.items():
            zs = component_slices.get(label_id, np.array([], dtype=np.int64))
            if zs.size == 0 or component_spans.get(label_id, 0.0) < float(min_span_mm):
                continue
            touch_z = np.asarray(touched_z_list, dtype=np.int64)
            touch_span = _span_mm(touch_z, pz)
            if touch_z.size < required_touch_slices or touch_span < float(min_span_mm):
                continue
            # First qualifying rod radius wins. Deterministic tie-break: most
            # Z slices touched, then axial span, then component size, then label.
            score = (
                float(touch_z.size),
                float(touch_span),
                float(component_spans[label_id]),
                float(component_sizes[label_id]),
                -float(label_id),
            )
            eligible.append((score, label_id, touch_z))
        if eligible:
            eligible.sort(key=lambda item: item[0], reverse=True)
            chosen_score, chosen_label, chosen_touch_indices = eligible[0]
            chosen_radius = float(radius)
            break

    if chosen_label is None:
        base_diag["failure_reason"] = "no_3d_component_met_rod_contact_and_z_span_criteria"
        base_diag["component_count"] = int(component_count)
        base_diag["required_rod_touch_slices"] = int(required_touch_slices)
        return empty, base_diag

    selected_component_roi = labels == chosen_label
    heart_mask = np.zeros(hu.shape, dtype=bool)
    heart_mask[z0:z1 + 1, y0:y1, x0:x1] = selected_component_roi
    component_z = component_slices[chosen_label] + z0
    global_touch_indices = chosen_touch_indices + z0
    base_diag.update({
        "status": "tracked_central_3d_component",
        "search_roi_yx": [[int(y0), int(y1)], [int(x0), int(x1)]],
        "selected_radius_mm": float(chosen_radius),
        "selected_component_label": int(chosen_label),
        "selected_component_voxels": int(heart_mask.sum()),
        "selected_z_range": [int(component_z[0]), int(component_z[-1])],
        "selected_component_span_mm": float(component_spans[chosen_label]),
        "rod_touch_z_range": [int(global_touch_indices[0]), int(global_touch_indices[-1])],
        "rod_touch_span_mm": float(_span_mm(global_touch_indices, pz)),
        "rod_touch_slice_count": int(global_touch_indices.size),
        "candidate_tissue_voxels": int(candidate.sum()),
        "component_count": int(component_count),
        "required_rod_touch_slices": int(required_touch_slices),
        "tie_break_score": [float(v) for v in chosen_score or ()],
        "failure_reason": None,
    })
    return heart_mask, base_diag


def associate_large_hole_components(
    large_holes_stack: np.ndarray,
    tracked_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Associate whole 3D large-hole components that overlap a rod-tracked mask.

    This preserves Stage 01's intent to recover non-cardiac large intrapulmonary
    masses, but only after a central component has been found. A hole component
    is classified as cardiac-associated only when it intersects the tracked
    component in 3D; non-intersecting large-hole components remain eligible for
    the existing lung-mask restoration path.
    """
    holes = np.asarray(large_holes_stack, dtype=bool)
    tracked = np.asarray(tracked_mask, dtype=bool)
    if holes.ndim != 3 or holes.shape != tracked.shape:
        raise ValueError("large_holes_stack and tracked_mask must have the same 3D ZYX shape")
    associated = np.zeros_like(holes, dtype=bool)
    labels, count = ndimage.label(holes, structure=np.ones((3, 3, 3), dtype=bool))
    matched_labels: list[int] = []
    if count:
        touched = np.unique(labels[tracked & holes])
        for val in touched:
            label_id = int(val)
            if label_id == 0:
                continue
            associated |= labels == label_id
            matched_labels.append(label_id)
    return associated, {
        "large_hole_component_count": int(count),
        "associated_large_hole_component_labels": matched_labels,
        "associated_large_hole_voxels": int(associated.sum()),
    }
