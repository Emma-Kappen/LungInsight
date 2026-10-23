"""Paravertebral gate for Stage 02 candidates.

The Stage 01 lung mask can include the posterior costovertebral recess, so a detection sitting on the vertebral
column passes the centre-in-lung rule although it is not a pulmonary nodule. This gate localises the vertebral
column from bone density in the posterior-central part of the body and rejects candidates whose detector box lies
within ``MIN_SPINE_DISTANCE_MM`` of it. When the column cannot be found (for example a cropped series) the gate
rejects nothing and says so in its report; it never invents a rejection.
"""
from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

BONE_HU = 200.0
MIN_SPINE_DISTANCE_MM = 10.0
SEARCH_MARGIN_MM = 14.0
MIN_SPINE_EXTENT_MM = 60.0
SPINE_HALF_WIDTH_MM = 45.0
POSTERIOR_FRACTION = 0.45
EXCLUSION_REASON = "paravertebral_bone_adjacent"


def locate_spine(
    volume_hu: np.ndarray,
    lung_mask: np.ndarray,
    spacing_zyx_mm: Sequence[float],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return a bool mask of the vertebral column (same shape as the CT) or None, plus a report."""
    spacing = np.asarray(spacing_zyx_mm, dtype=np.float64)
    lung = np.asarray(lung_mask, dtype=bool)
    if volume_hu.ndim != 3 or lung.shape != volume_hu.shape or not lung.any():
        return None, {"status": "spine_not_found", "reason": "invalid_or_empty_lung_mask"}
    columns = np.where(lung.any(axis=(0, 1)))[0]
    rows = np.where(lung.any(axis=(0, 2)))[0]
    middle = (columns.min() + columns.max()) / 2.0
    x0 = max(int(middle - SPINE_HALF_WIDTH_MM / spacing[2]), 0)
    x1 = min(int(middle + SPINE_HALF_WIDTH_MM / spacing[2]), volume_hu.shape[2])
    y0 = int(rows.min() + POSTERIOR_FRACTION * (rows.max() - rows.min()))
    band = np.asarray(volume_hu[:, y0:, x0:x1]) > BONE_HU
    band = ndimage.binary_opening(band, iterations=1)
    labels, count = ndimage.label(band)
    if count == 0:
        return None, {"status": "spine_not_found", "reason": "no_bone_in_posterior_central_band"}
    extents = ndimage.find_objects(labels)
    best = max(range(count), key=lambda i: extents[i][0].stop - extents[i][0].start)
    z_extent_mm = float((extents[best][0].stop - extents[best][0].start) * spacing[0])
    if z_extent_mm < MIN_SPINE_EXTENT_MM:
        return None, {"status": "spine_not_found", "reason": "bone_column_too_short", "z_extent_mm": round(z_extent_mm, 1)}
    spine = np.zeros(volume_hu.shape, dtype=bool)
    spine[:, y0:, x0:x1] = labels == best + 1
    return spine, {"status": "spine_located", "z_extent_mm": round(z_extent_mm, 1), "voxels": int(spine.sum())}


def box_distance_to_spine_mm(
    spine: np.ndarray,
    box_start_zyx: Sequence[float],
    box_end_zyx: Sequence[float],
    spacing_zyx_mm: Sequence[float],
) -> float | None:
    """Minimum distance in mm from a box to the spine, or None when the spine is farther than the search margin."""
    spacing = np.asarray(spacing_zyx_mm, dtype=np.float64)
    shape = np.asarray(spine.shape)
    margin = np.ceil(SEARCH_MARGIN_MM / spacing).astype(int)
    start = np.floor(np.asarray(box_start_zyx, dtype=np.float64)).astype(int)
    end = np.ceil(np.asarray(box_end_zyx, dtype=np.float64)).astype(int)
    low = np.maximum(start - margin, 0)
    high = np.minimum(end + margin, shape)
    window = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    region = spine[window]
    if not region.any():
        return None
    distance = ndimage.distance_transform_edt(~region, sampling=spacing)
    inner = tuple(
        slice(int(max(s, a) - a), int(min(e, b) - a))
        for s, e, a, b in zip(start, end, low, high)
    )
    inside = distance[inner]
    return float(inside.min()) if inside.size else None


def apply_paravertebral_gate(
    accepted: Sequence[dict[str, Any]],
    volume_hu: np.ndarray,
    lung_mask: np.ndarray,
    spacing_zyx_mm: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split already lung-gated candidates into (kept, newly_rejected, report)."""
    spine, spine_report = locate_spine(volume_hu, lung_mask, spacing_zyx_mm)
    report: dict[str, Any] = {
        "rule": f"detector box within {MIN_SPINE_DISTANCE_MM} mm of the vertebral column is rejected",
        "bone_threshold_hu": BONE_HU,
        "min_spine_distance_mm": MIN_SPINE_DISTANCE_MM,
        **spine_report,
        "rejected_ids": [],
    }
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for original in accepted:
        row = copy.deepcopy(original)
        distance = None
        if spine is not None:
            box_start = row.get("bbox_start_zyx")
            box_end = row.get("bbox_end_zyx") or row.get("bbox_end_zyx_exclusive")
            if box_start is not None and box_end is not None:
                distance = box_distance_to_spine_mm(spine, box_start, box_end, spacing_zyx_mm)
        row["distance_to_spine_mm"] = None if distance is None else round(distance, 2)
        if distance is not None and distance <= MIN_SPINE_DISTANCE_MM:
            row["candidate_in_lung_mask"] = False
            row["exclusion_reason"] = EXCLUSION_REASON
            rejected.append(row)
            report["rejected_ids"].append(str(row.get("stable_nodule_id") or row.get("candidate_id")))
        else:
            kept.append(row)
    report["rejected_count"] = len(rejected)
    return kept, rejected, report
