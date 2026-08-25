"""Shared contracts and geometry helpers for the imaging pipeline.

Array coordinates are always ordered ``(z, y, x)``. Bboxes are half-open:
``(z0, z1, y0, y1, x0, x1)``. Patient coordinates use DICOM's LPS frame.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np


BBoxZYX = Tuple[int, int, int, int, int, int]


def bbox_from_ranges(ranges: Sequence[Sequence[int]]) -> BBoxZYX:
    """Convert three half-open ``(start, stop)`` ranges to a bbox."""
    if len(ranges) != 3:
        raise ValueError("three axis ranges are required")
    return make_bbox(
        (ranges[0][0], ranges[1][0], ranges[2][0]),
        (ranges[0][1], ranges[1][1], ranges[2][1]),
    )


def bbox_ranges(bbox: BBoxZYX):
    """Return a bbox as ``((z0,z1), (y0,y1), (x0,x1))``."""
    return tuple((int(bbox[i]), int(bbox[i + 3])) for i in range(3))


def translate_bbox(bbox: BBoxZYX, offset_zyx: Sequence[int]) -> BBoxZYX:
    """Translate a half-open bbox by a voxel offset."""
    offset = tuple(int(v) for v in offset_zyx)
    if len(offset) != 3:
        raise ValueError("offset_zyx must contain three values")
    return make_bbox(
        tuple(bbox[i] + offset[i] for i in range(3)),
        tuple(bbox[i + 3] + offset[i] for i in range(3)),
    )


def stable_candidate_id(center_zyx: Sequence[int], source: str = "seed") -> str:
    """Create a stable, human-readable identity independent of list order."""
    center = tuple(int(v) for v in center_zyx)
    if len(center) != 3:
        raise ValueError("center_zyx must contain three values")
    normalized_source = "".join(ch if ch.isalnum() else "_" for ch in str(source)).strip("_")
    return f"{normalized_source or 'seed'}:{center[0]}:{center[1]}:{center[2]}"


def candidate_geometry(
    center_zyx: Sequence[int],
    bbox: BBoxZYX,
    geometry: "VolumeGeometry",
    source: str = "seed",
) -> Dict[str, Any]:
    """Return the canonical geometry fields shared by pipeline stages."""
    center = tuple(int(v) for v in center_zyx)
    if len(center) != 3:
        raise ValueError("center_zyx must contain three values")
    patient = geometry.voxel_to_patient(center)
    return {
        "candidate_id": stable_candidate_id(center, source),
        "center_zyx": list(center),
        "bbox_zyx": list(bbox),
        "crop_offset_zyx": list(geometry.crop_offset_zyx),
        "patient_lps_mm": [float(v) for v in patient],
        "spacing_zyx_mm": [float(v) for v in geometry.spacing_zyx_mm],
    }


@dataclass(frozen=True)
class VolumeGeometry:
    """Authoritative voxel-to-patient geometry for a saved volume."""

    shape_zyx: Tuple[int, int, int]
    spacing_zyx_mm: Tuple[float, float, float]
    origin_lps_mm: Tuple[float, float, float]
    direction_lps: Tuple[Tuple[float, float, float], ...]
    crop_offset_zyx: Tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        if tuple(self.shape_zyx) != tuple(int(v) for v in self.shape_zyx):
            raise ValueError("shape_zyx must contain integers")
        if any(v <= 0 for v in self.shape_zyx):
            raise ValueError("shape_zyx must be positive")
        if any(v <= 0 for v in self.spacing_zyx_mm):
            raise ValueError("spacing_zyx_mm must be positive")
        direction = np.asarray(self.direction_lps, dtype=float)
        if direction.shape != (3, 3) or not np.all(np.isfinite(direction)):
            raise ValueError("direction_lps must be a finite 3x3 matrix")

    @property
    def spacing(self) -> np.ndarray:
        return np.asarray(self.spacing_zyx_mm, dtype=float)

    @property
    def origin(self) -> np.ndarray:
        return np.asarray(self.origin_lps_mm, dtype=float)

    @property
    def direction(self) -> np.ndarray:
        return np.asarray(self.direction_lps, dtype=float)

    def voxel_to_patient(self, voxel_zyx: Sequence[float]) -> np.ndarray:
        voxel = np.asarray(voxel_zyx, dtype=float)
        if voxel.shape != (3,):
            raise ValueError("voxel coordinate must have shape (3,)")
        return self.origin + self.direction @ (voxel * self.spacing)

    def patient_to_voxel(self, patient_lps_mm: Sequence[float]) -> np.ndarray:
        patient = np.asarray(patient_lps_mm, dtype=float)
        if patient.shape != (3,):
            raise ValueError("patient coordinate must have shape (3,)")
        return (np.linalg.inv(self.direction) @ (patient - self.origin)) / self.spacing


def make_bbox(start_zyx: Iterable[int], stop_zyx: Iterable[int]) -> BBoxZYX:
    start = tuple(int(v) for v in start_zyx)
    stop = tuple(int(v) for v in stop_zyx)
    if len(start) != 3 or len(stop) != 3:
        raise ValueError("bbox coordinates must contain three axes")
    if any(a < 0 or b <= a for a, b in zip(start, stop)):
        raise ValueError(f"invalid half-open bbox: start={start}, stop={stop}")
    return start + stop


def clamp_bbox(bbox: BBoxZYX, shape_zyx: Sequence[int]) -> BBoxZYX:
    if len(shape_zyx) != 3:
        raise ValueError("shape_zyx must contain three axes")
    start = tuple(max(0, min(int(a), int(size) - 1)) for a, size in zip(bbox[:3], shape_zyx))
    stop = tuple(max(a + 1, min(int(b), int(size))) for a, b, size in zip(start, bbox[3:], shape_zyx))
    if any(b <= a for a, b in zip(start, stop)):
        raise ValueError(f"bbox is empty after clamping: {bbox}")
    return start + stop


def bbox_slices(bbox: BBoxZYX):
    return tuple(slice(bbox[i], bbox[i + 3]) for i in range(3))


def geometry_from_meta(meta: dict, shape_zyx: Sequence[int]) -> VolumeGeometry:
    pixel = meta.get("pixel_spacing_mm", [1.0, 1.0])
    spacing = tuple(float(v) for v in meta.get("spacing_zyx_mm", [meta.get("slice_spacing_mm", 1.0), pixel[0], pixel[1]]))
    direction = meta.get("direction_lps", np.eye(3).tolist())
    return VolumeGeometry(
        shape_zyx=tuple(int(v) for v in shape_zyx),
        spacing_zyx_mm=spacing,
        origin_lps_mm=tuple(float(v) for v in meta.get("origin_mm", [0.0, 0.0, 0.0])),
        direction_lps=tuple(tuple(float(v) for v in row) for row in direction),
        crop_offset_zyx=tuple(int(v) for v in meta.get("crop_offset_zyx", [0, 0, 0])),
    )
