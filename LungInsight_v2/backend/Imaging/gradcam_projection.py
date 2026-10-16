"""Geometry-checked projection of patch CAMs into the native CT voxel frame."""
from __future__ import annotations

import numpy as np
from collections.abc import Sequence


def project_patch_cam_to_ct(
    patch_cam: np.ndarray,
    crop_start_zyx: Sequence[int],
    ct_shape_zyx: Sequence[int],
) -> np.ndarray:
    cam = np.asarray(patch_cam, dtype=np.float32)
    start = np.asarray(crop_start_zyx, dtype=np.int64)
    shape = np.asarray(ct_shape_zyx, dtype=np.int64)
    if cam.ndim != 3 or not np.isfinite(cam).all():
        raise ValueError("patch_cam must be a finite 3-D array.")
    if start.shape != (3,) or shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError("crop_start_zyx and ct_shape_zyx must be three-element vectors.")
    end = start + np.asarray(cam.shape, dtype=np.int64)
    ct_start = np.maximum(start, 0)
    ct_end = np.minimum(end, shape)
    projected = np.full(tuple(int(v) for v in shape), np.nan, dtype=np.float32)
    if np.any(ct_end <= ct_start):
        return projected
    patch_start = ct_start - start
    patch_end = patch_start + (ct_end - ct_start)
    ct_slices = tuple(slice(int(a), int(b)) for a, b in zip(ct_start, ct_end))
    patch_slices = tuple(slice(int(a), int(b)) for a, b in zip(patch_start, patch_end))
    projected[ct_slices] = cam[patch_slices]
    return projected
