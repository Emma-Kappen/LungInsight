"""Deterministic stable identity for accepted imaging nodules."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence


def stable_nodule_id(
    patient_id: str,
    study_uid: str,
    series_uid: str,
    centroid_lps_mm: Sequence[float],
    detector_source: str,
) -> str:
    """Build an identity from study identity, physical position and source.

    Position is quantized to micrometres before hashing so harmless JSON
    float-format differences do not change an ID, while detector output
    precision is retained. The detector source is the fused record's source
    (ViTDet3D, LoG or Merged), not an array/list position.
    """
    if detector_source not in {"ViTDet3D", "LoG", "Merged"}:
        raise ValueError("detector_source must be one of ViTDet3D, LoG, or Merged.")
    identity = {
        "patient_id": _required_text(patient_id, "patient_id"),
        "study_uid": _required_text(study_uid, "study_uid"),
        "series_uid": _required_text(series_uid, "series_uid"),
        "centroid_lps_mm": _canonical_point(centroid_lps_mm),
        "detector_source": _required_text(detector_source, "detector_source"),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "nodule_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _required_text(value: str, name: str) -> str:
    result = str(value).strip()
    if not result or result.upper() in {"UNKNOWN", "N/A", "NA", "NONE", "PLACEHOLDER"}:
        raise ValueError(f"{name} must be a verified, non-placeholder string to create a stable nodule ID.")
    return result


def _canonical_point(point: Sequence[float]) -> list[int]:
    if isinstance(point, (str, bytes)) or len(point) != 3:
        raise ValueError("centroid_lps_mm must contain three finite coordinates.")
    values: list[int] = []
    for coordinate in point:
        value = float(coordinate)
        if not math.isfinite(value):
            raise ValueError("centroid_lps_mm must contain three finite coordinates.")
        values.append(round(value * 1000))
    return values
