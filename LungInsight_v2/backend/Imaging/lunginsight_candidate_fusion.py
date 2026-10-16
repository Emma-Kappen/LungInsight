#!/usr/bin/env python3
"""Schema-aware, deterministic LoG + ViTDet3D merge and lung/heart gating.

This is a standalone safety utility. It deliberately does not import the
project's unsupplied pipeline_schema.py or model modules. It never predicts
attributes, malignancy, Grad-CAM, or SHAP values. The filtered JSON output is
an audit/intermediate artifact, not a replacement for Stage 02's patch manifest
until explicitly integrated with the original patch-extraction contract.

Canonical geometry is read from meta_schema.json and validated structurally:
  * CT arrays and center_zyx are ordered [z, y, x].
  * spacing_zyx_mm is [z, y, x] in millimetres.
  * points must refer to the same stage01_final_cropped_ct frame.
  * bbox starts are inclusive and bbox ends exclusive when present.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from nodule_identity import stable_nodule_id


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON at {path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def unwrap_records(payload: Any, *, path: str = "input") -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("candidates", payload.get("predictions"))
        if records is None:
            # A few Stage 02 outputs are single candidate objects, not collections.
            raise ValueError(f"{path}: expected a list or an object with 'candidates'/'predictions'.")
    else:
        raise ValueError(f"{path}: expected a JSON list or object.")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"{path}: candidate collection must be a list of objects.")
    return [dict(row) for row in records]


def load_canonical_contract(schema_path: Path) -> dict[str, Any]:
    """Read and structurally validate canonical geometry without rewriting schema.

    Schema versions are not hard-coded: newer schemas are accepted only when
    they explicitly preserve the coordinate contract this utility needs. This
    avoids rejecting a schema solely because its version was incremented, while
    still failing closed if axes, physical coordinates, or CT frame disagree.
    """
    schema = read_json(schema_path)
    if not isinstance(schema, dict):
        raise ValueError(f"Schema must be a JSON object: {schema_path}")
    version = schema.get("schema_version")
    if version is None or not str(version).strip():
        raise ValueError(f"Schema is missing a non-empty schema_version: {schema_path}")
    fmt = schema.get("coordinate_format")
    stage02 = schema.get("stage_02_detect_and_patch")
    if not isinstance(fmt, dict) or not isinstance(stage02, dict):
        raise ValueError("Schema is missing coordinate_format or stage_02_detect_and_patch.")

    top_axes = schema.get("array_axes")
    fmt_axes = fmt.get("array_axes")
    if top_axes is not None and fmt_axes is not None and top_axes != fmt_axes:
        raise ValueError(f"Schema array_axes conflict: top-level={top_axes!r}, coordinate_format={fmt_axes!r}.")
    axes = fmt_axes if fmt_axes is not None else top_axes
    if axes != ["z", "y", "x"]:
        raise ValueError(f"Unsupported canonical array axes: {axes!r}; expected ['z','y','x'].")

    top_system = schema.get("coordinate_system")
    fmt_system = fmt.get("coordinate_system")
    if top_system is not None and fmt_system is not None and top_system != fmt_system:
        raise ValueError(f"Schema coordinate_system conflict: top-level={top_system!r}, coordinate_format={fmt_system!r}.")
    coordinate_system = fmt_system if fmt_system is not None else top_system
    if coordinate_system != "DICOM_LPS":
        raise ValueError(f"Unsupported canonical physical coordinate system: {coordinate_system!r}; expected DICOM_LPS.")

    stage_space = stage02.get("space")
    format_space = fmt.get("ct_frame_space")
    if stage_space is not None and format_space is not None and str(stage_space) != str(format_space):
        raise ValueError(
            f"Schema CT-frame conflict: stage_02_detect_and_patch.space={stage_space!r}, "
            f"coordinate_format.ct_frame_space={format_space!r}. Resolve canonical geometry before fusion."
        )
    ct_space = stage_space if stage_space is not None else format_space
    if ct_space is None or not str(ct_space).strip():
        raise ValueError("Schema must define stage_02_detect_and_patch.space or coordinate_format.ct_frame_space.")

    formula = fmt.get("index_to_physical_formula", "")
    return {
        "schema_version": str(version),
        "array_axes": list(axes),
        "coordinate_system": str(coordinate_system),
        "ct_space": str(ct_space),
        "index_to_physical_formula": str(formula),
        "schema": schema,
    }


def _finite_vec(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric three-vector.") from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite three-vector; got {result!r}.")
    return result


def _source(record: dict[str, Any]) -> str:
    raw = record.get("detection_source", record.get("source"))
    if raw is None:
        raw = "LoG" if ("log_response" in record or "diagnostic_id" in record) else "ViTDet3D"
    source = str(raw).strip()
    lower = source.lower()
    if "log" in lower:
        return "LoG"
    if "vit" in lower or "detr" in lower:
        return "ViTDet3D"
    raise ValueError(f"Unrecognized detector source {raw!r}; expected LoG or ViTDet3D.")


def _candidate_identity(record: dict[str, Any], center: np.ndarray) -> str:
    for key in ("candidate_id", "diagnostic_id", "id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"{_source(record)}:{','.join(f'{v:.4f}' for v in center)}"


def _bbox(record: dict[str, Any], center: np.ndarray, spacing: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    aliases = (
        ("bbox_start_zyx", "bbox_end_zyx"),
        ("bbox_low_zyx", "bbox_high_zyx"),
        ("low_zyx", "high_zyx"),
    )
    for low_key, high_key in aliases:
        if record.get(low_key) is not None and record.get(high_key) is not None:
            low = _finite_vec(record[low_key], low_key)
            high = _finite_vec(record[high_key], high_key)
            if np.any(high <= low):
                raise ValueError(f"Candidate {record.get('candidate_id', record.get('diagnostic_id'))}: invalid bbox {low_key}/{high_key}.")
            return low, high
    # LoG records in the supplied source encode a physical diameter, not an
    # explicit box. Use that scale only to create a geometrical comparison box.
    scale = record.get("scale_mm", record.get("diameter_mm"))
    try:
        scale_f = float(scale)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scale_f) or scale_f <= 0:
        return None
    half_extent_vox = np.maximum(scale_f / (2.0 * spacing), 0.5)
    low = np.floor(center - half_extent_vox)
    high = np.maximum(np.ceil(center + half_extent_vox), low + 1.0)
    return low, high


def _box_iou(a: tuple[np.ndarray, np.ndarray] | None, b: tuple[np.ndarray, np.ndarray] | None) -> float:
    if a is None or b is None:
        return 0.0
    lo_a, hi_a = a
    lo_b, hi_b = b
    intersection_extent = np.maximum(0.0, np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b))
    intersection = float(np.prod(intersection_extent))
    if intersection <= 0:
        return 0.0
    volume_a = float(np.prod(np.maximum(0.0, hi_a - lo_a)))
    volume_b = float(np.prod(np.maximum(0.0, hi_b - lo_b)))
    denominator = volume_a + volume_b - intersection
    return intersection / denominator if denominator > 0 else 0.0


def _normalise_detection(
    record: dict[str, Any], *, spacing: np.ndarray, expected_space: str | None
) -> dict[str, Any]:
    row = copy.deepcopy(record)
    source = _source(row)
    center = _finite_vec(row.get("center_zyx"), "center_zyx")
    if row.get("coordinate_order") is not None:
        order = str(row["coordinate_order"]).replace("[", "").replace("]", "").replace(",", "").replace(" ", "").upper()
        if order not in ("ZYX", "ZYXVOXEL", "VOXELZYX"):
            raise ValueError(f"Candidate {_candidate_identity(row, center)} uses coordinate_order={row['coordinate_order']!r}; expected ZYX.")
    space = row.get("space", row.get("source_space"))
    if space is not None and expected_space is not None and str(space) != expected_space:
        raise ValueError(
            f"Candidate {_candidate_identity(row, center)} is in space {space!r}, expected {expected_space!r}; "
            "convert geometry before comparing detectors."
        )
    if expected_space and space is None:
        row["space"] = expected_space
    row["center_zyx"] = center.tolist()
    row["_fusion_source"] = source
    row["_fusion_center"] = center
    row["_fusion_bbox"] = _bbox(row, center, spacing)
    row["_fusion_identity"] = _candidate_identity(row, center)
    if source == "ViTDet3D":
        score = row.get("detector_score", row.get("score"))
        score_name = "detector_score"
    else:
        score = row.get("log_response", row.get("score"))
        score_name = "log_response"
    try:
        score_value = float(score)
        if not math.isfinite(score_value):
            score_value = None
    except (TypeError, ValueError):
        score_value = None
    row["_fusion_score_name"] = score_name
    row["_fusion_score_value"] = score_value
    return row


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    # Do not compare unlike score scales. ViTDet3D is preferred for geometry
    # because the supplied schema declares it authoritative. Within one source,
    # a finite detector score ranks candidates; stable identity breaks ties.
    source_priority = 0 if row["_fusion_source"] == "ViTDet3D" else 1
    score = row["_fusion_score_value"]
    return (source_priority, 0 if score is not None else 1, -(score or 0.0), row["_fusion_identity"])


def _public_record(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_fusion_")}


def _matches(
    a: dict[str, Any], b: dict[str, Any], spacing: np.ndarray,
    center_distance_mm: float, max_center_distance_mm: float, min_overlap_iou: float,
) -> tuple[bool, float, float]:
    distance = float(np.linalg.norm((a["_fusion_center"] - b["_fusion_center"]) * spacing))
    iou = _box_iou(a["_fusion_bbox"], b["_fusion_bbox"])
    matched = distance <= center_distance_mm or (distance <= max_center_distance_mm and iou >= min_overlap_iou)
    return matched, distance, iou


def rank_filtered_log_candidates(
    filtered_candidates: Sequence[dict[str, Any]], *, limit: int = 50
) -> list[dict[str, Any]]:
    """Rank already quality-filtered LoG records by native response only."""
    if limit < 0:
        raise ValueError("limit must be non-negative.")
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for row in filtered_candidates:
        if not isinstance(row, dict):
            raise ValueError("Every filtered LoG candidate must be a dictionary.")
        value = row.get("log_response", row.get("detector_score"))
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Filtered LoG candidate {row.get('diagnostic_id', row.get('candidate_id'))!r} "
                "has no finite log_response for ranking."
            ) from exc
        if not math.isfinite(score):
            raise ValueError(
                f"Filtered LoG candidate {row.get('diagnostic_id', row.get('candidate_id'))!r} "
                "has no finite log_response for ranking."
            )
        identity = str(row.get("diagnostic_id", row.get("candidate_id", "")))
        ranked.append((-score, identity, copy.deepcopy(row)))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in ranked[:limit]]


def select_fused_candidates(
    fused_candidates: Sequence[dict[str, Any]], *, limit: int = 50
) -> list[dict[str, Any]]:
    """Select a deterministic, source-aware top-N from the fused set.

    Detector scores retain their native semantics. ViTDet3D candidates are
    ordered by detector score, followed by LoG-only candidates ordered by
    LoG response; scores are never compared across detector types.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative.")
    rows = [copy.deepcopy(row) for row in fused_candidates]
    for row in rows:
        source = str(row.get("source", row.get("detection_source", "")))
        if source not in {"ViTDet3D", "LoG", "Merged"}:
            raise ValueError(f"Fused candidate has unsupported source {source!r}.")
        row["_fusion_source"] = "ViTDet3D" if source in {"ViTDet3D", "Merged"} else "LoG"
        score = row.get("vitdet_score") if source in {"ViTDet3D", "Merged"} else row.get("log_score")
        try:
            value = float(score)
        except (TypeError, ValueError):
            value = None
        row["_fusion_score_value"] = value if value is not None and math.isfinite(value) else None
        row["_fusion_identity"] = str(row.get("stable_nodule_id", row.get("nodule_id", row.get("candidate_id", ""))))
    rows.sort(key=_rank_key)
    return [{key: value for key, value in row.items() if not key.startswith("_fusion_")} for row in rows[:limit]]


def merge_detections(
    log_detections: Sequence[dict[str, Any]],
    vitdet_detections: Sequence[dict[str, Any]],
    *,
    spacing_zyx_mm: Sequence[float],
    center_distance_mm: float = 3.0,
    max_center_distance_mm: float = 8.0,
    min_overlap_iou: float = 0.25,
    expected_space: str | None = None,
    patient_id: str | None = None,
    study_uid: str | None = None,
    series_uid: str | None = None,
    origin_lps_mm: Sequence[float] | None = None,
    direction_lps: Sequence[Sequence[float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge two detector lists and return (deduplicated records, count report).

    Match policy: merge when physical center distance <= center_distance_mm;
    otherwise merge only when center distance <= max_center_distance_mm AND
    3-D box IoU >= min_overlap_iou. Distances multiply ZYX index differences
    by spacing_zyx_mm. This handles anisotropic voxels.

    Conflict policy: preserve the ViTDet3D candidate's center and box whenever
    a group contains one; otherwise preserve the LoG candidate with the
    strongest finite LoG response. Never average locations or compare LoG
    response to ViTDet3D detector_score. Other source records/scores are kept
    under detector_provenance and detector_scores. Grouping and tie-breaking
    are deterministic for a fixed candidate set and configuration.
    """
    spacing = _finite_vec(spacing_zyx_mm, "spacing_zyx_mm")
    if np.any(spacing <= 0):
        raise ValueError("spacing_zyx_mm values must be positive.")
    for name, value in (("center_distance_mm", center_distance_mm), ("max_center_distance_mm", max_center_distance_mm)):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if max_center_distance_mm < center_distance_mm:
        raise ValueError("max_center_distance_mm must be >= center_distance_mm.")
    if not 0.0 <= float(min_overlap_iou) <= 1.0:
        raise ValueError("min_overlap_iou must be between 0 and 1.")

    raw_log = list(log_detections or [])
    raw_vit = list(vitdet_detections or [])
    if not all(isinstance(x, dict) for x in raw_log + raw_vit):
        raise ValueError("Every detector record must be a dictionary.")
    normalised = [
        _normalise_detection(row, spacing=spacing, expected_space=expected_space)
        for row in raw_log + raw_vit
    ]
    normalised.sort(key=_rank_key)

    # Each cluster representative is the best-ranked record; cluster membership
    # is not transitive by default, avoiding a chain of overlapping boxes from
    # collapsing several nearby nodules into one candidate.
    clusters: list[list[dict[str, Any]]] = []
    for row in normalised:
        options: list[tuple[float, float, int]] = []
        for index, cluster in enumerate(clusters):
            representative = cluster[0]
            matched, distance, iou = _matches(
                row, representative, spacing, float(center_distance_mm),
                float(max_center_distance_mm), float(min_overlap_iou),
            )
            if matched:
                options.append((distance, -iou, index))
        if options:
            _distance, _negative_iou, target_index = min(options)
            clusters[target_index].append(row)
        else:
            clusters.append([row])

    results: list[dict[str, Any]] = []
    for cluster in clusters:
        cluster.sort(key=_rank_key)
        winner = cluster[0]
        output = _public_record(winner)
        sources = sorted({item["_fusion_source"] for item in cluster})
        center = winner["_fusion_center"]
        bbox = winner["_fusion_bbox"]
        if all(value is not None for value in (patient_id, study_uid, series_uid, origin_lps_mm, direction_lps)):
            origin = _finite_vec(origin_lps_mm, "origin_lps_mm")
            direction = np.asarray(direction_lps, dtype=np.float64)
            if direction.shape != (3, 3) or not np.isfinite(direction).all():
                raise ValueError("direction_lps must be a finite 3x3 matrix.")
            center_lps = origin + direction @ (center * spacing)
            stable_id = stable_nodule_id(
                str(patient_id), str(study_uid), str(series_uid), center_lps,
                "Merged" if len(sources) > 1 else sources[0],
            )
        elif any(value is not None for value in (patient_id, study_uid, series_uid, origin_lps_mm, direction_lps)):
            raise ValueError(
                "Stable identity requires patient_id, study_uid, series_uid, origin_lps_mm, and direction_lps together."
            )
        else:
            # Compatibility for existing standalone callers that only use
            # this geometry utility. Production Stage 02 always supplies the
            # full physical study identity above.
            legacy_identity = {
                "center_zyx": [round(float(v), 3) for v in center],
                "bbox": None if bbox is None else {
                    "start": [round(float(v), 3) for v in bbox[0]],
                    "end": [round(float(v), 3) for v in bbox[1]],
                },
            }
            stable_id = "nodule_" + hashlib.sha256(
                json.dumps(legacy_identity, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
        detector_scores: dict[str, Any] = {}
        provenance = []
        source_members = []
        vitdet_score = None
        log_score = None
        for item in sorted(cluster, key=lambda r: (r["_fusion_source"], r["_fusion_identity"])):
            src = item["_fusion_source"]
            score_name = item["_fusion_score_name"]
            score_value = item["_fusion_score_value"]
            src_key = "ViTDet3D" if src == "ViTDet3D" else "LoG"
            detector_scores.setdefault(src_key, []).append({"field": score_name, "value": score_value})
            member_id = str(item.get("candidate_id", item.get("diagnostic_id", item["_fusion_identity"])))
            source_members.append({
                "candidate_id": member_id,
                "source": src,
                "score_field": score_name,
                "score": score_value,
                "original_purpose": item.get("purpose"),
            })
            if src == "ViTDet3D" and vitdet_score is None:
                vitdet_score = score_value
            if src == "LoG" and log_score is None:
                log_score = score_value
            provenance.append({
                "detector": src,
                "candidate_id": member_id,
                "center_zyx": item["_fusion_center"].tolist(),
                "bbox_start_zyx": None if item["_fusion_bbox"] is None else item["_fusion_bbox"][0].tolist(),
                "bbox_end_zyx": None if item["_fusion_bbox"] is None else item["_fusion_bbox"][1].tolist(),
                "score_field": score_name,
                "score": score_value,
            })
        output["nodule_id"] = stable_id
        output["stable_nodule_id"] = stable_id
        output["candidate_id"] = stable_id
        output["source"] = "Merged" if len(sources) > 1 else sources[0]
        output["source_members"] = source_members
        output["vitdet_score"] = vitdet_score
        output["log_score"] = log_score
        output["detector_provenance"] = sources
        output["detector_scores"] = detector_scores
        output["provenance_records"] = provenance
        output["primary_geometry_source"] = winner["_fusion_source"]
        output["merged_member_count"] = len(cluster)
        output["detection_source"] = output["source"]
        output["center_lps_mm"] = (
            center_lps.tolist()
            if all(value is not None for value in (patient_id, study_uid, series_uid, origin_lps_mm, direction_lps))
            else output.get("center_lps_mm")
        )
        if bbox is not None:
            output.setdefault("bbox_start_zyx", np.floor(bbox[0]).astype(int).tolist())
            output.setdefault("bbox_end_zyx", np.ceil(bbox[1]).astype(int).tolist())
        output["detection_source"] = output["source"]
        results.append(output)

    results.sort(key=lambda row: (
        tuple(round(float(v), 6) for v in row.get("center_zyx", (math.inf, math.inf, math.inf))),
        str(row["nodule_id"]),
    ))
    report = {
        "log_count": len(raw_log),
        "vitdet3d_count": len(raw_vit),
        "union_count_before_deduplication": len(raw_log) + len(raw_vit),
        "deduplicated_count": len(results),
        "duplicates_removed": len(raw_log) + len(raw_vit) - len(results),
        "merge_policy": {
            "coordinate_order": "ZYX voxel indices in one CT frame",
            "coordinate_space": expected_space,
            "spacing_zyx_mm": spacing.tolist(),
            "center_distance_mm": float(center_distance_mm),
            "max_center_distance_mm": float(max_center_distance_mm),
            "min_overlap_iou": float(min_overlap_iou),
            "conflict_resolution": "Prefer ViTDet3D geometry; otherwise highest finite LoG response. Never average coordinates or compare unlike score scales.",
        },
    }
    return results, report


def _sample_index(center_zyx: Any, shape: Sequence[int]) -> tuple[tuple[int, int, int] | None, str | None]:
    try:
        center = _finite_vec(center_zyx, "center_zyx")
    except ValueError:
        return None, "missing_or_invalid_center_zyx"
    rounded = np.rint(center).astype(np.int64)
    if any(rounded[axis] < 0 or rounded[axis] >= int(shape[axis]) for axis in range(3)):
        return tuple(int(v) for v in rounded), "center_outside_ct_volume"
    return tuple(int(v) for v in rounded), None


def filter_detections_by_lung_mask(
    detections: Sequence[dict[str, Any]],
    lung_mask: np.ndarray,
    heart_mask: np.ndarray | None = None,
    *,
    min_bbox_lung_occupancy: float = 0.0,
    spacing_zyx_mm: Sequence[float] | None = None,
    expected_space: str | None = None,
    require_nonempty_heart_mask: bool = False,
    require_heart_mask_reviewed: bool = False,
    heart_mask_reviewed: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply the center-in-lung rule with explicit heart exclusion.

    Production/audit CLI callers set ``require_nonempty_heart_mask=True``. In
    that mode, a zero-voxel heart mask makes anatomical exclusion unverified,
    so this function fail-closes by accepting no candidates and marking every
    row ``heart_mask_empty_unverified``. This is deliberate: an empty mask is
    not evidence that the heart occupies no voxels.

    A valid center otherwise samples True in ``lung_mask & ~heart_mask``.
    Optional bbox occupancy is a stricter additional gate and defaults to 0.0
    to preserve the schema's center-in-mask rule. Masks must share CT geometry.
    """
    lung = np.asarray(lung_mask, dtype=bool)
    if lung.ndim != 3:
        raise ValueError(f"lung_mask must be 3-D ZYX; got shape {lung.shape}.")
    if heart_mask is None:
        heart = np.zeros_like(lung, dtype=bool)
    else:
        heart = np.asarray(heart_mask, dtype=bool)
        if heart.shape != lung.shape:
            raise ValueError(f"lung_mask/heart_mask shape mismatch: {lung.shape} vs {heart.shape}.")
        if heart.ndim != 3:
            raise ValueError("heart_mask must be 3-D ZYX.")
    if not 0.0 <= float(min_bbox_lung_occupancy) <= 1.0:
        raise ValueError("min_bbox_lung_occupancy must be between 0 and 1.")
    occupancy_spacing = np.ones(3, dtype=np.float64) if spacing_zyx_mm is None else _finite_vec(spacing_zyx_mm, "spacing_zyx_mm")
    if np.any(occupancy_spacing <= 0):
        raise ValueError("spacing_zyx_mm values must be positive.")
    valid_lung = lung & ~heart
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}

    heart_mask_found = bool(heart.any())
    block_reason = None
    block_status = None
    if require_nonempty_heart_mask and not heart_mask_found:
        block_reason = "heart_mask_empty_unverified"
        block_status = "blocked_empty_heart_mask"
    elif require_heart_mask_reviewed and not heart_mask_reviewed:
        block_reason = "heart_mask_review_required"
        block_status = "blocked_heart_mask_review_required"

    if block_reason is not None:
        # Fail closed before center-only lung membership can falsely imply that
        # cardiac anatomy was excluded. Preserve every candidate for audit.
        for original in detections:
            if not isinstance(original, dict):
                raise ValueError("Every detection to filter must be a dictionary.")
            row = copy.deepcopy(original)
            row["candidate_in_lung_mask"] = None
            row["candidate_in_heart_mask"] = None
            row["exclusion_reason"] = block_reason
            rejected.append(row)
        if rejected:
            reasons[block_reason] = len(rejected)
        summary = {
            "input_count": len(detections),
            "accepted_count": 0,
            "rejected_count": len(rejected),
            "rejection_counts": dict(sorted(reasons.items())),
            "center_inclusion_rule": "rounded center_zyx must be inside lung_mask and outside heart_mask",
            "min_bbox_lung_occupancy": float(min_bbox_lung_occupancy),
            "heart_mask_voxels": int(heart.sum()),
            "heart_mask_found": heart_mask_found,
            "heart_mask_required": bool(require_nonempty_heart_mask),
            "heart_mask_review_required": bool(require_heart_mask_reviewed),
            "heart_mask_reviewed": bool(heart_mask_reviewed),
            "gate_status": block_status,
            "lung_mask_voxels": int(lung.sum()),
            "effective_lung_mask_voxels": int(valid_lung.sum()),
            "shape_zyx": list(lung.shape),
            "coordinate_space": expected_space,
        }
        return accepted, rejected, summary

    for original in detections:
        if not isinstance(original, dict):
            raise ValueError("Every detection to filter must be a dictionary.")
        row = copy.deepcopy(original)
        space = row.get("space", row.get("source_space"))
        if space is not None and expected_space is not None and str(space) != expected_space:
            raise ValueError(f"Candidate {row.get('candidate_id', row.get('nodule_id'))}: space {space!r} != {expected_space!r}.")
        index, outside_reason = _sample_index(row.get("center_zyx"), lung.shape)
        reason = outside_reason
        if reason is None and index is not None and bool(heart[index]):
            reason = "center_inside_heart_mask"
        if reason is None and index is not None and not bool(valid_lung[index]):
            reason = "center_outside_lung_mask"

        occupancy: float | None = None
        if reason is None and float(min_bbox_lung_occupancy) > 0:
            box = _bbox(row, np.asarray(row["center_zyx"], dtype=float), occupancy_spacing)
            if box is None:
                reason = "bbox_missing_for_occupancy_check"
            else:
                lo = np.floor(box[0]).astype(int)
                hi = np.ceil(box[1]).astype(int)
                start = np.maximum(lo, 0)
                end = np.minimum(hi, np.asarray(lung.shape, dtype=int))
                if np.any(end <= start):
                    occupancy = 0.0
                else:
                    region = valid_lung[tuple(slice(int(start[a]), int(end[a])) for a in range(3))]
                    occupancy = float(region.mean()) if region.size else 0.0
                if occupancy < float(min_bbox_lung_occupancy):
                    reason = "bbox_lung_mask_occupancy_below_threshold"
        row["center_index_zyx_rounded"] = list(index) if index is not None else None
        row["center_inside_heart_mask"] = bool(index is not None and all(0 <= index[a] < lung.shape[a] for a in range(3)) and heart[index])
        row["center_inside_lung_mask"] = bool(index is not None and all(0 <= index[a] < lung.shape[a] for a in range(3)) and lung[index])
        row["candidate_in_lung_mask"] = reason is None
        row["lung_mask_occupancy"] = occupancy
        row["exclusion_reason"] = reason
        if reason is None:
            accepted.append(row)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
            rejected.append(row)
    summary = {
        "input_count": len(detections),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "rejection_counts": dict(sorted(reasons.items())),
        "center_inclusion_rule": "rounded center_zyx must be inside lung_mask and outside heart_mask",
        "min_bbox_lung_occupancy": float(min_bbox_lung_occupancy),
        "heart_mask_voxels": int(heart.sum()),
        "heart_mask_found": heart_mask_found,
        "heart_mask_required": bool(require_nonempty_heart_mask),
        "heart_mask_review_required": bool(require_heart_mask_reviewed),
        "heart_mask_reviewed": bool(heart_mask_reviewed),
        "gate_status": "passed" if ((heart_mask_found or not require_nonempty_heart_mask) and (heart_mask_reviewed or not require_heart_mask_reviewed)) else "blocked_heart_mask_review_required",
        "lung_mask_voxels": int(lung.sum()),
        "effective_lung_mask_voxels": int(valid_lung.sum()),
        "shape_zyx": list(lung.shape),
        "coordinate_space": expected_space,
    }
    return accepted, rejected, summary


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-dir", type=Path, required=True,
                        help="Stage 01 patient output directory containing volume_hu_cropped.npy, masks and meta.json")
    parser.add_argument("--schema", type=Path, default=Path(__file__).resolve().parents[2] / "meta_schema.json")
    parser.add_argument("--vitdet-candidates", type=Path, required=True, help="Stage 02 candidates.json")
    parser.add_argument("--log-candidates", type=Path, required=True, help="Stage 02 log_candidates_diagnostic.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--center-distance-mm", type=float, default=3.0)
    parser.add_argument("--max-center-distance-mm", type=float, default=8.0)
    parser.add_argument("--min-overlap-iou", type=float, default=0.25)
    parser.add_argument("--min-bbox-lung-occupancy", type=float, default=0.0)
    args = parser.parse_args()

    contract = load_canonical_contract(args.schema.resolve())
    patient_dir = args.patient_dir.resolve()
    ct_path = patient_dir / "volume_hu_cropped.npy"
    lung_path = patient_dir / "lung_mask.npy"
    heart_path = patient_dir / "heart_mask.npy"
    meta_path = patient_dir / "meta.json"
    for path in (ct_path, lung_path, heart_path, meta_path, args.vitdet_candidates, args.log_candidates):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")
    ct = np.load(ct_path, mmap_mode="r", allow_pickle=False)
    lung = np.asarray(np.load(lung_path, mmap_mode="r", allow_pickle=False), dtype=bool)
    heart = np.asarray(np.load(heart_path, mmap_mode="r", allow_pickle=False), dtype=bool)
    if ct.ndim != 3 or lung.shape != ct.shape or heart.shape != ct.shape:
        raise ValueError(f"CT/mask geometry mismatch: CT={ct.shape}, lung={lung.shape}, heart={heart.shape}")
    meta = read_json(meta_path)
    spacing = _finite_vec(meta.get("spacing_zyx_mm"), "meta.json spacing_zyx_mm")
    if np.any(spacing <= 0):
        raise ValueError("meta.json spacing_zyx_mm must contain positive values.")
    # Stage 01 establishes the geometry shared by volume and masks; shape checks
    # plus exact CT-frame metadata are used here. The utility cannot prove that
    # source DICOM geometry was correct if Stage 01 meta.json is itself wrong.
    vit = unwrap_records(read_json(args.vitdet_candidates), path=str(args.vitdet_candidates))
    log = unwrap_records(read_json(args.log_candidates), path=str(args.log_candidates))
    merged, merge_report = merge_detections(
        log, vit, spacing_zyx_mm=spacing,
        center_distance_mm=args.center_distance_mm,
        max_center_distance_mm=args.max_center_distance_mm,
        min_overlap_iou=args.min_overlap_iou,
        expected_space=contract["ct_space"],
    )
    accepted, rejected, filter_report = filter_detections_by_lung_mask(
        merged, lung, heart,
        min_bbox_lung_occupancy=args.min_bbox_lung_occupancy,
        spacing_zyx_mm=spacing,
        expected_space=contract["ct_space"],
        require_nonempty_heart_mask=True,
        require_heart_mask_reviewed=True,
        heart_mask_reviewed=(meta.get("heart_mask_exclusion_verified") is True),
    )
    output_dir = (args.output_dir or (patient_dir / "02_candidate_fusion_audit")).resolve()
    write_json(output_dir / "merged_candidates.json", {"meta_schema_version": contract["schema_version"], "candidates": merged})
    write_json(output_dir / "filtered_candidates.json", {"meta_schema_version": contract["schema_version"], "candidates": accepted})
    write_json(output_dir / "rejected_candidates.json", {"meta_schema_version": contract["schema_version"], "candidates": rejected})
    gate_status = str(filter_report.get("gate_status", "unknown"))
    heart_gate_blocked = gate_status != "passed"
    gate_messages = {
        "blocked_empty_heart_mask": "heart_mask.npy contains zero True voxels; heart exclusion is unverified and all candidates were rejected fail-closed.",
        "blocked_heart_mask_review_required": "The central-rod mask has not been explicitly approved after visual QC; all candidates were rejected fail-closed.",
    }
    report = {
        "status": gate_status if heart_gate_blocked else "completed_candidate_merge_and_lung_gate_only",
        "safe_to_use_downstream": False,
        "blockers": ([gate_messages.get(gate_status, "Anatomical candidate gate did not pass; all candidates were rejected fail-closed.")] if heart_gate_blocked else []) + [
            "This utility creates an audit/intermediate artifact, not the Stage 02 patch manifest."
        ],
        "meta_schema_version": contract["schema_version"],
        "patient_dir": str(patient_dir),
        "ct_shape_zyx": list(ct.shape),
        "spacing_zyx_mm": spacing.tolist(),
        "merge": merge_report,
        "filter": filter_report,
        "output_dir": str(output_dir),
        "downstream_inference_run": False,
        "note": "Never point Stage 03 at this audit JSON. Stage 02 patch extraction/manifest and schema interfaces must be explicitly integrated; candidates are not downstream-eligible from this utility.",
    }
    write_json(output_dir / "merge_filter_report.json", report)
    print(json.dumps({
        "status": report["status"],
        "safe_to_use_downstream": report["safe_to_use_downstream"],
        "blockers": report["blockers"],
        "log_count": merge_report["log_count"],
        "vitdet3d_count": merge_report["vitdet3d_count"],
        "union_count_before_deduplication": merge_report["union_count_before_deduplication"],
        "deduplicated_count": merge_report["deduplicated_count"],
        "accepted_after_lung_heart_filter": filter_report["accepted_count"],
        "rejected_after_lung_heart_filter": filter_report["rejected_count"],
        "output_dir": str(output_dir),
        "downstream_inference_run": False,
    }, indent=2))
    # Return non-zero for an empty heart mask so shell automation cannot treat
    # a zero-voxel mask as a successful anatomical exclusion gate. Reports are
    # still written for inspection before this return.
    return 2 if heart_gate_blocked else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
