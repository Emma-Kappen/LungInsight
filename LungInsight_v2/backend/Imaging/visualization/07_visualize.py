#!/usr/bin/env python3
"""Stage 07: interactive axial CT viewer and validated nodule QC report.

Reports are written before the GUI opens. Only candidates with an in-lung
center and a valid, aligned 3-D Stage 02 bounding box are included in the
visible localized list. Missing or invalid boxes are blocked and retained in
the separate audit CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

IMAGING_DIR = Path(__file__).resolve().parent
if IMAGING_DIR.name in {"stages", "inference", "visualization", "validation"}:
    IMAGING_DIR = IMAGING_DIR.parent
if str(IMAGING_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGING_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons, Slider
from scipy import ndimage

from pipeline_schema import get_required, load_schema, check_meta_schema_version, validate_stage_version
from pipeline_logging import add_verbose_argument, configure_console, pipeline_print as print, stage_complete
from html_reports import generate_patient_report

SCHEMA = load_schema(__file__)
STAGE01 = get_required(SCHEMA, "stage_01_hu_and_crop", expected_type=dict)
STAGE02 = get_required(SCHEMA, "stage_02_detect_and_patch", expected_type=dict)
STAGE03 = get_required(SCHEMA, "stage_03_nodule_classification", expected_type=dict)
STAGE04 = get_required(SCHEMA, "stage_04_malignancy_prediction", expected_type=dict)
STAGE06 = get_required(SCHEMA, "stage_06_full_ct_gradcam", expected_type=dict)
STAGE07 = get_required(SCHEMA, "stage_07_visualize", expected_type=dict)
PATHS = get_required(SCHEMA, "paths", expected_type=dict)
STAGE07_PATHS = get_required(PATHS, "stage_07", expected_type=dict)
ATTRIBUTES = tuple(get_required(SCHEMA, "attribute_contracts.stage03_ranked_attributes", expected_type=list))
REPORT_JSON_NAME = str(get_required(SCHEMA, str(STAGE07["outputs"]["report_json_ref"]), expected_type=str))
REPORT_CSV_NAME = str(get_required(SCHEMA, str(STAGE07["outputs"]["report_csv_ref"]), expected_type=str))
EXCLUDED_CSV_NAME = str(get_required(SCHEMA, str(STAGE07["outputs"]["excluded_outside_lung_csv_ref"]), expected_type=str))
VIEWER_SCHEMA = get_required(STAGE07, "viewer", expected_type=dict)
DISPLAY_WINDOW = get_required(SCHEMA, str(STAGE07["display_window_ref"]), expected_type=dict)
V2_ROOT = IMAGING_DIR.parent.parent
DEFAULT_OUTPUT_ROOT = V2_ROOT / get_required(SCHEMA, "paths.output_root")


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}, line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    except OSError as exc:
        raise OSError(f"Could not read {path}: {exc}") from exc


def unwrap_records(payload: Any, path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, list):
        records, metadata = payload, {}
    elif isinstance(payload, dict):
        records = payload.get("predictions")
        metadata = payload
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a list under 'predictions'.")
        check_meta_schema_version(payload, SCHEMA, context=str(path), require=False)
    else:
        raise ValueError(f"{path}: expected a JSON list or object with 'predictions'.")
    if any(not isinstance(row, dict) for row in records):
        raise ValueError(f"{path}: every prediction record must be a JSON object.")
    return records, metadata


def rating_value(item: Any) -> float | None:
    if isinstance(item, dict):
        item = item.get("expected_rating", item.get("rating_original_scale", item.get("rating_rounded")))
    try:
        value = float(item)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def build_score_lookup(payload: Any, path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records, _ = unwrap_records(payload, path)
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        candidate_id = str(record.get("candidate_id"))
        try:
            patch_id = int(record.get("patch_id"))
        except (TypeError, ValueError):
            continue
        lookup[(candidate_id, patch_id)] = record
    return lookup


def sample_mask(mask: np.ndarray, center_zyx: Any) -> tuple[bool, tuple[int, int, int] | None, str | None]:
    try:
        center = np.asarray(center_zyx, dtype=np.float64)
    except (TypeError, ValueError):
        return False, None, "missing_or_non_numeric_center_zyx"
    if center.shape != (3,) or not np.isfinite(center).all():
        return False, None, "invalid_center_zyx"
    index = tuple(int(v) for v in np.rint(center).astype(np.int64))
    if any(v < 0 or v >= mask.shape[axis] for axis, v in enumerate(index)):
        return False, index, "center_outside_ct_volume"
    if not bool(mask[index]):
        return False, index, "center_outside_lung_mask"
    return True, index, None


def _score_row(
    prediction: dict[str, Any], score_lookup: dict[tuple[str, int], dict[str, Any]],
    stage06_record: dict[str, Any], *, source: str,
) -> dict[str, Any]:
    candidate_id = str(prediction.get("candidate_id"))
    try:
        patch_id = int(prediction.get("patch_id"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Prediction has invalid patch_id: {prediction.get('patch_id')!r}") from exc
    score = score_lookup.get((candidate_id, patch_id), {})
    if not stage06_record or str(stage06_record.get("candidate_id")) != candidate_id or int(stage06_record.get("patch_id", -1)) != patch_id:
        raise ValueError(f"Stage 06 record does not match candidate/patch identity {(candidate_id, patch_id)}.")
    attrs = prediction.get("attributes")
    if not isinstance(attrs, dict):
        attrs = {}
    center = prediction.get("center_zyx")
    center_lps = prediction.get("center_lps_mm")
    anatomy = prediction.get("anatomical_location")
    if not isinstance(anatomy, dict):
        anatomy = {}
    row: dict[str, Any] = {
        "patient_id": prediction.get("patient_id"),
        "detection_source": source,
        "source": stage06_record.get("detector", {}).get("source", source),
        "source_members": stage06_record.get("detector", {}).get("source_members", []),
        "vitdet_score": stage06_record.get("detector", {}).get("vitdet_score"),
        "purpose": stage06_record["purpose"],
        "patch_id": patch_id,
        "candidate_id": prediction.get("candidate_id"),
        "candidate_in_lung_mask": True,
        "center_z_voxel": float(center[0]) if isinstance(center, (list, tuple)) and len(center) == 3 else None,
        "center_y_voxel": float(center[1]) if isinstance(center, (list, tuple)) and len(center) == 3 else None,
        "center_x_voxel": float(center[2]) if isinstance(center, (list, tuple)) and len(center) == 3 else None,
        "center_lps_x_mm": center_lps[0] if isinstance(center_lps, (list, tuple)) and len(center_lps) == 3 else None,
        "center_lps_y_mm": center_lps[1] if isinstance(center_lps, (list, tuple)) and len(center_lps) == 3 else None,
        "center_lps_z_mm": center_lps[2] if isinstance(center_lps, (list, tuple)) and len(center_lps) == 3 else None,
        "detector_score": prediction.get("detector_score"),
        "brock_probability": (score.get("selected_prediction") or {}).get("probability"),
        "brock_status": (score.get("selected_prediction") or {}).get("status", "unavailable"),
        "center_inside_lung_mask": True,
        "center_inside_air_mask": None,
        "center_inside_heart_mask": anatomy.get("center_inside_heart_mask"),
        "location_primary": anatomy.get("primary"),
        "location_region": anatomy.get("region"),
        "location_compartment": anatomy.get("compartment"),
        "laterality": anatomy.get("laterality"),
        "vertical_region": anatomy.get("vertical_region"),
        "wall_relationship": anatomy.get("wall_relationship"),
        "distance_to_lung_mask_mm": anatomy.get("distance_to_lung_mask_mm"),
        "airway_type": anatomy.get("airway_type"),
        "airway_proximity_mm": anatomy.get("airway_proximity_mm"),
        "anatomy_location_method": anatomy.get("method", "derived_from_stage01_masks_fallback"),
        "stable_nodule_id": stage06_record["nodule_id"],
        "bbox_start_zyx": stage06_record["geometry"]["bbox_start_zyx"],
        "bbox_end_zyx_exclusive": stage06_record["geometry"]["bbox_end_zyx_exclusive"],
        "stage06_record": stage06_record,
    }
    for name in ATTRIBUTES:
        row[name] = rating_value(attrs.get(name))
    return row


def classify_fallback_location(
    row: dict[str, Any], index: tuple[int, int, int], *, lung_distance_mm: float,
    anatomy: dict[str, Any], air_mask: np.ndarray, heart_mask: np.ndarray,
) -> None:
    """Populate conservative location labels only if Stage 02 did not provide them."""
    z, y, x = index
    row["center_inside_air_mask"] = bool(air_mask[index])
    row["center_inside_heart_mask"] = bool(heart_mask[index])
    if anatomy and anatomy.get("primary"):
        for field, key in (
            ("location_primary", "primary"), ("location_region", "region"),
            ("location_compartment", "compartment"), ("laterality", "laterality"),
            ("vertical_region", "vertical_region"), ("wall_relationship", "wall_relationship"),
            ("distance_to_lung_mask_mm", "distance_to_lung_mask_mm"),
            ("airway_type", "airway_type"), ("airway_proximity_mm", "airway_proximity_mm"),
        ):
            if anatomy.get(key) is not None:
                row[field] = anatomy[key]
        row["anatomy_location_method"] = anatomy.get("method", "stage02_heuristic")
        return
    if lung_distance_mm <= float(get_required(SCHEMA, "stage_02_detect_and_patch.anatomical_localization.default_parameters.wall_distance_mm")):
        row["location_primary"] = "lung_wall"
        row["location_region"] = "lung_wall"
        row["location_compartment"] = "lung_wall"
        row["wall_relationship"] = "wall_adjacent"
    else:
        row["location_primary"] = "lung_parenchyma"
        row["location_region"] = "lung_parenchyma"
        row["location_compartment"] = "parenchyma"
        row["wall_relationship"] = "not_wall_adjacent"
    row["laterality"] = row.get("laterality") or "unknown"
    row["vertical_region"] = row.get("vertical_region") or "unknown"
    row["distance_to_lung_mask_mm"] = float(lung_distance_mm)
    row["airway_type"] = "unclassified_without_stage02_location"
    row["airway_proximity_mm"] = None


def validate_localization_bbox(geometry: Any, ct_shape: tuple[int, int, int]) -> tuple[bool, str | None]:
    if not isinstance(geometry, dict):
        return False, "missing_geometry"
    try:
        start = np.asarray(geometry["bbox_start_zyx"], dtype=np.float64)
        end = np.asarray(geometry["bbox_end_zyx_exclusive"], dtype=np.float64)
        center = np.asarray(geometry["center_zyx"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False, "missing_or_non_numeric_geometry"
    if (
        start.shape != (3,)
        or end.shape != (3,)
        or center.shape != (3,)
        or not np.isfinite(start).all()
        or not np.isfinite(end).all()
        or not np.isfinite(center).all()
    ):
        return False, "bbox_coordinates_not_finite_or_three_dimensional"
    if np.any(end <= start):
        return False, "bbox_has_non_positive_extent"
    if np.any(start != np.rint(start)) or np.any(end != np.rint(end)):
        return False, "bbox_coordinates_are_not_voxel_boundaries"
    if np.any(center < start) or np.any(center >= end):
        return False, "center_outside_bbox"
    shape = np.asarray(ct_shape, dtype=np.float64)
    if np.any(start < 0) or np.any(end > shape):
        return False, "bbox_outside_displayed_ct"
    frame = geometry.get("space", geometry.get("coordinate_frame", "stage01_final_cropped_ct"))
    if frame != "stage01_final_cropped_ct":
        return False, f"bbox_coordinate_frame_mismatch:{frame}"
    return True, None


def load_candidate_rows(patient_root: Path, lung_mask: np.ndarray, air_mask: np.ndarray,
                       heart_mask: np.ndarray, spacing_zyx: np.ndarray):
    stage3_dir = patient_root / get_required(PATHS, "stage_03.directory")
    stage4_dir = patient_root / get_required(PATHS, "stage_04.directory")
    primary_path = stage3_dir / get_required(STAGE03, "outputs.predictions_json")
    primary_mal_path = stage4_dir / get_required(PATHS, "stage_04.output_json")
    stage06_paths = get_required(PATHS, "stage_06", expected_type=dict)
    primary_stage06_path = patient_root / stage06_paths["directory"] / STAGE06["record_filename"]
    for path in (primary_path, primary_mal_path, primary_stage06_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required fused-pipeline report input is missing: {path}. Run Stages 03 through 06 first."
            )
    primary_records, _ = unwrap_records(read_json(primary_path), primary_path)
    primary_scores = build_score_lookup(read_json(primary_mal_path), primary_mal_path)
    stage06_payload = read_json(primary_stage06_path)
    validate_stage_version(
        stage06_payload.get("schema_version"),
        STAGE06["record_schema_version"],
        context="Stage 06 fused records",
    )
    stage06_records = stage06_payload.get("records")
    if not isinstance(stage06_records, list) or any(not isinstance(record, dict) for record in stage06_records):
        raise ValueError(f"{primary_stage06_path}: Stage 06 records must be a list of objects.")
    stage06_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for record in stage06_records:
        try:
            key = (str(record["candidate_id"]), int(record["patch_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{primary_stage06_path}: invalid stable candidate/patch identity.") from exc
        if record.get("stable_nodule_id", record.get("nodule_id")) != key[0]:
            raise ValueError(f"{primary_stage06_path}: stable_nodule_id mismatch for {key}.")
        if key in stage06_lookup:
            raise ValueError(f"{primary_stage06_path}: duplicate candidate/patch identity {key}.")
        stage06_lookup[key] = record
    if int(stage06_payload.get("accepted_record_count", -1)) != len(stage06_lookup):
        raise ValueError(f"{primary_stage06_path}: accepted_record_count does not reconcile with records.")

    prediction_lookup = {
        (str(row.get("candidate_id")), int(row["patch_id"])): row
        for row in primary_records
        if row.get("candidate_id") is not None and row.get("patch_id") is not None
    }
    if set(prediction_lookup) != set(stage06_lookup):
        raise ValueError(
            "Stage 03 and Stage 06 fused candidate identities differ; "
            f"missing_in_stage06={sorted(set(prediction_lookup) - set(stage06_lookup))[:10]}, "
            f"missing_in_stage03={sorted(set(stage06_lookup) - set(prediction_lookup))[:10]}."
        )
    all_source_rows = []
    accepted_rows = []
    excluded_rows = []
    lung_distance_inside = ndimage.distance_transform_edt(lung_mask.astype(bool), sampling=tuple(float(v) for v in spacing_zyx))

    for key, stage06_record in stage06_lookup.items():
        prediction = prediction_lookup[key]
        center = prediction.get("center_zyx")
        inside, index, reason = sample_mask(lung_mask, center)
        center_list = list(index) if index is not None else None
        dist = float(lung_distance_inside[index]) if inside and index is not None else None
        source = str(stage06_record.get("detector", {}).get("source", ""))
        row = _score_row(
            prediction, primary_scores, stage06_record,
            source=source,
        )
        row["stage06_record_path"] = str(primary_stage06_path)
        row["candidate_in_lung_mask"] = bool(inside)
        row["center_inside_lung_mask"] = bool(inside)
        row["center_index_zyx_rounded"] = json.dumps(center_list)
        row["exclusion_reason"] = reason
        bbox_valid, bbox_reason = validate_localization_bbox(
            stage06_record.get("geometry"), tuple(int(v) for v in lung_mask.shape)
        )
        row["localization_status"] = "VERIFIED" if bbox_valid else "BLOCKED"
        row["bbox_validation_reason"] = bbox_reason
        anatomy = prediction.get("anatomical_location") if isinstance(prediction.get("anatomical_location"), dict) else {}
        inside_heart = bool(heart_mask[index]) if index is not None else False
        row["center_inside_heart_mask"] = inside_heart
        if inside and not inside_heart and index is not None and bbox_valid:
            classify_fallback_location(row, index, lung_distance_mm=dist or 0.0, anatomy=anatomy,
                                       air_mask=air_mask, heart_mask=heart_mask)
            accepted_rows.append(row)
        else:
            if index is not None:
                row["center_inside_air_mask"] = bool(air_mask[index])
            if inside_heart:
                row["exclusion_reason"] = "center_inside_heart_mask"
                fallback_location = "heart_mask"
            elif not inside:
                fallback_location = "outside_lung"
            else:
                fallback_location = "localization_blocked"
            row["location_primary"] = anatomy.get("primary", fallback_location)
            row["location_region"] = anatomy.get("region", fallback_location)
            row["location_compartment"] = anatomy.get("compartment", fallback_location)
            row["laterality"] = anatomy.get("laterality", "unknown")
            row["vertical_region"] = anatomy.get("vertical_region", "unknown")
            row["wall_relationship"] = anatomy.get("wall_relationship", fallback_location)
            row["distance_to_lung_mask_mm"] = anatomy.get("distance_to_lung_mask_mm")
            row["airway_type"] = anatomy.get("airway_type")
            row["airway_proximity_mm"] = anatomy.get("airway_proximity_mm")
            if not bbox_valid and inside:
                row["exclusion_reason"] = f"missing_or_invalid_bbox:{bbox_reason}"
            excluded_rows.append(row)
        all_source_rows.append(row)
    accepted_rows.sort(
        key=lambda row: (
            -float(row["brock_probability"])
            if isinstance(row.get("brock_probability"), (int, float))
            else float("inf"),
            str(row["stable_nodule_id"]),
        )
    )
    return accepted_rows, excluded_rows, all_source_rows, {
        "primary_predictions_json": str(primary_path.resolve()),
        "primary_malignancy_json": str(primary_mal_path.resolve()),
        "primary_stage06_records": str(primary_stage06_path.resolve()),
    }


def render_nodule_visualizations(
    ct: np.ndarray,
    rows: list[dict[str, Any]],
    patient_root: Path,
    output_dir: Path,
    patient_id: str,
    level: float,
    width: float,
) -> list[str]:
    """Write a headless, identity-bound CT/CAM/Brock summary for each in-lung record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for row in rows:
        record = row["stage06_record"]
        stable_nodule_id = str(
            row.get("stable_nodule_id")
            or record.get("stable_nodule_id")
            or record.get("nodule_id")
            or record.get("candidate_id")
            or ""
        )
        if not stable_nodule_id:
            raise ValueError("Accepted Stage 06 record lacks a stable nodule ID.")
        geometry = record["geometry"]
        center = np.rint(np.asarray(geometry["center_zyx"], dtype=np.float64)).astype(int)
        z, y, x = (int(value) for value in center)
        if not (0 <= z < ct.shape[0] and 0 <= y < ct.shape[1] and 0 <= x < ct.shape[2]):
            raise ValueError(f"Stage 06 record {record['nodule_id']} has an out-of-volume center.")
        bbox_start = np.asarray(geometry["bbox_start_zyx"], dtype=np.float64)
        bbox_end = np.asarray(geometry["bbox_end_zyx_exclusive"], dtype=np.float64)
        if bbox_start.shape != (3,) or bbox_end.shape != (3,) or not np.isfinite(bbox_start).all() or not np.isfinite(bbox_end).all():
            raise ValueError(f"Stage 06 record {record['nodule_id']} has invalid bbox geometry.")
        crop_start = np.asarray(geometry["crop_start_zyx"], dtype=np.int64)
        bbox_valid, bbox_reason = validate_localization_bbox(geometry, tuple(int(v) for v in ct.shape))
        if not bbox_valid:
            raise ValueError(
                f"Stage 06 record {record['nodule_id']} has invalid display geometry: {bbox_reason}."
            )

        fig = plt.figure(figsize=(17, 12))
        grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.7])
        ct_ax = fig.add_subplot(grid[0, 0])
        ct_ax.imshow(window_ct(np.asarray(ct[z]), level, width), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if bbox_start[0] <= z < bbox_end[0]:
            left = max(float(bbox_start[2]), -0.5)
            top = max(float(bbox_start[1]), -0.5)
            right = min(float(bbox_end[2]), ct.shape[2] - 0.5)
            bottom = min(float(bbox_end[1]), ct.shape[1] - 0.5)
            if right > left and bottom > top:
                ct_ax.add_patch(Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor="yellow", linewidth=1.5))
        ct_ax.scatter([x], [y], marker="x", s=70, color="cyan")
        ct_ax.annotate(
            stable_nodule_id,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.15", "fc": "black", "alpha": 0.65},
        )
        ct_ax.set_title(
            f"{stable_nodule_id} | CT z={z} | "
            f"diameter={geometry.get('diameter_mm', 'unavailable')} mm"
        )
        ct_ax.set_xlim(-0.5, ct.shape[2] - 0.5)
        ct_ax.set_ylim(ct.shape[1] - 0.5, -0.5)

        cam_refs = record["explanations"]["gradcam"]
        cam_positions = ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0))
        for attribute, (grid_row, grid_column) in zip(ATTRIBUTES, cam_positions):
            ax = fig.add_subplot(grid[grid_row, grid_column])
            ax.imshow(window_ct(np.asarray(ct[z]), level, width), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            cam_ref = cam_refs.get(attribute, {})
            scaled_reference = cam_ref.get("scaled_path")
            raw_reference = cam_ref.get("raw_signed_path")
            reference = scaled_reference or raw_reference
            if reference:
                cam_path = Path(reference)
                if not cam_path.is_absolute():
                    cam_path = patient_root / cam_path
                if not cam_path.is_file():
                    raise FileNotFoundError(
                        f"Stage 06 record {record['nodule_id']} references missing CAM: {cam_path}."
                    )
                cam = np.asarray(np.load(cam_path, allow_pickle=False), dtype=np.float32)
                patch_size = tuple(int(v) for v in geometry["patch_size_vox_zyx"])
                if tuple(cam.shape) != patch_size or not np.isfinite(cam).all():
                    raise ValueError(f"Invalid CAM shape/values for {record['nodule_id']} {attribute}: {cam.shape}.")
                patch_z = z - int(crop_start[0])
                if 0 <= patch_z < cam.shape[0]:
                    cam_plane = cam[patch_z]
                    if scaled_reference:
                        if float(cam_plane.min()) < -1e-4 or float(cam_plane.max()) > 1.0001:
                            raise ValueError(f"Scaled CAM values must be in [0,1]: {cam_path}.")
                        ax.imshow(
                            cam_plane, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.48,
                            extent=(crop_start[2] - 0.5, crop_start[2] + cam.shape[2] - 0.5,
                                    crop_start[1] + cam.shape[1] - 0.5, crop_start[1] - 0.5),
                            interpolation="nearest",
                        )
                    else:
                        limit = max(float(np.max(np.abs(cam_plane))), 1e-8)
                        ax.imshow(
                            cam_plane, cmap="coolwarm", vmin=-limit, vmax=limit, alpha=0.48,
                            extent=(crop_start[2] - 0.5, crop_start[2] + cam.shape[2] - 0.5,
                                    crop_start[1] + cam.shape[1] - 0.5, crop_start[1] - 0.5),
                            interpolation="nearest",
                        )
                ax.set_title(f"{attribute} CAM overlay")
            else:
                ax.text(0.5, 0.5, "CAM unavailable", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{attribute} CAM")
            ax.set_xlim(-0.5, ct.shape[2] - 0.5)
            ax.set_ylim(ct.shape[1] - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])

        brock_ax = fig.add_subplot(grid[2, 1:])
        contributions = record["malignancy"].get("brock", {}).get("feature_contributions") or []
        names = [str(item.get("feature")) for item in contributions if isinstance(item, dict)]
        values = [float(item.get("contribution_logit")) for item in contributions if isinstance(item, dict)]
        if names:
            brock_ax.barh(names, values, color=["#d95f02" if value >= 0 else "#1b9e77" for value in values])
            brock_ax.axvline(0.0, color="black", linewidth=0.8)
            brock_ax.set_title("Brock formula logit contributions")
        else:
            brock_ax.text(0.5, 0.5, "Brock explanation unavailable", ha="center", va="center", transform=brock_ax.transAxes)
        brock_ax.set_xlabel(f"Brock probability={row.get('brock_probability', 'unavailable')}")

        predicted_attributes = record.get("attributes")
        predicted_attributes = predicted_attributes if isinstance(predicted_attributes, dict) else {}
        classification_text = ", ".join(
            f"{name}={rating_value(predicted_attributes.get(name))}"
            for name in ATTRIBUTES
            if rating_value(predicted_attributes.get(name)) is not None
        ) or "classification unavailable"
        fig.suptitle(
            f"{patient_id} | {stable_nodule_id} | {row.get('detection_source', record.get('detection_source', 'unknown'))} | "
            f"diameter={geometry.get('diameter_mm', 'unavailable')} mm | "
            f"Brock risk={row.get('brock_probability', 'unavailable')} ({row.get('brock_status', 'unavailable')})\n"
            f"Predicted attributes (not diagnosis): {classification_text}",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        filename = f"nodule_{int(record['patch_id']):03d}_explanation.png"
        image_path = output_dir / filename
        fig.savefig(image_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        row["visualization_path"] = str(image_path.resolve())
        written.append(str(image_path.resolve()))
    return written


def csv_safe(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    declared = get_required(STAGE07, "csv_columns_order", expected_type=list)
    preferred: list[str] = []
    for column in declared:
        if column == "{attribute_columns}":
            preferred.extend(ATTRIBUTES)
        else:
            preferred.append(str(column))
    internal = {"stage06_record", "stage06_record_path"}
    extras = sorted({key for row in rows for key in row if key not in preferred and key not in internal})
    fields = preferred + extras
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_safe(row.get(key)) for key in fields})


def window_ct(slice_hu: np.ndarray, level: float, width: float) -> np.ndarray:
    lower, upper = level - width / 2.0, level + width / 2.0
    return np.clip((slice_hu.astype(np.float32) - lower) / (upper - lower), 0.0, 1.0)


def launch_viewer(ct: np.ndarray, lung_mask: np.ndarray, air_mask: np.ndarray, heart_mask: np.ndarray,
                  rows: list[dict[str, Any]], patient_id: str, level: float, width: float,
                  ) -> None:
    """Open slider-controlled axial viewer; only in-lung rows are passed here."""
    if ct.ndim != 3:
        raise ValueError(f"Expected 3D CT volume, got shape={ct.shape}")
    z_count = int(ct.shape[0])
    valid_rows = []
    for row in rows:
        z, y, x = (int(round(float(row[k]))) for k in ("center_z_voxel", "center_y_voxel", "center_x_voxel"))
        if 0 <= z < z_count and 0 <= y < ct.shape[1] and 0 <= x < ct.shape[2] and bool(lung_mask[z, y, x]):
            valid_rows.append((row, (z, y, x)))
    initial_z = valid_rows[0][1][0] if valid_rows else z_count // 2

    fig, ax = plt.subplots(figsize=(11, 8))
    plt.subplots_adjust(bottom=0.24, left=0.28)
    slider_ax = fig.add_axes([0.32, 0.11, 0.62, 0.035])
    slider = Slider(slider_ax, "Axial slice z", 0, max(z_count - 1, 1), valinit=initial_z, valstep=1)
    check_ax = fig.add_axes([0.02, 0.035, 0.22, 0.14])
    default_mask_overlays = get_required(VIEWER_SCHEMA, "default_mask_overlays", expected_type=dict)
    labels = ["Lung mask", "Air mask", "Heart mask"]
    initial_mask_states = [bool(default_mask_overlays["lung_mask"]), bool(default_mask_overlays["air_mask"]), bool(default_mask_overlays["heart_mask"])]
    mask_checks = CheckButtons(check_ax, labels, initial_mask_states)
    mask_states = dict(zip(labels, initial_mask_states))

    def redraw(value: Any = None) -> None:
        z = int(round(slider.val))
        ax.clear()
        ax.imshow(window_ct(np.asarray(ct[z]), level, width), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if mask_states["Lung mask"] and lung_mask[z].any():
            ax.contour(lung_mask[z].astype(np.uint8), levels=[0.5], colors=["lime"], linewidths=0.6)
        if mask_states["Air mask"] and air_mask[z].any():
            ax.contour(air_mask[z].astype(np.uint8), levels=[0.5], colors=["deepskyblue"], linewidths=0.45)
        if mask_states["Heart mask"] and heart_mask[z].any():
            ax.contour(heart_mask[z].astype(np.uint8), levels=[0.5], colors=["gold"], linewidths=0.7)

        visible = []
        for row, idx in valid_rows:
            start = np.asarray(row["bbox_start_zyx"], dtype=np.int64)
            end = np.asarray(row["bbox_end_zyx_exclusive"], dtype=np.int64)
            if start[0] <= z < end[0]:
                visible.append((row, idx, start, end))
        for row, (center_z, y, x), start, end in visible:
            available = row.get("brock_status") == "available"
            source = row.get("detection_source")
            marker = str(VIEWER_SCHEMA["viTdet3d_marker"])
            color = str(VIEWER_SCHEMA["brock_available_color"]) if available else str(VIEWER_SCHEMA["brock_unavailable_color"])
            left = max(float(start[2]), -0.5)
            top = max(float(start[1]), -0.5)
            right = min(float(end[2]), ct.shape[2] - 0.5)
            bottom = min(float(end[1]), ct.shape[1] - 0.5)
            if right <= left or bottom <= top:
                raise ValueError(f"Accepted candidate {row.get('stable_nodule_id')} has no visible bbox on CT slice {z}.")
            ax.add_patch(Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor="yellow", linewidth=1.2))
            if center_z == z:
                ax.scatter([x], [y], marker=marker, s=75, c=[color], edgecolors="white", linewidths=0.8)
            attributes = row["stage06_record"].get("attributes")
            attributes = attributes if isinstance(attributes, dict) else {}
            classification = ", ".join(
                f"{name}={rating_value(attributes.get(name))}"
                for name in ATTRIBUTES
                if rating_value(attributes.get(name)) is not None
            ) or "classification unavailable"
            label = str(row.get("stable_nodule_id") or "")
            probability = row.get("brock_probability")
            geometry = row["stage06_record"].get("geometry", {})
            label += (
                f"\nDiameter {geometry.get('diameter_mm', 'unknown')} mm; "
                f"volume {geometry.get('volume_mm3', 'unavailable')} mm3 | "
                f"Brock {probability if probability is not None else 'unavailable'} ({row.get('brock_status')})"
                f"\n{classification}"
            )
            ax.annotate(label, (left, top), xytext=(3, 3), textcoords="offset points", fontsize=5,
                        color="white", bbox={"boxstyle": "round,pad=0.15", "fc": "black", "alpha": 0.6})
        ax.set_title(f"{patient_id} — axial z={z}/{z_count - 1} | in-lung candidates on slice: {len(visible)}")
        ax.set_xlabel("X voxel")
        ax.set_ylabel("Y voxel")
        ax.set_xlim(-0.5, ct.shape[2] - 0.5)
        ax.set_ylim(ct.shape[1] - 0.5, -0.5)
        legend_items = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markeredgecolor="white", label="ViTDet3D: Brock available", markersize=7),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="cyan", markeredgecolor="white", label="ViTDet3D: Brock unavailable", markersize=7),
        ]
        ax.legend(handles=legend_items, loc="upper right", fontsize=7)
        fig.canvas.draw_idle()

    def toggle_mask(label: str) -> None:
        mask_states[label] = not mask_states[label]
        redraw()

    slider.on_changed(redraw)
    mask_checks.on_clicked(toggle_mask)
    redraw()
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id", default=str(get_required(SCHEMA, "defaults.patient_id")))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for the JSON/CSV report; defaults from meta_schema.json.")
    parser.add_argument("--gui", dest="open_gui", action="store_true", help="Open the optional interactive slider viewer after writing reports.")
    parser.add_argument("--no-gui", dest="open_gui", action="store_false", help="Write reports without opening the slider viewer.")
    parser.set_defaults(open_gui=False)
    add_verbose_argument(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_console(args.verbose)
    if args.open_gui:
        try:
            plt.switch_backend("TkAgg")
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("The optional --gui viewer requires a working Tk GUI backend.") from exc
    patient_root = DEFAULT_OUTPUT_ROOT / args.patient_id
    output_dir = args.output_dir.resolve() if args.output_dir else patient_root / STAGE07_PATHS["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = get_required(STAGE01, "filenames", expected_type=dict)
    ct_path = patient_root / filenames["volume_hu_cropped.npy"]
    lung_path = patient_root / filenames["lung_mask.npy"]
    air_path = patient_root / filenames["air_mask.npy"]
    heart_path = patient_root / filenames["heart_mask.npy"]
    metadata_path = patient_root / filenames["meta.json"]
    for path in (ct_path, lung_path, air_path, heart_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Stage 07 input is missing: {path}. Re-run Stage 01 or restore the required output.")
    ct = np.load(ct_path, mmap_mode="r", allow_pickle=False)
    lung_mask = np.asarray(np.load(lung_path, mmap_mode="r", allow_pickle=False), dtype=bool)
    air_mask = np.asarray(np.load(air_path, mmap_mode="r", allow_pickle=False), dtype=bool)
    heart_mask = np.asarray(np.load(heart_path, mmap_mode="r", allow_pickle=False), dtype=bool)
    if ct.ndim != 3 or any(mask.shape != ct.shape for mask in (lung_mask, air_mask, heart_mask)):
        raise ValueError(f"CT/mask shape mismatch: CT={ct.shape}, lung={lung_mask.shape}, air={air_mask.shape}, heart={heart_mask.shape}.")
    meta = read_json(metadata_path)
    spacing = np.asarray(meta.get("spacing_zyx_mm"), dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"{metadata_path}: invalid/missing spacing_zyx_mm required for anatomical location report.")

    accepted, excluded, _all_rows, input_paths = load_candidate_rows(
        patient_root, lung_mask, air_mask, heart_mask, spacing
    )
    visualization_root = output_dir / str(STAGE07["outputs"]["nodule_visualization_directory"])
    visualization_paths = render_nodule_visualizations(
        ct,
        accepted,
        patient_root,
        visualization_root,
        args.patient_id,
        float(DISPLAY_WINDOW["level"]),
        float(DISPLAY_WINDOW["width"]),
    )
    report_json = output_dir / REPORT_JSON_NAME
    report_csv = output_dir / REPORT_CSV_NAME
    excluded_csv = output_dir / EXCLUDED_CSV_NAME
    write_csv(report_csv, accepted)
    write_csv(excluded_csv, excluded)

    def report_reference(value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        path = Path(value)
        if path.is_absolute():
            try:
                return path.resolve().relative_to(patient_root.resolve()).as_posix()
            except ValueError:
                return None
        return path.as_posix()

    eligible_nodules = []
    for row in accepted:
        record = row["stage06_record"]
        geometry = record["geometry"]
        attributes = record.get("attributes")
        attributes = attributes if isinstance(attributes, dict) else {}
        brock = record.get("malignancy", {}).get("brock", {})
        brock = brock if isinstance(brock, dict) else {}
        classification_attributes = {}
        for name in ATTRIBUTES:
            item = attributes.get(name)
            value = rating_value(item)
            classification_attributes[name] = {
                "value": value,
                "status": "predicted" if value is not None else "unavailable",
                "semantics": item.get("score_semantics") if isinstance(item, dict) else None,
                "model_version": item.get("model_version") if isinstance(item, dict) else None,
            }
        gradcam_refs = {}
        explanation = record.get("explanations")
        explanation = explanation if isinstance(explanation, dict) else {}
        gradcam = explanation.get("gradcam")
        if isinstance(gradcam, dict):
            for name, refs in gradcam.items():
                if not isinstance(refs, dict):
                    continue
                gradcam_refs[str(name)] = {
                    key: report_reference(refs.get(key))
                    for key in ("patch_cam_path", "scaled_path", "raw_signed_path", "ct_cam_path")
                    if refs.get(key)
                }
        eligible_nodules.append({
            "stable_nodule_id": row["stable_nodule_id"],
            "eligibility_status": "eligible_in_lung",
            "anatomical_location": {
                key: row.get(key)
                for key in (
                    "location_primary", "location_region", "location_compartment",
                    "laterality", "vertical_region", "wall_relationship",
                    "airway_type", "airway_proximity_mm",
                )
            },
            "centroid_voxel_zyx": geometry.get("center_zyx"),
            "centroid_lps_mm": geometry.get("center_lps_mm"),
            "bbox_voxel_zyx": {
                "start": geometry.get("bbox_start_zyx"),
                "end_exclusive": geometry.get("bbox_end_zyx_exclusive"),
                "coordinate_frame": geometry.get("source_space"),
            },
            "measurements": {
                "diameter_mm": geometry.get("diameter_mm"),
                "volume_mm3": geometry.get("volume_mm3"),
                "source": "detector_estimate",
            },
            "detection": {
                "source": row.get("source"),
                "source_members": row.get("source_members", []),
                "vitdet_score": row.get("vitdet_score"),
            },
            "classification": {
                "status": "available" if any(item["value"] is not None for item in classification_attributes.values()) else "unavailable",
                "source": "Stage 03 model predictions",
                "attributes": classification_attributes,
            },
            "brock_risk": {
                "status": brock.get("status", "unavailable"),
                "probability": brock.get("probability"),
                "missing_inputs": brock.get("features_missing", brock.get("missing_inputs", [])),
                "model_name": brock.get("model_name"),
                "model_version": brock.get("model_version"),
                "input_completeness": record.get("input_completeness"),
            },
            "quality_status": row.get("localization_status"),
            "explanation_refs": {"gradcam": gradcam_refs},
            "artifact_refs": {
                "stage06_record": report_reference(row.get("stage06_record_path")),
                "visualization": report_reference(row.get("visualization_path")),
            },
        })

    count_by_source: dict[str, dict[str, int]] = {}
    for source in ("ViTDet3D", "Merged"):
        source_rows = [row for row in accepted if row["detection_source"] == source]
        excluded_source_rows = [row for row in excluded if row["detection_source"] == source]
        count_by_source[source] = {
            "input_candidates": len([row for row in _all_rows if row["detection_source"] == source]),
            "accepted_center_inside_lung_mask": len(source_rows),
            "excluded_from_accepted_list": len(excluded_source_rows),
            "center_outside_lung_mask": sum(
                row.get("exclusion_reason") in {"center_outside_lung_mask", "center_outside_ct_volume"}
                for row in excluded_source_rows
            ),
            "center_inside_heart_mask": sum(
                row.get("exclusion_reason") == "center_inside_heart_mask"
                for row in excluded_source_rows
            ),
            "invalid_localization": sum(
                row.get("localization_status") == "BLOCKED"
                for row in excluded_source_rows
            ),
            "brock_available_count": sum(row["brock_status"] == "available" for row in source_rows),
        }
    summary = {
        "schema_version": STAGE07["report_schema_version"],
        "meta_schema_version": SCHEMA["schema_version"],
        "stage": "07_visualize",
        "status": "completed",
        "patient_id": args.patient_id,
        "created_from": {key: report_reference(value) for key, value in input_paths.items()},
        "ct_volume": report_reference(str(ct_path.resolve())),
        "lung_mask": report_reference(str(lung_path.resolve())),
        "ct_shape_zyx": list(ct.shape),
        "spacing_zyx_mm": spacing.tolist(),
        "candidate_center_filter": STAGE07["center_in_mask_rule"],
        "input_candidate_count": len(_all_rows),
        "nodule_count_center_inside_lung_mask": len(accepted),
        "nodule_count_excluded_from_display": len(excluded),
        "nodule_count_center_outside_lung_mask": sum(
            row.get("exclusion_reason") in {"center_outside_lung_mask", "center_outside_ct_volume"}
            for row in excluded
        ),
        "nodule_count_center_inside_heart_mask": sum(
            row.get("exclusion_reason") == "center_inside_heart_mask"
            for row in excluded
        ),
        "nodule_count_localization_blocked": sum(
            row.get("localization_status") == "BLOCKED" for row in excluded
        ),
        "localization_gate": "GATE_06_TO_07",
        "localization_gate_status": (
            "passed"
            if not any(row.get("localization_status") == "BLOCKED" for row in excluded)
            else "passed_with_blocked_candidates"
        ),
        "brock_available_count_inside_lung_mask": sum(row["brock_status"] == "available" for row in accepted),
        "counts_by_source": count_by_source,
        "accepted_csv": report_reference(str(report_csv.resolve())),
        "excluded_outside_lung_csv": report_reference(str(excluded_csv.resolve())),
        "visualization_directory": report_reference(str(visualization_root.resolve())),
        "visualization_count": len(visualization_paths),
        "eligible_nodules": eligible_nodules,
        "artifact_refs": {
            "report_json": report_reference(str(report_json.resolve())),
            "accepted_csv": report_reference(str(report_csv.resolve())),
            "excluded_csv": report_reference(str(excluded_csv.resolve())),
            "visualization_directory": report_reference(str(visualization_root.resolve())),
            "html_report": f"../../reports/{args.patient_id}/index.html",
        },
        "warnings": [
            f"{row['stable_nodule_id']}: Brock status is {row['brock_status']}."
            for row in accepted
            if row["brock_status"] != "available"
        ],
        "errors": [],
        "provenance": {
            "coordinate_frame": "stage01_final_cropped_ct",
            "coordinate_order": "ZYX",
            "coordinate_system": str(meta.get("coordinate_system", "unknown")),
            "meta_schema_version": meta.get("meta_schema_version"),
            "spacing_zyx_mm": spacing.tolist(),
            "geometry_preserved_from": "Stage 01 volume_hu_cropped.npy",
        },
        "limitations": list(STAGE07["clinical_limitations"]),
    }
    declared_report_fields = set(get_required(STAGE07, "report_fields", expected_type=list))
    if set(summary) != declared_report_fields:
        missing = sorted(declared_report_fields - set(summary))
        extra = sorted(set(summary) - declared_report_fields)
        raise RuntimeError(
            f"Stage 07 report/schema field mismatch. Missing={missing}; extra={extra}. "
            "Update meta_schema.json and this report writer together."
        )
    report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
    print(f"Report JSON: {report_json}")
    print(f"Accepted in-lung candidates CSV: {report_csv}")
    print(f"Excluded outside-lung candidates CSV: {excluded_csv}")
    print(f"Per-nodule CT/CAM/Brock visualizations: {len(visualization_paths)} in {visualization_root}")
    print(f"Candidates inspected: {len(_all_rows)}")
    print(f"Candidates accepted (rounded center inside lung_mask): {len(accepted)}")
    print(f"Candidates excluded (center outside lung_mask or CT): {len(excluded)}")
    print(f"Brock probabilities available within lung mask: {summary['brock_available_count_inside_lung_mask']}")
    html_report_dir = DEFAULT_OUTPUT_ROOT / "reports" / args.patient_id
    report_pages = generate_patient_report(
        patient_id=args.patient_id,
        patient_root=patient_root,
        report_root=html_report_dir,
        accepted=accepted,
        excluded=excluded,
        ct=np.asarray(ct),
        lung_mask=lung_mask,
        heart_mask=heart_mask,
        input_paths=input_paths,
    )
    print(f"HTML report: {html_report_dir / 'index.html'} ({len(report_pages)} pages)")

    if args.open_gui:
        launch_viewer(ct, lung_mask, air_mask, heart_mask, accepted, args.patient_id,
                      float(DISPLAY_WINDOW["level"]), float(DISPLAY_WINDOW["width"]),
                      )
    stage_complete(7, "07")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
