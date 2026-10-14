#!/usr/bin/env python3
"""Stage 07: interactive axial CT viewer and lung-mask-filtered nodule report.

Reports are written before the GUI opens. Only candidate centers that sample
True in Stage 01 lung_mask.npy are included in the visible accepted list and
nodule_report.csv. Rejected/outside centers are kept in a separate audit CSV.
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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons, Slider
from scipy import ndimage

from pipeline_schema import get_required, load_schema, check_meta_schema_version, validate_stage_version

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
    stage06_record: dict[str, Any], *, source: str, malignancy_threshold: int,
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
        "purpose": stage06_record["purpose"],
        "patch_id": patch_id,
        "candidate_id": prediction.get("candidate_id"),
        "diagnostic_id": prediction.get("diagnostic_id"),
        "candidate_in_lung_mask": True,
        "center_z_voxel": float(center[0]) if isinstance(center, (list, tuple)) and len(center) == 3 else None,
        "center_y_voxel": float(center[1]) if isinstance(center, (list, tuple)) and len(center) == 3 else None,
        "center_x_voxel": float(center[2]) if isinstance(center, (list, tuple)) and len(center) == 3 else None,
        "center_lps_x_mm": center_lps[0] if isinstance(center_lps, (list, tuple)) and len(center_lps) == 3 else None,
        "center_lps_y_mm": center_lps[1] if isinstance(center_lps, (list, tuple)) and len(center_lps) == 3 else None,
        "center_lps_z_mm": center_lps[2] if isinstance(center_lps, (list, tuple)) and len(center_lps) == 3 else None,
        "detector_score": prediction.get("detector_score"),
        "log_response": prediction.get("log_response"),
        "scale_mm": prediction.get("scale_mm"),
        "predicted_malignancy_rating_continuous": score.get("predicted_malignancy_rating_continuous"),
        "predicted_malignancy_rating_rounded": score.get("predicted_malignancy_rating_rounded"),
        "high_suspicion_rating_ge_threshold": False,
        "malignancy_threshold_used": malignancy_threshold,
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
    rounded = score.get("predicted_malignancy_rating_rounded")
    try:
        row["high_suspicion_rating_ge_threshold"] = int(rounded) >= malignancy_threshold
    except (TypeError, ValueError):
        pass
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


def load_candidate_rows(patient_root: Path, lung_mask: np.ndarray, air_mask: np.ndarray,
                       heart_mask: np.ndarray, spacing_zyx: np.ndarray, malignancy_threshold: int):
    stage3_dir = patient_root / get_required(PATHS, "stage_03.directory")
    stage4_dir = patient_root / get_required(PATHS, "stage_04.directory")
    primary_path = stage3_dir / get_required(STAGE03, "outputs.predictions_json")
    log_path = stage3_dir / get_required(STAGE03, "outputs.log_diagnostic_predictions_json")
    primary_mal_path = stage4_dir / get_required(PATHS, "stage_04.output_json")
    log_mal_path = stage4_dir / get_required(PATHS, "stage_04.log_diagnostic_output_json")
    stage06_paths = get_required(PATHS, "stage_06", expected_type=dict)
    primary_stage06_path = patient_root / stage06_paths["directory"] / STAGE06["record_filename"]
    log_stage06_path = patient_root / stage06_paths["log_diagnostic_directory"] / STAGE06["record_filename"]
    for path in (primary_path, log_path, primary_mal_path, log_mal_path, primary_stage06_path, log_stage06_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required Stage 03/04 report input is missing: {path}. Run Stages 03 and 04 for both authoritative and LoG predictions first."
            )
    primary_records, _ = unwrap_records(read_json(primary_path), primary_path)
    log_records, _ = unwrap_records(read_json(log_path), log_path)
    primary_scores = build_score_lookup(read_json(primary_mal_path), primary_mal_path)
    log_scores = build_score_lookup(read_json(log_mal_path), log_mal_path)
    stage06_lookups: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for source, path in (("ViTDet3D", primary_stage06_path), ("LoG", log_stage06_path)):
        payload = read_json(path)
        validate_stage_version(
            payload.get("schema_version"),
            STAGE06["record_schema_version"],
            context=f"Stage 06 {source} records",
        )
        if payload.get("detection_source") != source:
            raise ValueError(f"{path}: expected Stage 06 source {source!r}, got {payload.get('detection_source')!r}.")
        records = payload.get("records")
        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
            raise ValueError(f"{path}: Stage 06 records must be a list of objects.")
        lookup: dict[tuple[str, int], dict[str, Any]] = {}
        for record in records:
            try:
                key = (str(record["candidate_id"]), int(record["patch_id"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid Stage 06 candidate/patch identity.") from exc
            if key in lookup:
                raise ValueError(f"{path}: duplicate Stage 06 candidate/patch identity {key}.")
            lookup[key] = record
        if int(payload.get("accepted_record_count", -1)) != len(lookup):
            raise ValueError(f"{path}: accepted_record_count does not reconcile with records.")
        stage06_lookups[source] = lookup
    all_source_rows = []
    accepted_rows = []
    excluded_rows = []
    lung_distance_inside = ndimage.distance_transform_edt(lung_mask.astype(bool), sampling=tuple(float(v) for v in spacing_zyx))

    for source, predictions, score_lookup in (
        ("ViTDet3D", primary_records, primary_scores),
        ("LoG", log_records, log_scores),
    ):
        stage06_lookup = stage06_lookups[source]
        prediction_lookup = {
            (str(row.get("candidate_id")), int(row["patch_id"])): row
            for row in predictions
            if row.get("candidate_id") is not None and row.get("patch_id") is not None
        }
        if set(prediction_lookup) != set(stage06_lookup):
            raise ValueError(
                f"Stage 03 and Stage 06 {source} candidate identities differ; "
                f"missing_in_stage06={sorted(set(prediction_lookup) - set(stage06_lookup))[:10]}, "
                f"missing_in_stage03={sorted(set(stage06_lookup) - set(prediction_lookup))[:10]}."
            )
        for key, stage06_record in stage06_lookup.items():
            prediction = prediction_lookup[key]
            center = prediction.get("center_zyx")
            inside, index, reason = sample_mask(lung_mask, center)
            if index is None:
                center_list = None
                dist = None
            else:
                center_list = list(index)
                dist = float(lung_distance_inside[index]) if inside else None
            row = _score_row(
                prediction, score_lookup, stage06_record,
                source=source, malignancy_threshold=malignancy_threshold,
            )
            row["candidate_in_lung_mask"] = bool(inside)
            row["center_inside_lung_mask"] = bool(inside)
            row["center_index_zyx_rounded"] = json.dumps(center_list)
            row["exclusion_reason"] = reason
            anatomy = prediction.get("anatomical_location") if isinstance(prediction.get("anatomical_location"), dict) else {}
            if inside and index is not None:
                classify_fallback_location(row, index, lung_distance_mm=dist or 0.0, anatomy=anatomy,
                                           air_mask=air_mask, heart_mask=heart_mask)
                accepted_rows.append(row)
            else:
                if index is not None:
                    row["center_inside_air_mask"] = bool(air_mask[index])
                    row["center_inside_heart_mask"] = bool(heart_mask[index])
                row["location_primary"] = anatomy.get("primary", "outside_lung")
                row["location_region"] = anatomy.get("region", "outside_lung_mask")
                row["location_compartment"] = anatomy.get("compartment", "outside_lung")
                row["laterality"] = anatomy.get("laterality", "unknown")
                row["vertical_region"] = anatomy.get("vertical_region", "unknown")
                row["wall_relationship"] = anatomy.get("wall_relationship", "outside_lung_mask")
                row["distance_to_lung_mask_mm"] = anatomy.get("distance_to_lung_mask_mm")
                row["airway_type"] = anatomy.get("airway_type")
                row["airway_proximity_mm"] = anatomy.get("airway_proximity_mm")
                excluded_rows.append(row)
            all_source_rows.append(row)
    return accepted_rows, excluded_rows, all_source_rows, {
        "primary_predictions_json": str(primary_path.resolve()),
        "log_predictions_json": str(log_path.resolve()),
        "primary_malignancy_json": str(primary_mal_path.resolve()),
        "log_malignancy_json": str(log_mal_path.resolve()),
        "primary_stage06_records": str(primary_stage06_path.resolve()),
        "log_stage06_records": str(log_stage06_path.resolve()),
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
    """Write a headless, identity-bound CT/CAM/SHAP summary for each in-lung record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for row in rows:
        record = row["stage06_record"]
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
        ct_ax.set_title(f"CT z={z}; patch {record['patch_id']}")
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

        shap_ax = fig.add_subplot(grid[2, 1:])
        contributions = record["explanations"]["shap"].get("feature_contributions") or {}
        shap_names = [name for name in ATTRIBUTES if name in contributions]
        if shap_names:
            shap_values = [float(contributions[name]) for name in shap_names]
            shap_ax.barh(shap_names, shap_values, color=["#d95f02" if value >= 0 else "#1b9e77" for value in shap_values])
            shap_ax.axvline(0.0, color="black", linewidth=0.8)
            shap_ax.set_title("SHAP contributions to raw Stage 04 rating (not probabilities)")
        else:
            shap_ax.text(0.5, 0.5, "SHAP explanation unavailable", ha="center", va="center", transform=shap_ax.transAxes)
        rating = row.get("predicted_malignancy_rating_continuous")
        shap_ax.set_xlabel(f"rating contribution | raw model score; displayed rating={rating}")

        fig.suptitle(
            f"{patient_id} | nodule {record['nodule_id']} | {record['detection_source']} | "
            f"LIDC reader-rating estimate={rating}",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
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
    extras = sorted({key for row in rows for key in row if key not in preferred})
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
                  malignancy_threshold: int) -> None:
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

        visible = [(row, idx) for row, idx in valid_rows if idx[0] == z]
        for row, (_, y, x) in visible:
            high = row.get("high_suspicion_rating_ge_threshold") is True
            source = row.get("detection_source")
            marker = str(VIEWER_SCHEMA["viTdet3d_marker"]) if source == "ViTDet3D" else str(VIEWER_SCHEMA["log_marker"])
            color = str(VIEWER_SCHEMA["rating_at_or_above_threshold_color"]) if high else str(VIEWER_SCHEMA["rating_below_threshold_color"])
            ax.scatter([x], [y], marker=marker, s=75, c=[color], edgecolors="white", linewidths=0.8)
            prefix = "V" if source == "ViTDet3D" else "L"
            label = f"{prefix}{row.get('patch_id')}"
            nodule_id = str(row.get("stable_nodule_id") or "")
            if nodule_id:
                label += f" {nodule_id[:8]}"
            rating = row.get("predicted_malignancy_rating_rounded")
            if rating is not None:
                label += f" R{rating}"
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=7,
                        color="white", bbox={"boxstyle": "round,pad=0.15", "fc": "black", "alpha": 0.6})
        ax.set_title(f"{patient_id} — axial z={z}/{z_count - 1} | in-lung candidates on slice: {len(visible)}")
        ax.set_xlabel("X voxel")
        ax.set_ylabel("Y voxel")
        ax.set_xlim(-0.5, ct.shape[2] - 0.5)
        ax.set_ylim(ct.shape[1] - 0.5, -0.5)
        legend_items = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="cyan", markeredgecolor="white", label="ViTDet3D: rating below threshold", markersize=7),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markeredgecolor="white", label=f"ViTDet3D: rating ≥ {malignancy_threshold}", markersize=7),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="cyan", markeredgecolor="white", label="LoG: rating below threshold", markersize=7),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="red", markeredgecolor="white", label=f"LoG: rating ≥ {malignancy_threshold}", markersize=7),
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
    parser.add_argument("--no-gui", dest="open_gui", action="store_false", default=bool(VIEWER_SCHEMA["default_open_gui"]), help="Write reports without opening the slider viewer.")
    parser.add_argument("--malignancy-rating-threshold", type=int,
                        default=int(STAGE07["malignancy_rating_threshold"]),
                        help="Operational high-suspicion rating threshold; not a cancer diagnosis.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.malignancy_rating_threshold <= 5:
        raise ValueError("--malignancy-rating-threshold must be in [1,5].")
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
        patient_root, lung_mask, air_mask, heart_mask, spacing, args.malignancy_rating_threshold
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

    count_by_source: dict[str, dict[str, int]] = {}
    for source in ("ViTDet3D", "LoG"):
        source_rows = [row for row in accepted if row["detection_source"] == source]
        count_by_source[source] = {
            "input_candidates": len([row for row in _all_rows if row["detection_source"] == source]),
            "accepted_center_inside_lung_mask": len(source_rows),
            "excluded_center_outside_lung_mask": len([row for row in excluded if row["detection_source"] == source]),
            "high_suspicion_rating_ge_threshold": sum(bool(row["high_suspicion_rating_ge_threshold"]) for row in source_rows),
        }
    summary = {
        "schema_version": STAGE07["report_schema_version"],
        "meta_schema_version": SCHEMA["schema_version"],
        "stage": "07_visualize",
        "patient_id": args.patient_id,
        "created_from": input_paths,
        "ct_volume": str(ct_path.resolve()),
        "lung_mask": str(lung_path.resolve()),
        "ct_shape_zyx": list(ct.shape),
        "spacing_zyx_mm": spacing.tolist(),
        "candidate_center_filter": STAGE07["center_in_mask_rule"],
        "input_candidate_count": len(_all_rows),
        "nodule_count_center_inside_lung_mask": len(accepted),
        "nodule_count_center_outside_lung_mask_excluded": len(excluded),
        "high_suspicion_or_malignancy_rating_count_inside_lung_mask": sum(bool(row["high_suspicion_rating_ge_threshold"]) for row in accepted),
        "malignancy_rating_threshold": args.malignancy_rating_threshold,
        "malignancy_count_interpretation": STAGE07["malignancy_threshold_semantics"],
        "counts_by_source": count_by_source,
        "accepted_csv": str(report_csv.resolve()),
        "excluded_outside_lung_csv": str(excluded_csv.resolve()),
        "visualization_directory": str(visualization_root.resolve()),
        "visualization_count": len(visualization_paths),
        "accepted_candidate_rows": accepted,
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
    print(f"Per-nodule CT/CAM/SHAP visualizations: {len(visualization_paths)} in {visualization_root}")
    print(f"Candidates inspected: {len(_all_rows)}")
    print(f"Candidates accepted (rounded center inside lung_mask): {len(accepted)}")
    print(f"Candidates excluded (center outside lung_mask or CT): {len(excluded)}")
    print(f"High-suspicion rating >= {args.malignancy_rating_threshold} within lung mask: {summary['high_suspicion_or_malignancy_rating_count_inside_lung_mask']}")
    print("Interpretation: Stage 04 reader-suspicion rating threshold; not cancer probability or a clinical diagnosis.")

    if args.open_gui:
        launch_viewer(ct, lung_mask, air_mask, heart_mask, accepted, args.patient_id,
                      float(DISPLAY_WINDOW["level"]), float(DISPLAY_WINDOW["width"]),
                      args.malignancy_rating_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
