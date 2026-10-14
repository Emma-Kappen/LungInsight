#!/usr/bin/env python3
"""LungInsight Stage 04: predict LIDC-style malignancy ratings from Stage 03.

This is a separately callable inference stage. It consumes the existing Stage 03
predictions.json and the frozen XML-trained Stage 04 model. It reads no CT/H5
pixels and never invokes or modifies Stage 03.

Typical use from the LungInsight_v2 repository root:
    python backend/Imaging/04_malignancy_prediction.py LIDC-IDRI-0001

The default output is:
    backend/output/<patient_id>/04_malignancy_prediction/malignancy_predictions.json
"""
from __future__ import annotations

import argparse
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

import joblib
import numpy as np
from pipeline_schema import get_required, load_schema, check_meta_schema_version, validate_stage_version
from stage04_shap import explain_regression, load_training_background, sha256_file

SCHEMA = load_schema(__file__)
STAGE04_SCHEMA = get_required(SCHEMA, "stage_04_malignancy_prediction", expected_type=dict)
STAGE03_SCHEMA = get_required(SCHEMA, "stage_03_nodule_classification", expected_type=dict)
STAGE04_PATHS = get_required(SCHEMA, "paths.stage_04", expected_type=dict)
STAGE03_PATHS = get_required(SCHEMA, "paths.stage_03", expected_type=dict)
FEATURES = list(get_required(SCHEMA, str(STAGE04_SCHEMA["feature_names_ref"]), expected_type=list))
INPUT_RATING_RANGE = tuple(float(v) for v in get_required(STAGE04_SCHEMA, "input_rating_range", expected_type=list))
TARGET_RATING_RANGE = tuple(float(v) for v in get_required(STAGE04_SCHEMA, "target_rating_range", expected_type=list))
PREDICTION_TRANSFORM = get_required(STAGE04_SCHEMA, "prediction_transform", expected_type=dict)
STAGE_ROOT = IMAGING_DIR.parent.parent
DEFAULT_OUTPUT_ROOT = STAGE_ROOT / get_required(SCHEMA, "paths.output_root")
DEFAULT_MODEL = STAGE_ROOT / get_required(STAGE04_SCHEMA, "model_path")
STAGE_NAME = str(STAGE04_PATHS["directory"])


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    except OSError as exc:
        raise OSError(f"Could not read JSON file {path}: {exc}") from exc


def unpack_stage03_payload(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
        metadata: dict[str, Any] = {}
    elif isinstance(payload, dict):
        records = payload.get("predictions")
        metadata = payload
        if not isinstance(records, list):
            raise ValueError(
                "Stage 03 JSON must contain a list-valued 'predictions' field; "
                f"top-level keys={sorted(map(str, payload.keys()))}"
            )
    else:
        raise ValueError("Stage 03 predictions JSON must be an object or legacy array")
    if not records:
        raise ValueError("Stage 03 predictions array is empty; no candidates to score")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every Stage 03 candidate must be a JSON object")
    return records, metadata


def get_attribute_rating(record: dict[str, Any], name: str) -> float:
    attrs = record.get("attributes")
    if not isinstance(attrs, dict):
        raise ValueError("missing 'attributes' object")
    item = attrs.get(name)
    if isinstance(item, dict):
        # Current Stage 03 schema emits expected_rating on the 1..5 ordinal scale.
        value = item.get("expected_rating")
        if value is None:
            value = item.get("rating_original_scale")
        if value is None:
            value = item.get("rating_rounded")
        if value is None:
            raise ValueError(
                f"{name} lacks expected_rating/rating_original_scale/rating_rounded"
            )
    else:
        value = item
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} rating is not numeric") from exc
    if not math.isfinite(value) or not INPUT_RATING_RANGE[0] <= value <= INPUT_RATING_RANGE[1]:
        raise ValueError(
            f"{name} rating must be finite and in {INPUT_RATING_RANGE}, got {value}; "
            "check Stage 03 attribute predictions and meta_schema.json."
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 04 malignancy-rating inference from existing Stage 03 attribute predictions."
    )
    parser.add_argument(
        "patient_id", nargs="?",
        help="Patient ID (e.g. LIDC-IDRI-0001); locates the existing Stage 03 output.",
    )
    parser.add_argument(
        "--predictions", type=Path,
        help="Explicit path to an existing Stage 03 predictions.json. If supplied, patient_id is optional.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help=f"Pipeline output root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help=f"Frozen XML-trained Stage 04 joblib model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Explicit output JSON path. Default: <output-root>/<patient>/04_malignancy_prediction/malignancy_predictions.json.",
    )
    parser.add_argument(
        "--skip-invalid", action="store_true",
        help="Skip candidates with missing/invalid attributes and record them in the output. Default: fail fast.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.predictions is not None:
        input_path = args.predictions
        patient_id_arg = args.patient_id
    else:
        if not args.patient_id:
            raise SystemExit("Provide patient_id or --predictions PATH")
        patient_id_arg = args.patient_id
        input_path = args.output_root / args.patient_id / STAGE03_PATHS["directory"] / STAGE03_SCHEMA["outputs"]["predictions_json"]

    input_path = input_path.resolve()
    model_path = args.model.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Existing Stage 03 predictions.json not found: {input_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Stage 04 model bundle not found: {model_path}")

    try:
        bundle = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Stage 04 joblib model bundle {model_path}: {exc}. "
            "Restore the existing model file and ensure the environment has a compatible scikit-learn/joblib version."
        ) from exc
    if not isinstance(bundle, dict) or bundle.get("feature_names") != FEATURES or "model" not in bundle:
        got = bundle.get("feature_names") if isinstance(bundle, dict) else type(bundle).__name__
        raise ValueError(
            "Model bundle does not match the Stage 04 six-feature contract. "
            f"Expected {FEATURES}; got {got}"
        )

    payload = read_json(input_path)
    if isinstance(payload, dict):
        check_meta_schema_version(payload, SCHEMA, context=str(input_path), require=False)
        validate_stage_version(payload.get("schema_version"), STAGE03_SCHEMA["schema_version"], context="Stage 03 predictions")
    records, metadata = unpack_stage03_payload(payload)
    feature_rows: list[list[float]] = []
    valid_records: list[tuple[int, dict[str, Any]]] = []
    errors: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        try:
            row = [get_attribute_rating(record, name) for name in FEATURES]
            feature_rows.append(row)
            valid_records.append((idx, record))
        except ValueError as exc:
            errors.append({
                "index": idx,
                "patch_id": record.get("patch_id"),
                "candidate_id": record.get("candidate_id"),
                "error": str(exc),
            })
            if not args.skip_invalid:
                raise ValueError(
                    f"Invalid Stage 03 candidate index {idx} "
                    f"(patch_id={record.get('patch_id')}, candidate_id={record.get('candidate_id')}): {exc}"
                ) from exc

    if not feature_rows:
        raise ValueError("No valid Stage 03 candidates remain to score")

    x = np.asarray(feature_rows, dtype=np.float64)
    raw_predictions = np.asarray(bundle["model"].predict(x), dtype=float)
    if not np.isfinite(raw_predictions).all():
        raise ValueError("Stage 04 model returned NaN or Inf predictions; check the existing model bundle and its runtime dependencies.")
    shap_schema = get_required(STAGE04_SCHEMA, "shap_explanations", expected_type=dict)
    background_path = (STAGE_ROOT / str(shap_schema["training_table_path"])).resolve()
    background, background_metadata = load_training_background(
        background_path,
        FEATURES,
        split_column=str(shap_schema["split_column"]),
        training_split_value=str(shap_schema["training_split_value"]),
        sample_size=int(shap_schema["background_sample_size"]),
        random_seed=int(shap_schema["background_random_seed"]),
    )
    shap_values, shap_base_values, shap_metadata = explain_regression(
        bundle["model"],
        background,
        x,
        additivity_tolerance=float(shap_schema["additivity_tolerance"]),
    )
    shap_model_sha256 = sha256_file(model_path)
    shap_metadata.update(
        {
            "status": "complete",
            "background_data": background_metadata,
            "feature_names_in_order": list(FEATURES),
            "preprocessing": (
                "No additional Stage 04 transform: the six Stage 03 expected ratings "
                "in [1,5] are passed directly to the frozen model."
            ),
            "model_file": str(model_path),
            "model_sha256": shap_model_sha256,
            "model_type": bundle.get("model_type", type(bundle["model"]).__name__),
            "limitations": list(shap_schema["limitations"]),
        }
    )
    if PREDICTION_TRANSFORM["clip_to_target_range"]:
        predictions = np.clip(raw_predictions, TARGET_RATING_RANGE[0], TARGET_RATING_RANGE[1])
    else:
        predictions = raw_predictions
    rounding_method = str(PREDICTION_TRANSFORM["rounding_method"])
    if rounding_method != "floor_plus_half":
        raise ValueError(
            f"Unsupported Stage 04 rounding method in meta_schema.json: {rounding_method!r}. "
            "Set a supported method without changing the trained model artifact."
        )
    parent_patient_id = (metadata.get("patient_id") if isinstance(metadata, dict) else None) or patient_id_arg
    output_predictions: list[dict[str, Any]] = []
    for result_index, ((idx, record), row, pred, raw_pred) in enumerate(
        zip(valid_records, x, predictions, raw_predictions)
    ):
        rounded = int(np.clip(math.floor(float(pred) + 0.5), TARGET_RATING_RANGE[0], TARGET_RATING_RANGE[1]))
        output_predictions.append({
            "patch_id": record.get("patch_id"),
            "candidate_id": record.get("candidate_id"),
            "detection_source": record.get("detection_source", "ViTDet3D"),
            "diagnostic_id": record.get("diagnostic_id"),
            "purpose": record.get("purpose"),
            "stage04_eligible": record.get("stage04_eligible", True),
            "center_zyx": record.get("center_zyx"),
            "center_lps_mm": record.get("center_lps_mm"),
            "anatomical_location": record.get("anatomical_location"),
            "patient_id": record.get("patient_id") or record.get("patient") or parent_patient_id,
            "predicted_malignancy_rating_continuous": float(pred),
            "predicted_malignancy_rating_rounded": rounded,
            "model_raw_prediction_before_rating_clip": float(raw_pred),
            "rating_scale": f"{int(TARGET_RATING_RANGE[0])}..{int(TARGET_RATING_RANGE[1])}",
            "target_semantics": STAGE04_SCHEMA["semantics"],
            "feature_values_used": {name: float(value) for name, value in zip(FEATURES, row)},
            "shap_explanation": {
                "base_value_raw": float(shap_base_values[result_index]),
                "feature_contributions": {
                    name: float(value)
                    for name, value in zip(FEATURES, shap_values[result_index])
                },
                "explained_output": "model_raw_prediction_before_rating_clip",
                "model_sha256": shap_model_sha256,
            },
            "stage03_prediction_index": idx,
            "ct_or_image_used_by_malignancy_predictor": False,
        })

    if args.output is not None:
        output_path = args.output.resolve()
    else:
        patient_id = parent_patient_id or patient_id_arg
        if not patient_id:
            raise ValueError("Cannot determine patient_id for default output path; supply --output")
        output_path = (args.output_root / patient_id / STAGE_NAME / STAGE04_PATHS["output_json"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prediction_source = (metadata.get("detection_source") if isinstance(metadata, dict) else None)
    if not prediction_source and records:
        prediction_source = records[0].get("detection_source", "ViTDet3D")
    prediction_source = str(prediction_source or "ViTDet3D")
    result = {
        "schema_version": STAGE04_SCHEMA["output_schema_version"],
        "meta_schema_version": SCHEMA["schema_version"],
        "stage": STAGE_NAME,
        "status": "completed",
        "patient_id": parent_patient_id,
        "detection_source": prediction_source,
        "purpose": "research_comparison_only" if prediction_source == "LoG" else "authoritative_candidate_scoring",
        "model_file": str(model_path),
        "model_version": model_path.parent.name,
        "model_type": bundle.get("model_type", "XML-trained regressor"),
        "feature_names_in_order": list(FEATURES),
        "input_source": f"existing Stage 03 scalar attribute predictions ({prediction_source})",
        "source_predictions_json": str(input_path),
        "stage03_schema_version": metadata.get("schema_version") if isinstance(metadata, dict) else None,
        "candidate_count": len(records),
        "scored_candidate_count": len(output_predictions),
        "invalid_candidates": errors,
        "target_semantics": STAGE04_SCHEMA["semantics"],
        "shap": shap_metadata,
        "ct_or_image_read": False,
        "h5_read": False,
        "stage03_modified": False,
        "validation_warning": (
            "Interface compatibility is verified separately from predictive validity. "
            "Existing validation showed weak discrimination on Stage 03 outputs."
        ),
        "predictions": output_predictions,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "stage": STAGE_NAME,
        "patient_id": parent_patient_id,
        "input": str(input_path),
        "model": str(model_path),
        "output": str(output_path),
        "input_candidates": len(records),
        "scored_candidates": len(output_predictions),
        "invalid_candidates": len(errors),
        "ct_or_image_read": False,
        "h5_read": False,
        "stage03_modified": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
