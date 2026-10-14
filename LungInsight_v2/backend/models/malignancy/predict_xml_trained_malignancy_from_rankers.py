#!/usr/bin/env python3
"""Run the XML-trained Stage 04 malignancy model on Stage 03 predictions.json.

Accepts both supported input forms:
  1) the current Stage 03 wrapper object with a `predictions` array;
  2) a legacy bare JSON array of candidate records.

Only six scalar Stage 03 attribute outputs are passed to the malignancy model.
No CT, image, H5, or ranker checkpoint is read here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

FEATURES = ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"]
DEFAULT_MODEL = Path(
    "LungInsight_v2/backend/models/malignancy/xml_nodule_level_v1/"
    "malignancy_from_nodule_level_xml.joblib"
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def unpack_stage03_payload(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return candidate records and top-level metadata from Stage 03 JSON."""
    if isinstance(payload, list):
        records = payload
        metadata: dict[str, Any] = {}
    elif isinstance(payload, dict):
        records = payload.get("predictions")
        metadata = payload
        if not isinstance(records, list):
            keys = sorted(str(k) for k in payload.keys())
            raise ValueError(
                "Stage 03 JSON is an object but has no list-valued 'predictions' field; "
                f"top-level keys={keys}"
            )
    else:
        raise ValueError(
            "Expected Stage 03 predictions.json to contain an object with a 'predictions' "
            "array or a legacy JSON array"
        )

    if not records:
        raise ValueError("Stage 03 predictions array is empty; no candidate records to score")
    if not all(isinstance(row, dict) for row in records):
        raise ValueError("Every entry in the Stage 03 predictions array must be an object")
    return records, metadata


def get_value(record: dict[str, Any], name: str) -> float:
    attrs = record.get("attributes")
    if not isinstance(attrs, dict):
        raise ValueError("missing 'attributes' object")

    item = attrs.get(name)
    if isinstance(item, dict):
        # Current Stage 03 schema: continuous expected ordinal rating on 1..5.
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

    if not math.isfinite(value) or not 1.0 <= value <= 5.0:
        raise ValueError(f"{name} rating must be finite and in [1,5], got {value}")

    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions_json", type=Path, help="Existing Stage 03 predictions.json")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                    help=f"Stage 04 model joblib (default: {DEFAULT_MODEL})")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output JSON path (default: malignancy_predictions.json beside input)")
    ap.add_argument("--skip-invalid", action="store_true",
                    help="Skip invalid candidates instead of stopping on the first error")
    args = ap.parse_args()

    if not args.predictions_json.is_file():
        raise FileNotFoundError(f"Stage 03 predictions JSON not found: {args.predictions_json}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Stage 04 model bundle not found: {args.model}")

    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or bundle.get("feature_names") != FEATURES or "model" not in bundle:
        got = bundle.get("feature_names") if isinstance(bundle, dict) else type(bundle).__name__
        raise ValueError(
            "Model bundle does not match six-feature Stage 04 contract. "
            f"Expected {FEATURES}; got {got}"
        )

    payload = read_json(args.predictions_json)
    records, metadata = unpack_stage03_payload(payload)

    valid_rows: list[tuple[int, dict[str, Any]]] = []
    feature_rows: list[list[float]] = []
    errors: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        try:
            vals = [get_value(record, name) for name in FEATURES]
            feature_rows.append(vals)
            valid_rows.append((i, record))
        except ValueError as exc:
            errors.append({
                "index": i,
                "error": str(exc),
                "patch_id": record.get("patch_id"),
                "candidate_id": record.get("candidate_id"),
            })
            if not args.skip_invalid:
                raise ValueError(
                    f"Stage 03 candidate index {i} "
                    f"(patch_id={record.get('patch_id')}, candidate_id={record.get('candidate_id')}): {exc}"
                ) from exc

    if not feature_rows:
        raise ValueError("No valid Stage 03 candidate records to score")

    X = np.asarray(feature_rows, dtype=np.float64)
    preds = np.clip(np.asarray(bundle["model"].predict(X), dtype=float), 1.0, 5.0)
    output_records: list[dict[str, Any]] = []
    parent_patient_id = metadata.get("patient_id") if isinstance(metadata, dict) else None

    for (idx, record), row, pred in zip(valid_rows, X, preds):
        rounded = int(np.clip(math.floor(float(pred) + 0.5), 1, 5))
        output_records.append({
            "patch_id": record.get("patch_id"),
            "candidate_id": record.get("candidate_id"),
            "patient_id": record.get("patient_id") or record.get("patient") or parent_patient_id,
            "predicted_malignancy_rating_continuous": float(pred),
            "predicted_malignancy_rating_rounded": rounded,
            "rating_scale": "1..5",
            "target_semantics": "LIDC reader-assigned malignancy/suspicion rating; not pathology-confirmed cancer probability",
            "feature_values_used": {name: float(value) for name, value in zip(FEATURES, row)},
            "stage03_prediction_index": idx,
            "ct_or_image_used_by_malignancy_predictor": False,
        })

    result = {
        "schema_version": "stage04-xml-trained-malignancy-output-v2",
        "model_file": str(args.model),
        "model_type": bundle.get("model_type", "ExtraTreesRegressor"),
        "feature_names_in_order": FEATURES,
        "input_source": "existing Stage 03 predictions.json scalar ordinal attribute outputs",
        "stage03_schema_version": metadata.get("schema_version") if isinstance(metadata, dict) else None,
        "patient_id": parent_patient_id,
        "source_predictions_json": str(args.predictions_json),
        "candidate_count": len(records),
        "scored_candidate_count": len(output_records),
        "invalid_candidates": errors,
        "target_semantics": "LIDC reader malignancy/suspicion rating, not verified cancer probability",
        "xml_to_ranker_input_shift_warning": (
            "Input shape compatibility does not establish predictive validity. Consult the model's "
            "training_report.json for held-out validation results."
        ),
        "ct_or_image_used_by_malignancy_predictor": False,
        "predictions": output_records,
    }
    output_path = args.output or args.predictions_json.with_name("malignancy_predictions.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "output": str(output_path),
        "input_schema": "wrapped Stage 03 object" if isinstance(payload, dict) else "legacy array",
        "input_candidates": len(records),
        "scored_candidates": len(output_records),
        "invalid_candidates": len(errors),
        "patient_id": parent_patient_id,
        "ct_or_image_read": False,
        "h5_read": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
