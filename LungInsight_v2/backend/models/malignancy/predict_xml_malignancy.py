#!/usr/bin/env python3
"""Predict a radiologist malignancy-suspicion rating from structured XML features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from xml_malignancy_common import (
    DISCLAIMER,
    FEATURE_COLUMNS,
    OUTPUT_DEFAULT,
    rows_to_matrix,
)


RATING_DESCRIPTIONS = {
    1: "highly unlikely",
    2: "moderately unlikely",
    3: "indeterminate",
    4: "moderately suspicious",
    5: "highly suspicious",
}


def predict_record(
    record: dict[str, Any], model_bundle: dict[str, Any], schema: dict[str, Any]
) -> dict[str, Any]:
    if "features" in record and isinstance(record["features"], dict):
        record = record["features"]
    expected_order = model_bundle.get("feature_order")
    if expected_order != list(FEATURE_COLUMNS):
        raise ValueError("Model feature order does not match this inference implementation")
    row = {name: record.get(name) for name in FEATURE_COLUMNS}
    warnings: list[str] = []
    missing = [name for name, value in row.items() if value is None or value == ""]
    if missing:
        warnings.append(
            f"Missing {len(missing)} predictor feature(s); training medians and indicators will be used: {', '.join(missing)}"
        )
    non_feature_keys = sorted(set(record) - set(FEATURE_COLUMNS))
    if non_feature_keys:
        warnings.append(f"Ignored non-feature fields: {', '.join(non_feature_keys)}")
    for name, value in row.items():
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            warnings.append(f"Feature {name} is not numeric and will be treated as missing")
            row[name] = None
            continue
        if not (numeric == numeric and abs(numeric) != float("inf")):
            warnings.append(f"Feature {name} is non-finite and will be treated as missing")
            row[name] = None
            continue
        spec = schema.get("features", {}).get(name, {})
        legal_range = spec.get("rating_range")
        if legal_range and not (legal_range[0] <= numeric <= legal_range[1]):
            warnings.append(
                f"Feature {name}={numeric:g} is outside its valid rating-derived range {legal_range}"
            )
        observed_range = spec.get("observed_range")
        if observed_range and not (observed_range[0] <= numeric <= observed_range[1]):
            warnings.append(
                f"Feature {name}={numeric:g} is outside the observed training range {observed_range}"
            )
        row[name] = numeric

    probabilities = model_bundle["model"].predict_proba(rows_to_matrix([row]))[0]
    expected_rating = float(probabilities @ [1, 2, 3, 4, 5])
    predicted_rating = int(probabilities.argmax() + 1)
    return {
        "predicted_ordinal_rating": predicted_rating,
        "predicted_rating_description": RATING_DESCRIPTIONS[predicted_rating],
        "expected_rating_score": expected_rating,
        "probabilities_by_rating": {
            str(rating): float(probabilities[rating - 1]) for rating in range(1, 6)
        },
        "interpretation": DISCLAIMER,
        "warnings": warnings,
        "model": model_bundle.get("model_name"),
    }


def _read_record(value: str) -> dict[str, Any]:
    path = Path(value)
    if path.is_file():
        parsed = json.loads(path.read_text(encoding="utf-8"))
    else:
        parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Inference input must be a JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--record", required=True, help="JSON file path or inline feature-record JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_dir = args.data_dir.resolve()
    bundle = joblib.load(data_dir / "best_model.pkl")
    schema = json.loads((data_dir / "feature_schema.json").read_text(encoding="utf-8"))
    result = predict_record(_read_record(args.record), bundle, schema)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())