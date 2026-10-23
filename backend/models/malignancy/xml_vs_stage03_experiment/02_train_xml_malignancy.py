#!/usr/bin/env python
"""Train a CPU-only XML-input malignancy-rating regressor.

Training inputs: six XML reader-rated attributes.
Target: malignancy rating from the same XML annotation.
Uses only train_xml.csv; val_xml.csv is used for a simple XML-input sanity report.
The model is deliberately not trained on Stage 03 predictions (Option C).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture")
TARGET = "malignancy"


def load_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    X, y, patients = [], [], []
    for row in rows:
        try:
            x = [float(row[name]) for name in FEATURES]
            target = float(row[TARGET])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(x).all() or not np.isfinite(target):
            continue
        if not (1 <= target <= 5) or any(v < 1 or v > 5 for v in x):
            continue
        X.append(x)
        y.append(target)
        patients.append(row.get("patient_id", ""))
    if not X:
        raise ValueError(f"No valid rows in {path}")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(patients)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="Directory containing train_xml.csv and val_xml.csv")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--model", choices=("extra-trees", "random-forest"), default="extra-trees")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--min-samples-leaf", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    train_x, train_y, train_p = load_csv(args.data_dir / "train_xml.csv")
    val_x, val_y, val_p = load_csv(args.data_dir / "val_xml.csv")
    overlap = set(train_p) & set(val_p)
    if overlap:
        raise ValueError(f"Patient leakage between train and validation XML rows: {sorted(overlap)[:10]}")

    common = dict(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
        random_state=args.seed,
        n_jobs=-1,
    )
    if args.model == "extra-trees":
        model = ExtraTreesRegressor(**common)
    else:
        model = RandomForestRegressor(**common)

    model.fit(train_x, train_y)
    val_pred = np.clip(model.predict(val_x), 1.0, 5.0)
    val_mae = float(mean_absolute_error(val_y, val_pred))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "xml_attribute_malignancy.joblib"
    joblib.dump(model, model_path)
    metadata = {
        "model_type": args.model,
        "input_features_in_order": list(FEATURES),
        "target": TARGET,
        "input_source_during_training": "LIDC XML-derived annotation manifest",
        "target_semantics": "reader-assigned malignancy/suspicion rating 1..5, not confirmed pathology",
        "training_rows": int(len(train_y)),
        "training_patients": int(len(set(train_p))),
        "validation_rows": int(len(val_y)),
        "validation_patients": int(len(set(val_p))),
        "validation_xml_input_mae": val_mae,
        "hyperparameters": common,
        "model_path": str(model_path.resolve()),
        "warning": (
            "Validation above uses XML-rated attributes and is not the primary deployment-style result. "
            "Use evaluate_stage03_continuous_vs_rounded.py for held-out Stage 03 inputs."
        ),
    }
    (args.output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
