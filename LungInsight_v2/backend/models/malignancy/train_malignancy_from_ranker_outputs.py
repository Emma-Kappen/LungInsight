#!/usr/bin/env python3
"""Train the downstream malignancy-rating predictor from cached Stage 03 outputs.

This script reads only frozen feature archives (NPZ) containing the six Stage 03
attribute predictions and XML-derived reader malignancy ratings. It does NOT read
H5/CT images, run or retrain attribute models, or open the held-out test archive.

Expected NPZ keys: X, feature_names, patient_id, malignancy_reader_ratings,
malignancy_reader_mask. The target is the mean of each nodule's valid reader
malignancy ratings (1..5), so reader repetitions do not become duplicate samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score

FEATURES = [
    "subtlety",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def to_strings(values: np.ndarray) -> list[str]:
    result = []
    for value in values.tolist():
        if isinstance(value, bytes):
            result.append(value.decode("utf-8", errors="replace"))
        else:
            result.append(str(value))
    return result


def rounded_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5).astype(np.int64)


def load_split(path: Path, split_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {split_name} feature archive: {path}\n"
            "Expected the existing backend/models/malignancy/frozen_features/*.npz archives."
        )

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "X", "feature_names", "patient_id",
            "malignancy_reader_ratings", "malignancy_reader_mask",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"{path} is missing required NPZ keys: {missing}")
        data = {key: archive[key].copy() for key in required}

    X = np.asarray(data["X"], dtype=np.float32)
    feature_names = to_strings(data["feature_names"])
    patients = to_strings(data["patient_id"])
    ratings = np.asarray(data["malignancy_reader_ratings"], dtype=np.float32)
    mask = np.asarray(data["malignancy_reader_mask"], dtype=bool)

    if X.ndim != 2 or X.shape[1] != len(FEATURES):
        raise ValueError(f"{split_name}: X must have shape (N, 6), got {X.shape}")
    if feature_names != FEATURES:
        raise ValueError(
            f"{split_name}: feature order mismatch.\n"
            f"Expected: {FEATURES}\nFound:    {feature_names}\n"
            "Refusing to train on an ambiguous feature order."
        )
    if len(patients) != X.shape[0]:
        raise ValueError(f"{split_name}: patient_id count does not match X rows")
    if ratings.ndim != 2 or mask.shape != ratings.shape or ratings.shape[0] != X.shape[0]:
        raise ValueError(
            f"{split_name}: expected reader ratings/mask with matching (N, readers) shape; "
            f"X={X.shape}, ratings={ratings.shape}, mask={mask.shape}"
        )
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).any(axis=1).sum())
        raise ValueError(f"{split_name}: X contains non-finite values in {n_bad} rows")

    valid_ratings = mask & np.isfinite(ratings) & (ratings >= 1.0) & (ratings <= 5.0)
    rating_counts = valid_ratings.sum(axis=1)
    keep = rating_counts > 0
    if not keep.any():
        raise ValueError(f"{split_name}: no nodules have a valid XML malignancy rating")
    y = (np.where(valid_ratings, ratings, 0.0).sum(axis=1) / np.maximum(rating_counts, 1))[keep]

    return {
        "X": X[keep],
        "y": y.astype(np.float32),
        "patients": np.asarray(patients, dtype=str),
        "kept_rows": int(keep.sum()),
        "dropped_no_target": int((~keep).sum()),
        "reader_observations": int(valid_ratings.sum()),
        "raw_rows": int(X.shape[0]),
        "feature_names": feature_names,
        "archive_sha256": sha256_file(path),
    }


def metric_report(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 1.0, 5.0)
    y_true = np.asarray(y_true, dtype=np.float64)
    baseline = np.full_like(y_true, float(np.mean(y_train)))
    y_true_round = np.clip(rounded_half_up(y_true), 1, 5)
    y_pred_round = np.clip(rounded_half_up(y_pred), 1, 5)
    base_round = np.clip(rounded_half_up(baseline), 1, 5)

    def qwk(a: np.ndarray, b: np.ndarray) -> float | None:
        try:
            value = cohen_kappa_score(a, b, weights="quadratic")
            return float(value) if np.isfinite(value) else None
        except Exception:
            return None

    return {
        "n_nodules": int(len(y_true)),
        "target_mean": float(np.mean(y_true)),
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred)),
        "mae_continuous": float(mean_absolute_error(y_true, y_pred)),
        "rmse_continuous": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "constant_train_mean_baseline_mae": float(mean_absolute_error(y_true, baseline)),
        "mae_rounded_half_up": float(mean_absolute_error(y_true_round, y_pred_round)),
        "accuracy_rounded_half_up": float(np.mean(y_true_round == y_pred_round)),
        "qwk_rounded_half_up": qwk(y_true_round, y_pred_round),
        "target_rounded_counts": {str(int(k)): int((y_true_round == k).sum()) for k in range(1, 6)},
        "prediction_rounded_counts": {str(int(k)): int((y_pred_round == k).sum()) for k in range(1, 6)},
        "baseline_rounded_qwk": qwk(y_true_round, base_round),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path,
        default=Path("backend/models/malignancy/frozen_features"),
        help="Existing directory containing train.npz and val.npz. test.npz is never opened.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("backend/models/malignancy/stage03_input_v2"),
        help="New output directory; existing non-empty directory is protected by default.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in --out-dir")
    args = parser.parse_args()

    train_path = args.features_dir / "train.npz"
    val_path = args.features_dir / "val.npz"
    # Intentionally do not construct, stat, load, or evaluate test.npz here.
    train = load_split(train_path, "train")
    val = load_split(val_path, "validation")

    train_patients = set(train["patients"].tolist())
    val_patients = set(val["patients"].tolist())
    overlap = train_patients.intersection(val_patients)
    if overlap:
        preview = sorted(overlap)[:10]
        raise ValueError(
            f"Patient leakage: {len(overlap)} patient IDs overlap between train and validation; "
            f"examples={preview}. Stop rather than changing the split automatically."
        )

    out_dir = args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}\n"
            "Choose a new --out-dir or pass --overwrite deliberately."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # One predeclared downstream model; no sweep and no test-set model selection.
    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=4,
        max_features=1.0,
        random_state=2026,
        n_jobs=-1,
    )
    model.fit(train["X"], train["y"])
    val_pred = np.clip(model.predict(val["X"]), 1.0, 5.0)
    metrics = metric_report(val["y"], val_pred, train["y"])

    model_path = out_dir / "malignancy_from_attributes.joblib"
    bundle = {
        "model": model,
        "feature_names": FEATURES,
        "target": "LIDC reader-assigned malignancy/suspicion rating",
        "target_scale": [1, 5],
        "target_aggregation": "per-nodule mean of valid finite XML malignancy reader ratings",
        "input_semantics": "frozen Stage 03 expected ordinal ratings; no CT/image inputs",
        "model_type": "ExtraTreesRegressor",
        "hyperparameters": {
            "n_estimators": 500,
            "min_samples_leaf": 4,
            "max_features": 1.0,
            "random_state": 2026,
        },
    }
    joblib.dump(bundle, model_path)

    pred_csv = out_dir / "validation_predictions.csv"
    import csv
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_in_val_archive", "patient_id", "target_reader_mean", "predicted_rating_continuous", "predicted_rating_rounded_half_up"])
        # patient array is one item per raw row. The report currently indicates all val rows have labels;
        # this mapping remains correct if a future archive has a missing target row.
        original_patients = val["patients"]
        # Re-read only val metadata to align row indices when any targets were dropped.
        with np.load(val_path, allow_pickle=False) as archive:
            val_ratings = np.asarray(archive["malignancy_reader_ratings"], dtype=np.float32)
            val_mask = np.asarray(archive["malignancy_reader_mask"], dtype=bool)
        valid = val_mask & np.isfinite(val_ratings) & (val_ratings >= 1.0) & (val_ratings <= 5.0)
        counts = valid.sum(axis=1)
        kept_indices = np.flatnonzero(counts > 0)
        for ix, patient, target, prediction in zip(kept_indices, original_patients[kept_indices], val["y"], val_pred):
            writer.writerow([int(ix), patient, f"{float(target):.6f}", f"{float(prediction):.6f}", int(np.clip(math.floor(float(prediction) + 0.5), 1, 5))])

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "features": FEATURES,
        "input_archives": {
            "train": {"path": str(train_path), "sha256": train["archive_sha256"], "raw_rows": train["raw_rows"], "used_rows": train["kept_rows"], "dropped_no_target": train["dropped_no_target"], "valid_reader_observations": train["reader_observations"]},
            "validation": {"path": str(val_path), "sha256": val["archive_sha256"], "raw_rows": val["raw_rows"], "used_rows": val["kept_rows"], "dropped_no_target": val["dropped_no_target"], "valid_reader_observations": val["reader_observations"]},
        },
        "patient_overlap_train_validation": 0,
        "validation_metrics": metrics,
        "test_archive_opened": False,
        "h5_or_ct_read_by_this_script": False,
        "attribute_models_retrained": False,
        "limitations": [
            "Targets are LIDC reader malignancy/suspicion ratings, not pathology-confirmed outcomes.",
            "The existing train feature archive was generated by fixed rankers trained on the canonical training split; those train features are in-sample, not out-of-fold.",
            "Validation is for assessment only. Do not use the held-out test set to choose hyperparameters.",
        ],
    }
    (out_dir / "training_report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": "trained",
        "model_path": str(model_path),
        "report_path": str(out_dir / "training_report.json"),
        "validation_predictions_path": str(pred_csv),
        "train_nodules": train["kept_rows"],
        "validation_nodules": val["kept_rows"],
        "validation_metrics": metrics,
        "test_archive_opened": False,
        "h5_or_ct_read": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
