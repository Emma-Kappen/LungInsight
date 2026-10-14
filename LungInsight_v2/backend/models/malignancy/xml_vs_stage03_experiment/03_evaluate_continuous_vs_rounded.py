#!/usr/bin/env python
"""Compare continuous vs half-up-rounded frozen Stage 03 inputs on held-out patients.

Consumes the actual frozen_features schema:
X, malignancy_reader_ratings, malignancy_reader_mask, patient_id, nodule_id.
The XML-trained model is evaluated on Stage 03-predicted attribute scores, so
this is an input-distribution-shift diagnostic, not a clean in-domain estimate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error

FEATURES = ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture")


def round_half_up_clamped(x: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(np.asarray(x, dtype=np.float64) + 0.5), 1, 5).astype(np.int64)


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) == 0:
        return None
    try:
        value = cohen_kappa_score(y_true, y_pred, labels=[1, 2, 3, 4, 5], weights="quadratic")
        return float(value) if np.isfinite(value) else None
    except Exception:
        return None


def class_counts(values: np.ndarray) -> dict[str, int]:
    return {str(i): int(np.sum(values == i)) for i in range(1, 6)}


def load_archive(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        needed = {"X", "malignancy_reader_ratings", "malignancy_reader_mask", "patient_id", "nodule_id"}
        missing = needed - set(z.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}; found={z.files}")
        X = np.asarray(z["X"], dtype=np.float64)
        y = np.asarray(z["malignancy_reader_ratings"], dtype=np.float64)
        mask = np.asarray(z["malignancy_reader_mask"], dtype=bool)
        patients = z["patient_id"].astype(str).reshape(-1)
        nodules = z["nodule_id"].astype(str).reshape(-1)
        names = z["feature_names"].astype(str).reshape(-1).tolist() if "feature_names" in z.files else []
    if names and tuple(names[:6]) != FEATURES:
        raise ValueError(f"{path}: unexpected feature order {names}; expected {list(FEATURES)}")
    if X.ndim != 2 or X.shape[1] < 6:
        raise ValueError(f"{path}: expected X shape [n, >=6], got {X.shape}")
    if y.ndim != 2 or mask.shape != y.shape or len(X) != len(y):
        raise ValueError(f"{path}: reader ratings/mask must align with X")
    if len(patients) != len(X) or len(nodules) != len(X):
        raise ValueError(f"{path}: ID arrays do not align with X")
    X = X[:, :6]
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: non-finite Stage 03 feature scores")
    if np.any((X < 1 - 1e-4) | (X > 5 + 1e-4)):
        raise ValueError(f"{path}: Stage 03 scores outside [1,5]")
    valid_y = mask & np.isfinite(y) & (y >= 1) & (y <= 5)
    return X, y, valid_y, patients, nodules


def score_variant(model, X, y, valid_y, variant_name):
    pred_nodule = np.clip(np.asarray(model.predict(X), dtype=np.float64).reshape(-1), 1.0, 5.0)
    counts = valid_y.sum(axis=1)
    target_mean = np.divide(
        np.where(valid_y, y, 0.0).sum(axis=1),
        counts,
        out=np.full(len(y), np.nan, dtype=np.float64),
        where=counts > 0,
    )
    keep = np.isfinite(target_mean)
    y_mean = target_mean[keep]
    pred = pred_nodule[keep]
    true_class = round_half_up_clamped(y_mean)
    pred_class = round_half_up_clamped(pred)
    result = {
        "variant": variant_name,
        "n_nodules_with_target": int(keep.sum()),
        "n_reader_ratings": int(valid_y.sum()),
        "target_mean_rating": float(np.mean(y_mean)) if len(y_mean) else None,
        "target_class_counts_from_reader_mean": class_counts(true_class),
        "predicted_class_counts": class_counts(pred_class),
        "prediction_mean": float(np.mean(pred)) if len(pred) else None,
        "prediction_std": float(np.std(pred)) if len(pred) else None,
        "MAE_prediction_vs_reader_mean": float(mean_absolute_error(y_mean, pred)) if len(pred) else None,
        "QWK_rounded_reader_mean_vs_rounded_prediction": qwk(true_class, pred_class) if len(pred) else None,
        "accuracy_rounded_reader_mean_vs_rounded_prediction": float(accuracy_score(true_class, pred_class)) if len(pred) else None,
    }
    return result, pred_nodule, target_mean


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature-dir", required=True, type=Path,
                    help="Directory containing frozen_features val.npz and test.npz")
    ap.add_argument("--model", required=True, type=Path,
                    help="xml_attribute_malignancy.joblib")
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    model = joblib.load(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "XML-rated attribute inputs used for training; frozen Stage 03 predicted attributes used for evaluation",
        "rounding": "nearest integer, half-up via floor(x+0.5), clamped to [1,5]",
        "feature_order": list(FEATURES),
        "target": "mean valid LIDC reader malignancy/suspicion rating per nodule",
        "note": "This comparison is subject to input-distribution shift. Test results should be reported once and not used for model selection.",
        "splits": {},
    }
    prediction_rows = []
    for split in ("val", "test"):
        X, y, valid_y, patients, nodules = load_archive(args.feature_dir / f"{split}.npz")
        X_round = round_half_up_clamped(X).astype(np.float64)
        cont, cont_pred, target_mean = score_variant(model, X, y, valid_y, "continuous")
        rounded, rounded_pred, _ = score_variant(model, X_round, y, valid_y, "rounded_half_up")
        results["splits"][split] = {
            "continuous": cont,
            "rounded_half_up": rounded,
            "difference_rounded_minus_continuous_MAE": (
                rounded["MAE_prediction_vs_reader_mean"] - cont["MAE_prediction_vs_reader_mean"]
                if rounded["MAE_prediction_vs_reader_mean"] is not None and cont["MAE_prediction_vs_reader_mean"] is not None else None
            ),
            "difference_rounded_minus_continuous_QWK": (
                rounded["QWK_rounded_reader_mean_vs_rounded_prediction"] - cont["QWK_rounded_reader_mean_vs_rounded_prediction"]
                if rounded["QWK_rounded_reader_mean_vs_rounded_prediction"] is not None and cont["QWK_rounded_reader_mean_vs_rounded_prediction"] is not None else None
            ),
        }
        for i in range(len(X)):
            prediction_rows.append({
                "split": split,
                "patient_id": patients[i],
                "nodule_id": nodules[i],
                "target_reader_mean": target_mean[i] if np.isfinite(target_mean[i]) else "",
                **{f"{FEATURES[j]}_stage03_continuous": float(X[i, j]) for j in range(6)},
                **{f"{FEATURES[j]}_stage03_rounded": int(X_round[i, j]) for j in range(6)},
                "malignancy_pred_continuous_inputs": float(cont_pred[i]),
                "malignancy_pred_rounded_inputs": float(rounded_pred[i]),
                "reader_ratings_available": int(valid_y[i].sum()),
            })

    metrics_path = args.output_dir / "continuous_vs_rounded_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    pred_path = args.output_dir / "heldout_predictions.csv"
    if prediction_rows:
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)
    print(json.dumps(results, indent=2))
    print(f"\nMetrics: {metrics_path.resolve()}")
    print(f"Per-nodule predictions: {pred_path.resolve()}")


if __name__ == "__main__":
    main()
