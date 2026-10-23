#!/usr/bin/env python3
"""Validation-only malignancy rounding and reader-median experiment.

Reads ONLY frozen_features/train.npz and frozen_features/val.npz. It does not
open test.npz, any H5/CT file, or load/retrain the Stage 03 attribute rankers.
The six Stage 03 inputs and existing patient split remain unchanged.

Questions answered:
  1) For the fixed current ExtraTrees model, compare floor, round-half-up, and
     ceiling on validation accuracy and quadratic weighted kappa (QWK).
  2) Fit the exact same ExtraTrees architecture/hyperparameters once to the
     per-nodule mean reader rating and once to the per-nodule median reader
     rating. Compare both models against the SAME validation-median target,
     alongside a constant train-median baseline.
  3) Quantify each MAE gain over its target-matched constant baseline, including
     a patient-cluster bootstrap interval as a rough uncertainty check.

This is diagnostic validation analysis, not a final model-selection claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error

FEATURES = ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"]
MODEL_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 4,
    "max_features": 1.0,
    "random_state": 2026,
    "n_jobs": -1,
}
ROUNDING_RULES = ("floor", "round_half_up", "ceiling")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def to_strings(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v) for v in values.tolist()]


def round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(values, dtype=float) + 0.5).astype(int)


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    val = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return float(val) if np.isfinite(val) else None


def apply_rounding(values: np.ndarray, rule: str) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=float), 1.0, 5.0)
    if rule == "floor":
        result = np.floor(x)
    elif rule == "round_half_up":
        result = np.floor(x + 0.5)
    elif rule == "ceiling":
        result = np.ceil(x)
    else:
        raise ValueError(f"Unknown rounding rule: {rule}")
    return np.clip(result, 1, 5).astype(int)


def load_split(path: Path, split_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {split_name} archive: {path}\n"
            "Expected existing frozen_features/train.npz and frozen_features/val.npz. "
            "Copy/run this script inside LungInsight_v2; it will not rebuild the archives."
        )
    required = {"X", "feature_names", "patient_id", "malignancy_reader_ratings", "malignancy_reader_mask"}
    with np.load(path, allow_pickle=False) as z:
        missing = required.difference(z.files)
        if missing:
            raise KeyError(f"{path} missing required keys: {sorted(missing)}")
        X = np.asarray(z["X"], dtype=np.float64)
        feature_names = to_strings(np.asarray(z["feature_names"]))
        patients_all = np.asarray(to_strings(np.asarray(z["patient_id"])), dtype=str)
        ratings = np.asarray(z["malignancy_reader_ratings"], dtype=np.float64)
        mask = np.asarray(z["malignancy_reader_mask"], dtype=bool)

    if feature_names != FEATURES:
        raise ValueError(f"{split_name} feature order mismatch: expected {FEATURES}; found {feature_names}")
    if X.ndim != 2 or X.shape[1] != 6 or not np.isfinite(X).all():
        raise ValueError(f"{split_name} X must be finite with shape (N,6); found {X.shape}")
    if patients_all.shape != (len(X),):
        raise ValueError(f"{split_name} patient_id row count does not match X")
    if ratings.ndim != 2 or mask.shape != ratings.shape or ratings.shape[0] != len(X):
        raise ValueError(f"{split_name} reader ratings/mask have inconsistent shapes")

    valid = mask & np.isfinite(ratings) & (ratings >= 1.0) & (ratings <= 5.0)
    counts = valid.sum(axis=1)
    keep = counts > 0
    if not keep.any():
        raise ValueError(f"{split_name} has no nodules with valid malignancy ratings")
    safe = np.where(valid, ratings, np.nan)
    y_mean_all = np.nanmean(safe, axis=1)
    y_median_all = np.nanmedian(safe, axis=1)
    return {
        "X": X[keep],
        "patients": patients_all[keep],
        "patients_all": patients_all,
        "y_mean": y_mean_all[keep].astype(float),
        "y_median": y_median_all[keep].astype(float),
        "ratings_per_nodule": counts[keep].astype(int),
        "reader_observations": int(valid.sum()),
        "rows_raw": int(len(X)),
        "rows_used": int(keep.sum()),
        "rows_dropped_no_target": int((~keep).sum()),
        "sha256": sha256_file(path),
    }


def clustered_bootstrap_gain(
    y: np.ndarray,
    pred: np.ndarray,
    baseline_value: float,
    patients: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 20261012,
) -> dict[str, Any]:
    """Patient-cluster bootstrap CI for baseline MAE minus model MAE; positive is better."""
    unique = np.unique(patients)
    by_patient = {p: np.flatnonzero(patients == p) for p in unique}
    rng = np.random.default_rng(seed)
    gains = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([by_patient[p] for p in sampled])
        base_mae = mean_absolute_error(y[ix], np.full(len(ix), baseline_value, dtype=float))
        model_mae = mean_absolute_error(y[ix], pred[ix])
        gains[b] = base_mae - model_mae
    return {
        "resampling_unit": "patient_id",
        "bootstrap_replicates": n_boot,
        "seed": seed,
        "mae_gain_baseline_minus_model": float(mean_absolute_error(y, np.full(len(y), baseline_value)) - mean_absolute_error(y, pred)),
        "gain_ci_95_percentile": [float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))],
        "ci_excludes_zero_positive_gain": bool(np.quantile(gains, 0.025) > 0.0),
        "note": "Exploratory uncertainty interval; validation data are diagnostic, not a final test set.",
    }


def target_metrics(y: np.ndarray, pred: np.ndarray, baseline_value: float, patients: np.ndarray) -> dict[str, Any]:
    pred = np.clip(np.asarray(pred, dtype=float), 1.0, 5.0)
    base = np.full_like(y, float(baseline_value), dtype=float)
    model_mae = float(mean_absolute_error(y, pred))
    baseline_mae = float(mean_absolute_error(y, base))
    return {
        "n_nodules": int(len(y)),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
        "mae": model_mae,
        "constant_baseline_value": float(baseline_value),
        "constant_baseline_mae": baseline_mae,
        "mae_gain_over_baseline": baseline_mae - model_mae,
        "relative_mae_gain_percent": (100.0 * (baseline_mae - model_mae) / baseline_mae) if baseline_mae else None,
        "patient_cluster_bootstrap": clustered_bootstrap_gain(y, pred, baseline_value, patients),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("backend/models/malignancy/frozen_features"))
    parser.add_argument("--out-dir", type=Path, default=Path("backend/models/malignancy/rounding_median_validation_v1"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Intentionally load exactly the existing train and validation files. Never touch test.npz.
    train = load_split(args.features_dir / "train.npz", "train")
    val = load_split(args.features_dir / "val.npz", "validation")
    overlap = sorted(set(train["patients_all"]) & set(val["patients_all"]))
    if overlap:
        raise ValueError(f"Train/validation patient overlap ({len(overlap)}); stopping. Examples: {overlap[:5]}")

    out = args.out_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}. Choose another --out-dir or use --overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    # Existing current downstream model configuration: same learner and hyperparameters
    # for mean-target and median-target fits, so target aggregation is the only change.
    mean_model = ExtraTreesRegressor(**MODEL_PARAMS)
    median_model = ExtraTreesRegressor(**MODEL_PARAMS)
    mean_model.fit(train["X"], train["y_mean"])
    median_model.fit(train["X"], train["y_median"])
    mean_pred = np.clip(mean_model.predict(val["X"]), 1.0, 5.0)
    median_pred = np.clip(median_model.predict(val["X"]), 1.0, 5.0)

    y_mean_val = val["y_mean"]
    y_median_val = val["y_median"]
    mean_baseline_value = float(np.mean(train["y_mean"]))
    median_baseline_value = float(np.median(train["y_median"]))

    # Rounding comparison is on the current mean-target ExtraTrees model, against the
    # established reader-mean target classes rounded half-up.
    y_class = np.clip(round_half_up(y_mean_val), 1, 5)
    mean_baseline_pred = np.full_like(y_mean_val, mean_baseline_value, dtype=float)
    rounding_results: dict[str, Any] = {}
    for name, continuous in (("current_extra_trees", mean_pred), ("constant_mean_baseline", mean_baseline_pred)):
        rounding_results[name] = {}
        for rule in ROUNDING_RULES:
            pclass = apply_rounding(continuous, rule)
            rounding_results[name][rule] = {
                "accuracy_exact_match": float(accuracy_score(y_class, pclass)),
                "qwk": qwk(y_class, pclass),
                "mae_between_integer_classes": float(mean_absolute_error(y_class, pclass)),
                "target_class_counts": {str(k): int((y_class == k).sum()) for k in range(1, 6)},
                "prediction_class_counts": {str(k): int((pclass == k).sum()) for k in range(1, 6)},
            }

    # Mean-trained and median-trained models are BOTH scored against validation medians,
    # so the core comparison uses the same target. Baseline is the train-set median of
    # per-nodule median targets.
    median_target_comparison = {
        "target_definition": "median of valid XML reader malignancy ratings per nodule; even-reader medians may be x.5",
        "validation_median_target_mean": float(np.mean(y_median_val)),
        "validation_median_target_std": float(np.std(y_median_val)),
        "mean_target_model_scored_against_validation_median": target_metrics(
            y_median_val, mean_pred, median_baseline_value, val["patients"]
        ),
        "median_target_model_scored_against_validation_median": target_metrics(
            y_median_val, median_pred, median_baseline_value, val["patients"]
        ),
        "constant_train_median_baseline": {
            "value": median_baseline_value,
            "mae": float(mean_absolute_error(y_median_val, np.full_like(y_median_val, median_baseline_value))),
        },
    }

    mean_target_comparison = {
        "current_mean_target_model": target_metrics(y_mean_val, mean_pred, mean_baseline_value, val["patients"]),
        "constant_train_mean_baseline": {
            "value": mean_baseline_value,
            "mae": float(mean_absolute_error(y_mean_val, np.full_like(y_mean_val, mean_baseline_value))),
        },
    }

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "scikit_learn_version": sklearn.__version__,
        "features_in_fixed_order": FEATURES,
        "model": {"type": "ExtraTreesRegressor", "hyperparameters": MODEL_PARAMS},
        "split_integrity": {
            "train_nodules_with_target": train["rows_used"],
            "validation_nodules_with_target": val["rows_used"],
            "train_patient_count": int(len(set(train["patients_all"]))),
            "validation_patient_count": int(len(set(val["patients_all"]))),
            "patient_overlap_count": 0,
        },
        "inputs": {
            "train": {"path": str(args.features_dir / "train.npz"), "sha256": train["sha256"], "rows": train["rows_raw"], "used_rows": train["rows_used"], "valid_reader_observations": train["reader_observations"]},
            "validation": {"path": str(args.features_dir / "val.npz"), "sha256": val["sha256"], "rows": val["rows_raw"], "used_rows": val["rows_used"], "valid_reader_observations": val["reader_observations"]},
        },
        "stage03_feature_std": {
            "train": {FEATURES[i]: float(train["X"][:, i].std()) for i in range(6)},
            "validation": {FEATURES[i]: float(val["X"][:, i].std()) for i in range(6)},
            "note": "This summarizes cached Stage 03 values. XML annotation standard deviations are in diagnostics_report.json, not these NPZ arrays.",
        },
        "rounding_comparison_against_reader_mean_classes": {
            "target_class_definition": "per-nodule mean of valid reader ratings, rounded half-up and clipped to 1..5",
            "results": rounding_results,
            "best_current_model_rule_by_accuracy": max(ROUNDING_RULES, key=lambda r: rounding_results["current_extra_trees"][r]["accuracy_exact_match"]),
            "best_current_model_rule_by_qwk": max(ROUNDING_RULES, key=lambda r: (rounding_results["current_extra_trees"][r]["qwk"] if rounding_results["current_extra_trees"][r]["qwk"] is not None else -math.inf)),
            "note": "Only discrete output mapping changes here; continuous predictions and continuous MAE do not change.",
        },
        "mean_target_comparison": mean_target_comparison,
        "median_target_comparison": median_target_comparison,
        "test_archive_opened": False,
        "h5_or_ct_read": False,
        "attribute_models_loaded_or_retrained": False,
        "model_selection_or_checkpoint_promotion": False,
        "limitations": [
            "Only the fixed train and validation feature archives are read; the test archive is not opened or queried.",
            "All Stage 03 features are the existing cached outputs; the six-feature interface is unchanged.",
            "Bootstrap interval resamples patients, but the sample is still a single validation split and cannot replace test evaluation.",
            "Reader-assigned malignancy/suspicion rating is not pathology-confirmed malignancy probability.",
        ],
    }

    report_path = out / "rounding_median_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = out / "validation_predictions.csv"
    header = ["row_in_filtered_val", "patient_id", "reader_rating_count", "malignancy_reader_mean", "malignancy_reader_median", "mean_target_model_prediction", "median_target_model_prediction"]
    for model_name in ("current_extra_trees", "constant_mean_baseline"):
        for rule in ROUNDING_RULES:
            header.append(f"{model_name}_{rule}_class")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        rule_preds = {
            (model_name, rule): apply_rounding(pred, rule)
            for model_name, pred in (("current_extra_trees", mean_pred), ("constant_mean_baseline", mean_baseline_pred))
            for rule in ROUNDING_RULES
        }
        for i in range(len(y_mean_val)):
            row = [i, val["patients"][i], int(val["ratings_per_nodule"][i]), f"{y_mean_val[i]:.6f}", f"{y_median_val[i]:.6f}", f"{mean_pred[i]:.6f}", f"{median_pred[i]:.6f}"]
            row.extend(int(rule_preds[(model_name, rule)][i]) for model_name in ("current_extra_trees", "constant_mean_baseline") for rule in ROUNDING_RULES)
            w.writerow(row)

    print(json.dumps({
        "status": "completed",
        "report_path": str(report_path),
        "predictions_path": str(csv_path),
        "test_archive_opened": False,
        "h5_or_ct_read": False,
        "patient_overlap_count": 0,
        "best_rounding_by_accuracy": report["rounding_comparison_against_reader_mean_classes"]["best_current_model_rule_by_accuracy"],
        "best_rounding_by_qwk": report["rounding_comparison_against_reader_mean_classes"]["best_current_model_rule_by_qwk"],
        "rounding_metrics": rounding_results["current_extra_trees"],
        "mean_target_mae_vs_mean_target": mean_target_comparison["current_mean_target_model"]["mae"],
        "mean_target_constant_baseline_mae": mean_target_comparison["constant_train_mean_baseline"]["mae"],
        "mean_model_mae_vs_median_target": median_target_comparison["mean_target_model_scored_against_validation_median"]["mae"],
        "median_model_mae_vs_median_target": median_target_comparison["median_target_model_scored_against_validation_median"]["mae"],
        "median_constant_baseline_mae": median_target_comparison["constant_train_median_baseline"]["mae"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
