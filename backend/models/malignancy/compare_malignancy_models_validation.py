#!/usr/bin/env python3
"""Compare a small, fixed set of malignancy regressors using train + validation NPZ only.

Reads only frozen_features/train.npz and val.npz. Never opens test.npz, H5, CT,
or imports/runs/retrains the attribute models. Primary model-selection metric is
continuous validation MAE; QWK/rounded metrics are descriptive secondary metrics.
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
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def strings(a: np.ndarray) -> list[str]:
    return [x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x) for x in a.tolist()]


def load_archive(path: Path, split: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} archive: {path}")
    with np.load(path, allow_pickle=False) as z:
        required = {"X", "feature_names", "patient_id", "malignancy_reader_ratings", "malignancy_reader_mask"}
        missing = required.difference(z.files)
        if missing:
            raise KeyError(f"{path} missing keys: {sorted(missing)}")
        X = np.asarray(z["X"], dtype=np.float64)
        names = strings(z["feature_names"])
        patients = np.asarray(strings(z["patient_id"]), dtype=str)
        ratings = np.asarray(z["malignancy_reader_ratings"], dtype=np.float64)
        mask = np.asarray(z["malignancy_reader_mask"], dtype=bool)
    if names != FEATURES:
        raise ValueError(f"{split}: feature order mismatch: {names}")
    if X.ndim != 2 or X.shape[1] != 6 or not np.isfinite(X).all():
        raise ValueError(f"{split}: X must be finite (N, 6), got {X.shape}")
    if patients.shape != (len(X),) or ratings.ndim != 2 or ratings.shape != mask.shape or ratings.shape[0] != len(X):
        raise ValueError(f"{split}: mismatched rows/reader rating dimensions")
    valid = mask & np.isfinite(ratings) & (ratings >= 1) & (ratings <= 5)
    nvalid = valid.sum(axis=1)
    keep = nvalid > 0
    if not keep.any():
        raise ValueError(f"{split}: no rows have valid malignancy labels")
    y = np.where(valid, ratings, 0).sum(axis=1) / np.maximum(nvalid, 1)
    return {
        "X": X[keep], "y": y[keep].astype(np.float64), "patients": patients[keep],
        "original_rows": np.flatnonzero(keep), "reader_observations": int(valid.sum()),
        "raw_rows": int(len(X)), "dropped_rows": int((~keep).sum()),
        "sha256": sha256_file(path),
    }


def round_half_up(x: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(np.asarray(x, dtype=float) + 0.5), 1, 5).astype(int)


def qwk(y: np.ndarray, p: np.ndarray) -> float | None:
    try:
        value = cohen_kappa_score(y, p, weights="quadratic")
        return float(value) if np.isfinite(value) else None
    except Exception:
        return None


def metrics(y: np.ndarray, pred: np.ndarray, train_mean: float) -> dict[str, Any]:
    pred = np.clip(np.asarray(pred, dtype=float), 1, 5)
    yr = round_half_up(y)
    pr = round_half_up(pred)
    br = round_half_up(np.full_like(y, train_mean, dtype=float))
    return {
        "n_nodules": int(len(y)),
        "mae_continuous": float(mean_absolute_error(y, pred)),
        "rmse_continuous": float(math.sqrt(mean_squared_error(y, pred))),
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
        "mae_rounded_half_up": float(mean_absolute_error(yr, pr)),
        "accuracy_rounded_half_up": float((yr == pr).mean()),
        "qwk_rounded_half_up": qwk(yr, pr),
        "rounded_prediction_counts": {str(k): int((pr == k).sum()) for k in range(1, 6)},
        "constant_train_mean_baseline_mae": float(mean_absolute_error(y, np.full_like(y, train_mean, dtype=float))),
        "constant_train_mean_baseline_rounded_mae": float(mean_absolute_error(yr, br)),
        "constant_train_mean_baseline_rounded_qwk": qwk(yr, br),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path, default=Path("backend/models/malignancy/frozen_features"))
    ap.add_argument("--out-dir", type=Path, default=Path("backend/models/malignancy/validation_model_comparison_v1"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    train = load_archive(args.features_dir / "train.npz", "train")
    val = load_archive(args.features_dir / "val.npz", "validation")
    overlap = sorted(set(train["patients"]).intersection(set(val["patients"])))
    if overlap:
        raise ValueError(f"Train/validation patient overlap ({len(overlap)}); stopping. Examples: {overlap[:5]}")

    out = args.out_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory not empty: {out}. Choose a new --out-dir or pass --overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    # Fixed, small candidate set. No hyperparameter sweep; only train and validation are read.
    candidates = {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "ridge_scaled_alpha_10": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest_leaf8": RandomForestRegressor(n_estimators=400, min_samples_leaf=8, max_features=1.0, random_state=2026, n_jobs=-1),
        "extra_trees_leaf8": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=8, max_features=1.0, random_state=2026, n_jobs=-1),
        "extra_trees_leaf4": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=4, max_features=1.0, random_state=2026, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=2.0, early_stopping=False, random_state=2026),
        "gradient_boosting_mae": GradientBoostingRegressor(loss="absolute_error", n_estimators=120, learning_rate=0.03, max_depth=2, min_samples_leaf=12, random_state=2026),
    }

    rows: list[dict[str, Any]] = []
    preds_by_model: dict[str, np.ndarray] = {}
    y_val = val["y"]
    train_mean = float(train["y"].mean())
    for name, model in candidates.items():
        model.fit(train["X"], train["y"])
        p = np.clip(model.predict(val["X"]), 1, 5)
        preds_by_model[name] = p
        m = metrics(y_val, p, train_mean)
        rows.append({"model": name, **m, "beats_mean_baseline_mae": bool(m["mae_continuous"] < m["constant_train_mean_baseline_mae"] - 1e-12)})

    # Select for reporting only, based on continuous validation MAE. The selected model is NOT
    # silently saved/promoted; inspect whether it meaningfully beats the baseline first.
    ranked = sorted(rows, key=lambda r: (r["mae_continuous"], -(r["qwk_rounded_half_up"] if r["qwk_rounded_half_up"] is not None else -1)))
    best = ranked[0]
    baseline = next(r for r in rows if r["model"] == "baseline_mean")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "scikit_learn_version": sklearn.__version__,
        "features": FEATURES,
        "primary_selection_metric": "continuous validation MAE (lower is better)",
        "split_integrity": {"train_nodules": int(len(train["y"])), "validation_nodules": int(len(val["y"])), "patient_overlap_count": len(overlap)},
        "input_archives": {
            "train": {"path": str(args.features_dir / "train.npz"), "sha256": train["sha256"], "rows": train["raw_rows"], "used_rows": len(train["y"]), "reader_observations": train["reader_observations"]},
            "validation": {"path": str(args.features_dir / "val.npz"), "sha256": val["sha256"], "rows": val["raw_rows"], "used_rows": len(val["y"]), "reader_observations": val["reader_observations"]},
        },
        "feature_std_train": {FEATURES[i]: float(train["X"][:, i].std()) for i in range(6)},
        "feature_std_validation": {FEATURES[i]: float(val["X"][:, i].std()) for i in range(6)},
        "results_ranked_by_validation_mae": ranked,
        "best_by_validation_mae": best["model"],
        "best_mae": best["mae_continuous"],
        "constant_mean_baseline_mae": baseline["mae_continuous"],
        "best_beats_baseline": bool(best["mae_continuous"] < baseline["mae_continuous"] - 1e-12),
        "test_archive_opened": False,
        "h5_or_ct_read": False,
        "attribute_models_retrained": False,
        "important_limitations": [
            "Train features are cached in-sample outputs from fixed attribute rankers; not out-of-fold predictions.",
            "This is a small, fixed model comparison, not a hyperparameter sweep.",
            "Best-by-validation is for diagnosis/model selection only; do not evaluate on the test set yet.",
            "Target is LIDC reader-assigned malignancy/suspicion mean, not pathology-confirmed malignancy probability.",
        ],
    }
    report_path = out / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = out / "validation_predictions_by_model.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_in_val_archive", "patient_id", "target_reader_mean", *[f"{k}_prediction" for k in candidates]])
        for j, (orig_ix, patient, target) in enumerate(zip(val["original_rows"], val["patients"], y_val)):
            w.writerow([int(orig_ix), patient, f"{target:.6f}", *[f"{preds_by_model[k][j]:.6f}" for k in candidates]])

    print(json.dumps({
        "status": "completed",
        "report_path": str(report_path),
        "predictions_path": str(csv_path),
        "test_archive_opened": False,
        "h5_or_ct_read": False,
        "best_by_validation_mae": report["best_by_validation_mae"],
        "best_validation_mae": report["best_mae"],
        "constant_mean_baseline_mae": report["constant_mean_baseline_mae"],
        "best_beats_baseline": report["best_beats_baseline"],
        "ranked_models": [{"model": r["model"], "mae": round(r["mae_continuous"], 4), "qwk": None if r["qwk_rounded_half_up"] is None else round(r["qwk_rounded_half_up"], 4)} for r in ranked],
    }, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
