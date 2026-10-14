#!/usr/bin/env python3
"""Validation-only test of marginal feature-scale alignment for Stage 04.

The existing XML-trained Stage 04 model was fitted on six XML reader-rating
features (roughly ordinal 1..5), while at inference it receives frozen Stage 03
predictions. This script compares direct use of those Stage 03 values against a
train-statistics-only affine mapping that matches each feature's Stage 03 TRAIN
mean/std to the XML TRAIN table's mean/std. It does not retrain Stage 03, parse
XML/DICOM, read H5/CT, or open test.npz.

This is a diagnostic experiment, not automatic promotion of the aligned model.
"""
from __future__ import annotations

import argparse
import csv
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
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error

FEATURES = ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"]
EPS = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decode_strings(arr: np.ndarray) -> list[str]:
    out = []
    for v in np.asarray(arr).tolist():
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(str(v))
    return out


def load_archive(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} archive: {path}")
    # Explicitly only caller paths train.npz and val.npz are loaded. This script
    # never constructs or opens a test.npz path.
    with np.load(path, allow_pickle=False) as z:
        needed = {"X", "feature_names", "patient_id", "malignancy_reader_ratings", "malignancy_reader_mask"}
        missing = needed - set(z.files)
        if missing:
            raise KeyError(f"{label} archive missing keys: {sorted(missing)}")
        X = np.asarray(z["X"], dtype=np.float64).copy()
        names = decode_strings(z["feature_names"])
        patients = np.asarray(decode_strings(z["patient_id"]), dtype=str)
        ratings = np.asarray(z["malignancy_reader_ratings"], dtype=np.float64).copy()
        mask = np.asarray(z["malignancy_reader_mask"], dtype=bool).copy()
    if names != FEATURES:
        raise ValueError(f"{label} feature order mismatch: {names}; expected {FEATURES}")
    if X.ndim != 2 or X.shape[1] != 6 or len(patients) != len(X):
        raise ValueError(f"{label} inconsistent X/patient shape: {X.shape}, {len(patients)}")
    if ratings.ndim != 2 or mask.shape != ratings.shape or ratings.shape[0] != len(X):
        raise ValueError(f"{label} inconsistent ratings/mask shapes")
    if not np.isfinite(X).all():
        raise ValueError(f"{label} contains non-finite Stage 03 input features")
    valid = mask & np.isfinite(ratings) & (ratings >= 1) & (ratings <= 5)
    counts = valid.sum(axis=1)
    keep = counts > 0
    medians = np.nanmedian(np.where(valid, ratings, np.nan), axis=1)
    means = np.nanmean(np.where(valid, ratings, np.nan), axis=1)
    return {
        "X": X[keep], "patients": patients[keep], "y_median": medians[keep], "y_mean": means[keep],
        "raw_rows": int(len(X)), "rows_used": int(keep.sum()), "reader_observations": int(valid[keep].sum()),
        "feature_names": names, "archive_sha256": sha256_file(path),
    }


def rounded_half_up(x: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(np.asarray(x, dtype=float) + 0.5), 1, 5).astype(int)


def metrics(y: np.ndarray, pred: np.ndarray, baseline: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 1.0, 5.0)
    yi, pi = rounded_half_up(y), rounded_half_up(pred)
    bi = rounded_half_up(np.full_like(y, float(baseline)))
    qwk = cohen_kappa_score(yi, pi, weights="quadratic")
    bqwk = cohen_kappa_score(yi, bi, weights="quadratic")
    mae = float(mean_absolute_error(y, pred))
    base_mae = float(mean_absolute_error(y, np.full_like(y, float(baseline))))
    return {
        "n_nodules": int(len(y)),
        "target_mean": float(np.mean(y)), "target_std": float(np.std(y)),
        "prediction_mean": float(np.mean(pred)), "prediction_std": float(np.std(pred)),
        "prediction_min": float(np.min(pred)), "prediction_max": float(np.max(pred)),
        "mae_continuous": mae, "rmse_continuous": float(math.sqrt(mean_squared_error(y, pred))),
        "constant_training_target_baseline": float(baseline), "constant_baseline_mae": base_mae,
        "mae_gain_over_baseline": base_mae - mae,
        "accuracy_rounded_half_up": float(accuracy_score(yi, pi)),
        "qwk_rounded_half_up": float(qwk) if np.isfinite(qwk) else None,
        "target_class_counts": {str(c): int((yi == c).sum()) for c in range(1, 6)},
        "prediction_class_counts": {str(c): int((pi == c).sum()) for c in range(1, 6)},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml-table", type=Path, required=True,
                   help="Existing nodule_level_xml_table.csv from XML training run")
    p.add_argument("--model", type=Path, required=True,
                   help="Existing malignancy_from_nodule_level_xml.joblib")
    p.add_argument("--features-dir", type=Path, required=True,
                   help="Directory containing the existing train.npz and val.npz only")
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    for path, label in [(a.xml_table, "XML feature table"), (a.model, "XML-trained model"),
                        (a.features_dir / "train.npz", "train feature archive"),
                        (a.features_dir / "val.npz", "validation feature archive")]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    out = a.out_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}; choose a new directory")

    # Only use XML rows explicitly labelled train to estimate training-domain stats.
    xml_train = []
    xml_train_y = []
    with a.xml_table.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"split", "malignancy_median", *FEATURES}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"XML table missing columns: {sorted(missing)}")
        for row in reader:
            if row.get("split") != "train":
                continue
            vals = [float(row[k]) for k in FEATURES]
            y = float(row["malignancy_median"])
            if np.isfinite(vals).all() and np.isfinite(y) and 1 <= y <= 5:
                xml_train.append(vals)
                xml_train_y.append(y)
    X_xml = np.asarray(xml_train, dtype=np.float64)
    y_xml = np.asarray(xml_train_y, dtype=np.float64)
    if X_xml.ndim != 2 or X_xml.shape[1] != 6 or len(X_xml) < 10:
        raise ValueError(f"Too few valid XML train rows: {len(X_xml)}")

    tr = load_archive(a.features_dir / "train.npz", "train")
    va = load_archive(a.features_dir / "val.npz", "validation")
    overlap = set(tr["patients"]).intersection(va["patients"])
    if overlap:
        raise ValueError(f"Patient leakage between train/validation archives: {len(overlap)}")

    # Distribution matching fitted only from the training split on both sides.
    mu_xml = X_xml.mean(axis=0)
    sd_xml = X_xml.std(axis=0)
    mu_ranker = tr["X"].mean(axis=0)
    sd_ranker = tr["X"].std(axis=0)
    nearly_constant = [FEATURES[i] for i, sd in enumerate(sd_ranker) if sd < EPS]
    if nearly_constant:
        raise ValueError(f"Cannot scale nearly constant ranker feature(s): {nearly_constant}; std < {EPS}")

    model_bundle = joblib.load(a.model)
    if not isinstance(model_bundle, dict) or "model" not in model_bundle:
        raise ValueError("Model artifact is not the expected dict bundle containing key 'model'")
    if model_bundle.get("feature_names_in_order", model_bundle.get("feature_names")) != FEATURES:
        raise ValueError("Model artifact feature order does not match the six-feature inference contract")
    model = model_bundle["model"]

    raw_pred = np.clip(model.predict(va["X"]), 1.0, 5.0)
    # Map a ranker output to its standardized position in ranker TRAIN outputs,
    # then map that position to the XML TRAIN feature distribution.
    X_val_aligned = ((va["X"] - mu_ranker) / sd_ranker) * sd_xml + mu_xml
    aligned_pred = np.clip(model.predict(X_val_aligned), 1.0, 5.0)

    # Baselines derive only from XML TRAIN labels. Compare each mapping against
    # the held-out ranker validation archive's reader medians and means.
    train_median = float(np.median(y_xml))
    train_mean = float(np.mean(y_xml))
    raw_metrics = {
        "against_validation_median_target": metrics(va["y_median"], raw_pred, train_median),
        "against_validation_mean_target": metrics(va["y_mean"], raw_pred, train_mean),
    }
    aligned_metrics = {
        "against_validation_median_target": metrics(va["y_median"], aligned_pred, train_median),
        "against_validation_mean_target": metrics(va["y_mean"], aligned_pred, train_mean),
    }
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "feature_scale_alignment_validation_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "target_median", "target_mean", "raw_prediction", "aligned_prediction", *[f"raw_{x}" for x in FEATURES], *[f"aligned_{x}" for x in FEATURES]])
        for i in range(len(va["X"])):
            w.writerow([va["patients"][i], f"{va['y_median'][i]:.6f}", f"{va['y_mean'][i]:.6f}",
                        f"{raw_pred[i]:.6f}", f"{aligned_pred[i]:.6f}",
                        *[f"{v:.8f}" for v in va["X"][i]], *[f"{v:.6f}" for v in X_val_aligned[i]]])

    report = {
        "status": "completed",
        "experiment": "train-statistics-only affine input-distribution alignment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__,
        "features_in_order": FEATURES,
        "xml_table": str(a.xml_table), "xml_table_sha256": sha256_file(a.xml_table),
        "model": str(a.model), "model_sha256": sha256_file(a.model),
        "train_archive": {"path": str(a.features_dir / "train.npz"), "sha256": tr["archive_sha256"], "rows_used": tr["rows_used"]},
        "validation_archive": {"path": str(a.features_dir / "val.npz"), "sha256": va["archive_sha256"], "rows_used": va["rows_used"]},
        "xml_training_rows": int(len(X_xml)),
        "train_validation_patient_overlap": len(overlap),
        "mapping_statistics": {"xml_train_mean": mu_xml.tolist(), "xml_train_std": sd_xml.tolist(),
                               "ranker_train_mean": mu_ranker.tolist(), "ranker_train_std": sd_ranker.tolist()},
        "raw_ranker_inputs": raw_metrics,
        "aligned_ranker_inputs": aligned_metrics,
        "outputs": {"predictions_csv": str(pred_path)},
        "test_archive_opened": False, "h5_or_ct_read": False, "stage03_rankers_modified": False,
        "warning": "An improvement on validation is diagnostic only. The XML grouping run produced more usable groups than the fixed feature archives; verify annotation universe alignment before promoting this model."
    }
    (out / "feature_scale_alignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status":"completed", "report_path":str(out / "feature_scale_alignment_report.json"),
                      "predictions_path":str(pred_path), "xml_training_rows":len(X_xml),
                      "raw_ranker_inputs":raw_metrics, "aligned_ranker_inputs":aligned_metrics,
                      "test_archive_opened":False, "h5_or_ct_read":False, "stage03_rankers_modified":False}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
