#!/usr/bin/env python3
"""Train a Stage 04 malignancy-rating regressor from LIDC XML reader annotations.

The six input features are individual XML reader ratings, ordered to match the
frozen Stage 03 ranker-output contract. The saved joblib bundle can be consumed
by predict_malignancy_from_attributes.py with six scalar ranker outputs.

Uses the existing canonical patient split: XML rows from TRAIN patients fit the
model; XML rows from VAL patients report annotation-level validation. If the
frozen Stage 03 validation archive exists, it also evaluates the same model on
ranker outputs against that archive's nodule-level reader-mean targets. The test
feature archive is never opened and no test metrics are produced.

No H5/CT files or Stage 03 checkpoints are read or changed. This script fits one
predeclared model and does not tune hyperparameters against validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
)

FEATURES = ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"]
MODEL_PARAMS = {
    "n_estimators": 400,
    "min_samples_leaf": 8,
    "max_depth": None,
    "random_state": 2026,
    "n_jobs": -1,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def numeric_rating(value: Any) -> float:
    if value is None or isinstance(value, bool) or str(value).strip() == "":
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def patient_ids_for_split(split_obj: Any, name: str) -> set[str]:
    if not isinstance(split_obj, dict) or name not in split_obj:
        raise ValueError(f"split.json must contain '{name}'")
    item = split_obj[name]
    patients = item.get("patients") if isinstance(item, dict) else item
    if not isinstance(patients, list):
        raise ValueError(f"split.json['{name}'] must be a list or contain a patients list")
    result = {str(x).strip() for x in patients if str(x).strip()}
    if not result:
        raise ValueError(f"No patient IDs found for split '{name}'")
    return result


def xml_rows_for_splits(annotation_obj: Any, train_patients: set[str], val_patients: set[str]):
    if not isinstance(annotation_obj, dict) or not isinstance(annotation_obj.get("patients"), list):
        raise ValueError("Expected lidc_all_annotations_match.json structure with a 'patients' list")

    interested = train_patients | val_patients
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    patients_seen: set[str] = set()

    for patient in annotation_obj["patients"]:
        patient_id = str(patient.get("patient_id", "")).strip()
        if not patient_id or patient_id not in interested:
            # Deliberately ignore all patient records outside train/validation.
            continue
        patients_seen.add(patient_id)
        split_name = "train" if patient_id in train_patients else "val"
        for study in patient.get("studies", []) or []:
            if not study.get("matched", False):
                stats[f"{split_name}_unmatched_studies"] += 1
                continue
            for session in study.get("reading_sessions", []) or []:
                reader_id = str(session.get("servicing_radiologist_id") or "unknown")
                for nodule in session.get("nodules", []) or []:
                    stats[f"{split_name}_annotation_records_seen"] += 1
                    y = numeric_rating(nodule.get("malignancy"))
                    xs = [numeric_rating(nodule.get(name)) for name in FEATURES]
                    if not math.isfinite(y) or not 1 <= y <= 5:
                        stats[f"{split_name}_dropped_bad_target"] += 1
                        continue
                    if not all(math.isfinite(v) and 1 <= v <= 5 for v in xs):
                        stats[f"{split_name}_dropped_incomplete_or_out_of_range_features"] += 1
                        continue
                    rows.append({
                        "split": split_name,
                        "patient_id": patient_id,
                        "study_instance_uid": str(study.get("study_instance_uid") or ""),
                        "series_instance_uid": str(study.get("series_instance_uid") or ""),
                        "reader_id": reader_id,
                        "nodule_id": str(nodule.get("nodule_id") or ""),
                        "X": xs,
                        "y": float(y),
                    })
                    stats[f"{split_name}_usable_reader_rows"] += 1
                    stats[f"{split_name}_target_class_{int(y)}"] += 1

    missing_train = train_patients - patients_seen
    missing_val = val_patients - patients_seen
    stats["train_patients_with_xml_records"] = len(train_patients & patients_seen)
    stats["val_patients_with_xml_records"] = len(val_patients & patients_seen)
    stats["train_patients_without_xml_records"] = len(missing_train)
    stats["val_patients_without_xml_records"] = len(missing_val)
    return rows, dict(stats)


def rounded_half_up(values: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(np.asarray(values, dtype=float) + 0.5), 1, 5).astype(int)


def score(y: np.ndarray, pred: np.ndarray, baseline: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 1.0, 5.0)
    y_int = rounded_half_up(y)
    p_int = rounded_half_up(pred)
    baseline_pred = np.full(len(y), float(baseline))
    b_int = rounded_half_up(baseline_pred)
    qwk = cohen_kappa_score(y_int, p_int, weights="quadratic")
    base_qwk = cohen_kappa_score(y_int, b_int, weights="quadratic")
    return {
        "n_rows": int(len(y)),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
        "mae_continuous": float(mean_absolute_error(y, pred)),
        "rmse_continuous": float(math.sqrt(mean_squared_error(y, pred))),
        "constant_train_mean": float(baseline),
        "constant_baseline_mae": float(mean_absolute_error(y, baseline_pred)),
        "mae_gain_over_baseline": float(mean_absolute_error(y, baseline_pred) - mean_absolute_error(y, pred)),
        "accuracy_rounded_half_up": float(accuracy_score(y_int, p_int)),
        "qwk_rounded_half_up": float(qwk) if math.isfinite(float(qwk)) else None,
        "baseline_accuracy_rounded_half_up": float(accuracy_score(y_int, b_int)),
        "baseline_qwk_rounded_half_up": float(base_qwk) if math.isfinite(float(base_qwk)) else None,
        "target_integer_counts": {str(c): int((y_int == c).sum()) for c in range(1, 6)},
        "prediction_integer_counts": {str(c): int((p_int == c).sum()) for c in range(1, 6)},
    }


def write_prediction_csv(path: Path, rows: list[dict[str, Any]], y: np.ndarray, pred: np.ndarray, target_name: str):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "patient_id", "nodule_id", "reader_id", target_name,
                         "predicted_malignancy_continuous", "predicted_malignancy_round_half_up"])
        for i, (row, target, value) in enumerate(zip(rows, y, pred)):
            writer.writerow([i, row.get("patient_id", ""), row.get("nodule_id", ""),
                             row.get("reader_id", ""), f"{float(target):.6f}", f"{float(value):.6f}",
                             int(rounded_half_up(np.asarray([value]))[0])])


def load_ranker_validation(path: Path):
    required = {"X", "feature_names", "patient_id", "malignancy_reader_ratings", "malignancy_reader_mask"}
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise KeyError(f"Ranker validation archive {path} missing keys: {sorted(missing)}")
        X = np.asarray(archive["X"], dtype=np.float64)
        feature_names = [v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
                         for v in np.asarray(archive["feature_names"]).tolist()]
        raw_patients = np.asarray(archive["patient_id"])
        patients = np.asarray([v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
                               for v in raw_patients.tolist()], dtype=str)
        ratings = np.asarray(archive["malignancy_reader_ratings"], dtype=np.float64)
        mask = np.asarray(archive["malignancy_reader_mask"], dtype=bool)
    if feature_names != FEATURES:
        raise ValueError(f"Ranker validation feature order mismatch: {feature_names}")
    if X.ndim != 2 or X.shape[1] != 6 or len(patients) != len(X):
        raise ValueError(f"Bad ranker validation X/patient shapes: {X.shape}, {patients.shape}")
    if ratings.ndim != 2 or ratings.shape != mask.shape or ratings.shape[0] != len(X):
        raise ValueError("Reader-rating and mask arrays have inconsistent shapes")
    valid = mask & np.isfinite(ratings) & (ratings >= 1) & (ratings <= 5)
    n_readers = valid.sum(axis=1)
    keep = (n_readers > 0) & np.isfinite(X).all(axis=1)
    if not keep.any():
        raise ValueError("No valid nodule rows in ranker validation archive")
    safe = np.where(valid[keep], ratings[keep], np.nan)
    y = np.nanmean(safe, axis=1)
    return {"X": X[keep], "patients": patients[keep], "y": y.astype(float),
            "raw_rows": int(len(X)), "rows_used": int(keep.sum()),
            "valid_reader_observations": int(valid[keep].sum()),
            "archive_sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("datasets/LIDC/lidc_all_annotations_match.json"))
    parser.add_argument("--split-json", type=Path,
                        default=Path("backend/models/nodule_ranker_3d_distributional_v1/split.json"))
    parser.add_argument("--ranker-val-archive", type=Path,
                        default=Path("backend/models/malignancy/frozen_features/val.npz"),
                        help="Optional existing Stage 03 validation archive; only val.npz is read")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("backend/models/malignancy/xml_annotation_ranker_compatible_v1"))
    parser.add_argument("--skip-ranker-validation", action="store_true",
                        help="Skip Stage 03-output validation; XML annotation validation is still reported")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path, description in ((args.annotations, "XML-derived annotation JSON"), (args.split_json, "canonical split JSON")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.out_dir}; choose another directory or pass --overwrite")

    split_obj = read_json(args.split_json)
    train_patients = patient_ids_for_split(split_obj, "train")
    val_patients = patient_ids_for_split(split_obj, "val")
    overlap = train_patients & val_patients
    if overlap:
        raise ValueError(f"Canonical train/val patient overlap ({len(overlap)}): {sorted(overlap)[:5]}")

    annotations_obj = read_json(args.annotations)
    rows, parse_stats = xml_rows_for_splits(annotations_obj, train_patients, val_patients)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if not train_rows or not val_rows:
        raise RuntimeError(f"Need valid XML rows in both splits; train={len(train_rows)}, val={len(val_rows)}")

    X_train = np.asarray([r["X"] for r in train_rows], dtype=np.float64)
    y_train = np.asarray([r["y"] for r in train_rows], dtype=np.float64)
    X_val = np.asarray([r["X"] for r in val_rows], dtype=np.float64)
    y_val = np.asarray([r["y"] for r in val_rows], dtype=np.float64)
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Non-finite XML feature remained after row validation")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training malignancy ratings contain fewer than two classes")

    model = ExtraTreesRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    xml_val_pred = np.clip(model.predict(X_val), 1.0, 5.0)
    train_mean = float(np.mean(y_train))
    xml_val_metrics = score(y_val, xml_val_pred, train_mean)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "malignancy_from_xml_annotations.joblib"
    bundle = {
        "model": model,
        "feature_names": FEATURES,
        "target": "LIDC reader-assigned malignancy/suspicion rating",
        "target_scale": [1, 5],
        "target_aggregation": "one target per reader-nodule XML annotation row",
        "input_semantics": "six ordinal attribute scores; compatible with frozen Stage 03 expected-rating scalar outputs",
        "model_type": "ExtraTreesRegressor",
        "hyperparameters": MODEL_PARAMS,
        "feature_training_source": "LIDC XML reader annotations",
        "patient_split": {"train": "canonical train patients", "validation": "canonical val patients"},
        "warning": "XML input validation does not measure accuracy when the model receives compressed Stage 03 outputs. See validation report for separate ranker-output validation if available.",
    }
    joblib.dump(bundle, model_path)

    write_prediction_csv(out / "xml_annotation_validation_predictions.csv", val_rows, y_val, xml_val_pred, "xml_reader_malignancy_rating")

    ranker_val_metrics = None
    ranker_val_path = None
    ranker_val_info = None
    if not args.skip_ranker_validation:
        ranker_val = load_ranker_validation(args.ranker_val_archive)
        if ranker_val is not None:
            overlap_with_train = set(ranker_val["patients"]) & train_patients
            if overlap_with_train:
                raise ValueError(f"Ranker validation archive has train-patient overlap ({len(overlap_with_train)}); refusing validation")
            ranker_pred = np.clip(model.predict(ranker_val["X"]), 1.0, 5.0)
            ranker_val_metrics = score(ranker_val["y"], ranker_pred, train_mean)
            ranker_val_info = {
                "path": str(args.ranker_val_archive),
                "sha256": ranker_val["archive_sha256"],
                "raw_rows": ranker_val["raw_rows"],
                "rows_used": ranker_val["rows_used"],
                "valid_reader_observations": ranker_val["valid_reader_observations"],
                "patient_count": int(len(set(ranker_val["patients"]))),
                "patient_overlap_with_xml_train": 0,
                "target_unit": "per-nodule mean of valid XML malignancy reader ratings",
            }
            ranker_val_path = out / "stage03_validation_predictions.csv"
            # Keep a minimal row record without opening any test archive.
            ranker_rows = [{"patient_id": str(p), "nodule_id": "", "reader_id": ""} for p in ranker_val["patients"]]
            write_prediction_csv(ranker_val_path, ranker_rows, ranker_val["y"], ranker_pred, "nodule_mean_reader_malignancy_rating")
        elif args.ranker_val_archive.exists():
            raise RuntimeError(f"Unable to load ranker validation archive: {args.ranker_val_archive}")
        else:
            ranker_val_info = {"skipped": True, "reason": "validation archive not found; use --skip-ranker-validation to suppress this check"}

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "annotation_json": str(args.annotations),
        "annotation_json_sha256": sha256_file(args.annotations),
        "split_json": str(args.split_json),
        "split_json_sha256": sha256_file(args.split_json),
        "features_in_order": FEATURES,
        "training_rows_are_individual_reader_annotations": True,
        "model_hyperparameters": MODEL_PARAMS,
        "counts": {
            "train_patient_ids_in_split": len(train_patients),
            "validation_patient_ids_in_split": len(val_patients),
            "patient_overlap_train_validation": 0,
            "train_xml_annotation_rows_used": len(train_rows),
            "validation_xml_annotation_rows_used": len(val_rows),
            "train_xml_annotation_rows_dropped": int(parse_stats.get("train_annotation_records_seen", 0) - len(train_rows)),
            "validation_xml_annotation_rows_dropped": int(parse_stats.get("val_annotation_records_seen", 0) - len(val_rows)),
        },
        "xml_parse_stats": parse_stats,
        "validation_on_held_out_xml_reader_rows": xml_val_metrics,
        "ranker_output_validation": ranker_val_metrics,
        "ranker_validation_archive": ranker_val_info,
        "outputs": {
            "model": str(model_path),
            "xml_validation_predictions": str(out / "xml_annotation_validation_predictions.csv"),
            "ranker_validation_predictions": str(ranker_val_path) if ranker_val_path else None,
        },
        "test_feature_archive_opened": False,
        "test_annotations_used_for_training_or_model_selection": False,
        "h5_or_ct_read": False,
        "stage03_rankers_loaded_or_modified": False,
        "patient_split_changed": False,
        "limitations": [
            "Target is a radiologist's LIDC malignancy/suspicion rating, not pathology-confirmed cancer.",
            "Rows are individual reader-nodule annotations; reader-specific attribute scores at validation are not the same input distribution as one model score per nodule.",
            "The model can consume Stage 03 scalar outputs, but the ranker-output validation metrics—not XML-input validation metrics—are the closer estimate of deployment behavior.",
            "No test split was used for model selection or evaluation.",
        ],
    }
    report_path = out / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": "trained",
        "model_path": str(model_path),
        "report_path": str(report_path),
        "xml_validation_predictions_path": str(out / "xml_annotation_validation_predictions.csv"),
        "ranker_validation_predictions_path": str(ranker_val_path) if ranker_val_path else None,
        "train_xml_rows": len(train_rows),
        "validation_xml_rows": len(val_rows),
        "xml_validation_metrics": xml_val_metrics,
        "ranker_output_validation_metrics": ranker_val_metrics,
        "test_feature_archive_opened": False,
        "h5_or_ct_read": False,
        "stage03_rankers_modified": False,
        "patient_overlap_count": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
