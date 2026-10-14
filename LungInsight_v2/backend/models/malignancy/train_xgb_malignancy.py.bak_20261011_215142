#!/usr/bin/env python3
"""Train/evaluate an XGBoost malignancy-rating model from frozen attributes only.

Training target: individual valid LIDC reader malignancy/suspicion ratings (1..5).
For nodules with multiple reader annotations, one training row is created per
valid reader rating, and observation weights sum to one per nodule. Output class
probabilities describe the distribution over reader ratings, not cancer risk.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from frozen_attribute_common import (  # noqa: E402
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_HASHES,
    FEATURE_NAMES,
    sha256_file,
    sha256_indices,
)

DEFAULT_FEATURE_DIR = HERE / "frozen_features"
DEFAULT_OUTPUT_DIR = HERE / "xgb_malignancy_model"
RATING_CLASSES = np.arange(1, 6, dtype=np.int32)

CANDIDATE_CONFIGS: list[dict[str, Any]] = [
    {"name": "depth2_regularized", "max_depth": 2, "min_child_weight": 3.0, "eta": 0.03, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 5.0, "reg_alpha": 0.0},
    {"name": "depth3_regularized", "max_depth": 3, "min_child_weight": 5.0, "eta": 0.03, "subsample": 0.85, "colsample_bytree": 0.90, "reg_lambda": 8.0, "reg_alpha": 0.0},
    {"name": "depth4_strong_regularization", "max_depth": 4, "min_child_weight": 8.0, "eta": 0.02, "subsample": 0.80, "colsample_bytree": 0.85, "reg_lambda": 12.0, "reg_alpha": 0.1},
]


def fail_if_nonfinite_features(name: str, X: np.ndarray) -> None:
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"{name}: expected X shape [N,{len(FEATURE_NAMES)}], got {X.shape}")
    if not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))[:10].tolist()
        raise ValueError(f"{name}: non-finite features found at (row,column) positions {bad}")


def load_archive(path: Path, expected_split: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {expected_split} feature archive: {path}. Run generate_frozen_attribute_features.py first.")
    with np.load(path, allow_pickle=False) as z:
        required = {
            "X", "feature_names", "sample_index", "identity_key", "patient_id",
            "malignancy_reader_ratings", "malignancy_reader_mask", "malignancy_mean",
            "malignancy_median", "malignancy_reader_count",
        }
        missing = sorted(required - set(z.files))
        if missing:
            raise ValueError(f"{path}: archive schema is missing fields {missing}")
        data = {key: np.asarray(z[key]) for key in required}
    names = tuple(str(v) for v in data["feature_names"].tolist())
    if names != FEATURE_NAMES:
        raise ValueError(f"{path}: feature schema mismatch. Expected {FEATURE_NAMES}, got {names}")
    X = np.asarray(data["X"], dtype=np.float32)
    fail_if_nonfinite_features(expected_split, X)
    n = len(X)
    indices = np.asarray(data["sample_index"], dtype=np.int64)
    if n != EXPECTED_SPLIT_COUNTS[expected_split]:
        raise ValueError(f"{path}: expected canonical {expected_split} count {EXPECTED_SPLIT_COUNTS[expected_split]}, got {n}")
    observed_split_hash = sha256_indices(indices)
    if observed_split_hash != EXPECTED_SPLIT_HASHES[expected_split]:
        raise ValueError(
            f"{path}: ordered sample_index hash does not match the canonical {expected_split} split. "
            "Do not reorder or regenerate the split."
        )
    for field in ("sample_index", "identity_key", "patient_id", "malignancy_mean", "malignancy_median", "malignancy_reader_count"):
        if len(data[field]) != n:
            raise ValueError(f"{path}: field {field} has {len(data[field])} rows, expected {n}")
    ratings = np.asarray(data["malignancy_reader_ratings"], dtype=np.float32)
    mask = np.asarray(data["malignancy_reader_mask"], dtype=bool)
    if ratings.ndim != 2 or mask.shape != ratings.shape or ratings.shape[0] != n:
        raise ValueError(f"{path}: reader rating/mask shapes disagree: ratings={ratings.shape}, mask={mask.shape}, X={X.shape}")
    if np.any(mask & (~np.isfinite(ratings) | (ratings < 1) | (ratings > 5))):
        raise ValueError(f"{path}: reader_mask marks a rating invalidly outside finite [1,5]")
    if np.any((~mask) & np.isfinite(ratings)):
        # Invalid data may be present outside mask, but generator's contract stores NaN.
        raise ValueError(f"{path}: invalid/masked reader target values should be stored as NaN")
    if len(set(str(v) for v in data["patient_id"].tolist())) == 0:
        raise ValueError(f"{path}: no patient IDs present")
    if np.any(np.asarray(data["malignancy_reader_count"]) != mask.sum(axis=1)):
        raise ValueError(f"{path}: malignancy_reader_count does not agree with mask")
    if np.any((mask.sum(axis=1) == 0) & np.isfinite(data["malignancy_mean"])):
        raise ValueError(f"{path}: a nodule with zero valid readers has a finite malignancy_mean")
    return data


def validate_feature_manifest(feature_dir: Path) -> dict[str, Any]:
    manifest_path = feature_dir / "feature_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing provenance manifest {manifest_path}; do not train on unverified/renamed archives.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage03_training_or_finetuning_performed") is not False:
        raise ValueError("Feature manifest does not confirm frozen inference-only Stage 03 checkpoints")
    if tuple(manifest.get("features", [])) != FEATURE_NAMES:
        raise ValueError("Feature manifest does not contain the exact six-feature spatial_hybrid schema")
    if manifest.get("canonical_split_hashes") != EXPECTED_SPLIT_HASHES:
        raise ValueError("Feature manifest canonical split hashes do not match the expected split")
    if manifest.get("canonical_split_counts") != EXPECTED_SPLIT_COUNTS:
        raise ValueError("Feature manifest split counts do not match the canonical split")
    for group in ("lobulation_spiculation", "subtlety", "sphericity", "margin", "texture"):
        meta = manifest.get("checkpoints", {}).get(group)
        if not meta or not meta.get("sha256"):
            raise ValueError(f"Missing checkpoint hash/provenance for {group} in feature manifest")
        checkpoint_path = Path(meta.get("path", ""))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Frozen checkpoint from manifest is no longer present: {checkpoint_path}")
        if sha256_file(checkpoint_path) != meta["sha256"]:
            raise ValueError(f"Frozen Stage 03 checkpoint hash changed since feature generation: {checkpoint_path}")
    return manifest


def patient_overlap_check(train: dict[str, np.ndarray], val: dict[str, np.ndarray], test_ids: np.ndarray) -> None:
    sets = {
        "train": set(str(v) for v in train["patient_id"].tolist()),
        "val": set(str(v) for v in val["patient_id"].tolist()),
        "test": set(str(v) for v in test_ids.tolist()),
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            raise ValueError(f"Patient leakage between {a} and {b}: {sorted(overlap)[:10]}")


def validate_ids(data: dict[str, np.ndarray], name: str) -> None:
    sample_indices = np.asarray(data["sample_index"], dtype=np.int64)
    if len(np.unique(sample_indices)) != len(sample_indices):
        raise ValueError(f"{name}: duplicate sample_index values")
    identity_keys = [str(v) for v in data["identity_key"].tolist()]
    if len(set(identity_keys)) != len(identity_keys):
        raise ValueError(f"{name}: duplicate nodule identity_key values")
    if any(not str(v).strip() for v in data["patient_id"].tolist()):
        raise ValueError(f"{name}: missing patient IDs")


def expand_reader_observations(data: dict[str, np.ndarray], split_name: str):
    """Expand nodule rows into valid reader-observation labels, weighted per nodule."""
    X = np.asarray(data["X"], dtype=np.float32)
    ratings = np.asarray(data["malignancy_reader_ratings"], dtype=np.float32)
    mask = np.asarray(data["malignancy_reader_mask"], dtype=bool)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    source_rows: list[int] = []
    per_nodule_counts = mask.sum(axis=1)
    for i in range(len(X)):
        values = ratings[i, mask[i]]
        if not len(values):
            continue
        if not np.all(np.equal(values, np.floor(values))):
            raise ValueError(f"{split_name}: reader malignancy ratings must be integer values 1..5; sample row {i} has {values}")
        count = len(values)
        for value in values.astype(np.int32):
            if value < 1 or value > 5:
                raise ValueError(f"{split_name}: invalid reader malignancy rating {value}")
            rows.append(X[i])
            labels.append(int(value - 1))  # XGBoost multi:softprob uses class indices 0..4.
            weights.append(1.0 / float(count))
            source_rows.append(i)
    if not rows:
        raise ValueError(f"{split_name}: no valid reader malignancy observations")
    return (
        np.stack(rows).astype(np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(weights, dtype=np.float32),
        np.asarray(source_rows, dtype=np.int64),
        per_nodule_counts.astype(np.int32),
    )


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2 or np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return None
    try:
        from scipy.stats import spearmanr
        value = float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def compute_metrics(data: dict[str, np.ndarray], probabilities: np.ndarray, split_name: str) -> dict[str, Any]:
    try:
        from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for QWK/MAE/accuracy metrics. Install scikit-learn.") from exc
    X = np.asarray(data["X"], dtype=np.float32)
    mean_target = np.asarray(data["malignancy_mean"], dtype=np.float32)
    median_target = np.asarray(data["malignancy_median"], dtype=np.float32)
    reader_ratings = np.asarray(data["malignancy_reader_ratings"], dtype=np.float32)
    reader_mask = np.asarray(data["malignancy_reader_mask"], dtype=bool)
    counts = reader_mask.sum(axis=1)
    if probabilities.shape != (len(X), 5):
        raise ValueError(f"{split_name}: expected probability matrix {(len(X),5)}, got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{split_name}: model produced non-finite class probabilities")
    if np.any(probabilities < -1e-6) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"{split_name}: model probability rows do not sum to 1")
    ratings = RATING_CLASSES.astype(np.float64)
    expected = probabilities @ ratings
    predicted_class = probabilities.argmax(axis=1).astype(np.int32) + 1

    node_ok = (counts > 0) & np.isfinite(mean_target)
    if not node_ok.any():
        raise ValueError(f"{split_name}: no nodules with a valid reader-mean malignancy target")
    node_y_mean = mean_target[node_ok].astype(np.float64)
    node_y_rounded = np.floor(node_y_mean + 0.5).astype(np.int32).clip(1, 5)
    node_pred_expected = expected[node_ok]
    node_pred_class = predicted_class[node_ok]
    node_qwk = cohen_kappa_score(node_y_rounded, node_pred_class, weights="quadratic") if len(node_y_rounded) else float("nan")
    node_acc = accuracy_score(node_y_rounded, node_pred_class) if len(node_y_rounded) else float("nan")
    node_mae = mean_absolute_error(node_y_mean, node_pred_expected) if len(node_y_mean) else float("nan")

    # Evaluate against each individual reader rating as well as the per-nodule consensus.
    row_idx: list[int] = []
    reader_y: list[int] = []
    reader_weight: list[float] = []
    for i in range(len(X)):
        values = reader_ratings[i, reader_mask[i]]
        if not len(values):
            continue
        for value in values:
            row_idx.append(i)
            reader_y.append(int(value))
            reader_weight.append(1.0 / float(len(values)))
    row_idx_np = np.asarray(row_idx, dtype=np.int64)
    reader_y_np = np.asarray(reader_y, dtype=np.int32)
    reader_weight_np = np.asarray(reader_weight, dtype=np.float64)
    reader_pred_expected = expected[row_idx_np]
    reader_pred_class = predicted_class[row_idx_np]
    reader_qwk = cohen_kappa_score(reader_y_np, reader_pred_class, weights="quadratic", sample_weight=reader_weight_np)
    reader_acc = accuracy_score(reader_y_np, reader_pred_class, sample_weight=reader_weight_np)
    reader_mae = mean_absolute_error(reader_y_np, reader_pred_expected, sample_weight=reader_weight_np)

    return {
        "split": split_name,
        "nodules_total": int(len(X)),
        "nodules_with_valid_malignancy_labels": int(node_ok.sum()),
        "nodules_without_valid_malignancy_labels": int((~node_ok).sum()),
        "valid_reader_observations": int(len(reader_y_np)),
        "rating_counts_reader_observations": {str(k): int(np.sum(reader_y_np == k)) for k in RATING_CLASSES},
        "nodule_consensus": {
            "target": "mean of valid reader malignancy ratings (continuous 1..5)",
            "quadratic_weighted_kappa": float(node_qwk) if math.isfinite(float(node_qwk)) else None,
            "mae": float(node_mae) if math.isfinite(float(node_mae)) else None,
            "accuracy_rounded_consensus_vs_predicted_class": float(node_acc) if math.isfinite(float(node_acc)) else None,
            "spearman_rho_expected_rating_vs_reader_mean": safe_spearman(node_y_mean, node_pred_expected),
            "median_target_mae_diagnostic": float(mean_absolute_error(median_target[node_ok], node_pred_expected)) if np.isfinite(median_target[node_ok]).all() else None,
        },
        "reader_observation": {
            "target": "each valid individual LIDC reader's malignancy/suspicion rating; reader observations weighted so each nodule has total weight 1",
            "quadratic_weighted_kappa": float(reader_qwk) if math.isfinite(float(reader_qwk)) else None,
            "mae_expected_rating_vs_reader_rating": float(reader_mae) if math.isfinite(float(reader_mae)) else None,
            "accuracy_predicted_class_vs_reader_rating": float(reader_acc) if math.isfinite(float(reader_acc)) else None,
            "spearman_rho_expected_rating_vs_reader_rating_unweighted": safe_spearman(reader_y_np.astype(np.float64), reader_pred_expected),
        },
    }


def predict_probabilities(booster: Any, X: np.ndarray, xgb: Any) -> np.ndarray:
    matrix = xgb.DMatrix(X, feature_names=list(FEATURE_NAMES))
    pred = np.asarray(booster.predict(matrix), dtype=np.float64)
    return pred.reshape(len(X), 5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-rounds", type=int, default=500)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    feature_dir = a.feature_dir.resolve()
    output_dir = a.output_dir.resolve()
    if a.max_rounds < 1 or a.early_stopping_rounds < 1:
        raise ValueError("--max-rounds and --early-stopping-rounds must be positive")
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is missing from the active l-env. Install it with: python -m pip install xgboost") from exc
    try:
        from sklearn.metrics import cohen_kappa_score, mean_absolute_error
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required. Install it with: python -m pip install scikit-learn") from exc

    feature_manifest = validate_feature_manifest(feature_dir)
    for split_name in ("train", "val", "test"):
        split_info = feature_manifest.get("splits", {}).get(split_name, {})
        expected_archive_hash = split_info.get("archive_sha256")
        archive_path = feature_dir / f"{split_name}.npz"
        if expected_archive_hash and sha256_file(archive_path) != expected_archive_hash:
            raise ValueError(f"Feature archive hash mismatch for {archive_path}; regenerate features rather than reuse altered files.")
    train = load_archive(feature_dir / "train.npz", "train")
    val = load_archive(feature_dir / "val.npz", "val")
    validate_ids(train, "train")
    validate_ids(val, "val")
    with np.load(feature_dir / "test.npz", allow_pickle=False) as z:
        if "patient_id" not in z.files or "sample_index" not in z.files:
            raise ValueError("test.npz lacks identity fields needed for leakage checking")
        test_patient_ids = np.asarray(z["patient_id"]).astype(str)
        test_sample_indices = np.asarray(z["sample_index"], dtype=np.int64)
    if len(np.unique(test_sample_indices)) != len(test_sample_indices):
        raise ValueError("test: duplicate sample_index values")
    patient_overlap_check(train, val, test_patient_ids)
    if len(train["X"]) != EXPECTED_SPLIT_COUNTS["train"] or len(val["X"]) != EXPECTED_SPLIT_COUNTS["val"] or len(test_patient_ids) != EXPECTED_SPLIT_COUNTS["test"]:
        raise ValueError("Feature archive row counts disagree with the canonical split. Re-run feature generation with the unchanged split.")
    X_train_obs, y_train, w_train, source_train, train_counts = expand_reader_observations(train, "train")
    X_val_obs, y_val, w_val, source_val, val_counts = expand_reader_observations(val, "val")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training split has fewer than two distinct malignancy rating classes")
    dtrain = xgb.DMatrix(X_train_obs, label=y_train, weight=w_train, feature_names=list(FEATURE_NAMES))
    dval = xgb.DMatrix(X_val_obs, label=y_val, weight=w_val, feature_names=list(FEATURE_NAMES))

    base_params = {
        "objective": "multi:softprob",
        "num_class": 5,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "seed": int(a.seed),
        "verbosity": 0,
    }
    candidate_results: list[dict[str, Any]] = []
    candidate_boosters: list[Any] = []
    print(f"XGBoost={xgb.__version__}; train nodules={len(train['X'])}, valid reader ratings={len(y_train)}; val nodules={len(val['X'])}, valid reader ratings={len(y_val)}")
    for config in CANDIDATE_CONFIGS:
        params = dict(base_params)
        for key in ("max_depth", "min_child_weight", "eta", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha"):
            params[key] = config[key]
        print(f"Training candidate {config['name']}...", flush=True)
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=int(a.max_rounds),
            evals=[(dval, "validation")],
            early_stopping_rounds=int(a.early_stopping_rounds),
            verbose_eval=False,
        )
        best_iteration = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
        # Core Booster predict honors iteration_range explicitly; use it here so
        # candidate selection uses the validation-logloss best iteration only.
        p_val_obs = np.asarray(booster.predict(dval, iteration_range=(0, best_iteration + 1))).reshape(-1, 5)
        p_val_nodes = np.asarray(booster.predict(xgb.DMatrix(val["X"], feature_names=list(FEATURE_NAMES)), iteration_range=(0, best_iteration + 1))).reshape(-1, 5)
        metrics = compute_metrics(val, p_val_nodes, "validation")
        qwk = metrics["nodule_consensus"]["quadratic_weighted_kappa"]
        mae = metrics["nodule_consensus"]["mae"]
        result = {
            "name": config["name"],
            "parameters": {k: config[k] for k in config if k != "name"},
            "best_iteration_zero_based": best_iteration,
            "best_rounds": best_iteration + 1,
            "best_validation_mlogloss": float(getattr(booster, "best_score", float("nan"))),
            "validation_metrics": metrics,
            "selection_qwk": qwk,
            "selection_mae": mae,
            "reader_observation_rows_train": int(len(y_train)),
            "reader_observation_rows_validation": int(len(y_val)),
        }
        candidate_results.append(result)
        candidate_boosters.append(booster)
        print(f"  best_rounds={best_iteration + 1}; validation QWK={qwk}; MAE={mae}", flush=True)

    def selection_key(i: int):
        r = candidate_results[i]
        qwk = r["selection_qwk"]
        mae = r["selection_mae"]
        # Highest QWK first; when QWK is undefined, use lower MAE; MAE also breaks QWK ties.
        return (float(qwk) if qwk is not None else -float("inf"), -float(mae) if mae is not None else -float("inf"))

    selected_index = max(range(len(candidate_results)), key=selection_key)
    selected_result = candidate_results[selected_index]
    selected_booster_full = candidate_boosters[selected_index]
    selected_rounds = int(selected_result["best_rounds"])
    # Strip patience rounds from the saved Booster. No validation/test labels are
    # used to fit a new model after selection; the saved model is this selected train-only fit.
    try:
        selected_booster = selected_booster_full[:selected_rounds]
    except Exception as exc:
        raise RuntimeError(
            "The installed XGBoost cannot trim a Booster to its validation-selected best iteration. "
            "Upgrade xgboost (python -m pip install --upgrade xgboost) rather than saving a model with extra patience rounds."
        ) from exc
    if int(selected_booster.num_boosted_rounds()) != selected_rounds:
        raise RuntimeError(
            f"Saved Booster has {selected_booster.num_boosted_rounds()} rounds, but validation selected {selected_rounds}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "malignancy_xgb.json"
    selected_booster.save_model(str(model_path))
    (output_dir / "feature_schema.json").write_text(json.dumps({
        "schema_version": "1.0",
        "model_type": "XGBoost multi:softprob, 5 classes",
        "feature_names_in_order": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "input_semantics": "Expected reader ratings from six fixed Stage 03 hybrid rankers; no reader/XML target labels used as features",
        "feature_rating_scales": {name: [1, 5] for name in FEATURE_NAMES},
        "missing_feature_policy": "Reject missing/non-finite attributes; no silent imputation",
        "rating_class_mapping": {str(i): int(i + 1) for i in range(5)},
        "output_probability_semantics": "Probability distribution over LIDC reader malignancy/suspicion rating classes 1..5, not calibrated cancer probabilities",
        "expected_rating_formula": "sum(probability[r] * r for r in 1..5)",
        "malignancy_or_cancer_probability": False,
    }, indent=2), encoding="utf-8")

    # Final held-out test is loaded/scored only after model/hyperparameter/iteration selection.
    test = load_archive(feature_dir / "test.npz", "test")
    validate_ids(test, "test")
    patient_overlap_check(train, val, np.asarray(test["patient_id"]).astype(str))
    test_probs = predict_probabilities(selected_booster, test["X"], xgb)
    test_metrics = compute_metrics(test, test_probs, "test")
    # Recompute the selected validation result in the recorded exact iteration space.
    selected_val_probs = predict_probabilities(selected_booster, val["X"], xgb)
    selected_validation_metrics = compute_metrics(val, selected_val_probs, "validation_selected_model")

    metrics_payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_description": "LIDC reader-rated malignancy/suspicion (1..5), not confirmed cancer and not a calibrated cancer probability",
        "validation_used_for_model_selection": True,
        "test_used_for_model_selection": False,
        "selected_candidate": selected_result["name"],
        "selected_rounds": selected_rounds,
        "selection_rule": "Maximize validation nodule-consensus quadratic weighted kappa; break ties using lower MAE; validation logloss early-stopping determines each candidate's iteration.",
        "selected_candidate_validation_metrics": selected_result["validation_metrics"],
        "selected_model_validation_metrics": selected_validation_metrics,
        "test_metrics_once_after_selection": test_metrics,
        "candidate_validation_results": candidate_results,
        "interpretation_warning": (
            "Train feature predictions are in-sample because the six frozen Stage 03 rankers were trained on the canonical train split. "
            "They are not out-of-fold predictions. Validation/test ranker predictions are on patient-disjoint held-out splits. "
            "Do not interpret reported metrics as leakage-free generalization from fully out-of-sample Stage 03 features."
        ),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, allow_nan=False), encoding="utf-8")

    train_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "xgboost_version": xgb.__version__,
        "base_parameters": base_params,
        "candidate_configs": CANDIDATE_CONFIGS,
        "max_rounds": int(a.max_rounds),
        "early_stopping_rounds": int(a.early_stopping_rounds),
        "seed": int(a.seed),
        "target_observations": "one training row per valid individual reader rating, rating encoded 0..4 for XGBoost, mapped back to 1..5 for reported outputs",
        "sample_weighting": "Each reader observation has weight 1/(number of valid malignancy readers for its nodule), so total row weight per nodule is one.",
        "rows": {
            "train_nodules": int(len(train["X"])),
            "train_valid_reader_observations": int(len(y_train)),
            "train_nodules_without_valid_label": int((train["malignancy_reader_count"] == 0).sum()),
            "validation_nodules": int(len(val["X"])),
            "validation_valid_reader_observations": int(len(y_val)),
            "validation_nodules_without_valid_label": int((val["malignancy_reader_count"] == 0).sum()),
            "test_nodules": int(len(test["X"])),
            "test_valid_reader_observations": int(np.asarray(test["malignancy_reader_mask"], dtype=bool).sum()),
            "test_nodules_without_valid_label": int((test["malignancy_reader_count"] == 0).sum()),
        },
        "feature_manifest_sha256": sha256_file(feature_dir / "feature_manifest.json"),
        "h5_sha256": feature_manifest.get("h5_sha256"),
        "split_file_sha256": feature_manifest.get("split_file_sha256"),
        "canonical_split_hashes": EXPECTED_SPLIT_HASHES,
        "stage03_checkpoint_sha256": {g: d["sha256"] for g, d in feature_manifest["checkpoints"].items()},
        "stage03_checkpoints_modified": False,
        "stage04_inference_implemented": False,
    }
    (output_dir / "training_config.json").write_text(json.dumps(train_config, indent=2, allow_nan=False), encoding="utf-8")
    training_manifest = {
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "feature_schema_path": str(output_dir / "feature_schema.json"),
        "metrics_path": str(metrics_path),
        "training_config_path": str(output_dir / "training_config.json"),
        "feature_directory": str(feature_dir),
        "stage03_weights_modified": False,
        "warning": "Training rows contain fixed-ranker train-split predictions, not OOF predictions.",
    }
    (output_dir / "model_manifest.json").write_text(json.dumps(training_manifest, indent=2), encoding="utf-8")

    print("\nSelected XGBoost malignancy-rating model")
    print(f"Candidate: {selected_result['name']} | rounds={selected_rounds}")
    print(f"Model: {model_path}")
    print("Held-out test metrics (only after validation selection):")
    print(json.dumps(test_metrics, indent=2))
    print(f"Metrics: {metrics_path}")
    print("NOTE: outputs predict LIDC reader suspicion ratings 1..5, not cancer probabilities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
