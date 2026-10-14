#!/usr/bin/env python3
"""Train a small regularized ordinal MLP on frozen Stage 03 nodule attributes.

The model consumes only the six expected ratings produced by the existing frozen
spatial_hybrid Stage 03 rankers. It predicts LIDC reader malignancy/suspicion
ratings (1..5), not confirmed cancer and not a calibrated cancer probability.

Ordinal formulation
-------------------
A scalar latent score s(x) is passed through four ordered logistic thresholds:
    P(Y > k | x) = sigmoid(s(x) - threshold[k]), k=1..4.
The thresholds are constrained to be strictly increasing. The four cumulative
probabilities define valid class probabilities for ratings 1..5. Training uses
weighted cumulative binary cross-entropy over individual valid reader ratings;
reader-observation weights sum to one for each nodule. Validation ordinal loss
controls early stopping/model checkpoint selection. The held-out test split is
not used until training and checkpoint selection are complete.

This script shares the archive, provenance and metric contracts with the
existing train_xgb_malignancy.py script, but it does not require XGBoost.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from frozen_attribute_common import (  # noqa: E402
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_HASHES,
    FEATURE_NAMES,
    sha256_file,
)
from train_xgb_malignancy import (  # noqa: E402
    compute_metrics,
    expand_reader_observations,
    load_archive,
    patient_overlap_check,
    validate_feature_manifest,
    validate_ids,
)

DEFAULT_FEATURE_DIR = HERE / "frozen_features"
DEFAULT_OUTPUT_DIR = HERE / "mlp_malignancy_model"
RATING_VALUES = np.arange(1, 6, dtype=np.float32)
MIN_THRESHOLD_GAP = 1e-3


class OrdinalMLP(nn.Module):
    """6 -> 32 -> 16 -> scalar MLP with ordered cumulative-logit thresholds."""

    def __init__(
        self,
        n_features: int,
        dropout: float,
        initial_thresholds: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1, bias=False),
        )
        if initial_thresholds is None:
            initial_thresholds = np.asarray([-2.0, -1.2, 0.0, 1.1], dtype=np.float32)
        initial_thresholds = np.asarray(initial_thresholds, dtype=np.float64)
        if initial_thresholds.shape != (4,) or not np.isfinite(initial_thresholds).all():
            raise ValueError("initial_thresholds must contain four finite increasing values")
        if not np.all(np.diff(initial_thresholds) > 0):
            raise ValueError(f"initial_thresholds must be strictly increasing, got {initial_thresholds}")
        self.threshold_start = nn.Parameter(torch.tensor(float(initial_thresholds[0]), dtype=torch.float32))
        increments = np.diff(initial_thresholds) - MIN_THRESHOLD_GAP
        increments = np.maximum(increments, 1e-4)
        # inverse softplus, numerically stable for these small, positive increments
        raw_increments = np.log(np.expm1(increments))
        self.threshold_raw_increments = nn.Parameter(torch.as_tensor(raw_increments, dtype=torch.float32))

    def thresholds(self) -> torch.Tensor:
        increments = F.softplus(self.threshold_raw_increments) + MIN_THRESHOLD_GAP
        return torch.cat((self.threshold_start.reshape(1), self.threshold_start + torch.cumsum(increments, dim=0)))

    def cumulative_logits(self, x: torch.Tensor) -> torch.Tensor:
        score = self.network(x).reshape(-1, 1)
        return score - self.thresholds().reshape(1, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return four logits for P(Y>1), P(Y>2), P(Y>3), P(Y>4)."""
        return self.cumulative_logits(x)

    @torch.no_grad()
    def class_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        cumulative = torch.sigmoid(self.cumulative_logits(x))
        # Ordered thresholds guarantee cumulative[:,0] >= ... >= cumulative[:,3].
        probabilities = torch.cat(
            (
                1.0 - cumulative[:, 0:1],
                cumulative[:, 0:1] - cumulative[:, 1:2],
                cumulative[:, 1:2] - cumulative[:, 2:3],
                cumulative[:, 2:3] - cumulative[:, 3:4],
                cumulative[:, 3:4],
            ),
            dim=1,
        )
        probabilities = torch.clamp(probabilities, min=1e-8)
        return probabilities / probabilities.sum(dim=1, keepdim=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=50, help="Early-stopping patience on validation ordinal BCE")
    p.add_argument("--min-delta", type=float, default=1e-5, help="Minimum validation-loss improvement")
    p.add_argument("--learning-rate", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is False")
    return torch.device(name)


def initial_thresholds_from_train(labels_1_to_5: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Initialize thresholds to the train-only weighted empirical rating frequencies."""
    labels = np.asarray(labels_1_to_5, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)
    if labels.ndim != 1 or weights.shape != labels.shape or len(labels) == 0:
        raise ValueError("Expected non-empty one-dimensional train labels and matching observation weights")
    if np.any((labels < 1) | (labels > 5)) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Invalid train labels or sample weights")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Training sample weights sum to zero")
    thresholds = []
    for k in range(1, 5):
        p = float(weights[labels > k].sum() / total)
        p = float(np.clip(p, 1e-3, 1.0 - 1e-3))
        thresholds.append(math.log((1.0 - p) / p))
    arr = np.asarray(thresholds, dtype=np.float64)
    # Empirical cumulative frequencies are monotone. Clipping/numerical ties are
    # made strictly ordered to satisfy the ordinal parameterization.
    for i in range(1, len(arr)):
        arr[i] = max(arr[i], arr[i - 1] + 0.05)
    return arr.astype(np.float32)


def weighted_ordinal_bce(logits: torch.Tensor, labels_1_to_5: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError(f"Expected ordinal logits with shape [N,4], got {tuple(logits.shape)}")
    targets = torch.stack(
        [(labels_1_to_5 > k).to(dtype=logits.dtype) for k in range(1, 5)],
        dim=1,
    )
    row_losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
    denominator = torch.clamp(weights.sum(), min=1e-8)
    return (row_losses * weights).sum() / denominator


def probabilities_numpy(model: OrdinalMLP, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    with torch.inference_mode():
        result = model.class_probabilities(tensor).detach().cpu().numpy().astype(np.float64)
    if result.shape != (len(X), 5) or not np.isfinite(result).all():
        raise ValueError(f"Invalid class probabilities output shape/values: {result.shape}")
    if np.any(result < 0) or not np.allclose(result.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Ordinal MLP class probabilities are invalid")
    return result


def validation_ordinal_loss(
    model: OrdinalMLP,
    X_observations: np.ndarray,
    y_observations: np.ndarray,
    weights: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    x_t = torch.as_tensor(X_observations, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_observations, dtype=torch.long, device=device)
    w_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    with torch.inference_mode():
        loss = weighted_ordinal_bce(model(x_t), y_t, w_t)
    result = float(loss.item())
    if not math.isfinite(result):
        raise ValueError(f"Non-finite validation ordinal BCE: {result}")
    return result


def write_predictions_csv(path: Path, data: dict[str, np.ndarray], probabilities: np.ndarray) -> None:
    ratings = RATING_VALUES.astype(np.float64)
    expected = probabilities @ ratings
    predicted = probabilities.argmax(axis=1) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "sample_index", "identity_key", "patient_id", "malignancy_mean",
            "malignancy_median", "malignancy_reader_count", "expected_rating", "predicted_rating",
            *[f"probability_rating_{i}" for i in range(1, 6)],
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(len(probabilities)):
            row = {
                "sample_index": int(data["sample_index"][i]),
                "identity_key": str(data["identity_key"][i]),
                "patient_id": str(data["patient_id"][i]),
                "malignancy_mean": float(data["malignancy_mean"][i]) if np.isfinite(data["malignancy_mean"][i]) else "",
                "malignancy_median": float(data["malignancy_median"][i]) if np.isfinite(data["malignancy_median"][i]) else "",
                "malignancy_reader_count": int(data["malignancy_reader_count"][i]),
                "expected_rating": float(expected[i]),
                "predicted_rating": int(predicted[i]),
            }
            for j in range(5):
                row[f"probability_rating_{j + 1}"] = float(probabilities[i, j])
            writer.writerow(row)


def split_metadata_only_for_leak_check(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        if "patient_id" not in z.files or "sample_index" not in z.files:
            raise ValueError(f"{path} lacks patient_id/sample_index needed for leakage validation")
        patient_ids = np.asarray(z["patient_id"]).astype(str)
        sample_indices = np.asarray(z["sample_index"], dtype=np.int64)
    if len(np.unique(sample_indices)) != len(sample_indices):
        raise ValueError(f"{path}: duplicate sample_index values")
    return patient_ids, sample_indices


def main() -> int:
    args = parse_args()
    if args.max_epochs < 1 or args.patience < 1:
        raise ValueError("--max-epochs and --patience must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1 or args.min_delta < 0:
        raise ValueError("Invalid optimizer/dropout/early stopping parameter")

    set_seed(args.seed)
    device = choose_device(args.device)
    feature_dir = args.feature_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest = validate_feature_manifest(feature_dir)
    for split_name in ("train", "val", "test"):
        archive_path = feature_dir / f"{split_name}.npz"
        split_info = manifest.get("splits", {}).get(split_name, {})
        expected_hash = split_info.get("archive_sha256")
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing feature archive: {archive_path}")
        if expected_hash and sha256_file(archive_path) != expected_hash:
            raise ValueError(f"Feature archive hash mismatch: {archive_path}; regenerate frozen features")

    train = load_archive(feature_dir / "train.npz", "train")
    val = load_archive(feature_dir / "val.npz", "val")
    validate_ids(train, "train")
    validate_ids(val, "val")
    test_patient_ids, test_sample_indices = split_metadata_only_for_leak_check(feature_dir / "test.npz")
    patient_overlap_check(train, val, test_patient_ids)
    if len(test_sample_indices) != EXPECTED_SPLIT_COUNTS["test"]:
        raise ValueError(f"Expected {EXPECTED_SPLIT_COUNTS['test']} test rows, got {len(test_sample_indices)}")

    X_train_obs, y_train_zero, w_train, _source_train, train_counts = expand_reader_observations(train, "train")
    X_val_obs, y_val_zero, w_val, _source_val, val_counts = expand_reader_observations(val, "val")
    y_train = y_train_zero.astype(np.int64) + 1
    y_val = y_val_zero.astype(np.int64) + 1
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training reader labels contain fewer than two rating classes")

    # Fit feature normalization on training nodules only (not val/test).
    X_train_nodes = np.asarray(train["X"], dtype=np.float32)
    feature_mean = X_train_nodes.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = X_train_nodes.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    if not np.isfinite(feature_mean).all() or not np.isfinite(feature_scale).all():
        raise ValueError("Non-finite training feature normalization statistics")

    def transform(X: np.ndarray) -> np.ndarray:
        Xn = (np.asarray(X, dtype=np.float32) - feature_mean) / feature_scale
        if not np.isfinite(Xn).all():
            raise ValueError("Normalization produced non-finite features")
        return Xn.astype(np.float32, copy=False)

    X_train_obs_s = transform(X_train_obs)
    X_val_obs_s = transform(X_val_obs)
    X_train_nodes_s = transform(train["X"])
    X_val_nodes_s = transform(val["X"])
    train_weights_t = torch.as_tensor(w_train, dtype=torch.float32, device=device)
    train_x_t = torch.as_tensor(X_train_obs_s, dtype=torch.float32, device=device)
    train_y_t = torch.as_tensor(y_train, dtype=torch.long, device=device)

    initial_thresholds = initial_thresholds_from_train(y_train, w_train)
    model = OrdinalMLP(len(FEATURE_NAMES), args.dropout, initial_thresholds).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    print(
        f"Device={device}; train nodules={len(train['X'])}, reader observations={len(y_train)}; "
        f"val nodules={len(val['X'])}, reader observations={len(y_val)}; "
        f"features={len(FEATURE_NAMES)}",
        flush=True,
    )
    print("Ordinal MLP: 6 -> 32 -> 16 -> 1; ReLU; dropout; 4 ordered cumulative-logit thresholds", flush=True)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(train_x_t)
        train_loss = weighted_ordinal_bce(train_logits, train_y_t, train_weights_t)
        if not torch.isfinite(train_loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        val_loss = validation_ordinal_loss(model, X_val_obs_s, y_val, w_val, device)
        val_probs = probabilities_numpy(model, X_val_nodes_s, device)
        val_metrics = compute_metrics(val, val_probs, "validation")
        qwk = val_metrics["nodule_consensus"]["quadratic_weighted_kappa"]
        mae = val_metrics["nodule_consensus"]["mae"]
        history.append({
            "epoch": epoch,
            "train_ordinal_bce": float(train_loss.detach().cpu().item()),
            "validation_ordinal_bce": val_loss,
            "validation_qwk": qwk,
            "validation_mae": mae,
            "validation_spearman": val_metrics["nodule_consensus"]["spearman_rho_expected_rating_vs_reader_mean"],
            "validation_accuracy": val_metrics["nodule_consensus"]["accuracy_rounded_consensus_vs_predicted_class"],
        })

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch == 1 or epoch % 25 == 0 or epochs_since_improvement == 0:
            print(
                f"epoch={epoch:4d} train_ordinal_bce={float(train_loss.item()):.5f} "
                f"val_ordinal_bce={val_loss:.5f} val_QWK={qwk} val_MAE={mae}",
                flush=True,
            )
        if epochs_since_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best validation ordinal BCE was epoch {best_epoch} ({best_val_loss:.6f})", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training ended without a valid validation-selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "mlp_malignancy.pt"
    torch.save({
        "schema_version": "1.0",
        "model_type": "small_regularized_ordinal_mlp_cumulative_logit",
        "model_state_dict": best_state,
        "architecture": {
            "input_features": list(FEATURE_NAMES),
            "hidden_units": [32, 16],
            "activation": "ReLU",
            "dropout": float(args.dropout),
            "output": "scalar latent score with four strictly ordered cumulative-logit thresholds",
            "ordinal_rating_mapping": {"1": "Y<=1", "2": "Y=2", "3": "Y=3", "4": "Y=4", "5": "Y>4"},
        },
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "thresholds": model.thresholds().detach().cpu().numpy(),
        "rating_values": RATING_VALUES.astype(np.int32),
        "selected_epoch": int(best_epoch),
        "selected_validation_ordinal_bce": float(best_val_loss),
    }, checkpoint_path)

    # At this point all training/validation choices are fixed. Load test labels
    # only now and score the held-out test set once.
    test = load_archive(feature_dir / "test.npz", "test")
    validate_ids(test, "test")
    patient_overlap_check(train, val, np.asarray(test["patient_id"]).astype(str))
    test_probs = probabilities_numpy(model, transform(test["X"]), device)
    val_probs = probabilities_numpy(model, X_val_nodes_s, device)
    train_probs = probabilities_numpy(model, X_train_nodes_s, device)
    test_metrics = compute_metrics(test, test_probs, "test")
    selected_val_metrics = compute_metrics(val, val_probs, "validation_selected_model")
    train_metrics = compute_metrics(train, train_probs, "train_in_sample_diagnostic")

    # Constant baselines derived only from the training reader labels.
    prior = np.bincount(y_train - 1, weights=w_train, minlength=5).astype(np.float64)
    prior = prior / prior.sum()
    prior_probs = np.tile(prior.reshape(1, 5), (len(val["X"]), 1))
    majority_probs = np.zeros((len(val["X"]), 5), dtype=np.float64)
    majority_probs[:, int(np.argmax(prior))] = 1.0
    val_prior_metrics = compute_metrics(val, prior_probs, "validation_train_rating_prior_baseline")
    val_majority_metrics = compute_metrics(val, majority_probs, "validation_train_majority_class_baseline")
    test_prior_probs = np.tile(prior.reshape(1, 5), (len(test["X"]), 1))
    test_majority_probs = np.zeros((len(test["X"]), 5), dtype=np.float64)
    test_majority_probs[:, int(np.argmax(prior))] = 1.0
    test_prior_metrics = compute_metrics(test, test_prior_probs, "test_train_rating_prior_baseline")
    test_majority_metrics = compute_metrics(test, test_majority_probs, "test_train_majority_class_baseline")

    # Save per-epoch trace for auditing early stopping.
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(history[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    write_predictions_csv(output_dir / "predictions_train.csv", train, train_probs)
    write_predictions_csv(output_dir / "predictions_val.csv", val, val_probs)
    write_predictions_csv(output_dir / "predictions_test.csv", test, test_probs)

    feature_schema = {
        "schema_version": "1.0",
        "model_type": "small_regularized_ordinal_mlp_cumulative_logit",
        "feature_names_in_order": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "normalization": "z-score using train.npz nodule rows only; feature_mean/feature_scale stored in mlp_malignancy.pt",
        "rating_classes": [1, 2, 3, 4, 5],
        "probability_semantics": "distribution over LIDC reader malignancy/suspicion ratings, not a cancer probability",
        "expected_rating_formula": "sum(probability[r] * r for r in 1..5)",
        "missing_feature_policy": "reject missing or non-finite features; no silent imputation",
    }
    (output_dir / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")

    metrics_payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_description": "LIDC reader-rated malignancy/suspicion (1..5); not confirmed cancer and not a calibrated cancer probability",
        "validation_used_for_early_stopping": True,
        "test_used_for_model_selection": False,
        "selection_rule": "Minimum validation cumulative ordinal binary cross-entropy; patience-based early stopping. The test split is opened for labels only after checkpoint selection.",
        "selected_epoch": int(best_epoch),
        "selected_validation_ordinal_bce": float(best_val_loss),
        "train_in_sample_diagnostic_metrics": train_metrics,
        "selected_model_validation_metrics": selected_val_metrics,
        "test_metrics_once_after_selection": test_metrics,
        "validation_baselines": {
            "train_derived_rating_prior": val_prior_metrics,
            "train_derived_majority_class": val_majority_metrics,
        },
        "test_baselines": {
            "train_derived_rating_prior": test_prior_metrics,
            "train_derived_majority_class": test_majority_metrics,
        },
        "training_history_rows": len(history),
        "interpretation_warning": (
            "Frozen Stage 03 rankers were trained on canonical train nodules, so train feature predictions are in-sample, not OOF. "
            "Validation/test predictions use patient-disjoint splits. These metrics evaluate reader-rated suspicion labels, not cancer diagnosis."
        ),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, allow_nan=False), encoding="utf-8")

    training_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pytorch_version": torch.__version__,
        "device": str(device),
        "architecture": "6 -> 32 -> 16 -> 1; ReLU; dropout; four strictly ordered cumulative-logit thresholds",
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "seed": int(args.seed),
        "ordinal_loss": "weighted cumulative BCE over targets [rating>1, rating>2, rating>3, rating>4]",
        "sample_weighting": "Each reader observation has weight 1/(valid reader ratings for that nodule), so each nodule total training weight is 1.",
        "feature_manifest_sha256": sha256_file(feature_dir / "feature_manifest.json"),
        "h5_sha256": manifest.get("h5_sha256"),
        "split_file_sha256": manifest.get("split_file_sha256"),
        "canonical_split_hashes": EXPECTED_SPLIT_HASHES,
        "stage03_checkpoint_sha256": {g: d["sha256"] for g, d in manifest["checkpoints"].items()},
        "stage03_checkpoints_modified": False,
        "stage04_inference_implemented": False,
        "test_used_for_training_or_selection": False,
        "train_feature_provenance_warning": "Stage 03 train-split attributes are in-sample predictions, not OOF predictions.",
    }
    config_path = output_dir / "training_config.json"
    config_path.write_text(json.dumps(training_config, indent=2, allow_nan=False), encoding="utf-8")
    manifest_payload = {
        "model_path": str(checkpoint_path),
        "model_sha256": sha256_file(checkpoint_path),
        "feature_schema_path": str(output_dir / "feature_schema.json"),
        "metrics_path": str(metrics_path),
        "training_config_path": str(config_path),
        "training_history_path": str(output_dir / "training_history.csv"),
        "feature_directory": str(feature_dir),
        "stage03_weights_modified": False,
        "stage04_inference_implemented": False,
    }
    (output_dir / "model_manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    print("\nOrdinal MLP training completed")
    print(f"Selected epoch: {best_epoch}; best validation ordinal BCE={best_val_loss:.6f}")
    print(f"Model: {checkpoint_path}")
    print("Held-out test metrics (only after validation selection):")
    print(json.dumps(test_metrics, indent=2))
    print(f"Metrics: {metrics_path}")
    print("Reminder: these are LIDC reader suspicion ratings, not cancer probabilities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
