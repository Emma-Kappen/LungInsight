#!/usr/bin/env python3
"""Evaluate a malignancy suspicion model against reader-level ratings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error, roc_auc_score)

from malignancy_fusion import MalignancyFusion
from train_malignancy import choose_device, load_rows


def main(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_dim = int(checkpoint["architecture"]["input_dim"])
    device = choose_device(args.device)
    model = MalignancyFusion(input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    features, ratings, _, _ = load_rows(Path(args.features).resolve(), input_dim)
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    features = (features - mean) / std
    with torch.inference_mode():
        output = model.predict(torch.from_numpy(features).to(device))
    distribution = output["class_probabilities"].cpu().numpy()
    expected = output["malignancy_expected"].cpu().numpy()
    p_ge4 = output["probability_malignant_ge4"].cpu().numpy()
    truth = ratings.astype(np.int64)
    predicted = distribution.argmax(axis=1) + 1
    qwk = cohen_kappa_score(truth, predicted, weights="quadratic", labels=[1, 2, 3, 4, 5])
    rho = spearmanr(truth, expected).statistic if len(np.unique(truth)) > 1 else float("nan")
    auc = roc_auc_score(truth >= 4, p_ge4) if len(np.unique(truth >= 4)) > 1 else float("nan")
    report = {
        "interpretation": "Radiologist malignancy suspicion score, not a cancer diagnosis.",
        "n_reader_ratings": int(len(truth)),
        "quadratic_weighted_kappa": float(qwk),
        "spearman": float(rho),
        "mae": float(mean_absolute_error(truth, expected)),
        "rmse": float(np.sqrt(mean_squared_error(truth, expected))),
        "accuracy": float(accuracy_score(truth, predicted)),
        "p_malignant_ge4_roc_auc": float(auc),
        "confusion_matrix_labels_1_to_5": confusion_matrix(truth, predicted, labels=[1, 2, 3, 4, 5]).tolist(),
        "rating_interpretation": ["very unlikely", "unlikely", "indeterminate", "suspicious", "highly suspicious"],
    }
    serialized = json.dumps(report, indent=2, allow_nan=True)
    if args.output:
        Path(args.output).resolve().write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    raise SystemExit(main(parser.parse_args()))