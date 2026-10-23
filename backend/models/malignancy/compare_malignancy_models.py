#!/usr/bin/env python3
"""Summarize existing XGBoost and ordinal-MLP malignancy metrics side-by-side.

This is a reporting-only script. It does not fit models, alter checkpoints, or
choose hyperparameters. Prefer validation metrics for model decisions; report
test metrics as held-out final comparisons, not as tuning criteria.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def score_row(model_name: str, split_name: str, metric: dict[str, Any]) -> dict[str, Any]:
    consensus = metric.get("nodule_consensus", {})
    return {
        "model": model_name,
        "split": split_name,
        "QWK": consensus.get("quadratic_weighted_kappa"),
        "MAE": consensus.get("mae"),
        "accuracy": consensus.get("accuracy_rounded_consensus_vs_predicted_class"),
        "Spearman": consensus.get("spearman_rho_expected_rating_vs_reader_mean"),
        "nodules": metric.get("nodules_total"),
    }


def format_num(v: Any, width: int = 9) -> str:
    if v is None:
        return f"{'NA':>{width}}"
    if isinstance(v, (float, int)):
        return f"{float(v):>{width}.4f}"
    return f"{str(v):>{width}}"


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--xgb-dir", type=Path, default=HERE / "xgb_malignancy_model")
    p.add_argument("--mlp-dir", type=Path, default=HERE / "mlp_malignancy_model")
    args = p.parse_args()

    xgb_dir = args.xgb_dir.resolve()
    mlp_dir = args.mlp_dir.resolve()
    xgb_metrics = read_json(xgb_dir / "metrics.json")
    mlp_metrics = read_json(mlp_dir / "metrics.json")
    xgb_config_path = xgb_dir / "training_config.json"
    mlp_config_path = mlp_dir / "training_config.json"
    xgb_config = read_json(xgb_config_path) if xgb_config_path.exists() else {}
    mlp_config = read_json(mlp_config_path) if mlp_config_path.exists() else {}

    xgb_manifest_hash = xgb_config.get("feature_manifest_sha256")
    mlp_manifest_hash = mlp_config.get("feature_manifest_sha256")
    if xgb_manifest_hash and mlp_manifest_hash and xgb_manifest_hash != mlp_manifest_hash:
        raise ValueError(
            "The two models were trained from different feature manifests. "
            "Do not report this as a controlled comparison; regenerate/retrain against the same archives."
        )
    xgb_split_hashes = xgb_config.get("canonical_split_hashes")
    mlp_split_hashes = mlp_config.get("canonical_split_hashes")
    if xgb_split_hashes and mlp_split_hashes and xgb_split_hashes != mlp_split_hashes:
        raise ValueError("Canonical split hashes differ between model configs; comparison aborted.")

    rows = [
        score_row("XGBoost", "validation", xgb_metrics.get("selected_model_validation_metrics", {})),
        score_row("Ordinal MLP", "validation", mlp_metrics.get("selected_model_validation_metrics", {})),
        score_row("XGBoost", "test", xgb_metrics.get("test_metrics_once_after_selection", {})),
        score_row("Ordinal MLP", "test", mlp_metrics.get("test_metrics_once_after_selection", {})),
    ]
    headers = ("Model", "Split", "QWK", "MAE", "Accuracy", "Spearman", "Nodules")
    print("\nMalignancy-model comparison (same six frozen Stage 03 inputs)")
    print("-" * 94)
    print(f"{headers[0]:<14}{headers[1]:<12}{headers[2]:>9}{headers[3]:>9}{headers[4]:>11}{headers[5]:>11}{headers[6]:>10}")
    print("-" * 94)
    for row in rows:
        print(
            f"{row['model']:<14}{row['split']:<12}"
            f"{format_num(row['QWK'])}{format_num(row['MAE'])}"
            f"{format_num(row['accuracy'], 11)}{format_num(row['Spearman'], 11)}"
            f"{format_num(row['nodules'], 10)}"
        )
    print("-" * 94)
    print("Validation metrics are for model development. Test metrics are final held-out descriptive results;")
    print("do not tune either model or repeatedly choose between them using test results.")
    print("QWK/accuracy use rounded reader-mean ratings; MAE/Spearman use expected ordinal ratings.")
    print("Neither model output is a confirmed cancer diagnosis or a calibrated cancer probability.")
    if xgb_manifest_hash and mlp_manifest_hash:
        print(f"Same feature manifest SHA-256: {xgb_manifest_hash}")
    else:
        print("WARNING: one or both configs lack the feature-manifest hash; archive identity could not be confirmed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
