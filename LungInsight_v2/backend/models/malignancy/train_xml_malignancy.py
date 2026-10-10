#!/usr/bin/env python3
"""Train and validation-select lightweight XML-only ordinal models."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from xml_malignancy_common import (
    FEATURE_COLUMNS,
    OUTPUT_DEFAULT,
    CumulativeOrdinalModel,
    evaluate_predictions,
    majority_and_median,
    make_evaluation_rows,
    rows_to_matrix,
    sha256_file,
)


ARTIFACT_NAMES = (
    "ordinal_model.pkl", "gradient_boosted_model.pkl", "best_model.pkl",
    "preprocessing.pkl", "model_selection.json", "history.csv",
)


def _load_nodule_rows(dataset_path: Path, split_names: set[str] | None = None) -> list[dict[str, Any]]:
    with dataset_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            row for row in reader
            if split_names is None or row.get("split") in split_names
        ]
    required = {"split", "target_ratings_json", *FEATURE_COLUMNS}
    missing = required - (set(rows[0]) if rows else set())
    if missing:
        raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")
    return rows


def _ratings_for_split(rows: list[dict[str, Any]], split_name: str) -> list[int]:
    return [
        int(rating)
        for row in rows if row["split"] == split_name
        for rating in json.loads(row["target_ratings_json"] or "[]")
    ]


def _constant_probabilities(ratings: np.ndarray, count: int) -> np.ndarray:
    probabilities = np.zeros((count, 5), dtype=float)
    if count:
        probabilities[:, int(ratings[0]) - 1] = 1.0
    return probabilities


def _baseline_reports(truth: np.ndarray, train_ratings: np.ndarray) -> dict[str, Any]:
    majority, median = majority_and_median(train_ratings)
    result = {}
    for name, rating in (("training_majority", majority), ("training_median", median)):
        baseline = _constant_probabilities(np.asarray([rating]), len(truth))
        result[name] = evaluate_predictions(truth, baseline)
        result[name]["constant_rating"] = rating
    return result


def _selection_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, str]:
    name, metrics = item
    qwk = metrics.get("qwk")
    mae = metrics.get("mae")
    return (
        float(qwk) if qwk is not None else -float("inf"),
        -float(mae) if mae is not None else -float("inf"),
        name,
    )


def train(data_dir: Path, seed: int = 42) -> None:
    start = time.perf_counter()
    out_dir = data_dir.resolve()
    for filename in ARTIFACT_NAMES:
        if (out_dir / filename).exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {out_dir / filename}")
    dataset_path = out_dir / "xml_features.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Structured dataset not found: {dataset_path}")
    rows = _load_nodule_rows(dataset_path, {"train", "val"})
    train_x, train_y, _ = make_evaluation_rows(rows, "train")
    val_x, val_y, _ = make_evaluation_rows(rows, "val")
    if not len(train_y) or not len(val_y):
        raise ValueError("Both train and validation splits require valid XML reader malignancy ratings")
    if any(row["split"] not in {"train", "val", "test"} for row in rows):
        raise ValueError("Dataset contains a row without a canonical split assignment")
    train_ratings = np.asarray(train_y, dtype=np.int64)
    if set(np.unique(train_ratings)) != set(range(1, 6)):
        raise ValueError(f"Training split must contain every rating 1-5; found {sorted(set(train_ratings))}")
    majority, median = majority_and_median(train_ratings)

    comparisons: dict[str, Any] = {}
    models: dict[str, CumulativeOrdinalModel] = {}
    for model_name, model_kind in (
        ("ordinal", "ordinal"),
        ("gradient_boosted", "gradient_boosted"),
    ):
        model_start = time.perf_counter()
        model = CumulativeOrdinalModel(model_kind=model_kind, seed=seed)
        model.fit(train_x, train_y)
        probabilities = model.predict_proba(val_x)
        metrics = evaluate_predictions(val_y, probabilities)
        metrics["training_majority_baseline"] = evaluate_predictions(
            val_y, _constant_probabilities(np.asarray([majority]), len(val_y))
        )
        metrics["training_majority_baseline"]["constant_rating"] = majority
        metrics["training_median_baseline"] = evaluate_predictions(
            val_y, _constant_probabilities(np.asarray([median]), len(val_y))
        )
        metrics["training_median_baseline"]["constant_rating"] = median
        comparisons[model_name] = {
            "validation": metrics,
            "training_rating_observations": int(len(train_y)),
            "validation_rating_observations": int(len(val_y)),
            "elapsed_seconds": round(time.perf_counter() - model_start, 3),
        }
        models[model_name] = model
        print(
            f"model={model_name} train_reader_ratings={len(train_y)} "
            f"val_reader_ratings={len(val_y)} val_qwk={metrics['qwk']} "
            f"val_mae={metrics['mae']:.4f} elapsed={comparisons[model_name]['elapsed_seconds']:.1f}s",
            flush=True,
        )

    selected_name = max(
        comparisons.items(), key=lambda item: _selection_key((item[0], item[1]["validation"]))
    )[0]
    selected = models[selected_name]
    print(f"Selected {selected_name} using validation QWK (tie-break: validation MAE)", flush=True)

    model_bundles = {}
    for name, model in models.items():
        model_bundles[name] = {
            "model": model,
            "model_name": name,
            "feature_order": list(FEATURE_COLUMNS),
            "target_definition": "XML reader malignancy suspicion rating, ordinal categories 1-5; not a cancer diagnosis.",
            "rating_categories": [1, 2, 3, 4, 5],
            "seed": seed,
        }
    joblib.dump(model_bundles["ordinal"], out_dir / "ordinal_model.pkl")
    joblib.dump(model_bundles["gradient_boosted"], out_dir / "gradient_boosted_model.pkl")
    joblib.dump(model_bundles[selected_name], out_dir / "best_model.pkl")
    joblib.dump(
        {
            "feature_order": list(FEATURE_COLUMNS),
            "transformers_by_threshold": [pipeline.steps[:-1] for pipeline in selected.estimators],
            "model_name": selected_name,
        },
        out_dir / "preprocessing.pkl",
    )

    selection = {
        "selected_model": selected_name,
        "selection_metric": "validation quadratic weighted kappa",
        "tie_breaker": "lower validation mean absolute error",
        "test_metrics_used_for_selection": False,
        "validation_comparison": comparisons,
        "train_patient_or_nodule_rows_used": sum(row["split"] == "train" for row in rows),
        "validation_patient_or_nodule_rows_used_for_model_fit": 0,
        "test_patient_or_nodule_rows_used_for_model_fit": 0,
        "train_reader_rating_count": int(len(train_y)),
        "validation_reader_rating_count": int(len(val_y)),
        "test_reader_rating_count_loaded_for_fit_or_selection": 0,
        "target_definition": "XML reader malignancy suspicion rating, ordinal categories 1-5; not a cancer diagnosis.",
        "reader_level_scoring_protocol": "Each valid reader malignancy rating remains one ordinal training/evaluation observation; nodule-level feature vectors are repeated for their ratings.",
        "training_only_majority_rating": majority,
        "training_only_median_rating": median,
        "seed": seed,
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    (out_dir / "model_selection.json").write_text(
        json.dumps(selection, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "validation_qwk", "validation_mae", "validation_rmse",
                        "validation_exact_rating_accuracy", "train_rating_count",
                        "validation_rating_count", "elapsed_seconds", "selected"),
        )
        writer.writeheader()
        for name, item in comparisons.items():
            metrics = item["validation"]
            writer.writerow({
                "model": name,
                "validation_qwk": metrics["qwk"],
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "validation_exact_rating_accuracy": metrics["exact_rating_accuracy"],
                "train_rating_count": item["training_rating_observations"],
                "validation_rating_count": item["validation_rating_observations"],
                "elapsed_seconds": item["elapsed_seconds"],
                "selected": name == selected_name,
            })

    manifest_path = out_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_selection"] = {
        "selected_model": selected_name,
        "metric": "validation_qwk",
        "tie_breaker": "validation_mae",
        "test_used": False,
    }
    manifest.setdefault("artifacts", {}).update({
        name: sha256_file(out_dir / name)
        for name in ARTIFACT_NAMES
    })
    manifest["runtime_seconds_training"] = round(time.perf_counter() - start, 3)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"Training complete: selected={selected_name}; "
        f"elapsed={time.perf_counter() - start:.1f}s; model={out_dir / 'best_model.pkl'}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    train(args.data_dir, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())