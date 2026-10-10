#!/usr/bin/env python3
"""Evaluate the validation-selected XML-only ordinal model."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from train_xml_malignancy import _constant_probabilities, _load_nodule_rows, _ratings_for_split
from xml_malignancy_common import (
    OUTPUT_DEFAULT,
    evaluate_predictions,
    majority_and_median,
    make_evaluation_rows,
    sha256_file,
)


def _evaluation_path(split_name: str) -> str:
    return "validation_report.json" if split_name == "val" else "test_report.json"


def evaluate(data_dir: Path, split_name: str) -> None:
    start = time.perf_counter()
    out_dir = data_dir.resolve()
    report_path = out_dir / _evaluation_path(split_name)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation report: {report_path}")
    selection_path = out_dir / "model_selection.json"
    model_path = out_dir / "best_model.pkl"
    if not selection_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("Train both models and select a validation model before evaluation")
    if split_name == "test" and not (out_dir / "validation_report.json").is_file():
        raise RuntimeError("Held-out test evaluation is blocked until validation_report.json exists")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    bundle = joblib.load(model_path)
    if bundle.get("model_name") != selection.get("selected_model"):
        raise ValueError("best_model.pkl does not match the validation-selected model")
    rows = _load_nodule_rows(out_dir / "xml_features.csv", {"train", split_name})
    selected_rows = [row for row in rows if row["split"] == split_name]
    if any(row["split"] not in {"train", split_name} for row in rows):
        raise ValueError("Unexpected split rows were loaded for evaluation")
    features, truth, nodule_ids = make_evaluation_rows(rows, split_name)
    if not len(truth):
        raise ValueError(f"No valid XML reader target ratings in {split_name}")
    training_ratings = np.asarray(_ratings_for_split(rows, "train"), dtype=np.int64)
    if not len(training_ratings):
        raise ValueError("Train-only baseline ratings are unavailable")

    probabilities = bundle["model"].predict_proba(features)
    metrics = evaluate_predictions(truth, probabilities)
    majority, median = majority_and_median(training_ratings)
    for baseline_name, rating in (("training_majority", majority), ("training_median", median)):
        baseline = evaluate_predictions(
            truth, _constant_probabilities(np.asarray([rating]), len(truth))
        )
        baseline["constant_rating"] = rating
        metrics[baseline_name] = baseline

    report: dict[str, Any] = {
        "split": split_name,
        "selected_model": selection["selected_model"],
        "selection_metric": selection["selection_metric"],
        "test_metrics_used_for_model_selection": False,
        "sample_count": int(len(truth)),
        "physical_nodule_count": len(set(nodule_ids)),
        "missing_target_nodule_count": sum(
            int(row["target_reader_count"] or 0) == 0 for row in selected_rows
        ),
        "targeted_physical_nodule_count": sum(
            int(row["target_reader_count"] or 0) > 0 for row in selected_rows
        ),
        "target_definition": "XML reader malignancy suspicion rating on the source 1-5 scale; not a cancer diagnosis.",
        "reader_disagreement_protocol": "Each valid reader rating is evaluated independently; per-nodule model probabilities are repeated for all available readers. No consensus rating is treated as ground truth.",
        "metrics": metrics,
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    manifest_path = out_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("artifacts", {})[report_path.name] = sha256_file(report_path)
    manifest[f"runtime_seconds_{split_name}_evaluation"] = report["runtime_seconds"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"Evaluation complete: split={split_name}; model={selection['selected_model']}; "
        f"reader_ratings={len(truth)}; nodules={report['physical_nodule_count']}; "
        f"missing_target_nodules={report['missing_target_nodule_count']}; "
        f"QWK={metrics['qwk']}; MAE={metrics['mae']:.4f}; "
        f"elapsed={report['runtime_seconds']:.1f}s report={report_path}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    evaluate(args.data_dir, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())