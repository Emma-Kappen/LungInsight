#!/usr/bin/env python3
"""Train and evaluate an optional Stage 04 pairwise malignancy-priority ranker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

HERE = Path(__file__).resolve()
IMAGING = HERE.parents[2] / "Imaging"
if str(IMAGING) not in sys.path:
    sys.path.insert(0, str(IMAGING))

from stage04_ranking import PairwiseLinearRanker, ranking_metrics

FEATURES = ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"]


def load_archive(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"X", "feature_names", "patient_id", "malignancy_reader_ratings", "malignancy_reader_mask"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
        x = np.asarray(archive["X"], dtype=np.float64)
        names = [value.decode() if isinstance(value, bytes) else str(value) for value in archive["feature_names"]]
        patients = np.asarray(archive["patient_id"], dtype=str)
        ratings = np.asarray(archive["malignancy_reader_ratings"], dtype=np.float64)
        mask = np.asarray(archive["malignancy_reader_mask"], dtype=bool)
    if names != FEATURES or x.ndim != 2 or x.shape[1] != len(FEATURES):
        raise ValueError(f"{path} does not match the six-feature Stage 04 contract")
    valid = mask & np.isfinite(ratings) & (ratings >= 1) & (ratings <= 5)
    counts = valid.sum(axis=1)
    keep = counts > 0
    y = np.where(valid, ratings, 0).sum(axis=1)[keep] / counts[keep]
    return {"X": x[keep], "y": y, "patients": patients[keep]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path(__file__).resolve().parent / "frozen_features")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "pairwise_ranker_v1.joblib")
    parser.add_argument("--max-pairs-per-patient", type=int, default=2000)
    args = parser.parse_args()
    train = load_archive(args.features_dir / "train.npz")
    val = load_archive(args.features_dir / "val.npz")
    overlap = set(train["patients"].tolist()) & set(val["patients"].tolist())
    if overlap:
        raise ValueError(f"Patient leakage between train and validation: {sorted(overlap)[:5]}")
    model = PairwiseLinearRanker()
    model.fit(
        train["X"], train["y"], train["patients"],
        max_pairs_per_patient=args.max_pairs_per_patient,
    )
    metrics = {
        "model": "PairwiseLinearRanker",
        "target_semantics": "relative malignancy-priority score from LIDC reader-assigned ratings",
        "feature_names": FEATURES,
        "training_config": {
            "learning_rate": model.learning_rate,
            "epochs": model.epochs,
            "l2": model.l2,
            "min_rating_difference": 1.0,
            "max_pairs_per_patient": args.max_pairs_per_patient,
            "random_seed": 2026,
            "split_policy": "existing frozen train/val patient split; no patient appears in both",
        },
        "train": ranking_metrics(train["y"], model.predict(train["X"]), train["patients"]),
        "validation": ranking_metrics(val["y"], model.predict(val["X"]), val["patients"]),
        "patient_overlap": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model_type": "pairwise_linear_ranker_v1",
        "feature_names": FEATURES,
        "model": model,
        "target_semantics": metrics["target_semantics"],
        "training_metadata": metrics,
    }, args.out)
    args.out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
