"""Ranking-aware Stage 04 utilities.

The ranker learns relative malignancy-priority from reader-assigned LIDC
ratings. Those ratings are ordinal suspicion labels, not pathology outcomes or
cancer probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PairwiseTrainingData:
    left: np.ndarray
    right: np.ndarray
    target: np.ndarray
    weights: np.ndarray
    pair_patient_ids: tuple[str, ...]


def build_pairwise_training_data(
    x: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    *,
    min_rating_difference: float = 1.0,
    max_pairs_per_patient: int = 2000,
    random_seed: int = 2026,
) -> PairwiseTrainingData:
    """Build deterministic, non-contradictory within-patient preference pairs."""
    features = np.asarray(x, dtype=np.float64)
    ratings = np.asarray(y, dtype=np.float64)
    patients = np.asarray(patient_ids, dtype=str)
    if features.ndim != 2 or ratings.ndim != 1 or patients.ndim != 1:
        raise ValueError("x must be 2-D and y/patient_ids must be 1-D")
    if len(features) != len(ratings) or len(ratings) != len(patients):
        raise ValueError("x, y, and patient_ids must have the same row count")
    if not np.isfinite(features).all() or not np.isfinite(ratings).all():
        raise ValueError("ranking inputs must be finite")
    if min_rating_difference <= 0 or max_pairs_per_patient <= 0:
        raise ValueError("min_rating_difference and max_pairs_per_patient must be positive")

    rng = np.random.default_rng(random_seed)
    left_rows: list[np.ndarray] = []
    right_rows: list[np.ndarray] = []
    pair_targets: list[float] = []
    pair_weights: list[float] = []
    pair_patients: list[str] = []
    for patient in sorted(set(patients.tolist())):
        indices = np.flatnonzero(patients == patient)
        candidates = [
            (int(i), int(j))
            for i, j in combinations(indices.tolist(), 2)
            if abs(float(ratings[i] - ratings[j])) >= min_rating_difference
        ]
        if len(candidates) > max_pairs_per_patient:
            selected = np.sort(rng.choice(len(candidates), size=max_pairs_per_patient, replace=False))
            candidates = [candidates[int(index)] for index in selected]
        for i, j in candidates:
            if ratings[i] > ratings[j]:
                higher, lower = i, j
            else:
                higher, lower = j, i
            left_rows.append(features[higher])
            right_rows.append(features[lower])
            pair_targets.append(1.0)
            pair_weights.append(float(abs(ratings[higher] - ratings[lower])))
            pair_patients.append(patient)
    if not left_rows:
        raise ValueError("No unambiguous within-patient ranking pairs were generated")
    return PairwiseTrainingData(
        left=np.asarray(left_rows, dtype=np.float64),
        right=np.asarray(right_rows, dtype=np.float64),
        target=np.asarray(pair_targets, dtype=np.float64),
        weights=np.asarray(pair_weights, dtype=np.float64),
        pair_patient_ids=tuple(pair_patients),
    )


def pairwise_logistic_loss(
    score_difference: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Stable weighted loss for preferred-left score differences."""
    difference = np.asarray(score_difference, dtype=np.float64)
    if not np.isfinite(difference).all():
        raise ValueError("score_difference must be finite")
    applied_weights = np.ones_like(difference) if weights is None else np.asarray(weights, dtype=np.float64)
    if applied_weights.shape != difference.shape or not np.isfinite(applied_weights).all():
        raise ValueError("weights must be finite and match score_difference")
    if np.any(applied_weights < 0) or not np.any(applied_weights > 0):
        raise ValueError("weights must be non-negative with at least one positive value")
    loss = np.logaddexp(0.0, -difference)
    return float(np.average(loss, weights=applied_weights))


class PairwiseLinearRanker:
    """Small deterministic linear ranker trained with weighted pairwise loss."""

    def __init__(self, *, learning_rate: float = 0.03, epochs: int = 500, l2: float = 0.01):
        if learning_rate <= 0 or epochs <= 0 or l2 < 0:
            raise ValueError("learning_rate and epochs must be positive; l2 must be non-negative")
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, **pair_kwargs: Any) -> "PairwiseLinearRanker":
        features = np.asarray(x, dtype=np.float64)
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError("x must be a finite 2-D array")
        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        mean = self.mean_
        scale = self.scale_
        standardized = (features - mean) / scale
        pairs = build_pairwise_training_data(standardized, y, patient_ids, **pair_kwargs)
        self.coef_ = np.zeros(features.shape[1], dtype=np.float64)
        self.intercept_ = 0.0
        coef = self.coef_
        for _ in range(self.epochs):
            difference = (pairs.left - pairs.right) @ coef
            sigmoid_negative = 1.0 / (1.0 + np.exp(np.clip(difference, -60.0, 60.0)))
            weighted = pairs.weights / pairs.weights.sum()
            gradient = -((weighted * sigmoid_negative)[:, None] * (pairs.left - pairs.right)).sum(axis=0)
            gradient += self.l2 * coef
            coef -= self.learning_rate * gradient
        return self

    def _check_fitted(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("PairwiseLinearRanker must be fitted before prediction")
        return self.mean_, self.scale_, self.coef_, self.intercept_

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        mean, scale, coef, intercept = self._check_fitted()
        features = np.asarray(x, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(coef) or not np.isfinite(features).all():
            raise ValueError("x must be a finite 2-D array with the fitted feature count")
        return ((features - mean) / scale) @ coef + intercept

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.decision_function(x)

    def shap_values(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, scale, coef, intercept = self._check_fitted()
        standardized = (np.asarray(x, dtype=np.float64) - mean) / scale
        values = standardized * coef
        base = np.full(len(standardized), intercept, dtype=np.float64)
        return values, base


def ranking_metrics(y: np.ndarray, scores: np.ndarray, patient_ids: np.ndarray) -> dict[str, float | int]:
    """Report within-patient pairwise accuracy and rank correlations."""
    ratings = np.asarray(y, dtype=np.float64)
    predictions = np.asarray(scores, dtype=np.float64)
    patients = np.asarray(patient_ids, dtype=str)
    if len(ratings) != len(predictions) or len(ratings) != len(patients):
        raise ValueError("y, scores, and patient_ids must have the same row count")
    total = correct = 0
    for patient in sorted(set(patients.tolist())):
        indices = np.flatnonzero(patients == patient)
        for i, j in combinations(indices.tolist(), 2):
            target_difference = ratings[i] - ratings[j]
            if target_difference == 0:
                continue
            predicted_difference = predictions[i] - predictions[j]
            total += 1
            if target_difference * predicted_difference > 0:
                correct += 1
    result: dict[str, float | int] = {
        "n_nodules": int(len(ratings)),
        "n_ordered_within_patient_pairs": total,
        "pairwise_accuracy": float(correct / total) if total else float("nan"),
    }
    return result
