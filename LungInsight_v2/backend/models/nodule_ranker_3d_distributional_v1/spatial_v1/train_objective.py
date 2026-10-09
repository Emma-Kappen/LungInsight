#!/usr/bin/env python3
"""Train the LungInsight v2 matched objective ablation.

TEST patches and TEST labels are never loaded by this trainer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr, rankdata
from torch.utils.data import DataLoader, Dataset

from model_objective import (
    FOLD_MEAN,
    FOLD_STD,
    HU_HIGH,
    HU_LOW,
    HU_SHIFT,
    INPUT_SHAPE,
    MODEL_TARGETS,
    NOMINAL_TARGETS,
    ORDERED_TARGETS,
    RATING_MAX,
    READER_NAMES,
    READER_TO_MODEL,
    SpatialObjectiveModel,
    architecture_config,
    preprocess_hu,
)

EXPECTED_N = {"train": 1859, "val": 437, "test": 355}
EXPECTED_HASH = {
    "train": "1814147c1f1f0ccfbda73b56c5dbb392c14412b72c6b0fe316ee7a81957ad922",
    "val": "0831f0b8e0ba4b54de794c914677e783d21197bedc5c6f93c695eb07011051f7",
    "test": "e3ebe6f752237c58240e0888492679295ab068d6929cd75be1df83b821e4b56c",
}

ATTRIBUTE_GROUPS = {
    "all": None,
    "final_hybrid": ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"],
    "lobulation_spiculation": ["lobulation", "spiculation"],
    "subtlety": ["subtlety"],
    "sphericity": ["sphericity"],
    "margin": ["margin"],
    "texture": ["texture"],
}


def selected_attribute_names(selection: str | Sequence[str]) -> list[str]:
    if isinstance(selection, str):
        if selection in ATTRIBUTE_GROUPS:
            return list(MODEL_TARGETS if ATTRIBUTE_GROUPS[selection] is None else ATTRIBUTE_GROUPS[selection])
        if selection == "all":
            return list(MODEL_TARGETS)
        if selection in MODEL_TARGETS:
            return [selection]
        raise ValueError(f"Unknown attribute selection: {selection}")
    names = list(selection)
    unknown = set(names) - set(MODEL_TARGETS)
    if unknown:
        raise ValueError(f"Unknown attributes: {sorted(unknown)}")
    return names


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, sort_keys=True), encoding="utf-8")


def hash_indices(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def decode_scalar(v: Any) -> str:
    return v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)


def read_h5_indices(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read arbitrary indices through sorted h5py fancy indexing, restoring order."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return np.empty((0,) + dataset.shape[1:], dtype=dataset.dtype)
    order = np.argsort(indices, kind="mergesort")
    sorted_idx = indices[order]
    sorted_values = np.asarray(dataset[sorted_idx])
    restore = np.argsort(order, kind="mergesort")
    return sorted_values[restore]


def parse_and_validate_split(path: Path, n: int, h5: h5py.File) -> dict[str, np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("split.json must be a JSON object")

    # Patient IDs are metadata, not TEST labels or patches. Read per split so
    # the trainer never loads TEST reader annotations or CT data.
    patient_ds = h5["patient_id"]
    split: dict[str, np.ndarray] = {}
    patients: dict[str, set[str]] = {}

    for name in ("train", "val", "test"):
        entry = raw.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"split file: '{name}' must be an object")
        idx = entry.get("sample_indices", entry.get("nodule_indices", entry.get("indices")))
        if not isinstance(idx, list) or not all(isinstance(v, int) for v in idx):
            raise ValueError(f"split file: '{name}' must contain integer sample indices")
        arr = np.asarray(idx, dtype=np.int64)
        if len(arr) != len(np.unique(arr)):
            raise ValueError(f"{name} split contains duplicate sample indices")
        if np.any(arr < 0) or np.any(arr >= n):
            raise ValueError(f"{name} split contains out-of-range indices")
        if len(arr) != EXPECTED_N[name]:
            raise ValueError(f"{name}: expected {EXPECTED_N[name]} samples, got {len(arr)}")
        got_hash = hash_indices(arr)
        if got_hash != EXPECTED_HASH[name]:
            raise ValueError(f"{name}: canonical split hash mismatch: {got_hash}")

        actual_patients = {decode_scalar(v) for v in read_h5_indices(patient_ds, arr)}
        listed = entry.get("patients", entry.get("patient_ids"))
        if listed is not None and {str(v) for v in listed} != actual_patients:
            raise ValueError(f"{name}: listed patients do not match sample indices")
        split[name] = arr
        patients[name] = actual_patients

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(split[a].tolist()) & set(split[b].tolist()):
            raise ValueError(f"sample leakage between {a} and {b}")
        if patients[a] & patients[b]:
            raise ValueError(f"patient leakage between {a} and {b}")

    if sum(len(v) for v in split.values()) != n:
        raise ValueError("canonical split does not partition the H5 exactly")
    return split


def build_target_arrays(reader_labels: np.ndarray, reader_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Construct targets without mutating labels; only valid reader ratings count."""
    labels = np.asarray(reader_labels, dtype=np.float32)[:, :, READER_TO_MODEL]
    mask = np.asarray(reader_mask, dtype=bool)
    if labels.shape[:2] != mask.shape:
        raise ValueError(f"reader label/mask mismatch: {labels.shape} vs {mask.shape}")

    n = labels.shape[0]
    kmax = max(RATING_MAX.values())
    distributions = np.zeros((n, len(MODEL_TARGETS), kmax), dtype=np.float32)
    means = np.full((n, len(MODEL_TARGETS)), np.nan, dtype=np.float32)
    medians = np.full((n, len(MODEL_TARGETS)), np.nan, dtype=np.float32)
    counts = np.zeros((n, len(MODEL_TARGETS)), dtype=np.int32)
    valid = np.zeros((n, len(MODEL_TARGETS)), dtype=bool)
    agreement = np.zeros((n, len(MODEL_TARGETS)), dtype=np.float32)
    reader_ratings = labels.copy()
    reader_valid = np.zeros(labels.shape, dtype=bool)
    invalid_records: list[dict[str, Any]] = []

    for h, name in enumerate(MODEL_TARGETS):
        k = RATING_MAX[name]
        values = labels[:, :, h]
        finite = np.isfinite(values)
        valid_reader = mask & finite & (values >= 1.0) & (values <= float(k))
        reader_valid[:, :, h] = valid_reader
        invalid = mask & finite & ~((values >= 1.0) & (values <= float(k)))

        bad_positions = np.argwhere(invalid)
        for sample_idx, reader_slot in bad_positions.tolist():
            invalid_records.append({
                "local_index": int(sample_idx),
                "reader_slot": int(reader_slot),
                "attribute": name,
                "value": float(values[sample_idx, reader_slot]),
            })

        counts[:, h] = valid_reader.sum(axis=1).astype(np.int32)
        valid[:, h] = counts[:, h] > 0
        denom = np.maximum(counts[:, h], 1).astype(np.float32)
        safe_values = np.where(valid_reader, values, 0.0)
        means[:, h] = (safe_values.sum(axis=1) / denom).astype(np.float32)
        for sample_idx in np.flatnonzero(valid[:, h]):
            medians[sample_idx, h] = np.median(values[sample_idx, valid_reader[sample_idx]])

        for rating in range(1, k + 1):
            hit = valid_reader & (values == float(rating))
            distributions[:, h, rating - 1] = hit.sum(axis=1).astype(np.float32) / denom
        agreement[:, h] = distributions[:, h, :k].max(axis=1)

        if np.any(valid[:, h]):
            if not np.allclose(distributions[valid[:, h], h, :k].sum(axis=1), 1.0, atol=1e-6):
                raise ValueError(f"{name}: empirical distributions do not sum to 1")

    return {
        "distribution": distributions,
        "mean": means,
        "median": medians,
        "count": counts,
        "valid": valid,
        "agreement": agreement,
        "reader_ratings": reader_ratings,
        "reader_valid": reader_valid,
        "invalid_records": invalid_records,
    }


class H5PatchDataset(Dataset):
    def __init__(
        self,
        h5_path: Path,
        indices: np.ndarray,
        targets: dict[str, np.ndarray],
        ranking_target: str = "mean",
    ):
        if ranking_target not in {"mean", "median", "mean_median_blend"}:
            if ranking_target != "reader_pairwise":
                raise ValueError(f"Unknown ranking target: {ranking_target}")
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = np.asarray(targets["mean"], dtype=np.float32)
        self.ranking_target = np.asarray(
            targets["mean"] if ranking_target in {"mean", "reader_pairwise"} else
            targets["median"] if ranking_target == "median" else
            0.5 * (targets["mean"] + targets["median"]),
            dtype=np.float32,
        )
        self.distribution = np.asarray(targets["distribution"], dtype=np.float32)
        self.agreement = np.asarray(targets["agreement"], dtype=np.float32)
        self.count = np.asarray(targets["count"], dtype=np.int32)
        self.valid = np.asarray(targets["valid"], dtype=bool)
        self.reader_ratings = np.asarray(targets["reader_ratings"], dtype=np.float32)
        self.reader_valid = np.asarray(targets["reader_valid"], dtype=bool)
        self._h5 = None

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        self._open()
        h5_index = int(self.indices[item])
        patch = np.asarray(self._h5["patches"][h5_index], dtype=np.float32)
        return {
            "x": torch.from_numpy(patch),
            "mean": torch.from_numpy(self.mean[item]),
            "ranking_target": torch.from_numpy(self.ranking_target[item]),
            "distribution": torch.from_numpy(self.distribution[item]),
            "agreement": torch.from_numpy(self.agreement[item]),
            "count": torch.from_numpy(self.count[item]),
            "valid": torch.from_numpy(self.valid[item]),
            "reader_ratings": torch.from_numpy(self.reader_ratings[item]),
            "reader_valid": torch.from_numpy(self.reader_valid[item]),
            "h5_index": h5_index,
        }

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "mean": torch.stack([b["mean"] for b in batch]),
        "ranking_target": torch.stack([b["ranking_target"] for b in batch]),
        "distribution": torch.stack([b["distribution"] for b in batch]),
        "agreement": torch.stack([b["agreement"] for b in batch]),
        "count": torch.stack([b["count"] for b in batch]),
        "valid": torch.stack([b["valid"] for b in batch]),
        "reader_ratings": torch.stack([b["reader_ratings"] for b in batch]),
        "reader_valid": torch.stack([b["reader_valid"] for b in batch]),
        "h5_index": np.asarray([b["h5_index"] for b in batch], dtype=np.int64),
    }


def choose_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return dev


def ordinal_cumulative_targets(reader_ratings: torch.Tensor, k: int) -> torch.Tensor:
    """Build per-reader (Y > threshold) targets with shape [B,R,K-1]."""
    if reader_ratings.ndim != 2:
        raise ValueError(f"Expected reader ratings [B,R], got {tuple(reader_ratings.shape)}")
    if k < 2:
        raise ValueError(f"Ordinal rating maximum must be at least 2, got {k}")
    thresholds = torch.arange(1, k, device=reader_ratings.device, dtype=reader_ratings.dtype)
    return (reader_ratings.unsqueeze(-1) > thresholds).to(dtype=torch.float32)


def ordinal_reader_node_losses(
    logits: torch.Tensor,
    reader_ratings: torch.Tensor,
    reader_valid: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each nodule's mean reader-level cumulative-link BCE and validity."""
    if logits.ndim != 2 or logits.shape[1] != k - 1:
        raise ValueError(f"Expected ordinal logits [B,{k - 1}], got {tuple(logits.shape)}")
    if reader_ratings.ndim != 2 or reader_valid.shape != reader_ratings.shape:
        raise ValueError("Expected reader ratings and validity with matching [B,R] shapes")
    if reader_ratings.shape[0] != logits.shape[0]:
        raise ValueError("Ordinal logits and reader target batch dimensions do not match")

    reader_ratings = reader_ratings.float()
    reader_valid = reader_valid.bool()
    safe_ratings = torch.where(reader_valid, reader_ratings, torch.zeros_like(reader_ratings))
    ordinal_targets = ordinal_cumulative_targets(safe_ratings, k)
    reader_logits = logits.float().unsqueeze(1).expand_as(ordinal_targets)
    reader_losses = F.binary_cross_entropy_with_logits(
        reader_logits, ordinal_targets, reduction="none"
    ).mean(dim=-1)
    weights = reader_valid.to(dtype=reader_losses.dtype)
    node_valid = reader_valid.any(dim=1)
    node_losses = (reader_losses * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return node_losses, node_valid


def nominal_soft_ce(outputs: tuple[torch.Tensor, ...], distribution: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for h, name in enumerate(MODEL_TARGETS):
        if name not in NOMINAL_TARGETS:
            continue
        k = RATING_MAX[name]
        logp = F.log_softmax(outputs[h].float(), dim=-1)
        target = distribution[:, h, :k].float()
        node_loss = -(target * logp).sum(dim=-1)
        losses.append(node_loss[valid[:, h]])
    return torch.cat(losses).mean() if losses else torch.zeros((), device=outputs[0].device)


def mean_regression_loss(outputs: tuple[torch.Tensor, ...], mean: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for h, name in enumerate(MODEL_TARGETS):
        if name not in ORDERED_TARGETS:
            continue
        k = float(RATING_MAX[name])
        target = ((mean[:, h].float() - 1.0) / (k - 1.0)).clamp(0.0, 1.0)
        pred = torch.sigmoid(outputs[h].float())
        loss = F.smooth_l1_loss(pred, target, beta=0.05, reduction="none")
        losses.append(loss[valid[:, h]])
    return torch.cat(losses).mean() if losses else torch.zeros((), device=outputs[0].device)


def pairwise_logistic_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    agreement: torch.Tensor,
    valid: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, int]:
    """Confidence-weighted logistic loss over valid, sufficiently separated pairs."""
    valid = valid.bool() & torch.isfinite(targets) & torch.isfinite(scores)
    indices = torch.nonzero(valid, as_tuple=True)[0]
    if indices.numel() < 2:
        return scores.sum() * 0.0, 0

    pair_indices = torch.triu_indices(indices.numel(), indices.numel(), offset=1, device=scores.device)
    left, right = indices[pair_indices[0]], indices[pair_indices[1]]
    target_difference = targets[left].float() - targets[right].float()
    keep = (target_difference.abs() >= threshold) & (target_difference != 0)
    if not torch.any(keep):
        return scores.sum() * 0.0, 0

    left = left[keep]
    right = right[keep]
    target_sign = target_difference[keep].sign()
    score_difference = scores[left].float() - scores[right].float()
    weights = agreement[left].float() * agreement[right].float()
    losses = F.softplus(-target_sign * score_difference)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return scores.sum() * 0.0, int(keep.sum())
    return (losses * weights).sum() / weight_sum, int(keep.sum())


def reader_pairwise_loss(
    scores: torch.Tensor,
    reader_ratings: torch.Tensor,
    reader_valid: torch.Tensor,
    min_pair_confidence: float,
    attribute: str | Sequence[str] = "all",
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Confidence-weighted BCE over pair targets formed from individual readers."""
    batch_size, _, n_attributes = reader_ratings.shape
    if batch_size < 2:
        zero = scores.sum() * 0.0
        return zero, {"pair_accuracy": float("nan"), "valid_pair_count": 0, "average_pair_confidence": float("nan")}

    pair_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=scores.device)
    left, right = pair_indices
    attribute_losses: list[torch.Tensor] = []
    confidence_sum = 0.0
    correct_count = 0
    valid_pair_count = 0

    selected = set(selected_attribute_names(attribute))
    for h, name in enumerate(MODEL_TARGETS[:n_attributes]):
        if name not in selected:
            continue
        left_ratings = reader_ratings[left, :, h].float()
        right_ratings = reader_ratings[right, :, h].float()
        left_valid = reader_valid[left, :, h].bool()
        right_valid = reader_valid[right, :, h].bool()
        comparisons_valid = left_valid[:, :, None] & right_valid[:, None, :]
        wins_left = ((left_ratings[:, :, None] > right_ratings[:, None, :]) & comparisons_valid).sum(dim=(1, 2))
        wins_right = ((left_ratings[:, :, None] < right_ratings[:, None, :]) & comparisons_valid).sum(dim=(1, 2))
        evidence = wins_left + wins_right
        pair_targets = wins_left.float() / evidence.clamp_min(1).float()
        confidence = 2.0 * (pair_targets - 0.5).abs()
        keep = (evidence > 0) & (confidence >= min_pair_confidence)
        if not torch.any(keep):
            continue

        delta = scores[left[keep], h].float() - scores[right[keep], h].float()
        weights = confidence[keep]
        losses = F.binary_cross_entropy_with_logits(delta, pair_targets[keep], reduction="none")
        weight_sum = weights.sum()
        if weight_sum > 0:
            attribute_losses.append((losses * weights).sum() / weight_sum)
        hard_target = pair_targets[keep] > 0.5
        predicted = delta > 0
        correct_count += int((predicted == hard_target).sum().item())
        valid_pair_count += int(keep.sum().item())
        confidence_sum += float(weights.sum().item())

    loss = torch.stack(attribute_losses).mean() if attribute_losses else scores.sum() * 0.0
    return loss, {
        "pair_accuracy": correct_count / valid_pair_count if valid_pair_count else float("nan"),
        "valid_pair_count": valid_pair_count,
        "average_pair_confidence": confidence_sum / valid_pair_count if valid_pair_count else float("nan"),
    }


def reader_pairwise_split_diagnostics(
    scores: np.ndarray,
    reader_ratings: np.ndarray,
    reader_valid: np.ndarray,
    min_pair_confidence: float,
    attribute: str | Sequence[str] = "all",
) -> dict[str, float | int]:
    """Compute exact split-wide pair statistics in bounded-memory row blocks."""
    scores = np.asarray(scores, dtype=np.float64)
    ratings = np.asarray(reader_ratings, dtype=np.float32)
    valid = np.asarray(reader_valid, dtype=bool)
    n, _, n_attributes = ratings.shape
    confidence_sum = 0.0
    correct_count = 0
    valid_pair_count = 0
    attribute_losses: list[float] = []
    block_size = 64

    selected = set(selected_attribute_names(attribute))
    for h, name in enumerate(MODEL_TARGETS[:n_attributes]):
        if name not in selected:
            continue
        k = RATING_MAX[name]
        hist = np.stack([
            np.sum(valid[:, :, h] & (ratings[:, :, h] == rating), axis=1)
            for rating in range(1, k + 1)
        ], axis=1).astype(np.float64)
        attribute_loss_sum = 0.0
        attribute_weight_sum = 0.0

        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            wins_left = np.zeros((stop - start, n), dtype=np.float64)
            wins_right = np.zeros_like(wins_left)
            for rating_index in range(k):
                count_left = hist[start:stop, rating_index, None]
                lower_right = hist[:, :rating_index].sum(axis=1)[None, :]
                upper_right = hist[:, rating_index + 1 :].sum(axis=1)[None, :]
                wins_left += count_left * lower_right
                wins_right += count_left * upper_right

            evidence = wins_left + wins_right
            pair_targets = wins_left / np.maximum(evidence, 1.0)
            confidence = 2.0 * np.abs(pair_targets - 0.5)
            row_indices = np.arange(start, stop)[:, None]
            col_indices = np.arange(n)[None, :]
            keep = (col_indices > row_indices) & (evidence > 0) & (confidence >= min_pair_confidence)
            if not np.any(keep):
                continue

            delta = scores[start:stop, h, None] - scores[None, :, h]
            selected_delta = delta[keep]
            selected_target = pair_targets[keep]
            selected_confidence = confidence[keep]
            attribute_loss_sum += float(np.sum(
                selected_confidence * (np.logaddexp(0.0, selected_delta) - selected_target * selected_delta)
            ))
            attribute_weight_sum += float(selected_confidence.sum())
            correct_count += int(np.sum((selected_delta > 0) == (selected_target > 0.5)))
            valid_pair_count += int(keep.sum())
            confidence_sum += float(selected_confidence.sum())

        if attribute_weight_sum > 0:
            attribute_losses.append(attribute_loss_sum / attribute_weight_sum)

    return {
        "reader_pairwise_loss": float(np.mean(attribute_losses)) if attribute_losses else float("nan"),
        "pair_accuracy": correct_count / valid_pair_count if valid_pair_count else float("nan"),
        "valid_pair_count": valid_pair_count,
        "average_pair_confidence": confidence_sum / valid_pair_count if valid_pair_count else float("nan"),
    }


def select_ranking_targets(targets: dict[str, np.ndarray], ranking_target: str) -> np.ndarray:
    if ranking_target == "mean":
        return np.asarray(targets["mean"], dtype=np.float32)
    if ranking_target == "median":
        return np.asarray(targets["median"], dtype=np.float32)
    if ranking_target == "mean_median_blend":
        return 0.5 * (np.asarray(targets["mean"], dtype=np.float32) + np.asarray(targets["median"], dtype=np.float32))
    raise ValueError(f"Unknown ranking target: {ranking_target}")


def ranking_pair_counts(
    targets: dict[str, np.ndarray], ranking_target: str, thresholds: tuple[float, ...] = (0.25, 0.5, 1.0)
) -> dict[str, dict[str, int]]:
    selected = select_ranking_targets(targets, ranking_target)
    counts: dict[str, dict[str, int]] = {}
    for h, name in enumerate(MODEL_TARGETS):
        if name not in ORDERED_TARGETS:
            continue
        values = selected[:, h]
        values = values[np.isfinite(values)]
        pairs = np.triu_indices(len(values), k=1)
        differences = np.abs(values[pairs[0]] - values[pairs[1]])
        counts[name] = {
            f"pairs_{int(round(threshold * 100)):03d}": int(
                np.count_nonzero((differences >= threshold) & (differences > 0))
            )
            for threshold in thresholds
        }
    return counts


def total_loss(
    model: SpatialObjectiveModel,
    outputs: tuple[torch.Tensor, ...],
    batch: dict[str, torch.Tensor],
    rank_loss_weight: float = 0.5,
    ranking_pair_threshold: float = 0.5,
    ranking_target: str = "mean",
    ranking_min_confidence: float = 0.10,
    attribute: str | Sequence[str] = "all",
) -> torch.Tensor:
    # Equal weight per attribute (not per semantic group). Each task's loss is
    # averaged over valid nodules/readers first; the eight task losses are then
    # averaged, keeping Objective A and B matched.
    task_losses: list[torch.Tensor] = []
    selected = set(selected_attribute_names(attribute))
    ordered_scores = model.expected_ratings(outputs) if rank_loss_weight != 0.0 else None
    ranking_losses: list[torch.Tensor] = []
    for h, name in enumerate(MODEL_TARGETS):
        if name not in selected:
            continue
        valid = batch["valid"][:, h]
        if not torch.any(valid):
            continue
        if name in ORDERED_TARGETS:
            if model.objective == "ordinal_reader":
                k = RATING_MAX[name]
                node_loss, node_valid = ordinal_reader_node_losses(
                    outputs[h],
                    batch["reader_ratings"][:, :, h],
                    batch["reader_valid"][:, :, h],
                    k,
                )
            else:
                k = float(RATING_MAX[name])
                target = ((batch["mean"][:, h].float() - 1.0) / (k - 1.0)).clamp(0.0, 1.0)
                target = torch.nan_to_num(target, nan=0.0)
                pred = torch.sigmoid(outputs[h].float())
                node_loss = F.smooth_l1_loss(pred, target, beta=0.05, reduction="none")
                node_valid = valid
        else:
            k = RATING_MAX[name]
            logp = F.log_softmax(outputs[h].float(), dim=-1)
            target_dist = batch["distribution"][:, h, :k].float()
            node_loss = -(target_dist * logp).sum(dim=-1)
            node_valid = valid
        task_losses.append(node_loss[node_valid].mean())
        if ranking_target == "reader_pairwise":
            continue
        if name in ORDERED_TARGETS and ordered_scores is not None:
            rank_loss, pair_count = pairwise_logistic_loss(
                ordered_scores[:, h],
                batch["ranking_target"][:, h],
                batch["agreement"][:, h],
                valid,
                ranking_pair_threshold,
            )
            if pair_count:
                ranking_losses.append(rank_loss)

    supervised_loss = torch.stack(task_losses).mean() if task_losses else torch.zeros((), device=outputs[0].device)
    if ranking_target == "reader_pairwise" and ordered_scores is not None:
        ranking_loss, _ = reader_pairwise_loss(
            ordered_scores,
            batch["reader_ratings"],
            batch["reader_valid"],
            ranking_min_confidence,
            attribute,
        )
    else:
        ranking_loss = torch.stack(ranking_losses).mean() if ranking_losses else torch.zeros_like(supervised_loss)
    return supervised_loss + rank_loss_weight * ranking_loss


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def finite_corr(pred: np.ndarray, true: np.ndarray, fn) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if len(p) < 3 or np.std(p) < 1e-12 or np.std(t) < 1e-12:
        return float("nan")
    return float(fn(p, t).statistic)


def metrics_for_predictions(
    pred: np.ndarray,
    nominal_probs: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    attribute: str | Sequence[str] = "all",
) -> tuple[dict[str, dict[str, Any]], float, float]:
    per: dict[str, dict[str, Any]] = {}
    m6_ge3_values: list[float] = []
    m6_all_values: list[float] = []
    selected = set(selected_attribute_names(attribute))

    for h, name in enumerate(MODEL_TARGETS):
        if name not in selected:
            continue
        y = targets["mean"][:, h]
        valid = targets["valid"][:, h]
        count = targets["count"][:, h]
        p = pred[:, h]
        if name in ORDERED_TARGETS:
            use_all = valid
            use_ge3 = valid & (count >= 3)
            pa, ya = p[use_all], y[use_all]
            pg, yg = p[use_ge3], y[use_ge3]
            spearman_all = finite_corr(pa, ya, spearmanr)
            spearman_ge3 = finite_corr(pg, yg, spearmanr)
            pearson_all = finite_corr(pa, ya, pearsonr)
            per[name] = {
                "n_all": int(use_all.sum()),
                "n_ge3": int(use_ge3.sum()),
                "mae": float(np.mean(np.abs(pa - ya))) if len(pa) else float("nan"),
                "rmse": float(np.sqrt(np.mean((pa - ya) ** 2))) if len(pa) else float("nan"),
                "pearson": pearson_all,
                "spearman": spearman_all,
                "spearman_ge3": spearman_ge3,
                "target_std": float(np.std(pa * 0 + ya)) if len(ya) else float("nan"),
                "pred_std": float(np.std(pa)) if len(pa) else float("nan"),
                "sd_ratio": float(np.std(pa) / np.std(ya)) if len(ya) and np.std(ya) > 1e-12 else float("nan"),
            }
            if math.isfinite(spearman_all):
                m6_all_values.append(spearman_all)
            if math.isfinite(spearman_ge3):
                m6_ge3_values.append(spearman_ge3)
        else:
            probs = nominal_probs[name]
            use = valid
            true_cls = np.rint(y[use]).astype(int) - 1
            pred_cls = np.argmax(probs[use], axis=1)
            onehot = targets["distribution"][use, h, :RATING_MAX[name]]
            pprob = np.clip(probs[use], 1e-7, 1.0)
            nll = float(-np.mean(np.sum(onehot * np.log(pprob), axis=1))) if len(pprob) else float("nan")
            per[name] = {
                "n": int(use.sum()),
                "nll": nll,
                "accuracy": float(np.mean(pred_cls == true_cls)) if len(true_cls) else float("nan"),
                "true_class_distribution": np.bincount(true_cls, minlength=RATING_MAX[name]).tolist() if len(true_cls) else [],
                "pred_class_distribution": np.bincount(pred_cls, minlength=RATING_MAX[name]).tolist() if len(pred_cls) else [],
            }

            if name == "calcification" and len(true_cls):
                present_true = (true_cls < 5).astype(np.int32)
                present_score = 1.0 - probs[use, 5]
                if len(np.unique(present_true)) == 2:
                    per[name]["present_absent_auroc"] = binary_auc(present_true, present_score)
                else:
                    per[name]["present_absent_auroc"] = float("nan")

    m6_ge3 = float(np.mean(m6_ge3_values)) if m6_ge3_values else float("nan")
    m6_all = float(np.mean(m6_all_values)) if m6_all_values else float("nan")
    return per, m6_ge3, m6_all


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(score, dtype=np.float64)
    ranks = rankdata(s, method="average")
    positives = y == 1
    negatives = y == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def prediction_outputs(model: SpatialObjectiveModel, outputs: tuple[torch.Tensor, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    batch_expected = model.expected_ratings(outputs).detach().cpu().numpy().astype(np.float32)
    nominal: dict[str, list[np.ndarray]] = {n: [] for n in NOMINAL_TARGETS}
    for h, name in enumerate(MODEL_TARGETS):
        if name in NOMINAL_TARGETS:
            nominal[name].append(torch.softmax(outputs[h].float(), dim=-1).detach().cpu().numpy().astype(np.float32))
    nominal_np = {name: vals[0] for name, vals in nominal.items()}
    return batch_expected, nominal_np


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scaler=None,
    amp_dtype=None,
    grad_clip=1.0,
    rank_loss_weight=0.5,
    ranking_pair_threshold=0.5,
    ranking_target="mean",
    ranking_min_confidence=0.10,
    attribute="all",
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss_value = 0.0
    total_n = 0
    for batch in loader:
        b = move_batch(batch, device)
        x = preprocess_hu(b["x"]).unsqueeze(1)
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast_enabled = amp_dtype is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            outputs = model(x)
            loss = total_loss(
                model, outputs, b, rank_loss_weight, ranking_pair_threshold,
                ranking_target, ranking_min_confidence, attribute,
            )
        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total_loss_value += float(loss.detach()) * len(b["x"])
        total_n += len(b["x"])
    return total_loss_value / max(total_n, 1)


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    targets: dict[str, np.ndarray],
    rank_loss_weight: float = 0.5,
    ranking_pair_threshold: float = 0.5,
    ranking_target: str = "mean",
    ranking_min_confidence: float = 0.10,
    attribute: str = "all",
) -> tuple[float, dict[str, dict[str, Any]], float, float, dict[str, float | int] | None]:
    model.eval()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    nominal: dict[str, list[np.ndarray]] = {n: [] for n in NOMINAL_TARGETS}
    for batch in loader:
        b = move_batch(batch, device)
        x = preprocess_hu(b["x"]).unsqueeze(1)
        outputs = model(x)
        loss = total_loss(
            model, outputs, b, rank_loss_weight, ranking_pair_threshold,
            ranking_target, ranking_min_confidence, attribute,
        )
        losses.append(float(loss.detach()) * len(b["x"]))
        expected = model.expected_ratings(outputs).cpu().numpy().astype(np.float32)
        preds.append(expected)
        for h, name in enumerate(MODEL_TARGETS):
            if name in NOMINAL_TARGETS:
                nominal[name].append(torch.softmax(outputs[h].float(), dim=-1).cpu().numpy().astype(np.float32))
    pred_np = np.concatenate(preds, axis=0)
    nominal_np = {n: np.concatenate(v, axis=0) for n, v in nominal.items()}
    ordered_target_slice = {k: v for k, v in targets.items()}
    per, m6_ge3, m6_all = metrics_for_predictions(pred_np, nominal_np, ordered_target_slice, attribute)
    pair_diagnostics = None
    if ranking_target == "reader_pairwise":
        pair_diagnostics = reader_pairwise_split_diagnostics(
            pred_np,
            targets["reader_ratings"],
            targets["reader_valid"],
            ranking_min_confidence,
            attribute,
        )
    return sum(losses) / max(len(loader.dataset), 1), per, m6_ge3, m6_all, pair_diagnostics


@torch.inference_mode()
def collect_pairwise_scores(model, loader, device) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    for batch in loader:
        b = move_batch(batch, device)
        outputs = model(preprocess_hu(b["x"]).unsqueeze(1))
        scores.append(model.expected_ratings(outputs).cpu().numpy().astype(np.float32))
    return np.concatenate(scores, axis=0)


def build_baselines(
    train_targets: dict[str, np.ndarray],
    val_targets: dict[str, np.ndarray],
    attribute: str | Sequence[str] = "all",
) -> dict[str, Any]:
    rows = []
    selected = set(selected_attribute_names(attribute))
    for h, name in enumerate(MODEL_TARGETS):
        if name not in selected:
            continue
        tr_valid = train_targets["valid"][:, h]
        va_valid = val_targets["valid"][:, h]
        tr = train_targets["mean"][tr_valid, h]
        va = val_targets["mean"][va_valid, h]
        row: dict[str, Any] = {"attribute": name, "train_valid_n": int(tr_valid.sum()), "val_valid_n": int(va_valid.sum())}
        if name in ORDERED_TARGETS:
            value = float(np.median(tr))
            row.update({
                "baseline": "train_median_reader_mean",
                "value": value,
                "mae": float(np.mean(np.abs(np.full_like(va, value) - va))) if len(va) else float("nan"),
            })
        else:
            k = RATING_MAX[name]
            global_dist = train_targets["distribution"][tr_valid, h, :k].sum(axis=0).astype(np.float64)
            global_dist = global_dist / max(global_dist.sum(), 1e-12)
            cls = int(np.argmax(global_dist))
            va_dist = val_targets["distribution"][va_valid, h, :k].astype(np.float64)
            nll = float(-np.mean(np.sum(va_dist * np.log(np.clip(global_dist[None, :], 1e-7, 1.0)), axis=1))) if len(va_dist) else float("nan")
            va_cls = np.rint(va).astype(int) - 1
            row.update({
                "baseline": "train_global_reader_distribution",
                "mode_class": cls + 1,
                "nll": nll,
                "accuracy": float(np.mean(va_cls == cls)) if len(va_cls) else float("nan"),
                "train_distribution": global_dist.tolist(),
            })
        rows.append(row)
    return {"rows": rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--h5", required=True)
    p.add_argument("--split-file", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--objective", choices=["ordinal_reader", "mean_regression"], required=True)
    p.add_argument("--rank-loss-weight", type=float, default=0.5)
    p.add_argument(
        "--ranking-target",
        choices=["mean", "median", "mean_median_blend", "reader_pairwise"],
        default=None,
        help="Defaults to median for named hybrid groups and mean for all/legacy runs.",
    )
    p.add_argument("--ranking-pair-threshold", type=float, default=0.5)
    p.add_argument("--ranking-min-confidence", type=float, default=0.10)
    p.add_argument(
        "--attribute",
        choices=["all", *ORDERED_TARGETS],
        default=None,
        help="Legacy selector: train all heads jointly or isolate one ordered attribute.",
    )
    p.add_argument(
        "--attribute-group",
        choices=list(ATTRIBUTE_GROUPS),
        default="all",
        help="Select the supervised, ranked, evaluated, and checkpointed attribute group.",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--min-lr-factor", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--attn-dim", type=int, default=96)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main(a: argparse.Namespace) -> int:
    seed_all(a.seed)
    h5_path = Path(a.h5).resolve()
    split_path = Path(a.split_file).resolve()
    if a.attribute is not None and a.attribute_group != "all":
        raise ValueError("Use either --attribute or --attribute-group, not both")
    selection = a.attribute if a.attribute is not None else a.attribute_group
    active_attributes = selected_attribute_names(selection)
    active_ordered_targets = [name for name in ORDERED_TARGETS if name in active_attributes]
    is_all = set(active_attributes) == set(MODEL_TARGETS)
    group_run = a.attribute is None and a.attribute_group != "all"
    selection_label = a.attribute_group if group_run else (a.attribute or "all")
    if a.ranking_target is None:
        a.ranking_target = "median" if group_run else "mean"
    if group_run and (
        a.objective != "ordinal_reader"
        or a.ranking_target != "median"
        or not math.isclose(a.rank_loss_weight, 0.5)
        or not math.isclose(a.ranking_pair_threshold, 0.5)
    ):
        raise ValueError(
            "Named hybrid groups require objective=ordinal_reader, ranking-target=median, "
            "rank-loss-weight=0.5, and ranking-pair-threshold=0.5"
        )
    run_variant = f"rank_{a.rank_loss_weight:g}_{a.ranking_target}_pair_{a.ranking_pair_threshold:g}"
    if a.ranking_target == "reader_pairwise":
        run_variant += f"_min_conf_{a.ranking_min_confidence:g}"
    if a.attribute is not None and a.attribute != "all":
        run_variant += f"_attribute_{a.attribute}"
    if group_run:
        out = Path(a.out_dir).resolve() / "hybrid" / a.attribute_group
    else:
        out = Path(a.out_dir).resolve() / a.objective / f"seed_{a.seed}" / run_variant
    out.mkdir(parents=True, exist_ok=True)
    device = choose_device(a.device)

    print(f"LUNGINSIGHT SPATIAL_OBJECTIVE_V1 | objective={a.objective} | seed={a.seed} | device={device}", flush=True)
    if a.rank_loss_weight < 0:
        raise ValueError("--rank-loss-weight must be non-negative")
    if a.ranking_pair_threshold < 0:
        raise ValueError("--ranking-pair-threshold must be non-negative")
    if not 0.0 <= a.ranking_min_confidence <= 1.0:
        raise ValueError("--ranking-min-confidence must be between 0 and 1")
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}", flush=True)

    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    # Only metadata and TRAIN+VAL labels are loaded. TEST CT/labels remain sealed.
    with h5py.File(h5_path, "r") as h5:
        required = ["patches", "reader_labels", "reader_mask", "patient_id"]
        missing = [k for k in required if k not in h5]
        if missing:
            raise RuntimeError(f"Missing H5 datasets: {missing}")
        n = int(h5["patches"].shape[0])
        if tuple(h5["patches"].shape[1:]) != INPUT_SHAPE:
            raise RuntimeError(f"Unexpected patch shape: {h5['patches'].shape}")
        split = parse_and_validate_split(split_path, n, h5)
        use_idx = np.concatenate([split["train"], split["val"]])
        # Sorted fancy indexing, then restore canonical train+val order.
        order = np.argsort(use_idx, kind="mergesort")
        sorted_idx = use_idx[order]
        rl_sorted = np.asarray(h5["reader_labels"][sorted_idx], dtype=np.float32)
        rm_sorted = np.asarray(h5["reader_mask"][sorted_idx], dtype=bool)
        restore = np.argsort(order, kind="mergesort")
        reader_labels = rl_sorted[restore]
        reader_mask = rm_sorted[restore]

    targets_all = build_target_arrays(reader_labels, reader_mask)
    split_targets: dict[str, dict[str, np.ndarray]] = {}
    train_n = len(split["train"])
    split_targets["train"] = {k: v[:train_n] for k, v in targets_all.items() if k != "invalid_records"}
    split_targets["val"] = {k: v[train_n:] for k, v in targets_all.items() if k != "invalid_records"}
    pair_counts = {} if a.ranking_target == "reader_pairwise" else ranking_pair_counts(split_targets["train"], a.ranking_target)
    if not is_all and pair_counts:
        pair_counts = {name: pair_counts[name] for name in active_ordered_targets}

    invalid_records = targets_all["invalid_records"]
    for rec in invalid_records:
        global_index = int(use_idx[rec["local_index"]])
        rec["h5_index"] = global_index
        rec.pop("local_index", None)
    print(f"TRAIN={len(split['train'])} VAL={len(split['val'])} | TEST labels/patches not loaded", flush=True)
    print(
        f"ranking_target={a.ranking_target} rank_loss_weight={a.rank_loss_weight:g} "
        f"pair_threshold={a.ranking_pair_threshold:g} min_confidence={a.ranking_min_confidence:g}",
        flush=True,
    )
    for name, counts in pair_counts.items():
        print(f"{name} " + " ".join(f"{key}={value}" for key, value in counts.items()), flush=True)
    if invalid_records:
        print(f"Excluded {len(invalid_records)} invalid active reader rating(s) from their affected attribute only", flush=True)
        save_json(out / "invalid_ratings_excluded.json", invalid_records)

    baseline = build_baselines(split_targets["train"], split_targets["val"], selection)
    save_json(out / "baselines.json", baseline)

    model = SpatialObjectiveModel(a.objective, a.attn_dim, a.head_hidden, a.dropout).to(device)
    print(f"parameters={model.parameter_count():,}", flush=True)
    print("spatial output=128x8x8x8 (512 tokens)", flush=True)
    print("TEST is not loaded.", flush=True)

    train_ds = H5PatchDataset(h5_path, split["train"], split_targets["train"], a.ranking_target)
    val_ds = H5PatchDataset(h5_path, split["val"], split_targets["val"], a.ranking_target)
    common_loader = dict(
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        persistent_workers=a.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common_loader)
    train_diagnostic_loader = DataLoader(train_ds, shuffle=False, drop_last=False, **common_loader)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, a.epochs * steps_per_epoch)
    warmup_steps = max(1, a.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return a.min_lr_factor + (1.0 - a.min_lr_factor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if a.amp == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif a.amp == "fp16" and device.type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16) if device.type == "cuda" else None

    best_score = -float("inf")
    best_epoch = -1
    best_metric = None
    best_attribute_spearman_ge3 = float("nan")
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    start_time = time.time()

    for epoch in range(1, a.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            amp_dtype,
            a.grad_clip,
            a.rank_loss_weight,
            a.ranking_pair_threshold,
            a.ranking_target,
            a.ranking_min_confidence,
            selection,
        )
        for _ in range(steps_per_epoch):
            scheduler.step()
        val_loss, val_metrics, m6_ge3, m6_all, val_pair_diagnostics = evaluate(
            model,
            val_loader,
            device,
            split_targets["val"],
            a.rank_loss_weight,
            a.ranking_pair_threshold,
            a.ranking_target,
            a.ranking_min_confidence,
            selection,
        )

        train_pair_diagnostics = None
        if a.ranking_target == "reader_pairwise":
            train_scores = collect_pairwise_scores(model, train_diagnostic_loader, device)
            train_pair_diagnostics = reader_pairwise_split_diagnostics(
                train_scores,
                split_targets["train"]["reader_ratings"],
                split_targets["train"]["reader_valid"],
                a.ranking_min_confidence,
                selection,
            )

        mean_ordered_pred_sd = float(np.mean([val_metrics[name]["pred_std"] for name in active_ordered_targets]))
        mean_ordered_target_sd = float(np.mean([val_metrics[name]["target_std"] for name in active_ordered_targets]))
        mean_ordered_sd_ratio = float(np.mean([val_metrics[name]["sd_ratio"] for name in active_ordered_targets]))

        finite_ge3 = sum(math.isfinite(val_metrics[n]["spearman_ge3"]) for n in active_ordered_targets)
        finite_all = sum(math.isfinite(val_metrics[n]["spearman"]) for n in active_ordered_targets)
        collapsed = [n for n, values in val_metrics.items() if values.get("pred_std", 1.0) < 1e-4]

        selected_ge3 = [val_metrics[name]["spearman_ge3"] for name in active_ordered_targets]
        if len(active_ordered_targets) == 1 and math.isfinite(selected_ge3[0]):
            score, selection_metric = selected_ge3[0], f"{active_ordered_targets[0]}.spearman_ge3"
        elif group_run and a.attribute_group == "lobulation_spiculation" and all(math.isfinite(value) for value in selected_ge3):
            score = float(np.mean(selected_ge3))
            selection_metric = "mean(lobulation.spearman_ge3,spiculation.spearman_ge3)"
        elif (is_all or (group_run and a.attribute_group == "final_hybrid")) and math.isfinite(m6_ge3):
            score, selection_metric = m6_ge3, "M6>=3"
        elif (is_all or (group_run and a.attribute_group == "final_hybrid")) and math.isfinite(m6_all):
            score, selection_metric = m6_all, "M6all"
        else:
            score, selection_metric = -val_loss, "-val_loss_fallback"

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_ordered_pred_sd": mean_ordered_pred_sd,
            "mean_ordered_target_sd": mean_ordered_target_sd,
            "mean_ordered_sd_ratio": mean_ordered_sd_ratio,
            "m6_ge3": m6_ge3,
            "m6_all": m6_all,
            "finite_ordered_spearman_ge3": finite_ge3,
            "finite_ordered_spearman_all": finite_all,
            "collapsed_heads": ",".join(collapsed),
            "selection_metric": selection_metric,
            "selection_score": score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if not is_all:
            row["attribute_group"] = selection_label
            if len(active_ordered_targets) == 1:
                name = active_ordered_targets[0]
                row["attribute"] = name
                row["attribute_spearman"] = val_metrics[name]["spearman"]
                row["attribute_spearman_ge3"] = val_metrics[name]["spearman_ge3"]
                row["attribute_pred_std"] = val_metrics[name]["pred_std"]
                row["attribute_sd_ratio"] = val_metrics[name]["sd_ratio"]
        if a.ranking_target == "reader_pairwise":
            row.update({
                "train_reader_pairwise_loss": train_pair_diagnostics["reader_pairwise_loss"],
                "train_average_pair_confidence": train_pair_diagnostics["average_pair_confidence"],
                "train_valid_pair_count": train_pair_diagnostics["valid_pair_count"],
                "train_pair_accuracy": train_pair_diagnostics["pair_accuracy"],
                "val_reader_pairwise_loss": val_pair_diagnostics["reader_pairwise_loss"],
                "val_average_pair_confidence": val_pair_diagnostics["average_pair_confidence"],
                "val_valid_pair_count": val_pair_diagnostics["valid_pair_count"],
                "val_pair_accuracy": val_pair_diagnostics["pair_accuracy"],
            })
        for name, metrics in val_metrics.items():
            for key, value in metrics.items():
                if isinstance(value, (int, float, str)):
                    row[f"{name}.{key}"] = value
        history.append(row)

        pair_log = ""
        if a.ranking_target == "reader_pairwise":
            pair_log = (
                f"train_reader_pairwise_loss={train_pair_diagnostics['reader_pairwise_loss']:.4f} "
                f"train_pair_accuracy={train_pair_diagnostics['pair_accuracy']:.4f} "
                f"train_valid_pair_count={train_pair_diagnostics['valid_pair_count']} "
                f"train_average_pair_confidence={train_pair_diagnostics['average_pair_confidence']:.4f} "
                f"val_reader_pairwise_loss={val_pair_diagnostics['reader_pairwise_loss']:.4f} "
                f"val_pair_accuracy={val_pair_diagnostics['pair_accuracy']:.4f} "
                f"val_valid_pair_count={val_pair_diagnostics['valid_pair_count']} "
                f"val_average_pair_confidence={val_pair_diagnostics['average_pair_confidence']:.4f} "
            )
        if selection_metric == "-val_loss_fallback":
            improved = best_metric in {None, "-val_loss_fallback"} and score > best_score
        else:
            improved = best_metric == "-val_loss_fallback" or score > best_score
        if not is_all:
            current_metrics = val_metrics[active_ordered_targets[0]] if len(active_ordered_targets) == 1 else None
            display_best_epoch = epoch if improved else best_epoch
            if current_metrics is not None:
                display_best_ge3 = current_metrics["spearman_ge3"] if improved else best_attribute_spearman_ge3
                print(
                    f"epoch={epoch:03d} best_epoch={display_best_epoch:03d} "
                    f"attribute={active_ordered_targets[0]} "
                    f"spearman={current_metrics['spearman']:+.4f} "
                    f"spearman_ge3={current_metrics['spearman_ge3']:+.4f} "
                    f"best_spearman_ge3={display_best_ge3:+.4f} "
                    f"pred_std={current_metrics['pred_std']:.4f} "
                    f"sd_ratio={current_metrics['sd_ratio']:.4f} "
                    f"train={train_loss:.4f} val={val_loss:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"epoch={epoch:03d} best_epoch={display_best_epoch:03d} "
                    f"attribute_group={selection_label} selection_score={score:+.4f} "
                    f"spearman_ge3={dict(zip(active_ordered_targets, selected_ge3))} "
                    f"train={train_loss:.4f} val={val_loss:.4f}",
                    flush=True,
                )
        else:
            print(
                f"epoch {epoch:03d} | train={train_loss:.4f} val={val_loss:.4f} "
                f"M6>=3={m6_ge3:+.4f} M6all={m6_all:+.4f} "
                f"mean_pred_sd={mean_ordered_pred_sd:.4f} "
                f"mean_target_sd={mean_ordered_target_sd:.4f} "
                f"mean_sd_ratio={mean_ordered_sd_ratio:.4f} "
                f"{pair_log}"
                f"finite_ordered={finite_ge3}/{len(active_ordered_targets)} collapsed={len(collapsed)}/8 "
                f"[{','.join(collapsed) if collapsed else 'none'}]",
                flush=True,
            )
        if improved:
            best_score = score
            best_epoch = epoch
            best_metric = selection_metric
            if len(active_ordered_targets) == 1:
                best_attribute_spearman_ge3 = val_metrics[active_ordered_targets[0]]["spearman_ge3"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": architecture_config(a.objective, a.attn_dim, a.head_hidden, a.dropout),
                    "epoch": epoch,
                    "seed": a.seed,
                    "best_metric": float(score),
                    "best_metric_name": selection_metric,
                    "validation_metrics": val_metrics,
                    "test_used_for_selection": False,
                    "split_hashes": {k: hash_indices(split[k]) for k in ("train", "val", "test")},
                    "target_semantics": {
                        "model_targets": MODEL_TARGETS,
                        "reader_names": READER_NAMES,
                        "reader_to_model": READER_TO_MODEL,
                        "rating_max": RATING_MAX,
                    },
                    "invalid_ratings_excluded": invalid_records,
                    "training_config": {**vars(a), "attribute": a.attribute or "all", "attribute_group": selection_label},
                    "ranking_pair_counts_train": pair_counts,
                    "ranking_target": a.ranking_target,
                    "attribute": a.attribute or "all",
                    "attribute_group": selection_label,
                    "best_attribute_spearman_ge3": best_attribute_spearman_ge3 if len(active_ordered_targets) == 1 else None,
                    "pair_accuracy": val_pair_diagnostics["pair_accuracy"] if val_pair_diagnostics else None,
                    "valid_pair_count": val_pair_diagnostics["valid_pair_count"] if val_pair_diagnostics else None,
                    "average_pair_confidence": val_pair_diagnostics["average_pair_confidence"] if val_pair_diagnostics else None,
                    "train_pair_accuracy": train_pair_diagnostics["pair_accuracy"] if train_pair_diagnostics else None,
                    "train_valid_pair_count": train_pair_diagnostics["valid_pair_count"] if train_pair_diagnostics else None,
                    "train_average_pair_confidence": train_pair_diagnostics["average_pair_confidence"] if train_pair_diagnostics else None,
                    "reader_pairwise_diagnostics": {
                        "train": train_pair_diagnostics,
                        "validation": val_pair_diagnostics,
                    } if a.ranking_target == "reader_pairwise" else None,
                },
                out / "best.pth",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= a.patience:
                print(f"Early stopping; best epoch={best_epoch}", flush=True)
                break

    report = {
        "experiment": "spatial_objective_v1",
        "objective": a.objective,
        "read_only_h5": True,
        "h5_modified": False,
        "split_modified": False,
        "test_loaded_during_training": False,
        "test_used_for_selection": False,
        "h5": str(h5_path),
        "split_file": str(split_path),
        "split_hashes": {k: hash_indices(split[k]) for k in ("train", "val", "test")},
        "canonical_counts": {k: len(split[k]) for k in ("train", "val", "test")},
        "architecture": architecture_config(a.objective, a.attn_dim, a.head_hidden, a.dropout),
        "ranking_config": {
            "rank_loss_weight": a.rank_loss_weight,
            "ranking_target": a.ranking_target,
            "ranking_pair_threshold": a.ranking_pair_threshold,
            "ranking_min_confidence": a.ranking_min_confidence,
            "attribute": a.attribute or "all",
            "attribute_group": selection_label,
            "pair_counts_train": pair_counts,
        },
        "model_parameter_count": model.parameter_count(),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_selection_metric": best_metric,
        "elapsed_seconds": time.time() - start_time,
        "baseline": baseline,
        "invalid_ratings_excluded": invalid_records,
        "last_validation_metrics": val_metrics if history else None,
        "last_variance_diagnostics": {
            "mean_ordered_pred_sd": mean_ordered_pred_sd,
            "mean_ordered_target_sd": mean_ordered_target_sd,
            "mean_ordered_sd_ratio": mean_ordered_sd_ratio,
        } if history else None,
        "variance_diagnostics_by_epoch": [
            {
                "epoch": row["epoch"],
                "mean_ordered_pred_sd": row["mean_ordered_pred_sd"],
                "mean_ordered_target_sd": row["mean_ordered_target_sd"],
                "mean_ordered_sd_ratio": row["mean_ordered_sd_ratio"],
            }
            for row in history
        ],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_count": os.cpu_count(),
        },
        "selection_rule": (
            f"{active_ordered_targets[0]}.spearman_ge3, else -val_loss_fallback when undefined"
            if len(active_ordered_targets) == 1 else
            "mean(lobulation.spearman_ge3, spiculation.spearman_ge3), else -val_loss_fallback when either is undefined"
            if group_run and a.attribute_group == "lobulation_spiculation" else
            "M6>=3, else M6all, else -val_loss_fallback only when all six ordered correlations are undefined"
        ),
        "files": {
            "checkpoint": str((out / "best.pth").resolve()),
            "history": str((out / "history.csv").resolve()),
            "run": str((out / "run.json").resolve()),
        },
    }
    save_json(out / "run.json", report)
    if history:
        fields = list(history[0].keys())
        with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows([clean(r) for r in history])
    print(f"Saved {out / 'best.pth'} | best {best_metric}={best_score:+.4f} epoch={best_epoch}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
