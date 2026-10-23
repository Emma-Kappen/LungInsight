#!/usr/bin/env python3
"""
LungInsight v2 — one-stop diagnostic for the LIDC nodule attribute model.

READ-ONLY:
  * Does not modify the H5.
  * Does not modify the split.
  * Does not touch the test labels.
  * Does not train or save a neural-network checkpoint.
  * Writes only diagnostic artifacts under --out-dir.

Designed to adjudicate the current competing hypotheses:
  1) target/cardinality/reader mapping problem
  2) severe target imbalance / weak baselines
  3) reader disagreement / target noise
  4) image-label signal is weak or absent
  5) current CNN representation is weak
  6) current objective is poorly aligned
  7) apparent collapse is mostly a marginal-prior effect
  8) the pretrained encoder may or may not contain useful signal

This script intentionally stops short of choosing a production architecture.

Required:
  Python 3.x, numpy, h5py

Recommended:
  scipy
  scikit-learn

Example:
  python backend/models/nodule_ranker_3d_distributional_v1/one_stop_diagnostic.py ^
      --h5 datasets/LIDC/lidc_patches.h5 ^
      --split-file backend/models/nodule_ranker_3d/checkpoints_model2_final/split.json ^
      --out-dir backend/models/nodule_ranker_3d_distributional_v1/one_stop_diagnostic ^
      --pretrained-encoder backend/models/nodule_ranker_3d/checkpoints/pretrained_encoder.pth
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np


MODEL_TARGETS = [
    "subtlety",
    "internal_structure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
]

READER_TARGETS = [
    "calcification",
    "internalStructure",
    "lobulation",
    "malignancy",
    "margin",
    "sphericity",
    "spiculation",
    "subtlety",
    "texture",
]

# Reader order -> model order.
READER_TO_MODEL = [7, 1, 0, 5, 4, 2, 6, 8]

# Authoritative LIDC semantics:
#   internal structure: 1..4
#   calcification:      1..6
#   other seven reader characteristics are not all the same type, but
#   the six remaining model attributes here use five-point ordered scales.
AUTHORITATIVE_CARDINALITY = {
    "subtlety": 5,
    "internal_structure": 4,
    "calcification": 6,
    "sphericity": 5,
    "margin": 5,
    "lobulation": 5,
    "spiculation": 5,
    "texture": 5,
}

# Historical/current implementation under audit.
CURRENT_CODE_CARDINALITY = {
    "subtlety": 5,
    "internal_structure": 5,  # intentionally flagged
    "calcification": 6,
    "sphericity": 5,
    "margin": 5,
    "lobulation": 5,
    "spiculation": 5,
    "texture": 5,
}

TEST_INDEX_KEYS = ("test", "test_indices", "test_nodule_indices")
TRAIN_INDEX_KEYS = ("train", "train_indices", "train_nodule_indices")
VAL_INDEX_KEYS = ("val", "validation", "val_indices", "validation_indices", "valid")


def log(msg: str) -> None:
    print(msg, flush=True)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Cannot JSON encode {type(obj)}")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def finite(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x)) if x.size else float("nan")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan")
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    return pearson(rx, ry)


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    d = np.asarray(pred, float) - np.asarray(true, float)
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d * d))) if d.size else float("nan")


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    d = np.abs(np.asarray(pred, float) - np.asarray(true, float))
    d = d[np.isfinite(d)]
    return float(np.mean(d)) if d.size else float("nan")


def exact(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred, float)
    t = np.asarray(true, float)
    ok = np.isfinite(p) & np.isfinite(t)
    return float(np.mean(np.rint(p[ok]) == np.rint(t[ok]))) if np.any(ok) else float("nan")


def within_one(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred, float)
    t = np.asarray(true, float)
    ok = np.isfinite(p) & np.isfinite(t)
    return float(np.mean(np.abs(p[ok] - t[ok]) <= 1.0)) if np.any(ok) else float("nan")


def percentile_dict(x: np.ndarray, ps=(0, 1, 5, 25, 50, 75, 95, 99, 100)) -> Dict[str, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {str(p): float(np.percentile(x, p)) for p in ps}


def load_split(split_path):
    """Load the canonical LungInsight split.json without modifying it."""
    with open(split_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    def as_int_array(value):
        return np.asarray(value, dtype=np.int64).reshape(-1)

    split = {}

    # Canonical schema:
    # train/val/test are dictionaries containing sample_indices and patients.
    for name in ("train", "val", "test"):
        if name not in raw:
            raise ValueError(f"Split file missing required section: {name}")

        section = raw[name]
        if not isinstance(section, dict):
            raise ValueError(
                f"Split section '{name}' must be a dict, got {type(section).__name__}"
            )

        if "sample_indices" not in section:
            raise ValueError(
                f"Split section '{name}' missing 'sample_indices'"
            )

        split[name] = as_int_array(section["sample_indices"])

        # Preserve patient information for leakage checks.
        if "patients" in section:
            split[f"{name}_patients"] = list(section["patients"])

        # Preserve declared counts for validation/reporting.
        if "sample_count" in section:
            split[f"{name}_declared_sample_count"] = int(section["sample_count"])

        if "patient_count" in section:
            split[f"{name}_declared_patient_count"] = int(section["patient_count"])

    # Also preserve the explicit top-level patient lists when present.
    for name in ("train_patients", "val_patients", "test_patients"):
        if name in raw:
            split[name] = list(raw[name])

    # Preserve metadata without trying to convert it to integer arrays.
    if "schema_version" in raw:
        split["schema_version"] = raw["schema_version"]

    if "seed" in raw:
        split["seed"] = raw["seed"]

    # Sanity-check consistency between section-level and top-level counts.
    for name in ("train", "val", "test"):
        declared = split.get(f"{name}_declared_sample_count")
        actual = len(split[name])
        if declared is not None and declared != actual:
            raise ValueError(
                f"{name}: declared sample_count={declared}, "
                f"but sample_indices contains {actual}"
            )

    return split


def decode_scalar(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return bytes(v).decode("utf-8", errors="replace")
    return str(v)


def read_h5(h5_path: Path) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as h5:
        keys = list(h5.keys())
        required = [
            "patches", "reader_labels", "reader_mask", "attribute_mean",
            "attribute_std", "attribute_count", "nodule_id", "patient_id",
            "reader_id", "annotation_id",
        ]
        missing = [k for k in required if k not in h5]
        if missing:
            raise RuntimeError(f"Missing required H5 datasets: {missing}")

        data = {
            "keys": keys,
            "attrs": {str(k): str(v) for k, v in h5.attrs.items()},
            "patches_shape": tuple(h5["patches"].shape),  # type: ignore
            "patches_dtype": str(h5["patches"].dtype),  # type: ignore
            "reader_labels": h5["reader_labels"][:],  # type: ignore
            "reader_mask": h5["reader_mask"][:].astype(bool),  # type: ignore
            "attribute_mean": h5["attribute_mean"][:],  # type: ignore
            "attribute_std": h5["attribute_std"][:],  # type: ignore
            "attribute_count": h5["attribute_count"][:],  # type: ignore
            "nodule_id": np.array([decode_scalar(v) for v in h5["nodule_id"][:]], dtype=object),  # type: ignore
            "patient_id": np.array([decode_scalar(v) for v in h5["patient_id"][:]], dtype=object),  # type: ignore
            "reader_id": np.array([[decode_scalar(v) for v in row] for row in h5["reader_id"][:]], dtype=object),  # type: ignore
            "annotation_id": h5["annotation_id"][:],  # type: ignore
        }

        # Optional arrays.
        for k in ("series_instance_uid", "study_instance_uid",
                  "malignancy_count", "malignancy_distribution",
                  "malignancy_mean", "malignancy_std"):
            if k in h5:
                if h5[k].dtype.kind in "OSU": # type: ignore
                    data[k] = np.array([decode_scalar(v) for v in h5[k][:]], dtype=object) # type: ignore
                else:
                    data[k] = h5[k][:] # type: ignore

        # Patches are not retained in RAM unless requested by the feature probe.
        data["h5_path"] = str(h5_path)
    return data


def validate_split(split: Dict[str, np.ndarray], n: int, patient_ids: np.ndarray) -> Dict[str, Any]:
    result = {}
    sets = {}
    for name in ("train", "val", "test"):
        idx = split[name]
        bad = idx[(idx < 0) | (idx >= n)]
        dup = len(idx) - len(np.unique(idx))
        result[name] = {
            "n": int(len(idx)),
            "bad_indices": int(len(bad)),
            "duplicate_indices": int(dup),
            "patients": int(len(np.unique(patient_ids[idx]))),
        }
        sets[name] = set(idx.tolist())

    result["overlap_train_val"] = len(sets["train"] & sets["val"])
    result["overlap_train_test"] = len(sets["train"] & sets["test"])
    result["overlap_val_test"] = len(sets["val"] & sets["test"])

    psets = {
        name: set(patient_ids[split[name]].tolist())
        for name in ("train", "val", "test")
    }
    result["patient_overlap_train_val"] = len(psets["train"] & psets["val"])
    result["patient_overlap_train_test"] = len(psets["train"] & psets["test"])
    result["patient_overlap_val_test"] = len(psets["val"] & psets["test"])
    return result


def build_targets(
    reader_labels,
    reader_mask,
    target_names,
    reader_to_model,
    cardinalities,
):
    """
    Build reader-distribution targets while treating out-of-range ratings
    as invalid for that specific attribute/reader.

    IMPORTANT:
      - Does not modify the underlying reader labels.
      - Does not clip or remap invalid ratings.
      - Invalid ratings are excluded only from the affected attribute.
      - Other valid attributes from the same reader remain usable.
      - Returns per-attribute validity so downstream diagnostics can
        distinguish unsupported targets from genuine zero-valued targets.
    """
    reader_labels = np.asarray(reader_labels, dtype=np.float32)
    reader_mask = np.asarray(reader_mask, dtype=bool)

    if reader_labels.ndim != 3:
        raise ValueError(
            f"reader_labels must have shape [N,R,A], got {reader_labels.shape}"
        )

    if reader_mask.shape != reader_labels.shape[:2]:
        raise ValueError(
            "reader_mask shape mismatch: "
            f"labels={reader_labels.shape[:2]} mask={reader_mask.shape}"
        )

    n_samples = reader_labels.shape[0]

    target_names = list(target_names)
    reader_to_model = list(reader_to_model)
    cardinalities = list(cardinalities)

    n_targets = len(target_names)

    if len(reader_to_model) != n_targets:
        raise ValueError(
            f"reader_to_model length {len(reader_to_model)} != "
            f"target count {n_targets}"
        )

    if len(cardinalities) != n_targets:
        raise ValueError(
            f"cardinalities length {len(cardinalities)} != "
            f"target count {n_targets}"
        )

    distributions = []
    means = []
    counts = []
    valid_targets = []

    print("\nTarget-rating validation:")

    for target_idx, name in enumerate(target_names):
        reader_attr_idx = int(reader_to_model[target_idx])
        K = int(cardinalities[target_idx])

        if reader_attr_idx < 0 or reader_attr_idx >= reader_labels.shape[2]:
            raise ValueError(
                f"{name}: reader attribute index {reader_attr_idx} "
                f"outside reader_labels dimension {reader_labels.shape[2]}"
            )

        values = reader_labels[:, :, reader_attr_idx]
        valid_reader_mask = reader_mask.copy()

        # A reader rating is usable only if:
        #   1. the reader slot is active, and
        #   2. the rating is finite, and
        #   3. the rating lies inside the authoritative category range.
        finite = np.isfinite(values)
        in_range = (values >= 1.0) & (values <= float(K))

        invalid_active = reader_mask & ~(finite & in_range)

        invalid_count = int(invalid_active.sum())

        if invalid_count:
            bad_values = values[invalid_active]
            unique_bad, unique_counts = np.unique(
                bad_values,
                return_counts=True,
            )

            bad_summary = {
                str(float(v)): int(c)
                for v, c in zip(unique_bad, unique_counts)
            }

            print(
                f"  WARNING {name}: excluded {invalid_count} invalid "
                f"active reader rating(s); invalid_values={bad_summary}"
            )

        valid_reader_mask &= finite & in_range

        # Build categorical empirical distribution.
        dist = np.zeros((n_samples, K), dtype=np.float32)
        mean = np.full(n_samples, np.nan, dtype=np.float32)
        count = valid_reader_mask.sum(axis=1).astype(np.int32)

        for i in range(n_samples):
            vals = values[i, valid_reader_mask[i]]

            if vals.size == 0:
                continue

            vals_int = vals.astype(np.int64)

            # Categories are 1..K; convert to zero-based bins.
            bincount = np.bincount(
                vals_int - 1,
                minlength=K,
            )[:K]

            dist[i] = bincount.astype(np.float32) / float(vals.size)
            mean[i] = float(vals.mean())

        valid_sample = count > 0

        distributions.append(dist)
        means.append(mean)
        counts.append(count)
        valid_targets.append(valid_sample)

    # Each attribute has its own cardinality K, so distributions have
    # different shapes: [N, K_attribute]. Keep them as a list.
    distributions = list(distributions)

    means = np.stack(means, axis=1)
    counts = np.stack(counts, axis=1)
    valid_targets = np.stack(valid_targets, axis=1)

    print(
        f"  Samples with at least one valid reader rating per target:"
    )

    for target_idx, name in enumerate(target_names):
        n_valid = int(valid_targets[:, target_idx].sum())
        n_total = n_samples
        print(
            f"    {name}: {n_valid}/{n_total} "
            f"({100.0 * n_valid / max(n_total, 1):.2f}%)"
        )

    return distributions, means, counts, valid_targets


def categorical_counts(
    reader_labels: np.ndarray,
    reader_mask: np.ndarray,
    model_idx: int,
    k: int,
) -> Dict[str, int]:
    rj = READER_TO_MODEL[model_idx]
    vals = reader_labels[:, :, rj]
    mask = reader_mask & np.isfinite(vals)
    vals = np.rint(vals[mask]).astype(int)
    return {str(i): int(np.sum(vals == i)) for i in range(1, k + 1)}


def per_nodule_mode(mean: np.ndarray, count: np.ndarray, j: int) -> np.ndarray:
    # This is a simple validation baseline based on training nodal means.
    return mean[:, j].copy()


def constant_baseline(train_y: np.ndarray, val_y: np.ndarray) -> Dict[str, Any]:
    tr = train_y[np.isfinite(train_y)]
    va = val_y[np.isfinite(val_y)]
    if tr.size == 0 or va.size == 0:
        return {"n_train": int(tr.size), "n_val": int(va.size)}
    pred = np.full_like(va, np.median(tr), dtype=float)
    return {
        "n_train": int(tr.size),
        "n_val": int(va.size),
        "train_median": float(np.median(tr)),
        "train_mean": float(np.mean(tr)),
        "train_std": float(np.std(tr)),
        "val_mae_median": mae(pred, va),
        "val_rmse_median": rmse(pred, va),
        "val_exact_median": exact(pred, va),
        "val_within1_median": within_one(pred, va),
    }


def mode_rating_baseline(
    reader_labels: np.ndarray,
    reader_mask: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    j: int,
    k: int,
) -> Dict[str, Any]:
    rj = READER_TO_MODEL[j]
    tr = reader_labels[train_idx, :, rj]
    tm = reader_mask[train_idx] & np.isfinite(tr)
    tv = reader_labels[val_idx, :, rj]
    vm = reader_mask[val_idx] & np.isfinite(tv)
    flat = np.rint(tr[tm]).astype(int)
    counts = np.bincount(flat, minlength=k + 1)[1:]
    mode = int(np.argmax(counts) + 1) if counts.sum() else 1
    v = np.rint(tv[vm]).astype(int)
    return {
        "train_reader_counts": {str(i + 1): int(counts[i]) for i in range(k)},
        "train_global_mode": mode,
        "val_reader_count": int(v.size),
        "val_accuracy_global_mode": float(np.mean(v == mode)) if v.size else float("nan"),
    }


def agreement_stats(
    reader_labels: np.ndarray,
    reader_mask: np.ndarray,
    j: int,
) -> Dict[str, Any]:
    rj = READER_TO_MODEL[j]
    vals = reader_labels[:, :, rj]
    mask = reader_mask & np.isfinite(vals)
    counts = mask.sum(axis=1)
    usable = counts >= 2

    disagreements = []
    pair_agree = []
    leave_one_out_mean_shift = []

    for i in np.where(usable)[0]:
        v = np.rint(vals[i, mask[i]]).astype(int)
        disagreements.append(int(np.unique(v).size > 1))
        if len(v) >= 2:
            pairs = 0
            agree = 0
            for a in range(len(v)):
                for b in range(a + 1, len(v)):
                    pairs += 1
                    agree += int(v[a] == v[b])
            pair_agree.append(agree / pairs)

        full_mean = float(np.mean(v))
        if len(v) >= 3:
            shifts = [abs(float(np.mean(np.delete(v, q))) - full_mean) for q in range(len(v))]
            leave_one_out_mean_shift.extend(shifts)

    return {
        "nodules_total": int(len(vals)),
        "nodules_with_2plus_readers": int(np.sum(usable)),
        "fraction_disagree_2plus": float(np.mean(disagreements)) if disagreements else float("nan"),
        "pairwise_exact_agreement": float(np.mean(pair_agree)) if pair_agree else float("nan"),
        "leave_one_reader_mean_abs_shift": float(np.mean(leave_one_out_mean_shift))
        if leave_one_out_mean_shift else float("nan"),
        "reader_count_distribution": {
            str(k): int(np.sum(counts == k)) for k in range(1, int(counts.max()) + 1)
        } if counts.size else {},
    }


def reader_count_target_stability(
    mean: np.ndarray,
    count: np.ndarray,
    j: int,
) -> Dict[str, Any]:
    rows = []
    for c in sorted(np.unique(count[:, j])):
        y = mean[count[:, j] == c, j]
        y = y[np.isfinite(y)]
        if y.size:
            rows.append({
                "reader_count": int(c),
                "n": int(y.size),
                "target_mean": float(np.mean(y)),
                "target_std": float(np.std(y)),
                "target_median": float(np.median(y)),
            })
    return rows # type: ignore


def split_target_stats(mean: np.ndarray, count: np.ndarray, split: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    rows = []
    for split_name in ("train", "val", "test"):
        idx = split[split_name]
        for j, name in enumerate(MODEL_TARGETS):
            y = mean[idx, j]
            y = y[np.isfinite(y)]
            rows.append({
                "split": split_name,
                "attribute": name,
                "n_nodules": int(y.size),
                "mean": safe_mean(y),
                "std": safe_std(y),
                "min": float(np.min(y)) if y.size else float("nan"),
                "max": float(np.max(y)) if y.size else float("nan"),
                "median": float(np.median(y)) if y.size else float("nan"),
                "p25": float(np.percentile(y, 25)) if y.size else float("nan"),
                "p75": float(np.percentile(y, 75)) if y.size else float("nan"),
                "mean_reader_count": safe_mean(count[idx, j]),
            })
    return rows


def feature_vector_from_patch(x: np.ndarray) -> np.ndarray:
    """
    Cheap, interpretable CT-only features. Assumes the H5 patch is already in HU
    and approximately [-1000, 400], as documented by the project.
    """
    x = np.asarray(x, dtype=np.float32)
    flat = x.reshape(-1)

    q = np.percentile(flat, [1, 5, 10, 25, 50, 75, 90, 95, 99]).astype(np.float32)
    feats = list(q)
    feats += [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.mean(x < -900)),
        float(np.mean((x >= -900) & (x < -300))),
        float(np.mean((x >= -300) & (x < 100))),
        float(np.mean((x >= 100) & (x < 300))),
        float(np.mean(x >= 300)),
        float(np.mean(x >= 200)),
        float(np.mean(x >= 0)),
    ]

    # Central crop statistics: 1/2-width and 1/4-width cubes.
    n = x.shape[0]
    c = n // 2
    h = max(1, n // 4)
    central = x[c - h:c + h, c - h:c + h, c - h:c + h]
    q2 = np.percentile(central.reshape(-1), [5, 25, 50, 75, 95]).astype(np.float32)
    feats += list(q2)
    feats += [
        float(np.mean(central)),
        float(np.std(central)),
        float(np.mean(central >= 200)),
        float(np.mean(central >= 300)),
    ]

    # Six coarse spatial blocks; mean/std and high-HU fraction.
    starts = [0, n // 4, n // 2, (3 * n) // 4]
    blocks = [
        x[:n//2, :n//2, :n//2],
        x[:n//2, n//2:, :n//2],
        x[:n//2, :n//2, n//2:],
        x[n//2:, :n//2, :n//2],
        x[n//2:, n//2:, n//2:],
        x[n//2:, n//2:, :n//2],
    ]
    for b in blocks:
        feats += [float(np.mean(b)), float(np.std(b)), float(np.mean(b >= 200))]

    return np.asarray(feats, dtype=np.float32)


def handcrafted_probe(
    h5_path: Path,
    mean: np.ndarray,
    split: Dict[str, np.ndarray],
    out_dir: Path,
    seed: int = 123,
    max_samples: int = 0,
) -> Dict[str, Any]:
    """
    Cheap CPU diagnostic. Uses only training labels to fit simple probes and
    validation labels for evaluation. No test labels.

    Models:
      * Ridge on continuous per-nodule mean targets.
      * Logistic regression for the two categorical attributes where possible.

    If sklearn is absent, this stage is skipped with an explicit reason.
    """
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        return {"status": "skipped", "reason": f"scikit-learn unavailable: {e}"}

    rng = np.random.default_rng(seed)
    train_idx = split["train"].copy()
    val_idx = split["val"].copy()

    if max_samples and len(train_idx) > max_samples:
        train_idx = np.sort(rng.choice(train_idx, size=max_samples, replace=False))
    if max_samples and len(val_idx) > max_samples:
        val_idx = np.sort(rng.choice(val_idx, size=max_samples, replace=False))

    log(f"[features] extracting {len(train_idx)} train + {len(val_idx)} val patches on CPU")
    Xtr = []
    Xva = []
    with h5py.File(h5_path, "r") as h5:
        patches = h5["patches"]
        for i in train_idx:
            Xtr.append(feature_vector_from_patch(patches[int(i)])) # type: ignore
        for i in val_idx:
            Xva.append(feature_vector_from_patch(patches[int(i)])) # type: ignore

    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xva = np.asarray(Xva, dtype=np.float32)
    np.save(out_dir / "handcrafted_train_features.npy", Xtr)
    np.save(out_dir / "handcrafted_val_features.npy", Xva)

    results = []
    for j, name in enumerate(MODEL_TARGETS):
        ytr = mean[train_idx, j]
        yva = mean[val_idx, j]
        oktr = np.isfinite(ytr)
        okva = np.isfinite(yva)
        if oktr.sum() < 20 or okva.sum() < 10:
            continue

        # Continuous probe against the reader mean for every head.
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        )
        model.fit(Xtr[oktr], ytr[oktr])
        pred = model.predict(Xva[okva])
        results.append({
            "attribute": name,
            "probe": "handcrafted_ridge_mean",
            "n_train": int(oktr.sum()),
            "n_val": int(okva.sum()),
            "mae": mae(pred, yva[okva]), # type: ignore
            "rmse": rmse(pred, yva[okva]), # type: ignore
            "spearman": spearman(pred, yva[okva]), # type: ignore
            "pearson": pearson(pred, yva[okva]), # type: ignore
            "pred_std": safe_std(pred), # type: ignore
            "true_std": safe_std(yva[okva]),
        })

        # For categorical targets, also try multinomial logistic regression
        # on the rounded per-nodule mean. This is a diagnostic only.
        if name in ("internal_structure", "calcification"):
            k = AUTHORITATIVE_CARDINALITY[name]
            ctr = np.rint(ytr[oktr]).astype(int)
            cva = np.rint(yva[okva]).astype(int)
            valid_tr = (ctr >= 1) & (ctr <= k)
            valid_va = (cva >= 1) & (cva <= k)
            classes = np.unique(ctr[valid_tr])
            if len(classes) >= 2:
                clf = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                )
                clf.fit(Xtr[oktr][valid_tr], ctr[valid_tr])
                cp = clf.predict(Xva[okva][valid_va])
                results.append({
                    "attribute": name,
                    "probe": "handcrafted_logistic_balanced",
                    "n_train": int(valid_tr.sum()),
                    "n_val": int(valid_va.sum()),
                    "accuracy": float(np.mean(cp == cva[valid_va])),
                    "macro_accuracy": None,
                    "pred_unique": sorted(np.unique(cp).astype(int).tolist()),
                    "true_unique": sorted(np.unique(cva[valid_va]).astype(int).tolist()),
                })

    write_csv(
        out_dir / "handcrafted_probe.csv",
        results,
        sorted({k for r in results for k in r.keys()}) if results else
        ["attribute", "probe"],
    )
    return {
        "status": "completed",
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "feature_dim": int(Xtr.shape[1]), # type: ignore
        "results": results,
    }


def inspect_current_code(
    project_root: Path,
) -> Dict[str, Any]:
    """
    Read-only static audit of the currently checked-in distributional-v1 model
    and trainer. This does not execute training.
    """
    result: Dict[str, Any] = {"status": "not_found"}
    model_path = project_root / "backend/models/nodule_ranker_3d_distributional_v1/model_3d.py"
    trainer_path = project_root / "backend/models/nodule_ranker_3d_distributional_v1/train_nodule_ranker_3d.py"

    result["model_file"] = str(model_path)
    result["trainer_file"] = str(trainer_path)

    if not model_path.exists() and not trainer_path.exists():
        return result

    text_parts = {}
    for p in (model_path, trainer_path):
        if p.exists():
            text_parts[p.name] = p.read_text(encoding="utf-8", errors="replace")
    result["files_present"] = list(text_parts.keys())

    # Cheap textual checks. These are deliberately conservative: "present" does
    # not mean "correct", only that the implementation contains the construct.
    trainer_text = text_parts.get(trainer_path.name, "")
    model_text = text_parts.get(model_path.name, "")

    result["static_checks"] = {
        "NoduleRanker3D_class_present": "class NoduleRanker3D" in model_text,
        "READER_TO_MODEL_present": "READER_TO_MODEL" in trainer_text,
        "internal_structure_5_present": "internal_structure" in trainer_text and "5" in trainer_text,
        "calcification_6_present": "calcification" in trainer_text and "6" in trainer_text,
        "distribution_loss_present": "distribution_loss" in trainer_text,
        "expected_rating_present": "expected_rating" in trainer_text,
        "pretrained_support_present": "--pretrained" in trainer_text or "pretrained" in trainer_text,
        "spearman_selection_present": "selection_score" in trainer_text and "spearman" in trainer_text.lower(),
    }

    # Extract likely target-cardinality constants if obvious.
    hits = []
    for line in trainer_text.splitlines():
        low = line.lower()
        if "cardinal" in low or "rating_max" in low or "internal_structure" in low or "calcification" in low:
            hits.append(line.strip())
    result["relevant_source_lines"] = hits[:100]
    result["status"] = "completed"
    return result


def optional_pretrained_encoder_audit(checkpoint: Optional[Path]) -> Dict[str, Any]:
    """
    Static checkpoint inspection only. It intentionally does not assume a
    particular encoder API, because the diagnostic must remain usable if the
    project model implementation changes.

    The report tells us whether the checkpoint is structurally loadable and what
    metadata it declares. A true frozen-embedding probe is left as an explicit
    follow-up if the model API can be verified.
    """
    if checkpoint is None:
        return {"status": "not_requested"}
    if not checkpoint.exists():
        return {"status": "missing", "path": str(checkpoint)}

    result = {"status": "completed", "path": str(checkpoint), "size_bytes": checkpoint.stat().st_size}
    try:
        import torch
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            result["top_level_keys"] = sorted(str(k) for k in ckpt.keys())
            for k in ("encoder_config", "pretrain_config", "input_shape",
                      "epochs_completed", "complete", "train_nodule_indices",
                      "train_patients", "preprocessing"):
                if k in ckpt:
                    v = ckpt[k]
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    result[k] = v
            sd = ckpt.get("encoder_state_dict")
            if isinstance(sd, dict):
                result["encoder_state_dict_keys"] = len(sd)
                result["encoder_parameter_count"] = int(sum(
                    int(v.numel()) for v in sd.values() if hasattr(v, "numel")
                ))
        else:
            result["object_type"] = str(type(ckpt))
    except Exception as e:
        result["load_error"] = repr(e)
        result["traceback"] = traceback.format_exc(limit=3)
    return result


def hash_indices(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = Path(args.h5).resolve()
    split_path = Path(args.split_file).resolve()
    project_root = Path(args.project_root).resolve() if args.project_root else h5_path.parents[2]

    log("=== LungInsight v2 / ONE-STOP DIAGNOSTIC ===")
    log("READ-ONLY: H5, split, and test labels are not modified.")
    log(f"H5:   {h5_path}")
    log(f"Split: {split_path}")
    log(f"Out:  {out_dir}")

    report: Dict[str, Any] = {
        "diagnostic_version": "1.0",
        "h5": str(h5_path),
        "split_file": str(split_path),
        "project_root": str(project_root),
        "read_only": True,
        "test_labels_used_for_model_selection": False,
        "model_targets": MODEL_TARGETS,
        "reader_targets": READER_TARGETS,
        "reader_to_model": READER_TO_MODEL,
        "authoritative_cardinality": AUTHORITATIVE_CARDINALITY,
        "current_code_cardinality": CURRENT_CODE_CARDINALITY,
    }

    # ------------------------------------------------------------------
    # 1. H5 integrity
    # ------------------------------------------------------------------
    log("\n[1/8] H5 integrity and schema")
    data = read_h5(h5_path)
    n = int(data["patches_shape"][0])
    h5_report = {
        "keys": data["keys"],
        "attrs": data["attrs"],
        "patches_shape": data["patches_shape"],
        "patches_dtype": data["patches_dtype"],
        "reader_labels_shape": list(data["reader_labels"].shape),
        "reader_mask_shape": list(data["reader_mask"].shape),
        "attribute_mean_shape": list(data["attribute_mean"].shape),
        "attribute_std_shape": list(data["attribute_std"].shape),
        "attribute_count_shape": list(data["attribute_count"].shape),
        "nonfinite_reader_labels": int(np.sum(~np.isfinite(data["reader_labels"]))),
        "nonfinite_attribute_mean": int(np.sum(~np.isfinite(data["attribute_mean"]))),
        "nonfinite_attribute_std": int(np.sum(~np.isfinite(data["attribute_std"]))),
        "nonfinite_patches": None,
        "constant_patches": None,
    }

    with h5py.File(h5_path, "r") as h5:
        # Chunked scan to avoid loading the full 2.6k*64^3 array.
        patches = h5["patches"]
        nonfinite = 0
        constant = 0
        mins = []
        maxs = []
        for start in range(0, n, args.patch_scan_batch):
            stop = min(n, start + args.patch_scan_batch)
            x = patches[start:stop] # type: ignore
            nonfinite += int(np.sum(~np.isfinite(x))) # type: ignore
            flat = x.reshape(x.shape[0], -1) # type: ignore
            constant += int(np.sum(np.ptp(flat, axis=1) == 0))
            mins.append(float(np.min(x))) # type: ignore
            maxs.append(float(np.max(x))) # type: ignore
    h5_report["nonfinite_patches"] = nonfinite
    h5_report["constant_patches"] = constant
    h5_report["patch_global_min"] = min(mins) if mins else None
    h5_report["patch_global_max"] = max(maxs) if maxs else None
    report["h5_integrity"] = h5_report

    # ------------------------------------------------------------------
    # 2. Split integrity
    # ------------------------------------------------------------------
    log("\n[2/8] Split integrity")
    split = load_split(split_path)
    split_report = validate_split(split, n, data["patient_id"])
    split_report["index_hashes"] = {
        name: hash_indices(split[name]) for name in ("train", "val", "test")
    }
    report["split_integrity"] = split_report
    log(json.dumps(split_report, indent=2))

    # ------------------------------------------------------------------
    # 3. Target semantics and construction audit
    # ------------------------------------------------------------------
    log("\n[3/8] Target semantics / cardinality / mapping")
    semantics = {
        "authoritative": AUTHORITATIVE_CARDINALITY,
        "current_implementation_assumption": CURRENT_CODE_CARDINALITY,
        "discrepancies": [],
    }
    for name in MODEL_TARGETS:
        if AUTHORITATIVE_CARDINALITY[name] != CURRENT_CODE_CARDINALITY[name]:
            semantics["discrepancies"].append({
                "attribute": name,
                "authoritative_K": AUTHORITATIVE_CARDINALITY[name],
                "current_code_K": CURRENT_CODE_CARDINALITY[name],
                "severity": "high",
            })

    dist, mean, count, valid = build_targets(
        data["reader_labels"], 
        data["reader_mask"], 
        AUTHORITATIVE_CARDINALITY.keys(), 
        READER_TO_MODEL, 
        AUTHORITATIVE_CARDINALITY.values())

    consistency_rows = []
    for j, name in enumerate(MODEL_TARGETS):
        k = AUTHORITATIVE_CARDINALITY[name]
        reader_j = READER_TO_MODEL[j]
        vals = data["reader_labels"][:, :, reader_j]
        mask = data["reader_mask"] & np.isfinite(vals)
        calc_mean = mean[:, j]
        calc_std = np.full(n, np.nan, dtype=np.float32)
        calc_count = count[:, j]
        for i in range(n):
            v = vals[i, mask[i]]
            if len(v):
                calc_std[i] = float(np.std(v))

        am = data["attribute_mean"][:, j]
        ast = data["attribute_std"][:, j]
        ac = data["attribute_count"][:, j]

        mean_diff = np.nanmax(np.abs(am - calc_mean)) if np.any(np.isfinite(am - calc_mean)) else 0.0
        std_diff = np.nanmax(np.abs(ast - calc_std)) if np.any(np.isfinite(ast - calc_std)) else 0.0
        count_diff = int(np.sum(ac != calc_count))

        consistency_rows.append({
            "attribute": name,
            "reader_index": reader_j,
            "K": k,
            "computed_valid_nodules": int(np.sum(valid[:, j])),
            "computed_mean": safe_mean(mean[:, j]),
            "computed_std": safe_std(mean[:, j]),
            "computed_reader_count_mean": safe_mean(count[:, j]),
            "max_abs_mean_difference": float(mean_diff),
            "max_abs_std_difference": float(std_diff),
            "count_mismatch_nodules": count_diff,
        })
    report["target_semantics"] = semantics
    report["target_consistency"] = consistency_rows
    write_csv(
        out_dir / "target_consistency.csv",
        consistency_rows,
        list(consistency_rows[0].keys()) if consistency_rows else ["attribute"],
    )

    # ------------------------------------------------------------------
    # 4. Reader agreement / label distribution
    # ------------------------------------------------------------------
    log("\n[4/8] Reader counts, agreement, distributions")
    agreement = {}
    category_rows = []
    for j, name in enumerate(MODEL_TARGETS):
        agreement[name] = agreement_stats(data["reader_labels"], data["reader_mask"], j)
        for split_name in ("train", "val", "test"):
            idx = split[split_name]
            counts = categorical_counts(
                data["reader_labels"], data["reader_mask"], j, AUTHORITATIVE_CARDINALITY[name]
            )
            # Overall category counts are useful for audit; split-specific
            # counts are recomputed below.
            vals = data["reader_labels"][idx, :, READER_TO_MODEL[j]]
            m = data["reader_mask"][idx] & np.isfinite(vals)
            vv = np.rint(vals[m]).astype(int)
            for rating in range(1, AUTHORITATIVE_CARDINALITY[name] + 1):
                category_rows.append({
                    "split": split_name,
                    "attribute": name,
                    "rating": rating,
                    "count": int(np.sum(vv == rating)),
                    "fraction": float(np.mean(vv == rating)) if vv.size else float("nan"),
                })
    report["reader_agreement"] = agreement
    write_csv(
        out_dir / "category_counts.csv",
        category_rows,
        ["split", "attribute", "rating", "count", "fraction"],
    )

    stability_rows = []
    for j, name in enumerate(MODEL_TARGETS):
        for row in reader_count_target_stability(mean, count, j):
            row["attribute"] = name # type: ignore
            stability_rows.append(row)
    write_csv(
        out_dir / "reader_count_target_stability.csv",
        stability_rows,
        ["attribute", "reader_count", "n", "target_mean", "target_std", "target_median"],
    )

    # ------------------------------------------------------------------
    # 5. Baselines + split shift
    # ------------------------------------------------------------------
    log("\n[5/8] Training-derived baselines and split shift")
    baseline_rows = []
    for j, name in enumerate(MODEL_TARGETS):
        tr = mean[split["train"], j]
        va = mean[split["val"], j]
        b = constant_baseline(tr, va)
        b["attribute"] = name
        b["baseline"] = "train_median_mean_target"
        baseline_rows.append(b)

        mb = mode_rating_baseline(
            data["reader_labels"], data["reader_mask"],
            split["train"], split["val"], j, AUTHORITATIVE_CARDINALITY[name]
        )
        mb["attribute"] = name
        mb["baseline"] = "train_global_reader_rating_mode"
        baseline_rows.append(mb)

    write_csv(
        out_dir / "baselines.csv",
        baseline_rows,
        sorted({k for r in baseline_rows for k in r.keys()}),
    )
    report["split_target_stats"] = split_target_stats(mean, count, split)

    # ------------------------------------------------------------------
    # 6. Handcrafted image-signal probe
    # ------------------------------------------------------------------
    log("\n[6/8] Cheap handcrafted CT-feature probe")
    probe = handcrafted_probe(
        h5_path, mean, split, out_dir,
        seed=args.seed,
        max_samples=args.max_feature_samples,
    )
    report["handcrafted_probe"] = {
        k: v for k, v in probe.items() if k != "results"
    }
    report["handcrafted_probe_results"] = probe.get("results", [])

    # ------------------------------------------------------------------
    # 7. Current-code + pretrained checkpoint audit
    # ------------------------------------------------------------------
    log("\n[7/8] Current implementation and pretrained checkpoint audit")
    report["current_code_audit"] = inspect_current_code(project_root)
    pretrained = Path(args.pretrained_encoder).resolve() if args.pretrained_encoder else None
    report["pretrained_encoder_audit"] = optional_pretrained_encoder_audit(pretrained)

    # ------------------------------------------------------------------
    # 8. One-stop adjudication
    # ------------------------------------------------------------------
    log("\n[8/8] Adjudicating hypotheses")
    adjudication = []

    def add_hypothesis(name: str, status: str, evidence: List[str], next_test: str) -> None:
        adjudication.append({
            "hypothesis": name,
            "status": status,
            "evidence": evidence,
            "next_decisive_test": next_test,
        })

    # Target cardinality.
    if semantics["discrepancies"]:
        add_hypothesis(
            "Internal-structure cardinality is specified incorrectly in the current implementation.",
            "SUPPORTED_AS_SPECIFICATION_PROBLEM",
            [
                "Authoritative LIDC schema uses internalStructure codes 1..4.",
                "Current implementation assumptions report internal_structure K=5.",
                "The discrepancy is testable directly from the H5 and model code.",
            ],
            "Audit the actual head dimension, target builder, and every loss/metric path before another training run.",
        )
    else:
        add_hypothesis(
            "Target cardinality is wrong in the current implementation.",
            "NOT_SUPPORTED_BY_THIS_AUDIT",
            ["No authoritative-vs-current cardinality discrepancy was found in the configured constants."],
            "Still inspect runtime model output dimensions if training code differs from these constants.",
        )

    # Split leakage.
    patient_overlap = (
        split_report["patient_overlap_train_val"]
        + split_report["patient_overlap_train_test"]
        + split_report["patient_overlap_val_test"]
    )
    add_hypothesis(
        "Patient-level split leakage explains the failure.",
        "SUPPORTED" if patient_overlap else "WEAKENED",
        [f"Patient overlap across split pairs = {patient_overlap}."],
        "If nonzero, stop model work and repair split. If zero, this hypothesis is weakened.",
    )

    # Baseline dominance.
    baseline_evidence = []
    for j, name in enumerate(MODEL_TARGETS):
        yv = mean[split["val"], j]
        yt = yv[np.isfinite(yv)]
        if yt.size:
            baseline_evidence.append(
                f"{name}: validation mean-target SD={np.std(yt):.3f}, "
                f"median-baseline MAE={mae(np.full_like(yt, np.median(mean[split['train'], j][np.isfinite(mean[split['train'], j])])), yt):.3f}"
            )
    add_hypothesis(
        "Several attributes are dominated by marginal prevalence, making raw accuracy/MAE misleading.",
        "SUPPORTED" if any(
            name in ("internal_structure", "calcification")
            and safe_std(mean[split["val"], j]) < 0.30
            for j, name in enumerate(MODEL_TARGETS)
        ) else "UNRESOLVED",
        baseline_evidence,
        "Compare the neural model against the training-derived baselines per attribute; do not use aggregate accuracy as the decision metric.",
    )

    # Reader noise.
    disagreement_vals = [
        agreement[name]["fraction_disagree_2plus"]
        for name in MODEL_TARGETS
        if np.isfinite(agreement[name]["fraction_disagree_2plus"])
    ]
    add_hypothesis(
        "Reader disagreement materially limits attainable performance.",
        "SUPPORTED_AS_CONTRIBUTOR" if disagreement_vals and np.mean(disagreement_vals) > 0.20 else "UNRESOLVED",
        [
            f"Mean fraction of 2+ reader nodules with disagreement across attributes = {np.mean(disagreement_vals):.3f}"
            if disagreement_vals else "Agreement statistics unavailable."
        ],
        "Use reader-count-stratified stability and agreement to estimate an attainable ceiling before judging architecture.",
    )

    # Image signal.
    hp = report["handcrafted_probe_results"]
    hp_spears = [
        r["spearman"] for r in hp
        if r.get("probe") == "handcrafted_ridge_mean" and np.isfinite(r.get("spearman", np.nan))
    ]
    add_hypothesis(
        "There is recoverable image signal in cheap CT statistics.",
        "SUPPORTED" if hp_spears and max(hp_spears) >= 0.20 else
        ("WEAKENED" if hp_spears else "UNRESOLVED"),
        [
            f"Best handcrafted Ridge validation Spearman = {max(hp_spears):.3f}"
            if hp_spears else "Handcrafted probe did not run."
        ],
        "If this probe is useful while the CNN is not, prioritize sample-efficient representations rather than a larger CNN.",
    )

    # Current model interpretation, using the known run as an external observation
    # rather than recomputing its checkpoint.
    add_hypothesis(
        "The observed best-checkpoint collapse is only an early-stopping artifact.",
        "WEAKENED",
        [
            "The prior full run's last checkpoint recovered prediction variance but remained near-zero in Spearman.",
            "Therefore variance recovery alone did not restore generalization.",
        ],
        "Do not respond to collapse with longer training alone.",
    )

    add_hypothesis(
        "The existing distributional objective is proven to be the root cause.",
        "NOT_PROVEN",
        [
            "Current diagnostics can establish target semantics and baselines, but cannot prove an objective is causal without a matched objective comparison.",
            "A small-data failure can arise from representation, supervision, objective, or their interaction.",
        ],
        "Only compare objectives after target/cardinality/mapping integrity and signal probes are established.",
    )

    add_hypothesis(
        "The pretrained encoder is the root cause.",
        "UNRESOLVED",
        [
            "Checkpoint metadata can be inspected here, but metadata alone cannot establish transfer quality.",
            "A frozen embedding probe is required to distinguish representation quality from head/objective failure.",
        ],
        "Run a frozen-encoder linear/nonlinear probe using the same train/validation split and target construction.",
    )

    add_hypothesis(
        "A wholesale architecture replacement is currently justified.",
        "NOT_JUSTIFIED_YET",
        [
            "Multiple opinions independently recommend diagnostic isolation before architecture search.",
            "The current evidence does not isolate the encoder as the bottleneck.",
        ],
        "Choose the next model only after the diagnostic report identifies the limiting component.",
    )

    report["adjudication"] = adjudication

    # Explicit conclusion.
    report["decision_gate"] = {
        "do_not_regenerate_h5": True,
        "do_not_change_split": True,
        "do_not_use_test_for_model_selection": True,
        "do_not_launch_large_architecture_grid_yet": True,
        "must_audit_internal_structure_cardinality": bool(semantics["discrepancies"]),
        "next_required_evidence": [
            "training-derived baselines",
            "reader agreement / target stability",
            "cheap image-feature probe",
            "runtime audit of internal_structure K",
            "frozen pretrained-encoder probe if feasible",
            "only then a matched objective comparison",
        ],
    }

    write_json(out_dir / "one_stop_diagnostic.json", report)

    # Human-readable summary.
    lines = [
        "LUNGINSIGHT v2 — ONE-STOP DIAGNOSTIC",
        "=" * 60,
        f"H5: {h5_path}",
        f"Split: {split_path}",
        f"Train/Val/Test: {len(split['train'])}/{len(split['val'])}/{len(split['test'])}",
        "",
        "DECISIVE FINDINGS",
    ]
    for h in adjudication:
        lines.append(f"- {h['status']}: {h['hypothesis']}")
        for e in h["evidence"][:3]:
            lines.append(f"    {e}")
    lines += [
        "",
        "TARGET CARDINALITIES",
        json.dumps({
            "authoritative": AUTHORITATIVE_CARDINALITY,
            "current_code_assumption": CURRENT_CODE_CARDINALITY,
        }, indent=2),
        "",
        "FILES WRITTEN",
        "  one_stop_diagnostic.json",
        "  target_consistency.csv",
        "  category_counts.csv",
        "  reader_count_target_stability.csv",
        "  baselines.csv",
        "  handcrafted_probe.csv (if sklearn available)",
        "  handcrafted_train_features.npy / handcrafted_val_features.npy (if probe ran)",
        "",
        "DECISION GATE",
        json.dumps(report["decision_gate"], indent=2),
    ]
    (out_dir / "ONE_STOP_SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")

    log("\n" + "\n".join(lines))
    log(f"\nComplete. Main report: {out_dir / 'one_stop_diagnostic.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True, help="Path to fixed lidc_patches.h5")
    p.add_argument("--split-file", required=True, help="Path to canonical split.json")
    p.add_argument("--out-dir", required=True, help="Fresh diagnostic output directory")
    p.add_argument("--project-root", default=None, help="LungInsight_v2 root; inferred if omitted")
    p.add_argument("--pretrained-encoder", default=None, help="Optional pretrained_encoder.pth for read-only checkpoint audit")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--patch-scan-batch", type=int, default=32)
    p.add_argument(
        "--max-feature-samples",
        type=int,
        default=0,
        help="Optional cap for handcrafted probe. 0 = all train/val nodules.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    raise SystemExit(run(args))
