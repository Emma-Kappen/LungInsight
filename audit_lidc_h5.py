#!/usr/bin/env python3
"""
LungInsight LIDC HDF5 signal/consistency audit.

Purpose:
    Audit the prepared HDF5 before another expensive Model-3 training run.

Checks:
    1. HDF5 schema and metadata.
    2. Patch value distribution and normalization regime.
    3. Patch-wise variance / near-constant patch fraction.
    4. Center-vs-shell intensity contrast.
    5. Patient/nodule/series identity counts.
    6. Reader-label validity and per-attribute distributions.
    7. Simple correlations between patch statistics and each attribute.
    8. Exact duplicate patch detection using SHA-1 hashes.
    9. Label means by patient and rough target variation.

Malignancy is reported only as metadata context and is never used as a
neural-network target or signal-selection criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr


ATTRIBUTE_NAMES = [
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
]

NUM_CLASSES = {
    "subtlety": 5,
    "internalStructure": 5,
    "calcification": 6,
    "sphericity": 5,
    "margin": 5,
    "lobulation": 5,
    "spiculation": 5,
    "texture": 5,
}

# Raw reader_labels feature order:
# calcification, internalStructure, lobulation, malignancy,
# margin, sphericity, spiculation, subtlety, texture
READER_TO_MODEL = [7, 1, 0, 5, 4, 2, 6, 8]


def decode_attr(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        s = value
        try:
            return json.loads(s)
        except Exception:
            return s
    return value


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return (
        float(pearsonr(x, y).statistic),
        float(spearmanr(x, y).statistic),
    )


def percentile_string(x):
    q = np.percentile(x, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return " ".join(f"{v:.5g}" for v in q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--hash-check", action="store_true")
    args = ap.parse_args()

    path = Path(args.h5)

    print("=" * 88)
    print("LIDC HDF5 SIGNAL / CONSISTENCY AUDIT")
    print("=" * 88)
    print("HDF5:", path)

    with h5py.File(path, "r") as h5:
        patches_ds = h5["patches"]
        reader_labels = np.asarray(h5["reader_labels"], dtype=np.float32)
        reader_mask = np.asarray(h5["reader_mask"], dtype=bool)
        patient_id = np.asarray(h5["patient_id"])
        nodule_id = np.asarray(h5["nodule_id"])
        series_uid = np.asarray(h5["series_instance_uid"])
        study_uid = np.asarray(h5["study_instance_uid"])
        attribute_mean = np.asarray(h5["attribute_mean"], dtype=np.float32)

        n = len(patches_ds)

        print()
        print("SCHEMA")
        print("-" * 88)
        print("samples:", n)
        print("patch shape:", patches_ds.shape)
        print("patch dtype:", patches_ds.dtype)
        print("reader_labels:", reader_labels.shape)
        print("reader_mask:", reader_mask.shape)
        print("attribute_mean:", attribute_mean.shape)

        print()
        print("METADATA")
        print("-" * 88)
        for key in [
            "source_dataset",
            "coordinate_order",
            "source_coordinate_order",
            "spacing_order",
            "source_spacing_order",
            "patch_mm",
            "patch_size",
            "hu_min",
            "hu_max",
            "reader_feature_names",
            "attribute_names",
        ]:
            if key in h5.attrs:
                print(f"{key}: {decode_attr(h5.attrs[key])}")

        # ------------------------------------------------------------------
        # Patch statistics
        # ------------------------------------------------------------------
        means = np.empty(n, dtype=np.float32)
        stds = np.empty(n, dtype=np.float32)
        mins = np.empty(n, dtype=np.float32)
        maxs = np.empty(n, dtype=np.float32)
        center_means = np.empty(n, dtype=np.float32)
        shell_means = np.empty(n, dtype=np.float32)

        global_min = np.inf
        global_max = -np.inf
        total_sum = 0.0
        total_sq = 0.0
        total_count = 0

        # Stable hash detection without storing all patch contents.
        hashes = [] if args.hash_check else None

        print()
        print("PATCH STATISTICS")
        print("-" * 88)

        for start in range(0, n, args.chunk_size):
            end = min(start + args.chunk_size, n)
            x = np.asarray(patches_ds[start:end], dtype=np.float32)

            means[start:end] = x.reshape(len(x), -1).mean(axis=1)
            stds[start:end] = x.reshape(len(x), -1).std(axis=1)
            mins[start:end] = x.reshape(len(x), -1).min(axis=1)
            maxs[start:end] = x.reshape(len(x), -1).max(axis=1)

            center = x[:, 28:36, 28:36, 28:36]
            center_means[start:end] = center.reshape(len(x), -1).mean(axis=1)

            shell = x.copy()
            shell[:, 27:37, 27:37, 27:37] = np.nan
            shell_means[start:end] = np.nanmean(
                shell.reshape(len(x), -1),
                axis=1,
            )

            finite = np.isfinite(x)
            vals = x[finite]
            if vals.size:
                global_min = min(global_min, float(vals.min()))
                global_max = max(global_max, float(vals.max()))
                total_sum += float(vals.sum(dtype=np.float64))
                total_sq += float(np.square(vals, dtype=np.float64).sum())
                total_count += int(vals.size)

            if hashes is not None:
                for patch in x:
                    hashes.append(hashlib.sha1(
                        np.ascontiguousarray(patch).tobytes()
                    ).hexdigest())

        global_mean = total_sum / max(total_count, 1)
        global_var = total_sq / max(total_count, 1) - global_mean**2
        global_std = np.sqrt(max(global_var, 0.0))

        print("global min/max:", global_min, global_max)
        print("global mean/std:", global_mean, global_std)
        print("per-patch mean percentiles:", percentile_string(means))
        print("per-patch std percentiles:", percentile_string(stds))
        print("per-patch min percentiles:", percentile_string(mins))
        print("per-patch max percentiles:", percentile_string(maxs))
        print("center 8^3 mean percentiles:", percentile_string(center_means))
        print("shell mean percentiles:", percentile_string(shell_means))
        print(
            "center-minus-shell percentiles:",
            percentile_string(center_means - shell_means),
        )
        print(
            "near-constant patches (std < 1e-5):",
            int(np.sum(stds < 1e-5)),
            f"/ {n}",
        )

        # ------------------------------------------------------------------
        # Identity checks
        # ------------------------------------------------------------------
        def unique_count(arr):
            values = [str(x) for x in arr]
            return len(set(values))

        print()
        print("IDENTITY / DUPLICATION")
        print("-" * 88)
        print("unique patients:", unique_count(patient_id))
        print("unique nodule IDs:", unique_count(nodule_id))
        print("unique series UIDs:", unique_count(series_uid))
        print("unique study UIDs:", unique_count(study_uid))

        pair_counts = Counter(
            (str(p), str(nid))
            for p, nid in zip(patient_id, nodule_id)
        )
        repeated_pairs = [
            (pair, count)
            for pair, count in pair_counts.items()
            if count > 1
        ]
        print("repeated (patient_id, nodule_id) pairs:", len(repeated_pairs))
        if repeated_pairs[:10]:
            print("examples:", repeated_pairs[:10])

        if hashes is not None:
            hash_counts = Counter(hashes)
            duplicate_hashes = [
                (h, c) for h, c in hash_counts.items() if c > 1
            ]
            print("exact duplicate patches:", sum(c - 1 for _, c in duplicate_hashes))
            print("duplicate hash groups:", len(duplicate_hashes))
            if duplicate_hashes[:10]:
                print("duplicate examples:", duplicate_hashes[:10])

        # ------------------------------------------------------------------
        # Reader-label audit in model order.
        # ------------------------------------------------------------------
        labels_model = reader_labels[:, :, READER_TO_MODEL].transpose(0, 2, 1)
        masks_model = reader_mask[:, None, :]

        print()
        print("READER-LABEL DISTRIBUTIONS")
        print("-" * 88)

        target_means = {}
        target_majority = {}

        for a, attr in enumerate(ATTRIBUTE_NAMES):
            y = labels_model[:, a, :]
            valid = (
                masks_model[:, 0, :]
                & np.isfinite(y)
                & (y >= 1)
                & (y <= NUM_CLASSES[attr])
            )

            counts = np.bincount(
                y[valid].astype(np.int64),
                minlength=NUM_CLASSES[attr] + 1,
            )[1:]

            sample_valid_count = valid.sum(axis=1)
            y_sum = np.where(valid, y, 0.0).sum(axis=1)
            sample_valid = sample_valid_count > 0

            sample_mean = (
                y_sum / np.maximum(sample_valid_count, 1)
            )
            target_means[attr] = sample_mean

            # Majority reader rating, where at least one rating exists.
            majority = np.full(n, np.nan, dtype=np.float32)
            for i in np.where(sample_valid)[0]:
                vals = y[i][valid[i]].astype(np.int64)
                binc = np.bincount(
                    vals,
                    minlength=NUM_CLASSES[attr] + 1,
                )
                majority[i] = float(np.argmax(binc[1:]) + 1)
            target_majority[attr] = majority

            print(
                f"{attr:<20} "
                f"valid ratings={int(valid.sum()):>6} "
                f"valid samples={int(sample_valid.sum()):>5} "
                f"mean={np.nanmean(sample_mean):.4f} "
                f"std={np.nanstd(sample_mean):.4f} "
                f"counts={counts.tolist()}"
            )

        # ------------------------------------------------------------------
        # Simple signal correlations.
        # ------------------------------------------------------------------
        patch_features = {
            "patch_mean": means,
            "patch_std": stds,
            "patch_min": mins,
            "patch_max": maxs,
            "center_mean": center_means,
            "shell_mean": shell_means,
            "center_minus_shell": center_means - shell_means,
        }

        print()
        print("PATCH-SUMMARY ↔ ATTRIBUTE-MEAN CORRELATIONS")
        print("-" * 88)
        print(
            f"{'ATTRIBUTE':<20}"
            f"{'BEST_FEATURE':<24}"
            f"{'PEARSON':>10}"
            f"{'SPEARMAN':>10}"
        )
        print("-" * 88)

        for attr in ATTRIBUTE_NAMES:
            y = target_means[attr]

            best = None
            for feature_name, feature in patch_features.items():
                p, s = safe_corr(feature, y)
                score = abs(s) if np.isfinite(s) else -1.0
                if best is None or score > best[0]:
                    best = (score, feature_name, p, s)

            _, feature_name, p, s = best
            print(
                f"{attr:<20}"
                f"{feature_name:<24}"
                f"{p:>10.4f}"
                f"{s:>10.4f}"
            )

        # ------------------------------------------------------------------
        # Patient-level label sanity.
        # ------------------------------------------------------------------
        print()
        print("PATIENT-LEVEL TARGET SANITY")
        print("-" * 88)

        patient_strings = np.asarray([str(x) for x in patient_id])
        unique_patients = np.unique(patient_strings)

        for attr in ATTRIBUTE_NAMES:
            vals = []
            for pid in unique_patients:
                idx = patient_strings == pid
                v = target_means[attr][idx]
                if np.isfinite(v).any():
                    vals.append(float(np.nanmean(v)))
            vals = np.asarray(vals)

            print(
                f"{attr:<20} "
                f"patient-mean range={np.nanmin(vals):.4f}..{np.nanmax(vals):.4f} "
                f"std={np.nanstd(vals):.4f}"
            )

        # ------------------------------------------------------------------
        # Centering sanity.
        # ------------------------------------------------------------------
        contrast = center_means - shell_means
        print()
        print("CENTERING SANITY")
        print("-" * 88)
        print(
            "fraction center > shell by >0.05:",
            float(np.mean(contrast > 0.05)),
        )
        print(
            "fraction center > shell by >0.10:",
            float(np.mean(contrast > 0.10)),
        )
        print(
            "fraction center < shell by <-0.05:",
            float(np.mean(contrast < -0.05)),
        )

        print()
        print("AUDIT COMPLETE")
        print("=" * 88)


if __name__ == "__main__":
    main()
