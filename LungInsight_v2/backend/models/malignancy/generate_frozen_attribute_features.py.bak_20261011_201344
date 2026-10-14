#!/usr/bin/env python3
"""Generate XGBoost input features using the existing Stage 03 checkpoints only.

The H5 patches are run through five frozen, evaluation-only hybrid rankers. No
Stage 03 optimizer/trainer is imported or invoked. Outputs preserve H5 index,
patient identity, optional stored patch/nodule IDs, and each valid reader-level
malignancy rating/mask.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from frozen_attribute_common import (  # noqa: E402
    CHECKPOINT_GROUPS,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_H5,
    DEFAULT_SPLIT,
    EXPECTED_PATCH_SHAPE,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_HASHES,
    FEATURE_NAMES,
    H5_MALIGNANCY_READER_COLUMN,
    choose_device,
    decode_h5_string,
    find_optional_h5_dataset,
    load_frozen_models,
    read_h5_indices,
    sha256_file,
    validate_and_load_split,
)

DEFAULT_OUTPUT_DIR = HERE / "frozen_features"


def json_safe_id_array(values: np.ndarray | None, *, n: int, fallback: str = "") -> np.ndarray:
    if values is None:
        return np.asarray([fallback] * n, dtype="U1")
    result = [decode_h5_string(v) for v in values]
    width = max(1, max((len(v) for v in result), default=1))
    return np.asarray(result, dtype=f"U{width}")


def unique_width(values: list[str]) -> np.ndarray:
    width = max(1, max((len(v) for v in values), default=1))
    return np.asarray(values, dtype=f"U{width}")


def get_ids(h5: h5py.File, n: int, split_indices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    patient_data = np.asarray(h5["patient_id"][:])
    patient_ids_all = np.asarray([decode_h5_string(v) for v in patient_data], dtype=object)
    optional = {
        "patch_id": ("patch_id", "patch_ids"),
        "nodule_id": ("nodule_id", "nodule_ids", "xml_nodule_id"),
        "candidate_id": ("candidate_id", "candidate_ids"),
        "series_instance_uid": ("series_instance_uid", "series_uid"),
        "study_instance_uid": ("study_instance_uid", "study_uid"),
    }
    found: dict[str, tuple[np.ndarray | None, str | None]] = {}
    for key, names in optional.items():
        ds, actual_name = find_optional_h5_dataset(h5, names, n)
        values = np.asarray(ds[:]) if ds is not None else None
        found[key] = (values, actual_name)

    identities: dict[str, np.ndarray] = {}
    # Identity audit over the whole H5. Use original IDs when present, but never
    # invent a nodule ID: sample_index remains the canonical fallback identity.
    for split_name, indices in split_indices.items():
        identity_keys: list[str] = []
        for idx in indices:
            i = int(idx)
            patient = str(patient_ids_all[i])
            study_data = found["study_instance_uid"][0]
            series_data = found["series_instance_uid"][0]
            nodule_data = found["nodule_id"][0]
            patch_data = found["patch_id"][0]
            candidate_data = found["candidate_id"][0]
            pieces = [patient]
            if study_data is not None and decode_h5_string(study_data[i]).strip():
                pieces.append("study=" + decode_h5_string(study_data[i]).strip())
            if series_data is not None and decode_h5_string(series_data[i]).strip():
                pieces.append("series=" + decode_h5_string(series_data[i]).strip())
            if nodule_data is not None and decode_h5_string(nodule_data[i]).strip():
                pieces.append("nodule=" + decode_h5_string(nodule_data[i]).strip())
            elif patch_data is not None and decode_h5_string(patch_data[i]).strip():
                pieces.append("patch=" + decode_h5_string(patch_data[i]).strip())
            elif candidate_data is not None and decode_h5_string(candidate_data[i]).strip():
                pieces.append("candidate=" + decode_h5_string(candidate_data[i]).strip())
            else:
                pieces.append(f"h5_index={i}")
            identity_keys.append("|".join(pieces))
        if len(identity_keys) != len(set(identity_keys)):
            duplicates = []
            seen: set[str] = set()
            for item in identity_keys:
                if item in seen:
                    duplicates.append(item)
                seen.add(item)
            raise ValueError(f"Duplicate nodule identities in {split_name}: {duplicates[:10]}")

        identities[f"{split_name}__patient_id"] = unique_width([str(patient_ids_all[int(i)]) for i in indices])
        identities[f"{split_name}__sample_index"] = np.asarray(indices, dtype=np.int64)
        identities[f"{split_name}__identity_key"] = unique_width(identity_keys)
        for field, (values, _source) in found.items():
            if values is None:
                identities[f"{split_name}__{field}"] = np.asarray([""] * len(indices), dtype="U1")
            else:
                identities[f"{split_name}__{field}"] = unique_width([
                    decode_h5_string(values[int(i)]) for i in indices
                ])
    return identities


def extract_labels(h5: h5py.File, indices: np.ndarray) -> dict[str, np.ndarray]:
    labels_ds = h5["reader_labels"]
    mask_ds = h5["reader_mask"]
    labels_shape = tuple(labels_ds.shape)
    if len(labels_shape) != 3 or labels_shape[2] <= H5_MALIGNANCY_READER_COLUMN:
        raise ValueError(
            "H5 reader_labels must have shape [N, reader_slots, >=9] in the source READER_NAMES order; "
            f"got {labels_shape}. The malignancy source column is {H5_MALIGNANCY_READER_COLUMN}."
        )
    labels = np.asarray(read_h5_indices(labels_ds, indices), dtype=np.float32)
    reader_mask = np.asarray(read_h5_indices(mask_ds, indices), dtype=bool)
    if reader_mask.shape != labels.shape[:2]:
        raise ValueError(f"reader_mask shape {reader_mask.shape} is incompatible with reader_labels {labels.shape}")
    ratings = labels[:, :, H5_MALIGNANCY_READER_COLUMN].copy()
    valid = reader_mask & np.isfinite(ratings) & (ratings >= 1.0) & (ratings <= 5.0)
    ratings[~valid] = np.nan
    count = valid.sum(axis=1).astype(np.int32)
    mean = np.full(len(indices), np.nan, dtype=np.float32)
    median = np.full(len(indices), np.nan, dtype=np.float32)
    for i in range(len(indices)):
        values = ratings[i, valid[i]]
        if len(values):
            mean[i] = float(np.mean(values, dtype=np.float64))
            median[i] = float(np.median(values))
    return {
        "malignancy_reader_ratings": ratings,
        "malignancy_reader_mask": valid.astype(bool),
        "malignancy_mean": mean,
        "malignancy_median": median,
        "malignancy_reader_count": count,
    }


def infer_split(
    h5: h5py.File,
    indices: np.ndarray,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    batch_size: int,
    split_name: str,
) -> np.ndarray:
    from frozen_attribute_common import _import_model_code  # local import keeps module interface narrow
    MODEL_TARGETS, _, _, _, preprocess_hu = _import_model_code()
    feature_index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    model_index = {name: MODEL_TARGETS.index(name) for name in FEATURE_NAMES}
    out = np.full((len(indices), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    total_batches = (len(indices) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_number, start in enumerate(range(0, len(indices), batch_size), start=1):
            batch_indices = indices[start:start + batch_size]
            patches = np.asarray(read_h5_indices(h5["patches"], batch_indices), dtype=np.float32)
            expected_shape = (len(batch_indices),) + EXPECTED_PATCH_SHAPE
            if tuple(patches.shape) != expected_shape:
                raise ValueError(f"H5 patches batch has shape {patches.shape}, expected {expected_shape}")
            if not np.isfinite(patches).all():
                raise ValueError(f"Non-finite HU voxels found in {split_name} indices {batch_indices[:10].tolist()}")
            x = preprocess_hu(torch.from_numpy(np.ascontiguousarray(patches))).unsqueeze(1).to(device)
            for group, attrs in CHECKPOINT_GROUPS.items():
                model = models[group]
                outputs = model(x)
                expected = model.expected_ratings(outputs).float().cpu().numpy()
                for name in attrs:
                    out[start:start + len(batch_indices), feature_index[name]] = expected[:, model_index[name]]
            print(f"[{split_name}] inference batch {batch_number}/{total_batches}", flush=True)
    if not np.isfinite(out).all():
        bad = np.argwhere(~np.isfinite(out))[:10].tolist()
        raise ValueError(f"Missing/non-finite frozen attribute predictions in {split_name}; sample positions: {bad}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--h5", type=Path, default=DEFAULT_H5)
    p.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--skip-h5-sha256", action="store_true", help="Skip full H5 hashing for faster debugging; manifest marks hash omitted")
    p.add_argument("--allow-noncanonical-checkpoint-provenance", action="store_true", help="Disable checkpoint provenance validation only for deliberate debugging")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    h5_path = a.h5.resolve()
    split_path = a.split_file.resolve()
    output_dir = a.output_dir.resolve()
    for path, label in ((h5_path, "H5 dataset"), (split_path, "canonical split")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if a.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Checking Stage 03 checkpoint provenance and hashes...", flush=True)
    device = choose_device(a.device)
    models, checkpoint_metadata = load_frozen_models(
        a.checkpoint_root, device,
        validate_provenance=not a.allow_noncanonical_checkpoint_provenance,
    )
    h5_hash = None if a.skip_h5_sha256 else sha256_file(h5_path)
    split_hash = sha256_file(split_path)

    with h5py.File(h5_path, "r") as h5:
        required = ("patches", "reader_labels", "reader_mask", "patient_id")
        missing = [key for key in required if key not in h5]
        if missing:
            raise ValueError(f"H5 missing required dataset(s): {missing}")
        n_samples = int(h5["patches"].shape[0])
        if tuple(h5["patches"].shape[1:]) != EXPECTED_PATCH_SHAPE:
            raise ValueError(f"Expected H5 patches shape [N,{EXPECTED_PATCH_SHAPE}], got {h5['patches'].shape}")
        if int(h5["reader_labels"].shape[0]) != n_samples or int(h5["reader_mask"].shape[0]) != n_samples:
            raise ValueError("H5 patch and reader-label row counts disagree")
        patient_ids_all = np.asarray(h5["patient_id"][:])
        split_indices, raw_split = validate_and_load_split(split_path, n_samples, patient_ids_all)
        identities = get_ids(h5, n_samples, split_indices)
        split_results: dict[str, dict[str, Any]] = {}
        for split_name in ("train", "val", "test"):
            indices = split_indices[split_name]
            X = infer_split(h5, indices, models, device, a.batch_size, split_name)
            labels = extract_labels(h5, indices)
            patient_ids = identities[f"{split_name}__patient_id"]
            if len(patient_ids) != len(indices) or np.any(patient_ids == ""):
                raise ValueError(f"Missing patient_id metadata in {split_name}")
            invalid_target_nodules = int((labels["malignancy_reader_count"] == 0).sum())
            if invalid_target_nodules == len(indices):
                raise ValueError(f"No valid reader malignancy targets exist in {split_name}; cannot train/evaluate XGBoost")
            archive_path = output_dir / f"{split_name}.npz"
            np.savez_compressed(
                archive_path,
                X=X.astype(np.float32),
                feature_names=np.asarray(FEATURE_NAMES, dtype="U"),
                sample_index=identities[f"{split_name}__sample_index"],
                identity_key=identities[f"{split_name}__identity_key"],
                patient_id=patient_ids,
                patch_id=identities[f"{split_name}__patch_id"],
                nodule_id=identities[f"{split_name}__nodule_id"],
                candidate_id=identities[f"{split_name}__candidate_id"],
                series_instance_uid=identities[f"{split_name}__series_instance_uid"],
                study_instance_uid=identities[f"{split_name}__study_instance_uid"],
                malignancy_reader_ratings=labels["malignancy_reader_ratings"],
                malignancy_reader_mask=labels["malignancy_reader_mask"],
                malignancy_mean=labels["malignancy_mean"],
                malignancy_median=labels["malignancy_median"],
                malignancy_reader_count=labels["malignancy_reader_count"],
            )
            split_results[split_name] = {
                "archive": str(archive_path),
                "archive_sha256": sha256_file(archive_path),
                "rows": len(indices),
                "feature_shape": list(X.shape),
                "reader_rating_slots": int(labels["malignancy_reader_ratings"].shape[1]),
                "valid_reader_observations": int(labels["malignancy_reader_mask"].sum()),
                "nodules_with_at_least_one_valid_malignancy_rating": int((labels["malignancy_reader_count"] > 0).sum()),
                "nodules_without_valid_malignancy_rating": invalid_target_nodules,
                "patients": int(len(set(patient_ids.tolist()))),
                "sample_index_hash": EXPECTED_SPLIT_HASHES[split_name],
            }
            print(
                f"[{split_name}] wrote {archive_path.name}: {len(indices)} nodules, "
                f"{split_results[split_name]['valid_reader_observations']} valid reader labels, "
                f"missing nodule-level target for {invalid_target_nodules}",
                flush=True,
            )

    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Frozen Stage 03 feature generation for reader-rated LIDC malignancy/suspicion prediction",
        "stage03_training_or_finetuning_performed": False,
        "stage03_mode": "model.eval() + torch.inference_mode() only",
        "stage03_optimizer_or_backward_called": False,
        "route": "spatial_hybrid",
        "features": list(FEATURE_NAMES),
        "feature_semantics": "Expected ordinal reader ratings from each frozen ranker, not calibrated probabilities",
        "unsupported_features_not_fabricated": ["internal_structure", "calcification"],
        "h5": str(h5_path),
        "h5_sha256": h5_hash,
        "h5_sha256_skipped": bool(a.skip_h5_sha256),
        "split_file": str(split_path),
        "split_file_sha256": split_hash,
        "canonical_split_hashes": EXPECTED_SPLIT_HASHES,
        "canonical_split_counts": EXPECTED_SPLIT_COUNTS,
        "checkpoint_root": str(Path(a.checkpoint_root).resolve()),
        "checkpoints": checkpoint_metadata,
        "reader_target": {
            "source_dataset": "reader_labels",
            "source_reader_names_order": ["calcification", "internalStructure", "lobulation", "malignancy", "margin", "sphericity", "spiculation", "subtlety", "texture"],
            "malignancy_column_index": H5_MALIGNANCY_READER_COLUMN,
            "validity_dataset": "reader_mask",
            "valid_rating_range": [1, 5],
            "saved_target_views": ["reader-level ratings + masks", "mean", "median", "valid-reader count"],
            "missing_labels": "Stored as NaN with mask=False; the trainer excludes invalid observations and reports counts.",
        },
        "training_feature_provenance_limitation": (
            "Train split features are generated from fixed rankers that were trained on the canonical train split. "
            "They are in-sample predictions, not out-of-fold predictions, and may be optimistic. Validation and test "
            "features are generated by the same frozen weights on disjoint patients. No ranker is retrained here."
        ),
        "splits": split_results,
    }
    manifest_path = output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
