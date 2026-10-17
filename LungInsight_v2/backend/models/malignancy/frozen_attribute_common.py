#!/usr/bin/env python3
"""Shared, inference-only helpers for LungInsight's frozen spatial-hybrid rankers.

This module intentionally contains no optimizer, training loop, gradient-enabled
path, or checkpoint-writing function for Stage 03 models.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# This file is expected at <repo>/LungInsight_v2/backend/models/malignancy/.
V2_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = V2_ROOT.parent
MODEL_ROOT = V2_ROOT / "backend" / "models" / "nodule_ranker_3d_distributional_v1"
SPATIAL_ROOT = MODEL_ROOT / "spatial_v1"
DEFAULT_CHECKPOINT_ROOT = SPATIAL_ROOT / "checkpoints" / "hybrid"
DEFAULT_SPLIT = MODEL_ROOT / "split.json"
DEFAULT_H5 = PROJECT_ROOT / "datasets" / "LIDC" / "lidc_patches.h5"

FEATURE_NAMES = (
    "subtlety",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
)
CHECKPOINT_GROUPS = {
    "lobulation_spiculation": ("lobulation", "spiculation"),
    "subtlety": ("subtlety",),
    "sphericity": ("sphericity",),
    "margin": ("margin",),
    "texture": ("texture",),
}
EXPECTED_SPLIT_COUNTS = {"train": 1859, "val": 437, "test": 355}
EXPECTED_SPLIT_HASHES = {
    "train": "1814147c1f1f0ccfbda73b56c5dbb392c14412b72c6b0fe316ee7a81957ad922",
    "val": "0831f0b8e0ba4b54de794c914677e783d21197bedc5c6f93c695eb07011051f7",
    "test": "e3ebe6f752237c58240e0888492679295ab068d6929cd75be1df83b821e4b56c",
}
H5_MALIGNANCY_READER_COLUMN = 3  # See train_objective.py READER_NAMES order.
EXPECTED_PATCH_SHAPE = (64, 64, 64)


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a file incrementally without loading it into memory."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def decode_h5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def read_h5_indices(dataset: Any, indices: np.ndarray) -> np.ndarray:
    """Read arbitrary HDF5 indices while preserving requested row order."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return np.empty((0,) + tuple(dataset.shape[1:]), dtype=dataset.dtype)
    order = np.argsort(indices, kind="mergesort")
    sorted_indices = indices[order]
    sorted_values = np.asarray(dataset[sorted_indices])
    restore = np.argsort(order, kind="mergesort")
    return sorted_values[restore]


def choose_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch reports CUDA unavailable.")
    return device


def _import_model_code():
    spatial_path = str(SPATIAL_ROOT)
    if spatial_path not in sys.path:
        sys.path.insert(0, spatial_path)
    try:
        from model_objective import (  # type: ignore[import-not-found]
            MODEL_TARGETS,
            ORDERED_TARGETS,
            RATING_MAX,
            SpatialObjectiveModel,
            preprocess_hu,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not import the architecture-compatible model_objective.py at "
            f"{SPATIAL_ROOT}. Keep the supplied spatial_v1 source directory in place."
        ) from exc
    return MODEL_TARGETS, ORDERED_TARGETS, RATING_MAX, SpatialObjectiveModel, preprocess_hu


def load_frozen_models(
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
    device: torch.device | None = None,
    *,
    validate_provenance: bool = True,
) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, Any]]]:
    """Load all five supplied checkpoints strictly, without changing weights."""
    _, _, _, SpatialObjectiveModel, _ = _import_model_code()
    checkpoint_root = Path(checkpoint_root).resolve()
    device = device or torch.device("cpu")
    models: dict[str, torch.nn.Module] = {}
    metadata: dict[str, dict[str, Any]] = {}
    common_arch: dict[str, Any] | None = None

    for group, _attrs in CHECKPOINT_GROUPS.items():
        path = checkpoint_root / group / "best.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen Stage 03 checkpoint: {path}")
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Could not load checkpoint {path}: {exc}") from exc
        if not isinstance(ckpt, dict) or "architecture" not in ckpt or "model_state_dict" not in ckpt:
            raise ValueError(f"Checkpoint lacks architecture/model_state_dict metadata: {path}")
        arch = ckpt["architecture"]
        # Validate every architecture and preprocessing field against the source
        # implementation, not merely tensor shapes or the objective label.
        spatial_path = str(SPATIAL_ROOT)
        if spatial_path not in sys.path:
            sys.path.insert(0, spatial_path)
        try:
            from model_objective import architecture_config  # type: ignore[import-not-found]
            expected_arch = architecture_config(
                arch.get("objective", ""),
                int(arch.get("attention_dim", -1)),
                int(arch.get("head_hidden", -1)),
                float(arch.get("dropout", float("nan"))),
            )
        except Exception as exc:
            raise RuntimeError(f"Could not validate architecture metadata from {path}: {exc}") from exc
        if arch != expected_arch:
            raise ValueError(
                f"Incompatible checkpoint architecture/preprocessing metadata in {path}. "
                "The checkpoint must exactly match model_objective.architecture_config; do not guess or resize input."
            )
        if ckpt.get("test_used_for_selection") is True:
            raise ValueError(f"Checkpoint metadata says test data was used for selection: {path}")
        if arch.get("objective") != "ordinal_reader":
            raise ValueError(f"{path}: expected objective='ordinal_reader', got {arch.get('objective')!r}")
        if tuple(arch.get("input_shape", [])) != EXPECTED_PATCH_SHAPE:
            raise ValueError(f"{path}: expected input shape {EXPECTED_PATCH_SHAPE}, got {arch.get('input_shape')}")
        if validate_provenance:
            actual_hashes = ckpt.get("split_hashes")
            if actual_hashes != EXPECTED_SPLIT_HASHES:
                raise ValueError(
                    f"Checkpoint provenance mismatch for {path}. Expected canonical split hashes "
                    f"{EXPECTED_SPLIT_HASHES}, found {actual_hashes}. Do not retrain or bypass this check."
                )
            recorded_group = ckpt.get("attribute_group")
            if recorded_group != group:
                raise ValueError(f"{path}: expected checkpoint group {group!r}, got {recorded_group!r}")
        if common_arch is None:
            common_arch = arch
        elif arch != common_arch:
            raise ValueError("The five hybrid checkpoints do not share an identical architecture/preprocessing config.")

        model = SpatialObjectiveModel(
            arch["objective"],
            int(arch["attention_dim"]),
            int(arch["head_hidden"]),
            float(arch["dropout"]),
        )
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        except Exception as exc:
            raise ValueError(f"Architecture/state-dict mismatch for {path}: {exc}") from exc
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[group] = model
        metadata[group] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "attribute_group": ckpt.get("attribute_group"),
            "epoch": ckpt.get("epoch"),
            "best_metric_name": ckpt.get("best_metric_name"),
            "architecture": arch,
            "split_hashes": ckpt.get("split_hashes"),
            "test_used_for_selection": ckpt.get("test_used_for_selection"),
        }
        del ckpt

    return models, metadata


def forward_with_expected_ratings_and_attention(
    model: torch.nn.Module,
    x_normalized: torch.Tensor,
    *,
    model_targets: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute model expected ratings and task-to-voxel attention maps.

    Input is already normalized exactly as model_objective.preprocess_hu specifies,
    with shape [B,1,64,64,64]. Attention maps are returned as [B,T,8,8,8].
    """
    MODEL_TARGETS, _, _, _, _ = _import_model_code()
    if model_targets is None:
        model_targets = MODEL_TARGETS
    # Same forward path as SpatialObjectiveModel.forward, with its internal
    # task-query softmax retained for explainability; no parameters are altered.
    feature_map = model.feature_map(x_normalized)  # type: ignore[attr-defined]
    attention_module = model.attention  # type: ignore[attr-defined]
    task_features = attention_module(feature_map)
    outputs = tuple(head(task_features[:, i]) for i, head in enumerate(model.heads))  # type: ignore[attr-defined]
    expected = model.expected_ratings(outputs).float()  # type: ignore[attr-defined]

    b, c, z, y, x = feature_map.shape
    tokens = attention_module.norm(feature_map.flatten(2).transpose(1, 2))
    coords = attention_module.coordinates(z, y, x, feature_map.device, feature_map.dtype)
    keys = attention_module.key(tokens) + attention_module.coord(coords).unsqueeze(0)
    queries = attention_module.query.unsqueeze(0).expand(b, -1, -1)
    scores = torch.einsum("bad,btd->bat", queries, keys) * attention_module.scale
    weights = torch.softmax(scores.float(), dim=-1).reshape(b, len(model_targets), z, y, x)
    return (
        expected.detach().cpu().numpy().astype(np.float32, copy=False),
        weights.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def stage03_prediction_map(
    patch_array: np.ndarray,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    *,
    collect_attention: bool = False,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Predict the six frozen hybrid attributes for one raw-HU patch."""
    predictions, _, attention = stage03_prediction_details(
        patch_array, models, device, collect_attention=collect_attention
    )
    return predictions, attention


def stage03_prediction_details(
    patch_array: np.ndarray,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    *,
    collect_attention: bool = False,
) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """Return expected ratings, ordinal output distributions, and optional attention maps."""
    MODEL_TARGETS, _, _, _, preprocess_hu = _import_model_code()
    if tuple(patch_array.shape) != EXPECTED_PATCH_SHAPE:
        raise ValueError(f"Expected a raw-HU patch shaped {EXPECTED_PATCH_SHAPE}, got {patch_array.shape}")
    patch = np.asarray(patch_array, dtype=np.float32)
    if not np.isfinite(patch).all():
        raise ValueError("Patch contains NaN or infinite HU values")
    x = preprocess_hu(torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0).unsqueeze(0)).to(device)
    attr_predictions: dict[str, float] = {}
    attr_distributions: dict[str, dict[str, Any]] = {}
    attr_attention: dict[str, np.ndarray] = {}
    attr_index = {name: MODEL_TARGETS.index(name) for name in FEATURE_NAMES}
    with torch.inference_mode():
        for group, attrs in CHECKPOINT_GROUPS.items():
            model = models[group]
            outputs = model(x)
            expected = model.expected_ratings(outputs)
            if collect_attention:
                _, weights = forward_with_expected_ratings_and_attention(model, x)
                for name in attrs:
                    j = attr_index[name]
                    attr_attention[name] = weights[0, j]
            for name in attrs:
                j = attr_index[name]
                attr_predictions[name] = float(expected[0, j].detach().cpu().item())
                probabilities = model.ordered_probabilities(outputs[j])[0].float()
                probability_values = probabilities.detach().cpu().numpy().astype(np.float64)
                probability_values /= probability_values.sum()
                ratings = np.arange(1, len(probability_values) + 1, dtype=np.float64)
                entropy = -float(np.sum(probability_values * np.log2(np.maximum(probability_values, 1e-12))))
                mean = float(np.sum(probability_values * ratings))
                variance = float(np.sum(probability_values * np.square(ratings - mean)))
                attr_distributions[name] = {
                    "rating_values": ratings.astype(int).tolist(),
                    "probabilities": probability_values.astype(float).tolist(),
                    "entropy_bits": entropy,
                    "rating_standard_deviation": float(np.sqrt(max(variance, 0.0))),
                    "calibrated_confidence": None,
                    "semantics": "ordinal_model_output_distribution_not_calibrated_confidence",
                }
    missing = [name for name in FEATURE_NAMES if name not in attr_predictions]
    if missing:
        raise RuntimeError(f"Hybrid inference omitted required feature(s): {missing}")
    if not all(np.isfinite(attr_predictions[n]) for n in FEATURE_NAMES):
        raise RuntimeError("Hybrid inference produced non-finite attribute rating(s)")
    for name, maximum in {
        "subtlety": 5, "sphericity": 5, "margin": 5,
        "lobulation": 5, "spiculation": 5, "texture": 5,
    }.items():
        val = attr_predictions[name]
        if not (1.0 - 1e-4 <= val <= maximum + 1e-4):
            raise RuntimeError(f"{name} expected rating is outside [1,{maximum}]: {val}")
    return (
        {name: attr_predictions[name] for name in FEATURE_NAMES},
        {name: attr_distributions[name] for name in FEATURE_NAMES},
        {name: attr_attention[name] for name in FEATURE_NAMES if name in attr_attention},
    )


def validate_and_load_split(split_path: str | Path, n_samples: int, patient_ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Validate the known canonical split without modifying/rebuilding it."""
    split_path = Path(split_path).resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"Canonical split file not found: {split_path}")
    raw = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("split.json must be a JSON object")
    split: dict[str, np.ndarray] = {}
    patient_sets: dict[str, set[str]] = {}
    for name in ("train", "val", "test"):
        entry = raw.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"split.json: missing {name!r} object")
        idx_list = entry.get("sample_indices", entry.get("nodule_indices", entry.get("indices")))
        if not isinstance(idx_list, list) or not all(isinstance(v, (int, np.integer)) for v in idx_list):
            raise ValueError(f"split.json: {name}.sample_indices must be a list of integers")
        idx = np.asarray(idx_list, dtype=np.int64)
        if len(idx) != EXPECTED_SPLIT_COUNTS[name]:
            raise ValueError(f"Canonical {name} count mismatch: expected {EXPECTED_SPLIT_COUNTS[name]}, got {len(idx)}")
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"Duplicate H5 sample indices in {name} split")
        if np.any(idx < 0) or np.any(idx >= int(n_samples)):
            raise ValueError(f"Out-of-range H5 sample index in {name} split for N={n_samples}")
        got_hash = sha256_indices(idx)
        if got_hash != EXPECTED_SPLIT_HASHES[name]:
            raise ValueError(f"Canonical {name} split hash mismatch: {got_hash}")
        listed_patients = entry.get("patients", entry.get("patient_ids"))
        actual_patients = {decode_h5_string(patient_ids[i]) for i in idx}
        if listed_patients is not None and {str(p) for p in listed_patients} != actual_patients:
            raise ValueError(f"The listed patient set for {name} does not agree with H5 patient_id values")
        split[name] = idx
        patient_sets[name] = actual_patients
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patient_sets[a] & patient_sets[b]
        if overlap:
            raise ValueError(f"Patient leakage between {a} and {b}: {sorted(overlap)[:10]}")
        if set(split[a].tolist()) & set(split[b].tolist()):
            raise ValueError(f"Sample-index leakage between {a} and {b}")
    if sum(len(v) for v in split.values()) != n_samples:
        raise ValueError(
            f"Canonical split counts sum to {sum(len(v) for v in split.values())}, but H5 has {n_samples} rows. "
            "Do not change the split to bypass this mismatch."
        )
    return split, raw


def find_optional_h5_dataset(h5: Any, names: tuple[str, ...], n: int):
    """Find an optional one-row-per-patch metadata dataset by likely basename."""
    dataset_paths: list[str] = []
    def visitor(name: str, obj: Any) -> None:
        if hasattr(obj, "shape") and len(obj.shape) >= 1 and int(obj.shape[0]) == n:
            dataset_paths.append(name)
    h5.visititems(visitor)
    by_basename: dict[str, list[str]] = {}
    for path in dataset_paths:
        by_basename.setdefault(path.split("/")[-1].lower(), []).append(path)
    for candidate in names:
        # Prefer exact root-level dataset, then any unique basename match.
        if candidate in h5 and hasattr(h5[candidate], "shape") and len(h5[candidate].shape) >= 1 and int(h5[candidate].shape[0]) == n:
            return h5[candidate], candidate
        paths = by_basename.get(candidate.lower(), [])
        if len(paths) == 1:
            return h5[paths[0]], paths[0]
    return None, None
