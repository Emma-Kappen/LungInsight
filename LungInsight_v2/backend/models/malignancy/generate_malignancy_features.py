#!/usr/bin/env python3
"""Generate leak-free hybrid attribute features and reader malignancy targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
SPATIAL_DIR = HERE.parent / "nodule_ranker_3d_distributional_v1" / "spatial_v1"
HYBRID_CHECKPOINT_DIR = SPATIAL_DIR / "checkpoints" / "hybrid"
sys.path.insert(0, str(SPATIAL_DIR))

from model_objective import MODEL_TARGETS, SpatialObjectiveModel, preprocess_hu  # noqa: E402
from train_objective import (  # noqa: E402
    H5PatchDataset,
    build_target_arrays,
    choose_device,
    collate,
    hash_indices,
    parse_and_validate_split,
    read_h5_indices,
    total_loss,
)


ATTRIBUTE_NAMES = ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture")
GROUPS = {
    "lobulation_spiculation": ("lobulation", "spiculation"),
    "subtlety": ("subtlety",),
    "sphericity": ("sphericity",),
    "margin": ("margin",),
    "texture": ("texture",),
}
CHECKPOINT_ARGS = {
    "lobulation_spiculation": "lobulation_spiculation",
    "subtlety": "subtlety",
    "sphericity": "sphericity",
    "margin": "margin",
    "texture": "texture",
}
MALIGNANCY_READER_INDEX = 3


def resolve_checkpoint_path(
    path: str | Path, group: str, checkpoint_dir: Path = HYBRID_CHECKPOINT_DIR
) -> Path:
    """Resolve explicit paths or the repository's canonical hybrid checkpoint path."""
    requested = Path(path).expanduser()
    resolved = requested.resolve()
    if resolved.is_file():
        return resolved

    canonical = (checkpoint_dir / group / "best.pth").resolve()
    if not requested.is_absolute() and canonical.is_file():
        return canonical

    raise FileNotFoundError(
        f"group '{group}': attribute checkpoint not found; requested path={resolved}; "
        f"canonical checkpoint path={canonical}. Pass the existing checkpoint path explicitly."
    )


def build_group_channel_mapping(
    groups: dict[str, tuple[str, ...]] = GROUPS,
    attribute_names: tuple[str, ...] = ATTRIBUTE_NAMES,
    model_targets: list[str] = MODEL_TARGETS,
) -> dict[str, tuple[tuple[str, int, int], ...]]:
    """Map each group's target names to (attribute slot, model output channel)."""
    if len(attribute_names) != len(set(attribute_names)):
        raise ValueError(f"Duplicate malignancy attributes: {attribute_names}")
    if len(model_targets) != len(set(model_targets)):
        raise ValueError(f"Duplicate model output targets: {model_targets}")

    mapping: dict[str, tuple[tuple[str, int, int], ...]] = {}
    assigned_by: dict[str, str] = {}
    for group, targets in groups.items():
        expected_targets = tuple(targets)
        if len(expected_targets) != len(set(expected_targets)):
            raise ValueError(
                f"group '{group}': duplicate target mapping; expected targets={expected_targets}; "
                "actual output dimensions=not available before inference"
            )
        entries = []
        for target in expected_targets:
            if target not in model_targets:
                raise ValueError(
                    f"group '{group}': unknown model target '{target}'; expected targets="
                    f"{expected_targets}; actual output dimensions=not available before inference; "
                    f"valid model targets={tuple(model_targets)}"
                )
            if target not in attribute_names:
                raise ValueError(
                    f"group '{group}': target '{target}' is not a malignancy feature; "
                    f"expected targets={expected_targets}; actual output dimensions="
                    "not available before inference; invalid mapping"
                )
            if target in assigned_by:
                raise ValueError(
                    f"group '{group}': duplicate assignment for '{target}' already mapped by "
                    f"group '{assigned_by[target]}'; expected targets={expected_targets}; "
                    "actual output dimensions=not available before inference"
                )
            assigned_by[target] = group
            entries.append((target, attribute_names.index(target), model_targets.index(target)))
        mapping[group] = tuple(entries)

    missing = tuple(name for name in attribute_names if name not in assigned_by)
    if missing:
        raise ValueError(
            f"missing malignancy attribute mappings={missing}; expected attributes={attribute_names}; "
            "actual output dimensions=not available before inference"
        )
    return mapping


def assemble_attribute_slots(
    group_outputs: dict[str, np.ndarray], sample_count: int,
    mapping: dict[str, tuple[tuple[str, int, int], ...]] | None = None,
) -> np.ndarray:
    """Assemble [N,6,2] mean/uncertainty slots from full model-target outputs."""
    mapping = mapping or build_group_channel_mapping()
    expected_groups = set(mapping)
    actual_groups = set(group_outputs)
    if actual_groups != expected_groups:
        raise ValueError(
            f"attribute groups do not match mapping; missing={sorted(expected_groups - actual_groups)}, "
            f"unknown={sorted(actual_groups - expected_groups)}; expected targets by group="
            f"{ {group: tuple(item[0] for item in entries) for group, entries in mapping.items()} }; "
            "actual output dimensions=unavailable for missing groups; invalid mapping"
        )

    slots = np.full((sample_count, len(ATTRIBUTE_NAMES), 2), np.nan, dtype=np.float32)
    assignment_counts = np.zeros(len(ATTRIBUTE_NAMES), dtype=np.int32)
    for group, entries in mapping.items():
        values = np.asarray(group_outputs[group])
        expected_shape = (sample_count, len(MODEL_TARGETS), 2)
        expected_targets = tuple(item[0] for item in entries)
        if values.shape != expected_shape:
            raise ValueError(
                f"group '{group}': incompatible output shape; expected targets={expected_targets}; "
                f"expected output dimensions={expected_shape} with model target order={tuple(MODEL_TARGETS)}; "
                f"actual output dimensions={values.shape}; invalid mapping"
            )
        for target, feature_slot, output_channel in entries:
            if output_channel < 0 or output_channel >= values.shape[1]:
                raise ValueError(
                    f"group '{group}': invalid channel mapping for target '{target}'; "
                    f"expected targets={expected_targets}; output channel={output_channel}; "
                    f"actual output dimensions={values.shape}; invalid mapping"
                )
            if assignment_counts[feature_slot]:
                raise ValueError(
                    f"group '{group}': duplicate feature-slot assignment for '{target}'; "
                    f"expected targets={expected_targets}; actual output dimensions={values.shape}; "
                    "invalid mapping"
                )
            selected = values[:, output_channel, :]
            if not np.isfinite(selected).all():
                raise ValueError(
                    f"group '{group}': nonfinite values for target '{target}'; "
                    f"expected targets={expected_targets}; actual output dimensions={values.shape}; "
                    "invalid mapping"
                )
            slots[:, feature_slot, :] = selected
            assignment_counts[feature_slot] += 1

    unfilled = tuple(ATTRIBUTE_NAMES[i] for i, count in enumerate(assignment_counts) if count != 1)
    if unfilled:
        raise ValueError(
            f"unfilled or multiply assigned malignancy feature slots={unfilled}; "
            f"expected attributes={ATTRIBUTE_NAMES}; actual slot shape={slots.shape}; "
            f"assignment counts={assignment_counts.tolist()}"
        )
    if not np.isfinite(slots).all():
        raise ValueError(
            f"nonfinite assembled feature slots; expected attributes={ATTRIBUTE_NAMES}; "
            f"actual slot shape={slots.shape}; assignment counts={assignment_counts.tolist()}"
        )
    return slots


def pack_feature_matrix(slots: np.ndarray) -> np.ndarray:
    """Pack six score means followed by their six corresponding uncertainties."""
    expected_shape = (slots.shape[0], len(ATTRIBUTE_NAMES), 2) if slots.ndim == 3 else None
    if slots.ndim != 3 or slots.shape != expected_shape:
        raise ValueError(
            f"Expected assembled slots [N,{len(ATTRIBUTE_NAMES)},2], got {slots.shape}"
        )
    if not np.isfinite(slots).all():
        raise ValueError(f"Cannot pack nonfinite feature slots with shape {slots.shape}")
    features = np.concatenate((slots[:, :, 0], slots[:, :, 1]), axis=1).astype(np.float32)
    if features.shape != (slots.shape[0], 12):
        raise ValueError(f"Expected 12 malignancy features, got shape {features.shape}")
    return features


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(temporary_file, **arrays)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_fold_cache(
    path: Path,
    expected_metadata: dict[str, Any],
    expected_positions: np.ndarray,
    expected_h5_indices: np.ndarray,
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata"].item()))
            positions = np.asarray(cache["train_positions"], dtype=np.int64)
            h5_indices = np.asarray(cache["h5_indices"], dtype=np.int64)
            features = np.asarray(cache["X"], dtype=np.float32)
        if metadata != expected_metadata:
            return None
        if not np.array_equal(positions, expected_positions):
            return None
        if not np.array_equal(h5_indices, expected_h5_indices):
            return None
        if features.shape != (len(expected_positions), 12) or not np.isfinite(features).all():
            return None
        return features
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def load_attribute_checkpoint(
    path: Path, device: torch.device, expected_group: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    if architecture.get("objective") != "ordinal_reader":
        raise ValueError(f"{path} must be an ordinal_reader attribute checkpoint")
    declared_targets = tuple(architecture.get("targets", ()))
    if declared_targets != tuple(MODEL_TARGETS):
        raise ValueError(
            f"group '{expected_group}': checkpoint target order mismatch; "
            f"expected targets={tuple(MODEL_TARGETS)}; actual checkpoint targets={declared_targets}; "
            "actual output dimensions are not verified until inference"
        )
    checkpoint_group = checkpoint.get("attribute_group") or checkpoint.get("training_config", {}).get("attribute_group")
    if checkpoint_group != expected_group:
        raise ValueError(
            f"group '{expected_group}': checkpoint group metadata mismatch; "
            f"expected targets={GROUPS.get(expected_group)}; actual checkpoint group={checkpoint_group!r}; "
            f"checkpoint output target order={declared_targets}"
        )
    model = SpatialObjectiveModel(
        architecture["objective"],
        architecture["attention_dim"],
        architecture["head_hidden"],
        architecture["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint, {"model": model.to(device), "architecture": architecture}


def mc_attribute_predictions(
    model: SpatialObjectiveModel,
    h5: h5py.File,
    indices: np.ndarray,
    device: torch.device,
    passes: int,
    batch_size: int,
    group: str,
) -> np.ndarray:
    """Return [N,8,2], ordered by MODEL_TARGETS, with mean and MC-dropout SD."""
    if passes < 2:
        raise ValueError("--mc-passes must be at least 2 to estimate uncertainty")
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
    results = np.empty((len(indices), len(MODEL_TARGETS), 2), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            patches = read_h5_indices(h5["patches"], batch_indices)
            x = preprocess_hu(torch.from_numpy(patches).to(device)).unsqueeze(1)
            scores = []
            for _ in range(passes):
                outputs = model(x)
                if len(outputs) != len(MODEL_TARGETS):
                    raise ValueError(
                        f"group '{group}': model returned {len(outputs)} output heads; "
                        f"expected targets={tuple(MODEL_TARGETS)} ({len(MODEL_TARGETS)} heads); "
                        f"actual output dimensions={[tuple(output.shape) for output in outputs]}"
                    )
                expected = model.expected_ratings(outputs)
                if expected.ndim != 2 or expected.shape[1] != len(MODEL_TARGETS):
                    raise ValueError(
                        f"group '{group}': expected ratings do not match target channels; "
                        f"expected targets={tuple(MODEL_TARGETS)}; "
                        f"actual output dimensions={tuple(expected.shape)}"
                    )
                scores.append(expected)
            draws = torch.stack(scores, dim=0)
            means = draws.mean(dim=0)
            stds = draws.std(dim=0, unbiased=False)
            results[start : start + len(batch_indices), :, 0] = means.cpu().numpy()
            results[start : start + len(batch_indices), :, 1] = stds.cpu().numpy()
    model.eval()
    return results


def make_patient_folds(patient_ids: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    patients = np.unique(patient_ids.astype(str))
    if folds < 2 or folds > len(patients):
        raise ValueError(f"folds must be between 2 and number of train patients ({len(patients)})")
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    patient_buckets = np.array_split(patients, folds)
    assignments = [np.flatnonzero(np.isin(patient_ids.astype(str), bucket)) for bucket in patient_buckets]
    if any(len(bucket) == 0 for bucket in assignments):
        raise RuntimeError("Patient fold assignment produced an empty fold")
    return assignments


def fold_training_config(checkpoint: dict[str, Any], group: str, fold_epochs: int | None) -> dict[str, Any]:
    config = dict(checkpoint.get("training_config") or {})
    architecture = checkpoint["architecture"]
    return {
        "attribute": config.get("attribute") or "all",
        "attribute_group": checkpoint.get("attribute_group") or config.get("attribute_group") or group,
        "ranking_target": checkpoint.get("ranking_target") or config.get("ranking_target") or "median",
        "rank_loss_weight": float(config.get("rank_loss_weight", 0.5)),
        "ranking_pair_threshold": float(config.get("ranking_pair_threshold", 0.5)),
        "ranking_min_confidence": float(config.get("ranking_min_confidence", 0.10)),
        "batch_size": int(config.get("batch_size", 8)),
        "epochs": int(fold_epochs or checkpoint.get("epoch") or config.get("epochs", 75)),
        "lr": float(config.get("lr", 5e-4)),
        "weight_decay": float(config.get("weight_decay", 0.05)),
        "warmup_epochs": int(config.get("warmup_epochs", 5)),
        "min_lr_factor": float(config.get("min_lr_factor", 0.05)),
        "grad_clip": float(config.get("grad_clip", 1.0)),
        "amp": str(config.get("amp", "bf16")),
        "attn_dim": int(architecture["attention_dim"]),
        "head_hidden": int(architecture["head_hidden"]),
        "dropout": float(architecture["dropout"]),
    }


def train_fold_model(
    group: str,
    checkpoint: dict[str, Any],
    h5_path: Path,
    indices: np.ndarray,
    targets: dict[str, np.ndarray],
    device: torch.device,
    seed: int,
    fold_epochs: int | None,
    num_workers: int,
) -> SpatialObjectiveModel:
    config = fold_training_config(checkpoint, group, fold_epochs)
    seed_all(seed)
    architecture = checkpoint["architecture"]
    model = SpatialObjectiveModel(
        architecture["objective"],
        config["attn_dim"],
        config["head_hidden"],
        config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    dataset = H5PatchDataset(h5_path, indices, targets, config["ranking_target"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
    steps_per_epoch = max(1, len(loader))
    total_steps = max(1, config["epochs"] * steps_per_epoch)
    warmup_steps = max(1, config["warmup_epochs"] * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return config["min_lr_factor"] + (1.0 - config["min_lr_factor"]) * 0.5 * (
            1.0 + math.cos(math.pi * min(1.0, progress))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if config["amp"] == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif config["amp"] == "fp16" and device.type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16) if device.type == "cuda" else None
    selected = config["attribute"] if config["attribute"] != "all" else config["attribute_group"]
    model.train()
    for epoch in range(config["epochs"]):
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            x = preprocess_hu(batch["x"]).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            autocast_enabled = amp_dtype is not None and device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                loss = total_loss(
                    model,
                    model(x),
                    batch,
                    rank_loss_weight=config["rank_loss_weight"],
                    ranking_pair_threshold=config["ranking_pair_threshold"],
                    ranking_target=config["ranking_target"],
                    ranking_min_confidence=config["ranking_min_confidence"],
                    attribute=selected,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite fold loss for {group} at epoch {epoch + 1}")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
            scheduler.step()
    return model.eval()


def malignancy_targets(h5: h5py.File, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = read_h5_indices(h5["reader_labels"], indices)[:, :, MALIGNANCY_READER_INDEX]
    masks = read_h5_indices(h5["reader_mask"], indices).astype(bool)
    valid = masks & np.isfinite(labels) & (labels >= 1) & (labels <= 5)
    labels = np.where(valid, labels, np.nan).astype(np.float32)
    return labels, valid


def save_archive(path: Path, features: np.ndarray, labels: np.ndarray, valid: np.ndarray,
                 patients: np.ndarray, nodules: np.ndarray, indices: np.ndarray,
                 split_name: str) -> None:
    metadata = json.dumps({"split": split_name, "feature_names": [
        f"{name}_score" for name in ATTRIBUTE_NAMES
    ] + [f"{name}_uncertainty" for name in ATTRIBUTE_NAMES]})
    atomic_save_npz(
        path,
        X=features.astype(np.float32),
        y=labels.astype(np.float32),
        y_mask=valid.astype(bool),
        patient_ids=patients.astype(str),
        nodule_ids=nodules.astype(str),
        h5_indices=indices.astype(np.int64),
        split=np.full(len(indices), split_name),
        metadata=np.asarray(metadata),
    )


def main(args: argparse.Namespace) -> int:
    device = choose_device(args.device)
    h5_path = Path(args.h5).resolve()
    split_path = Path(args.split_file).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    group_mapping = build_group_channel_mapping()
    checkpoint_paths = {
        group: resolve_checkpoint_path(getattr(args, CHECKPOINT_ARGS[group]), group)
        for group in GROUPS
    }
    loaded = {
        group: load_attribute_checkpoint(path, device, expected_group=group)
        for group, path in checkpoint_paths.items()
    }
    architectures = [item[0]["architecture"] for item in loaded.values()]
    if any(architecture != architectures[0] for architecture in architectures[1:]):
        raise ValueError("All five hybrid checkpoints must share one architecture configuration")
    input_hashes = {
        "h5_sha256": sha256_file(h5_path),
        "split_sha256": sha256_file(split_path),
        "checkpoint_sha256": {
            group: sha256_file(path) for group, path in checkpoint_paths.items()
        },
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }

    with h5py.File(h5_path, "r") as h5:
        splits = parse_and_validate_split(split_path, int(h5["patches"].shape[0]), h5)
        expected_train_hash = hash_indices(splits["train"])
        for group, (checkpoint, _) in loaded.items():
            split_hashes = checkpoint.get("split_hashes") or {}
            checkpoint_train_hash = split_hashes.get("train")
            if checkpoint_train_hash != expected_train_hash:
                raise ValueError(f"{group} checkpoint was not trained on the requested train split")
            if checkpoint.get("test_used_for_selection", False):
                raise ValueError(f"{group} checkpoint used the test split for model selection")

        train_indices = splits["train"]
        train_patients = read_h5_indices(h5["patient_id"], train_indices)
        folds = make_patient_folds(train_patients, args.folds, args.seed)
        train_features = np.empty((len(train_indices), 12), dtype=np.float32)
        raw_labels = read_h5_indices(h5["reader_labels"], train_indices)
        raw_masks = read_h5_indices(h5["reader_mask"], train_indices).astype(bool)
        target_arrays = build_target_arrays(raw_labels, raw_masks)
        train_positions = np.arange(len(train_indices))
        fold_assignment = np.full(len(train_indices), -1, dtype=np.int64)
        for fold_number, positions in enumerate(folds):
            if np.any(fold_assignment[positions] != -1):
                raise ValueError(f"Training sample assigned to multiple OOF folds: {fold_number}")
            fold_assignment[positions] = fold_number
        if np.any(fold_assignment < 0):
            missing_positions = np.flatnonzero(fold_assignment < 0).tolist()
            raise ValueError(f"Training samples missing OOF fold assignments: {missing_positions[:10]}")
        fold_assignment_hash = hashlib.sha256(fold_assignment.tobytes()).hexdigest()
        fold_cache_dir = output_dir / "fold_cache"

        for fold_number, heldout_positions in enumerate(folds):
            fit_positions = np.setdiff1d(train_positions, heldout_positions, assume_unique=True)
            fold_indices = train_indices[fit_positions]
            heldout_h5_indices = train_indices[heldout_positions]
            fold_training_epochs = {
                group: fold_training_config(checkpoint, group, args.fold_epochs)["epochs"]
                for group, (checkpoint, _) in loaded.items()
            }
            fold_metadata = {
                **input_hashes,
                "fold_number": fold_number,
                "fold_count": args.folds,
                "fold_assignment_sha256": fold_assignment_hash,
                "heldout_train_positions": heldout_positions.tolist(),
                "heldout_h5_indices": heldout_h5_indices.tolist(),
                "heldout_patient_ids": np.asarray(train_patients[heldout_positions]).astype(str).tolist(),
                "seed": args.seed,
                "mc_passes": args.mc_passes,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "device": str(device),
                "fold_training_epochs": fold_training_epochs,
                "model_target_order": list(MODEL_TARGETS),
                "attribute_order": list(ATTRIBUTE_NAMES),
                "feature_order": [f"{name}_score" for name in ATTRIBUTE_NAMES]
                + [f"{name}_uncertainty" for name in ATTRIBUTE_NAMES],
            }
            cache_path = fold_cache_dir / f"fold_{fold_number:02d}.npz"
            print(
                f"Fold {fold_number + 1}/{args.folds}: fit={len(fit_positions)} "
                f"heldout={len(heldout_positions)} patients={len(np.unique(train_patients[heldout_positions]))}",
                flush=True,
            )
            cached_features = load_fold_cache(
                cache_path, fold_metadata, heldout_positions, heldout_h5_indices
            )
            if cached_features is not None:
                train_features[heldout_positions] = cached_features
                print(f"Reusing validated fold cache {cache_path}", flush=True)
                continue
            if cache_path.exists():
                print(f"Ignoring incomplete or incompatible fold cache {cache_path}", flush=True)
            fold_targets = {
                key: value[fit_positions] if isinstance(value, np.ndarray) and value.shape[0] == len(train_indices) else value
                for key, value in target_arrays.items()
            }
            fold_outputs: dict[str, np.ndarray] = {}
            for group, (checkpoint, _) in loaded.items():
                model = train_fold_model(
                    group,
                    checkpoint,
                    h5_path,
                    fold_indices,
                    fold_targets,
                    device,
                    args.seed + fold_number * 1009 + list(GROUPS).index(group),
                    args.fold_epochs,
                    args.num_workers,
                )
                mc_seed = args.seed + 500_000 + fold_number * 1009 + list(GROUPS).index(group)
                seed_all(mc_seed)
                fold_outputs[group] = mc_attribute_predictions(
                    model, h5, train_indices[heldout_positions], device,
                    args.mc_passes, args.batch_size, group,
                )
                del model
            assembled = assemble_attribute_slots(
                fold_outputs, len(heldout_positions), group_mapping
            )
            fold_features = pack_feature_matrix(assembled)
            atomic_save_npz(
                cache_path,
                X=fold_features,
                train_positions=heldout_positions.astype(np.int64),
                h5_indices=heldout_h5_indices.astype(np.int64),
                metadata=np.asarray(json.dumps(fold_metadata, sort_keys=True)),
            )
            train_features[heldout_positions] = fold_features
            print(f"Saved completed fold cache {cache_path}", flush=True)

        for split_name in ("val", "test"):
            indices = splits[split_name]
            split_outputs: dict[str, np.ndarray] = {}
            split_index = 0 if split_name == "val" else 1
            for group, (_, entry) in loaded.items():
                model = entry["model"]
                seed_all(args.seed + 700_001 + split_index * 1009 + list(GROUPS).index(group))
                split_outputs[group] = mc_attribute_predictions(
                    model, h5, indices, device, args.mc_passes, args.batch_size, group
                )
            combined = assemble_attribute_slots(split_outputs, len(indices), group_mapping)
            features = pack_feature_matrix(combined)
            labels, valid = malignancy_targets(h5, indices)
            patients = read_h5_indices(h5["patient_id"], indices)
            nodules = read_h5_indices(h5["nodule_id"], indices)
            save_archive(output_dir / f"{split_name}.npz", features, labels, valid, patients, nodules, indices, split_name)

        train_labels, train_valid = malignancy_targets(h5, train_indices)
        train_nodules = read_h5_indices(h5["nodule_id"], train_indices)
        save_archive(
            output_dir / "train_oof.npz", train_features, train_labels, train_valid,
            train_patients, train_nodules, train_indices, "train_oof",
        )
        all_features = np.concatenate((train_features, *[
            np.load(output_dir / f"{split}.npz")["X"] for split in ("val", "test")
        ]))
        all_labels = np.concatenate((train_labels, *[
            np.load(output_dir / f"{split}.npz")["y"] for split in ("val", "test")
        ]))
        all_valid = np.concatenate((train_valid, *[
            np.load(output_dir / f"{split}.npz")["y_mask"] for split in ("val", "test")
        ]))
        all_patients = np.concatenate((train_patients, *[
            read_h5_indices(h5["patient_id"], splits[split]) for split in ("val", "test")
        ]))
        all_nodules = np.concatenate((train_nodules, *[
            read_h5_indices(h5["nodule_id"], splits[split]) for split in ("val", "test")
        ]))
        all_indices = np.concatenate([splits[name] for name in ("train", "val", "test")])
        all_split = np.concatenate([
            np.full(len(splits[name]), "train_oof" if name == "train" else name)
            for name in ("train", "val", "test")
        ])
        atomic_save_npz(
            output_dir / "features.npz",
            X=all_features.astype(np.float32), y=all_labels.astype(np.float32),
            y_mask=all_valid.astype(bool), patient_ids=all_patients.astype(str),
            nodule_ids=all_nodules.astype(str), h5_indices=all_indices.astype(np.int64),
            split=all_split, metadata=np.asarray(json.dumps({"oof_folds": args.folds, "mc_passes": args.mc_passes})),
        )
    print(f"Saved train_oof.npz, val.npz, test.npz, and features.npz to {output_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--lobulation-spiculation", required=True)
    parser.add_argument("--subtlety", required=True)
    parser.add_argument("--sphericity", required=True)
    parser.add_argument("--margin", required=True)
    parser.add_argument("--texture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for patient-grouped out-of-fold generation.",
    )
    parser.add_argument("--fold-epochs", type=int, default=None,
                        help="Override fold training epochs; defaults to each checkpoint's selected epoch")
    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    raise SystemExit(main(parser.parse_args()))