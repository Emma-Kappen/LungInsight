#!/usr/bin/env python3
"""Generate leak-free hybrid attribute features and reader malignancy targets."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
SPATIAL_DIR = HERE.parent / "nodule_ranker_3d_distributional_v1" / "spatial_v1"
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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_attribute_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    if architecture.get("objective") != "ordinal_reader":
        raise ValueError(f"{path} must be an ordinal_reader attribute checkpoint")
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
) -> np.ndarray:
    """Return [N, 6, 2] mean and MC-dropout SD of expected ratings."""
    if passes < 2:
        raise ValueError("--mc-passes must be at least 2 to estimate uncertainty")
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
    results = np.empty((len(indices), len(ATTRIBUTE_NAMES), 2), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            patches = read_h5_indices(h5["patches"], batch_indices)
            x = preprocess_hu(torch.from_numpy(patches).to(device)).unsqueeze(1)
            scores = []
            for _ in range(passes):
                expected = model.expected_ratings(model(x))
                score_indices = [MODEL_TARGETS.index(name) for name in ATTRIBUTE_NAMES]
                scores.append(expected[:, score_indices])
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
    np.savez_compressed(
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
    checkpoint_paths = {
        group: Path(getattr(args, CHECKPOINT_ARGS[group])).resolve() for group in GROUPS
    }
    loaded = {group: load_attribute_checkpoint(path, device) for group, path in checkpoint_paths.items()}
    architectures = [item[0]["architecture"] for item in loaded.values()]
    if any(architecture != architectures[0] for architecture in architectures[1:]):
        raise ValueError("All five hybrid checkpoints must share one architecture configuration")

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

        for fold_number, heldout_positions in enumerate(folds):
            fit_positions = np.setdiff1d(train_positions, heldout_positions, assume_unique=True)
            fold_indices = train_indices[fit_positions]
            print(
                f"Fold {fold_number + 1}/{args.folds}: fit={len(fit_positions)} "
                f"heldout={len(heldout_positions)} patients={len(np.unique(train_patients[heldout_positions]))}",
                flush=True,
            )
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
                fold_outputs[group] = mc_attribute_predictions(
                    model, h5, train_indices[heldout_positions], device, args.mc_passes, args.batch_size
                )
                del model
            assembled = np.empty((len(heldout_positions), 6, 2), dtype=np.float32)
            for group, names in GROUPS.items():
                group_values = fold_outputs[group]
                model_names = ("lobulation", "spiculation") if group == "lobulation_spiculation" else names
                for name in model_names:
                    assembled[:, ATTRIBUTE_NAMES.index(name), :] = group_values[:, MODEL_TARGETS.index(name), :]
            train_features[heldout_positions] = np.concatenate(
                (assembled[:, :, 0], assembled[:, :, 1]), axis=1
            )

        for split_name in ("val", "test"):
            indices = splits[split_name]
            combined = np.full((len(indices), 6, 2), np.nan, dtype=np.float32)
            for group, (_, entry) in loaded.items():
                model = entry["model"]
                predictions = mc_attribute_predictions(
                    model, h5, indices, device, args.mc_passes, args.batch_size
                )
                for name in GROUPS[group]:
                    combined[:, ATTRIBUTE_NAMES.index(name), :] = predictions[:, MODEL_TARGETS.index(name), :]
            features = np.concatenate((combined[:, :, 0], combined[:, :, 1]), axis=1)
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
        np.savez_compressed(
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
    parser.add_argument("--fold-epochs", type=int, default=None,
                        help="Override fold training epochs; defaults to each checkpoint's selected epoch")
    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    raise SystemExit(main(parser.parse_args()))