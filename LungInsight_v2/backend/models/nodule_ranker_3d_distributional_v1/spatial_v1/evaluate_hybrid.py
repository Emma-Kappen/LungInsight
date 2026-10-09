#!/usr/bin/env python3
"""Evaluate the five hybrid checkpoints together on a canonical H5 split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from model_objective import MODEL_TARGETS, ORDERED_TARGETS, SpatialObjectiveModel, preprocess_hu
from train_objective import (
    H5PatchDataset,
    build_target_arrays,
    clean,
    collate,
    hash_indices,
    metrics_for_predictions,
    parse_and_validate_split,
)


CHECKPOINT_GROUPS = {
    "lobulation_spiculation": ("lobulation", "spiculation"),
    "subtlety": ("subtlety",),
    "sphericity": ("sphericity",),
    "margin": ("margin",),
    "texture": ("texture",),
}


def choose_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = SpatialObjectiveModel(
        architecture["objective"],
        architecture["attention_dim"],
        architecture["head_hidden"],
        architecture["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint, architecture


def main(args: argparse.Namespace) -> int:
    device = choose_device(args.device)
    h5_path = Path(args.h5).resolve()
    split_path = Path(args.split_file).resolve()
    checkpoint_paths = {
        "lobulation_spiculation": Path(args.lobulation_spiculation_checkpoint).resolve(),
        "subtlety": Path(args.subtlety_checkpoint).resolve(),
        "sphericity": Path(args.sphericity_checkpoint).resolve(),
        "margin": Path(args.margin_checkpoint).resolve(),
        "texture": Path(args.texture_checkpoint).resolve(),
    }
    if args.split == "test":
        print("WARNING: explicit TEST evaluation. This must be the final held-out evaluation and must not feed back into model development.", flush=True)

    with h5py.File(h5_path, "r") as h5:
        split = parse_and_validate_split(split_path, int(h5["patches"].shape[0]), h5)
        indices = split[args.split]
        order = np.argsort(indices, kind="mergesort")
        sorted_indices = indices[order]
        reader_labels = np.asarray(h5["reader_labels"][sorted_indices], dtype=np.float32)
        reader_mask = np.asarray(h5["reader_mask"][sorted_indices], dtype=bool)
        restore = np.argsort(order, kind="mergesort")
        targets = build_target_arrays(reader_labels[restore], reader_mask[restore])

    dataset_targets = {key: value for key, value in targets.items() if key != "invalid_records"}
    predictions = np.full((len(indices), len(MODEL_TARGETS)), np.nan, dtype=np.float32)
    loader = DataLoader(
        H5PatchDataset(h5_path, indices, dataset_targets, "median"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )

    common_architecture = None
    for group, checkpoint_path in checkpoint_paths.items():
        model, _, architecture = load_model(checkpoint_path, device)
        if architecture["objective"] != "ordinal_reader":
            raise ValueError(f"{group} checkpoint must use ordinal_reader")
        if common_architecture is None:
            common_architecture = architecture
        elif architecture != common_architecture:
            raise ValueError("Hybrid checkpoints must share an identical architecture configuration")
        batch_predictions = []
        with torch.inference_mode():
            for batch in loader:
                x = preprocess_hu(batch["x"].to(device, non_blocking=True)).unsqueeze(1)
                batch_predictions.append(model.expected_ratings(model(x)).cpu().numpy().astype(np.float32))
        group_predictions = np.concatenate(batch_predictions, axis=0)
        for name in CHECKPOINT_GROUPS[group]:
            column = MODEL_TARGETS.index(name)
            predictions[:, column] = group_predictions[:, column]

    if any(not np.isfinite(predictions[:, MODEL_TARGETS.index(name)]).all() for name in ORDERED_TARGETS):
        raise RuntimeError("A hybrid checkpoint did not provide every ordered attribute")
    metrics, m6_ge3, m6_all = metrics_for_predictions(
        predictions,
        {},
        dataset_targets,
        ORDERED_TARGETS,
    )
    report = {
        "checkpoints": {group: str(path) for group, path in checkpoint_paths.items()},
        "objective": "ordinal_reader",
        "split": args.split,
        "test_warning": args.split == "test",
        "split_hash": hash_indices(indices),
        "M6>=3": m6_ge3,
        "M6all": m6_all,
        "m6_ge3": m6_ge3,
        "m6_all": m6_all,
        "metrics": metrics,
        "invalid_ratings_excluded": targets["invalid_records"],
    }
    serialized = json.dumps(clean(report), indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
        print(f"Saved {output_path}")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--lobulation-spiculation-checkpoint", required=True)
    parser.add_argument("--subtlety-checkpoint", required=True)
    parser.add_argument("--sphericity-checkpoint", required=True)
    parser.add_argument("--margin-checkpoint", required=True)
    parser.add_argument("--texture-checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default=None)
    raise SystemExit(main(parser.parse_args()))
