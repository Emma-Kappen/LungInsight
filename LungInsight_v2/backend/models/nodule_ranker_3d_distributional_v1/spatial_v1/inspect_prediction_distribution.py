#!/usr/bin/env python3
"""Inspect ordered-attribute prediction distributions on the validation split."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from evaluate_objective import choose_device, load_checkpoint
from model_objective import MODEL_TARGETS, ORDERED_TARGETS, RATING_MAX, preprocess_hu
from train_objective import (
    H5PatchDataset,
    build_target_arrays,
    collate,
    finite_corr,
    parse_and_validate_split,
)


def save_histograms(output_dir: Path, predictions: np.ndarray, targets: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for h, name in enumerate(MODEL_TARGETS):
        if name not in ORDERED_TARGETS:
            continue
        pred = predictions[:, h]
        target = targets[:, h]
        finite = np.isfinite(pred) & np.isfinite(target)
        k = RATING_MAX[name]
        edges = np.linspace(0.5, k + 0.5, 51)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(target[finite], bins=np.arange(0.5, k + 1.5, 1.0), density=True, alpha=0.6, label="Target")
        ax.hist(pred[finite], bins=edges, density=True, alpha=0.6, label="Prediction")
        ax.set(title=f"{name}: predictions and reader means", xlabel="Rating", ylabel="Density")
        ax.set_xlim(0.5, k + 0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", dpi=160)
        plt.close(fig)


@torch.inference_mode()
def collect_predictions(model, loader, device: torch.device) -> np.ndarray:
    batches: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        outputs = model(preprocess_hu(x).unsqueeze(1))
        batches.append(model.expected_ratings(outputs).cpu().numpy().astype(np.float32))
    if not batches:
        raise ValueError("Validation split is empty; no predictions to inspect")
    return np.concatenate(batches, axis=0)


def main(args: argparse.Namespace) -> int:
    device = choose_device(args.device)
    checkpoint = Path(args.checkpoint).resolve()
    h5_path = Path(args.h5).resolve()
    split_path = Path(args.split_file).resolve()
    model, ck = load_checkpoint(checkpoint, device)

    with h5py.File(h5_path, "r") as h5:
        split = parse_and_validate_split(split_path, int(h5["patches"].shape[0]), h5)
        indices = split[args.split]
        order = np.argsort(indices, kind="mergesort")
        sorted_indices = indices[order]
        reader_labels = np.asarray(h5["reader_labels"][sorted_indices], dtype=np.float32)
        reader_mask = np.asarray(h5["reader_mask"][sorted_indices], dtype=bool)
        restore = np.argsort(order, kind="mergesort")
        targets = build_target_arrays(reader_labels[restore], reader_mask[restore])

    dataset = H5PatchDataset(
        h5_path,
        indices,
        {key: value for key, value in targets.items() if key != "invalid_records"},
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )
    predictions = collect_predictions(model, loader, device)
    target_means = targets["mean"]

    print(f"Checkpoint: {checkpoint}")
    print(f"Objective: {ck['architecture']['objective']} | split: {args.split} | n={len(indices)}")
    for h, name in enumerate(MODEL_TARGETS):
        if name not in ORDERED_TARGETS:
            continue
        valid = targets["valid"][:, h]
        pred = predictions[valid, h].astype(np.float64)
        target = target_means[valid, h].astype(np.float64)
        if not len(pred):
            print(f"\n{name}\n---------\nn=0 (no valid targets)")
            continue

        pred_std = float(np.std(pred))
        target_std = float(np.std(target))
        sd_ratio = pred_std / target_std if target_std > 1e-12 else float("nan")
        print(f"\n{name}\n---------")
        print(f"n={len(pred)}")
        print(f"pred_mean={np.mean(pred):.3f}")
        print(f"pred_std={pred_std:.3f}")
        print(f"pred_min={np.min(pred):.3f}")
        print(f"pred_max={np.max(pred):.3f}")
        print(f"\ntarget_mean={np.mean(target):.3f}")
        print(f"target_std={target_std:.3f}")
        print(f"target_min={np.min(target):.3f}")
        print(f"target_max={np.max(target):.3f}")
        print(f"\nsd_ratio={sd_ratio:.4f}")
        print(f"spearman={finite_corr(pred, target, spearmanr):.3f}")
        print(f"pearson={finite_corr(pred, target, pearsonr):.3f}")
        percentiles = np.percentile(pred, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        print("\nPrediction percentiles:")
        for percentile, value in zip([0, 1, 5, 25, 50, 75, 95, 99, 100], percentiles):
            print(f"{percentile:>3}% = {value:.3f}")

    if args.save_histograms:
        save_histograms(Path(args.save_histograms).resolve(), predictions, target_means)
        print(f"\nSaved histograms to {Path(args.save_histograms).resolve()}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-histograms", default=None, metavar="OUTPUT_DIR")
    raise SystemExit(main(parser.parse_args()))