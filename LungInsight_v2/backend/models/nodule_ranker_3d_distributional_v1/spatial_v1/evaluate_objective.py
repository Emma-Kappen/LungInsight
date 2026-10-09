#!/usr/bin/env python3
"""One-shot validation/test evaluator for a spatial_objective_v1 checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from model_objective import MODEL_TARGETS, NOMINAL_TARGETS, ORDERED_TARGETS, RATING_MAX, SpatialObjectiveModel, preprocess_hu
from train_objective import ATTRIBUTE_GROUPS, H5PatchDataset, build_target_arrays, clean, collate, hash_indices, metrics_for_predictions, parse_and_validate_split, total_loss as compute_total_loss


def load_checkpoint(path: Path, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck["architecture"]
    model = SpatialObjectiveModel(
        arch["objective"],
        arch["attention_dim"],
        arch["head_hidden"],
        arch["dropout"],
    )
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return model.to(device).eval(), ck


def choose_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


@torch.inference_mode()
def evaluate_split(
    model,
    loader,
    device,
    targets,
    rank_loss_weight=0.5,
    ranking_pair_threshold=0.5,
    ranking_target="mean",
    ranking_min_confidence=0.10,
    attribute="all",
):
    preds = []
    nominal = {n: [] for n in NOMINAL_TARGETS}
    loss_total = 0.0
    total_n = 0
    for batch in loader:
        b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        x = preprocess_hu(b["x"]).unsqueeze(1)
        outputs = model(x)
        loss = compute_total_loss(
            model, outputs, b, rank_loss_weight, ranking_pair_threshold,
            ranking_target, ranking_min_confidence, attribute,
        )
        total_loss_value = float(loss.detach()) * len(b["x"])
        loss_total += total_loss_value
        total_n += len(b["x"])
        preds.append(model.expected_ratings(outputs).cpu().numpy().astype(np.float32))
        for h, name in enumerate(MODEL_TARGETS):
            if name in NOMINAL_TARGETS:
                nominal[name].append(torch.softmax(outputs[h].float(), dim=-1).cpu().numpy().astype(np.float32))

    pred = np.concatenate(preds, axis=0)
    nominal_np = {k: np.concatenate(v, axis=0) for k, v in nominal.items()}
    per, m6_ge3, m6_all = metrics_for_predictions(pred, nominal_np, targets, attribute)
    return loss_total / max(total_n, 1), per, m6_ge3, m6_all


def main(a: argparse.Namespace) -> int:
    device = choose_device(a.device)
    checkpoint = Path(a.checkpoint).resolve()
    h5_path = Path(a.h5).resolve()
    split_path = Path(a.split_file).resolve()
    if a.split == "test":
        print("WARNING: explicit TEST evaluation. This must be the final held-out evaluation and must not feed back into model development.", flush=True)

    model, ck = load_checkpoint(checkpoint, device)
    training_config = ck.get("training_config", {})
    ranking_target = training_config.get("ranking_target", "mean")
    rank_loss_weight = float(training_config.get("rank_loss_weight", 0.0))
    ranking_pair_threshold = float(training_config.get("ranking_pair_threshold", 0.5))
    ranking_min_confidence = float(training_config.get("ranking_min_confidence", 0.10))
    attribute_group = training_config.get("attribute_group", ck.get("attribute_group"))
    attribute = (
        attribute_group
        if attribute_group in ATTRIBUTE_GROUPS and attribute_group != "all"
        else training_config.get("attribute", ck.get("attribute", "all"))
    )
    with h5py.File(h5_path, "r") as h5:
        n = int(h5["patches"].shape[0])
        split = parse_and_validate_split(split_path, n, h5)
        idx = split[a.split]
        order = np.argsort(idx, kind="mergesort")
        sorted_idx = idx[order]
        rl_sorted = np.asarray(h5["reader_labels"][sorted_idx], dtype=np.float32)
        rm_sorted = np.asarray(h5["reader_mask"][sorted_idx], dtype=bool)
        restore = np.argsort(order, kind="mergesort")
        targets = build_target_arrays(rl_sorted[restore], rm_sorted[restore])

    ds = H5PatchDataset(
        h5_path,
        idx,
        {k: v for k, v in targets.items() if k != "invalid_records"},
        ranking_target,
    )
    loader = DataLoader(
        ds,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        persistent_workers=a.num_workers > 0,
    )
    loss, per, m6_ge3, m6_all = evaluate_split(
        model,
        loader,
        device,
        {k: v for k, v in targets.items() if k != "invalid_records"},
        rank_loss_weight,
        ranking_pair_threshold,
        ranking_target,
        ranking_min_confidence,
        attribute,
    )
    report = {
        "checkpoint": str(checkpoint),
        "objective": ck["architecture"]["objective"],
        "split": a.split,
        "test_warning": a.split == "test",
        "split_hash": hash_indices(idx),
        "loss": loss,
        "ranking_config": {
            "ranking_target": ranking_target,
            "rank_loss_weight": rank_loss_weight,
            "ranking_pair_threshold": ranking_pair_threshold,
            "ranking_min_confidence": ranking_min_confidence,
            "attribute": attribute,
            "attribute_group": attribute_group or "all",
        },
        "m6_ge3": m6_ge3,
        "m6_all": m6_all,
        "metrics": per,
        "invalid_ratings_excluded": targets["invalid_records"],
    }
    if a.output:
        out = Path(a.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(clean(report), indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved {out}")
    else:
        print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--split-file", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output", default=None)
    raise SystemExit(main(p.parse_args()))
