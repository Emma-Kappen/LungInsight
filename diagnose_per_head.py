#!/usr/bin/env python3
"""
Read-only validation diagnostics for distributional_v1 checkpoints.

Uses the exact model/training utilities from this directory:
  - model_3d.load_trained_model
  - model_3d.preprocess_hu
  - train_nodule_ranker_3d.build_targets
  - train_nodule_ranker_3d.resolve_split
  - train_nodule_ranker_3d.compute_metrics

No training, H5 mutation, split mutation, or test evaluation.
Evaluates every checkpoint matching best.pth/last.pth or a supplied .pth file.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model_3d import load_trained_model, preprocess_hu
from train_nodule_ranker_3d import (
    MODEL_TARGETS,
    Amp,
    build_targets,
    compute_metrics,
    get_device,
    load_meta,
    parse_split_file,
    read_patches,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--split-file", required=True)
    p.add_argument("--checkpoint", required=True,
                   help=".pth file or directory containing best.pth/last.pth")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def json_clean(x):
    if isinstance(x, dict):
        return {str(k): json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def checkpoint_paths(spec: str):
    p = Path(spec)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(p)
    found = []
    for name in ("best.pth", "last.pth"):
        q = p / name
        if q.exists():
            found.append(q)
    if not found:
        found = sorted(p.glob("*.pth"))
    if not found:
        raise FileNotFoundError(f"No .pth checkpoints found in {p}")
    return found


def load_split_indices(split_file, meta):
    # Use the exact canonical parser from the trainer.
    return parse_split_file(Path(split_file), meta)


def make_loader(x, batch_size, workers, device):
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(x)))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    chunks = []
    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        with amp.ctx():
            logits = model(preprocess_hu(x).unsqueeze(1))
            pred = model.expected_rating(logits)
        chunks.append(pred.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def head_distribution(pred, true_mean, h):
    p = pred[:, h].astype(np.float64)
    t = true_mean[:, h].astype(np.float64)
    rounded = np.clip(np.floor(p + 0.5), 1, 100).astype(np.int64)
    return {
        "pred_mean": float(p.mean()),
        "true_mean": float(t.mean()),
        "pred_std": float(p.std()),
        "true_std": float(t.std()),
        "std_ratio": float(p.std() / max(t.std(), 1e-9)),
        "pred_min": float(p.min()),
        "pred_max": float(p.max()),
        "true_min": float(t.min()),
        "true_max": float(t.max()),
        "pred_quantiles": {
            "q01": float(np.quantile(p, .01)),
            "q10": float(np.quantile(p, .10)),
            "q25": float(np.quantile(p, .25)),
            "q50": float(np.quantile(p, .50)),
            "q75": float(np.quantile(p, .75)),
            "q90": float(np.quantile(p, .90)),
            "q99": float(np.quantile(p, .99)),
        },
        "true_quantiles": {
            "q01": float(np.quantile(t, .01)),
            "q10": float(np.quantile(t, .10)),
            "q25": float(np.quantile(t, .25)),
            "q50": float(np.quantile(t, .50)),
            "q75": float(np.quantile(t, .75)),
            "q90": float(np.quantile(t, .90)),
            "q99": float(np.quantile(t, .99)),
        },
        "unique_rounded_predictions": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(rounded, return_counts=True))
        },
    }


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    amp = Amp(args.amp, device)

    print(f"device={device} amp={args.amp}")
    print(f"H5={args.h5}")
    print(f"split={args.split_file}")

    meta = load_meta(Path(args.h5))
    split = load_split_indices(args.split_file, meta)
    val_idx = np.asarray(split["val"], dtype=np.int64)
    print(f"validation nodules={len(val_idx)}")

    targets = build_targets(meta["reader_labels"], meta["reader_mask"])
    val_mean = targets["mean"][val_idx]
    val_count = targets["count"][val_idx]
    val_valid = targets["valid"][val_idx]

    # Read only validation patches.
    patches = read_patches(Path(args.h5), val_idx)
    loader = make_loader(patches, args.batch_size, args.num_workers, device)

    results = {}
    rows = []

    for ckpt in checkpoint_paths(args.checkpoint):
        print(f"\n===== {ckpt.name} =====")
        model, ck = load_trained_model(ckpt, device=device)
        pred = predict(model, loader, device, amp)
        metrics = compute_metrics(pred, val_mean, val_count, val_valid, min_readers=1)

        # Additional distribution diagnostics beyond the trainer's metrics.
        extra = {}
        for h, name in enumerate(MODEL_TARGETS):
            extra[name] = head_distribution(pred, val_mean, h)

        result = {
            "checkpoint": str(ckpt),
            "checkpoint_epoch": int(ck.get("epoch", -1)),
            "model_identity": ck.get("model_identity"),
            "architecture": ck.get("architecture"),
            "aggregate_metrics": metrics["aggregate"],
            "per_head_metrics": metrics["per_head"],
            "distribution_diagnostics": extra,
            "min_readers": 1,
            "n_validation": int(len(val_idx)),
        }
        results[ckpt.name] = json_clean(result)

        a = metrics["aggregate"]
        print(
            f"aggregate MAE={a['mae']:.4f} RMSE={a['rmse']:.4f} "
            f"Spearman={a['mean_spearman']:.4f} "
            f"Pearson={a['mean_pearson']:.4f} "
            f"SD-ratio={a['mean_std_ratio']:.3f} "
            f"collapsed={a['n_collapsed_heads']}/8"
        )
        for h, name in enumerate(MODEL_TARGETS):
            m = metrics["per_head"][name]
            d = extra[name]
            print(
                f"  {name:<19} n={m['n']:3d} "
                f"MAE={m['mae']:.3f} spear={m['spearman']:.3f} "
                f"predSD={m['pred_std']:.3f} trueSD={m['true_std']:.3f} "
                f"ratio={m['std_ratio']:.3f} "
                f"rounded={d['unique_rounded_predictions']}"
                f"{' COLLAPSED' if m['collapsed'] else ''}"
            )

        for h, name in enumerate(MODEL_TARGETS):
            m = metrics["per_head"][name]
            d = extra[name]
            rows.append({
                "checkpoint": ckpt.name,
                "epoch": int(ck.get("epoch", -1)),
                "head": name,
                "n": m["n"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "exact_acc": m["exact_acc"],
                "within1_acc": m["within1_acc"],
                "pearson": m["pearson"],
                "spearman": m["spearman"],
                "pred_mean": d["pred_mean"],
                "true_mean": d["true_mean"],
                "pred_std": d["pred_std"],
                "true_std": d["true_std"],
                "std_ratio": d["std_ratio"],
                "pred_min": d["pred_min"],
                "pred_max": d["pred_max"],
                "true_min": d["true_min"],
                "true_max": d["true_max"],
                "unique_rounded_predictions": json.dumps(d["unique_rounded_predictions"], sort_keys=True),
                "collapsed": m["collapsed"],
            })

    payload = {
        "note": "Validation-only diagnostic. No test evaluation. Only saved checkpoints can be evaluated; this run has best.pth and last.pth, so epoch trajectories are not reconstructable.",
        "h5": str(args.h5),
        "split_file": str(args.split_file),
        "validation_n": int(len(val_idx)),
        "targets": MODEL_TARGETS,
        "results": results,
    }
    (out / "per_head_diagnostics.json").write_text(
        json.dumps(json_clean(payload), indent=2), encoding="utf-8"
    )

    fields = list(rows[0].keys()) if rows else []
    with (out / "per_head_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with (out / "per_epoch_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "epoch", "mean_spearman", "mean_pearson", "mean_std_ratio", "collapsed_heads", "mae", "rmse"])
        for name, r in results.items():
            a = r["aggregate_metrics"]
            w.writerow([name, r["checkpoint_epoch"], a["mean_spearman"], a["mean_pearson"], a["mean_std_ratio"], a["n_collapsed_heads"], a["mae"], a["rmse"]])

    print(f"\nWrote: {out / 'per_head_diagnostics.json'}")
    print(f"Wrote: {out / 'per_head_summary.csv'}")
    print(f"Wrote: {out / 'per_epoch_metrics.csv'}")


if __name__ == "__main__":
    main()
