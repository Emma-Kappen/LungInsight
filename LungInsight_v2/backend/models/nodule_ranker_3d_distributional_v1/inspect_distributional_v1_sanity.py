#!/usr/bin/env python3
"""Inspect the exact 16-nodule REAL-H5 subset used by distributional_v1 sanity.

Read-only. Does not modify the H5, split, model, or checkpoints.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import numpy as np


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--model-dir", required=True,
                    help="distributional_v1 directory containing model_3d.py and train_nodule_ranker_3d.py")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-nodules", type=int, default=16)
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    train_mod = load_module(model_dir / "train_nodule_ranker_3d.py", "dist_train_inspect")

    meta = train_mod.load_meta(args.h5)
    idx, split = train_mod.resolve_split(
        argparse.Namespace(split_file=args.split_file), meta
    )
    targets = train_mod.build_targets(meta["reader_labels"], meta["reader_mask"])

    sub = np.sort(np.random.RandomState(args.seed).choice(
        idx["train"], args.n_nodules, replace=False
    ))

    print("=== EXACT SANITY SUBSET ===")
    print(f"seed={args.seed} n={args.n_nodules}")
    print("indices:", sub.tolist())
    print()

    print("=== CALCIFICATION PER-NODULE TARGETS ===")
    h = train_mod.MODEL_TARGETS.index("calcification")
    k = train_mod.KS[h]
    labels = meta["reader_labels"][sub, :, train_mod.READER_TO_MODEL[h]]
    masks = meta["reader_mask"][sub]
    dist = targets["distribution"][sub, h, :k]
    means = targets["mean"][sub, h]
    counts = targets["count"][sub, h]

    reader_values = []
    for row, mask in zip(labels, masks):
        vals = row[mask & np.isfinite(row)].astype(int).tolist()
        reader_values.append(vals)

    for i, (idx0, vals, d, m, c) in enumerate(zip(sub, reader_values, dist, means, counts)):
        dstr = " ".join(f"{x:.3f}" for x in d)
        print(f"{idx0:4d}  readers={int(c):d}  ratings={vals}  mean={m:.3f}  dist=[{dstr}]")

    print()
    print("=== CALCIFICATION SUBSET SUMMARY ===")
    print(f"mean-of-nodule-means = {means.mean():.6f}")
    print(f"std-of-nodule-means  = {means.std():.6f}")
    print(f"min/max nodule mean  = {means.min():.6f} / {means.max():.6f}")
    print(f"unique nodule means  = {np.unique(np.round(means, 6)).tolist()}")
    print(f"mean target distribution = {[round(float(x), 6) for x in dist.mean(0)]}")

    # Aggregate reader-vote distribution over the selected nodules.
    valid_reader_vals = labels[masks & np.isfinite(labels)].astype(int)
    vote_counts = np.bincount(valid_reader_vals, minlength=k + 1)[1:]
    vote_dist = vote_counts / max(1, vote_counts.sum())
    print(f"aggregate reader-vote distribution = {[round(float(x), 6) for x in vote_dist]}")
    print(f"aggregate reader votes = {int(vote_counts.sum())}")

    # Constant expected-rating baselines. These make a collapsed head interpretable.
    for name, cst in [
        ("subset mean", float(means.mean())),
        ("subset median", float(np.median(means))),
        ("subset vote-distribution mean", float((vote_dist * np.arange(1, k + 1)).sum())),
    ]:
        mae = float(np.abs(means - cst).mean())
        print(f"constant baseline: {name:<28} pred={cst:.6f} MAE={mae:.6f}")

    print()
    print("=== TRAIN-WIDE CALCIFICATION TARGETS ===")
    tr = idx["train"]
    trm = targets["mean"][tr, h]
    trd = targets["distribution"][tr, h, :k]
    print(f"train nodules = {len(tr)}")
    print(f"mean-of-nodule-means = {trm.mean():.6f}")
    print(f"std-of-nodule-means  = {trm.std():.6f}")
    print(f"min/max nodule mean  = {trm.min():.6f} / {trm.max():.6f}")
    print(f"selected subset covers mean range = {means.min():.3f} .. {means.max():.3f}")
    print(f"train mean target distribution = {[round(float(x), 6) for x in trd.mean(0)]}")
    tr_labels = meta["reader_labels"][tr, :, train_mod.READER_TO_MODEL[h]]
    tr_masks = meta["reader_mask"][tr]
    tr_vals = tr_labels[tr_masks & np.isfinite(tr_labels)].astype(int)
    tr_counts = np.bincount(tr_vals, minlength=k + 1)[1:]
    tr_vote_dist = tr_counts / max(1, tr_counts.sum())
    print(f"train aggregate reader-vote distribution = {[round(float(x), 6) for x in tr_vote_dist]}")
    print(f"train aggregate reader votes = {int(tr_counts.sum())}")

    print()
    print("=== INTERPRETATION ===")
    if means.std() < 0.5:
        print("The 16-sample calcification target is low-variance; a near-constant prediction can be a subset-composition effect.")
    else:
        print("The 16-sample calcification target has substantial variation; inspect optimization/gradient behavior more closely.")


if __name__ == "__main__":
    main()
