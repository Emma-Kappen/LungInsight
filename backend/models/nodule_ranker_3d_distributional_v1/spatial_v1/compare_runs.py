#!/usr/bin/env python3
"""Compare two completed spatial_objective_v1 validation runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def metrics_from_checkpoint(run: dict) -> dict:
    ckpt = Path(run["files"]["checkpoint"])
    if ckpt.is_file():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        return ck.get("validation_metrics", {})
    return run.get("last_validation_metrics", {}) or {}


def fmt(v):
    if v is None:
        return "nan"
    try:
        return f"{float(v):+.4f}"
    except (TypeError, ValueError):
        return str(v)


def main(a):
    ra = load(Path(a.objective_a))
    rb = load(Path(a.objective_b))
    print("LUNGINSIGHT SPATIAL_OBJECTIVE_V1 COMPARISON")
    print("=" * 84)
    print(f"A: {ra['objective']} | best epoch={ra['best_epoch']} | {ra['best_selection_metric']}={fmt(ra['best_selection_score'])}")
    print(f"B: {rb['objective']} | best epoch={rb['best_epoch']} | {rb['best_selection_metric']}={fmt(rb['best_selection_score'])}")
    print(f"Delta A-B: {fmt(ra['best_selection_score'] - rb['best_selection_score'])}")
    print("\nOrdered attributes")
    print(f"{'Attribute':20s} {'A rho>=3':>12s} {'B rho>=3':>12s} {'delta':>12s} {'A MAE':>10s} {'B MAE':>10s}")
    print("-" * 84)
    ma = metrics_from_checkpoint(ra)
    mb = metrics_from_checkpoint(rb)
    for name in ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"]:
        aa = ma.get(name, {})
        bb = mb.get(name, {})
        ar = aa.get("spearman_ge3")
        br = bb.get("spearman_ge3")
        delta = (ar - br) if ar is not None and br is not None else None
        print(f"{name:20s} {fmt(ar):>12s} {fmt(br):>12s} {fmt(delta):>12s} {fmt(aa.get('mae')):>10s} {fmt(bb.get('mae')):>10s}")

    print("\nNominal attributes")
    print(f"{'Attribute':20s} {'A NLL':>12s} {'B NLL':>12s} {'A Acc':>12s} {'B Acc':>12s}")
    print("-" * 72)
    for name in ["internal_structure", "calcification"]:
        aa = ma.get(name, {})
        bb = mb.get(name, {})
        print(f"{name:20s} {fmt(aa.get('nll')):>12s} {fmt(bb.get('nll')):>12s} {fmt(aa.get('accuracy')):>12s} {fmt(bb.get('accuracy')):>12s}")

    print("\nNOTE: These are validation-development results. TEST must remain sealed until the final model is frozen.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--objective-a", required=True)
    p.add_argument("--objective-b", required=True)
    raise SystemExit(main(p.parse_args()))
