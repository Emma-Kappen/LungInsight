"""
12_gradcam_border_artifact_check.py

Follow-up diagnostic to 11_gradcam_sanity_check.py.

11_gradcam_sanity_check.py showed that against the REAL checkpoint,
several heads' Grad-CAM "active_grid" fraction is small (~6-11%) AND
suspiciously CONSTANT across completely different synthetic blob
positions. A signal that's genuinely tracking where the blob is should
have its active cells MOVE as the blob moves. A signal that's actually
dominated by a fixed artifact (e.g. the network having learned to use
zero-padding at the patch border as an implicit, content-independent
positional cue -- a documented phenomenon in padded CNNs) will have
active cells that DON'T move, regardless of blob position.

This script makes that distinction directly: run the same head at
several different blob positions and check whether the "active"
activation-grid cells (from cir_multihead_pipeline's gradient
reliability diagnostic) are the same set every time.

Usage
-----
python Imaging/12_gradcam_border_artifact_check.py \
    --checkpoint Imaging/checkpoints/best_model_gpu_v2.pth \
    --target-layer backbone.layer2
"""

import argparse

import numpy as np
import torch

from cir_multihead_pipeline import (
    FEATURE_NAMES,
    PATCH_SIZE,
    _GradCAMCapture,
    _resolve_layer,
    create_multihead_model,
)

try:
    from inference_cpu import load_checkpoint
except Exception:  # pragma: no cover
    load_checkpoint = None


def make_blob(center, patch_size=PATCH_SIZE, radius=4, background=-800.0,
              delta=120.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = background + 60.0 * torch.randn(1, 1, patch_size, patch_size,
                                         patch_size, generator=g)
    zz, yy, xx = np.ogrid[:patch_size, :patch_size, :patch_size]
    mask = (
        (zz - center[0]) ** 2
        + (yy - center[1]) ** 2
        + (xx - center[2]) ** 2
    ) <= radius * radius
    x[0, 0][torch.from_numpy(mask)] += delta
    return x


def grid_coords_near_border(coord, grid_size=8, margin=1):
    return any(c < margin or c >= grid_size - margin for c in coord)


def run(checkpoint_path, target_layer="backbone.layer2", seed=0, top_k=16):
    device = torch.device("cpu")
    torch.manual_seed(seed)

    model = create_multihead_model(device=device)
    if checkpoint_path:
        if load_checkpoint is None:
            raise RuntimeError("inference_cpu.load_checkpoint unavailable")
        model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    positions = [
        (16, 16, 16),
        (32, 32, 32),
        (48, 20, 40),
        (10, 50, 30),
        (55, 55, 10),
    ]

    print("=" * 78)
    print("GRAD-CAM BORDER-ARTIFACT CHECK")
    print(f"Target layer: {target_layer}")
    print("For each head: does the set of 'active' activation-grid cells")
    print("MOVE when the blob moves (real signal), or stay FIXED (a")
    print("content-independent artifact -- most likely zero-padding at")
    print("the patch border acting as a positional shortcut)?")
    print("=" * 78)

    layer = _resolve_layer(model, target_layer)

    per_head_coord_sets = {head: [] for head in FEATURE_NAMES}

    for pos in positions:
        x = make_blob(pos, seed=seed)

        capture = _GradCAMCapture()
        fh = layer.register_forward_hook(capture.forward_hook)
        bh = layer.register_full_backward_hook(capture.backward_hook)

        model.zero_grad(set_to_none=True)
        outputs = model(x)  # ONE forward per position, reused for all heads

        for head_idx, head in enumerate(FEATURE_NAMES):
            model.zero_grad(set_to_none=True)
            retain = head_idx < len(FEATURE_NAMES) - 1
            outputs[head].sum().backward(retain_graph=retain)

            grad = capture.gradients.detach()
            cell_norm = grad[0].norm(dim=0)  # (d, h, w)
            flat = cell_norm.reshape(-1)

            k = min(top_k, flat.numel())
            top_indices = torch.topk(flat, k).indices.tolist()
            grid_shape = cell_norm.shape
            top_coords = frozenset(
                tuple(int(c) for c in np.unravel_index(idx, grid_shape))
                for idx in top_indices
            )
            per_head_coord_sets[head].append(top_coords)

        fh.remove()
        bh.remove()
        del outputs
        import gc
        gc.collect()

    print()
    print(f"top_k = {top_k} cells (out of {8*8*8} total) per test")
    print()
    print(f"{'head':14s} {'intersection':>13s} {'stable?':>9s}  border_pct")
    print("-" * 78)

    for head in FEATURE_NAMES:
        sets = per_head_coord_sets[head]
        intersection = set.intersection(*[set(s) for s in sets]) if sets else set()

        # "Stable" = a meaningful chunk of each test's top-K cells recur
        # in EVERY differently-positioned blob's top-K -- i.e. the same
        # handful of cells win regardless of where the real signal is.
        stable_fraction = len(intersection) / top_k if top_k else 0.0
        is_stable = stable_fraction >= 0.3

        border_count = sum(
            1 for c in intersection if grid_coords_near_border(c)
        )
        border_pct = (
            100.0 * border_count / len(intersection) if intersection else 0.0
        )

        flag = "FIXED (!)" if is_stable else "moves ok"

        print(f"{head:14s} {len(intersection):13d} {flag:>9s}  {border_pct:5.1f}%")

        if is_stable:
            print(
                f"    -> {len(intersection)}/{top_k} of the top gradient "
                f"cells are IDENTICAL across every blob position tested, "
                f"regardless of where the blob actually was. "
                f"{border_pct:.0f}% of those recurring cells sit at the "
                f"grid border (index 0 or 7 on some axis). This head's "
                f"Grad-CAM at '{target_layer}' is not reliably tracking "
                f"input content."
            )
            print(f"    Recurring cells: {sorted(intersection)}")

    print()
    print("=" * 78)
    print("Interpretation:")
    print("  'FIXED (!)' + high border_pct  -> the head is very likely")
    print("      exploiting zero-padding as a content-independent")
    print("      positional shortcut. Its Grad-CAM peak/geometry_status")
    print("      should NOT be trusted at this target layer. This is a")
    print("      property of the TRAINED WEIGHTS, not of")
    print("      cir_multihead_pipeline.py's implementation -- the fix is")
    print("      either a different --target-layer for this head, or")
    print("      addressing it during training/architecture (e.g. replicate")
    print("      padding instead of zero padding, or an anti-aliased/")
    print("      padding-free downsampling path).")
    print("  'moves ok'                     -> this head's CAM genuinely")
    print("      tracks input content at this layer; its geometry verdicts")
    print("      can be trusted.")
    print("=" * 78)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--target-layer", default="backbone.layer2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=16,
                         help="Number of highest-gradient activation-grid "
                              "cells to compare across blob positions "
                              "(default: 16, i.e. top ~3%% of the 8^3 grid).")
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.checkpoint, target_layer=args.target_layer, seed=args.seed,
        top_k=args.top_k)


if __name__ == "__main__":
    main()
