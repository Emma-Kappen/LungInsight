"""
11_gradcam_sanity_check.py

GEOMETRY UNIT TEST for cir_multihead_pipeline.py's Grad-CAM++ implementation.

Purpose
-------
Answer the recurring debugging question "is an off-center CAM peak a real
geometry bug, or legitimate model attention?" -- WITHOUT needing a trained
checkpoint or a real patient volume.

How it works
------------
1. Build an untrained (or optionally checkpointed) multi-head model.
2. Plant a synthetic bright blob at a KNOWN voxel location in an otherwise
   blank 64^3 patch (a stand-in for "the thing the model should attend to").
3. Run generate_characteristic_heatmaps() and check that the CAM peak lands
   near the planted location.

Interpretation
---------------
Because Grad-CAM here targets backbone.layer2 (an 8^3 native activation
grid, upsampled x8 to 64^3), a peak within roughly one grid cell
(~8-14 voxels, i.e. sqrt(3)*8) of the planted blob is EXPECTED and means
the pipeline's geometry (axis order, layer resolution, upsampling,
per-head gradient isolation) is working correctly.

If, on real checkpoints/candidates, this same test script reports peaks
consistently tens of voxels off for a bright, unambiguous synthetic target,
THAT indicates a real geometry bug (e.g. an axis swap or stale-activation
issue) rather than model attention -- go back to cir_multihead_pipeline.py.
If it reports peaks close to the target (as it does against the untrained
model in this repo), then off-center peaks you see on real patient data are
model behavior, not a pipeline bug.

Usage
-----
python Imaging/11_gradcam_sanity_check.py
python Imaging/11_gradcam_sanity_check.py --checkpoint Imaging/checkpoints/best_model_gpu_v2.pth
"""

import argparse

import numpy as np
import torch

from cir_multihead_pipeline import (
    PATCH_SIZE,
    create_multihead_model,
    generate_characteristic_heatmaps,
)

try:
    from inference_cpu import load_checkpoint
except Exception:  # pragma: no cover
    load_checkpoint = None


# Native activation grid size for backbone.layer2 on a 64^3 input.
# (64 -> stem/2 -> 32 -> maxpool/2 -> 16 -> layer1(x1) -> 16 -> layer2/2 -> 8)
NATIVE_GRID = 8
GRID_CELL_VOXELS = PATCH_SIZE // NATIVE_GRID  # 8
# A peak within one diagonal grid cell is "resolution-limited", not a bug.
EXPECTED_MAX_DIST = GRID_CELL_VOXELS * (3 ** 0.5)  # ~13.9 voxels


def make_blob(center, patch_size=PATCH_SIZE, radius=4, value=400.0):
    x = torch.zeros(1, 1, patch_size, patch_size, patch_size)
    zz, yy, xx = np.ogrid[:patch_size, :patch_size, :patch_size]
    mask = (
        (zz - center[0]) ** 2
        + (yy - center[1]) ** 2
        + (xx - center[2]) ** 2
    ) <= radius * radius
    x[0, 0][torch.from_numpy(mask)] = value
    return x


def run_sanity_check(checkpoint_path=None, target_layer="backbone.layer2", seed=0):
    device = torch.device("cpu")
    torch.manual_seed(seed)

    model = create_multihead_model(device=device)
    if checkpoint_path:
        if load_checkpoint is None:
            raise RuntimeError("inference_cpu.load_checkpoint unavailable")
        model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    test_positions = [
        (16, 16, 16),
        (32, 32, 32),
        (48, 20, 40),
        (10, 50, 30),
    ]

    print("=" * 78)
    print("GRAD-CAM GEOMETRY SANITY CHECK")
    print(f"Target layer: {target_layer}  |  native grid: {NATIVE_GRID}^3  "
          f"|  cell size: {GRID_CELL_VOXELS} voxels  "
          f"|  pass threshold: <= {EXPECTED_MAX_DIST:.1f} voxels")
    print("=" * 78)

    all_pass = True
    any_unreliable = False

    for pos in test_positions:
        x = make_blob(pos)
        heatmaps, diagnostics = generate_characteristic_heatmaps(
            model, x, device=device, target_layer=target_layer,
            return_diagnostics=True,
        )

        print(f"\nPlanted blob at ZYX={pos}")
        for head, cam in heatmaps.items():
            peak = np.unravel_index(np.argmax(cam), cam.shape)
            dist = float(np.linalg.norm(np.array(peak) - np.array(pos)))
            reliable = diagnostics[head]["reliable"]
            active_pct = diagnostics[head]["active_cell_fraction"] * 100

            if not reliable:
                status = "UNRELIABLE"
                any_unreliable = True
                # A localization failure caused by gradient sparsity is a
                # DIFFERENT problem from a geometry bug -- don't count it
                # against the geometry pass/fail verdict.
            else:
                status = "PASS" if dist <= EXPECTED_MAX_DIST else "FAIL"
                all_pass &= (status == "PASS")

            print(f"  {head:14s} peak={peak}  dist={dist:5.1f}  "
                  f"active_grid={active_pct:5.1f}%  [{status}]")

    print("\n" + "=" * 78)
    if any_unreliable:
        print("RESULT: Grad-CAM GRADIENT RELIABILITY issue detected.")
        print("        One or more heads' gradients are concentrated in a")
        print("        tiny fraction of the target layer's spatial grid.")
        print("        Their CAM peaks are not trustworthy regardless of")
        print("        whether the pipeline's geometry code is correct --")
        print("        the argmax is being decided by numerical noise, not")
        print("        signal. This is DIFFERENT from a geometry bug: the")
        print("        axis handling / upsampling / per-head isolation may")
        print("        be completely correct and this can still happen.")
        print("        Try: a different --target-layer for the affected")
        print("        head(s), or inspect training for that head (dead")
        print("        units, near-constant predictions, etc).")
    elif all_pass:
        print("RESULT: Grad-CAM geometry PASSES sanity check.")
        print("        Off-center peaks on real patient data are model")
        print("        attention (or the layer's coarse native grid),")
        print("        not a pipeline geometry bug.")
    else:
        print("RESULT: Grad-CAM geometry FAILS sanity check.")
        print("        A synthetic, unambiguous target was not localized,")
        print("        AND gradients were reliable (not sparse). This points")
        print("        to a real geometry bug -- re-inspect")
        print("        cir_multihead_pipeline.py: axis order, target-layer")
        print("        resolution, per-head gradient isolation.")
    print("=" * 78)

    return all_pass and not any_unreliable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                         help="Optional checkpoint path. Without it, an "
                              "untrained model is used, which is sufficient "
                              "to validate GEOMETRY (not attention quality).")
    parser.add_argument("--target-layer", default="backbone.layer2")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    run_sanity_check(
        checkpoint_path=args.checkpoint,
        target_layer=args.target_layer,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
