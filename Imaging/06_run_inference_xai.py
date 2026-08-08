"""
06_run_inference_xai.py

STEP 6: bridges the candidate pipeline to the trained classifier, and
produces a ranked, Grad-CAM-explained shortlist of likely-malignant
nodules:

    01_dicom_to_hu.py           -> DICOM -> HU volume
    02_mask_and_crop.py         -> lung segmentation + non-lung blanking + Z-crop
    03_visualize.py             -> viewing
    04_detect_and_patch.py      -> multi-scale 3D LoG candidate detection
    05_shape_filter_and_grow.py -> reject tubular candidates, grow +
                                    tightly crop each real nodule
    06_run_inference_xai.py     <- this file: run the TRAINED classifier
                                    (cir_multihead_pipeline.py /
                                    inference_cpu.py) on every surviving
                                    nodule patch, generate per-head
                                    Grad-CAM heatmaps, and rank
                                    candidates by malignancy score

=== Why this file exists (the actual integration gap, CONFIRMED) ===

01-05 and cir_multihead_pipeline.py/inference_cpu.py were built to two
different specs and don't share a module boundary. Concretely:

  * inference_cpu.py has an UNCONDITIONAL, module-level
    `from detect_candidates_cpu import (...)` (detect_nodules_log,
    extract_patch, get_spacing_mm, load_volume_hu, segment_lungs).
    detect_candidates_cpu.py does not exist anywhere in this project.
    This was verified directly: `import inference_cpu` raises
    ModuleNotFoundError immediately, before any of ITS OWN code runs
    -- so inference_cpu.py currently cannot be used at all, via either
    its --patch or --patient-id path, independent of anything else in
    this pipeline.
  * Even if detect_candidates_cpu.py were created, it would just be a
    second, less thorough reimplementation of what 01-05 already do
    (02's lung segmentation alone is considerably more careful than a
    single `segment_lungs` call name suggests -- see that file's own
    docstring: 3D-connectivity organ identity, a dedicated heart mask,
    size-limited hole filling, etc.)
  * 05's nodule patches are exactly the right INPUT for the
    classifier -- shape-filtered, size-adaptively grown, real
    (unblanked) HU -- there's no reason to detect candidates twice.

So this script imports create_multihead_model and
generate_characteristic_heatmaps from cir_multihead_pipeline.py
directly (that file's own imports are all fine), but does NOT import
from inference_cpu.py -- save_candidate_results is reimplemented here
verbatim instead (it has no real dependency on detect_candidates_cpu,
only on os/numpy/pandas), specifically to sidestep inference_cpu.py's
broken top-level import rather than requiring you to first create a
throwaway detect_candidates_cpu.py stub just to unblock imports. If
inference_cpu.py's import gets fixed/guarded later, that duplication
can be removed.

=== Known compatibility requirements (verify these before trusting output) ===

  1. PATCH SIZE: cir_multihead_pipeline.py hardcodes PATCH_SIZE = 64
     (64x64x64 patches). 05_shape_filter_and_grow.py's own default is
     --patch-shape 32. THESE MUST MATCH or inference_cpu.load_patch's
     shape check will reject every patch. Run 05 with
     `--patch-shape 64` before this script, or pass a different
     PATCH_SIZE-consistent checkpoint. This script checks patch shape
     per-candidate and raises immediately with this exact fix if it's
     wrong, rather than silently resampling around the mismatch (that
     would feed the classifier a scale it was never trained on).
  2. VALUE CONVENTION: verified consistent -- 05's patches are raw
     float32 HU (see 05's own docstring: "growth and shape analysis
     ... read from volume_hu ... never from volume_hu_masked"), and
     cir_multihead_pipeline.LIDCPatchDataset / inference_cpu.load_patch
     both load patches with no additional normalization beyond a
     float32 cast. No windowing/rescaling mismatch between what 05
     produces and what the classifier expects.
  3. CHECKPOINT FORMAT: inference_cpu.load_checkpoint assumes the .pth
     file IS the raw model state_dict. Many training setups (including
     typical Colab/GradNorm checkpointing, which usually saves the
     optimizer state and per-task loss weights alongside the model)
     instead save a WRAPPER dict, e.g.
     {'model_state_dict': ..., 'optimizer_state_dict': ..., 'epoch': ...}.
     This script tries the raw state_dict first (matching
     inference_cpu.load_checkpoint's own assumption exactly) and falls
     back to unwrapping common wrapper keys
     ('model_state_dict'/'state_dict'/'model') if the first attempt
     fails, printing which path was used either way so you can confirm
     it loaded correctly rather than silently guessing.
  4. GRAD-CAM TARGET LAYER: CONFIRMED against the real architecture
     (se_resnet3d.py, confirmed as the authoritative se_resnet50_3d --
     see note below). MultiHeadSEResNet3D explicitly aliases
     `self.layer4 = self.backbone.layer4` in its own __init__,
     specifically so `getattr(model, 'layer4')` (the default
     --target-layer) resolves without reaching into `.backbone` --
     the file's own docstring says as much. --target-layer's default
     of 'layer4' is correct out of the box for this model; the
     fallback warning below (printing named_modules()) is kept only
     as a safety net in case a different checkpoint/architecture is
     ever swapped in later.
  5. SIGMOID CONVENTION: CONFIRMED -- se_resnet3d.py's
     MultiHeadSEResNet3D.forward() applies `torch.sigmoid(logit)` to
     every head before returning, so outputs are genuine [0,1]
     probabilities. This matches inference_cpu.py's own documented
     assumption exactly ("does NOT apply an additional sigmoid").
     (An earlier version of this project also contained a SECOND,
     incompatible se_resnet50_3d inside a senet/se_resnet.py that
     returned raw un-sigmoided logits -- se_resnet3d.py has since been
     confirmed as the one actually in use, so that ambiguity is
     resolved. --raw-logits below is kept as a manual override, in
     case a future checkpoint swap reintroduces a raw-logit model, but
     is NOT needed for the current se_resnet3d.py-based checkpoint.)
  6. COMPUTE COST: generate_characteristic_heatmaps does ONE backward
     pass PER HEAD (10 heads here) on top of the forward pass, per
     candidate, on CPU. That's normal for Grad-CAM (each head needs
     its own gradient w.r.t. the shared trunk activation) but means
     cost scales with (num_kept_candidates x num_heads) backward
     passes -- fine for the typically small number of candidates that
     survive 05's shape filter per scan, worth knowing if you batch
     this across many patients.

=== What this script outputs ===

For every candidate 05 marked kept=True (i.e. survived the shape
filter and has a patch):
  - runs the classifier -> a [0,1] confidence score per head
    (FEATURE_NAMES from cir_multihead_pipeline.py, includes
    'malignancy')
  - generates a Grad-CAM heatmap per head (if the target layer attaches
    successfully)
  - saves both via inference_cpu.save_candidate_results (one .npz per
    candidate + a running candidate_results_manifest.csv), unchanged
    from inference_cpu.py's own format

Then, specific to finding likely-cancerous nodules:
  - ranked_candidates.csv: every scored candidate, one row each, with
    every head's score plus voxel location and 05's own
    equivalent_diameter_mm/sphericity, SORTED by --malignancy-head
    score descending
  - reports/rankNN_candidate_XXXX.png: for the top --top-k candidates
    (by malignancy score) that have a heatmap, a 2x3 panel of axial/
    coronal/sagittal center slices through the grown patch, plain CT
    on top and the malignancy-head Grad-CAM overlaid (red/yellow =
    higher contribution to the malignancy score) on the bottom -- a
    fast visual sanity check of WHERE the model is actually looking,
    not just what score it produced.

Usage:
    python 06_run_inference_xai.py \
        output/LIDC-IDRI-0001_masked \
        output/LIDC-IDRI-0001_candidates \
        output/LIDC-IDRI-0001_candidates_nodules \
        --checkpoint checkpoints/multihead_best.pth \
        --out-dir output/LIDC-IDRI-0001_xai \
        --target-layer layer4 --malignancy-head malignancy --top-k 10

    # If Grad-CAM warns the target layer doesn't exist, find the real
    # name first:
    python -c "
    import torch
    from cir_multihead_pipeline import create_multihead_model
    m = create_multihead_model(device='cpu')
    for name, _ in m.named_modules():
        print(name)
    "
"""

import argparse
import csv
import json
import os
import sys

# --- Make cir_multihead_pipeline.py (and everything IT imports, e.g.
# se_resnet3d.py) findable, regardless of the caller's cwd. ---
#
# Uses path_setup.py (lives alongside this script in Imaging/), which
# walks up from this file's own location to find the Imaging/ directory
# by name, then adds it plus its known subdirectories to sys.path. More
# robust than hardcoding paths relative to this file, since it keeps
# working even if this script itself moves within the project.
from path_setup import ensure_project_paths
ensure_project_paths(__file__)

import numpy as np

try:
    import torch
except ImportError:
    print(
        "torch is required. Install it with:\n"
        "    pip install torch --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from cir_multihead_pipeline import (
        FEATURE_NAMES,
        PATCH_SIZE,
        create_multihead_model,
        generate_characteristic_heatmaps,
    )
except ImportError as e:
    print(
        "Could not import cir_multihead_pipeline.py -- make sure it (and "
        "the local `senet` package it depends on for se_resnet50_3d) is "
        f"on PYTHONPATH / in the current directory.\nOriginal error: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

# NOTE: inference_cpu.save_candidate_results is intentionally NOT
# imported here. inference_cpu.py has an unconditional, module-level
# `from detect_candidates_cpu import (...)` -- and detect_candidates_cpu.py
# does not exist anywhere in this project. That means inference_cpu.py
# currently cannot be imported AT ALL (confirmed directly: `import
# inference_cpu` raises ModuleNotFoundError before any of its own code
# runs), regardless of whether you use its --patch or --patient-id path.
# save_candidate_results itself has no real dependency on
# detect_candidates_cpu (it only uses os/numpy/pandas), so it's
# reimplemented here verbatim rather than importing a module that
# can't currently load. Once detect_candidates_cpu.py either gets
# created or that import gets removed/guarded in inference_cpu.py, you
# can switch this back to a real import if you'd rather keep one
# implementation.


def save_candidate_results(output_dir: str, candidate_id: str, patch: np.ndarray,
                            probs: dict, heatmaps: dict, center_zyx=None):
    """Verbatim copy of inference_cpu.py's save_candidate_results -- see
    the note above for why this isn't a cross-file import."""
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "candidate_results_manifest.csv")
    result_path = os.path.join(output_dir, f"{candidate_id}_results.npz")

    payload = {
        "candidate_id": np.array(candidate_id, dtype=object),
        "patch": patch.astype(np.float32),
        "center_zyx": np.asarray(center_zyx, dtype=np.float32) if center_zyx is not None
                      else np.array([np.nan, np.nan, np.nan], dtype=np.float32),
    }
    for head in FEATURE_NAMES:
        payload[f"{head}_score"] = np.asarray(probs[head], dtype=np.float32)
        payload[f"{head}_heatmap"] = np.asarray(heatmaps[head], dtype=np.float32)
    np.savez_compressed(result_path, **payload)

    row = {"candidate_id": candidate_id, "result_path": result_path}
    if center_zyx is not None:
        row["center_z"], row["center_y"], row["center_x"] = center_zyx
    for head in FEATURE_NAMES:
        row[f"{head}_score"] = probs[head]

    if os.path.isfile(manifest_path):
        manifest = pd.read_csv(manifest_path)
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    else:
        manifest = pd.DataFrame([row])
    manifest.to_csv(manifest_path, index=False)
    return result_path


def _unwrap_state_dict(raw, checkpoint_path: str):
    """Given whatever torch.load() returned, return (state_dict, description).
    A real state_dict's values are all tensors; a wrapper dict's top-level
    values are typically a mix (nested state_dict, optimizer state, an int
    epoch, etc.) -- that's what distinguishes the two cases here, no
    trial-and-error load attempt needed."""
    if isinstance(raw, dict):
        if raw and all(torch.is_tensor(v) for v in raw.values()):
            return raw, "raw state_dict"
        for key in ("model_state_dict", "state_dict", "model"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key], f"wrapper dict, raw['{key}']"
        raise RuntimeError(
            f"Checkpoint at '{checkpoint_path}' is a dict but doesn't look "
            f"like a raw state_dict (not all top-level values are tensors), "
            f"and none of the wrapper keys 'model_state_dict'/'state_dict'/"
            f"'model' were found either. Top-level keys: {list(raw.keys())}."
        )
    raise RuntimeError(
        f"Checkpoint at '{checkpoint_path}' is not a dict at all "
        f"(got {type(raw)}) -- can't load it."
    )


def load_state_dict_from_checkpoint(checkpoint_path: str, device: "torch.device"):
    """Load and unwrap a checkpoint file into a plain state_dict."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location=device)
    state_dict, description = _unwrap_state_dict(raw, checkpoint_path)
    print(f"[info] Checkpoint loaded ({description}).")
    return state_dict


def infer_head_names_from_state_dict(state_dict, checkpoint_path: str):
    """
    Read the ACTUAL head names this checkpoint was trained with, straight
    off its own 'heads.<name>.weight' keys -- rather than trusting
    cir_multihead_pipeline.FEATURE_NAMES, which is a hand-maintained
    constant that can (and, for the real checkpoint this script was
    developed against, DID) drift out of sync with what a given .pth
    actually contains. This is what makes the model's head list
    self-consistent with whatever checkpoint you point --checkpoint at,
    including future retrains with a different head set, with no code
    change needed here.
    """
    import re
    names = [m.group(1) for k in state_dict.keys()
             if (m := re.match(r"^heads\.([^.]+)\.weight$", k))]
    if not names:
        sample = list(state_dict.keys())[:10]
        raise RuntimeError(
            f"Could not find any 'heads.<name>.weight' keys in "
            f"'{checkpoint_path}' -- can't infer this checkpoint's head "
            f"names. First few keys found: {sample}"
        )
    return names




def load_pipeline_candidates(candidates_dir: str):
    """Load candidates.json from 04_detect_and_patch.py -- needed for
    voxel/world coordinates, since 05's nodules.json only stores
    candidate_id (a positional index into THIS list), not coordinates
    directly."""
    path = os.path.join(candidates_dir, "candidates.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Pass 04_detect_and_patch.py's --out-dir "
            f"as candidates_dir."
        )
    with open(path) as f:
        data = json.load(f)
    return data["candidates"]


def load_nodules(nodules_dir: str):
    """Load nodules.json from 05_shape_filter_and_grow.py."""
    path = os.path.join(nodules_dir, "nodules.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Pass 05_shape_filter_and_grow.py's "
            f"--out-dir as nodules_dir."
        )
    with open(path) as f:
        data = json.load(f)
    return data["nodules"], data.get("params", {})


def run_inference_xai(
    masked_dir: str,
    candidates_dir: str,
    nodules_dir: str,
    checkpoint_path: str,
    out_dir: str,
    target_layer: str = "layer4",
    malignancy_head: str = "malignancy",
    top_k: int = 10,
    device_str: str = "cpu",
    raw_logits: bool = False,
):
    global FEATURE_NAMES  # reassigned below from the checkpoint's own heads

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device_str)

    print(f"[info] Loading meta.json from '{masked_dir}'...")
    with open(os.path.join(masked_dir, "meta.json")) as f:
        meta = json.load(f)
    patient_id = meta.get("patient_id", "unknown")

    print(f"[info] Loading candidates from '{candidates_dir}'...")
    candidates = load_pipeline_candidates(candidates_dir)

    print(f"[info] Loading shape-filtered/grown nodules from '{nodules_dir}'...")
    nodules, grow_params = load_nodules(nodules_dir)
    kept = [n for n in nodules if n.get("kept") and n.get("patch_file")]
    print(f"[info] {len(nodules)} candidates evaluated by 05, {len(kept)} "
          f"kept with a patch to classify.")

    if kept:
        expected_patch_shape = grow_params.get("patch_shape")
        if expected_patch_shape is not None and expected_patch_shape != PATCH_SIZE:
            raise ValueError(
                f"05_shape_filter_and_grow.py was run with --patch-shape "
                f"{expected_patch_shape}, but the classifier requires "
                f"PATCH_SIZE={PATCH_SIZE}. Re-run 05 with "
                f"--patch-shape {PATCH_SIZE}, then re-run this script."
            )

    print(f"[info] Reading checkpoint from '{checkpoint_path}'...")
    state_dict = load_state_dict_from_checkpoint(checkpoint_path, device)
    checkpoint_head_names = infer_head_names_from_state_dict(state_dict, checkpoint_path)
    if set(checkpoint_head_names) != set(FEATURE_NAMES):
        print(
            f"[info] This checkpoint's heads do not match "
            f"cir_multihead_pipeline.FEATURE_NAMES -- that constant reflects "
            f"an earlier/aspirational head list, not necessarily this "
            f"checkpoint. Using the checkpoint's own {len(checkpoint_head_names)} "
            f"heads instead: {checkpoint_head_names}"
        )
    FEATURE_NAMES = checkpoint_head_names

    if malignancy_head not in FEATURE_NAMES:
        raise ValueError(
            f"--malignancy-head '{malignancy_head}' is not one of this "
            f"checkpoint's heads {FEATURE_NAMES}. Pass the exact head name "
            f"your checkpoint was trained with."
        )

    print(f"[info] Building model with heads {FEATURE_NAMES}...")
    model = create_multihead_model(head_names=FEATURE_NAMES, device=device_str)
    model.load_state_dict(state_dict)
    print("[info] Checkpoint weights loaded into model (all keys matched).")
    model.eval()

    heatmap_ok = hasattr(model, target_layer)
    if not heatmap_ok:
        print(f"\n[warn] model has no attribute '{target_layer}' -- Grad-CAM "
              f"heatmaps will be SKIPPED for every candidate (classification "
              f"scores are unaffected). This model's actual module names:")
        for name, _ in model.named_modules():
            if name:
                print(f"    {name}")
        print(f"[warn] Find the last convolutional block above and re-run "
              f"with --target-layer <that name>.\n")

    results = []
    out_of_range_seen = []
    for nodule_row in kept:
        candidate_id = nodule_row["candidate_id"]
        cand = candidates[candidate_id]  # positional match: 05 enumerated
                                          # 04's candidates list in order

        patch_path = os.path.join(nodules_dir, nodule_row["patch_file"])
        patch = np.load(patch_path)
        if patch.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(
                f"Patch '{patch_path}' has shape {patch.shape}, but the "
                f"classifier requires ({PATCH_SIZE},{PATCH_SIZE},{PATCH_SIZE}). "
                f"Re-run 05_shape_filter_and_grow.py with "
                f"--patch-shape {PATCH_SIZE}."
            )

        x = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(x)
        probs = {}
        for head in FEATURE_NAMES:
            score = outputs[head]
            if score.dim() > 1 and score.size(1) == 1:
                score = score.squeeze(1)
            value = float(score.cpu().item())
            if raw_logits:
                value = 1.0 / (1.0 + np.exp(-value))
            else:
                out_of_range_seen.append(value < 0.0 or value > 1.0)
            probs[head] = value

        heatmaps = {}
        result_path = None
        if heatmap_ok:
            try:
                heatmaps = generate_characteristic_heatmaps(
                    model, x, device=device_str, target_layer=target_layer
                )
            except Exception as e:
                print(f"[warn] Grad-CAM failed for candidate {candidate_id}: {e}")
                heatmaps = {}

        center_zyx = (int(cand["voxel_z"]), int(cand["voxel_y"]), int(cand["voxel_x"]))
        cand_str_id = f"{patient_id}_nodule_{candidate_id:04d}"
        if heatmaps:
            result_path = save_candidate_results(
                out_dir, cand_str_id, patch, probs, heatmaps, center_zyx=center_zyx
            )

        results.append({
            "candidate_id": candidate_id,
            "voxel_z": center_zyx[0], "voxel_y": center_zyx[1], "voxel_x": center_zyx[2],
            "equivalent_diameter_mm": nodule_row.get("equivalent_diameter_mm"),
            "sphericity": nodule_row.get("sphericity"),
            "bbox_used": nodule_row.get("bbox_used"),
            **{f"{h}_score": probs[h] for h in FEATURE_NAMES},
            "has_heatmap": bool(heatmaps),
            "result_path": result_path,
        })

        if len(results) % 10 == 0 or len(results) == len(kept):
            print(f"[info] Classified {len(results)}/{len(kept)} candidates...")

    if not raw_logits and any(out_of_range_seen):
        print(
            "\n[WARNING] At least one head score fell OUTSIDE [0,1]. "
            "Sigmoid output is mathematically guaranteed to lie in [0,1], "
            "so this is conclusive: the loaded model is NOT already "
            "applying sigmoid (despite se_resnet3d.py's own forward() "
            "doing so -- double check load_checkpoint_robust actually "
            "loaded the expected architecture/weights). Scores in "
            "ranked_candidates.csv below are RAW, not probabilities. "
            "Re-run with --raw-logits to get correct [0,1] probabilities.\n"
        )

    results.sort(key=lambda r: r[f"{malignancy_head}_score"], reverse=True)

    ranked_csv = os.path.join(out_dir, "ranked_candidates.csv")
    if results:
        fieldnames = [k for k in results[0].keys() if k != "result_path"]
        with open(ranked_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in fieldnames})

    print(f"\n[done] Scored {len(results)} candidates.")
    if results:
        top = results[0]
        print(f"[done] Ranked by '{malignancy_head}' descending -> '{ranked_csv}'")
        print(f"[done] Top candidate: id={top['candidate_id']} "
              f"{malignancy_head}={top[f'{malignancy_head}_score']:.3f} "
              f"diameter~{top['equivalent_diameter_mm']}mm at voxel "
              f"(z={top['voxel_z']}, y={top['voxel_y']}, x={top['voxel_x']})")
    else:
        print("[info] No candidates survived to classify -- nothing to rank. "
              "Check 05's kept/rejected counts.")

    render_top_k_report(results, out_dir, top_k, malignancy_head)
    return results


def render_top_k_report(results, out_dir: str, top_k: int, malignancy_head: str):
    """
    For the top_k candidates BY MALIGNANCY SCORE that have a saved
    heatmap, render a 2x3 panel (axial/coronal/sagittal center slices,
    plain CT on top row and malignancy Grad-CAM overlaid on the bottom
    row) so you can see WHERE the model is looking, not just the
    number it produced -- a fast faithfulness sanity check (e.g. a
    high malignancy score with a heatmap centered on the patch edge or
    on an obvious vessel, rather than the grown nodule region, is a
    sign to distrust that score).
    """
    to_render = [r for r in results if r.get("result_path")][:top_k]
    if not to_render:
        print("[info] No heatmaps available to render (target layer didn't "
              "attach, or no candidates survived).")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available -- skipping report images "
              "(scores/heatmaps are still saved in the .npz files).")
        return

    reports_dir = os.path.join(out_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Rough lung-window-style display stretch for the raw HU patch:
    # centered near typical soft-tissue/nodule density, wide enough to
    # show the nodule against surrounding parenchyma without extra
    # windowing machinery pulled in from 03_visualize.py.
    def to_display(hu_slice):
        return np.clip((hu_slice + 1000.0) / 1400.0, 0.0, 1.0)

    for rank, r in enumerate(to_render, start=1):
        data = np.load(r["result_path"])
        patch = data["patch"]
        heatmap = data[f"{malignancy_head}_heatmap"]
        c = patch.shape[0] // 2

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        planes = [
            ("axial", patch[c, :, :], heatmap[c, :, :]),
            ("coronal", patch[:, c, :], heatmap[:, c, :]),
            ("sagittal", patch[:, :, c], heatmap[:, :, c]),
        ]
        for col, (name, img, hm) in enumerate(planes):
            disp = to_display(img)
            axes[0, col].imshow(disp, cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(f"{name} (CT)")
            axes[0, col].axis("off")

            axes[1, col].imshow(disp, cmap="gray", vmin=0, vmax=1)
            axes[1, col].imshow(hm, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            axes[1, col].set_title(f"{name} + Grad-CAM")
            axes[1, col].axis("off")

        malignancy_score = r[f"{malignancy_head}_score"]
        fig.suptitle(
            f"Rank {rank}: candidate {r['candidate_id']} -- "
            f"{malignancy_head}={malignancy_score:.3f}, "
            f"~{r.get('equivalent_diameter_mm')}mm, "
            f"voxel (z={r['voxel_z']}, y={r['voxel_y']}, x={r['voxel_x']})"
        )
        fig.tight_layout()
        out_path = os.path.join(reports_dir, f"rank{rank:02d}_candidate_{r['candidate_id']:04d}.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"[done] Wrote {len(to_render)} ranked XAI report image(s) -> '{reports_dir}'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="STEP 6: run the trained multi-head classifier + "
        "Grad-CAM on 05's nodule patches, and rank candidates by "
        "malignancy score."
    )
    parser.add_argument("masked_dir", help="02_mask_and_crop.py --out-dir (for meta.json).")
    parser.add_argument("candidates_dir", help="04_detect_and_patch.py --out-dir (for candidate coordinates).")
    parser.add_argument("nodules_dir", help="05_shape_filter_and_grow.py --out-dir (for nodules.json + patches/).")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: '<nodules_dir>_xai').")
    parser.add_argument("--target-layer", default="layer4", help="Model attribute name for Grad-CAM hooks (default: 'layer4').")
    parser.add_argument("--malignancy-head", default="malignancy", help="Head name to rank candidates by (default: 'malignancy').")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top-ranked candidates to render XAI report images for (default: 10).")
    parser.add_argument("--device", default="cpu", help="torch device string (default: 'cpu', matching inference_cpu.py).")
    parser.add_argument(
        "--raw-logits", action="store_true",
        help="Manually apply sigmoid to every head's output before treating "
        "it as a probability. NOT needed for the current se_resnet3d.py-"
        "based checkpoint (its MultiHeadSEResNet3D already applies sigmoid "
        "internally) -- kept as an override in case a future checkpoint "
        "swap reintroduces a raw-logit model (default: False).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.nodules_dir.rstrip("/\\") + "_xai")
    run_inference_xai(
        args.masked_dir, args.candidates_dir, args.nodules_dir,
        args.checkpoint, out_dir,
        target_layer=args.target_layer,
        malignancy_head=args.malignancy_head,
        top_k=args.top_k,
        device_str=args.device,
        raw_logits=args.raw_logits,
    )


if __name__ == "__main__":
    main()