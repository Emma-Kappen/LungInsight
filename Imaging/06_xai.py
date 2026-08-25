"""
06_xai.py

STEP 6 (classification + explainability) in the pipeline:
    01_dicom_to_hu.py              -> DICOM -> HU volume
    02_mask_and_crop.py            -> lung segmentation + non-lung blanking + Z-crop
    03_visualize.py                -> viewing
    04_detect_candidates.py        -> nodule candidate detection (ViTDet3D)
    05_extract_candidate_patches.py -> crop 64^3 classifier patches
    06_xai.py                      <- this file: classify each patch with the
                                       multi-head SE-ResNet3D (se_resnet3d.py +
                                       best_model_gpu_v2.pth) and compute
                                       explainability maps
    07_visualize.py                -> overlay the explainability maps back onto
                                       the full CT volume with a slice slider

This intentionally builds the classifier straight from se_resnet3d.py rather
than a separate "cir_multihead_pipeline" helper module: best_model_gpu_v2.pth
is a plain state_dict for se_resnet3d.MultiHeadSEResNet3D, and its 8 head
names (spiculation, lobulation, calcification, margin, texture, sphericity,
subtlety, malignancy) are read directly out of the checkpoint's
'heads.<name>.weight' keys below, so this script can't silently drift out of
sync with whatever heads a given checkpoint was actually trained with.

The model returns raw logits (see se_resnet3d.py's MultiHeadSEResNet3D
docstring). Sigmoid is applied exactly once here when exposing probabilities.

Explainability metrics computed per candidate patch, per head:
    - confidence score:  sigmoid(logit)
    - Grad-CAM++ heatmap: localization from an intermediate conv block,
      upsampled to the full 64^3 patch. Answers "where in the patch did
      the model look."
    - vanilla gradient saliency: |d(logit)/d(input)|, normalized. Answers
      "which individual voxels most influenced the score," at full input
      resolution (finer-grained but noisier than Grad-CAM++).

A note on --target-layer and why it defaults to 'backbone.layer3', not 'layer4':
se_resnet3d.py's stem + layer1..layer4 downsample a 64^3 input by a total
factor of 32 (stem: /4, layer2: /2, layer3: /2, layer4: /2), so for THIS
patch size the per-layer spatial resolution is:
    layer1: 16^3 (4096 cells)   layer2: 8^3 (512 cells)
    layer3:  4^3   (64 cells)   layer4: 2^3    (8 cells)
layer4 -- the conventional Grad-CAM choice for 2D ImageNet-scale
backbones -- collapses to only 8 spatial cells here. Trilinearly
upsampling 8 corner values to a 64^3 volume cannot produce a localized
blob; it can only produce a smooth gradient smeared across the ENTIRE
patch, which looks like the model is "highlighting" the whole square
regardless of where (if anywhere) it actually focused. 'layer3' (64
cells) is the default instead: still deep/semantic, but with enough
spatial resolution for the upsampled CAM to actually localize something
within the patch rather than just interpolating 8 corners. Pass
--target-layer backbone.layer2 for even finer (but less semantic)
localization. ('layer4' alone, without the 'backbone.' prefix, also still
works -- MultiHeadSEResNet3D aliases it directly, see se_resnet3d.py --
but layer1-3 are only reachable via the 'backbone.' prefix.)

Both maps come from ONE forward pass per patch (gradients for both are
pulled with torch.autograd.grad against the same cached activations/logits,
looping only over heads -- not re-running the backbone per head, per map).

Usage:
    python 06_xai.py output/LIDC-IDRI-0001_patches \
        --checkpoint checkpoints/best_model_gpu_v2.pth \
        --out-dir output/LIDC-IDRI-0001_xai

Outputs (written to --out-dir):
    <candidate_id>_xai.npz -> patch, center_zyx, and per head:
        {head}_score (float), {head}_gradcam (patch_size^3 float32,
        [0,1]), {head}_saliency (patch_size^3 float32, [0,1])
    xai_manifest.csv -> candidate_id, xai_result_path, patch_path,
        center_z/y/x (carried from patch_manifest.csv), plus one
        {head}_score column per head for quick triage/sorting
    meta.json -> checkpoint path, resolved head names, target layer,
        number of candidates processed
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from se_resnet3d import se_resnet50_3d

PATCH_SIZE = 64
EPS = 1e-8


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def resolve_head_names(state_dict: dict):
    """Read head names straight out of a state_dict's 'heads.<name>.weight'
    keys, so the model we construct always matches what the checkpoint was
    actually trained with (rather than trusting a hardcoded list that could
    drift if the checkpoint changes).
    """
    names = sorted({
        k.split('.')[1] for k in state_dict.keys()
        if k.startswith('heads.') and k.endswith('.weight')
    })
    if not names:
        raise ValueError(
            "Could not find any 'heads.<name>.weight' keys in the checkpoint "
            "-- is this really a se_resnet3d.MultiHeadSEResNet3D state_dict?"
        )
    return names


def load_classifier(checkpoint_path: str, device: torch.device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Some checkpoints wrap the state_dict in a training-harness dict
    # (e.g. {'model_state_dict': ..., 'epoch': ..., 'optimizer_state_dict': ...}).
    # best_model_gpu_v2.pth is a raw state_dict, but handle the wrapped case
    # too so this script keeps working if that ever changes.
    if isinstance(checkpoint, dict) and not all(torch.is_tensor(v) for v in checkpoint.values()):
        for key in ('model_state_dict', 'state_dict', 'model'):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break

    head_names = resolve_head_names(checkpoint)
    model = se_resnet50_3d(in_channels=1, head_names=head_names)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, head_names


def resolve_target_layer(model: torch.nn.Module, target_layer: str):
    """Resolve a dotted attribute path (default 'layer4') on the model.
    se_resnet3d.py's MultiHeadSEResNet3D exposes layer4 directly as a plain
    attribute (see its docstring) specifically for this kind of Grad-CAM hook.
    """
    module = model
    for part in target_layer.split('.'):
        if not hasattr(module, part):
            raise AttributeError(
                f"Model has no attribute '{part}' while resolving target "
                f"layer '{target_layer}'."
            )
        module = getattr(module, part)
    return module


# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------

def compute_gradcampp(activation: torch.Tensor, grad_act: torch.Tensor, output_size):
    """3D Grad-CAM++ for a single (1,C,D,H,W) activation/gradient pair.

    Standard Grad-CAM++ weighting (Chattopadhyay et al.), extended from 2D
    spatial sums to 3D spatial sums:
        alpha = grad^2 / (2*grad^2 + sum_spatial(A)*grad^3 + eps)
        weights_c = sum_spatial(alpha_c * relu(grad_c))
        cam = relu(sum_c weights_c * A_c)
    then min-max normalized to [0,1] and trilinearly upsampled to
    output_size (the original patch's spatial size).
    """
    grad2 = grad_act.pow(2)
    grad3 = grad_act.pow(3)
    sum_act = activation.sum(dim=(2, 3, 4), keepdim=True)
    denom = 2 * grad2 + sum_act * grad3
    denom = torch.where(denom != 0, denom, torch.full_like(denom, EPS))
    alpha = grad2 / denom
    weights = (alpha * F.relu(grad_act)).sum(dim=(2, 3, 4), keepdim=True)  # (1,C,1,1,1)

    cam = F.relu((weights * activation).sum(dim=1, keepdim=True))  # (1,1,d,h,w)
    cam = F.interpolate(cam, size=output_size, mode='trilinear', align_corners=False)
    cam = cam.squeeze(0).squeeze(0)

    cam_min, cam_max = cam.min(), cam.max()
    if (cam_max - cam_min).item() > EPS:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)
    return cam.detach().cpu().numpy().astype(np.float32)


def compute_saliency(grad_x: torch.Tensor):
    """Vanilla gradient saliency: |d(logit)/d(input)|, min-max normalized
    to [0,1]. grad_x is (1,1,D,H,W).
    """
    sal = grad_x.abs().squeeze(0).squeeze(0)
    sal_min, sal_max = sal.min(), sal.max()
    if (sal_max - sal_min).item() > EPS:
        sal = (sal - sal_min) / (sal_max - sal_min)
    else:
        sal = torch.zeros_like(sal)
    return sal.detach().cpu().numpy().astype(np.float32)


def explain_patch(model: torch.nn.Module, x: torch.Tensor, target_layer_module,
                   head_names, compute_saliency_maps: bool = True):
    """Single forward pass, then one torch.autograd.grad call per head
    (retaining the graph) to pull both the Grad-CAM++ activation gradient
    and the input-level saliency gradient without re-running the backbone.

    Returns (scores, gradcams, saliencies) dicts keyed by head name.
    """
    activation_holder = {}

    def _capture(_module, _inp, out):
        activation_holder['act'] = out

    handle = target_layer_module.register_forward_hook(_capture)
    x = x.clone().requires_grad_(True)
    outputs = model(x)
    handle.remove()
    activation = activation_holder['act']

    scores, gradcams, saliencies = {}, {}, {}
    output_size = tuple(x.shape[2:])
    n_heads = len(head_names)

    for i, head in enumerate(head_names):
        logit = outputs[head]
        if logit.dim() > 1 and logit.size(1) == 1:
            logit = logit.squeeze(1)
        scores[head] = float(torch.sigmoid(logit).detach().cpu().item())

        is_last = (i == n_heads - 1) and not compute_saliency_maps
        grad_targets = [activation, x] if compute_saliency_maps else [activation]
        grads = torch.autograd.grad(
            logit, grad_targets, retain_graph=not is_last, create_graph=False,
        )
        grad_act = grads[0]
        gradcams[head] = compute_gradcampp(activation.detach(), grad_act, output_size)
        if compute_saliency_maps:
            saliencies[head] = compute_saliency(grads[1])

    return scores, gradcams, saliencies


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def load_patch(patch_path: str) -> torch.Tensor:
    if not os.path.isfile(patch_path):
        raise FileNotFoundError(f'Patch file not found: {patch_path}')
    patch = np.load(patch_path)
    if patch.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(
            f'Expected a ({PATCH_SIZE},{PATCH_SIZE},{PATCH_SIZE}) patch, got {patch.shape}'
        )
    return torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def save_xai_result(out_dir: str, candidate_id: str, patch: np.ndarray,
                     center_zyx, scores: dict, gradcams: dict, saliencies: dict):
    result_path = os.path.join(out_dir, f'{candidate_id}_xai.npz')
    payload = {
        'candidate_id': np.array(candidate_id, dtype=object),
        'patch': patch.astype(np.float32),
        'center_zyx': np.asarray(center_zyx, dtype=np.float32),
    }
    for head, score in scores.items():
        payload[f'{head}_score'] = np.float32(score)
        payload[f'{head}_gradcam'] = gradcams[head]
        if head in saliencies:
            payload[f'{head}_saliency'] = saliencies[head]
    np.savez_compressed(result_path, **payload)
    return result_path


def run_xai(patches_dir: str, checkpoint_path: str, out_dir: str,
            target_layer: str = 'backbone.layer3', device_str: str = 'cpu',
            heads_filter=None, skip_saliency: bool = False):
    manifest_path = os.path.join(patches_dir, 'patch_manifest.csv')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"'{manifest_path}' not found. Run 05_extract_candidate_patches.py "
            f"first, and pass its --out-dir here as patches_dir."
        )
    patch_manifest = pd.read_csv(manifest_path)

    device = torch.device(device_str)
    print(f"[info] Loading classifier checkpoint from '{checkpoint_path}'...")
    model, head_names = load_classifier(checkpoint_path, device)
    if heads_filter:
        unknown = set(heads_filter) - set(head_names)
        if unknown:
            raise ValueError(
                f"--heads requested {sorted(unknown)}, which are not among "
                f"this checkpoint's heads: {head_names}"
            )
        head_names = [h for h in head_names if h in heads_filter]
    print(f"[info] Heads: {head_names}")

    target_layer_module = resolve_target_layer(model, target_layer)

    os.makedirs(out_dir, exist_ok=True)
    manifest_rows = []
    for _, row in patch_manifest.iterrows():
        candidate_id = row['candidate_id']
        patch_path = row['patch_path']
        x = load_patch(patch_path).to(device)
        patch_np = x.squeeze(0).squeeze(0).detach().cpu().numpy()

        scores, gradcams, saliencies = explain_patch(
            model, x, target_layer_module, head_names,
            compute_saliency_maps=not skip_saliency,
        )

        center_zyx = (row.get('center_z', np.nan), row.get('center_y', np.nan), row.get('center_x', np.nan))
        result_path = save_xai_result(
            out_dir, candidate_id, patch_np, center_zyx, scores, gradcams, saliencies,
        )

        manifest_row = row.to_dict()
        manifest_row['xai_result_path'] = os.path.abspath(result_path)
        for head, score in scores.items():
            manifest_row[f'{head}_score'] = score
        manifest_rows.append(manifest_row)
        print(f"[info] {candidate_id}: " +
              ", ".join(f"{h}={scores[h]:.3f}" for h in head_names))

    manifest_df = pd.DataFrame(manifest_rows)
    xai_manifest_path = os.path.join(out_dir, 'xai_manifest.csv')
    manifest_df.to_csv(xai_manifest_path, index=False)

    run_meta = {
        'source_patches_dir': os.path.abspath(patches_dir),
        'checkpoint_path': os.path.abspath(checkpoint_path),
        'head_names': head_names,
        'target_layer': target_layer,
        'saliency_computed': not skip_saliency,
        'num_candidates': len(manifest_rows),
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(run_meta, f, indent=2)

    print(f"[done] Wrote {len(manifest_rows)} XAI result(s) + xai_manifest.csv -> '{out_dir}'")
    print("[done] Next: run 07_visualize.py to overlay these heatmaps on the "
          "full CT volume with a slice slider.")
    return manifest_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="STEP 6: Classify each candidate patch with the multi-head "
        "SE-ResNet3D and compute Grad-CAM++ / saliency explainability maps."
    )
    parser.add_argument(
        'patches_dir',
        help="Directory containing patch_manifest.csv (the --out-dir from "
        "05_extract_candidate_patches.py).",
    )
    parser.add_argument(
        '--checkpoint', required=True,
        help="Path to the CLASSIFIER checkpoint (se_resnet3d.py architecture, "
        "e.g. checkpoints/best_model_gpu_v2.pth). This is NOT the detector "
        "checkpoint used by 04_detect_candidates.py.",
    )
    parser.add_argument(
        '--out-dir', default=None,
        help="Directory to write <candidate_id>_xai.npz + xai_manifest.csv "
        "(default: '<patches_dir>_xai').",
    )
    parser.add_argument(
        '--target-layer', default='backbone.layer3',
        help="Dotted attribute path to the conv layer Grad-CAM++ hooks into "
        "(default: 'backbone.layer3' -- 4^3=64 spatial cells for a 64^3 "
        "patch. 'layer4' is only 2^3=8 cells here and upsamples to a "
        "meaningless smooth gradient across the whole patch rather than a "
        "localized blob; 'backbone.layer2' (8^3=512 cells) gives finer "
        "localization at the cost of a less semantic layer. See the module "
        "docstring.",
    )
    parser.add_argument(
        '--heads', default=None,
        help="Comma-separated subset of head names to compute (default: all "
        "heads present in the checkpoint). E.g. 'malignancy,spiculation'.",
    )
    parser.add_argument(
        '--skip-saliency', action='store_true',
        help="Skip the vanilla-gradient saliency map and only compute "
        "Grad-CAM++ (faster; saliency maps roughly double the backward "
        "passes per candidate).",
    )
    parser.add_argument(
        '--device', default='cpu', choices=['cpu', 'cuda'],
        help="Device to run the classifier on (default: cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.patches_dir.rstrip('/\\') + '_xai')
    heads_filter = [h.strip() for h in args.heads.split(',')] if args.heads else None
    run_xai(
        args.patches_dir, args.checkpoint, out_dir,
        target_layer=args.target_layer, device_str=args.device,
        heads_filter=heads_filter, skip_saliency=args.skip_saliency,
    )


if __name__ == '__main__':
    main()