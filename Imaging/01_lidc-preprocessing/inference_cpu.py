"""
inference_cpu.py

Local CPU-only inference and explainability script for a single 3D patch.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGING_ROOT = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, IMAGING_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from path_setup import ensure_project_paths

ensure_project_paths(__file__)

import argparse
import os
import zipfile

import numpy as np
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from cir_multihead_pipeline import create_multihead_model, FEATURE_NAMES


class DictModelHeadTarget(ClassifierOutputTarget):
    def __init__(self, head_name: str):
        super().__init__(category=0)
        self.head_name = head_name

    def __call__(self, model_output):
        if not isinstance(model_output, dict):
            raise TypeError('Model output must be a dict of head_name->logits')
        if self.head_name not in model_output:
            raise ValueError(f'Head {self.head_name} not found in model output')
        out = model_output[self.head_name]
        if out.dim() > 1 and out.size(1) == 1:
            out = out.squeeze(1)
        return out


def load_patch(patch_path: str) -> torch.Tensor:
    if not os.path.isfile(patch_path):
        raise FileNotFoundError(f'Patch file not found: {patch_path}')
    patch = np.load(patch_path)
    if patch.ndim != 3 or patch.shape != (64, 64, 64):
        raise ValueError(f'Input patch must be shape (64,64,64), got {patch.shape}')
    return torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def inference_with_heatmaps(patch_path: str, checkpoint_path: str, target_layer: str = 'layer4'):
    device = torch.device('cpu')
    x = load_patch(patch_path).to(device)

    model = create_multihead_model(device=device)
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    with torch.no_grad():
        outputs = model(x)

    probs = {}
    for head in FEATURE_NAMES:
        logits = outputs[head]
        if logits.dim() > 1 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        probs[head] = float(torch.sigmoid(logits).cpu().item())

    heatmaps = {}
    target_layer_module = getattr(model, target_layer, None)
    if target_layer_module is None:
        raise AttributeError(f'Model does not contain layer {target_layer}')

    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer_module], use_cuda=False)
    for head in FEATURE_NAMES:
        target = DictModelHeadTarget(head)
        grayscale_cam = cam(x, targets=[target])
        if grayscale_cam.ndim == 4:
            heatmap = grayscale_cam[0]
        elif grayscale_cam.ndim == 3:
            heatmap = grayscale_cam
        else:
            raise RuntimeError(f'Unexpected heatmap shape: {grayscale_cam.shape}')
        if heatmap.shape != (64, 64, 64):
            if heatmap.shape[0] == 1 and heatmap.shape[1:] == (64, 64, 64):
                heatmap = heatmap[0]
            else:
                raise RuntimeError(f'Unsupported heatmap shape: {heatmap.shape}')
        heatmap = np.clip((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8), 0.0, 1.0)
        heatmaps[head] = heatmap.astype(np.float32)

    out_path = os.path.splitext(patch_path)[0] + '_inference_results.npz'
    np.savez_compressed(out_path, patch=x.cpu().numpy(), **{f'{head}_prob': probs[head] for head in FEATURE_NAMES}, **{f'{head}_heatmap': heatmaps[head] for head in FEATURE_NAMES})

    return probs, heatmaps, out_path


def parse_args():
    parser = argparse.ArgumentParser(description='Run CPU inference and save explainability heatmaps for one 3D patch.')
    parser.add_argument('--patch', required=True, help='Path to the 3D .npy patch file')
    parser.add_argument('--checkpoint', required=True, help='Path to the trained model checkpoint file')
    parser.add_argument('--target-layer', default='layer4', help='Final convolutional layer for GradCAMPlusPlus')
    return parser.parse_args()


def main():
    args = parse_args()
    probs, heatmaps, out_path = inference_with_heatmaps(args.patch, args.checkpoint, args.target_layer)
    print('Inference probabilities:')
    for head, prob in probs.items():
        print(f'  {head}: {prob:.4f}')
    print(f'Saved inference results to: {out_path}')


if __name__ == '__main__':
    main()
