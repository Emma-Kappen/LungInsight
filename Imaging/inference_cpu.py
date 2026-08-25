"""
inference_cpu.py

Local CPU-only inference and explainability script for 3D CT patches or
full-volume scan inference.

The canonical model returns raw logits. This wrapper applies sigmoid exactly
once when exposing probabilities to callers.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

try:
    import pylidc as pl
except Exception:  # pragma: no cover - optional dependency in test stubs
    pl = None

try:
    from cir_multihead_pipeline import (
        FEATURE_NAMES,
        PATCH_SIZE,
        create_multihead_model,
        generate_characteristic_heatmaps,
    )
except Exception:  # pragma: no cover - exercised by lightweight import tests
    FEATURE_NAMES = [
        'spiculation', 'lobulation', 'calcification', 'margin',
        'texture', 'sphericity', 'subtlety', 'malignancy'
    ]
    PATCH_SIZE = 64
    create_multihead_model = None
    generate_characteristic_heatmaps = None

try:
    from detect_candidates_cpu import (
        detect_nodules_log,
        extract_patch,
        get_spacing_mm,
        load_volume_hu,
        segment_lungs,
    )
except ImportError:  # The canonical 01-05 pipeline replaces this old detector.
    detect_nodules_log = None
    extract_patch = None
    get_spacing_mm = None
    load_volume_hu = None
    segment_lungs = None


def load_patch(patch_path: str) -> torch.Tensor:
    if not os.path.isfile(patch_path):
        raise FileNotFoundError(f'Patch file not found: {patch_path}')
    patch = np.load(patch_path)
    if patch.ndim != 3 or patch.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(
            f'Input patch must be shape ({PATCH_SIZE},{PATCH_SIZE},{PATCH_SIZE}), got {patch.shape}'
        )
    return torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and not all(torch.is_tensor(v) for v in checkpoint.values()):
        for key in ('model_state_dict', 'state_dict', 'model'):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    state_dict = checkpoint
    model.load_state_dict(state_dict)
    return model


def save_candidate_results(output_dir: str, candidate_id: str, patch: np.ndarray,
                            probs: dict, heatmaps: dict, center_zyx=None):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, 'candidate_results_manifest.csv')
    result_path = os.path.join(output_dir, f'{candidate_id}_results.npz')

    payload = {
        'candidate_id': np.array(candidate_id, dtype=object),
        'patch': patch.astype(np.float32),
        'center_zyx': np.asarray(center_zyx, dtype=np.float32) if center_zyx is not None else np.array([np.nan, np.nan, np.nan], dtype=np.float32),
    }
    for head in FEATURE_NAMES:
        payload[f'{head}_score'] = np.asarray(probs[head], dtype=np.float32)
        if head in heatmaps:
            payload[f'{head}_heatmap'] = np.asarray(heatmaps[head], dtype=np.float32)
    np.savez_compressed(result_path, **payload)

    row = {'candidate_id': candidate_id, 'result_path': result_path}
    if center_zyx is not None:
        row['center_z'], row['center_y'], row['center_x'] = center_zyx
    for head in FEATURE_NAMES:
        row[f'{head}_score'] = probs[head]

    if os.path.isfile(manifest_path):
        manifest = pd.read_csv(manifest_path)
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    else:
        manifest = pd.DataFrame([row])
    manifest.to_csv(manifest_path, index=False)
    return result_path


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
        score = outputs[head]
        if score.dim() > 1 and score.size(1) == 1:
            score = score.squeeze(1)
        probs[head] = float(torch.sigmoid(score).cpu().item())

    heatmaps = generate_characteristic_heatmaps(model, x, device=device, target_layer=target_layer)
    out_path = os.path.splitext(patch_path)[0] + '_inference_results.npz'
    saved_path = save_candidate_results(
        os.path.dirname(out_path),
        os.path.splitext(os.path.basename(out_path))[0],
        x.squeeze(0).squeeze(0).cpu().numpy(),
        probs,
        heatmaps,
        center_zyx=(0.0, 0.0, 0.0),
    )
    return probs, heatmaps, saved_path


def run_scan_inference(scan, checkpoint_path: str, output_dir: str, target_layer: str = 'layer4'):
    if pl is None:
        raise RuntimeError('pylidc is required for full-scan inference')
    if any(fn is None for fn in (detect_nodules_log, extract_patch, get_spacing_mm, load_volume_hu, segment_lungs)):
        raise RuntimeError(
            'Legacy full-scan inference is unavailable because detect_candidates_cpu.py '
            'is not part of this repository. Use the canonical 01-05-06 pipeline.'
        )

    device = torch.device('cpu')
    volume_hu = load_volume_hu(scan)
    spacing_mm = get_spacing_mm(scan)
    lung_mask = segment_lungs(volume_hu)
    detections = detect_nodules_log(volume_hu, lung_mask, spacing_mm, verbose=False)

    model = create_multihead_model(device=device)
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    result_paths = []
    for idx, det in enumerate(detections):
        patch = extract_patch(volume_hu, det['centroid_zyx'], patch_size=PATCH_SIZE)
        if patch is None:
            continue
        patch_tensor = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(patch_tensor)
        probs = {}
        for head in FEATURE_NAMES:
            score = outputs[head]
            if score.dim() > 1 and score.size(1) == 1:
                score = score.squeeze(1)
            probs[head] = float(torch.sigmoid(score).cpu().item())
        heatmaps = generate_characteristic_heatmaps(model, patch_tensor, device=device, target_layer=target_layer)
        cand_id = f"{scan.patient_id}_cand{idx:03d}"
        result_path = save_candidate_results(
            output_dir,
            cand_id,
            patch,
            probs,
            heatmaps,
            center_zyx=det['centroid_zyx'],
        )
        result_paths.append(result_path)
    return result_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run CPU inference and save explainability heatmaps for one 3D patch or one full CT scan.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--patch', help='Path to the 3D .npy patch file')
    group.add_argument('--patient-id', help='pylidc patient identifier for full-volume scan inference')
    parser.add_argument('--checkpoint', required=True, help='Path to the trained model checkpoint file')
    parser.add_argument('--output-dir', default='inference_output', help='Directory for saved candidate results')
    parser.add_argument('--target-layer', default='layer4', help='Final convolutional layer for Grad-CAM++ heatmaps')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.patch:
        probs, heatmaps, out_path = inference_with_heatmaps(args.patch, args.checkpoint, args.target_layer)
        print('Inference confidence scores:')
        for head, prob in probs.items():
            print(f'  {head}: {prob:.4f}')
        print(f'Saved inference results to: {out_path}')
    else:
        if pl is None:
            raise RuntimeError('pylidc is required for full-scan inference')
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == args.patient_id).first()
        if scan is None:
            raise ValueError(f'No pylidc scan found for patient id {args.patient_id}')
        result_paths = run_scan_inference(scan, args.checkpoint, args.output_dir, args.target_layer)
        print(f'Saved {len(result_paths)} candidate result files to {args.output_dir}')


if __name__ == '__main__':
    main()