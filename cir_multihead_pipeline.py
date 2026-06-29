"""
cir_multihead_pipeline.py

Modular pipeline to:
- aggregate LIDC-IDRI nodules via pylidc and produce an extended CIR-style manifest
- extract 64x64x64 isotropic patches and save as .npy
- assemble a multi-head 3D SE-ResNet50 model (10 independent sigmoid heads)
- generate per-head 3D Grad-CAM heatmaps for explainability
"""
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

try:
    import pylidc as pl
except Exception:
    pl = None

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from senet import se_resnet50_3d
except Exception:
    se_resnet50_3d = None

FEATURE_NAMES = [
    'spiculation', 'lobulation', 'density', 'calcification', 'margin',
    'texture', 'sphericity', 'subtlety', 'internalStructure', 'malignancy'
]

BINARIZE_THRESHOLD = 3.0
PATCH_SIZE = 64


def _get_feature_value(annotation, feature_name: str):
    if hasattr(annotation, feature_name):
        return getattr(annotation, feature_name)
    low = feature_name.lower()
    if hasattr(annotation, low):
        return getattr(annotation, low)
    try:
        fv = annotation.feature_vals()
        return fv.get(feature_name) or fv.get(low)
    except Exception:
        return None


def _to_hu(volume: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    vol = volume.astype(np.float64)
    if slope != 1:
        vol = vol * slope
    vol = vol + intercept
    return vol.astype(np.int16)


def aggregate_and_extract(output_dir: str,
                          patch_size: int = PATCH_SIZE,
                          manifest_name: str = 'extended_cir_manifest.csv',
                          scans: Optional[List] = None) -> pd.DataFrame:
    if pl is None:
        raise RuntimeError('pylidc is required but not available in the environment')

    os.makedirs(output_dir, exist_ok=True)
    if scans is None:
        scans = pl.query(pl.Scan).all()

    rows = []
    for scan in scans:
        patient_id = scan.patient_id
        intercept = getattr(scan, 'RescaleIntercept', None)
        slope = getattr(scan, 'RescaleSlope', None)
        try:
            imgs = scan.load_all_dicom_images(verbose=False)
            slice0 = imgs[0]
            intercept = slice0.RescaleIntercept if intercept is None else intercept
            slope = slice0.RescaleSlope if slope is None else slope
        except Exception:
            intercept = 0.0 if intercept is None else intercept
            slope = 1.0 if slope is None else slope

        clusters = scan.cluster_annotations()
        for n_idx, nodule_annotations in enumerate(clusters, start=1):
            if not isinstance(nodule_annotations, list):
                nodule_annotations = [nodule_annotations]

            nodule_uid = f"{patient_id}_{n_idx:03d}"
            filename = f"nodule_{nodule_uid}.npy"
            file_path = os.path.join(output_dir, filename)

            feat_vals = {}
            for feat in FEATURE_NAMES:
                vals = []
                for ann in nodule_annotations:
                    v = _get_feature_value(ann, feat)
                    if v is None:
                        continue
                    try:
                        vals.append(float(v))
                    except Exception:
                        continue
                avg = float(np.mean(vals)) if len(vals) > 0 else np.nan
                feat_vals[feat] = 1 if (not np.isnan(avg) and avg >= BINARIZE_THRESHOLD) else 0

            try:
                ann = nodule_annotations[0]
                vol, mask = ann.uniform_cubic_resample(side_length=patch_size, verbose=False)
                if vol.shape != (patch_size, patch_size, patch_size):
                    if vol.ndim == 3:
                        z, y, x = vol.shape
                        if min(z, y, x) < patch_size:
                            raise ValueError(f'Patch too small: {vol.shape}')
                        cz, cy, cx = z // 2, y // 2, x // 2
                        vol = vol[cz - patch_size//2: cz + patch_size//2,
                                  cy - patch_size//2: cy + patch_size//2,
                                  cx - patch_size//2: cx + patch_size//2]
                vol = _to_hu(vol, intercept, slope)
                np.save(file_path, vol)
            except Exception as e:
                print(f"Skipping {nodule_uid} extraction: {e}")
                continue

            try:
                center = getattr(ann, 'centroid', None)
                if callable(center):
                    center = center()
                if center is None:
                    center = (np.nan, np.nan, np.nan)
            except Exception:
                center = (np.nan, np.nan, np.nan)

            row = {
                'nodule_id': nodule_uid,
                'file_path': os.path.abspath(file_path),
                'patient_id': patient_id,
                'center_x': center[0],
                'center_y': center[1],
                'center_z': center[2],
            }
            for feat in FEATURE_NAMES:
                row[f'{feat}_label'] = int(feat_vals.get(feat, 0))
            rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(output_dir, manifest_name), index=False)
    return manifest


class LIDCPatchDataset(Dataset):
    def __init__(self, manifest_csv: str, transform=None, device='cpu'):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vol = np.load(row['file_path'])
        if vol.ndim == 3:
            vol = vol.astype(np.float32)
            tensor = torch.from_numpy(vol).unsqueeze(0)
        elif vol.ndim == 4:
            tensor = torch.from_numpy(vol).permute(3, 0, 1, 2).float()
        else:
            raise ValueError(f'Unsupported volume ndim: {vol.ndim}')

        if self.transform is not None:
            tensor = self.transform(tensor)

        targets = {feat: torch.tensor(int(row[f'{feat}_label']), dtype=torch.long)
                   for feat in FEATURE_NAMES}
        return tensor.to(self.device), targets


def create_multihead_model(head_names: Optional[List[str]] = None, device='cpu'):
    if se_resnet50_3d is None:
        raise RuntimeError('se_resnet50_3d not available; ensure the local senet package is importable')
    head_names = head_names or FEATURE_NAMES
    model = se_resnet50_3d(in_channels=1, head_names=head_names)
    return model.to(device)


def generate_characteristic_heatmaps(model: torch.nn.Module,
                                     input_patch: torch.Tensor,
                                     device: str = 'cpu',
                                     target_layer: str = 'layer4') -> Dict[str, np.ndarray]:
    model.eval()
    device = torch.device(device)
    input_patch = input_patch.to(device)

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out.detach()
        def _save_grad(grad):
            gradients['value'] = grad.detach()
        out.register_hook(_save_grad)

    if not hasattr(model, target_layer):
        raise ValueError(f'Model has no attribute {target_layer} to attach hooks')
    handle = getattr(model, target_layer).register_forward_hook(forward_hook)

    outputs = model(input_patch)
    heatmaps = {}
    head_items = list(outputs.items())

    for name, out in head_items:
        model.zero_grad()
        score = out.squeeze() if out.dim() > 1 else out
        scalar = score.sum()
        scalar.backward(retain_graph=True)

        act = activations.get('value')
        grad = gradients.get('value')
        if act is None or grad is None:
            raise RuntimeError('Failed to capture activations or gradients for Grad-CAM')

        weights = torch.mean(grad, dim=(2, 3, 4), keepdim=True)
        gcam = F.relu(torch.sum(weights * act, dim=1, keepdim=True))
        up = F.interpolate(gcam, size=input_patch.shape[2:], mode='trilinear', align_corners=False)
        up_np = up.detach().cpu().numpy()
        bm = up_np[0, 0]
        bm = (bm - np.min(bm)) / (np.max(bm) - np.min(bm)) if np.max(bm) - np.min(bm) > 0 else np.zeros_like(bm)
        heatmaps[name] = bm

    handle.remove()
    return heatmaps


if __name__ == '__main__':
    device = 'cpu'
    model = None
    try:
        model = create_multihead_model(head_names=FEATURE_NAMES, device=device)
    except Exception:
        print('Model creation skipped (senet not importable).')

    if model is not None:
        x = torch.randn(1, 1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, device=device)
        outs = model(x)
        print('Model forward ok. Heads:', list(outs.keys()))
        try:
            maps = generate_characteristic_heatmaps(model, x, device=device)
            print('Generated heatmaps for heads:', list(maps.keys()))
        except Exception as e:
            print('Heatmap generation failed:', e)
