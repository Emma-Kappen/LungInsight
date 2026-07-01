"""
cir_multihead_pipeline.py

Modular pipeline to:
- aggregate LIDC-IDRI nodules via pylidc and produce an extended CIR-style manifest
  with per-feature CONFIDENCE SCORES normalized to [0, 1] (not binary labels)
- extract 64x64x64 isotropic patches and save as .npy
- assemble a multi-head 3D SE-ResNet50 model (10 independent sigmoid confidence heads)
- generate per-head 3D Grad-CAM heatmaps for explainability

This is the single canonical dataset/model module for the project. Other
scripts (preprocess_cpu.py, inference_cpu.py, the training/validation
notebooks) import FEATURE_NAMES, LIDCPatchDataset, create_multihead_model,
and generate_characteristic_heatmaps from here.
"""
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

try:
    # pylidc 0.2.3 (latest on PyPI as of writing) calls the long-removed
    # configparser.SafeConfigParser internally (Scan.py:55), which raises
    # AttributeError on Python 3.12+. SafeConfigParser was a cosmetic alias
    # for ConfigParser, so restoring it as an attribute is a safe shim --
    # it does not change pylidc's behavior, only lets the import succeed.
    import configparser
    if not hasattr(configparser, 'SafeConfigParser'):
        configparser.SafeConfigParser = configparser.ConfigParser

    import pylidc as pl
except Exception:
    pl = None

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from se_resnet3d import se_resnet50_3d

FEATURE_NAMES = [
    'spiculation', 'lobulation', 'density', 'calcification', 'margin',
    'texture', 'sphericity', 'subtlety', 'internalStructure', 'malignancy'
]

# pylidc rating scales per feature: (min, max) as annotated.
# Most semantic features are rated 1-5. calcification and internalStructure
# are rated on pylidc's 1-6 scale.
FEATURE_RANGES = {feat: (1.0, 5.0) for feat in FEATURE_NAMES}
FEATURE_RANGES['calcification'] = (1.0, 6.0)
FEATURE_RANGES['internalStructure'] = (1.0, 6.0)

# Features where a LOWER raw rating corresponds to MORE benign / LESS
# malignant-suspicious appearance would need inversion after min-max scaling
# to keep "higher score = more malignant-suspicious" consistent. Plain
# min-max already satisfies that direction for every feature in this set
# (verified against pylidc's coding: calcification 1=popcorn/benign ...
# 6=absent/suspicious; internalStructure 1=soft tissue/benign ... higher
# values trend toward less-common/more-suspicious patterns), so no feature
# currently needs inversion. Kept as an explicit, empty set rather than
# hardcoded math so a future feature with reversed semantics can be added
# here without touching _normalize_feature.
INVERT_FEATURES = set()

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


def _normalize_feature(feature_name: str, raw_avg: float) -> float:
    """Map a raw averaged pylidc rating to a [0, 1] confidence score.

    Higher output always means "more pronounced / more malignant-suspicious",
    regardless of whether the underlying pylidc scale runs in the opposite
    direction for a given feature (see INVERT_FEATURES).
    """
    lo, hi = FEATURE_RANGES[feature_name]
    if np.isnan(raw_avg):
        return np.nan
    clipped = min(max(raw_avg, lo), hi)
    norm = (clipped - lo) / (hi - lo) if hi != lo else 0.0
    if feature_name in INVERT_FEATURES:
        norm = 1.0 - norm
    return float(norm)


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

            feat_scores = {}
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
                raw_avg = float(np.mean(vals)) if len(vals) > 0 else np.nan
                feat_scores[feat] = _normalize_feature(feat, raw_avg)

            try:
                ann = nodule_annotations[0]
                vol, mask = ann.uniform_cubic_resample(side_length=patch_size, verbose=False)
                if vol.shape != (patch_size, patch_size, patch_size):
                    if vol.ndim == 3:
                        z, y, x = vol.shape
                        if min(z, y, x) < patch_size:
                            raise ValueError(f'Patch too small: {vol.shape}')
                        cz, cy, cx = z // 2, y // 2, x // 2
                        vol = vol[cz - patch_size // 2: cz + patch_size // 2,
                                  cy - patch_size // 2: cy + patch_size // 2,
                                  cx - patch_size // 2: cx + patch_size // 2]
                # Guard against silent off-by-one crops (e.g. odd source dims).
                if vol.shape != (patch_size, patch_size, patch_size):
                    raise ValueError(
                        f'Cropped patch shape {vol.shape} != expected '
                        f'({patch_size}, {patch_size}, {patch_size})'
                    )
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
                # Continuous confidence score in [0, 1], NaN if no annotations
                # provided a rating for this feature.
                row[f'{feat}_score'] = feat_scores.get(feat, np.nan)
            rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(output_dir, manifest_name), index=False)
    return manifest


class LIDCPatchDataset(Dataset):
    """Canonical dataset for loading 3D patches and per-feature confidence
    score targets in [0, 1]. This is the single dataset class used across
    the project (preprocess_cpu.py and both training/validation notebooks
    import this rather than defining their own)."""

    def __init__(self, manifest_csv: str, transform=None, device: str = 'cpu'):
        if not os.path.isfile(manifest_csv):
            raise FileNotFoundError(f'Manifest file not found: {manifest_csv}')
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.device = torch.device(device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_path = row['file_path']
        if not os.path.isfile(patch_path):
            raise FileNotFoundError(f'Patch file not found: {patch_path}')

        vol = np.load(patch_path)
        if vol.ndim == 3:
            if vol.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
                raise ValueError(
                    f'Invalid patch shape {vol.shape} in {patch_path}, '
                    f'expected ({PATCH_SIZE}, {PATCH_SIZE}, {PATCH_SIZE})'
                )
            tensor = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0)
        elif vol.ndim == 4:
            tensor = torch.from_numpy(vol).permute(3, 0, 1, 2).float()
        else:
            raise ValueError(f'Unsupported volume ndim: {vol.ndim}')

        if self.transform is not None:
            tensor = self.transform(tensor)

        # Confidence-score regression targets in [0, 1], not class indices.
        targets = {
            feat: torch.tensor(float(row[f'{feat}_score']), dtype=torch.float32)
            for feat in FEATURE_NAMES
        }
        return tensor.to(self.device), targets


def create_multihead_model(head_names: Optional[List[str]] = None, device='cpu'):
    """Construct the multi-head 3D SE-ResNet50 model.

    Backbone: se_resnet3d.se_resnet50_3d, an independent 3D reimplementation
    of moskomule/senet.pytorch's se_resnet50 ([3, 4, 6, 3] SEBottleneck
    layout), inflated to 3D convolutions for single-channel HU volumes.

    Returns a model where model(x) -> dict[str, Tensor] mapping each name in
    head_names to a sigmoid-activated confidence score tensor of shape (N,)
    in [0, 1]. `model.layer4` is exposed for Grad-CAM hooking.
    """
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
        activations['value'] = out
        def _save_grad(grad):
            gradients['value'] = grad.detach()
        out.register_hook(_save_grad)

    if not hasattr(model, target_layer):
        raise ValueError(f'Model has no attribute {target_layer} to attach hooks')
    handle = getattr(model, target_layer).register_forward_hook(forward_hook)

    outputs = model(input_patch)
    heatmaps = {}
    head_items = list(outputs.items())

    try:
        for i, (name, out) in enumerate(head_items):
            # Clear stale activation/gradient state from the previous head so
            # a silently-failed hook can never cause us to reuse a prior
            # head's gradient instead of erroring out.
            gradients.pop('value', None)

            model.zero_grad()
            score = out.squeeze() if out.dim() > 1 else out
            scalar = score.sum()
            is_last = (i == len(head_items) - 1)
            scalar.backward(retain_graph=not is_last)

            act = activations.get('value')
            grad = gradients.get('value')
            if act is None or grad is None:
                raise RuntimeError(
                    f'Failed to capture activations or gradients for head "{name}" '
                    f'-- the {target_layer} hook may not be on the path to this head\'s output'
                )

            weights = torch.mean(grad, dim=(2, 3, 4), keepdim=True)
            gcam = F.relu(torch.sum(weights * act.detach(), dim=1, keepdim=True))
            up = F.interpolate(gcam, size=input_patch.shape[2:], mode='trilinear', align_corners=False)
            up_np = up.detach().cpu().numpy()
            bm = up_np[0, 0]
            bm_max, bm_min = np.max(bm), np.min(bm)
            bm = (bm - bm_min) / (bm_max - bm_min) if bm_max - bm_min > 0 else np.zeros_like(bm)
            heatmaps[name] = bm
    finally:
        handle.remove()

    return heatmaps


if __name__ == '__main__':
    device = 'cpu'
    model = create_multihead_model(head_names=FEATURE_NAMES, device=device)
    x = torch.randn(1, 1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, device=device)
    outs = model(x)
    print('Model forward ok. Heads:', list(outs.keys()))
    try:
        maps = generate_characteristic_heatmaps(model, x, device=device)
        print('Generated heatmaps for heads:', list(maps.keys()))
    except Exception as e:
        print('Heatmap generation failed:', e)