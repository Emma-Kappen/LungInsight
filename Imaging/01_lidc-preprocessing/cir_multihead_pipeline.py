"""
cir_multihead_pipeline.py

Modular pipeline to:
- aggregate LIDC-IDRI nodules via pylidc and produce an extended CIR-style manifest
- extract 64x64x64 isotropic patches and save as .npy
- assemble a multi-head 3D SE-ResNet50 model (10 independent sigmoid heads)
- generate per-head 3D Grad-CAM heatmaps for explainability

Author: Generated
"""
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

try:
    import pylidc as pl
except Exception as e:
    pl = None

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Import the 3D SE-ResNet implementation from se_resnet3d.py.
#
# NOTE: this used to be `from Imaging.se_resnet3d import se_resnet50_3d`,
# which assumes Imaging/ is itself an importable package from a parent
# directory on sys.path (e.g. LungInsight/ on the path, Imaging/ containing
# an __init__.py). That's not how this project's path_setup.py works: it
# adds the Imaging/ directory ITSELF to sys.path (see
# ensure_project_paths()), which makes se_resnet3d importable as a
# top-level module -- `Imaging.se_resnet3d` can never resolve that way, so
# the import silently failed and se_resnet50_3d was always None (masked by
# the bare `except Exception` below, which is why the resulting
# create_multihead_model() error message pointed at "the local senet
# package" -- a red herring from an earlier design, not the real cause).
try:
    from se_resnet3d import se_resnet50_3d
except Exception as _se_resnet3d_import_error:
    se_resnet50_3d = None
    print(f"[warn] Could not import se_resnet50_3d from se_resnet3d.py: "
          f"{_se_resnet3d_import_error!r}")

# --- Configuration ---
FEATURE_NAMES = [
    'spiculation', 'lobulation', 'density', 'calcification', 'margin',
    'texture', 'sphericity', 'subtlety', 'internalStructure', 'malignancy'
]

BINARIZE_THRESHOLD = 3.0
PATCH_SIZE = 64


def _get_feature_value(annotation, feature_name: str):
    # pylidc annotations may use different attribute casing; handle both
    if hasattr(annotation, feature_name):
        return getattr(annotation, feature_name)
    low = feature_name.lower()
    if hasattr(annotation, low):
        return getattr(annotation, low)
    # fallback: feature might be inside annotation.feature_vals()
    try:
        fv = annotation.feature_vals()
        return fv.get(feature_name) or fv.get(low)
    except Exception:
        return None


def _to_hu(volume: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    """Convert resampled volume to Hounsfield Units using slope/intercept."""
    vol = volume.astype(np.float64)
    if slope != 1:
        vol = vol * slope
    vol = vol + intercept
    return vol.astype(np.int16)


def aggregate_and_extract(output_dir: str,
                          patch_size: int = PATCH_SIZE,
                          manifest_name: str = 'extended_cir_manifest.csv',
                          scans: Optional[List] = None) -> pd.DataFrame:
    """
    Query pylidc (or use provided `scans`) to extract 64^3 patches and build a manifest.

    Returns a pandas DataFrame and writes `manifest_name` in `output_dir`.
    """
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
        # fallback to helper from pylidc scans
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

            # compute average per-feature score and binarize
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
                if len(vals) > 0:
                    avg = float(np.mean(vals))
                else:
                    avg = np.nan
                feat_vals[feat] = 1 if (not np.isnan(avg) and avg >= BINARIZE_THRESHOLD) else 0

            # attempt to extract patch (pick first annotation as center)
            try:
                ann = nodule_annotations[0]
                vol, mask = ann.uniform_cubic_resample(side_length=patch_size, verbose=False)
                if vol.shape != (patch_size, patch_size, patch_size):
                    # Some pylidc versions return shape (D,H,W) but with different order; validate
                    if vol.ndim == 3:
                        # attempt to pad/crop to required shape
                        z, y, x = vol.shape
                        if min(z, y, x) < patch_size:
                            raise ValueError(f'Patch too small: {vol.shape}')
                        # center-crop as fallback
                        cz, cy, cx = z // 2, y // 2, x // 2
                        vol = vol[cz - patch_size//2: cz + patch_size//2,
                                  cy - patch_size//2: cy + patch_size//2,
                                  cx - patch_size//2: cx + patch_size//2]
                vol = _to_hu(vol, intercept, slope)
                np.save(file_path, vol)
            except Exception as e:
                # skip nodules that cannot be extracted safely
                print(f"Skipping {nodule_uid} extraction: {e}")
                continue

            # compute centroid if available
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
            # add binary labels
            for feat in FEATURE_NAMES:
                row[f'{feat}_label'] = int(feat_vals.get(feat, 0))

            rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(output_dir, manifest_name), index=False)
    return manifest


class LIDCPatchDataset(Dataset):
    """PyTorch Dataset that yields (tensor, targets_dict) from the manifest.

    Each volume is returned as a torch.FloatTensor with shape (1, D, H, W).
    Targets are returned as a dict mapping feature->torch.LongTensor(0/1).
    """
    def __init__(self, manifest_csv: str, transform=None, device='cpu'):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vol = np.load(row['file_path'])
        # ensure shape (1,D,H,W)
        if vol.ndim == 3:
            vol = vol.astype(np.float32)
            tensor = torch.from_numpy(vol).unsqueeze(0)
        elif vol.ndim == 4:
            tensor = torch.from_numpy(vol).permute(3, 0, 1, 2).float()
        else:
            raise ValueError(f'Unsupported volume ndim: {vol.ndim}')

        if self.transform is not None:
            tensor = self.transform(tensor)

        # collect binary targets
        targets = {feat: torch.tensor(int(row[f'{feat}_label']), dtype=torch.long)
                   for feat in FEATURE_NAMES}

        return tensor.to(self.device), targets


def create_multihead_model(head_names: Optional[List[str]] = None, device='cpu'):
    """Instantiate a 3D SE-ResNet50 with independent sigmoid heads for each characteristic."""
    if se_resnet50_3d is None:
        raise RuntimeError('se_resnet50_3d not available; ensure the local senet package is importable')
    head_names = head_names or FEATURE_NAMES
    model = se_resnet50_3d(in_channels=1, head_names=head_names)
    return model.to(device)


def generate_characteristic_heatmaps(model: torch.nn.Module,
                                     input_patch: torch.Tensor,
                                     device: str = 'cpu',
                                     target_layer: str = 'layer4') -> Dict[str, np.ndarray]:
    """
    Generate a 3D Grad-CAM heatmap (64x64x64) for each characteristic head.

    - `model` must be the 3D SE-ResNet50 instance with named heads
    - `input_patch` tensor shape: (1,1,64,64,64) or (B,1,64,64,64)
    Returns a dict mapping head_name -> heatmap numpy array (D,H,W) normalized 0-1.
    """
    model.eval()
    device = torch.device(device)
    input_patch = input_patch.to(device)

    activations = {}
    gradients = {}

    # forward hook to capture activations from the target layer (module output)
    def forward_hook(module, inp, out):
        activations['value'] = out.detach()

        # register hook on the activation tensor to capture gradients during backward
        def _save_grad(grad):
            gradients['value'] = grad.detach()

        out.register_hook(_save_grad)

    # attach hook
    if not hasattr(model, target_layer):
        raise ValueError(f'Model has no attribute {target_layer} to attach hooks')
    handle = getattr(model, target_layer).register_forward_hook(forward_hook)

    # forward pass
    outputs = model(input_patch)

    heatmaps = {}

    # ensure heads order deterministic
    head_items = list(outputs.items())

    for name, out in head_items:
        model.zero_grad()
        # out shape: (B,) or (B,1); reduce to scalar(s) per batch
        if out.dim() > 1:
            score = out.squeeze()
        else:
            score = out

        # for batch processing, sum the batch scores so backward works for batch>1
        scalar = score.sum()
        scalar.backward(retain_graph=True)

        # now we have activations['value'] and gradients['value']
        act = activations.get('value')  # shape (B, C, d, h, w)
        grad = gradients.get('value')  # shape (B, C, d, h, w)
        if act is None or grad is None:
            raise RuntimeError('Failed to capture activations or gradients for Grad-CAM')

        # global-average-pool the gradients to obtain channel-wise weights
        weights = torch.mean(grad, dim=(2, 3, 4), keepdim=True)  # shape (B, C, 1,1,1)

        # weighted combination of activations
        gcam = F.relu(torch.sum(weights * act, dim=1, keepdim=True))  # shape (B,1,d,h,w)

        # upsample to input spatial size
        up = F.interpolate(gcam, size=input_patch.shape[2:], mode='trilinear', align_corners=False)
        # normalize per sample
        up_np = up.detach().cpu().numpy()
        # if batch >1, return a heatmap per sample by stacking; here we return first sample heatmap
        bm = up_np[0, 0]
        # normalize to 0-1
        if np.max(bm) - np.min(bm) > 0:
            bm = (bm - np.min(bm)) / (np.max(bm) - np.min(bm))
        else:
            bm = np.zeros_like(bm)
        heatmaps[name] = bm

    # remove hook
    handle.remove()
    return heatmaps


if __name__ == '__main__':
    # Quick self-test (dry run) — does not require pylidc or real data.
    device = 'cpu'
    model = None
    try:
        model = create_multihead_model(head_names=FEATURE_NAMES, device=device)
    except Exception:
        print('Model creation skipped (senet not importable).')

    # synthesize a dummy input and run heatmap generator if model present
    if model is not None:
        x = torch.randn(1, 1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, device=device)
        outs = model(x)
        print('Model forward ok. Heads:', list(outs.keys()))
        try:
            maps = generate_characteristic_heatmaps(model, x, device=device)
            print('Generated heatmaps for heads:', list(maps.keys()))
        except Exception as e:
            print('Heatmap generation failed:', e)
            