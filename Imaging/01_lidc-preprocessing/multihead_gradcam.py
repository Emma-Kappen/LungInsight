"""
multihead_gradcam.py

3D Grad-CAM++ wrapper for multi-head dictionary-output models.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGING_ROOT = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, IMAGING_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from path_setup import ensure_project_paths

ensure_project_paths(__file__)

import numpy as np
import torch

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError as exc:
    raise ImportError(
        'pytorch_grad_cam is required for multi-head GradCAM; install jacobgil/pytorch-grad-cam'
    ) from exc

from cir_multihead_pipeline import FEATURE_NAMES


class DictModelHeadTarget(ClassifierOutputTarget):
    """Target wrapper for dict-based model outputs with separate head logits."""

    def __init__(self, head_name: str):
        super().__init__(category=0)
        self.head_name = head_name

    def __call__(self, model_output):
        if isinstance(model_output, dict):
            if self.head_name not in model_output:
                raise ValueError(f'Head {self.head_name} not found in model output')
            out = model_output[self.head_name]
        elif isinstance(model_output, torch.Tensor):
            raise TypeError(
                'DictModelHeadTarget expects dict output from the model, but got Tensor'
            )
        else:
            raise TypeError(
                f'Unsupported model output type for Grad-CAM target: {type(model_output)}'
            )

        if isinstance(out, torch.Tensor):
            if out.dim() > 1 and out.size(1) == 1:
                out = out.squeeze(1)
            return out

        raise TypeError('Selected head output is not a torch.Tensor')


def _resolve_target_layer(model: torch.nn.Module, target_layer_name: str):
    if hasattr(model, target_layer_name):
        target_layer = getattr(model, target_layer_name)
        if target_layer is None:
            raise AttributeError(f'Target layer {target_layer_name} is None')
        return target_layer

    if '.' in target_layer_name:
        current = model
        for part in target_layer_name.split('.'):
            if not hasattr(current, part):
                raise AttributeError(f'Layer path invalid at {part}')
            current = getattr(current, part)
        return current

    raise AttributeError(
        f'Model has no attribute or submodule named {target_layer_name}'
    )


def get_multihead_3d_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer_name: str = 'layer4',
    device: Optional[str] = None,
    head_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate independent 3D Grad-CAM++ heatmaps for every multi-head output.

    Args:
        model: 3D SE-ResNet50 model returning dict(head_name->logits)
        input_tensor: Tensor shape (B, 1, 64, 64, 64)
        target_layer_name: final shared 3D conv layer name
        device: optional device string ('cpu' or 'cuda')
        head_names: ordered list of head names; defaults to FEATURE_NAMES

    Returns:
        dict mapping each head name to a (64,64,64) numpy heatmap.
    """
    head_names = head_names or FEATURE_NAMES
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    model = model.to(device)

    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError('input_tensor must be a torch.Tensor')
    if input_tensor.ndim != 5:
        raise ValueError('input_tensor must have shape (B, 1, 64, 64, 64)')
    if input_tensor.shape[1] != 1:
        raise ValueError('input_tensor channel dimension must equal 1')
    if input_tensor.shape[2:] != (64, 64, 64):
        raise ValueError('input_tensor spatial dimensions must equal (64, 64, 64)')
    if input_tensor.device != device:
        input_tensor = input_tensor.to(device)

    target_layer = _resolve_target_layer(model, target_layer_name)
    use_cuda = device.type == 'cuda' and torch.cuda.is_available()

    cam = GradCAMPlusPlus(
        model=model,
        target_layers=[target_layer],
        use_cuda=use_cuda,
    )

    if input_tensor.shape[0] != 1:
        # Grad-CAM++ is only returning a single heatmap per head in this wrapper
        raise ValueError('Batch size > 1 is not supported by get_multihead_3d_gradcam')

    heatmaps: Dict[str, np.ndarray] = {}
    for head in head_names:
        target = DictModelHeadTarget(head)
        try:
            grayscale_cam = cam(input_tensor, targets=[target])
        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():
                torch.cuda.empty_cache()
                raise RuntimeError('CUDA out of memory during GradCAM computation') from exc
            raise

        if not isinstance(grayscale_cam, np.ndarray):
            raise RuntimeError('GradCAM output expected to be numpy array')
        if grayscale_cam.ndim == 4:
            cam_map = grayscale_cam[0]
        elif grayscale_cam.ndim == 3:
            cam_map = grayscale_cam
        else:
            raise RuntimeError(f'Unexpected GradCAM output shape {grayscale_cam.shape}')

        if cam_map.shape != (64, 64, 64):
            cam_map = np.asarray(cam_map)
            if cam_map.shape[0] == 1 and cam_map.shape[1:] == (64, 64, 64):
                cam_map = cam_map[0]
            else:
                raise RuntimeError(
                    f'Unsupported heatmap shape from GradCAM: {cam_map.shape}'
                )

        mn, mx = float(np.min(cam_map)), float(np.max(cam_map))
        if mx > mn:
            cam_map = (cam_map - mn) / (mx - mn)
        else:
            cam_map = np.zeros_like(cam_map, dtype=np.float32)

        heatmaps[head] = cam_map.astype(np.float32)

    return heatmaps
