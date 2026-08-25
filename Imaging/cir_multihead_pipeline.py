"""
cir_multihead_pipeline.py

Canonical model construction and Grad-CAM++ explainability for the
LungInsight 3D CT multi-head classifier.

Coordinate convention
---------------------
All volumetric arrays use:

    (Z, Y, X)

PyTorch tensors use:

    (N, C, Z, Y, X)

Heatmaps returned by generate_characteristic_heatmaps() use:

    (Z, Y, X)

Therefore the returned heatmap can be indexed directly against the
corresponding 3D CT patch without axis permutation.

Model output contract
---------------------
The current training pipeline uses masked MSE directly on the model outputs.

Therefore each classifier head returns a raw scalar regression prediction.

No sigmoid is applied by this module.

Grad-CAM++
----------
A separate Grad-CAM++ heatmap is generated for every model head.

The target layer must be a module actually traversed during the model
forward pass.

Canonical targets:

    backbone.layer1
    backbone.layer2
    backbone.layer3
    backbone.layer4

For 64^3 input patches, backbone.layer3 is the default because it provides
a better semantic/spatial trade-off than backbone.layer4.

Approximate spatial resolutions:

    input:             64^3
    stem:              16^3
    backbone.layer1:   16^3
    backbone.layer2:    8^3
    backbone.layer3:    4^3
    backbone.layer4:    2^3
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from se_resnet3d import se_resnet50_3d


# ---------------------------------------------------------------------
# MODEL CONTRACT
# ---------------------------------------------------------------------

FEATURE_NAMES = [
    "calcification",
    "lobulation",
    "malignancy",
    "margin",
    "sphericity",
    "spiculation",
    "subtlety",
    "texture",
]

PATCH_SIZE = 64

DEFAULT_GRADCAM_LAYER = "backbone.layer3"


# ---------------------------------------------------------------------
# LAYER RESOLUTION
# ---------------------------------------------------------------------

def _resolve_layer(
    model: torch.nn.Module,
    dotted_name: str,
) -> Optional[torch.nn.Module]:
    """
    Resolve a dotted module path.

    Example:

        backbone.layer3

    resolves to:

        model.backbone.layer3
    """

    if not dotted_name:
        return None

    obj = model

    for part in dotted_name.split("."):

        if not hasattr(obj, part):
            return None

        obj = getattr(obj, part)

        if obj is None:
            return None

    if not isinstance(obj, torch.nn.Module):
        return None

    return obj


def _list_matching_modules(
    model: torch.nn.Module,
    query: str,
):
    """
    Return module names containing the requested text.

    Used only for diagnostics.
    """

    query = query.lower()

    matches = []

    for name, module in model.named_modules():

        if query in name.lower():

            matches.append(
                (
                    name,
                    module.__class__.__name__,
                )
            )

    return matches


def _describe_model_layers(
    model: torch.nn.Module,
) -> str:
    """
    Return a compact diagnostic list of important modules.
    """

    lines = []

    for name, module in model.named_modules():

        class_name = (
            module.__class__.__name__.lower()
        )

        if (
            "conv" in class_name
            or "bottleneck" in class_name
            or "layer" in name.lower()
        ):
            lines.append(
                f"    {name}: "
                f"{module.__class__.__name__}"
            )

    if not lines:
        return "    <no obvious convolutional modules found>"

    return "\n".join(lines[:120])


# ---------------------------------------------------------------------
# MODEL CONSTRUCTION
# ---------------------------------------------------------------------

def create_multihead_model(
    device: Optional[torch.device] = None,
    head_names=None,
) -> torch.nn.Module:
    """
    Construct the canonical LungInsight multi-head SE-ResNet50 3D model.

    Outputs are raw regression predictions.
    """

    if device is None:
        device = torch.device("cpu")

    names = list(
        head_names
        if head_names is not None
        else FEATURE_NAMES
    )

    model = se_resnet50_3d(
        in_channels=1,
        head_names=names,
    )

    model.to(device)

    return model


# ---------------------------------------------------------------------
# MODEL OUTPUT HELPERS
# ---------------------------------------------------------------------

def _extract_head_outputs(
    outputs,
) -> Dict[str, torch.Tensor]:
    """
    Validate the canonical multi-head output representation.
    """

    if not isinstance(outputs, dict):
        raise TypeError(
            "Expected the multi-head model to return a dictionary "
            f"of head predictions, got {type(outputs).__name__}."
        )

    return outputs


# ---------------------------------------------------------------------
# GRAD-CAM CAPTURE
# ---------------------------------------------------------------------

class _GradCAMCapture:
    """
    Stores activations and gradients for one target layer.
    """

    def __init__(self):

        self.activations = None
        self.gradients = None

        self.forward_called = False
        self.backward_called = False

    def forward_hook(
        self,
        module,
        inputs,
        output,
    ):

        self.forward_called = True

        if isinstance(output, (tuple, list)):
            output = output[0]

        if not torch.is_tensor(output):
            raise RuntimeError(
                "Grad-CAM target layer returned a non-tensor output: "
                f"{type(output).__name__}"
            )

        self.activations = output

    def backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.backward_called = True

        if not grad_output:
            return

        gradient = grad_output[0]

        if gradient is None:
            return

        self.gradients = gradient


# ---------------------------------------------------------------------
# CAM NORMALIZATION
# ---------------------------------------------------------------------

def _normalize_cam(
    cam: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize each batch CAM independently to [0, 1].

    Input:
        (N, 1, Z, Y, X)

    Output:
        (N, 1, Z, Y, X)
    """

    if cam.ndim != 5:
        raise ValueError(
            "CAM must have shape (N,1,Z,Y,X), "
            f"got {tuple(cam.shape)}"
        )

    batch_size = cam.shape[0]

    flat = cam.reshape(
        batch_size,
        -1,
    )

    cam_min = flat.min(
        dim=1,
    ).values.reshape(
        batch_size,
        1,
        1,
        1,
        1,
    )

    cam_max = flat.max(
        dim=1,
    ).values.reshape(
        batch_size,
        1,
        1,
        1,
        1,
    )

    denominator = cam_max - cam_min

    normalized = torch.where(
        denominator > 1e-8,
        (cam - cam_min) / denominator,
        torch.zeros_like(cam),
    )

    return normalized.clamp(
        0.0,
        1.0,
    )


# ---------------------------------------------------------------------
# GRAD-CAM++ COMPUTATION
# ---------------------------------------------------------------------

def _compute_gradcam_pp(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    output_size: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Compute Grad-CAM++.

    Parameters
    ----------
    activations:
        (N, C, z, y, x)

    gradients:
        (N, C, z, y, x)

    output_size:
        Final (Z, Y, X) heatmap size.

    Returns
    -------
    Tensor:
        (N, 1, Z, Y, X)
    """

    if activations.ndim != 5:
        raise RuntimeError(
            "Grad-CAM activations must have shape "
            "(N,C,Z,Y,X), got "
            f"{tuple(activations.shape)}"
        )

    if gradients.ndim != 5:
        raise RuntimeError(
            "Grad-CAM gradients must have shape "
            "(N,C,Z,Y,X), got "
            f"{tuple(gradients.shape)}"
        )

    if activations.shape != gradients.shape:
        raise RuntimeError(
            "Activation/gradient shape mismatch: "
            f"activation={tuple(activations.shape)}, "
            f"gradient={tuple(gradients.shape)}"
        )

    gradient_squared = gradients.pow(2)

    gradient_cubed = gradients.pow(3)

    spatial_term = (
        activations
        * gradient_cubed
    ).sum(
        dim=(2, 3, 4),
        keepdim=True,
    )

    denominator = (
        2.0 * gradient_squared
        + spatial_term
    )

    denominator = torch.where(
        denominator.abs() > 1e-12,
        denominator,
        torch.ones_like(denominator),
    )

    alpha = (
        gradient_squared
        / denominator
    )

    positive_gradients = F.relu(
        gradients
    )

    weights = (
        alpha
        * positive_gradients
    ).sum(
        dim=(2, 3, 4),
    )

    cam = (
        weights[
            :,
            :,
            None,
            None,
            None,
        ]
        * activations
    ).sum(
        dim=1,
        keepdim=True,
    )

    cam = F.relu(cam)

    cam = F.interpolate(
        cam,
        size=output_size,
        mode="trilinear",
        align_corners=False,
    )

    cam = _normalize_cam(cam)

    return cam


# ---------------------------------------------------------------------
# GRADIENT RELIABILITY DIAGNOSTIC
# ---------------------------------------------------------------------

GRADIENT_ACTIVE_CELL_RATIO = 0.05
GRADIENT_ACTIVE_CELL_THRESHOLD = 0.10


def _gradient_reliability(
    gradients: torch.Tensor,
) -> Dict[str, object]:
    """
    Measure whether gradients are distributed across the target layer's
    spatial grid.

    This diagnostic is intentionally restricted to batch size 1 because
    LungInsight generates candidate-level Grad-CAM maps one patch at a time.
    """

    if gradients.ndim != 5:
        raise ValueError(
            "Expected gradients with shape (N,C,Z,Y,X), "
            f"got {tuple(gradients.shape)}"
        )

    if gradients.shape[0] != 1:
        raise ValueError(
            "Gradient reliability diagnostics require batch size 1, "
            f"got batch size {gradients.shape[0]}"
        )

    cell_norm = gradients[0].norm(
        dim=0,
    )

    flat = cell_norm.reshape(-1)

    max_value = float(
        flat.max().item()
    )

    if max_value <= 0.0:

        return {
            "max_cell_grad_norm": 0.0,
            "active_cell_fraction": 0.0,
            "reliable": False,
        }

    active_mask = (
        flat
        >= GRADIENT_ACTIVE_CELL_THRESHOLD
        * max_value
    )

    active_fraction = float(
        active_mask.float().mean().item()
    )

    return {
        "max_cell_grad_norm": max_value,
        "active_cell_fraction": active_fraction,
        "reliable": (
            active_fraction
            >= GRADIENT_ACTIVE_CELL_RATIO
        ),
    }


# ---------------------------------------------------------------------
# MAIN HEATMAP GENERATOR
# ---------------------------------------------------------------------

def generate_characteristic_heatmaps(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: Optional[torch.device] = None,
    target_layer: str = DEFAULT_GRADCAM_LAYER,
    return_diagnostics: bool = False,
):
    """
    Generate one Grad-CAM++ heatmap for every classifier head.

    Parameters
    ----------
    model:
        Loaded multi-head regression model.

    x:
        Input tensor with shape:

            (1, 1, Z, Y, X)

        Normally:

            (1, 1, 64, 64, 64)

    device:
        Torch device.

    target_layer:
        Executed module path.

        Recommended:

            backbone.layer3

        Other valid choices:

            backbone.layer1
            backbone.layer2
            backbone.layer4

    return_diagnostics:
        If True, return:

            heatmaps, diagnostics

    Returns
    -------
    dict[str, np.ndarray]

        Each heatmap has shape:

            (Z, Y, X)

        Values are normalized to:

            [0, 1]
    """

    if device is None:

        try:
            device = next(
                model.parameters()
            ).device

        except StopIteration:
            device = torch.device("cpu")

    model.eval()

    x = x.to(
        device=device,
        dtype=torch.float32,
    )

    if x.ndim != 5:
        raise ValueError(
            "Grad-CAM input must have shape "
            "(N,C,Z,Y,X), "
            f"got {tuple(x.shape)}"
        )

    if x.shape[0] != 1:
        raise ValueError(
            "LungInsight candidate Grad-CAM currently requires "
            "batch size 1, "
            f"got {x.shape[0]}"
        )

    layer = _resolve_layer(
        model,
        target_layer,
    )

    if layer is None:

        query = target_layer.split(".")[-1]

        matches = _list_matching_modules(
            model,
            query,
        )

        if matches:

            match_text = "\n".join(
                f"    {name}: {class_name}"
                for name, class_name in matches
            )

        else:

            match_text = (
                "    <no matching modules>"
            )

        raise AttributeError(
            "\n"
            "Grad-CAM target layer could not be resolved.\n"
            f"Requested: {target_layer}\n\n"
            "Matching modules:\n"
            f"{match_text}\n\n"
            "Available model layers:\n"
            f"{_describe_model_layers(model)}"
        )

    capture = _GradCAMCapture()

    forward_handle = (
        layer.register_forward_hook(
            capture.forward_hook
        )
    )

    backward_handle = (
        layer.register_full_backward_hook(
            capture.backward_hook
        )
    )

    try:

        model.zero_grad(
            set_to_none=True
        )

        outputs = model(x)

        outputs = _extract_head_outputs(
            outputs
        )

        if not capture.forward_called:

            raise RuntimeError(
                "Grad-CAM target layer did not execute during "
                "the model forward pass.\n"
                f"Requested target: {target_layer}\n"
                f"Resolved module: "
                f"{layer.__class__.__name__}"
            )

        activations = capture.activations

        if activations is None:

            raise RuntimeError(
                "Grad-CAM target layer executed but no activation "
                "tensor was captured."
            )

        if activations.ndim != 5:

            raise RuntimeError(
                "Target activation must have shape "
                "(N,C,Z,Y,X), got "
                f"{tuple(activations.shape)}"
            )

        if activations.shape[0] != x.shape[0]:

            raise RuntimeError(
                "Target activation batch size does not match "
                "input batch size."
            )

        heatmaps: Dict[
            str,
            np.ndarray
        ] = {}

        diagnostics: Dict[
            str,
            Dict[str, object]
        ] = {}

        for head_name in FEATURE_NAMES:

            if head_name not in outputs:

                raise RuntimeError(
                    f"Model output is missing head "
                    f"'{head_name}'. "
                    f"Available outputs: "
                    f"{list(outputs.keys())}"
                )

            model.zero_grad(
                set_to_none=True
            )

            capture.gradients = None
            capture.backward_called = False

            prediction = outputs[head_name]

            if prediction.ndim > 1:

                prediction = prediction.reshape(
                    prediction.shape[0],
                    -1,
                ).mean(
                    dim=1
                )

            target_score = prediction.sum()

            target_score.backward(
                retain_graph=True
            )

            if not capture.backward_called:

                raise RuntimeError(
                    "Grad-CAM backward hook did not fire for "
                    f"head '{head_name}' at "
                    f"'{target_layer}'."
                )

            gradients = capture.gradients

            if gradients is None:

                raise RuntimeError(
                    "No Grad-CAM gradients were captured for "
                    f"head '{head_name}' at "
                    f"'{target_layer}'."
                )

            act = activations.detach()

            grad = gradients.detach()

            diagnostics[head_name] = (
                _gradient_reliability(
                    grad
                )
            )

            cam = _compute_gradcam_pp(
                activations=act,
                gradients=grad,
                output_size=tuple(
                    x.shape[2:]
                ),
            )

            cam_np = (
                cam[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if cam_np.shape != tuple(
                x.shape[2:]
            ):

                raise RuntimeError(
                    "Grad-CAM geometry mismatch for "
                    f"'{head_name}': "
                    f"expected {tuple(x.shape[2:])}, "
                    f"got {cam_np.shape}"
                )

            if not np.isfinite(
                cam_np
            ).all():

                raise RuntimeError(
                    f"Grad-CAM heatmap for "
                    f"'{head_name}' contains "
                    "NaN or infinite values."
                )

            heatmaps[head_name] = cam_np

        if return_diagnostics:

            return (
                heatmaps,
                diagnostics,
            )

        return heatmaps

    finally:

        forward_handle.remove()

        backward_handle.remove()


# ---------------------------------------------------------------------
# SELF TEST
# ---------------------------------------------------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("CIR MULTI-HEAD PIPELINE SELF TEST")
    print("=" * 72)

    model = create_multihead_model(
        device=device,
    )

    x = torch.randn(
        1,
        1,
        PATCH_SIZE,
        PATCH_SIZE,
        PATCH_SIZE,
        device=device,
    )

    print("\nModel outputs:")

    with torch.no_grad():

        outputs = model(x)

    for name, value in outputs.items():

        print(
            f"  {name:16s}: "
            f"shape={tuple(value.shape)}"
        )

    test_layers = [
        "backbone.layer1",
        "backbone.layer2",
        "backbone.layer3",
        "backbone.layer4",
    ]

    for target in test_layers:

        print(
            "\n"
            + "-" * 72
        )

        print(
            f"Testing Grad-CAM target: "
            f"{target}"
        )

        print(
            "-" * 72
        )

        try:

            heatmaps, diagnostics = (
                generate_characteristic_heatmaps(
                    model=model,
                    x=x,
                    device=device,
                    target_layer=target,
                    return_diagnostics=True,
                )
            )

            for name, heatmap in heatmaps.items():

                diagnostic = diagnostics[name]

                print(
                    f"  {name:16s}: "
                    f"shape={heatmap.shape}, "
                    f"min={heatmap.min():.4f}, "
                    f"max={heatmap.max():.4f}, "
                    f"mean={heatmap.mean():.4f}, "
                    f"active="
                    f"{diagnostic['active_cell_fraction'] * 100:.1f}%, "
                    f"reliable="
                    f"{diagnostic['reliable']}"
                )

        except Exception as exc:

            print(
                f"  FAILED: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )