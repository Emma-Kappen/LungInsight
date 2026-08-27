"""
07_visualize_gradcam.py

LungInsight — Stage 07
======================

Generate candidate-level 3D Grad-CAM explanations for ALL classifier heads.

Coordinate convention
---------------------

ALL spatial coordinates use:

    Z, Y, X

Three spatial spaces are explicitly preserved:

1. GLOBAL / NATIVE CT SPACE
   Full Stage 02 native CT volume.

2. PHYSICAL PATCH SPACE
   Continuous physical coordinates corresponding to the Stage 05
   classifier patch.

3. LOCAL / CANDIDATE PATCH SPACE
   The 64 x 64 x 64 Stage 05 classifier patch.

Grad-CAM is generated at the target convolutional feature-map resolution
and interpolated ONLY inside the local 64³ patch.

Stage 07 NEVER projects Grad-CAM into the native CT.

Stage 08 performs the local -> physical -> native/global projection.

Classifier preprocessing contract
---------------------------------

Stage 05 stores classifier patches normalized to:

    normalized = (HU + 1000) / 1400

Therefore Stage 07 feeds the saved [0,1] patch DIRECTLY into the model.

NO SECOND NORMALIZATION IS PERFORMED.

For visualization only:

    HU = normalized * 1400 - 1000

Grad-CAM contract
-----------------

Every classifier head receives an independent forward/backward pass.

For each candidate:

    calcification.npy
    lobulation.npy
    malignancy.npy
    margin.npy
    sphericity.npy
    spiculation.npy
    subtlety.npy
    texture.npy

is produced.

Every CAM is:

    [64,64,64]

in LOCAL candidate-patch coordinates.

No CAMs are accumulated, averaged, or merged between heads.

Output structure
----------------

07_gradcam/
    candidate_<id>/
        gradcam/
            calcification.npy
            lobulation.npy
            malignancy.npy
            margin.npy
            sphericity.npy
            spiculation.npy
            subtlety.npy
            texture.npy

        montages/
            calcification.png
            lobulation.png
            malignancy.png
            margin.png
            sphericity.png
            spiculation.png
            subtlety.png
            texture.png

        metadata.json

    gradcam_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# CONFIGURATION
# ============================================================================

PATCH_SIZE = 64

HU_MIN = -1000.0
HU_MAX = 400.0

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

DEFAULT_TARGET = "malignancy"

# Opt-in verbose per-head gradient/activation diagnostics. Off by default
# so production runs stay quiet; set LUNGINSIGHT_GRADCAM_DEBUG=1 to enable
# when investigating a specific head's CAM.
_GRADCAM_DEBUG = bool(
    int(os.environ.get("LUNGINSIGHT_GRADCAM_DEBUG", "0"))
)

DEFAULT_CHECKPOINT = (
    "Imaging/checkpoints/best_model_gpu_v3.pth"
)

LOCAL_CENTER_ZYX = [
    (PATCH_SIZE - 1) / 2.0,
    (PATCH_SIZE - 1) / 2.0,
    (PATCH_SIZE - 1) / 2.0,
]


# ============================================================================
# JSON HELPERS
# ============================================================================

def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================================
# PARSING
# ============================================================================

def _is_empty(value: Any) -> bool:
    return value is None or value == ""


def parse_float_zyx(value: Any) -> Optional[List[float]]:
    """
    Parse a floating-point Z,Y,X vector.
    """

    if value is None:
        return None

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(
                f"Expected 3-element ZYX vector, got {value}"
            )

        return [float(x) for x in value]

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return None

        try:
            parsed = json.loads(text)

            if isinstance(parsed, (list, tuple)):
                return parse_float_zyx(parsed)

        except json.JSONDecodeError:
            pass

        parts = [
            p.strip()
            for p in text.split(",")
        ]

        if len(parts) == 3:
            return [float(x) for x in parts]

    raise ValueError(
        f"Could not parse ZYX coordinate: {value!r}"
    )


def parse_int_zyx(value: Any) -> Optional[List[int]]:
    parsed = parse_float_zyx(value)

    if parsed is None:
        return None

    return [
        int(round(x))
        for x in parsed
    ]


def first_value(
    mapping: Dict[str, Any],
    keys: Tuple[str, ...],
) -> Any:
    for key in keys:
        if key not in mapping:
            continue

        value = mapping[key]

        if not _is_empty(value):
            return value

    return None


def first_float_zyx(
    mapping: Dict[str, Any],
    keys: Tuple[str, ...],
) -> Optional[List[float]]:
    for key in keys:
        if key not in mapping:
            continue

        value = mapping[key]

        if _is_empty(value):
            continue

        try:
            return parse_float_zyx(value)
        except (TypeError, ValueError):
            continue

    return None


def first_int_zyx(
    mapping: Dict[str, Any],
    keys: Tuple[str, ...],
) -> Optional[List[int]]:
    value = first_float_zyx(
        mapping,
        keys,
    )

    if value is None:
        return None

    return [
        int(round(x))
        for x in value
    ]


# ============================================================================
# RECORD / GEOMETRY HELPERS
# ============================================================================

def geometry_dicts(record: dict) -> List[dict]:
    """
    Return likely locations of Stage 05 spatial metadata.

    Stage 06 may preserve Stage 05 geometry under different nesting.
    """

    result: List[dict] = []

    def add(value):
        if isinstance(value, dict) and value not in result:
            result.append(value)

    add(record)

    for key in (
        "geometry",
        "patch_geometry",
        "stage05_geometry",
        "crop_geometry",
        "spatial_metadata",
        "metadata",
        "spatial_context",
        "crop",
    ):
        value = record.get(key)

        if isinstance(value, dict):
            add(value)

            nested = value.get("geometry")

            if isinstance(nested, dict):
                add(nested)

            nested_spatial = value.get("spatial_context")

            if isinstance(nested_spatial, dict):
                add(nested_spatial)

    return result


def extract_crop_geometry(record: dict) -> Dict[str, Any]:
    """
    Extract Stage 05 geometry without reconstructing the crop.

    Priority:

        1. explicit Stage 05 physical/local geometry
        2. explicit native geometry
        3. mathematically derivable quantities

    No spatial crop is generated here.
    """

    candidates = geometry_dicts(record)

    geometry: Dict[str, Any] = {}

    # -----------------------------------------------------------------
    # Candidate/global native center.
    # -----------------------------------------------------------------

    candidate_center_zyx = None

    for mapping in candidates:
        candidate_center_zyx = first_float_zyx(
            mapping,
            (
                "candidate_center_zyx",
                "patch_center_native_zyx",
                "native_center_zyx",
                "center_zyx",
                "voxel_center_zyx",
            ),
        )

        if candidate_center_zyx is not None:
            break

    if candidate_center_zyx is None:
        raise RuntimeError(
            "Stage 05/06 record does not contain a native "
            "candidate center in Z,Y,X order.\n"
            f"Record keys: {sorted(record.keys())}"
        )

    geometry["candidate_center_zyx"] = candidate_center_zyx

    # -----------------------------------------------------------------
    # Native volume shape.
    # -----------------------------------------------------------------

    native_shape = None

    for mapping in candidates:
        native_shape = first_int_zyx(
            mapping,
            (
                "native_volume_shape_zyx",
                "native_shape_zyx",
                "source_shape_zyx",
                "volume_shape_zyx",
                "global_shape_zyx",
                "ct_shape_zyx",
            ),
        )

        if native_shape is not None:
            break

    geometry["native_volume_shape_zyx"] = native_shape

    # -----------------------------------------------------------------
    # Native spacing.
    # -----------------------------------------------------------------

    native_spacing = None

    for mapping in candidates:
        native_spacing = first_float_zyx(
            mapping,
            (
                "native_spacing_zyx_mm",
                "spacing_zyx_mm",
                "source_spacing_zyx_mm",
                "volume_spacing_zyx_mm",
            ),
        )

        if native_spacing is not None:
            break

    geometry["native_spacing_zyx_mm"] = native_spacing

    # -----------------------------------------------------------------
    # Local patch shape.
    # -----------------------------------------------------------------

    patch_shape = None

    for mapping in candidates:
        patch_shape = first_int_zyx(
            mapping,
            (
                "patch_shape_zyx",
                "local_patch_shape_zyx",
                "output_shape_zyx",
            ),
        )

        if patch_shape is not None:
            break

    if patch_shape is None:
        patch_shape = [
            PATCH_SIZE,
            PATCH_SIZE,
            PATCH_SIZE,
        ]

    if patch_shape != [
        PATCH_SIZE,
        PATCH_SIZE,
        PATCH_SIZE,
    ]:
        raise RuntimeError(
            "Stage 05 patch shape is not 64³.\n"
            f"Got: {patch_shape}"
        )

    geometry["patch_shape_zyx"] = patch_shape

    # -----------------------------------------------------------------
    # Local patch center.
    # -----------------------------------------------------------------

    local_center = None

    for mapping in candidates:
        local_center = first_float_zyx(
            mapping,
            (
                "local_patch_center_zyx",
                "patch_center_local_zyx",
                "local_center_zyx",
            ),
        )

        if local_center is not None:
            break

    if local_center is None:
        local_center = list(LOCAL_CENTER_ZYX)

    geometry["local_patch_center_zyx"] = local_center

    # -----------------------------------------------------------------
    # Patch spacing.
    # -----------------------------------------------------------------

    patch_spacing = None

    for mapping in candidates:
        patch_spacing = first_float_zyx(
            mapping,
            (
                "patch_spacing_zyx_mm",
                "local_spacing_zyx_mm",
                "output_spacing_zyx_mm",
            ),
        )

        if patch_spacing is not None:
            break

    if patch_spacing is None:
        patch_spacing = [
            1.0,
            1.0,
            1.0,
        ]

    geometry["patch_spacing_zyx_mm"] = patch_spacing

    # -----------------------------------------------------------------
    # Patch FOV.
    # -----------------------------------------------------------------

    patch_fov = None

    for mapping in candidates:
        patch_fov = first_float_zyx(
            mapping,
            (
                "patch_fov_zyx_mm",
                "physical_fov_zyx_mm",
                "fov_zyx_mm",
            ),
        )

        if patch_fov is not None:
            break

    if patch_fov is None:
        patch_fov = [
            patch_shape[i] * patch_spacing[i]
            for i in range(3)
        ]

    geometry["patch_fov_zyx_mm"] = patch_fov

    # -----------------------------------------------------------------
    # Physical patch center.
    #
    # Prefer Stage 05's explicit value.
    # -----------------------------------------------------------------

    physical_center = None

    for mapping in candidates:
        physical_center = first_float_zyx(
            mapping,
            (
                "patch_center_physical_zyx_mm",
                "patch_center_physical_mm",
                "candidate_center_physical_zyx_mm",
                "candidate_center_physical_mm",
                "physical_center_zyx_mm",
            ),
        )

        if physical_center is not None:
            break

    # -----------------------------------------------------------------
    # Physical center. Stage 05 is authoritative. If an old Stage 05
    # manifest lacks it, derive it using the Stage 02 physical origin;
    # never assume voxel index 0 is physical coordinate 0.
    # -----------------------------------------------------------------

    physical_center_source = "stage05"

    if physical_center is None:
        native_origin = None
        for mapping in candidates:
            native_origin = first_float_zyx(
                mapping,
                (
                    "native_origin_zyx_mm",
                    "origin_zyx_mm",
                    "origin_mm",
                ),
            )
            if native_origin is not None:
                break

        if native_spacing is not None:
            if native_origin is None:
                native_origin = [0.0, 0.0, 0.0]
                physical_center_source = "derived_with_zero_origin"
            else:
                physical_center_source = "derived_from_native_origin"

            physical_center = [
                native_origin[i]
                + candidate_center_zyx[i] * native_spacing[i]
                for i in range(3)
            ]
        else:
            physical_center_source = "unavailable"

    geometry[
        "patch_center_physical_zyx_mm"
    ] = physical_center

    geometry[
        "physical_center_source"
    ] = physical_center_source

    # -----------------------------------------------------------------
    # Preserve exact legacy Stage 05 fields when available.
    #
    # Stage 08 may still use these if present.
    # -----------------------------------------------------------------

    for key in (
        "native_start_zyx",
        "native_end_zyx",
        "source_intersection_start_zyx",
        "source_intersection_end_zyx",
        "source_start_zyx",
        "source_end_zyx",
    ):
        for mapping in candidates:
            if key in mapping and not _is_empty(mapping[key]):
                try:
                    geometry[key] = parse_int_zyx(
                        mapping[key]
                    )
                except (TypeError, ValueError):
                    geometry[key] = mapping[key]
                break

    return geometry


# ============================================================================
# EXPLICIT COORDINATE METADATA
# ============================================================================

def build_spatial_context(
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build explicit global / physical / local / Grad-CAM coordinate metadata.

    The local coordinate system is authoritative for the saved CAM.
    """

    shape = geometry[
        "patch_shape_zyx"
    ]

    local_center = geometry[
        "local_patch_center_zyx"
    ]

    native_center = geometry[
        "candidate_center_zyx"
    ]

    native_spacing = geometry.get(
        "native_spacing_zyx_mm"
    )

    patch_spacing = geometry[
        "patch_spacing_zyx_mm"
    ]

    physical_center = geometry.get(
        "patch_center_physical_zyx_mm"
    )

    local_to_native = (
        "native_zyx = "
        "candidate_center_zyx + "
        "(local_zyx - local_patch_center_zyx) * "
        "patch_spacing_zyx_mm / "
        "native_spacing_zyx_mm"
    )

    local_to_physical = (
        "physical_zyx_mm = "
        "patch_center_physical_zyx_mm + "
        "(local_zyx - local_patch_center_zyx) * "
        "patch_spacing_zyx_mm"
    )

    return {
        "coordinate_order": "ZYX",

        "global": {
            "space": "stage02_native_ct",

            "shape_zyx":
                geometry.get(
                    "native_volume_shape_zyx"
                ),

            "spacing_zyx_mm":
                native_spacing,

            "candidate_center_zyx":
                native_center,
        },

        "physical": {
            "space":
                "patient_physical_patch_space",

            "center_zyx_mm":
                physical_center,

            "spacing_zyx_mm":
                patch_spacing,

            "fov_zyx_mm":
                geometry[
                    "patch_fov_zyx_mm"
                ],

            "center_source":
                geometry[
                    "physical_center_source"
                ],
        },

        "local": {
            "space":
                "candidate_patch",

            "shape_zyx":
                shape,

            "center_zyx":
                local_center,

            "coordinate_range_zyx": [
                [0, shape[0] - 1],
                [0, shape[1] - 1],
                [0, shape[2] - 1],
            ],
        },

        "gradcam": {
            "space":
                "candidate_patch",

            "shape_zyx":
                shape,

            "center_zyx":
                local_center,

            "coordinate_range_zyx": [
                [0, shape[0] - 1],
                [0, shape[1] - 1],
                [0, shape[2] - 1],
            ],
        },

        "transform": {
            "local_to_physical":
                local_to_physical,

            "local_to_native":
                local_to_native,

            "authority":
                "stage05",
        },

        "geometry_authority":
            "stage05",
    }


# ============================================================================
# MODEL
# ============================================================================

def load_project_model(
    checkpoint_path: str,
    device: torch.device,
):
    """
    Construct the canonical LungInsight classifier.
    """

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    if str(project_root) not in sys.path:
        sys.path.insert(
            0,
            str(project_root),
        )

    try:

        from cir_multihead_pipeline import (
            create_multihead_model,
            FEATURE_NAMES as MODEL_FEATURE_NAMES,
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not import cir_multihead_pipeline.py.\n"
            f"Project root: {project_root}\n"
            f"Original error: {exc}"
        ) from exc

    if list(MODEL_FEATURE_NAMES) != FEATURE_NAMES:

        raise RuntimeError(
            "Classifier feature order mismatch.\n\n"
            f"Model:    {MODEL_FEATURE_NAMES}\n"
            f"Stage 07: {FEATURE_NAMES}"
        )

    checkpoint_path = os.path.abspath(
        checkpoint_path
    )

    if not os.path.isfile(checkpoint_path):

        raise FileNotFoundError(
            "Classifier checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    model = create_multihead_model(
        head_names=FEATURE_NAMES,
        device=device,
    )

    state = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if not isinstance(state, dict):

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    if "state_dict" in state:
        state = state["state_dict"]

    elif "model_state_dict" in state:
        state = state["model_state_dict"]

    cleaned_state = {}

    for key, value in state.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state[key] = value

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state,
            strict=False,
        )
    )

    if missing:

        raise RuntimeError(
            "Classifier checkpoint is missing model parameters:\n"
            + "\n".join(
                str(x)
                for x in missing[:20]
            )
        )

    if unexpected:

        raise RuntimeError(
            "Classifier checkpoint contains unexpected parameters:\n"
            + "\n".join(
                str(x)
                for x in unexpected[:20]
            )
        )

    model.eval()

    return model


# ============================================================================
# STAGE 05 / 06 LOADING
# ============================================================================

def load_stage06_classification(
    stage06_dir: str,
) -> Optional[List[dict]]:

    path = os.path.join(
        stage06_dir,
        "classification.json",
    )

    if not os.path.isfile(path):
        return None

    data = load_json(path)

    if not isinstance(data, dict):

        raise RuntimeError(
            f"Invalid Stage 06 classification file:\n{path}"
        )

    candidates = data.get("candidates")

    if not isinstance(candidates, list):

        raise RuntimeError(
            "Stage 06 classification.json does not "
            "contain a 'candidates' list."
        )

    print(
        f"Stage 06 classification: {path}"
    )

    return candidates


def find_stage05_manifest(
    stage05_dir: str,
) -> Optional[str]:

    names = [
        "patches.json",
        "candidate_patches.json",
        "patch_manifest.json",
        "candidates.json",
    ]

    for name in names:

        path = os.path.join(
            stage05_dir,
            name,
        )

        if os.path.isfile(path):
            return path

    return None


def load_stage05_records(
    stage05_dir: str,
) -> List[dict]:

    manifest_path = find_stage05_manifest(
        stage05_dir
    )

    if manifest_path is None:

        raise FileNotFoundError(
            "Could not locate Stage 05 patch manifest."
        )

    data = load_json(
        manifest_path
    )

    if isinstance(data, dict):

        for key in (
            "patches",
            "candidates",
            "records",
            "items",
        ):

            if isinstance(data.get(key), list):
                return data[key]

        raise RuntimeError(
            f"Could not identify candidate list in:\n"
            f"{manifest_path}"
        )

    if isinstance(data, list):
        return data

    raise RuntimeError(
        f"Unsupported Stage 05 manifest structure:\n"
        f"{manifest_path}"
    )


# ============================================================================
# RECORD HELPERS
# ============================================================================

def get_record_value(
    record: dict,
    keys: List[str],
    default=None,
):
    for key in keys:

        if key not in record:
            continue

        value = record[key]

        if _is_empty(value):
            continue

        return value

    return default


def record_identifier(
    record: dict,
    index: int,
) -> str:

    value = get_record_value(
        record,
        [
            "candidate_id",
            "id",
            "candidate_index",
            "annotation_index",
        ],
        index,
    )

    return str(value)


# ============================================================================
# PATCH PATH
# ============================================================================

def resolve_patch_path(
    record: dict,
    stage05_dir: str,
) -> Optional[str]:

    possible_keys = [
        "patch_file",
        "patch_path",
        "file_path",
        "path",
        "patch",
        "npy_path",
    ]

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    for key in possible_keys:

        value = record.get(key)

        if value is None:
            continue

        value = str(value)

        candidates = [
            value,
            os.path.join(
                stage05_dir,
                value,
            ),
            str(project_root / value),
        ]

        for candidate in candidates:

            if os.path.isfile(candidate):

                return os.path.abspath(
                    candidate
                )

    return None


# ============================================================================
# PATCH
# ============================================================================

def load_patch(
    path: str,
) -> np.ndarray:
    """
    Load Stage 05 normalized [0,1] patch.
    """

    if not os.path.isfile(path):

        raise FileNotFoundError(path)

    patch = np.load(
        path,
        allow_pickle=False,
    )

    patch = np.asarray(
        patch,
        dtype=np.float32,
    )

    if patch.ndim == 4 and patch.shape[0] == 1:
        patch = patch[0]

    expected_shape = (
        PATCH_SIZE,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    if patch.shape != expected_shape:

        raise ValueError(
            "Expected classifier patch "
            f"{expected_shape}, got {patch.shape}\n"
            f"path={path}"
        )

    if not np.isfinite(patch).all():

        raise ValueError(
            f"Patch contains NaN or Inf:\n{path}"
        )

    patch_min = float(patch.min())
    patch_max = float(patch.max())

    tolerance = 1e-4

    if (
        patch_min < -tolerance
        or patch_max > 1.0 + tolerance
    ):

        raise ValueError(
            "Stage 05 patch is not normalized to [0,1].\n"
            f"min={patch_min:.6f}\n"
            f"max={patch_max:.6f}\n"
            f"path={path}"
        )

    return patch


def normalized_to_hu(
    patch: np.ndarray,
) -> np.ndarray:

    return (
        patch * (HU_MAX - HU_MIN)
        + HU_MIN
    ).astype(
        np.float32,
        copy=False,
    )


def patch_to_classifier_tensor(
    patch: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    No normalization occurs here.
    """

    tensor = torch.from_numpy(
        patch
    )

    tensor = tensor[
        None,
        None,
        ...
    ]

    return tensor.to(device)


# ============================================================================
# MODEL OUTPUT
# ============================================================================

def get_target_output(
    outputs,
    target_name: str,
) -> torch.Tensor:

    if isinstance(outputs, dict):

        if target_name not in outputs:

            raise KeyError(
                f"Model output does not contain "
                f"'{target_name}'. "
                f"Available: {list(outputs.keys())}"
            )

        value = outputs[target_name]

    elif isinstance(outputs, (tuple, list)):

        index = FEATURE_NAMES.index(
            target_name
        )

        value = outputs[index]

    else:

        raise TypeError(
            "Unsupported model output type: "
            f"{type(outputs)}"
        )

    if value.ndim == 0:
        return value

    return value.reshape(-1)[0]


# ============================================================================
# GRAD-CAM
# ============================================================================

class GradCAM:
    """
    Candidate-local 3D Grad-CAM.

    Raw CAM:
        target feature-map spatial resolution

    Final CAM:
        [64,64,64]

    Every call to generate() is independent.
    """

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = output

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ):
        if not grad_output:
            self.gradients = None
        else:
            self.gradients = grad_output[0]

    def remove(self):

        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        patch: torch.Tensor,
        target_name: str,
    ) -> Tuple[np.ndarray, float, List[int]]:

        self.model.zero_grad(
            set_to_none=True
        )

        self.activations = None
        self.gradients = None

        outputs = self.model(patch)

        target = get_target_output(
            outputs,
            target_name,
        )

        prediction = float(
            target.detach()
            .cpu()
            .item()
        )

        target.backward()

        if self.activations is None:

            raise RuntimeError(
                f"Grad-CAM did not capture activations "
                f"for head '{target_name}'."
            )

        if self.gradients is None:

            raise RuntimeError(
                f"Grad-CAM did not capture gradients "
                f"for head '{target_name}'."
            )

        activations = self.activations
        gradients = self.gradients

        if activations.ndim != 5:

            raise RuntimeError(
                "Grad-CAM target layer must produce "
                "(B,C,D,H,W).\n"
                f"Got: {activations.shape}"
            )

        if gradients.shape != activations.shape:

            raise RuntimeError(
                "Activation/gradient shape mismatch.\n"
                f"Activations: {activations.shape}\n"
                f"Gradients:   {gradients.shape}"
            )

        raw_cam_shape = [
            int(activations.shape[2]),
            int(activations.shape[3]),
            int(activations.shape[4]),
        ]

        # -------------------------------------------------------------
        # Standard Grad-CAM channel weights.
        # -------------------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3, 4),
            keepdim=True,
        )

        signed_cam = (
            weights * activations
        ).sum(dim=1)[0]

        cam = F.relu(signed_cam)

        # -------------------------------------------------------------
        # Some heads legitimately have a weighted activation sum that
        # is negative everywhere (the head's positive evidence isn't
        # concentrated in this target layer's channels for this
        # candidate). Standard Grad-CAM's ReLU then discards the whole
        # map, producing an all-zero heatmap that carries no
        # information. When that happens, fall back to the signed map
        # so the head still yields a real localization instead of a
        # silently empty one; this only engages per-head, per-candidate
        # and never touches other heads' independent passes.
        # -------------------------------------------------------------

        if float(cam.max()) <= 0.0:

            cam = signed_cam

        if _GRADCAM_DEBUG:

            print(
                f"      [debug] {target_name:14s} "
                f"grad_abs_max={float(gradients.abs().max()):.6e} "
                f"weights_abs_max={float(weights.abs().max()):.6e} "
                f"activ_abs_max={float(activations.abs().max()):.6e} "
                f"cam_pre_norm_max={float(cam.max()):.6e} "
                f"fallback_signed="
                f"{bool(float(cam.max()) <= 0.0)}"
            )

        # -------------------------------------------------------------
        # Normalize RAW CAM before interpolation.
        # -------------------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        if float(cam_max - cam_min) > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = torch.zeros_like(cam)

        # -------------------------------------------------------------
        # LOCAL ONLY.
        #
        # This interpolation has no relationship to global CT space.
        # -------------------------------------------------------------

        cam = F.interpolate(
            cam[None, None],
            size=(
                PATCH_SIZE,
                PATCH_SIZE,
                PATCH_SIZE,
            ),
            mode="trilinear",
            align_corners=False,
        )[0, 0]

        heatmap = (
            cam.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        return (
            heatmap,
            prediction,
            raw_cam_shape,
        )


def generate_all_head_gradcams(
    gradcam: GradCAM,
    patch: torch.Tensor,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate one completely independent local CAM per classifier head.
    """

    results: Dict[str, Dict[str, Any]] = {}

    for head in FEATURE_NAMES:

        heatmap, prediction, raw_cam_shape = (
            gradcam.generate(
                patch=patch,
                target_name=head,
            )
        )

        expected_shape = (
            PATCH_SIZE,
            PATCH_SIZE,
            PATCH_SIZE,
        )

        if heatmap.shape != expected_shape:

            raise RuntimeError(
                f"Head '{head}' produced invalid CAM shape "
                f"{heatmap.shape}; expected {expected_shape}."
            )

        results[head] = {
            "heatmap": heatmap,
            "prediction": prediction,
            "raw_cam_shape": raw_cam_shape,
        }

    return results


# ============================================================================
# TARGET LAYER
# ============================================================================

def find_gradcam_target_layer(
    model,
    preferred_stage: str = "layer3",
):
    """
    Locate a Grad-CAM target layer.

    Deeper stages (layer4) are more semantically specific but, for a
    64^3 input passed through a typical stem(4x) + 3 strided stages,
    end up spatially tiny (e.g. 2x2x2). At that resolution the ReLU'd
    Grad-CAM weighted sum can legitimately be <= 0 at every single
    spatial location, which silently zeroes the entire CAM. Preferring
    an earlier stage keeps more spatial resolution while still being
    deep enough to carry semantic information.
    """

    if hasattr(model, "backbone"):

        backbone = model.backbone

        if hasattr(backbone, preferred_stage):

            stage = getattr(
                backbone,
                preferred_stage,
            )

            if len(stage) > 0:

                block = stage[-1]

                if hasattr(block, "conv3"):

                    name = (
                        f"model.backbone."
                        f"{preferred_stage}[-1].conv3"
                    )

                    print(
                        "Grad-CAM target layer:"
                    )

                    print(
                        f"  {name}"
                    )

                    return (
                        block.conv3,
                        name,
                    )

                if hasattr(block, "conv2"):

                    name = (
                        f"model.backbone."
                        f"{preferred_stage}[-1].conv2"
                    )

                    print(
                        "Grad-CAM target layer:"
                    )

                    print(
                        f"  {name}"
                    )

                    return (
                        block.conv2,
                        name,
                    )

        print(
            f"  [WARN] backbone.{preferred_stage} not found or "
            "empty; falling back to layer4."
        )

        if hasattr(backbone, "layer4"):

            layer4 = backbone.layer4

            if len(layer4) > 0:

                block = layer4[-1]

                if hasattr(block, "conv3"):

                    name = (
                        "model.backbone."
                        "layer4[-1].conv3"
                    )

                    print(
                        "Grad-CAM target layer:"
                    )

                    print(
                        f"  {name}"
                    )

                    return (
                        block.conv3,
                        name,
                    )

    last_conv = None
    last_name = None

    for name, module in model.named_modules():

        if isinstance(
            module,
            torch.nn.Conv3d,
        ):

            last_conv = module
            last_name = name

    if last_conv is None:

        raise RuntimeError(
            "Could not find a Conv3d layer for Grad-CAM."
        )

    print(
        "Grad-CAM target layer:"
    )

    print(
        f"  {last_name}"
    )

    return (
        last_conv,
        last_name,
    )


# ============================================================================
# VISUALIZATION
# ============================================================================

def percentile_window(
    patch_hu: np.ndarray,
) -> Tuple[float, float]:

    values = patch_hu[
        np.isfinite(patch_hu)
    ]

    if values.size == 0:
        return HU_MIN, HU_MAX

    low = float(
        np.percentile(
            values,
            1,
        )
    )

    high = float(
        np.percentile(
            values,
            99,
        )
    )

    low = max(
        low,
        HU_MIN,
    )

    high = min(
        high,
        HU_MAX,
    )

    if high <= low:
        high = low + 1.0

    return low, high


def choose_peak_slices(
    heatmap: np.ndarray,
    count: int = 5,
) -> List[int]:
    """
    Select strongest LOCAL axial z slices.

    z is always local patch z.
    """

    scores = heatmap.reshape(
        heatmap.shape[0],
        -1,
    ).mean(axis=1)

    order = np.argsort(scores)[::-1]

    selected = []

    for z in order:

        z = int(z)

        if all(
            abs(z - previous) >= 2
            for previous in selected
        ):
            selected.append(z)

        if len(selected) >= count:
            break

    if not selected:
        selected = [
            heatmap.shape[0] // 2
        ]

    return sorted(selected)


def save_gradcam_volume(
    heatmap: np.ndarray,
    output_path: str,
):

    expected_shape = (
        PATCH_SIZE,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    if heatmap.shape != expected_shape:

        raise ValueError(
            f"Grad-CAM must remain local {expected_shape}; "
            f"got {heatmap.shape}"
        )

    np.save(
        output_path,
        heatmap.astype(
            np.float32,
            copy=False,
        ),
    )


def save_montage(
    patch_hu: np.ndarray,
    heatmap: np.ndarray,
    slices: List[int],
    output_path: str,
    title: str,
):
    """
    Save CT + one HEAD'S Grad-CAM.

    CT and CAM are in exactly the same local coordinate system.
    """

    n = len(slices)

    fig, axes = plt.subplots(
        2,
        n,
        figsize=(
            3.5 * n,
            7,
        ),
    )

    if n == 1:

        axes = np.asarray(
            axes
        ).reshape(
            2,
            1,
        )

    vmin, vmax = percentile_window(
        patch_hu
    )

    for column, z in enumerate(slices):

        ct = patch_hu[z]
        cam = heatmap[z]

        axes[0, column].imshow(
            ct,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

        axes[0, column].set_title(
            f"Local CT\nz={z}"
        )

        axes[0, column].axis("off")

        axes[1, column].imshow(
            ct,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

        masked_cam = np.ma.masked_where(
            cam <= 0.05,
            cam,
        )

        axes[1, column].imshow(
            masked_cam,
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
            alpha=0.55,
        )

        axes[1, column].set_title(
            f"Grad-CAM\nz={z}"
        )

        axes[1, column].axis("off")

    fig.suptitle(
        title,
        fontsize=13,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate independent 3D Grad-CAM "
            "volumes for all LungInsight classifier heads."
        )
    )

    parser.add_argument(
        "patient_id",
        help="LIDC patient ID, e.g. LIDC-IDRI-0141",
    )

    parser.add_argument(
        "--output-root",
        default="output",
        help="Pipeline output root.",
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Trained classifier checkpoint.",
    )

    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        choices=FEATURE_NAMES,
        help=(
            "Head whose peak slices are reported prominently. "
            "ALL heads are still generated."
        ),
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Maximum number of candidates to explain.",
    )

    parser.add_argument(
        "--peak-slices",
        type=int,
        default=5,
        help="Number of strongest local axial slices per head.",
    )

    parser.add_argument(
        "--gradcam-layer",
        default="layer3",
        choices=["layer1", "layer2", "layer3", "layer4"],
        help=(
            "Backbone stage to hook for Grad-CAM. Deeper stages "
            "(layer4) are more semantic but spatially coarse "
            "(e.g. 2x2x2 for a 64^3 input with 32x downsampling), "
            "which can make the ReLU'd weighted CAM collapse to "
            "all-zero. Shallower stages (layer3/layer2) trade some "
            "semantic specificity for usable spatial resolution."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():

    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print(
        "LUNGINSIGHT STAGE 07 — MULTI-HEAD GRAD-CAM"
    )
    print("=" * 72)

    print(
        f"Patient      : {args.patient_id}"
    )

    print(
        f"Checkpoint   : {args.checkpoint}"
    )

    print(
        f"Device       : {device}"
    )

    print()
    print(
        "SPATIAL CONTRACT"
    )

    print(
        "  Coordinate order : Z,Y,X"
    )

    print(
        "  Global space     : Stage 02 native CT"
    )

    print(
        "  Physical space   : Stage 05 patch geometry"
    )

    print(
        "  Local space      : Stage 05 candidate patch"
    )

    print(
        "  Local shape      : [64,64,64]"
    )

    print(
        "  Grad-CAM space   : LOCAL candidate patch"
    )

    print(
        "  Grad-CAM shape   : [64,64,64]"
    )

    print(
        "  Projection       : Stage 08"
    )

    print()
    print(
        "HEADS"
    )

    for head in FEATURE_NAMES:
        print(
            f"  - {head}"
        )

    # -----------------------------------------------------------------
    # Patient directory.
    # -----------------------------------------------------------------

    patient_dir = os.path.join(
        args.output_root,
        args.patient_id,
    )

    if not os.path.isdir(patient_dir):

        raise FileNotFoundError(
            "Patient output directory not found:\n"
            f"{patient_dir}"
        )

    # -----------------------------------------------------------------
    # Stage 05.
    # -----------------------------------------------------------------

    stage05_candidates = [
        os.path.join(
            patient_dir,
            "stage05_classifier_patches",
        ),
        os.path.join(
            patient_dir,
            "05_classifier_patches",
        ),
        os.path.join(
            patient_dir,
            "05_extract_candidate_patches",
        ),
        os.path.join(
            patient_dir,
            "05",
        ),
    ]

    stage05_dir = None

    for candidate in stage05_candidates:

        if os.path.isdir(candidate):

            stage05_dir = candidate
            break

    if stage05_dir is None:

        raise FileNotFoundError(
            "Could not locate Stage 05 output directory "
            f"under:\n{patient_dir}"
        )

    # -----------------------------------------------------------------
    # Stage 06.
    # -----------------------------------------------------------------

    stage06_candidates = [
        os.path.join(
            patient_dir,
            "06_classification",
        ),
        os.path.join(
            patient_dir,
            "stage06_classification",
        ),
    ]

    stage06_dir = None

    for candidate in stage06_candidates:

        classification_path = os.path.join(
            candidate,
            "classification.json",
        )

        if os.path.isfile(
            classification_path
        ):

            stage06_dir = candidate
            break

    # -----------------------------------------------------------------
    # Stage 07.
    # -----------------------------------------------------------------

    stage07_dir = os.path.join(
        patient_dir,
        "07_gradcam",
    )

    os.makedirs(
        stage07_dir,
        exist_ok=True,
    )

    print()
    print(
        f"Stage 05    : {stage05_dir}"
    )

    print(
        f"Stage 06    : "
        f"{stage06_dir if stage06_dir else 'not found'}"
    )

    print(
        f"Stage 07    : {stage07_dir}"
    )

    # -----------------------------------------------------------------
    # Load candidates.
    #
    # Stage 06 preferred.
    # Stage 05 fallback.
    # -----------------------------------------------------------------

    records = None

    if stage06_dir is not None:

        records = load_stage06_classification(
            stage06_dir
        )

        records_source = (
            "Stage 06 classification.json"
        )

    else:

        print()
        print(
            "Stage 06 classification.json not found."
        )

        print(
            "Falling back to Stage 05 manifest."
        )

        records = load_stage05_records(
            stage05_dir
        )

        records_source = (
            "Stage 05 manifest"
        )

    print(
        f"Candidate source: {records_source}"
    )

    print(
        f"Candidates      : {len(records)}"
    )

    # Stage 05 is the geometry authority. Stage 06 supplies predictions,
    # but it must never be allowed to replace Stage 05 spatial metadata.
    stage05_records = load_stage05_records(stage05_dir)
    stage05_by_id = {}
    for idx, stage05_record in enumerate(stage05_records):
        sid = record_identifier(stage05_record, idx)
        stage05_by_id[sid] = stage05_record

    print(f"Stage 05 geometry records: {len(stage05_by_id)}")

    if args.max_candidates is not None:

        records = records[
            :args.max_candidates
        ]

        print(
            f"Limited to      : {len(records)}"
        )

    # -----------------------------------------------------------------
    # Model.
    # -----------------------------------------------------------------

    print()
    print(
        "Loading trained classifier..."
    )

    model = load_project_model(
        args.checkpoint,
        device,
    )

    target_layer, target_layer_name = (
        find_gradcam_target_layer(
            model,
            preferred_stage=args.gradcam_layer,
        )
    )

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    results = []

    # -----------------------------------------------------------------
    # Candidate loop.
    # -----------------------------------------------------------------

    try:

        for index, record in enumerate(records):

            candidate_id = record_identifier(
                record,
                index,
            )

            print()
            print(
                f"[{index + 1}/{len(records)}] "
                f"Candidate {candidate_id}"
            )

            try:

                # =====================================================
                # PATCH
                # =====================================================

                patch_path = resolve_patch_path(
                    record,
                    stage05_dir,
                )

                if patch_path is None:

                    raise FileNotFoundError(
                        "Could not resolve Stage 05 patch path."
                    )

                print(
                    f"  Patch: {patch_path}"
                )

                patch = load_patch(
                    patch_path
                )

                print(
                    f"  Patch shape: {patch.shape}"
                )

                print(
                    f"  Patch range: "
                    f"{patch.min():.5f} .. "
                    f"{patch.max():.5f}"
                )

                # =====================================================
                # GEOMETRY
                # =====================================================

                stage05_geometry_record = stage05_by_id.get(candidate_id)
                if stage05_geometry_record is None:
                    raise RuntimeError(
                        f"No Stage 05 geometry record for candidate {candidate_id}."
                    )

                # Merge only for geometry extraction. Stage 05 remains the
                # authoritative source; Stage 06 values cannot overwrite it.
                geometry_input = dict(record)
                geometry_input["stage05_geometry"] = stage05_geometry_record

                geometry = extract_crop_geometry(
                    geometry_input
                )

                spatial_context = (
                    build_spatial_context(
                        geometry
                    )
                )

                print()
                print(
                    "  COORDINATES"
                )

                print(
                    "    Global candidate center ZYX: "
                    f"{geometry['candidate_center_zyx']}"
                )

                print(
                    "    Physical patch center ZYX mm: "
                    f"{geometry['patch_center_physical_zyx_mm']}"
                )

                print(
                    "    Local patch center ZYX: "
                    f"{geometry['local_patch_center_zyx']}"
                )

                print(
                    "    Local shape ZYX: "
                    f"{geometry['patch_shape_zyx']}"
                )

                print(
                    "    Patch spacing ZYX mm: "
                    f"{geometry['patch_spacing_zyx_mm']}"
                )

                print(
                    "    Physical FOV ZYX mm: "
                    f"{geometry['patch_fov_zyx_mm']}"
                )

                # =====================================================
                # CLASSIFIER INPUT
                #
                # IMPORTANT:
                #
                # Stage 05 already normalized the patch.
                #
                # No normalization here.
                # =====================================================

                input_tensor = (
                    patch_to_classifier_tensor(
                        patch,
                        device,
                    )
                )

                # Grad-CAM requires gradients.
                input_tensor.requires_grad_()

                # =====================================================
                # ALL HEADS
                # =====================================================

                all_head_results = (
                    generate_all_head_gradcams(
                        gradcam=gradcam,
                        patch=input_tensor,
                    )
                )

                print()
                print(
                    "  HEAD RESULTS"
                )

                for head in FEATURE_NAMES:

                    result = all_head_results[
                        head
                    ]

                    print(
                        f"    {head:14s} "
                        f"prediction="
                        f"{result['prediction']:.5f} "
                        f"raw="
                        f"{result['raw_cam_shape']} "
                        f"local="
                        f"{result['heatmap'].shape}"
                    )

                # =====================================================
                # VISUALIZATION-ONLY HU
                # =====================================================

                patch_hu = normalized_to_hu(
                    patch
                )

                # =====================================================
                # OUTPUT DIRECTORIES
                # =====================================================

                candidate_dir = os.path.join(
                    stage07_dir,
                    f"candidate_{candidate_id}",
                )

                gradcam_dir = os.path.join(
                    candidate_dir,
                    "gradcam",
                )

                montage_dir = os.path.join(
                    candidate_dir,
                    "montages",
                )

                os.makedirs(
                    gradcam_dir,
                    exist_ok=True,
                )

                os.makedirs(
                    montage_dir,
                    exist_ok=True,
                )

                # =====================================================
                # SAVE EACH HEAD SEPARATELY
                # =====================================================

                head_metadata = {}

                for head in FEATURE_NAMES:

                    head_result = (
                        all_head_results[head]
                    )

                    heatmap = (
                        head_result["heatmap"]
                    )

                    prediction = (
                        head_result["prediction"]
                    )

                    raw_cam_shape = (
                        head_result[
                            "raw_cam_shape"
                        ]
                    )

                    # -------------------------------------------------
                    # CAM
                    # -------------------------------------------------

                    heatmap_path = os.path.join(
                        gradcam_dir,
                        f"{head}.npy",
                    )

                    save_gradcam_volume(
                        heatmap,
                        heatmap_path,
                    )

                    # -------------------------------------------------
                    # Peak local slices
                    # -------------------------------------------------

                    slices = choose_peak_slices(
                        heatmap,
                        count=args.peak_slices,
                    )

                    # -------------------------------------------------
                    # Montage
                    # -------------------------------------------------

                    montage_path = os.path.join(
                        montage_dir,
                        f"{head}.png",
                    )

                    save_montage(
                        patch_hu=patch_hu,
                        heatmap=heatmap,
                        slices=slices,
                        output_path=montage_path,
                        title=(
                            f"{args.patient_id} | "
                            f"Candidate {candidate_id} | "
                            f"{head} = "
                            f"{prediction:.3f}"
                        ),
                    )

                    head_metadata[head] = {
                        "prediction":
                            prediction,

                        "raw_cam_shape_zyx":
                            raw_cam_shape,

                        "output_shape_zyx":
                            list(
                                heatmap.shape
                            ),

                        "gradcam_space":
                            "candidate_patch",

                        "coordinate_order":
                            "ZYX",

                        "peak_slices_local_z":
                            slices,

                        "heatmap_path":
                            os.path.abspath(
                                heatmap_path
                            ),

                        "montage_path":
                            os.path.abspath(
                                montage_path
                            ),
                    }

                # =====================================================
                # METADATA
                # =====================================================

                metadata = {
                    "stage": 7,

                    "patient_id":
                        args.patient_id,

                    "candidate_id":
                        candidate_id,

                    "candidate_index":
                        record.get(
                            "candidate_index",
                            candidate_id,
                        ),

                    "checkpoint":
                        os.path.abspath(
                            args.checkpoint
                        ),

                    "target_layer":
                        target_layer_name,

                    "coordinate_convention":
                        "ZYX",

                    # -------------------------------------------------
                    # Explicit spatial contract.
                    # -------------------------------------------------

                    "spatial_context":
                        spatial_context,

                    "geometry":
                        geometry,

                    # Flat copies make the Stage 07 -> Stage 08 contract
                    # robust to older readers that do not inspect nested
                    # metadata. These values are copied from Stage 05.
                    "native_volume_shape_zyx":
                        geometry.get("native_volume_shape_zyx"),
                    "native_spacing_zyx_mm":
                        geometry.get("native_spacing_zyx_mm"),
                    "native_origin_zyx_mm":
                        geometry.get("native_origin_zyx_mm"),
                    "candidate_center_zyx":
                        geometry.get("candidate_center_zyx"),
                    "patch_center_physical_zyx_mm":
                        geometry.get("patch_center_physical_zyx_mm"),
                    "patch_spacing_zyx_mm":
                        geometry.get("patch_spacing_zyx_mm"),
                    "local_patch_center_zyx":
                        geometry.get("local_patch_center_zyx"),
                    "patch_shape_zyx":
                        geometry.get("patch_shape_zyx"),
                    "stage02_crop_offset_zyx":
                        geometry.get("stage02_crop_offset_zyx", [0, 0, 0]),

                    # -------------------------------------------------
                    # Global coordinates.
                    # -------------------------------------------------

                    "global": {
                        "space":
                            "stage02_native_ct",

                        "shape_zyx":
                            geometry.get(
                                "native_volume_shape_zyx"
                            ),

                        "candidate_center_zyx":
                            geometry[
                                "candidate_center_zyx"
                            ],

                        "spacing_zyx_mm":
                            geometry.get(
                                "native_spacing_zyx_mm"
                            ),
                    },

                    # -------------------------------------------------
                    # Physical coordinates.
                    # -------------------------------------------------

                    "physical": {
                        "space":
                            "patient_physical_patch_space",

                        "center_zyx_mm":
                            geometry.get(
                                "patch_center_physical_zyx_mm"
                            ),

                        "spacing_zyx_mm":
                            geometry[
                                "patch_spacing_zyx_mm"
                            ],

                        "fov_zyx_mm":
                            geometry[
                                "patch_fov_zyx_mm"
                            ],

                        "center_source":
                            geometry[
                                "physical_center_source"
                            ],
                    },

                    # -------------------------------------------------
                    # Local classifier coordinates.
                    # -------------------------------------------------

                    "local": {
                        "space":
                            "candidate_patch",

                        "shape_zyx":
                            geometry[
                                "patch_shape_zyx"
                            ],

                        "center_zyx":
                            geometry[
                                "local_patch_center_zyx"
                            ],

                        "coordinate_range_zyx": [
                            [0, PATCH_SIZE - 1],
                            [0, PATCH_SIZE - 1],
                            [0, PATCH_SIZE - 1],
                        ],

                        "spacing_zyx_mm":
                            geometry[
                                "patch_spacing_zyx_mm"
                            ],
                    },

                    # -------------------------------------------------
                    # Grad-CAM coordinates.
                    #
                    # Same local coordinate system as classifier patch.
                    # -------------------------------------------------

                    "gradcam": {
                        "space":
                            "candidate_patch",

                        "coordinate_order":
                            "ZYX",

                        "shape_zyx": [
                            PATCH_SIZE,
                            PATCH_SIZE,
                            PATCH_SIZE,
                        ],

                        "center_zyx":
                            geometry[
                                "local_patch_center_zyx"
                            ],

                        "coordinate_range_zyx": [
                            [0, PATCH_SIZE - 1],
                            [0, PATCH_SIZE - 1],
                            [0, PATCH_SIZE - 1],
                        ],

                        "projection_to_global":
                            "stage08",
                    },

                    # -------------------------------------------------
                    # Explicit transform.
                    # -------------------------------------------------

                    "coordinate_transform":
                        spatial_context[
                            "transform"
                        ],

                    # -------------------------------------------------
                    # Per-head CAMs.
                    # -------------------------------------------------

                    "heads":
                        head_metadata,

                    # -------------------------------------------------
                    # Classifier preprocessing.
                    # -------------------------------------------------

                    "classifier_input": {
                        "shape_bchw_dhw": [
                            1,
                            1,
                            PATCH_SIZE,
                            PATCH_SIZE,
                            PATCH_SIZE,
                        ],

                        "shape_zyx": [
                            PATCH_SIZE,
                            PATCH_SIZE,
                            PATCH_SIZE,
                        ],

                        "value_range":
                            "[0,1]",

                        "normalization":
                            "(HU + 1000) / 1400",

                        "stage05_normalized":
                            True,

                        "stage07_normalized_again":
                            False,

                        "visualization_hu_conversion":
                            "HU = normalized * 1400 - 1000",
                    },

                    # -------------------------------------------------
                    # Original Stage 06 / Stage 05 record.
                    # -------------------------------------------------

                    "record":
                        record,
                }

                metadata_path = os.path.join(
                    candidate_dir,
                    "metadata.json",
                )

                with open(
                    metadata_path,
                    "w",
                    encoding="utf-8",
                ) as file:

                    json.dump(
                        metadata,
                        file,
                        indent=2,
                        default=json_default,
                    )

                results.append(
                    metadata
                )

                # =====================================================
                # PER-CANDIDATE gradcam_summary.json
                #
                # Required by the LungInsight Stage 07 spec: one
                # summary file per candidate, one entry per head,
                # each entry following the mandated flat schema.
                # =====================================================

                candidate_gradcam_summary = {
                    head: {
                        "candidate_id":
                            candidate_id,

                        "head":
                            head,

                        "coordinate_order":
                            "ZYX",

                        "gradcam_space":
                            "candidate_patch",

                        "shape_zyx": [
                            PATCH_SIZE,
                            PATCH_SIZE,
                            PATCH_SIZE,
                        ],

                        "local_center_zyx":
                            LOCAL_CENTER_ZYX,

                        "value_range":
                            "[0,1]",

                        "target_layer":
                            target_layer_name,

                        "source_patch":
                            patch_path,

                        "projection_authority":
                            "stage08",
                    }
                    for head in FEATURE_NAMES
                }

                candidate_gradcam_summary_path = os.path.join(
                    candidate_dir,
                    "gradcam_summary.json",
                )

                with open(
                    candidate_gradcam_summary_path,
                    "w",
                    encoding="utf-8",
                ) as file:

                    json.dump(
                        candidate_gradcam_summary,
                        file,
                        indent=2,
                        default=json_default,
                    )

                print()
                print(
                    "  Saved independent CAMs:"
                )

                for head in FEATURE_NAMES:

                    print(
                        f"    {head}: "
                        f"{head_metadata[head]['heatmap_path']}"
                    )

                print(
                    f"  Metadata: "
                    f"{metadata_path}"
                )

            except Exception as exc:

                print(
                    f"  [ERROR] Candidate "
                    f"{candidate_id}: {exc}"
                )

    finally:

        gradcam.remove()

    # -----------------------------------------------------------------
    # Summary.
    # -----------------------------------------------------------------

    # NOTE: this is the run-level rollup across all candidates, kept
    # separate from each candidate's own gradcam_summary.json (which
    # is the file mandated by the Stage 07 spec, written per-head,
    # per-candidate, above).
    summary_path = os.path.join(
        stage07_dir,
        "run_summary.json",
    )

    summary = {
        "stage": 7,

        "description": (
            "LungInsight candidate-level independent "
            "multi-head Grad-CAM in local Stage 05 "
            "candidate-patch coordinates."
        ),

        "patient_id":
            args.patient_id,

        "checkpoint":
            os.path.abspath(
                args.checkpoint
            ),

        "target_layer":
            target_layer_name,

        "coordinate_convention":
            "ZYX",

        "heads":
            FEATURE_NAMES,

        "spatial_contract": {
            "global": {
                "space":
                    "stage02_native_ct",

                "description":
                    "Full native CT coordinate space.",
            },

            "physical": {
                "space":
                    "patient_physical_patch_space",

                "description":
                    "Continuous physical coordinates "
                    "defined by Stage 05 geometry.",
            },

            "local": {
                "space":
                    "candidate_patch",

                "shape_zyx": [
                    PATCH_SIZE,
                    PATCH_SIZE,
                    PATCH_SIZE,
                ],

                "description":
                    "Stage 05 classifier patch.",
            },

            "gradcam": {
                "space":
                    "candidate_patch",

                "shape_zyx": [
                    PATCH_SIZE,
                    PATCH_SIZE,
                    PATCH_SIZE,
                ],

                "description":
                    "Independent per-head Grad-CAM "
                    "within local candidate-patch space.",
            },

            "projection_to_global":
                "stage08",

            "geometry_authority":
                "stage05",
        },

        "preprocessing": {
            "stage05_patch_already_normalized":
                True,

            "stage07_second_normalization":
                False,

            "classifier_value_range":
                "[0,1]",

            "normalization":
                "(HU + 1000) / 1400",

            "visualization_conversion":
                "HU = normalized * 1400 - 1000",
        },

        "num_candidates_requested":
            len(records),

        "num_candidates_processed":
            len(results),

        "results":
            results,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            default=json_default,
        )

    # -----------------------------------------------------------------
    # Final report.
    # -----------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "STAGE 07 COMPLETE"
    )
    print("=" * 72)

    print(
        f"Candidates processed : {len(results)}"
    )

    print(
        f"Heads per candidate  : {len(FEATURE_NAMES)}"
    )

    print(
        f"Target layer         : {target_layer_name}"
    )

    print(
        "Local patch          : [64,64,64]"
    )

    print(
        "Grad-CAM             : [64,64,64] per head"
    )

    print(
        "Grad-CAM coordinate  : LOCAL candidate-patch Z,Y,X"
    )

    print(
        "Global projection    : Stage 08"
    )

    print()
    print(
        f"Output directory     : {stage07_dir}"
    )

    print(
        f"Summary              : {summary_path}"
    )


if __name__ == "__main__":
    main()