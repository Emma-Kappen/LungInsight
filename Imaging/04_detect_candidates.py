"""
04_detect_candidates.py

LungInsight Stage 04 -- Pretrained ViTDet3D Candidate Detection
=================================================================

This is a corrected version of the previous Stage 04 script, brought
into compliance with LungInsight_Imaging_Pipeline_Architecture.pdf.

WHAT CHANGED AND WHY (audit summary)
-------------------------------------

1. LoG is no longer fused into the authoritative candidate list.

   The architecture is explicit: "Stage 04 = pretrained ViTDet3D
   lesion/nodule detection. It should not be replaced by LoG,
   handcrafted blob detection, or heuristic candidate generation.
   Those techniques may be retained only as: diagnostic tools,
   fallback analysis, research comparison baselines."

   The previous version ran ViTDet3D and LoG in parallel and then
   spatially FUSED them with `fuse_detectors()` -- LoG-only detections
   (no ViTDet3D support at all) were written into the primary
   `candidates.json` with equal standing to ViTDet3D detections, and
   the fused/weighted-average center could pull a ViTDet3D center
   away from what the detector actually predicted. That makes LoG a
   co-equal detector, not a diagnostic baseline, and Stage 05 (which
   trusts `candidates.json` as ground truth for patch centers) had no
   way to tell a LoG-only guess from a real ViTDet3D proposal.

   Fixed: `candidates.json` (the file Stage 05 consumes) now contains
   ONLY ViTDet3D proposals. LoG still runs (optional, on by default)
   but is written to its own clearly-labeled diagnostic file, plus a
   read-only agreement report comparing the two -- neither of which
   is ever merged back into `candidates.json`.

2. Proposal-center authority is preserved through de-duplication.

   The architecture: "The detector proposal center (`center_zyx`)
   must serve as the authoritative downstream center for patch
   cropping. Do not recalculate or re-center candidates on
   bounding-box midpoints."

   ViTDet3D's sliding-window inference legitimately produces multiple
   overlapping detections of the same physical nodule (75% window
   overlap by design). The previous version collapsed those with a
   *score-weighted average* of centers -- which is itself a kind of
   silent re-centering, just averaged across windows instead of a
   single bbox midpoint. Fixed: de-duplication is now plain
   greedy NMS by physical distance -- the highest-scoring window's
   raw, unmodified `center_zyx` is kept verbatim, and only the other
   (lower-scoring, redundant) window detections in that neighborhood
   are discarded. No coordinate is ever synthesized or averaged.

3. Output schema now matches the required candidate record exactly.

   `candidates.json` records now contain the fields the architecture
   specifies: `candidate_id`, `source`, `detector_score`,
   `coordinate_order`, `center_zyx`, `bbox_start_zyx`, `bbox_end_zyx`,
   `bbox_size_zyx_voxels`, `spacing_zyx_mm`, `space`, and `model`
   (name / checkpoint / model_version). The previous schema used a
   nested `bbox_zyx: [[low], [high]]` pair, had no per-candidate
   `space`/`coordinate_order`/`spacing_zyx_mm`/`model` fields, and
   mixed in ViT- and LoG-specific fields (`vit_score`, `log_score`,
   `vit_logit`, `log_scale_mm`, `sources`) that don't belong in a
   single-detector authoritative record.

4. Stage separation / spatial tracking.

   `space` is now stamped on every candidate as
   `"stage02_native_ct"`, and `04_candidates/detector_metadata.json`
   records the exact Stage 02 volume shape and spacing the detector
   ran against, so a candidate's `center_zyx` can always be traced
   back to Stage 02 native voxel space without ambiguity (per
   "Retain coordinate mappings to ensure candidate positions align
   directly with stage02_native_ct").

   Note on Stage 03: this script still reads directly from the
   Stage 02 output directory and performs ViTDet3D's own windowing/
   normalization inline, rather than consuming a separately-persisted
   `03_detector_input/` tensor. That is unchanged from the previous
   version. If a real Stage 03 script/output directory exists in your
   deployment, prefer loading its persisted tensor + metadata.json
   instead of re-deriving it here, and validate its
   `input_shape_zyx`/`padding_*_zyx` against `VIT_WINDOW` before
   inference -- the hook point is `load_stage02()` /
   `run_vitdet3d_raw()` below.

5. Output layout matches the architecture's file tree.

   `output/<PATIENT>/04_candidates/` now contains `candidates.json`
   (authoritative, ViTDet3D-only) and `detector_metadata.json`
   (provenance -- matching the architecture doc's named files)
   alongside the additional diagnostic-only files
   `log_candidates_diagnostic.json`,
   `candidates_diagnostic_agreement.json`, and `candidates.csv` (a
   flat, human-readable mirror of `candidates.json`, not a second
   source of truth).

COORDINATE CONVENTION
----------------------

All internal voxel coordinates are (Z, Y, X).
All physical coordinates are (Z, Y, X) in millimetres.
PyTorch tensors are (B, C, Z, Y, X) / (N, C, Dz, Dy, Dx).
This is never silently transposed to X,Y,Z anywhere in this file.

VITDET3D
--------

The pretrained detector used here is:

    rlsn/DeTr4LungNodule

a 3D ViT detector trained on LUNA16. Its detector input is
40 x 128 x 128 voxels, and its published implementation normalizes
with:

    mean = -775.657161489884
    std  =  962.3208802005623

These values are used ONLY for ViTDet3D. They must NOT be reused by
the Stage 06 classifier, which instead uses
`clip(HU, -1000, 400)` then `(HU + 1000) / 1400` (applied in Stage 05,
not here).

INPUT
-----

Expected Stage 02 directory:

    output/<PATIENT>/02/
        volume_hu.npy
        lung_mask.npy
        meta.json

OUTPUT
------

    output/<PATIENT>/04_candidates/
        candidates.json                     <- AUTHORITATIVE (ViTDet3D only)
        candidates.csv                      <- flat mirror of the above
        detector_metadata.json              <- provenance / run config
        log_candidates_diagnostic.json      <- LoG, diagnostic only
        candidates_diagnostic_agreement.json<- ViTDet3D vs LoG comparison

Only `candidates.json` should ever be read by Stage 05.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_laplace, maximum_filter


# ============================================================================
# CONSTANTS
# ============================================================================

COORDINATE_ORDER = "ZYX"
STAGE02_SPACE = "stage02_native_ct"

# ---------------------------------------------------------------------------
# ViTDet3D input geometry
# ---------------------------------------------------------------------------

VIT_WINDOW = np.asarray([40, 128, 128], dtype=np.int32)

# Original rlsn implementation uses 75% overlap:
#   40  * 0.75 = 30
#   128 * 0.75 = 96
VIT_STRIDE = np.asarray([30, 96, 96], dtype=np.int32)

VIT_MEAN = -775.657161489884
VIT_STD = 962.3208802005623

# The original detector evaluation code used a raw logit threshold of
# -5.0 (~0.0067 probability). We retain this permissive value because
# Stage 04 is intended to generate high-recall candidates; precision
# is handled downstream (Stage 06).
DEFAULT_VIT_LOGIT_THRESHOLD = -5.0

# Distance (mm) used to de-duplicate overlapping-window ViTDet3D
# detections of the same physical lesion. This is NMS, not fusion --
# see module docstring point 2. It only ever discards redundant
# lower-scoring detections; it never averages or recomputes a center.
DEFAULT_NMS_DISTANCE_MM = 10.0

# ---------------------------------------------------------------------------
# LoG (diagnostic / research-comparison detector only -- NOT part of
# the authoritative candidate list; see module docstring point 1).
# ---------------------------------------------------------------------------

LOG_DIAMETERS_MM = (4.0, 6.0, 8.0, 10.0, 14.0, 18.0, 24.0)
DEFAULT_LOG_THRESHOLD = 0.035
DEFAULT_LOG_MIN_DISTANCE_MM = 5.0

# Distance (mm) within which a LoG diagnostic detection is considered
# to "agree" with a ViTDet3D candidate, purely for reporting purposes.
DEFAULT_DIAGNOSTIC_AGREEMENT_DISTANCE_MM = 10.0

# ---------------------------------------------------------------------------
# CT values
# ---------------------------------------------------------------------------

AIR_HU = -1000.0


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DetectorCandidate:
    """
    The authoritative Stage 04 candidate record.

    Field set matches the architecture's Stage 04 candidate schema
    exactly (see LungInsight_Imaging_Pipeline_Architecture.pdf,
    "Recommended record" / "B. ViTDet3D Candidate Record"). This is
    ONLY ever populated from ViTDet3D output -- see module docstring.
    """

    candidate_id: int
    source: str                      # always "ViTDet3D"
    detector_score: float
    coordinate_order: str            # always "ZYX"
    center_zyx: List[float]
    bbox_start_zyx: List[int]
    bbox_end_zyx: List[int]
    bbox_size_zyx_voxels: List[int]
    spacing_zyx_mm: List[float]
    space: str                       # always "stage02_native_ct"
    model: Dict[str, Any]            # {"name", "checkpoint", "model_version"}

    # Extra, non-schema-breaking provenance kept alongside (additive
    # fields; every field the architecture requires is still present
    # and unambiguous). Downstream code that only reads the required
    # keys is unaffected.
    raw_logit: Optional[float] = None
    center_mm_zyx: Optional[List[float]] = None
    diameter_mm: Optional[float] = None


@dataclass
class LogDiagnosticCandidate:
    """
    A LoG (Laplacian-of-Gaussian) detection.

    This is explicitly a *diagnostic / research-comparison* record,
    never an authoritative candidate -- it is written to its own file
    and is never merged into `candidates.json`.
    """

    diagnostic_id: str
    purpose: str                     # always "research_comparison_only"
    coordinate_order: str
    center_zyx: List[float]
    center_mm_zyx: List[float]
    scale_mm: float
    log_response: float
    space: str


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def json_default(obj: Any):
    """JSON serializer for numpy values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=json_default)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# STAGE 02 INPUT LOADING
# ============================================================================

def _find_first(obj: Dict[str, Any], keys: Sequence[str]):
    """Find the first matching key in a metadata dictionary."""
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def load_stage02(
    stage02_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the Stage 02 volume, lung mask, and geometry.

    Returns
    -------
    volume_hu      : (Z, Y, X) float32
    lung_mask      : (Z, Y, X) bool
    spacing_zyx    : (3,) float32, mm/voxel
    origin_zyx     : (3,) float32, physical coordinate of voxel [0,0,0]
    metadata       : original Stage 02 metadata dict
    """

    volume_path = os.path.join(stage02_dir, "volume_hu.npy")
    mask_path = os.path.join(stage02_dir, "lung_mask.npy")
    meta_path = os.path.join(stage02_dir, "meta.json")

    if not os.path.isfile(volume_path):
        raise FileNotFoundError(f"Missing Stage 02 volume:\n{volume_path}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Missing Stage 02 lung mask:\n{mask_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing Stage 02 metadata:\n{meta_path}")

    volume = np.asarray(np.load(volume_path, allow_pickle=False), dtype=np.float32)
    mask = np.asarray(np.load(mask_path, allow_pickle=False))
    metadata = load_json(meta_path)

    if volume.ndim != 3:
        raise RuntimeError(f"Expected volume shape (Z,Y,X), got {volume.shape}")
    if mask.shape != volume.shape:
        raise RuntimeError(
            f"Volume/mask shape mismatch:\nvolume={volume.shape}\nmask={mask.shape}"
        )
    mask = mask.astype(bool)

    # ------------------------------------------------------------------
    # Spacing. Tolerate a few common metadata layouts, but never
    # silently transpose ZYX <-> XYZ without an explicit declared order.
    # ------------------------------------------------------------------
    spacing_value = _find_first(
        metadata,
        ["spacing_zyx_mm", "spacing_zyx", "spacing", "voxel_spacing", "spacing_xyz"],
    )
    if spacing_value is None:
        raise RuntimeError(
            "Could not find voxel spacing in meta.json.\n"
            "Expected one of: spacing_zyx_mm, spacing_zyx, spacing, "
            "voxel_spacing, spacing_xyz."
        )
    spacing = np.asarray(spacing_value, dtype=np.float32)
    if spacing.shape != (3,):
        raise RuntimeError(f"Invalid spacing shape: {spacing.shape}")

    spacing_order = metadata.get(
        "spacing_order", metadata.get("coordinate_order", "Z,Y,X")
    )
    if str(spacing_order).upper().replace(" ", "") in ("X,Y,Z", "XYZ"):
        spacing = spacing[::-1]

    # ------------------------------------------------------------------
    # Origin. Zero is correct if Stage 02 defines no physical origin.
    # ------------------------------------------------------------------
    origin_value = _find_first(metadata, ["origin_mm", "origin_zyx", "origin", "origin_xyz"])
    if origin_value is None:
        origin = np.zeros(3, dtype=np.float32)
    else:
        origin = np.asarray(origin_value, dtype=np.float32)
        if origin.shape != (3,):
            raise RuntimeError(f"Invalid origin shape: {origin.shape}")
        origin_order = metadata.get(
            "origin_order", metadata.get("coordinate_order", "Z,Y,X")
        )
        if str(origin_order).upper().replace(" ", "") in ("X,Y,Z", "XYZ"):
            origin = origin[::-1]

    return volume, mask, spacing, origin, metadata


# ============================================================================
# GEOMETRY
# ============================================================================

def voxel_to_mm(
    center_zyx: np.ndarray, spacing_zyx: np.ndarray, origin_zyx: np.ndarray
) -> np.ndarray:
    """Convert Z,Y,X voxel coordinates to Z,Y,X physical coordinates."""
    return origin_zyx + center_zyx * spacing_zyx


def bbox_from_center(
    center_zyx: np.ndarray, diameter_mm: float, spacing_zyx: np.ndarray
) -> np.ndarray:
    """Cubic voxel-space bounding box around a center, given a physical diameter."""
    radius_vox = (diameter_mm / 2.0) / spacing_zyx
    low = center_zyx - radius_vox
    high = center_zyx + radius_vox
    return np.concatenate([low, high])


def euclidean_mm(a_zyx: np.ndarray, b_zyx: np.ndarray, spacing_zyx: np.ndarray) -> float:
    """Physical Euclidean distance between two VOXEL-space centers."""
    delta = (a_zyx - b_zyx) * spacing_zyx
    return float(np.linalg.norm(delta))


def euclidean_mm_physical(a_mm: np.ndarray, b_mm: np.ndarray) -> float:
    """Physical Euclidean distance between two already-physical (mm) centers."""
    return float(np.linalg.norm(np.asarray(a_mm) - np.asarray(b_mm)))


# ============================================================================
# VITDET3D WINDOWING
# ============================================================================

def make_vit_windows(
    volume: np.ndarray, window_size: np.ndarray, stride: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate fixed-size ViTDet3D sliding windows.

    Returns
    -------
    offsets : (N, 3) int, Z,Y,X global voxel offsets into the (possibly
              air-padded) volume
    windows : (N, D, H, W) float32

    Air padding (AIR_HU) is used if the Stage 02 volume is smaller than
    the detector window along any axis.
    """
    shape = np.asarray(volume.shape, dtype=np.int32)
    padded_shape = np.maximum(shape, window_size)

    padded = np.full(tuple(padded_shape), AIR_HU, dtype=np.float32)
    padded[: shape[0], : shape[1], : shape[2]] = volume

    offsets_per_axis = []
    for dim in range(3):
        max_start = padded_shape[dim] - window_size[dim]
        if max_start <= 0:
            offsets = np.asarray([0], dtype=np.int32)
        else:
            offsets = np.arange(0, max_start + 1, stride[dim], dtype=np.int32)
            if offsets[-1] != max_start:
                offsets = np.concatenate([offsets, np.asarray([max_start], dtype=np.int32)])
        offsets_per_axis.append(offsets)

    offsets, windows = [], []
    for z in offsets_per_axis[0]:
        for y in offsets_per_axis[1]:
            for x in offsets_per_axis[2]:
                offsets.append([z, y, x])
                windows.append(
                    padded[
                        z : z + window_size[0],
                        y : y + window_size[1],
                        x : x + window_size[2],
                    ]
                )

    return (
        np.asarray(offsets, dtype=np.int32),
        np.asarray(windows, dtype=np.float32),
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


# ============================================================================
# VITDET3D MODEL DEFINITION
#
# The published checkpoint (rlsn/DeTr4LungNodule on the Hugging Face Hub)
# is NOT a class that ships with the `transformers` library -- "VitDet3D"
# has never existed in `transformers`, and the Hub repo contains no
# modeling_*.py / auto_map, so `trust_remote_code=True` cannot load it
# either. It is a custom architecture, defined in the author's own
# `model.py` (github.com/rlsn/LungNoduleDetection), that happens to
# subclass `transformers.PreTrainedModel` with `config_class = ViTConfig`.
# Subclassing PreTrainedModel is exactly what makes
# `VitDet3D.from_pretrained(repo_id)` work -- but only once this exact
# class is defined locally, the same way the author's own training code
# defined it. This is a straight port of that class, kept 1:1 with the
# upstream implementation so the published pytorch_model.bin state_dict
# loads without key mismatches.
#
# Confirmed against the checkpoint's own config.json:
#     "architectures": ["VitDet3D"], "model_type": "vit"
#     "image_size": [40, 128, 128], "patch_size": [4, 16, 16]
# ("image_size"/"patch_size" here are exactly VIT_WINDOW's [D,H,W] shape.)
#
# There is also no preprocessor_config.json in the repo, so
# AutoImageProcessor was never going to work either -- normalization is
# (and always was) handled manually via VIT_MEAN / VIT_STD above.
# ============================================================================

from transformers import PreTrainedModel, ViTConfig  # noqa: E402
from transformers.utils import ModelOutput  # noqa: E402

try:
    from transformers.activations import ACT2FN
except ImportError:  # pragma: no cover - extremely old/new transformers
    ACT2FN = {"gelu": nn.GELU()}


# ----------------------------------------------------------------------
# Self-contained ViT transformer block, ported 1:1 (module names and all)
# from transformers==4.34.0's transformers/models/vit/modeling_vit.py --
# the version this checkpoint was trained and saved with.
#
# We do NOT import ViTEncoder/ViTLayer/ViTPooler from the *installed*
# `transformers` package here. Those internals are not a stable public
# API and have already changed shape upstream: `ViTEncoder` has been
# removed entirely in some releases, and `ViTLayer` collapsed its
# `intermediate` + `output` submodules into a single `mlp` submodule in
# others -- either change would silently break loading this checkpoint's
# pytorch_model.bin (wrong or missing state_dict keys) even though the
# import itself might succeed. Pinning our own copy of these classes,
# with the original 4.34.0 module names, keeps this immune to future
# `transformers` refactors.
# ----------------------------------------------------------------------

class _ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


class _ViTSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class _ViTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = _ViTSelfAttention(config)
        self.output = _ViTSelfOutput(config)

    def forward(self, hidden_states):
        self_output = self.attention(hidden_states)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class _ViTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = (
            ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str)
            else config.hidden_act
        )

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class _ViTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class _ViTLayer(nn.Module):
    """Matches transformers==4.34.0's ViTLayer module names exactly."""

    def __init__(self, config):
        super().__init__()
        self.attention = _ViTAttention(config)
        self.intermediate = _ViTIntermediate(config)
        self.output = _ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        attention_output = self.attention(self.layernorm_before(hidden_states))
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)

        return layer_output


class _ViTEncoder3D(nn.Module):
    """Matches transformers==4.34.0's ViTEncoder module names exactly."""

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([_ViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return (hidden_states,)


class _ViTPooler(nn.Module):
    """Matches transformers==4.34.0's ViTPooler module names exactly."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class _ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=[3, 3, 3],
            stride=stride, padding=1, bias=False,
        )
        self.downsample = downsample
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=[3, 3, 3],
            padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class _CNNFeatureExtractor3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = 64
        self.out_size = [3, 8, 8]
        self.conv1 = nn.Conv3d(
            config.num_channels, self.in_channels,
            kernel_size=7, stride=2, padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)

    def _make_layer(self, num_channels, num_layers, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, num_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(num_channels),
            )
        layers = [_ResBlock3D(self.in_channels, num_channels, stride, downsample)]
        self.in_channels = num_channels
        for _ in range(1, num_layers):
            layers.append(_ResBlock3D(self.in_channels, num_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class _PosEmbedding3D(nn.Module):
    def __init__(self, config, in_channels, in_size):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.seq_len = int(np.prod(in_size))
        self.projection = nn.Linear(in_channels, config.hidden_size)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.seq_len + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = x.flatten(2).transpose(1, 2)
        x = self.projection(x)
        embeddings = torch.cat((cls_tokens, x), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class VitDet3D(PreTrainedModel):
    """
    3D CNN-stem + ViT-encoder detector, ported 1:1 from
    rlsn/LungNoduleDetection's model.py so that the published
    rlsn/DeTr4LungNodule checkpoint loads with matching state_dict keys.
    """

    config_class = ViTConfig

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.cnn = _CNNFeatureExtractor3D(config)
        self.embeddings = _PosEmbedding3D(config, self.cnn.in_channels, self.cnn.out_size)
        self.encoder = _ViTEncoder3D(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = _ViTPooler(config) if add_pooling_layer else None
        self.classification_head = _MLP(config.hidden_size, config.num_labels, 3)
        self.bbox_head = _MLP(config.hidden_size, 6, 3)
        self.config = config

        # The original rlsn/LungNoduleDetection model.py never called this.
        # That was harmless under transformers==4.34.0 (the version this
        # checkpoint was trained with), but current transformers releases
        # populate required from_pretrained() bookkeeping here (tied-weight
        # keys, fp32-keep-modules, etc.) and will crash without it.
        self.post_init()

    def forward(self, pixel_values, labels=None, bbox=None):
        feature_maps = self.cnn(pixel_values)
        embeddings = self.embeddings(feature_maps)
        encoder_outputs = self.encoder(embeddings)
        sequence_output = self.layernorm(encoder_outputs[0])
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        logits = self.classification_head(pooled_output)
        bbox_pred = self.bbox_head(pooled_output)

        loss = None
        if labels is not None and bbox is not None:
            loss_bbox_fn = nn.MSELoss(reduction="none")
            if self.config.num_labels == 1:
                loss = nn.BCEWithLogitsLoss()(logits.view(-1), labels.float())
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)
            mask = labels.unsqueeze(-1).bool()
            loss = loss + (loss_bbox_fn(bbox_pred, bbox) * mask).mean()

        return ModelOutput(
            loss=loss,
            logits=logits,
            bbox=bbox_pred,
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


def load_vitdet3d(model_name: str, device: torch.device, revision: Optional[str] = None):
    """
    Load the published ViTDet3D checkpoint.

    "VitDet3D" is not a class in the `transformers` package -- it is
    defined locally above. We instantiate it from the checkpoint's own
    ViTConfig and let `PreTrainedModel.from_pretrained` (inherited, not
    custom) pull down and load the matching weights.

    Returns (model, resolved_commit_hash_or_None). The commit hash, when
    available, is recorded in each candidate's `model.checkpoint` field
    so a run is exactly reproducible (see module docstring point 3/4).
    """

    print()
    print("Loading ViTDet3D:")
    print(f"  {model_name}" + (f" @ {revision}" if revision else ""))

    kwargs = {"revision": revision} if revision else {}

    try:
        model = VitDet3D.from_pretrained(model_name, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load ViTDet3D checkpoint '{model_name}'.\n"
            "This requires network access to huggingface.co and a "
            "transformers install new enough to provide ViTEncoder/"
            "ViTPooler (transformers.models.vit.modeling_vit)."
        ) from exc

    model = model.to(device)
    model.eval()

    resolved_commit = getattr(model, "config", None)
    resolved_commit = getattr(resolved_commit, "_commit_hash", None)

    print("ViTDet3D loaded.")

    # No processor: the repo ships no preprocessor_config.json.
    # Normalization is handled explicitly via VIT_MEAN / VIT_STD.
    return model, resolved_commit


# ============================================================================
# VITDET3D INFERENCE
# ============================================================================

@torch.no_grad()
def run_vitdet3d_raw(
    volume_hu: np.ndarray,
    device: torch.device,
    model,
    batch_size: int,
    logit_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Run ViTDet3D over sliding windows and return RAW, per-window
    detections (voxel-space, in the Stage 02 global frame). No
    de-duplication and no schema formatting happens here -- that keeps
    this function a pure, auditable mapping from
    "ViTDet3D input space" -> "Stage 02 global CT coordinates"
    (architecture requirement: "map ViTDet3D input space back to Stage
    02 global CT coordinates without spatial shift").

    Each returned dict has:
        center_zyx : np.ndarray (3,) float, GLOBAL voxel coordinates
        low_zyx    : np.ndarray (3,) float, GLOBAL voxel coordinates
        high_zyx   : np.ndarray (3,) float, GLOBAL voxel coordinates
        score      : float, sigmoid(logit)
        logit      : float, raw detector logit
    """

    offsets, windows = make_vit_windows(volume_hu, VIT_WINDOW, VIT_STRIDE)
    print()
    print(f"ViTDet3D windows: {len(windows)}")

    volume_shape = np.asarray(volume_hu.shape)
    raw_detections: List[Dict[str, Any]] = []

    for start in range(0, len(windows), batch_size):
        end = min(start + batch_size, len(windows))
        batch = windows[start:end]
        batch_offsets = offsets[start:end]

        # The published ViTDet3D model was trained using LUNA16
        # mean/std normalization. This is ONLY for ViTDet3D -- see
        # module docstring.
        batch = (batch - VIT_MEAN) / VIT_STD

        tensor = torch.from_numpy(batch).unsqueeze(1).to(device=device, dtype=torch.float32)
        outputs = model(pixel_values=tensor)

        logits = outputs.logits.detach().cpu().numpy().reshape(-1)
        bbox = outputs.bbox.detach().cpu().numpy()

        for j in range(len(logits)):
            logit = float(logits[j])
            if logit <= logit_threshold:
                continue

            # ViTDet3D's bbox head has no explicit sigmoid; the
            # published bbox targets are normalized to local-window
            # [0, 1] coordinates, so clip defensively.
            local_bbox = np.clip(bbox[j], 0.0, 1.0)
            local_low, local_high = local_bbox[:3], local_bbox[3:]

            # Protect against malformed/reversed predictions -- this is
            # sanitizing the detector's own bbox, not re-deriving a
            # center from it (the center below always comes from this
            # same bbox prediction, never a bounding-box "midpoint of a
            # separately-cropped patch").
            low = np.minimum(local_low, local_high)
            high = np.maximum(local_low, local_high)

            global_low = batch_offsets[j] + low * VIT_WINDOW
            global_high = batch_offsets[j] + high * VIT_WINDOW
            center = (global_low + global_high) / 2.0

            # Reject predictions whose center falls entirely outside
            # the real (unpadded) Stage 02 volume.
            if np.any(center < 0) or np.any(center >= volume_shape):
                continue

            raw_detections.append(
                {
                    "center_zyx": center,
                    "low_zyx": global_low,
                    "high_zyx": global_high,
                    "score": float(sigmoid(np.asarray([logit]))[0]),
                    "logit": logit,
                }
            )

    print(f"ViTDet3D raw detections (pre-NMS): {len(raw_detections)}")
    return raw_detections


def nms_by_distance(
    raw_detections: List[Dict[str, Any]],
    spacing_zyx: np.ndarray,
    distance_mm: float,
) -> List[Dict[str, Any]]:
    """
    Greedy non-maximum suppression over the SAME detector's overlapping
    sliding-window detections, by physical distance between centers.

    This exists only because ViTDet3D's 75%-overlap sliding windows
    will detect the same physical lesion from several neighboring
    windows. Per the architecture's "Proposal Center Authority" rule,
    a kept detection's `center_zyx` is returned EXACTLY as the detector
    produced it -- never averaged, never recomputed from a bounding-box
    midpoint of some other structure. All this function does is decide
    which single raw detection to keep from a cluster and discard the
    (lower-scoring, redundant) rest.
    """

    if not raw_detections:
        return []

    ordered = sorted(raw_detections, key=lambda d: d["score"], reverse=True)
    kept: List[Dict[str, Any]] = []

    for det in ordered:
        center = det["center_zyx"]
        suppressed = False
        for kept_det in kept:
            if euclidean_mm(center, kept_det["center_zyx"], spacing_zyx) <= distance_mm:
                suppressed = True
                break
        if not suppressed:
            kept.append(det)

    print(f"ViTDet3D detections after NMS: {len(kept)}")
    return kept


def build_detector_candidates(
    kept_detections: List[Dict[str, Any]],
    spacing_zyx: np.ndarray,
    origin_zyx: np.ndarray,
    model_name: str,
    checkpoint_ref: str,
    model_version: str,
) -> List[DetectorCandidate]:
    """
    Format NMS-kept raw ViTDet3D detections into the architecture's
    exact Stage 04 candidate schema. `candidate_id` is assigned here,
    sequentially, after NMS -- this is the ONLY place candidate
    identity is created; nothing downstream should re-derive it.
    """

    # Highest score first, for a stable/reproducible, score-ordered
    # candidate_id assignment.
    ordered = sorted(kept_detections, key=lambda d: d["score"], reverse=True)

    candidates: List[DetectorCandidate] = []
    for candidate_id, det in enumerate(ordered):
        center = det["center_zyx"]
        low = det["low_zyx"]
        high = det["high_zyx"]

        bbox_start = np.floor(low).astype(int)
        bbox_end = np.ceil(high).astype(int)
        bbox_size = np.maximum(bbox_end - bbox_start, 1)

        center_mm = voxel_to_mm(center, spacing_zyx, origin_zyx)
        side_mm = np.maximum((high - low) * spacing_zyx, 0.1)
        diameter_mm = float(np.cbrt(np.prod(side_mm)))

        candidates.append(
            DetectorCandidate(
                candidate_id=candidate_id,
                source="ViTDet3D",
                detector_score=float(det["score"]),
                coordinate_order=COORDINATE_ORDER,
                center_zyx=[float(v) for v in center.tolist()],
                bbox_start_zyx=[int(v) for v in bbox_start.tolist()],
                bbox_end_zyx=[int(v) for v in bbox_end.tolist()],
                bbox_size_zyx_voxels=[int(v) for v in bbox_size.tolist()],
                spacing_zyx_mm=[float(v) for v in spacing_zyx.tolist()],
                space=STAGE02_SPACE,
                model={
                    "name": "ViTDet3D",
                    "checkpoint": checkpoint_ref,
                    "model_version": model_version,
                },
                raw_logit=float(det["logit"]),
                center_mm_zyx=[float(v) for v in center_mm.tolist()],
                diameter_mm=diameter_mm,
            )
        )

    return candidates

def validate_detector_candidate_schema(
    candidate: Dict[str, Any],
    volume_shape_zyx: Sequence[int],
    spacing_zyx_mm: Sequence[float],
) -> None:
    """Validate the Stage 04 spatial contract before serialization."""

    required = (
        "candidate_id",
        "center_zyx",
        "bbox_start_zyx",
        "bbox_end_zyx",
        "coordinate_order",
        "space",
        "spacing_zyx_mm",
    )
    missing = [key for key in required if key not in candidate]
    if missing:
        raise RuntimeError(f"Stage 04 candidate missing required fields: {missing}")

    if candidate["coordinate_order"] != COORDINATE_ORDER:
        raise RuntimeError(
            f"candidate coordinate_order must be {COORDINATE_ORDER!r}, "
            f"got {candidate['coordinate_order']!r}"
        )
    if candidate["space"] != STAGE02_SPACE:
        raise RuntimeError(
            f"candidate space must be {STAGE02_SPACE!r}, got {candidate['space']!r}"
        )

    center = np.asarray(candidate["center_zyx"], dtype=np.float64)
    shape = np.asarray(volume_shape_zyx, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise RuntimeError(f"Invalid center_zyx: {candidate['center_zyx']}")
    if np.any(center < 0) or np.any(center >= shape):
        raise RuntimeError(
            f"Candidate center outside Stage 02 volume: "
            f"center={center.tolist()}, shape={shape.tolist()}"
        )

    spacing = np.asarray(candidate["spacing_zyx_mm"], dtype=np.float64)
    expected_spacing = np.asarray(spacing_zyx_mm, dtype=np.float64)
    if spacing.shape != (3,) or not np.allclose(
        spacing, expected_spacing, atol=1e-4, rtol=0.0
    ):
        raise RuntimeError(
            "Stage 04 candidate spacing does not match detector volume spacing."
        )


# ============================================================================
# LOG DETECTOR (diagnostic / research-comparison only)
# ============================================================================

def normalize_for_log(volume_hu: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
    """
    Normalize HU values for the LoG response calculation.

    This normalization is used ONLY for the diagnostic LoG detector.
    It is never written into Stage 05 classifier patches, and it never
    touches `candidates.json`.
    """
    valid = volume_hu[lung_mask]
    if valid.size == 0:
        raise RuntimeError("Lung mask contains no voxels.")

    p_low, p_high = np.percentile(valid, [1.0, 99.0])
    if p_high <= p_low:
        p_low, p_high = float(np.min(valid)), float(np.max(valid))
    if p_high <= p_low:
        return np.zeros_like(volume_hu, dtype=np.float32)

    image = (volume_hu - p_low) / (p_high - p_low)
    image = np.clip(image, 0.0, 1.0)
    image[~lung_mask] = 0.0
    return image.astype(np.float32)


def local_maxima(response: np.ndarray, footprint: np.ndarray, threshold: float) -> np.ndarray:
    """Return voxel coordinates of local maxima above threshold."""
    maximum = maximum_filter(response, footprint=footprint, mode="constant", cval=0.0)
    mask = (response == maximum) & (response >= threshold) & np.isfinite(response)
    return np.argwhere(mask)


def run_log_detector(
    volume_hu: np.ndarray,
    lung_mask: np.ndarray,
    spacing_zyx: np.ndarray,
    origin_zyx: np.ndarray,
    threshold: float,
    min_distance_mm: float,
) -> List[LogDiagnosticCandidate]:
    """
    Multi-scale Laplacian-of-Gaussian blob detector.

    Per the architecture, LoG is retained only as a diagnostic /
    research-comparison baseline (it must not replace or be fused into
    ViTDet3D's Stage 04 output). Every record this returns is tagged
    `purpose="research_comparison_only"` and is written to its own
    file, never merged into `candidates.json`.
    """

    image = normalize_for_log(volume_hu, lung_mask)
    raw: List[Dict[str, Any]] = []

    print()
    print("Running multi-scale LoG (diagnostic only):")

    for diameter_mm in LOG_DIAMETERS_MM:
        # For a 3D Gaussian blob, diameter ~= 2 * sqrt(3) * sigma.
        sigma_mm = diameter_mm / (2.0 * math.sqrt(3.0))
        sigma_vox = sigma_mm / spacing_zyx

        response = -gaussian_laplace(image, sigma=tuple(sigma_vox.tolist()), mode="nearest")
        response *= sigma_mm ** 2
        response[~lung_mask] = 0.0

        radius_vox = np.maximum(
            np.ceil(min_distance_mm / spacing_zyx).astype(np.int32), 1
        )
        z = np.arange(-radius_vox[0], radius_vox[0] + 1)
        y = np.arange(-radius_vox[1], radius_vox[1] + 1)
        x = np.arange(-radius_vox[2], radius_vox[2] + 1)
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        footprint = (
            (zz * spacing_zyx[0]) ** 2
            + (yy * spacing_zyx[1]) ** 2
            + (xx * spacing_zyx[2]) ** 2
            <= min_distance_mm ** 2
        )

        coords = local_maxima(response, footprint, threshold)
        print(f"  diameter={diameter_mm:5.1f} mm sigma={sigma_mm:5.2f} mm maxima={len(coords)}")

        for coord in coords:
            center = coord.astype(np.float32)
            raw.append(
                {
                    "center_zyx": center,
                    "score": float(response[tuple(coord)]),
                    "scale_mm": float(diameter_mm),
                }
            )

    print(f"LoG raw diagnostic detections: {len(raw)}")

    # Same NMS-by-distance approach as ViTDet3D -- keep the strongest
    # raw detection's own center unchanged, discard redundant neighbors.
    kept = nms_by_distance(
        [{"center_zyx": r["center_zyx"], "low_zyx": r["center_zyx"], "high_zyx": r["center_zyx"],
          "score": r["score"], "logit": r["score"]} for r in raw],
        spacing_zyx,
        distance_mm=min_distance_mm,
    )
    # Recover scale_mm for kept detections (matched by identity of the
    # numpy array object, since nms_by_distance only forwards score).
    scale_lookup = {id(r["center_zyx"]): r["scale_mm"] for r in raw}

    diagnostics: List[LogDiagnosticCandidate] = []
    for diagnostic_id, det in enumerate(sorted(kept, key=lambda d: d["score"], reverse=True)):
        center = det["center_zyx"]
        center_mm = voxel_to_mm(center, spacing_zyx, origin_zyx)
        diagnostics.append(
            LogDiagnosticCandidate(
                diagnostic_id=f"log_diag_{diagnostic_id:05d}",
                purpose="research_comparison_only",
                coordinate_order=COORDINATE_ORDER,
                center_zyx=[float(v) for v in center.tolist()],
                center_mm_zyx=[float(v) for v in center_mm.tolist()],
                scale_mm=float(scale_lookup.get(id(center), float("nan"))),
                log_response=float(det["score"]),
                space=STAGE02_SPACE,
            )
        )

    print(f"LoG diagnostic candidates after NMS: {len(diagnostics)}")
    return diagnostics


# ============================================================================
# DIAGNOSTIC-ONLY COMPARISON (never fed back into candidates.json)
# ============================================================================

def compute_diagnostic_agreement(
    vit_candidates: List[DetectorCandidate],
    log_candidates: List[LogDiagnosticCandidate],
    agreement_distance_mm: float,
) -> Dict[str, Any]:
    """
    Report-only comparison of ViTDet3D candidates against LoG
    diagnostic detections. This NEVER modifies either candidate list --
    it only annotates, for research purposes, which ViTDet3D candidates
    have a nearby LoG detection and vice versa.
    """

    per_candidate = []
    vit_with_log_support = 0

    for c in vit_candidates:
        c_mm = np.asarray(c.center_mm_zyx, dtype=np.float64)
        best_dist = None
        best_log_id = None
        for log_c in log_candidates:
            d = euclidean_mm_physical(c_mm, log_c.center_mm_zyx)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_log_id = log_c.diagnostic_id

        has_support = best_dist is not None and best_dist <= agreement_distance_mm
        if has_support:
            vit_with_log_support += 1

        per_candidate.append(
            {
                "candidate_id": c.candidate_id,
                "nearest_log_diagnostic_id": best_log_id,
                "distance_mm": best_dist,
                "log_agrees_within_threshold": bool(has_support),
            }
        )

    log_with_vit_support = 0
    for log_c in log_candidates:
        log_mm = np.asarray(log_c.center_mm_zyx, dtype=np.float64)
        nearest = min(
            (euclidean_mm_physical(log_mm, c.center_mm_zyx) for c in vit_candidates),
            default=None,
        )
        if nearest is not None and nearest <= agreement_distance_mm:
            log_with_vit_support += 1

    return {
        "purpose": "research_comparison_only -- does NOT affect candidates.json",
        "agreement_distance_mm": agreement_distance_mm,
        "num_vitdet3d_candidates": len(vit_candidates),
        "num_log_diagnostic_candidates": len(log_candidates),
        "vitdet3d_candidates_with_log_support": vit_with_log_support,
        "log_candidates_with_vitdet3d_support": log_with_vit_support,
        "per_candidate": per_candidate,
    }


# ============================================================================
# OUTPUT
# ============================================================================

def write_candidates_csv(path: str, candidates: List[DetectorCandidate]):
    """Flat, human-readable mirror of candidates.json (ViTDet3D only)."""

    fields = [
        "candidate_id", "source", "detector_score", "coordinate_order",
        "center_z", "center_y", "center_x",
        "bbox_start_z", "bbox_start_y", "bbox_start_x",
        "bbox_end_z", "bbox_end_y", "bbox_end_x",
        "bbox_size_z", "bbox_size_y", "bbox_size_x",
        "spacing_z_mm", "spacing_y_mm", "spacing_x_mm",
        "space", "model_name", "model_checkpoint", "model_version",
        "diameter_mm",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in candidates:
            writer.writerow(
                {
                    "candidate_id": c.candidate_id,
                    "source": c.source,
                    "detector_score": c.detector_score,
                    "coordinate_order": c.coordinate_order,
                    "center_z": c.center_zyx[0], "center_y": c.center_zyx[1], "center_x": c.center_zyx[2],
                    "bbox_start_z": c.bbox_start_zyx[0], "bbox_start_y": c.bbox_start_zyx[1], "bbox_start_x": c.bbox_start_zyx[2],
                    "bbox_end_z": c.bbox_end_zyx[0], "bbox_end_y": c.bbox_end_zyx[1], "bbox_end_x": c.bbox_end_zyx[2],
                    "bbox_size_z": c.bbox_size_zyx_voxels[0], "bbox_size_y": c.bbox_size_zyx_voxels[1], "bbox_size_x": c.bbox_size_zyx_voxels[2],
                    "spacing_z_mm": c.spacing_zyx_mm[0], "spacing_y_mm": c.spacing_zyx_mm[1], "spacing_x_mm": c.spacing_zyx_mm[2],
                    "space": c.space,
                    "model_name": c.model.get("name"),
                    "model_checkpoint": c.model.get("checkpoint"),
                    "model_version": c.model.get("model_version"),
                    "diameter_mm": c.diameter_mm,
                }
            )


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LungInsight Stage 04: pretrained ViTDet3D candidate detection."
    )

    parser.add_argument("patient_id", help="Patient/output identifier, e.g. LIDC-IDRI-0141")
    parser.add_argument("--output-root", default="output", help="Root output directory. Default: output")

    parser.add_argument(
        "--vitdet-model", default="rlsn/DeTr4LungNodule",
        help="Hugging Face ViTDet3D model identifier.",
    )
    parser.add_argument(
        "--vitdet-revision", default=None,
        help="Optional pinned Hugging Face revision/commit for reproducibility.",
    )
    parser.add_argument(
        "--model-version", default="v1.0",
        help="Human-readable model version tag recorded in each candidate's model.model_version.",
    )

    parser.add_argument("--device", default=None, help="Device: cuda, cpu, or auto. Default: auto.")
    parser.add_argument("--batch-size", type=int, default=4, help="ViTDet3D sliding-window batch size.")
    parser.add_argument(
        "--vit-logit-threshold", type=float, default=DEFAULT_VIT_LOGIT_THRESHOLD,
        help="Minimum ViTDet3D raw logit. Default: -5.",
    )
    parser.add_argument(
        "--nms-distance-mm", type=float, default=DEFAULT_NMS_DISTANCE_MM,
        help="Distance (mm) used to de-duplicate overlapping-window ViTDet3D "
        "detections of the same lesion. Does not fuse across detectors.",
    )

    parser.add_argument(
        "--with-log-diagnostic", dest="with_log_diagnostic", action="store_true", default=True,
        help="Also run the LoG diagnostic/research-comparison detector (default: on). "
        "Output is written separately and never merged into candidates.json.",
    )
    parser.add_argument(
        "--skip-log", dest="with_log_diagnostic", action="store_false",
        help="Skip the LoG diagnostic detector entirely.",
    )
    parser.add_argument("--log-threshold", type=float, default=DEFAULT_LOG_THRESHOLD, help="LoG response threshold.")
    parser.add_argument(
        "--log-min-distance-mm", type=float, default=DEFAULT_LOG_MIN_DISTANCE_MM,
        help="Minimum LoG candidate separation.",
    )
    parser.add_argument(
        "--diagnostic-agreement-distance-mm", type=float,
        default=DEFAULT_DIAGNOSTIC_AGREEMENT_DISTANCE_MM,
        help="Distance (mm) used only for the ViTDet3D-vs-LoG agreement report.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 76)
    print("LUNGINSIGHT STAGE 04")
    print("PRETRAINED VITDET3D CANDIDATE DETECTION")
    print("=" * 76)
    print(f"Patient : {args.patient_id}")
    print(f"Device  : {device}")
    print()

    # ------------------------------------------------------------------
    # Stage 02 input
    # ------------------------------------------------------------------
    stage02_dir = os.path.join(args.output_root, args.patient_id, "02")
    volume_hu, lung_mask, spacing_zyx, origin_zyx, stage02_meta = load_stage02(stage02_dir)

    print(f"Volume  : {volume_hu.shape}")
    print(f"Spacing : {spacing_zyx[0]:.4f}, {spacing_zyx[1]:.4f}, {spacing_zyx[2]:.4f} mm")
    print(f"Lung voxels: {int(lung_mask.sum()):,}")

    stage04_dir = os.path.join(args.output_root, args.patient_id, "04_candidates")
    os.makedirs(stage04_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # ViTDet3D -- the authoritative Stage 04 detector
    # ------------------------------------------------------------------
    model, resolved_commit = load_vitdet3d(args.vitdet_model, device, revision=args.vitdet_revision)
    checkpoint_ref = args.vitdet_model
    if args.vitdet_revision:
        checkpoint_ref += f"@{args.vitdet_revision}"
    elif resolved_commit:
        checkpoint_ref += f"@{resolved_commit}"

    raw_detections = run_vitdet3d_raw(
        volume_hu=volume_hu,
        device=device,
        model=model,
        batch_size=args.batch_size,
        logit_threshold=args.vit_logit_threshold,
    )
    kept_detections = nms_by_distance(raw_detections, spacing_zyx, distance_mm=args.nms_distance_mm)
    vit_candidates = build_detector_candidates(
        kept_detections,
        spacing_zyx=spacing_zyx,
        origin_zyx=origin_zyx,
        model_name="ViTDet3D",
        checkpoint_ref=checkpoint_ref,
        model_version=args.model_version,
    )

    for candidate in (asdict(item) for item in vit_candidates):
        validate_detector_candidate_schema(
            candidate,
            volume_shape_zyx=volume_hu.shape,
            spacing_zyx_mm=spacing_zyx,
        )

    # AUTHORITATIVE output -- this is the only file Stage 05 should read.
    save_json(
        os.path.join(stage04_dir, "candidates.json"),
        [asdict(c) for c in vit_candidates],
    )
    write_candidates_csv(os.path.join(stage04_dir, "candidates.csv"), vit_candidates)

    # ------------------------------------------------------------------
    # LoG -- diagnostic / research-comparison only (never authoritative)
    # ------------------------------------------------------------------
    log_candidates: List[LogDiagnosticCandidate] = []
    if args.with_log_diagnostic:
        log_candidates = run_log_detector(
            volume_hu=volume_hu,
            lung_mask=lung_mask,
            spacing_zyx=spacing_zyx,
            origin_zyx=origin_zyx,
            threshold=args.log_threshold,
            min_distance_mm=args.log_min_distance_mm,
        )
    else:
        print()
        print("LoG diagnostic detector: SKIPPED")

    save_json(
        os.path.join(stage04_dir, "log_candidates_diagnostic.json"),
        [asdict(c) for c in log_candidates],
    )

    agreement_report = compute_diagnostic_agreement(
        vit_candidates, log_candidates,
        agreement_distance_mm=args.diagnostic_agreement_distance_mm,
    )
    save_json(
        os.path.join(stage04_dir, "candidates_diagnostic_agreement.json"),
        agreement_report,
    )

    # ------------------------------------------------------------------
    # Provenance / run metadata
    # ------------------------------------------------------------------
    detector_metadata = {
        "patient_id": args.patient_id,
        "stage": 4,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_order": COORDINATE_ORDER,
        "space": STAGE02_SPACE,
        "volume_shape_zyx": list(volume_hu.shape),
        "spacing_zyx_mm": spacing_zyx.tolist(),
        "origin_zyx_mm": origin_zyx.tolist(),
            "crop_offset_zyx": stage02_meta.get("crop_offset_zyx", [0, 0, 0]),
        "authoritative_detector": {
            "name": "ViTDet3D",
            "model": args.vitdet_model,
            "checkpoint": checkpoint_ref,
            "model_version": args.model_version,
            "window_zyx": VIT_WINDOW.tolist(),
            "stride_zyx": VIT_STRIDE.tolist(),
            "normalization_mean": VIT_MEAN,
            "normalization_std": VIT_STD,
            "logit_threshold": args.vit_logit_threshold,
            "nms_distance_mm": args.nms_distance_mm,
            "num_raw_detections": len(raw_detections),
            "num_candidates": len(vit_candidates),
        },
        "diagnostic_log_detector": {
            "enabled": args.with_log_diagnostic,
            "note": "Diagnostic/research-comparison only; not fused into candidates.json.",
            "diameters_mm": list(LOG_DIAMETERS_MM),
            "threshold": args.log_threshold,
            "min_distance_mm": args.log_min_distance_mm,
            "num_candidates": len(log_candidates),
        },
        "output_files": {
            "candidates.json": "AUTHORITATIVE - ViTDet3D only. Consumed by Stage 05.",
            "candidates.csv": "Flat mirror of candidates.json.",
            "log_candidates_diagnostic.json": "Diagnostic only - not authoritative.",
            "candidates_diagnostic_agreement.json": "Comparison report only.",
        },
    }
    save_json(os.path.join(stage04_dir, "detector_metadata.json"), detector_metadata)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print()
    print("=" * 76)
    print("STAGE 04 COMPLETE")
    print("=" * 76)
    print(f"ViTDet3D candidates (authoritative) : {len(vit_candidates)}")
    print(f"LoG diagnostic candidates           : {len(log_candidates)}")
    print(f"ViTDet3D candidates with LoG support : {agreement_report['vitdet3d_candidates_with_log_support']}")
    print()
    print("Output:")
    print(f"  {os.path.abspath(stage04_dir)}")
    print()
    print("Files:")
    print("  candidates.json                      <- AUTHORITATIVE (feed to Stage 05)")
    print("  candidates.csv")
    print("  detector_metadata.json")
    print("  log_candidates_diagnostic.json       <- diagnostic only")
    print("  candidates_diagnostic_agreement.json <- diagnostic only")


if __name__ == "__main__":
    main()