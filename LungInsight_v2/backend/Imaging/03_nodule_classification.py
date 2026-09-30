"""
LungInsight v2 -- Stage 03: Nodule Attribute Classification / Characterization
===============================================================================

This stage applies the supplied LungInsight nodule-ranker network:
    backend/models/nodule_ranker/model.py
    backend/models/nodule_ranker/components.py
    backend/models/nodule_ranker/initialisation.py

The supplied network is the attention-enhanced cross-task lung-nodule model.
Its forward pass returns:
    final         : 9 sigmoid attribute scores in [0, 1]
    aux           : 9 auxiliary attribute scores in [0, 1]
    slice_weights : slice-attention weights
    caam_weights  : cross-attribute attention weights

The nine attributes, in the model's required order, are:
    0 subtlety
    1 internal_structure
    2 calcification
    3 sphericity
    4 margin
    5 lobulation
    6 spiculation
    7 texture
    8 malignancy

The original reference implementation uses one nodule volume at a time and
passes the slice dimension as the PyTorch batch dimension to the 2-D backbone:
    input patch: (Z, H, W)
    model input: (Z, 1, H, W)
This is important because SliceAttention aggregates information across the
first dimension; batching multiple nodules together would mix their slices.

Input:
    backend/output/<patient_id>/02_detect_and_patch/
        patches/arrays/patch_000.npy
        patches/patches_manifest.json

Patch requirements:
    * float32 HU arrays in ZYX order
    * the Stage 02 default is 64 x 64 x 64
    * spatial dimensions must be 64 x 64 because the supplied model's
      three 2-D pooling stages reduce 64 -> 32 -> 16 -> 8 and its final
      AvgPool2d is fixed at 8 x 8.
    * depth (Z) may vary; SliceAttention is designed to aggregate the
      available slices.

Reference-model input normalization:
    The original model repository normalises each prepared CT slice as:
        (image - fold_mean) / fold_std
    The published config supplies five fold-specific means/stds. For this
    v2 integration, --fold-id selects the corresponding pair (default 1).

    Stage 02 stores physical CT values as HU. The original nodule-ranker
    training data are not stored as raw negative-HU values; the fold means
    are in the shifted CT-intensity domain. Therefore the Stage 03 adapter
    uses the model-input transform:
        clipped = clip(HU, -1000, 400)
        model_intensity = clipped + 1000
    followed by the original fold normalization.

Outputs:
    backend/output/<patient_id>/03_nodule_classification/
        predictions.json
        predictions.csv
        classification_metadata.json
        attention/
            patch_000_slice_attention.npy
            patch_000_attribute_attention.npy

Important:
    The nine sigmoid outputs are the model's normalized attribute ratings,
    not calibrated probabilities. `malignancy` is a learned 1--5 LIDC rating
    after inverse scaling, not a binary benign/malignant probability.

The classification metadata also preserves the Stage 02 patch geometry,
DICOM-LPS coordinate frame, anatomical-location object, detector score,
and patch provenance so predictions stay traceable to the original CT.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
V2_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_ROOT = V2_ROOT / "backend" / "output"
DEFAULT_RANKER_DIR = V2_ROOT / "backend" / "models" / "nodule_ranker"


# -----------------------------------------------------------------------------
# Model/data constants from the supplied reference implementation/config
# -----------------------------------------------------------------------------

ATTRIBUTE_NAMES = [
    "subtlety",
    "internal_structure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
    "malignancy",
]

FEATURE_IDS = list(range(9))
N_FEATURES = len(ATTRIBUTE_NAMES)
INITIAL_DEPTH = 32

# The supplied public config contains five fold-specific image means/stds.
# These values are the reference model's data-normalisation values.
FOLD_MEANS = [357.87565, 356.72941, 360.50329, 355.27085, 357.84044]
FOLD_STDS = [288.47657, 287.1689, 290.45408, 286.90142, 288.45889]

# Model input representation inferred from the supplied reference pipeline:
# HU -> clip[-1000,400] -> shift +1000 -> fold normalization.
HU_LOW = -1000.0
HU_HIGH = 400.0
HU_SHIFT = 1000.0

# The reference test code maps normalized outputs back to LIDC rating scales:
# feature 1: internal structure -> 1..4
# feature 2: calcification     -> 1..6
# all other features           -> 1..5
ATTRIBUTE_RATING_SCALES = {
    "subtlety": {"min": 1.0, "max": 5.0, "multiplier": 4.0, "offset": 1.0},
    "internal_structure": {"min": 1.0, "max": 4.0, "multiplier": 3.0, "offset": 1.0},
    "calcification": {"min": 1.0, "max": 6.0, "multiplier": 5.0, "offset": 1.0},
    "sphericity": {"min": 1.0, "max": 5.0, "multiplier": 4.0, "offset": 1.0},
    "margin": {"min": 1.0, "max": 5.0, "multiplier": 4.0, "offset": 1.0},
    "lobulation": {"min": 1.0, "max": 5.0, "multiplier": 4.0, "offset": 1.0},
    "spiculation": {"min": 1.0, "max": 5.0, "multiplier": 4.0, "offset": 1.0},
    "texture": {"min": 1.0, "max": 5.0, "multiplier": 4.0, "offset": 1.0},
    "malignancy": {"min": 1.0, "max": 5.0, "multiplier": 4.0, "offset": 1.0},
}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=json_default, allow_nan=False)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_3(value: Any, name: str, dtype=float) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values; got {value}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values: {value}")
    return arr


def normalize_order(value: Any, name: str) -> str:
    text = str(value).upper().replace(" ", "").replace(",", "")
    if text != "ZYX":
        raise ValueError(f"{name} must be ZYX; got {value!r}")
    return "ZYX"


# -----------------------------------------------------------------------------
# Reference model import
# -----------------------------------------------------------------------------


def _load_module_from_file(module_name: str, path: Path, package: str | None = None) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Missing nodule-ranker source file: {path}")
    kwargs = {"submodule_search_locations": [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(module_name, str(path), **kwargs)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    if package:
        module.__package__ = package
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_ranker_network(ranker_dir: Path):
    """
    Load Network from the supplied model.py/components.py/initialisation.py.

    The supplied components.py uses the original repository's historical
    spelling `.intialisation`. The user's v2 file is named `initialisation.py`.
    We provide that spelling as an in-memory module alias so the source files
    themselves do not need to be edited.
    """
    model_path = ranker_dir / "model.py"
    components_path = ranker_dir / "components.py"
    init_correct = ranker_dir / "initialisation.py"
    init_original = ranker_dir / "intialisation.py"

    if init_original.is_file():
        init_path = init_original
    elif init_correct.is_file():
        init_path = init_correct
    else:
        raise FileNotFoundError(
            "Could not find initialisation.py or intialisation.py in "
            f"{ranker_dir}"
        )

    package_name = "_lunginsight_nodule_ranker"
    package = ModuleType(package_name)
    package.__path__ = [str(ranker_dir)]
    sys.modules[package_name] = package

    init_module_name = f"{package_name}.intialisation"
    _load_module_from_file(init_module_name, init_path, package_name)

    components_module_name = f"{package_name}.components"
    _load_module_from_file(components_module_name, components_path, package_name)

    model_module_name = f"{package_name}.model"
    model_module = _load_module_from_file(model_module_name, model_path, package_name)

    if not hasattr(model_module, "Network"):
        raise ImportError(f"Network class not found in {model_path}")

    return model_module.Network


# -----------------------------------------------------------------------------
# Checkpoint handling
# -----------------------------------------------------------------------------


def _checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"epoch[_-](\d+)", path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def discover_checkpoint(ranker_dir: Path) -> Path:
    """Find one clear default checkpoint without guessing among experiments."""
    preferred_names = [
        "best.pth",
        "best_model.pth",
        "model_best.pth",
        "latest.pth",
        "checkpoint.pth",
    ]
    for name in preferred_names:
        candidate = ranker_dir / name
        if candidate.is_file():
            return candidate

    root_epoch_files = list(ranker_dir.glob("epoch_*.pth"))
    if root_epoch_files:
        root_epoch_files.sort(key=lambda p: (_checkpoint_epoch(p) is not None, _checkpoint_epoch(p) or -1, p.name))
        return root_epoch_files[-1]

    all_files = [p for p in ranker_dir.rglob("*.pth") if p.is_file()]
    if len(all_files) == 1:
        return all_files[0]

    if not all_files:
        raise FileNotFoundError(
            "No .pth checkpoint found under "
            f"{ranker_dir}. Supply one with --checkpoint."
        )

    examples = "\n".join(f"  {p}" for p in sorted(all_files)[:20])
    raise RuntimeError(
        "Multiple nodule-ranker checkpoints were found, so Stage 03 will not "
        "guess which trained weights you intend to use. Supply --checkpoint.\n"
        f"Candidates:\n{examples}"
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    state_dict = checkpoint
    checkpoint_epoch = None
    if isinstance(checkpoint, dict):
        checkpoint_epoch = checkpoint.get("epoch")
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                state_dict = candidate
                break

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Unsupported checkpoint format in {checkpoint_path}; expected a state_dict or dict containing model_state_dict/state_dict/model."
        )

    # Handle DataParallel checkpoints that prefix parameters with "module.".
    if state_dict and all(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {str(k)[7:]: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Nodule-ranker checkpoint does not match the supplied Network architecture.\n"
            f"Missing keys ({len(missing)}): {missing[:10]}\n"
            f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}"
        )

    return {
        "checkpoint_epoch": int(checkpoint_epoch) if checkpoint_epoch is not None else None,
        "state_dict_keys": int(len(state_dict)),
    }


# -----------------------------------------------------------------------------
# Stage 02 patch loading
# -----------------------------------------------------------------------------


def load_patch_manifest(stage02_dir: Path) -> dict[str, Any]:
    manifest_path = stage02_dir / "patches" / "patches_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Stage 02 patch manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("patches_manifest.json must contain an object")
    return manifest


def load_patch_array(patch_root: Path, record: dict[str, Any]) -> tuple[Path, np.ndarray]:
    array_path = record.get("array_path")
    if array_path:
        candidate_path = Path(str(array_path))
        if not candidate_path.is_absolute():
            # Manifest paths are usually relative to the stage output directory
            # or the working directory. Prefer exact path first, then patch root.
            options = [candidate_path, patch_root / candidate_path]
        else:
            options = [candidate_path]
    else:
        patch_id = int(record["patch_id"])
        options = [patch_root / "arrays" / f"patch_{patch_id:03d}.npy"]

    for path in options:
        if path.is_file():
            patch = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
            return path, patch

    # Last fallback: canonical array filename under the manifest's patch root.
    patch_id = int(record["patch_id"])
    fallback = patch_root / "arrays" / f"patch_{patch_id:03d}.npy"
    raise FileNotFoundError(f"Patch array not found. Tried: {options + [fallback]}")


def validate_patch_record(record: dict[str, Any], patch: np.ndarray) -> None:
    if str(record.get("coordinate_order", "ZYX")).upper() != "ZYX":
        raise ValueError(f"Patch {record.get('patch_id')} is not ZYX")
    if patch.ndim != 3:
        raise ValueError(f"Patch {record.get('patch_id')} must be 3-D ZYX; got {patch.shape}")
    if patch.shape[1:] != (64, 64):
        raise ValueError(
            f"Patch {record.get('patch_id')} has spatial shape {patch.shape[1:]}; "
            "the supplied nodule-ranker expects 64 x 64 spatial slices."
        )


# -----------------------------------------------------------------------------
# Model-input preprocessing
# -----------------------------------------------------------------------------


def hu_to_ranker_intensity(patch_hu: np.ndarray) -> np.ndarray:
    """
    Convert Stage 02 HU into the intensity domain used by the reference
    nodule-ranker data pipeline.
    """
    clipped = np.clip(patch_hu.astype(np.float32), HU_LOW, HU_HIGH)
    return clipped + HU_SHIFT


def normalize_for_ranker(patch_intensity: np.ndarray, fold_id: int) -> np.ndarray:
    if fold_id < 1 or fold_id > len(FOLD_MEANS):
        raise ValueError(f"--fold-id must be 1..{len(FOLD_MEANS)}")
    mean = FOLD_MEANS[fold_id - 1]
    std = FOLD_STDS[fold_id - 1]
    if std <= 0:
        raise ValueError(f"Invalid fold std: {std}")
    return ((patch_intensity - mean) / std).astype(np.float32)


def prepare_model_input(patch_hu: np.ndarray, fold_id: int) -> torch.Tensor:
    model_domain = hu_to_ranker_intensity(patch_hu)
    normalized = normalize_for_ranker(model_domain, fold_id)

    # Reference test path: batch_x is [1,Z,H,W], then permute to [Z,1,H,W].
    tensor = torch.from_numpy(normalized).float().unsqueeze(0)  # [1,Z,H,W]
    tensor = tensor.permute(1, 0, 2, 3).contiguous()            # [Z,1,H,W]
    return tensor


# -----------------------------------------------------------------------------
# Prediction post-processing
# -----------------------------------------------------------------------------


def inverse_rating_scale(score_01: float, attribute_name: str) -> float:
    cfg = ATTRIBUTE_RATING_SCALES[attribute_name]
    return float(score_01 * cfg["multiplier"] + cfg["offset"])


def build_attribute_predictions(final_scores: np.ndarray, aux_scores: np.ndarray) -> dict[str, Any]:
    if final_scores.shape != (N_FEATURES,):
        raise ValueError(f"Expected 9 final scores, got {final_scores.shape}")
    if aux_scores.shape != (N_FEATURES,):
        raise ValueError(f"Expected 9 auxiliary scores, got {aux_scores.shape}")

    attributes: dict[str, Any] = {}
    for idx, name in enumerate(ATTRIBUTE_NAMES):
        score = float(np.clip(final_scores[idx], 0.0, 1.0))
        aux = float(np.clip(aux_scores[idx], 0.0, 1.0))
        attributes[name] = {
            "feature_id": idx,
            "score_0_1": score,
            "rating_original_scale": inverse_rating_scale(score, name),
            "rating_scale": [
                ATTRIBUTE_RATING_SCALES[name]["min"],
                ATTRIBUTE_RATING_SCALES[name]["max"],
            ],
            "auxiliary_score_0_1": aux,
        }
    return attributes


def attention_to_json(slice_weights: np.ndarray, caam_weights: np.ndarray) -> tuple[list[float], list[list[float]]]:
    slice_arr = np.asarray(slice_weights, dtype=np.float32).squeeze()
    if slice_arr.ndim != 1:
        slice_arr = slice_arr.reshape(-1)

    caam_arr = np.asarray(caam_weights, dtype=np.float32)
    if caam_arr.ndim == 1:
        caam_arr = caam_arr.reshape(1, -1)

    return slice_arr.astype(float).tolist(), caam_arr.astype(float).tolist()


# -----------------------------------------------------------------------------
# Single-patch inference
# -----------------------------------------------------------------------------


def infer_one(
    model: torch.nn.Module,
    patch_hu: np.ndarray,
    fold_id: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_input = prepare_model_input(patch_hu, fold_id).to(device)

    with torch.inference_mode():
        final, aux, slice_weights, caam_weights = model(model_input)

    final_np = final.squeeze(0).detach().cpu().numpy().astype(np.float32)
    aux_np = aux.squeeze(0).detach().cpu().numpy().astype(np.float32)
    slice_np = slice_weights.detach().cpu().numpy().astype(np.float32)
    caam_np = caam_weights.detach().cpu().numpy().astype(np.float32)

    return final_np, aux_np, slice_np, caam_np


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def write_predictions_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "patch_id",
        "candidate_id",
        "anatomical_primary",
        "anatomical_region",
        "laterality",
        "compartment",
        "detector_score",
    ]
    for name in ATTRIBUTE_NAMES:
        fields.extend([
            f"{name}_score_0_1",
            f"{name}_rating_original_scale",
            f"{name}_auxiliary_score_0_1",
        ])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            location = record.get("anatomical_location") or {}
            row = {
                "patch_id": record["patch_id"],
                "candidate_id": record["candidate_id"],
                "anatomical_primary": location.get("primary"),
                "anatomical_region": location.get("region"),
                "laterality": location.get("laterality"),
                "compartment": location.get("compartment"),
                "detector_score": record.get("detector_score"),
            }
            for name in ATTRIBUTE_NAMES:
                pred = record["attributes"][name]
                row[f"{name}_score_0_1"] = pred["score_0_1"]
                row[f"{name}_rating_original_scale"] = pred["rating_original_scale"]
                row[f"{name}_auxiliary_score_0_1"] = pred["auxiliary_score_0_1"]
            writer.writerow(row)


def build_prediction_record(
    patch_record: dict[str, Any],
    patch_path: Path,
    attributes: dict[str, Any],
    slice_weights: np.ndarray,
    caam_weights: np.ndarray,
) -> dict[str, Any]:
    location = patch_record.get("anatomical_location")
    center_lps = patch_record.get("center_lps_mm", patch_record.get("center_world_mm"))
    patch_origin_lps = patch_record.get("patch_origin_lps_mm")
    direction_lps = patch_record.get("direction_lps")

    slice_attention_json, caam_attention_json = attention_to_json(slice_weights, caam_weights)

    return {
        "patch_id": int(patch_record["patch_id"]),
        "candidate_id": patch_record["candidate_id"],
        "prediction_space": patch_record.get("space", "patch_local_ct"),
        "source_ct_space": patch_record.get("source_space", "stage01_final_cropped_ct"),
        "coordinate_order": "ZYX",
        "patch_size_vox_zyx": patch_record.get("patch_size_vox_zyx"),
        "spacing_zyx_mm": patch_record.get("spacing_zyx_mm"),
        "origin_lps_mm": patch_record.get("origin_lps_mm"),
        "patch_origin_lps_mm": patch_origin_lps,
        "direction_lps": direction_lps,
        "center_zyx": patch_record.get("center_zyx"),
        "center_lps_mm": center_lps,
        "patch_center_local_zyx": patch_record.get("patch_center_local_zyx"),
        "detector_score": patch_record.get("detector_score"),
        "raw_logit": patch_record.get("raw_logit"),
        "diameter_mm": patch_record.get("diameter_mm"),
        "anatomical_location": location,
        "attributes": attributes,
        "attention": {
            "slice_attention": {
                "num_slices": len(slice_attention_json),
                "weights": slice_attention_json,
                "weight_semantics": "one attention weight per input axial slice; weights are generated by SliceAttention and sum approximately to 1 across slices.",
                "array_path": None,
            },
            "cross_attribute_attention": {
                "shape": [len(caam_attention_json), len(caam_attention_json[0]) if caam_attention_json else 0],
                "weights": caam_attention_json,
                "row_semantics": "one row per output attribute head",
                "column_semantics": "one column per attribute-specific feature representation",
                "array_path": None,
            },
        },
        "patch_array_path": str(patch_path),
        "reference_model": {
            "architecture": "attention-enhanced cross-task nodule attribute network",
            "module": "backend.models.nodule_ranker.model.Network",
            "initial_depth": INITIAL_DEPTH,
            "n_features": N_FEATURES,
            "feature_ids": FEATURE_IDS,
        },
    }


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    patient_root = output_root / args.patient_id
    stage02_dir = patient_root / "02_detect_and_patch"
    stage03_dir = patient_root / "03_nodule_classification"
    patch_root = stage02_dir / "patches"

    if not stage02_dir.is_dir():
        raise FileNotFoundError(f"Stage 02 output not found: {stage02_dir}")

    manifest = load_patch_manifest(stage02_dir)
    patch_records = manifest.get("patches", [])
    if not isinstance(patch_records, list):
        raise ValueError("patches_manifest.json field 'patches' must be a list")

    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else discover_checkpoint(Path(args.ranker_dir).resolve())
    ranker_dir = Path(args.ranker_dir).resolve()

    if args.fold_id < 1 or args.fold_id > len(FOLD_MEANS):
        raise ValueError(f"--fold-id must be 1..{len(FOLD_MEANS)}")

    if args.device in (None, "auto"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 78)
    print("LUNGINSIGHT STAGE 03")
    print("NODULE ATTRIBUTE CLASSIFICATION / CHARACTERIZATION")
    print("=" * 78)
    print(f"Patient       : {args.patient_id}")
    print(f"Device        : {device}")
    print(f"Input patches : {patch_root}")
    print(f"Checkpoint    : {checkpoint_path}")
    print(f"Fold          : {args.fold_id}")
    print()

    Network = load_ranker_network(ranker_dir)
    opts = SimpleNamespace(initial_depth=INITIAL_DEPTH)
    model = Network(opts, N_FEATURES).to(device)
    checkpoint_info = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    stage03_dir.mkdir(parents=True, exist_ok=True)
    attention_dir = stage03_dir / "attention"
    attention_dir.mkdir(parents=True, exist_ok=True)

    prediction_records: list[dict[str, Any]] = []

    print(f"Loaded {len(patch_records)} Stage 02 patches.")

    for index, patch_record in enumerate(patch_records):
        patch_path, patch_hu = load_patch_array(patch_root, patch_record)
        validate_patch_record(patch_record, patch_hu)

        final_np, aux_np, slice_np, caam_np = infer_one(
            model,
            patch_hu,
            args.fold_id,
            device,
        )

        attributes = build_attribute_predictions(final_np, aux_np)
        record = build_prediction_record(
            patch_record,
            patch_path,
            attributes,
            slice_np,
            caam_np,
        )

        slice_attention_path = attention_dir / f"patch_{int(patch_record['patch_id']):03d}_slice_attention.npy"
        caam_attention_path = attention_dir / f"patch_{int(patch_record['patch_id']):03d}_attribute_attention.npy"
        np.save(slice_attention_path, np.asarray(slice_np, dtype=np.float32))
        np.save(caam_attention_path, np.asarray(caam_np, dtype=np.float32))

        record["attention"]["slice_attention"]["array_path"] = str(slice_attention_path)
        record["attention"]["cross_attribute_attention"]["array_path"] = str(caam_attention_path)

        prediction_records.append(record)

        malignancy = attributes["malignancy"]["rating_original_scale"]
        print(
            f"Classified patch_{int(patch_record['patch_id']):03d}: "
            f"candidate={patch_record['candidate_id']} "
            f"malignancy_rating={malignancy:.3f}/5.000"
        )

    created_at = datetime.now(timezone.utc).isoformat()
    mean = FOLD_MEANS[args.fold_id - 1]
    std = FOLD_STDS[args.fold_id - 1]

    metadata = {
        "schema_version": "03.1",
        "stage": "03",
        "module": "Nodule Attribute Classification / Characterization",
        "generated_at_utc": created_at,
        "patient_id": args.patient_id,
        "coordinate_order": "ZYX",
        "coordinate_system": "DICOM_LPS",
        "source_space": "stage01_final_cropped_ct",
        "prediction_space": "patch_local_ct",
        "inputs": {
            "stage02_directory": str(stage02_dir),
            "patch_manifest": str(stage02_dir / "patches" / "patches_manifest.json"),
            "patch_arrays": str(patch_root / "arrays"),
        },
        "model": {
            "module": "backend.models.nodule_ranker.model.Network",
            "architecture": "attention-enhanced cross-task network",
            "initial_depth": INITIAL_DEPTH,
            "n_features": N_FEATURES,
            "feature_ids": FEATURE_IDS,
            "attribute_names": ATTRIBUTE_NAMES,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_info["checkpoint_epoch"],
            "state_dict_keys": checkpoint_info["state_dict_keys"],
        },
        "device": str(device),
        "preprocessing": {
            "source_units": "HU",
            "clip_low_hu": HU_LOW,
            "clip_high_hu": HU_HIGH,
            "shift_after_clip": HU_SHIFT,
            "fold_id": args.fold_id,
            "fold_mean": mean,
            "fold_std": std,
            "formula": "normalized = (clip(HU, -1000, 400) + 1000 - fold_mean) / fold_std",
            "spatial_shape_requirement": [64, 64],
            "depth_behavior": "arbitrary Z depth is preserved; each axial slice is processed by the 2-D backbone and aggregated by SliceAttention.",
        },
        "attribute_scales": ATTRIBUTE_RATING_SCALES,
        "prediction_semantics": {
            "score_0_1": "Model sigmoid output corresponding to the normalized LIDC attribute rating target.",
            "rating_original_scale": "Inverse mapping to the LIDC attribute rating scale used by the reference test code.",
            "malignancy": "A 1-to-5 malignancy rating, not a calibrated binary malignancy probability.",
            "auxiliary_score_0_1": "Auxiliary ASCMM prediction retained for provenance; final is the authoritative Network output.",
        },
        "patch_provenance": {
            "center_authority": "Stage 02 authoritative ViTDet3D center remains unchanged; Stage 03 never re-centers the patch.",
            "anatomical_location": "Copied from Stage 02 authoritative candidate/patch record without reinterpretation.",
            "coordinate_mapping": "Stage 02 patch geometry remains authoritative: p_LPS_mm = patch_origin_lps_mm + direction_lps @ ([z,y,x] * spacing_zyx_mm).",
        },
        "outputs": {
            "predictions_json": str(stage03_dir / "predictions.json"),
            "predictions_csv": str(stage03_dir / "predictions.csv"),
            "classification_metadata_json": str(stage03_dir / "classification_metadata.json"),
            "attention_directory": str(attention_dir),
        },
        "prediction_count": len(prediction_records),
    }

    save_json(stage03_dir / "predictions.json", prediction_records)
    write_predictions_csv(stage03_dir / "predictions.csv", prediction_records)
    save_json(stage03_dir / "classification_metadata.json", metadata)

    print()
    print("=" * 78)
    print("STAGE 03 COMPLETE")
    print("=" * 78)
    print(f"Classified patches : {len(prediction_records)}")
    print(f"Output             : {stage03_dir}")
    print(f"Predictions        : {stage03_dir / 'predictions.json'}")
    print(f"Metadata           : {stage03_dir / 'classification_metadata.json'}")

    return stage03_dir


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LungInsight Stage 03: classify/characterize Stage 02 nodule "
            "patches with the supplied attention-enhanced cross-task network."
        )
    )
    parser.add_argument(
        "patient_id",
        help="Patient identifier, e.g. LIDC-IDRI-0001",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Pipeline output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--ranker-dir",
        type=Path,
        default=DEFAULT_RANKER_DIR,
        help=f"Directory containing model.py/components.py/initialisation.py and checkpoint(s). Default: {DEFAULT_RANKER_DIR}",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit trained nodule-ranker .pth checkpoint. If omitted, Stage 03 auto-discovers a single clear checkpoint.",
    )
    parser.add_argument(
        "--fold-id",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Reference fold-normalization pair used by the original model. Default: 1.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cuda, or cpu. Default: auto.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
