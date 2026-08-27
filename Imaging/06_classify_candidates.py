"""
06_classify_candidates.py

LungInsight — Stage 06
=======================

Run the trained 3D multi-head classifier on Stage 05 candidate patches.

Pipeline
--------
04_detect_candidates.py
        v
05_extract_candidate_patches.py
        v
06_classify_candidates.py       <-- THIS SCRIPT
        v
07_visualize_gradcam.py


SPATIAL / GEOMETRY CONTRACT
============================

Stage 05 is the sole geometry authority. Stage 06 does not reconstruct,
recompute, or round any spatial value it receives from Stage 05 -- it
reads the following fields verbatim from each Stage 05 patch record and
republishes them unchanged:

    candidate_id
    patch_file
    coordinate_order            ("ZYX")
    candidate_center_zyx        (continuous, NOT rounded)
    patch_shape_zyx              -> [64, 64, 64]
    patch_spacing_zyx_mm         -> isotropic mm per patch voxel
    geometry_authority            "stage05"

Stage 06 never derives these values from anything else (e.g. it does not
infer patch_spacing_zyx_mm from a field-of-view / shape ratio, and it
does not re-key or rename any Stage 05 field). If a required field is
missing from a Stage 05 record, that candidate fails loudly instead of
being silently filled in.


TENSOR CONTRACT
================

Stage 05 persists each patch as a bare (64, 64, 64) float32 .npy array,
already HU-clipped to [-1000, 400] and normalized to [0, 1] via

    (HU + 1000) / 1400

Stage 06:
    * reshapes the on-disk (64, 64, 64) array to the classifier's
      required (1, 1, 64, 64, 64) input tensor -- this is a pure
      reshape, not a numeric transform,
    * verifies (does not correct) that values already lie in [0, 1],
    * performs NO additional HU clipping and NO additional
      normalization of any kind.


CLASSIFIER HEADS
=================

Exactly 8 independent diagnostic heads, evaluated for every candidate:

    calcification, lobulation, malignancy, margin,
    sphericity, spiculation, subtlety, texture


OUTPUT
=======

    output/<patient_id>/06_classification/classification.json

A JSON array. Each element is one candidate record, in exactly this
shape (values are illustrative):

    {
      "candidate_id": 17,
      "patch_file": "patches/candidate_17.npy",
      "coordinate_order": "ZYX",
      "predictions": {
        "calcification": 0.21,
        "lobulation": 0.78,
        "malignancy": 0.64,
        "margin": 0.59,
        "sphericity": 0.33,
        "spiculation": 0.71,
        "subtlety": 0.81,
        "texture": 0.48
      },
      "candidate_center_zyx": [182.4, 211.7, 143.2],
      "patch_shape_zyx": [64, 64, 64],
      "patch_spacing_zyx_mm": [1.0, 1.0, 1.0],
      "geometry_authority": "stage05"
    }

No other top-level fields are added. Candidates that fail (missing
geometry fields, unreadable patch, bad shape, out-of-range values) are
skipped and reported on stderr/stdout -- they are not written as
partial/null records.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


# ============================================================================
# CONFIGURATION
# ============================================================================

CLASSIFIER_SIZE = 64

INPUT_TENSOR_SHAPE = (1, 1, CLASSIFIER_SIZE, CLASSIFIER_SIZE, CLASSIFIER_SIZE)

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

NORMALIZED_MIN = 0.0
NORMALIZED_MAX = 1.0
NORMALIZATION_TOLERANCE = 1e-4

# Fields Stage 06 must read verbatim from each Stage 05 patch record and
# republish unchanged. Names match 05_extract_candidate_patches.py's
# manifest exactly -- no aliasing, no fallback key-guessing.
REQUIRED_STAGE05_KEYS = (
    "candidate_id",
    "patch_file",
    "coordinate_order",
    "candidate_center_zyx",
    "patch_shape_zyx",
    "patch_spacing_zyx_mm",
    "geometry_authority",
)


# ============================================================================
# PROJECT MODEL IMPORT
# ============================================================================

def import_project_model():
    """
    Import the canonical LungInsight multi-head model implementation.
    """
    project_root = Path(__file__).resolve().parent.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from cir_multihead_pipeline import create_multihead_model, FEATURE_NAMES as MODEL_FEATURE_NAMES
    except Exception as exc:
        raise RuntimeError(
            "Could not import cir_multihead_pipeline.py.\n"
            f"Project root: {project_root}\n"
            f"Original error: {exc}"
        ) from exc

    return create_multihead_model, MODEL_FEATURE_NAMES


def resolve_checkpoint(checkpoint: str) -> str:
    path = os.path.abspath(checkpoint)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Classifier checkpoint not found:\n{path}")
    return path


def load_model(checkpoint_path: str, device: torch.device):
    """
    Construct the canonical multi-head classifier and load its checkpoint.
    Fails loudly on any head-name mismatch against the required 8 heads.
    """
    create_multihead_model, model_feature_names = import_project_model()

    if list(model_feature_names) != FEATURE_NAMES:
        raise RuntimeError(
            "Classifier head order/name mismatch.\n\n"
            f"Model heads   : {list(model_feature_names)}\n"
            f"Required heads: {FEATURE_NAMES}"
        )

    print(f"Loading classifier checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Heads : {FEATURE_NAMES}")

    model = create_multihead_model(head_names=model_feature_names, device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unsupported checkpoint format. Expected a state_dict or checkpoint dictionary."
        )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    if missing:
        raise RuntimeError(
            "Classifier checkpoint is missing model parameters:\n"
            + "\n".join(str(x) for x in missing[:20])
        )
    if unexpected:
        raise RuntimeError(
            "Classifier checkpoint contains unexpected parameters:\n"
            + "\n".join(str(x) for x in unexpected[:20])
        )

    model.eval()
    print("Checkpoint loaded successfully.")
    return model


# ============================================================================
# STAGE 05 MANIFEST
# ============================================================================

def load_stage05_manifest(stage05_dir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(stage05_dir, "patches.json")

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Stage 05 manifest not found:\n{manifest_path}\n\nRun Stage 05 first."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict) or not isinstance(manifest.get("patches"), list):
        raise ValueError("Stage 05 patches.json must be an object containing a 'patches' list.")

    return manifest


def validate_stage05_record(record: Dict[str, Any]) -> None:
    """
    Confirm every field Stage 06 must republish verbatim is present.
    Stage 06 never fabricates or infers a missing geometry field.
    """
    if not isinstance(record, dict):
        raise ValueError("Stage 05 patch record must be a JSON object.")

    missing = [key for key in REQUIRED_STAGE05_KEYS if key not in record]
    if missing:
        raise KeyError(
            "Stage 05 record is missing required field(s): "
            f"{missing}. Stage 06 does not reconstruct geometry, so this "
            "candidate cannot be classified."
        )

    if record["coordinate_order"] != "ZYX":
        raise ValueError(
            f"Unexpected coordinate_order {record['coordinate_order']!r}; "
            "Stage 06 requires strict ZYX ordering."
        )

    for field in ("candidate_center_zyx", "patch_shape_zyx", "patch_spacing_zyx_mm"):
        value = record[field]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{field} must be a 3-element ZYX vector, got {value!r}")


# ============================================================================
# PATCH LOADING (no renormalization, no geometry recomputation)
# ============================================================================

def resolve_patch_path(stage05_dir: str, patch_file: str) -> str:
    path = patch_file if os.path.isabs(patch_file) else os.path.join(stage05_dir, patch_file)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Classifier patch not found:\n{path}")
    return path


def load_classifier_patch(path: str) -> torch.Tensor:
    """
    Load one Stage 05 patch and reshape (64,64,64) -> (1,1,64,64,64).

    This is a pure reshape. Stage 06 performs NO HU clipping and NO
    normalization -- it only verifies Stage 05 already normalized the
    patch into [0, 1], which is a check, not a correction.
    """
    patch = np.load(path, allow_pickle=False)
    patch = np.asarray(patch, dtype=np.float32)

    expected_3d = (CLASSIFIER_SIZE, CLASSIFIER_SIZE, CLASSIFIER_SIZE)

    if patch.shape == expected_3d:
        patch = patch[np.newaxis, np.newaxis, ...]
    elif patch.shape == INPUT_TENSOR_SHAPE:
        pass
    else:
        raise ValueError(
            f"Unexpected classifier patch shape: {patch.shape}\n"
            f"Expected {expected_3d} (as persisted by Stage 05) "
            f"or {INPUT_TENSOR_SHAPE}."
        )

    if not np.isfinite(patch).all():
        raise ValueError(f"Patch contains NaN or Inf values:\n{path}")

    patch_min, patch_max = float(patch.min()), float(patch.max())
    if patch_min < NORMALIZED_MIN - NORMALIZATION_TOLERANCE or patch_max > NORMALIZED_MAX + NORMALIZATION_TOLERANCE:
        raise ValueError(
            "Stage 05 classifier patch is not normalized to [0,1]: "
            f"min={patch_min:.6f}, max={patch_max:.6f}\nPatch: {path}\n"
            "Stage 06 does not renormalize -- fix Stage 05 output instead."
        )

    return torch.from_numpy(patch)


# ============================================================================
# INFERENCE
# ============================================================================

def scalar_prediction(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().reshape(-1)[0])


@torch.no_grad()
def classify_patch(model, patch: torch.Tensor, device: torch.device) -> Dict[str, float]:
    patch = patch.to(device, non_blocking=True)
    outputs = model(patch)

    predictions: Dict[str, float] = {}
    for feature in FEATURE_NAMES:
        if feature not in outputs:
            raise RuntimeError(f"Model output is missing required head '{feature}'.")
        predictions[feature] = round(scalar_prediction(outputs[feature]), 4)

    return predictions


def build_output_record(record: Dict[str, Any], predictions: Dict[str, float]) -> Dict[str, Any]:
    """
    Assemble the strict output schema. Every geometry/identity field is
    copied verbatim from the Stage 05 record -- nothing is recomputed,
    rounded, or renamed.
    """
    return {
        "candidate_id": record["candidate_id"],
        "patch_file": record["patch_file"],
        "coordinate_order": record["coordinate_order"],
        "predictions": predictions,
        "candidate_center_zyx": list(record["candidate_center_zyx"]),
        "patch_shape_zyx": list(record["patch_shape_zyx"]),
        "patch_spacing_zyx_mm": list(record["patch_spacing_zyx_mm"]),
        "geometry_authority": record["geometry_authority"],
    }


# ============================================================================
# MAIN CLASSIFICATION LOOP
# ============================================================================

def classify_candidates(
    model,
    manifest: Dict[str, Any],
    stage05_dir: str,
    device: torch.device,
) -> Dict[str, Any]:
    patch_records = manifest["patches"]
    results: List[Dict[str, Any]] = []
    failed = 0

    print(f"Candidates: {len(patch_records)}")

    for index, record in enumerate(patch_records):
        try:
            validate_stage05_record(record)
            patch_path = resolve_patch_path(stage05_dir, record["patch_file"])
            patch = load_classifier_patch(patch_path)
            predictions = classify_patch(model, patch, device)
            results.append(build_output_record(record, predictions))

            print(
                f"[{index + 1}/{len(patch_records)}] "
                f"candidate_id={record['candidate_id']} -> "
                + ", ".join(f"{k}={v:.3f}" for k, v in predictions.items())
            )

        except Exception as exc:
            failed += 1
            cand_id = record.get("candidate_id", "?") if isinstance(record, dict) else "?"
            print(f"[{index + 1}/{len(patch_records)}] [ERROR] candidate_id={cand_id}: {exc}")

    return {
        "results": results,
        "num_candidates": len(patch_records),
        "num_classified": len(results),
        "num_failed": failed,
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LungInsight Stage 06 -- classify Stage 05 candidate patches."
    )
    parser.add_argument("patient_id", help="Patient/output identifier, e.g. LIDC-IDRI-0141")
    parser.add_argument("--checkpoint", required=True, help="Path to trained classifier checkpoint.")
    parser.add_argument("--output-root", default="output", help="Pipeline output root. Default: output")
    parser.add_argument(
        "--stage05-dir",
        default=None,
        help="Stage 05 directory. Default: <output-root>/<patient_id>/05_classifier_patches",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Stage 06 output directory. Default: <output-root>/<patient_id>/06_classification",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patient_output = os.path.join(args.output_root, args.patient_id)

    stage05_dir = (
        os.path.abspath(args.stage05_dir)
        if args.stage05_dir
        else os.path.join(patient_output, "05_classifier_patches")
    )
    stage06_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(patient_output, "06_classification")
    )

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    os.makedirs(stage06_dir, exist_ok=True)

    print("=" * 72)
    print("LUNGINSIGHT -- STAGE 06: 3D MULTI-HEAD CLASSIFIER INFERENCE")
    print("=" * 72)
    print(f"Patient    : {args.patient_id}")
    print(f"Stage 05   : {stage05_dir}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Output     : {stage06_dir}")
    print(f"Device     : {device}")
    print()

    manifest = load_stage05_manifest(stage05_dir)
    print(f"Loaded Stage 05 manifest with {len(manifest['patches'])} patches.")
    print()

    model = load_model(checkpoint_path, device)
    print()

    classification = classify_candidates(
        model=model,
        manifest=manifest,
        stage05_dir=stage05_dir,
        device=device,
    )

    output_document = {
        "stage": 6,
        "patient_id": args.patient_id,
        "checkpoint": checkpoint_path,
        "device": str(device),
        "num_candidates": classification["num_candidates"],
        "num_classified": classification["num_classified"],
        "num_failed": classification["num_failed"],
        # Stage 07's load_stage06_classification() requires a top-level
        # object with a "candidates" list -- a bare list is rejected.
        "candidates": classification["results"],
    }

    json_path = os.path.join(stage06_dir, "classification.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_document, f, indent=2)

    print()
    print("=" * 72)
    print("STAGE 06 COMPLETE")
    print("=" * 72)
    print(f"Candidates seen : {classification['num_candidates']}")
    print(f"Classified      : {classification['num_classified']}")
    print(f"Failed          : {classification['num_failed']}")
    print()
    print(f"Classification JSON: {os.path.abspath(json_path)}")


if __name__ == "__main__":
    main()