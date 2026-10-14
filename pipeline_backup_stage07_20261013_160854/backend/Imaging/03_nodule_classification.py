#!/usr/bin/env python3
"""LungInsight v2 Stage 03: frozen spatial-hybrid nodule attribute inference.

Reads the authoritative Stage 02 patches manifest and a separate LoG diagnostic
patch manifest. It uses the five pre-existing hybrid checkpoints unchanged and
writes six attribute ratings (not internal structure, calcification, or malignancy).
Authoritative predictions stay in predictions.json; LoG proposals go to separate
diagnostic outputs and are never Stage 04 malignancy inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pipeline_schema import get_required, load_schema, check_meta_schema_version, validate_stage_version

SCHEMA = load_schema(__file__)
STAGE02_SCHEMA = get_required(SCHEMA, "stage_02_detect_and_patch", expected_type=dict)
STAGE03_SCHEMA = get_required(SCHEMA, "stage_03_nodule_classification", expected_type=dict)
STAGE03_MODEL = get_required(STAGE03_SCHEMA, "model", expected_type=dict)
STAGE03_OUTPUTS = get_required(STAGE03_SCHEMA, "outputs", expected_type=dict)
STAGE03_PATHS = get_required(SCHEMA, "paths.stage_03", expected_type=dict)
TRANSFER_IDENTITY_FIELDS = tuple(get_required(STAGE03_SCHEMA, "transfer_identity_fields", expected_type=list))
V2_ROOT = Path(__file__).resolve().parents[2]
MALIGNANCY_HELPERS = V2_ROOT / "backend" / "models" / "malignancy"
if str(MALIGNANCY_HELPERS) not in sys.path:
    sys.path.insert(0, str(MALIGNANCY_HELPERS))
try:
    from frozen_attribute_common import (  # noqa: E402
        CHECKPOINT_GROUPS as HELPER_CHECKPOINT_GROUPS,
        DEFAULT_CHECKPOINT_ROOT as HELPER_DEFAULT_CHECKPOINT_ROOT,
        FEATURE_NAMES as HELPER_FEATURE_NAMES,
        EXPECTED_PATCH_SHAPE as HELPER_EXPECTED_PATCH_SHAPE,
        choose_device,
        load_frozen_models,
        sha256_file,
        stage03_prediction_map,
    )
except ImportError as exc:
    raise ImportError(
        f"Stage 03 requires the existing inference helper at {MALIGNANCY_HELPERS / 'frozen_attribute_common.py'}. "
        "Restore that project file and its compatible dependencies; the refactor does not retrain or recreate it."
    ) from exc

FEATURE_NAMES = tuple(get_required(SCHEMA, str(STAGE03_MODEL["attribute_order_ref"]), expected_type=list))
CHECKPOINT_GROUPS = {str(k): tuple(v) for k, v in get_required(STAGE03_MODEL, "checkpoint_groups", expected_type=dict).items()}
EXPECTED_PATCH_SHAPE = tuple(get_required(STAGE03_SCHEMA, "inputs.patch_shape_zyx", expected_type=list))
DEFAULT_CHECKPOINT_ROOT = V2_ROOT / get_required(STAGE03_MODEL, "checkpoint_root")
DEFAULT_OUTPUT_ROOT = V2_ROOT / get_required(SCHEMA, "paths.output_root")

if tuple(HELPER_EXPECTED_PATCH_SHAPE) != EXPECTED_PATCH_SHAPE:
    raise RuntimeError(
        "Stage 03 patch-shape mismatch: frozen_attribute_common.EXPECTED_PATCH_SHAPE="
        f"{tuple(HELPER_EXPECTED_PATCH_SHAPE)}, but meta_schema.json requires {EXPECTED_PATCH_SHAPE}. "
        "Correct the schema to describe the existing checkpoints; do not resize patches or alter checkpoints."
    )
if tuple(HELPER_FEATURE_NAMES) != FEATURE_NAMES:
    raise RuntimeError(
        "Stage 03 feature-order mismatch: frozen_attribute_common.FEATURE_NAMES="
        f"{tuple(HELPER_FEATURE_NAMES)}, but meta_schema.json requires {FEATURE_NAMES}. "
        "Update the schema only if it accurately describes the existing checkpoint heads; do not retrain or alter checkpoints."
    )
if {k: tuple(v) for k, v in HELPER_CHECKPOINT_GROUPS.items()} != CHECKPOINT_GROUPS:
    raise RuntimeError(
        "Stage 03 checkpoint-group mismatch between frozen_attribute_common and meta_schema.json. "
        "Correct the schema to match the existing frozen checkpoints before inference."
    )
if tuple(HELPER_DEFAULT_CHECKPOINT_ROOT.resolve().parts[-len(DEFAULT_CHECKPOINT_ROOT.parts):]) != tuple(DEFAULT_CHECKPOINT_ROOT.resolve().parts[-len(DEFAULT_CHECKPOINT_ROOT.parts):]):
    # Do not fail solely on differing absolute repo prefixes; compare normalized repo-relative suffixes below.
    helper_suffix = Path(*HELPER_DEFAULT_CHECKPOINT_ROOT.parts[-len(DEFAULT_CHECKPOINT_ROOT.parts):])
    schema_suffix = Path(*DEFAULT_CHECKPOINT_ROOT.parts[-len(DEFAULT_CHECKPOINT_ROOT.parts):])
    if helper_suffix != schema_suffix:
        print(f"[warning] schema checkpoint root {DEFAULT_CHECKPOINT_ROOT} differs from helper default {HELPER_DEFAULT_CHECKPOINT_ROOT}; using schema path.", file=sys.stderr)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def resolve_patch_array(record: dict[str, Any], manifest_path: Path) -> Path:
    raw = record.get("array_path")
    filename = f"patch_{int(record['patch_id']):03d}.npy"
    candidates: list[Path] = []
    if raw:
        p = Path(str(raw)).expanduser()
        candidates.append(p if p.is_absolute() else manifest_path.parent / p)
    # Standard Stage 02 layout, independent of stale absolute paths in a manifest.
    candidates.extend([
        manifest_path.parent / "arrays" / filename,
        manifest_path.parent.parent / "arrays" / filename,
    ])
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        f"Patch array for patch_id={record.get('patch_id')} was not found. Tried: "
        + "; ".join(str(p) for p in candidates)
    )


def transfer_identity_fields(patch: dict[str, Any]) -> dict[str, Any]:
    out = {k: patch[k] for k in TRANSFER_IDENTITY_FIELDS if k in patch}
    out["prediction_space"] = STAGE03_SCHEMA["space"]
    out["detection_source"] = patch.get("detection_source", "ViTDet3D")
    out["patch_array_path"] = patch.get("array_path")
    return out


def run_log_diagnostic_inference(
    patient_dir: Path,
    models: dict[str, torch.nn.Module],
    checkpoint_metadata: dict[str, Any],
    device: torch.device,
    *,
    output_dir: Path,
    collect_attention: bool,
) -> dict[str, Any]:
    """Infer LoG-derived patches separately so they never enter Stage 04's primary input."""
    patient_id = patient_dir.name
    rel_manifest = Path(str(STAGE03_SCHEMA["inputs"]["log_diagnostic_patch_manifest"]))
    manifest_path = patient_dir / rel_manifest
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing Stage 02 LoG diagnostic patch manifest: {manifest_path}. "
            "Rerun Stage 02 with its default LoG diagnostic detector enabled, or explicitly skip LoG inference with --skip-log-diagnostic."
        )
    manifest = load_json(manifest_path)
    check_meta_schema_version(manifest, SCHEMA, context=str(manifest_path), require=False)
    validate_stage_version(
        manifest.get("schema_version"), STAGE02_SCHEMA["manifest_schema_version"],
        context="Stage 02 LoG diagnostic patch manifest",
    )
    patches = manifest.get("patches")
    if not isinstance(patches, list) or not all(isinstance(patch, dict) for patch in patches):
        raise ValueError(f"{manifest_path}: expected a list of JSON objects under 'patches'.")
    patch_ids = [patch.get("patch_id") for patch in patches]
    if any(not isinstance(pid, int) for pid in patch_ids) or len(patch_ids) != len(set(patch_ids)):
        raise ValueError(f"{manifest_path}: patch_id values must be unique integers.")
    patches = sorted(patches, key=lambda patch: int(patch["patch_id"]))

    log_attention_name = str(STAGE03_OUTPUTS["log_attention_directory"]).strip("/")
    log_attention_dir = output_dir / log_attention_name
    if collect_attention:
        log_attention_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for n, patch in enumerate(patches, start=1):
        array_path = resolve_patch_array(patch, manifest_path)
        patch_array = np.load(array_path, allow_pickle=False)
        if tuple(patch_array.shape) != EXPECTED_PATCH_SHAPE:
            raise ValueError(
                f"{array_path}: Stage 03 requires raw-HU shape {EXPECTED_PATCH_SHAPE}; "
                f"got {patch_array.shape}. Do not silently resize/resample diagnostic patches."
            )
        if patch_array.dtype != np.float32:
            raise ValueError(f"{array_path}: expected float32 patch data, got {patch_array.dtype}.")
        predictions, attention_by_attribute = stage03_prediction_map(
            patch_array, models, device, collect_attention=collect_attention
        )
        record = transfer_identity_fields(patch)
        record["patch_array_path"] = str(array_path)
        record["patient_id"] = patient_id
        record["detection_source"] = "LoG"
        record["purpose"] = "research_comparison_only"
        record["stage04_eligible"] = False
        record["attributes"] = {
            name: {
                "expected_rating": float(predictions[name]),
                "rating_scale": list(STAGE03_SCHEMA["attribute_scales"][name]["rating_range"]),
                "score_semantics": STAGE03_SCHEMA["attribute_scales"][name]["semantics"],
                "checkpoint_group": next(g for g, attrs in CHECKPOINT_GROUPS.items() if name in attrs),
            }
            for name in FEATURE_NAMES
        }
        record["warnings"] = [
            "log_proposal_is_diagnostic_not_authoritative",
            "attribute_predictions_are_not_validated_for_LoG_proposals",
        ]
        if collect_attention:
            attention_tensor = np.stack([attention_by_attribute[name] for name in FEATURE_NAMES]).astype(np.float32)
            attention_rel = Path(log_attention_name) / f"patch_{int(patch['patch_id']):05d}_attribute_attention.npy"
            np.save(output_dir / attention_rel, attention_tensor, allow_pickle=False)
            record["attention"] = {
                "attribute_spatial_attention_path": attention_rel.as_posix(),
                "attribute_order": list(FEATURE_NAMES),
                "grid_shape_zyx": list(STAGE03_MODEL["attention_grid_shape_zyx"]),
                "semantics": "softmax attention over spatial tokens; not a segmentation mask",
            }
        records.append(record)
        anatomical = patch.get("anatomical_location")
        flat: dict[str, Any] = {
            "patient_id": patient_id,
            "patch_id": patch.get("patch_id"),
            "candidate_id": patch.get("candidate_id"),
            "diagnostic_id": patch.get("diagnostic_id"),
            "detection_source": "LoG",
            "log_response": patch.get("log_response"),
            "scale_mm": patch.get("scale_mm"),
            "detector_score": patch.get("detector_score"),
            "center_lps_mm": json.dumps(patch.get("center_lps_mm")),
            "anatomical_region": (anatomical.get("region") if isinstance(anatomical, dict) else None),
            "warnings": ";".join(record["warnings"]),
        }
        flat.update({name: predictions[name] for name in FEATURE_NAMES})
        flat_rows.append(flat)
        if len(patches) <= 20 or n % 50 == 0 or n == len(patches):
            print(
                f"[{patient_id}][LoG] {n}/{len(patches)} patch={patch['patch_id']} -> "
                + ", ".join(f"{key}={predictions[key]:.3f}" for key in FEATURE_NAMES),
                flush=True,
            )

    output = {
        "schema_version": STAGE03_SCHEMA["schema_version"],
        "meta_schema_version": SCHEMA["schema_version"],
        "stage": STAGE03_PATHS["directory"],
        "patient_id": patient_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "route": "spatial_hybrid",
        "objective": "ordinal_reader",
        "detection_source": "LoG",
        "purpose": "research_comparison_only",
        "stage04_eligible": False,
        "prediction_space": STAGE03_SCHEMA["space"],
        "attribute_order": list(FEATURE_NAMES),
        "attribute_outputs": STAGE03_SCHEMA["prediction_semantics"]["expected_rating"],
        "malignancy_prediction_included": False,
        "unsupported_attributes_omitted": list(STAGE03_SCHEMA["unsupported_attributes"]),
        "patch_manifest": str(manifest_path.resolve()),
        "patch_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_metadata": checkpoint_metadata,
        "attention_enabled": bool(collect_attention),
        "n_patches": len(records),
        "diagnostic_warning": (
            "These rows originate from LoG proposals, not the authoritative ViTDet3D detector. "
            "Their attribute outputs are exploratory and are kept separate from predictions.json, "
            "which is the only Stage 03 input used by the primary Stage 04 malignancy-rating step."
        ),
        "predictions": records,
    }
    json_path = output_dir / STAGE03_OUTPUTS["log_diagnostic_predictions_json"]
    json_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    csv_path = output_dir / STAGE03_OUTPUTS["log_diagnostic_predictions_csv"]
    fieldnames = list(flat_rows[0].keys()) if flat_rows else [
        "patient_id", "patch_id", "candidate_id", "diagnostic_id", "detection_source", "log_response", "scale_mm",
        "detector_score", "center_lps_mm", "anatomical_region", "warnings", *FEATURE_NAMES,
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"Saved Stage 03 LoG diagnostic predictions: {json_path} ({len(records)} patches)")
    return output


def run_patient(
    patient_dir: Path,
    models: dict[str, torch.nn.Module],
    checkpoint_metadata: dict[str, Any],
    device: torch.device,
    *,
    output_root: Path,
    collect_attention: bool,
    process_log_diagnostic: bool = True,
) -> dict[str, Any]:
    patient_id = patient_dir.name
    patch_manifest_path = patient_dir / STAGE02_SCHEMA["patch_outputs"]["manifest"]
    if not patch_manifest_path.is_file():
        raise FileNotFoundError(f"Missing Stage 02 patch manifest: {patch_manifest_path}")
    patch_manifest = load_json(patch_manifest_path)
    check_meta_schema_version(patch_manifest, SCHEMA, context=str(patch_manifest_path), require=False)
    validate_stage_version(
        patch_manifest.get("schema_version"),
        STAGE02_SCHEMA["manifest_schema_version"],
        context="Stage 02 patch manifest",
    )
    patches = patch_manifest.get("patches")
    if not isinstance(patches, list):
        raise ValueError(f"{patch_manifest_path}: expected a list under 'patches'")
    if not all(isinstance(patch, dict) for patch in patches):
        raise ValueError(f"{patch_manifest_path}: every patch record must be a JSON object.")
    patch_ids = [p.get("patch_id") for p in patches]
    if any(not isinstance(pid, int) for pid in patch_ids):
        raise ValueError(f"{patch_manifest_path}: every patch requires an integer patch_id")
    if len(patch_ids) != len(set(patch_ids)):
        raise ValueError(f"{patch_manifest_path}: duplicate patch_id values")
    patches = sorted(patches, key=lambda p: int(p["patch_id"]))

    out_dir = patient_dir / STAGE03_PATHS["directory"]
    attention_dir = out_dir / STAGE03_OUTPUTS["attention_directory"].strip("/")
    out_dir.mkdir(parents=True, exist_ok=True)
    if collect_attention:
        attention_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for n, patch in enumerate(patches, start=1):
        array_path = resolve_patch_array(patch, patch_manifest_path)
        patch_array = np.load(array_path, allow_pickle=False)
        if tuple(patch_array.shape) != EXPECTED_PATCH_SHAPE:
            raise ValueError(
                f"{array_path}: hybrid rankers require raw-HU shape {EXPECTED_PATCH_SHAPE}; "
                f"got {patch_array.shape}. Do not silently resize/resample adaptive patches."
            )
        if patch_array.dtype != np.float32:
            raise ValueError(f"{array_path}: expected float32 patch data, got {patch_array.dtype}.")
        predictions, attention_by_attribute = stage03_prediction_map(
            patch_array, models, device, collect_attention=collect_attention
        )
        record = transfer_identity_fields(patch)
        record["patch_array_path"] = str(array_path)
        record["patient_id"] = patient_id
        record["detection_source"] = str(patch.get("detection_source", "ViTDet3D"))
        record["stage04_eligible"] = True
        record["attributes"] = {
            name: {
                "expected_rating": float(predictions[name]),
                "rating_scale": list(STAGE03_SCHEMA["attribute_scales"][name]["rating_range"]),
                "score_semantics": STAGE03_SCHEMA["attribute_scales"][name]["semantics"],
                "checkpoint_group": next(g for g, attrs in CHECKPOINT_GROUPS.items() if name in attrs),
            }
            for name in FEATURE_NAMES
        }
        anatomical = patch.get("anatomical_location")
        warnings: list[str] = []
        if isinstance(anatomical, dict) and anatomical.get("center_inside_lung_mask") is False:
            warnings.append("detector_center_outside_lung_mask")
        if isinstance(anatomical, dict) and anatomical.get("region") == "outside_lung_mask":
            warnings.append("anatomical_region_outside_lung_mask")
        record["warnings"] = warnings
        if collect_attention:
            # Canonical FEATURE_NAMES order, then the model's 8x8x8 spatial grid.
            attention_tensor = np.stack([attention_by_attribute[name] for name in FEATURE_NAMES]).astype(np.float32)
            attention_rel = Path(str(STAGE03_OUTPUTS["attention_directory"])) / f"patch_{int(patch['patch_id']):03d}_attribute_attention.npy"
            np.save(out_dir / attention_rel, attention_tensor, allow_pickle=False)
            record["attention"] = {
                "attribute_spatial_attention_path": str(attention_rel).replace("\\", "/"),
                "attribute_order": list(FEATURE_NAMES),
                "grid_shape_zyx": list(STAGE03_MODEL["attention_grid_shape_zyx"]),
                "semantics": "softmax attention over the model's 512 spatial tokens; not a segmentation mask",
            }
        records.append(record)
        csv_record: dict[str, Any] = {
            "patient_id": patient_id,
            "patch_id": patch.get("patch_id"),
            "candidate_id": patch.get("candidate_id"),
            "detection_source": str(patch.get("detection_source", "ViTDet3D")),
            "detector_score": patch.get("detector_score"),
            "center_lps_mm": json.dumps(patch.get("center_lps_mm")),
            "anatomical_region": (anatomical.get("region") if isinstance(anatomical, dict) else None),
            "warnings": ";".join(warnings),
        }
        csv_record.update({name: predictions[name] for name in FEATURE_NAMES})
        flat_rows.append(csv_record)
        print(f"[{patient_id}] {n}/{len(patches)} patch={patch['patch_id']} -> " + ", ".join(f"{k}={predictions[k]:.3f}" for k in FEATURE_NAMES), flush=True)

    log_output = None
    if process_log_diagnostic:
        log_output = run_log_diagnostic_inference(
            patient_dir,
            models,
            checkpoint_metadata,
            device,
            output_dir=out_dir,
            collect_attention=collect_attention,
        )

    output = {
        "schema_version": STAGE03_SCHEMA["schema_version"],
        "meta_schema_version": SCHEMA["schema_version"],
        "stage": STAGE03_PATHS["directory"],
        "patient_id": patient_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "route": "spatial_hybrid",
        "objective": "ordinal_reader",
        "detection_source": "ViTDet3D",
        "stage04_eligible": True,
        "log_diagnostic_predictions_json": (
            STAGE03_OUTPUTS["log_diagnostic_predictions_json"] if log_output is not None else None
        ),
        "n_log_diagnostic_patches_inferred": (int(log_output["n_patches"]) if log_output is not None else 0),
        "prediction_space": STAGE03_SCHEMA["space"],
        "attribute_order": list(FEATURE_NAMES),
        "attribute_outputs": STAGE03_SCHEMA["prediction_semantics"]["expected_rating"],
        "malignancy_prediction_included": False,
        "unsupported_attributes_omitted": list(STAGE03_SCHEMA["unsupported_attributes"]),
        "patch_manifest": str(patch_manifest_path.resolve()),
        "patch_manifest_sha256": sha256_file(patch_manifest_path),
        "checkpoint_metadata": checkpoint_metadata,
        "attention_enabled": bool(collect_attention),
        "n_patches": len(records),
        "predictions": records,
    }
    json_path = out_dir / STAGE03_OUTPUTS["predictions_json"]
    json_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    csv_path = out_dir / STAGE03_OUTPUTS["predictions_csv"]
    fieldnames = list(flat_rows[0].keys()) if flat_rows else [
        "patient_id", "patch_id", "candidate_id", "detector_score", "center_lps_mm",
        "anatomical_region", "warnings", *FEATURE_NAMES,
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    meta_path = out_dir / STAGE03_OUTPUTS["classification_metadata_json"]
    metadata = {
        "schema_version": output["schema_version"],
        "meta_schema_version": SCHEMA["schema_version"],
        "stage": STAGE03_PATHS["directory"],
        "patient_id": patient_id,
        "route": "spatial_hybrid",
        "objective": "ordinal_reader",
        "attribute_order": list(FEATURE_NAMES),
        "n_patches": len(records),
        "n_log_diagnostic_patches_inferred": (int(log_output["n_patches"]) if log_output is not None else 0),
        "log_diagnostic_predictions_json": (STAGE03_OUTPUTS["log_diagnostic_predictions_json"] if log_output is not None else None),
        "patch_shape_required": list(EXPECTED_PATCH_SHAPE),
        "input_intensity": STAGE03_SCHEMA["preprocessing"]["source_units"],
        "preprocessing": dict(STAGE03_SCHEMA["preprocessing"]),
        "output_semantics": output["attribute_outputs"],
        "checkpoint_hashes": {g: v["sha256"] for g, v in checkpoint_metadata.items()},
        "malignancy_prediction_included": False,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved Stage 03 predictions: {json_path}")
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--patient-id", help="Process one patient ID; default processes all output patient folders with Stage 02 manifests")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    p.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch device spec such as cuda:0")
    p.add_argument("--collect-attention", action="store_true", help="Write spatial attention tensors for each processed patch")
    p.add_argument("--skip-log-diagnostic", action="store_true", help="Skip the separate LoG diagnostic patch manifest; the default processes LoG proposals for Stage 03 attributes")
    p.add_argument("--allow-noncanonical-checkpoint-provenance", action="store_true", help="Disable split-hash/group provenance checks only for deliberate debugging")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    output_root = a.output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"Stage 01/02 output root does not exist: {output_root}")
    device = choose_device(a.device)
    if not a.checkpoint_root.is_dir():
        raise FileNotFoundError(
            f"Stage 03 checkpoint directory not found: {a.checkpoint_root}. "
            "Fix: restore the existing frozen spatial-hybrid checkpoint directory or pass --checkpoint-root. "
            "No GPU or retraining is required if compatible checkpoints are available."
        )
    models, checkpoint_metadata = load_frozen_models(
        a.checkpoint_root, device,
        validate_provenance=not a.allow_noncanonical_checkpoint_provenance,
    )
    if a.patient_id:
        patient_dirs = [output_root / a.patient_id]
    else:
        manifest_rel = Path(str(STAGE02_SCHEMA["patch_outputs"]["manifest"]))
        patient_dirs = sorted(p for p in output_root.iterdir() if p.is_dir() and
                              (p / manifest_rel).is_file())
    if not patient_dirs:
        raise RuntimeError(f"No patient Stage 02 patch manifests found under {output_root}")
    for patient_dir in patient_dirs:
        if not patient_dir.is_dir():
            raise FileNotFoundError(f"Patient output directory not found: {patient_dir}")
        run_patient(
            patient_dir, models, checkpoint_metadata, device,
            output_root=output_root,
            collect_attention=a.collect_attention,
            process_log_diagnostic=(
                bool(STAGE03_SCHEMA["inputs"]["process_log_diagnostic_default"])
                and not a.skip_log_diagnostic
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
