#!/usr/bin/env python3
"""Validate the LungInsight pipeline schema and six-stage script contracts without model inference."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
IMAGING = ROOT / "backend" / "Imaging"
if str(IMAGING) not in sys.path:
    sys.path.insert(0, str(IMAGING))

from pipeline_schema import PipelineSchemaError, get_required, load_schema  # noqa: E402

SCRIPT_FILES = (
    "01_hu_and_crop.py",
    "02_detect_and_patch.py",
    "03_nodule_classification.py",
    "04_malignancy_prediction.py",
    "04_malignancy_classifier.py",
    "05_gradcam.py",
    "06_full_ct_gradcam.py",
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise PipelineSchemaError(message)


def validate_schema_contracts(schema: dict[str, Any]) -> list[str]:
    """Check schema references and cross-stage contracts; return concise pass lines."""
    passed: list[str] = []
    version = get_required(schema, "schema_version", expected_type=str)
    supported = get_required(schema, "metadata_policy.supported_schema_versions", expected_type=list)
    _assert(version in supported, f"Schema version {version!r} is not in metadata_policy.supported_schema_versions.")

    attributes = get_required(schema, "attribute_contracts.stage03_ranked_attributes", expected_type=list)
    _assert(bool(attributes), "The canonical Stage 03 attribute list is empty.")
    _assert(all(isinstance(x, str) and x for x in attributes), "The canonical attribute list must contain non-empty strings only.")
    _assert(len(attributes) == len(set(attributes)), "The canonical Stage 03 attribute list contains duplicate feature names.")
    consumers = (
        "stage_03_nodule_classification.model.attribute_order_ref",
        "stage_04_malignancy_prediction.feature_names_ref",
        "stage_04_malignancy_classifier.feature_names_ref",
        "stage_05_gradcam.attributes_ref",
        "stage_06_full_ct_gradcam.attributes_ref",
    )
    for ref_path in consumers:
        ref = get_required(schema, ref_path, expected_type=str)
        resolved = get_required(schema, ref, expected_type=list)
        _assert(resolved == attributes, f"{ref_path} resolves to a feature order different from the canonical list.")
    stage3_scales = get_required(schema, "stage_03_nodule_classification.attribute_scales", expected_type=dict)
    _assert(set(attributes).issubset(stage3_scales), "Stage 03 is missing an attribute_scales entry for one or more canonical attributes.")
    passed.append(f"Schema v{version}; canonical attribute order shared by Stages 03–06.")

    # Stage 01 filenames are the sole source for Stage 01 artifacts used downstream.
    stage1_files = get_required(schema, "stage_01_hu_and_crop.filenames", expected_type=dict)
    stage2_input = get_required(schema, "stage_02_detect_and_patch.input_volume.filename_key", expected_type=str)
    _assert(stage2_input in stage1_files, f"Stage 02 input volume filename_key {stage2_input!r} is not defined by Stage 01.")

    paths2 = get_required(schema, "paths.stage_02", expected_type=dict)
    patch_outputs = get_required(schema, "stage_02_detect_and_patch.patch_outputs", expected_type=dict)
    patch_root = Path(get_required(patch_outputs, "root", expected_type=str))
    expected_patch_root = Path(paths2["directory"]) / paths2["patch_directory"]
    _assert(patch_root.as_posix().rstrip("/") == expected_patch_root.as_posix(),
            f"Stage 02 patch root {patch_root} conflicts with canonical paths {expected_patch_root}.")
    for key in ("array_directory", "visualization_directory"):
        path = Path(get_required(patch_outputs, key, expected_type=str))
        try:
            path.relative_to(patch_root)
        except ValueError as exc:
            raise PipelineSchemaError(f"Stage 02 {key}={path} must be nested under patch root {patch_root}.") from exc
    manifest = Path(get_required(patch_outputs, "manifest", expected_type=str))
    declared_manifest = Path(get_required(schema, "stage_03_nodule_classification.inputs.patch_manifest", expected_type=str))
    _assert(manifest == declared_manifest, f"Stage 03 expects manifest {declared_manifest}, but Stage 02 writes {manifest}.")
    _assert(manifest.name.endswith(".json"), f"Stage 02 manifest path is not a JSON filename: {manifest}.")
    passed.append("Stage 01 -> Stage 02 filenames, and Stage 02 -> Stage 03 manifest path, are consistent.")

    # LoG proposals are given their own patch tree/manifest and Stage 03 outputs.
    log_patch_outputs = get_required(schema, "stage_02_detect_and_patch.diagnostic_patch_outputs", expected_type=dict)
    log_root = Path(get_required(log_patch_outputs, "root", expected_type=str))
    log_arrays = Path(get_required(log_patch_outputs, "array_directory", expected_type=str))
    log_visualizations = Path(get_required(log_patch_outputs, "visualization_directory", expected_type=str))
    for label, path in (("array directory", log_arrays), ("visualization directory", log_visualizations)):
        try:
            path.relative_to(log_root)
        except ValueError as exc:
            raise PipelineSchemaError(f"Stage 02 diagnostic {label} {path} must be nested under {log_root}.") from exc
    log_manifest = Path(get_required(log_patch_outputs, "manifest", expected_type=str))
    log_input = Path(get_required(schema, "stage_03_nodule_classification.inputs.log_diagnostic_patch_manifest", expected_type=str))
    _assert(log_manifest == log_input, f"Stage 03 LoG manifest input {log_input} differs from Stage 02 diagnostic output {log_manifest}.")
    _assert(log_manifest != manifest, "LoG and authoritative patch manifests must remain separate.")
    _assert(bool(get_required(schema, "stage_03_nodule_classification.outputs.log_diagnostic_predictions_json", expected_type=str)),
            "Stage 03 diagnostic predictions output filename is missing.")
    _assert(bool(get_required(schema, "stage_03_nodule_classification.outputs.log_diagnostic_predictions_csv", expected_type=str)),
            "Stage 03 diagnostic CSV output filename is missing.")
    _assert(bool(get_required(schema, "stage_02_detect_and_patch.diagnostic_log_detector.stage03_attribute_inference_enabled_by_default", expected_type=bool))
            == bool(get_required(schema, "stage_03_nodule_classification.inputs.process_log_diagnostic_default", expected_type=bool)),
            "Stage 02 LoG-to-Stage 03 default settings disagree.")
    passed.append("LoG patch manifest and Stage 03 diagnostic outputs are configured separately from authoritative predictions.")

    stage3_inputs = get_required(schema, "stage_03_nodule_classification.inputs", expected_type=dict)
    stage3_shape = tuple(get_required(stage3_inputs, "patch_shape_zyx", expected_type=list))
    stage5_shape = tuple(get_required(schema, "stage_05_gradcam.patch_shape_zyx", expected_type=list))
    _assert(stage3_shape == stage5_shape, f"Stage 03 patch shape {stage3_shape} differs from Stage 05 Grad-CAM shape {stage5_shape}.")
    _assert(len(stage3_shape) == 3 and all(isinstance(n, int) and n > 0 for n in stage3_shape),
            f"Expected a positive 3D Stage 03 patch shape, got {stage3_shape}.")
    patch_axes = tuple(get_required(schema, "stage_02_detect_and_patch.patch_outputs.array_axes", expected_type=list))
    _assert(patch_axes == ("z", "y", "x"),
            "Stage 02 patch arrays must preserve ZYX axes for the existing checkpoints.")
    patch_dtype = get_required(schema, "stage_02_detect_and_patch.patch_outputs.array_dtype", expected_type=str)
    input_dtype = get_required(schema, "stage_03_nodule_classification.inputs.dtype", expected_type=str)
    _assert(patch_dtype == "float32" and input_dtype == "float32",
            f"Stage 02/03 patch dtype contract must remain float32; got {patch_dtype!r}/{input_dtype!r}.")
    passed.append(f"Patch contract consistent: shape={stage3_shape}, axes=ZYX, dtype={patch_dtype}.")

    stage4 = get_required(schema, "stage_04_malignancy_prediction", expected_type=dict)
    target_range = get_required(stage4, "target_rating_range", expected_type=list)
    range_ref = get_required(stage4, "prediction_transform.target_range_ref", expected_type=str)
    _assert(get_required(schema, range_ref, expected_type=list) == target_range,
            "Stage 04 prediction transform target_range_ref does not resolve to its target_rating_range.")
    _assert(target_range[0] < target_range[1], f"Invalid Stage 04 target rating range: {target_range}.")
    _assert(get_required(stage4, "model_path", expected_type=str), "Stage 04 joblib model_path is empty.")
    ordinal = get_required(schema, "stage_04_malignancy_classifier", expected_type=dict)
    ordinal_range = get_required(ordinal, "input_rating_range", expected_type=list)
    classes = get_required(ordinal, "rating_classes", expected_type=list)
    groups = get_required(ordinal, "rating_groups", expected_type=dict)
    group_values = [rating for values in groups.values() for rating in values]
    _assert(len(ordinal_range) == 2 and ordinal_range[0] < ordinal_range[1], "Invalid Stage 04 ordinal input_rating_range.")
    _assert(classes and len(classes) == len(set(classes)), "Stage 04 ordinal rating_classes must be unique and non-empty.")
    _assert(set(classes) == set(group_values) and len(group_values) == len(set(group_values)),
            "Stage 04 ordinal rating_groups must partition the rating_classes without omissions or duplicates.")
    threshold = float(get_required(ordinal, "confidence_threshold"))
    _assert(0.0 <= threshold <= 1.0, "Stage 04 ordinal confidence_threshold must be within [0,1].")
    paths4 = get_required(schema, "paths.stage_04", expected_type=dict)
    _assert(paths4["directory"] and paths4["output_json"].endswith(".json"), "Stage 04 output path config is invalid.")
    passed.append("Stage 04 feature order and primary/alternative rating transforms are valid; no checkpoint is changed.")

    # Visualization contracts are shared; only Stage 06 adds the detector bounding-box color.
    palette = get_required(schema, "visualization.rainbow_palette", expected_type=list)
    _assert(len(palette) >= 2, "Rainbow palette must contain at least two stops.")
    positions = [float(item["position"]) for item in palette]
    _assert(positions == sorted(positions) and positions[0] == 0.0 and positions[-1] == 1.0,
            f"Rainbow palette positions must be sorted from 0 to 1; got {positions}.")
    for section in ("stage_05_gradcam", "stage_06_full_ct_gradcam"):
        ref = get_required(schema, f"{section}.palette_ref", expected_type=str)
        _assert(get_required(schema, ref, expected_type=list) == palette, f"{section} palette reference is inconsistent.")
    display_ref = get_required(schema, "stage_06_full_ct_gradcam.display_window_ref", expected_type=str)
    display_window = get_required(schema, display_ref, expected_type=dict)
    _assert(set(("level", "width")).issubset(display_window), "Shared CT display window must define level and width.")
    _assert(float(display_window["width"]) > 0, "Shared CT display-window width must be positive.")
    _assert(0.0 <= float(get_required(schema, "stage_05_gradcam.default_alpha")) <= 1.0,
            "Stage 05 default_alpha must be within [0,1].")
    _assert(0.0 <= float(get_required(schema, "stage_06_full_ct_gradcam.default_alpha")) <= 1.0,
            "Stage 06 default_alpha must be within [0,1].")
    passed.append("Stages 05–06 share palette, outline color, and schema-driven CT display-window defaults.")
    return passed


def check_artifacts(schema: dict[str, Any]) -> list[str]:
    """Check runtime files that must exist locally; remote ViTDet3D cache is not mandatory."""
    missing: list[str] = []
    stage3_model = get_required(schema, "stage_03_nodule_classification.model", expected_type=dict)
    checkpoint_root = ROOT / stage3_model["checkpoint_root"]
    helper_paths = [
        ROOT / "backend" / "models" / "malignancy" / "frozen_attribute_common.py",
        ROOT / "backend" / "models" / "nodule_ranker_3d_distributional_v1" / "spatial_v1" / "model_objective.py",
    ]
    missing.extend(str(p.relative_to(ROOT)) for p in helper_paths if not p.is_file())
    stage5 = get_required(schema, "stage_05_gradcam.checkpoint_by_attribute", expected_type=dict)
    for checkpoint_rel in sorted(set(stage5.values())):
        path = checkpoint_root / checkpoint_rel
        if not path.is_file():
            missing.append(str(path.relative_to(ROOT)))
    model_path = ROOT / get_required(schema, "stage_04_malignancy_prediction.model_path", expected_type=str)
    if not model_path.is_file():
        missing.append(str(model_path.relative_to(ROOT)))
    if missing:
        raise FileNotFoundError(
            "Required local inference artifacts are missing:\n  " + "\n  ".join(missing) +
            "\nFix: restore the exact existing helper/model/checkpoint files or correct paths in meta_schema.json. "
            "Do not retrain or edit checkpoints."
        )
    return ["All required local Stage 03 and Stage 04 inference artifacts are present."]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-artifacts", action="store_true", help="Also verify local model helper files and trained checkpoints exist.")
    args = parser.parse_args()
    schema = load_schema(__file__)
    messages = validate_schema_contracts(schema)

    errors: list[str] = []
    for filename in SCRIPT_FILES:
        script = IMAGING / filename
        if not script.is_file():
            errors.append(f"Missing pipeline script: {script.relative_to(ROOT)}")
            continue
        try:
            source = script.read_text(encoding="utf-8-sig")
            compile(source, str(script), "exec")
        except (OSError, SyntaxError) as exc:
            errors.append(f"Script syntax/read failure in {script.relative_to(ROOT)}: {exc}")
    if errors:
        raise RuntimeError("Pipeline script validation failed:\n  " + "\n  ".join(errors))
    messages.append(f"All {len(SCRIPT_FILES)} supplied stage scripts parse successfully.")

    # Compile checks do not catch undefined global names; this guards previously observed path-name failures.
    stage4_source = (IMAGING / "04_malignancy_prediction.py").read_text(encoding="utf-8-sig")
    stage4_tree = ast.parse(stage4_source)
    assigned_names = {
        target.id
        for node in ast.walk(stage4_tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in ((node.targets if isinstance(node, ast.Assign) else [node.target]))
        if isinstance(target, ast.Name)
    }
    _assert("STAGE03_PATHS" in assigned_names,
            "Stage 04 references Stage 03 paths but never loads STAGE03_PATHS from meta_schema.json. Fix backend/Imaging/04_malignancy_prediction.py.")
    messages.append("Stage 04 Stage 03-input path configuration is defined (undefined-path regression guard passed).")
    if args.check_artifacts:
        messages.extend(check_artifacts(schema))
    for message in messages:
        print(f"[PASS] {message}")
    print("[NOTE] Static validation does not execute CT processing, download ViTDet3D, or run trained-model inference.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PipelineSchemaError, FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(2)
