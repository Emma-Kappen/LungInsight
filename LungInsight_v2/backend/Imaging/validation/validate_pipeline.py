#!/usr/bin/env python3
"""Validate the LungInsight pipeline schema and Stage 01–07 contracts without model inference."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2] if SCRIPT_DIR.name == "validation" else SCRIPT_DIR
IMAGING = ROOT / "backend" / "Imaging"
if str(IMAGING) not in sys.path:
    sys.path.insert(0, str(IMAGING))

from pipeline_schema import PipelineSchemaError, get_required, load_schema  # noqa: E402

SCRIPT_FILES = (
    "01_hu_and_crop.py",
    "02_detect_and_patch.py",
    "03_nodule_classification.py",
    "04_malignancy_prediction.py",
    "05_gradcam.py",
    "06_full_ct_gradcam.py",
    "07_visualize.py",
    "attest_heart_mask_review.py",
    "heart_mask_review.py",
)
SCRIPT_SUBDIRECTORIES = {
    "01_hu_and_crop.py": "stages",
    "02_detect_and_patch.py": "stages",
    "03_nodule_classification.py": "stages",
    "04_malignancy_prediction.py": "stages",
    "05_gradcam.py": "inference",
    "06_full_ct_gradcam.py": "visualization",
    "07_visualize.py": "visualization",
    "attest_heart_mask_review.py": "validation",
}


def pipeline_script_path(filename: str) -> Path:
    """Resolve a stage source, accepting the previous flat layout during migration."""
    subdirectory = SCRIPT_SUBDIRECTORIES.get(filename)
    relocated = IMAGING / subdirectory / filename if subdirectory else IMAGING / filename
    legacy = IMAGING / filename
    return relocated if relocated.is_file() or not legacy.is_file() else legacy


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
    for key in (
        "heart_mask_review_montage_pattern",
        "heart_mask_review_metrics.json",
        "heart_mask_review_attestation.json",
    ):
        _assert(isinstance(stage1_files.get(key), str) and bool(stage1_files[key]),
                f"Stage 01 review output filename {key!r} is missing.")
    review_qc = get_required(schema, "stage_01_hu_and_crop.review_qc", expected_type=dict)
    _assert(int(get_required(review_qc, "slices_per_montage")) > 0,
            "Stage 01 review_qc.slices_per_montage must be positive.")
    _assert(int(get_required(review_qc, "columns")) > 0,
            "Stage 01 review_qc.columns must be positive.")
    passed.append("Stage 01 review montage, per-slice metrics, and hash-bound attestation paths are defined.")
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
    passed.append("Optional LoG audit artifacts remain isolated while validated LoG proposals enter the fused production manifest.")

    stage2_filter = get_required(schema, "stage_02_detect_and_patch.candidate_acceptance_filter", expected_type=dict)
    _assert(stage2_filter.get("require_nonempty_heart_mask") is True,
            "Production candidate filtering must fail closed on an empty heart mask.")
    _assert(stage2_filter.get("require_human_review_attestation") is True,
            "Production candidate filtering must require a human review attestation.")
    for key in (
        "candidate_filter_report",
        "rejected_candidates_audit",
        "log_candidate_filter_report",
        "log_rejected_candidates_audit",
    ):
        _assert(bool(get_required(schema, f"stage_02_detect_and_patch.filenames.{key}", expected_type=str)),
                f"Stage 02 is missing the configured output filename {key!r}.")
    passed.append("Stage 02 requires reviewed heart-mask gating before authoritative or diagnostic patch extraction.")

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
    _assert(stage4.get("role") == "brock_only_research_risk_inference",
            "Stage 04 must be configured as Brock-only research risk inference.")
    _assert(get_required(stage4, "model_name", expected_type=str) == "Brock_PanCan_full",
            "Stage 04 must identify the Brock/PanCan model.")
    _assert(get_required(stage4, "model_version", expected_type=str),
            "Stage 04 Brock model version is missing.")
    paths4 = get_required(schema, "paths.stage_04", expected_type=dict)
    _assert(paths4["directory"] and paths4["output_json"].endswith(".json"), "Stage 04 output path config is invalid.")
    passed.append("Stage 04 is configured for Brock-only research risk inference; no reader-rating model or fallback is configured.")

    stage06_fields = get_required(schema, "stage_06_full_ct_gradcam.record_required_fields", expected_type=list)
    _assert(
        {"nodule_id", "candidate_id", "patch_id", "geometry", "detector", "attributes",
         "malignancy", "model_identity", "uncertainty_and_limitations", "explanations"}.issubset(stage06_fields),
        "Stage 06 nodule records do not define the required identity, geometry, model, and explanation fields.",
    )
    passed.append("Stage 06 schema defines stable nodule records and Brock formula provenance/configuration.")

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
    missing_dependencies: list[str] = []
    required_modules = {
        "numpy": "numpy",
        "torch": "torch",
        "pydicom": "pydicom",
        "transformers": "transformers",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "PIL": "pillow",
        "joblib": "joblib",
        "sklearn": "scikit-learn",
        "jsonschema": "jsonschema",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
    }
    for module_name, package_name in required_modules.items():
        if importlib.util.find_spec(module_name) is None:
            missing_dependencies.append(package_name)
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
    if missing or missing_dependencies:
        details = []
        if missing_dependencies:
            details.append("Missing Python packages: " + ", ".join(missing_dependencies) +
                           ". Install the declared dependencies with `python -m pip install -r requirements.txt`.")
        if missing:
            details.append("Required local inference artifacts are missing:\n  " + "\n  ".join(missing))
        raise FileNotFoundError(
            "\n".join(details) +
            "\nFix: restore the exact existing helper/model/checkpoint files or correct paths in meta_schema.json. "
            "Do not retrain or edit checkpoints."
        )
    return ["Required Stage 03/04 model files and pipeline Python dependencies are present."]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-artifacts", action="store_true", help="Also verify local model helper files and trained checkpoints exist.")
    args = parser.parse_args()
    schema = load_schema(__file__)
    messages = validate_schema_contracts(schema)

    errors: list[str] = []
    for filename in SCRIPT_FILES:
        script = pipeline_script_path(filename)
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
    stage4_source = pipeline_script_path("04_malignancy_prediction.py").read_text(encoding="utf-8-sig")
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

    stage5_source = pipeline_script_path("05_gradcam.py").read_text(encoding="utf-8-sig")
    stage5_tree = ast.parse(stage5_source)
    stage5_assigned_names = {
        target.id
        for node in ast.walk(stage5_tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in ((node.targets if isinstance(node, ast.Assign) else [node.target]))
        if isinstance(target, ast.Name)
    }
    _assert("PATCH_SHAPE" in stage5_assigned_names,
            "Stage 05 uses PATCH_SHAPE but does not define it from meta_schema.json. Fix backend/Imaging/05_gradcam.py.")
    _assert('get_required(STAGE05_SCHEMA, "patch_shape_zyx", expected_type=list)' in stage5_source,
            "Stage 05 PATCH_SHAPE must be sourced from stage_05_gradcam.patch_shape_zyx in meta_schema.json.")
    _assert("if PATCH_SHAPE != STAGE03_PATCH_SHAPE:" in stage5_source,
            "Stage 05 must validate its patch shape against Stage 03's frozen model input shape.")
    _assert("MODEL_TARGET_INDEX = validate_target_contract(MODEL_TARGETS, ATTRIBUTES)" in stage5_source,
            "Stage 05 must validate requested attributes as a subset of the frozen model targets.")
    _assert("score = model.expected_ratings(outputs)[" in stage5_source,
            "Stage 05 must differentiate the selected expected-rating scalar.")
    _assert("low_resolution_cam = (weights * activation).sum(" in stage5_source,
            "Stage 05 must preserve the signed raw Grad-CAM rather than applying an attribute-specific polarity inversion.")
    _assert("if tuple(MODEL_TARGETS) != ATTRIBUTES:" not in stage5_source,
            "Stage 05 must not require exact equality between the six requested attributes and the model's eight-target output order.")
    messages.append("Stage 05 Grad-CAM patch shape is schema-driven and the six requested attributes are validated as a subset of the frozen model targets.")

    stage2_source = pipeline_script_path("02_detect_and_patch.py").read_text(encoding="utf-8-sig")
    review_check = stage2_source.index("review_attestation = validate_review_attestation(")
    detector_call = stage2_source.index("raw_detections = run_vitdet3d_raw(")
    candidate_gate = stage2_source.index(
        "accepted_candidates, rejected_candidates, candidate_filter_report = filter_detections_by_lung_mask("
    )
    accepted_patch_call = stage2_source.index("records = extract_all(")
    log_gate = stage2_source.index(
        "accepted_log_candidates, rejected_log_candidates, log_filter_report = filter_detections_by_lung_mask("
    )
    log_patch_call = stage2_source.index("log_records = extract_all(")
    _assert(review_check < detector_call, "Stage 02 must validate the current Stage 01 review before detector inference.")
    _assert(candidate_gate < accepted_patch_call, "Stage 02 must gate authoritative candidates before patch extraction.")
    _assert(log_gate < log_patch_call, "Stage 02 must gate LoG proposals before diagnostic patch extraction.")
    messages.append("Stage 02 review, authoritative-candidate, and LoG gates precede detector/patch downstream boundaries.")

    # Optional LoG-specific paths remain for audit compatibility; production inference uses the fused Stage 02 manifest.
    stage4_log_output = get_required(schema, "paths.stage_04.log_diagnostic_output_json", expected_type=str)
    stage3_log_input = get_required(schema, "stage_03_nodule_classification.outputs.log_diagnostic_predictions_json", expected_type=str)
    _assert(stage4_log_output.endswith(".json") and stage3_log_input.endswith(".json"), "Stage 04 diagnostic JSON paths must be JSON filenames.")
    stage5_batch = get_required(schema, "stage_05_gradcam.batch_processing", expected_type=dict)
    _assert(bool(stage5_batch.get("reuse_loaded_checkpoints")), "Stage 05 diagnostic batch processing must reuse loaded checkpoint models.")
    _assert("{patch_id" in str(stage5_batch.get("patch_output_directory_pattern", "")), "Stage 05 batch patch provenance pattern must include patch_id.")
    default_log_limit = get_required(schema, "stage_02_detect_and_patch.diagnostic_log_detector.patch_limit_default")
    _assert(type(default_log_limit) is int and default_log_limit > 0, "The canonical LoG patch limit must be a positive integer (use --all-log-patches for all proposals).")
    stage7 = get_required(schema, "stage_07_visualize", expected_type=dict)
    stage7_outputs = get_required(stage7, "outputs", expected_type=dict)
    for output_key in ("report_json_ref", "report_csv_ref", "excluded_outside_lung_csv_ref"):
        ref = get_required(stage7_outputs, output_key, expected_type=str)
        _assert(get_required(schema, ref, expected_type=str).endswith(".json" if output_key == "report_json_ref" else ".csv"),
                f"Stage 07 output reference {ref!r} is invalid.")
    csv_order = get_required(stage7, "csv_columns_order", expected_type=list)
    _assert(csv_order.count("{attribute_columns}") == 1, "Stage 07 CSV schema must contain exactly one canonical attribute-column placeholder.")
    canonical_attributes = get_required(schema, "attribute_contracts.stage03_ranked_attributes", expected_type=list)
    expanded_csv_columns = [name for item in csv_order for name in (canonical_attributes if item == "{attribute_columns}" else [item])]
    _assert(set(canonical_attributes).issubset(expanded_csv_columns), "Stage 07 CSV must include all canonical Stage 03 attributes.")
    _assert(len(expanded_csv_columns) == len(set(expanded_csv_columns)), "Stage 07 expanded CSV columns contain duplicates.")
    report_fields = get_required(stage7, "report_fields", expected_type=list)
    _assert(len(report_fields) == len(set(report_fields)), "Stage 07 summary JSON contains duplicate schema field names.")
    stage7_source = pipeline_script_path("07_visualize.py").read_text(encoding="utf-8-sig")
    _assert("sample_mask(lung_mask, center)" in stage7_source, "Stage 07 must directly sample lung_mask.npy before accepting/displaying centers.")
    _assert('get_required(STAGE07, "csv_columns_order", expected_type=list)' in stage7_source, "Stage 07 CSV columns must be sourced from meta_schema.json.")
    _assert("accepted_rows.append(row)" in stage7_source and "excluded_rows.append(row)" in stage7_source
            and "validate_localization_bbox(" in stage7_source
            and 'row["localization_status"] = "VERIFIED"' in stage7_source,
            "Stage 07 must separate in-lung records and block candidates without valid aligned boxes.")
    stage6_source = pipeline_script_path("06_full_ct_gradcam.py").read_text(encoding="utf-8-sig")
    _assert('filter_report.get("gate_status") != "passed"' in stage6_source,
            "Stage 06 must require a passing Stage 02 review/filter gate.")
    _assert("build_stage06_nodule_records(" in stage6_source and "if candidate_id not in accepted_ids:" in stage6_source,
            "Stage 06 must build identity-linked records only from accepted Stage 02 candidates.")
    _assert('STAGE06["record_filename"]' in stage7_source,
            "Stage 07 must load the Stage 06 identity-linked nodule record set.")
    _assert("set(prediction_lookup) != set(stage06_lookup)" in stage7_source,
            "Stage 07 must reject identity-set drift from Stage 06.")
    fusion_source = (IMAGING / "lunginsight_candidate_fusion.py").read_text(encoding="utf-8-sig")
    _assert("def select_fused_candidates(" in fusion_source and "def merge_detections(" in fusion_source,
            "Candidate fusion must deduplicate the detector union and select from the fused set.")
    merge_at = stage2_source.index("fused_candidates, fusion_report = merge_detections(")
    select_at = stage2_source.index("candidate_dicts = select_fused_candidates(fused_candidates, limit=50)")
    _assert(merge_at < select_at, "Stage 02 must select Top-50 only after detector fusion and deduplication.")
    _assert("DETECTION_FUSION_AUDIT.md" in stage2_source and "stable_nodule_id" in fusion_source,
            "Stage 02 fusion audit and stable nodule identity must be present.")
    _assert("filter_accepted_patches(" in stage5_source and "stage05_acceptance_audit.json" in stage5_source,
            "Stage 05 must block Grad-CAM inference for candidates not accepted by Stage 04.")
    _assert("create_contract_file(patient_root, args.patient_id, record_path)" in stage6_source,
            "Stage 06 must generate the validated imaging contract from its actual records.")
    _assert("generate_patient_report(" in stage7_source,
            "Stage 07 must generate static patient HTML reports.")
    contract_schema_path = IMAGING / "schemas" / "imaging_clinical_contract_v1.schema.json"
    openapi_path = IMAGING / "api" / "openapi.json"
    _assert(contract_schema_path.is_file() and openapi_path.is_file(),
            "The versioned imaging contract schema and OpenAPI document must exist.")
    openapi = json.loads(openapi_path.read_text(encoding="utf-8"))
    for api_path in (
        "/api/v1/imaging/study",
        "/api/v1/imaging/study/{study_id}",
        "/api/v1/imaging/nodule/{nodule_id}",
        "/api/v1/imaging/artifact/{artifact_id}",
    ):
        _assert(api_path in openapi.get("paths", {}), f"OpenAPI is missing endpoint {api_path}.")
    messages.append("Fused LoG + ViTDet3D candidates, accepted-only CAM inference, validated contract/API, HTML reports, and Stage 07 box/lung localization gating are wired.")
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
