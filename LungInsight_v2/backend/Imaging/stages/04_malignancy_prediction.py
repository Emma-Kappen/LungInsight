#!/usr/bin/env python3
"""Stage 04: Brock/PanCan research-only pulmonary nodule risk estimation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

IMAGING_DIR = Path(__file__).resolve().parent
if IMAGING_DIR.name in {"stages", "inference", "visualization", "validation"}:
    IMAGING_DIR = IMAGING_DIR.parent
if str(IMAGING_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGING_DIR))

from brock import estimate_brock
from pipeline_logging import add_verbose_argument, configure_console, pipeline_print as print
from pipeline_schema import check_meta_schema_version, get_required, load_schema, validate_stage_version
from stage02_eligibility import load_validated_stage02

SUPPORTED_MALIGNANCY_METHODS = ("brock",)
BROCK_MODEL_VERSION = "published-coefficients-verified-v1"

SCHEMA = load_schema(__file__)
STAGE04_SCHEMA = get_required(SCHEMA, "stage_04_malignancy_prediction", expected_type=dict)
STAGE03_SCHEMA = get_required(SCHEMA, "stage_03_nodule_classification", expected_type=dict)
STAGE03_PATHS = get_required(SCHEMA, "paths.stage_03", expected_type=dict)
STAGE04_PATHS = get_required(SCHEMA, "paths.stage_04", expected_type=dict)
V2_ROOT = IMAGING_DIR.parent.parent
DEFAULT_OUTPUT_ROOT = V2_ROOT / get_required(SCHEMA, "paths.output_root")


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc


def unpack_stage03_payload(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, list):
        records, metadata = payload, {}
    elif isinstance(payload, dict):
        records, metadata = payload.get("predictions"), payload
        if not isinstance(records, list):
            raise ValueError("Stage 03 JSON must contain a list-valued 'predictions' field.")
        check_meta_schema_version(payload, SCHEMA, context="Stage 03 predictions", require=False)
        validate_stage_version(
            payload.get("schema_version"),
            STAGE03_SCHEMA["schema_version"],
            context="Stage 03 predictions",
        )
    else:
        raise ValueError("Stage 03 predictions JSON must be an object or list.")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Stage 03 predictions must contain only object records.")
    return records, metadata


def _invalid_brock_inputs(stable_nodule_id: str, reason: str) -> dict[str, Any]:
    return {
        "status": "invalid_input",
        "method": "brock_formula_decomposition",
        "model_name": "Brock_PanCan_full",
        "model_version": BROCK_MODEL_VERSION,
        "target": "pulmonary_nodule_malignancy_probability",
        "probability": None,
        "features_missing": [],
        "feature_contributions": [],
        "failure_reason": f"{stable_nodule_id}: {reason}",
        "limitations": ["No probability was generated from malformed Brock inputs."],
    }


def selected_prediction(brock: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "brock",
        "model_version": brock.get("model_version", BROCK_MODEL_VERSION),
        "status": brock.get("status", "error"),
        "probability": brock.get("probability"),
        "value": brock.get("probability"),
        "inputs_used": brock.get("inputs_used", []),
        "missing_inputs": brock.get("features_missing", []),
        "validation_warnings": [brock["failure_reason"]] if brock.get("failure_reason") else [],
        "explanation": brock,
        "limitations": brock.get("limitations", []),
        "research_only": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 04 Brock-only research malignancy-risk inference."
    )
    parser.add_argument(
        "patient_id",
        nargs="?",
        help="Patient ID used to locate the existing Stage 03 predictions.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Explicit Stage 03 predictions.json path; patient_id is optional when supplied.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Pipeline output root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--malignancy-method",
        choices=SUPPORTED_MALIGNANCY_METHODS,
        default="brock",
        help="Only supported malignancy-risk method: Brock/PanCan.",
    )
    parser.add_argument(
        "--brock-inputs",
        type=Path,
        default=None,
        help=(
            "JSON envelope containing patient-level inputs and a nodules object keyed "
            "by stable_nodule_id. Missing inputs produce an explicit incomplete result."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output JSON path.",
    )
    add_verbose_argument(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_console(args.verbose)
    if args.predictions is not None:
        input_path = args.predictions.resolve()
        patient_id_arg = args.patient_id
    else:
        if not args.patient_id:
            raise SystemExit("Provide patient_id or --predictions PATH")
        patient_id_arg = args.patient_id
        input_path = (
            args.output_root
            / args.patient_id
            / STAGE03_PATHS["directory"]
            / STAGE03_SCHEMA["outputs"]["predictions_json"]
        ).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Existing Stage 03 predictions.json not found: {input_path}")

    payload = read_json(input_path)
    records, metadata = unpack_stage03_payload(payload)
    if not isinstance(metadata, dict) or metadata.get("stage04_eligible") is not True:
        raise RuntimeError("Stage 04 requires a Stage 03 output explicitly marked stage04_eligible=true.")
    manifest_value = metadata.get("patch_manifest")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise RuntimeError("Stage 03 output does not identify the gated Stage 02 primary manifest.")
    manifest_path = Path(manifest_value).expanduser().resolve()
    patient_root = manifest_path.parents[2]
    stage02_manifest, accepted_patches, accepted_candidates = load_validated_stage02(patient_root, manifest_path)
    del stage02_manifest
    expected_ids = {
        (str(row["candidate_id"]), int(row["patch_id"]))
        for row in accepted_patches
    }
    declared_hash = metadata.get("patch_manifest_sha256")
    actual_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if not isinstance(declared_hash, str) or declared_hash != actual_hash:
        raise RuntimeError("Stage 03 predictions are not bound to the current Stage 02 manifest hash.")
    seen_stage03: set[tuple[str, int]] = set()
    eligible_records: list[dict[str, Any]] = []
    excluded_candidates: list[dict[str, Any]] = []
    for record in records:
        stable_id = record.get("stable_nodule_id", record.get("nodule_id"))
        try:
            identity = (str(record["candidate_id"]), int(record["patch_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Stage 03 record lacks a valid candidate_id/patch_id identity.") from exc
        if identity in seen_stage03:
            raise ValueError(f"Stage 03 contains duplicate candidate/patch identity {identity}.")
        seen_stage03.add(identity)
        if (
            identity not in expected_ids
            or stable_id != identity[0]
            or identity[0] not in accepted_candidates
            or record.get("stage04_eligible") is not True
        ):
            excluded_candidates.append({
                "candidate_id": identity[0],
                "patch_id": identity[1],
                "stable_nodule_id": stable_id,
                "reason": "not_in_current_selected_stage02_set_or_not_explicitly_eligible",
            })
            continue
        eligible_records.append(record)
    if seen_stage03 != expected_ids:
        raise RuntimeError("Stage 03 records do not exactly match the selected Stage 02 patch set.")
    records = eligible_records

    brock_payload: dict[str, Any] = {}
    if args.brock_inputs is not None:
        loaded = read_json(args.brock_inputs.resolve())
        if (
            not isinstance(loaded, dict)
            or not isinstance(loaded.get("patient"), dict)
            or not isinstance(loaded.get("nodules"), dict)
        ):
            raise ValueError(
                "Brock input envelope must contain object-valued 'patient' and 'nodules' fields."
            )
        patient_fields = {"age_years", "sex", "family_history_lung_cancer", "emphysema"}
        nodule_fields = {"nodule_size_mm", "nodule_type", "upper_lobe", "nodule_count", "spiculation"}
        unknown_patient = set(loaded["patient"]) - patient_fields
        if unknown_patient:
            raise ValueError(f"Unknown patient-level Brock input(s): {sorted(unknown_patient)}")
        for stable_id, nodule_values in loaded["nodules"].items():
            if not isinstance(stable_id, str) or not isinstance(nodule_values, dict):
                raise ValueError("Brock nodules must map stable nodule IDs to input objects.")
            unknown_nodule = set(nodule_values) - nodule_fields
            if unknown_nodule:
                raise ValueError(f"Unknown Brock inputs for {stable_id}: {sorted(unknown_nodule)}")
            if set(nodule_values) & patient_fields:
                raise ValueError(f"Patient-level Brock predictors must not be overridden per nodule: {stable_id}")
        known_ids = {str(row["candidate_id"]) for row in accepted_patches}
        unknown_ids = set(loaded["nodules"]) - known_ids
        if unknown_ids:
            raise ValueError(f"Brock input envelope contains unselected candidate IDs: {sorted(unknown_ids)}")
        brock_payload = loaded

    predictions: list[dict[str, Any]] = []
    invalid_candidates: list[dict[str, Any]] = []
    for record in records:
        stable_id = record.get("stable_nodule_id", record.get("nodule_id"))
        if not isinstance(stable_id, str) or not stable_id.strip():
            raise ValueError("Eligible Stage 03 record is missing stable_nodule_id.")
        nodule_inputs = brock_payload.get("nodules", {}).get(stable_id, {})
        if not isinstance(nodule_inputs, dict):
            brock = _invalid_brock_inputs(stable_id, "nodule Brock inputs must be an object")
        else:
            brock = estimate_brock({
                **brock_payload.get("patient", {}),
                **nodule_inputs,
            })
        predictions.append({
            "stable_nodule_id": stable_id,
            "nodule_id": stable_id,
            "candidate_id": record.get("candidate_id"),
            "patient_id": record.get("patient_id") or record.get("patient") or metadata.get("patient_id") or patient_id_arg,
            "source": record.get("source", record.get("detection_source")),
            "detection_source": record.get("detection_source", "fused_candidate_set"),
            "stage04_eligible": True,
            "model_name": "Brock_PanCan_full",
            "model_version": BROCK_MODEL_VERSION,
            "formula_version": STAGE04_SCHEMA["formula_version"],
            "model_file": None,
            "center_zyx": record.get("center_zyx"),
            "center_lps_mm": record.get("center_lps_mm"),
            "anatomical_location": record.get("anatomical_location"),
            "geometry": record.get("geometry", {}),
            "malignancy": {
                "brock": brock,
                "selected_prediction": selected_prediction(brock),
            },
            "brock": brock,
            "selected_prediction": selected_prediction(brock),
            "ct_or_image_used_by_malignancy_predictor": False,
            "input_completeness": {
                "status": brock.get("status"),
                "required_fields": [
                    "age_years", "sex", "family_history_lung_cancer", "emphysema",
                    "nodule_size_mm", "nodule_type", "upper_lobe", "nodule_count", "spiculation",
                ],
                "fields_missing": brock.get("features_missing", []),
                "source": str(args.brock_inputs.resolve()) if args.brock_inputs is not None else None,
            },
            "input_provenance": {
                "patient_fields": sorted(brock_payload.get("patient", {})),
                "nodule_fields": sorted(nodule_inputs) if isinstance(nodule_inputs, dict) else [],
                "nodule_size_unit": "mm",
                "stage03_predictions_used_as_brock_predictors": False,
                "stage02_detector_size_used_as_brock_size": False,
            },
            "risk_uncertainty": {
                "status": "not_estimated",
                "reason": "The published Brock formula output is a point risk estimate; calibrated patient-specific uncertainty is unavailable.",
            },
            "diagnosis": {"status": "not_predicted", "value": None},
            "pathology": {"status": "not_provided", "value": None},
        })

    parent_patient_id = (
        metadata.get("patient_id") if isinstance(metadata, dict) else None
    ) or patient_id_arg
    output_path = (
        args.output.resolve()
        if args.output is not None
        else (args.output_root / str(parent_patient_id) / STAGE04_PATHS["directory"] / STAGE04_PATHS["output_json"]).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": STAGE04_SCHEMA["output_schema_version"],
        "meta_schema_version": SCHEMA["schema_version"],
        "stage": "04_malignancy_prediction",
        "status": "completed" if predictions else "no_eligible_candidates",
        "patient_id": parent_patient_id,
        "malignancy_method": "brock",
        "model_name": "Brock_PanCan_full",
        "model_version": BROCK_MODEL_VERSION,
        "target_semantics": "research-only pulmonary nodule malignancy-risk probability; not pathology diagnosis",
        "input_source": f"validated Brock envelope plus Stage 03 stable nodule identifiers ({input_path})",
        "source_predictions_json": str(input_path),
        "source_predictions_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "candidate_count": len(records),
        "scored_candidate_count": len(predictions),
        "invalid_candidates": invalid_candidates,
        "excluded_candidates": excluded_candidates,
        "input_completeness": {
            "status": "provided" if args.brock_inputs is not None else "missing",
            "envelope": str(args.brock_inputs.resolve()) if args.brock_inputs is not None else None,
            "incomplete_prediction_count": sum(row["brock"].get("status") == "incomplete" for row in predictions),
        },
        "diagnosis": {"status": "not_predicted", "value": None},
        "pathology": {"status": "not_provided", "value": None},
        "stage02_manifest": str(manifest_path),
        "stage02_manifest_sha256": actual_hash,
        "research_only": True,
        "limitations": [
            "Brock is the only malignancy-risk model supported by this POC.",
            "This research estimate is not clinically validated and is not a diagnosis.",
            "Missing or invalid inputs produce no probability and are never replaced by another score.",
        ],
        "predictions": predictions,
    }
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "stage": result["stage"],
        "patient_id": parent_patient_id,
        "output": str(output_path),
        "malignancy_method": "brock",
        "scored_candidates": len(predictions),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
