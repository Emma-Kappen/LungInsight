from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator, FormatChecker

from .harmonization import ClinicalValidationError, validate_clinician_inputs
from .imaging import ImagingEvidenceError, validate_imaging_evidence

CLINICAL_DIR = Path(__file__).resolve().parent
INPUT_SCHEMA_PATH = CLINICAL_DIR / "schemas" / "clinical_inference_v1.schema.json"
OUTPUT_SCHEMA_PATH = CLINICAL_DIR / "schemas" / "clinical_prediction_v1.schema.json"


class ClinicalInferenceError(ValueError):
    """Raised when the validated Clinical inference contract cannot be satisfied."""


def _load_schema(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    errors = sorted(
        Draft202012Validator(
            _load_schema(INPUT_SCHEMA_PATH),
            format_checker=FormatChecker(),
        ).iter_errors(request),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors[:5]
        )
        raise ClinicalInferenceError(f"Clinical input validation failed: {details}")
    if not isinstance(request["patient_id"], str) or not request["patient_id"].strip():
        raise ClinicalInferenceError("patient_id must be a non-empty string.")
    try:
        clinician = validate_clinician_inputs(request["clinician_inputs"])
        imaging = validate_imaging_evidence(
            request["imaging_evidence"],
            patient_id=request["patient_id"],
        )
    except (ClinicalValidationError, ImagingEvidenceError) as exc:
        raise ClinicalInferenceError(str(exc)) from exc
    requested_method = request.get("requested_malignancy_method")
    if requested_method is not None and requested_method not in imaging["malignancy_methods"]:
        raise ClinicalInferenceError(
            f"Requested malignancy method {requested_method!r} was not selected in the supplied "
            "Imaging contract; regenerate Stage 04 with --malignancy-method and resubmit its contract."
        )
    return {"clinician": clinician, "imaging": imaging, "requested_malignancy_method": requested_method}


def _feature_frame(
    clinician: dict[str, Any],
    feature_order: list[str],
) -> pd.DataFrame:
    values = {
        "age_at_baseline_years": clinician.get("age_at_baseline_years"),
        "sex_at_baseline": clinician.get("sex_at_baseline"),
        "kps_category": clinician.get("kps_category"),
    }
    unknown = set(feature_order) - set(values)
    if unknown:
        raise ClinicalInferenceError(f"Model declares unsupported features: {sorted(unknown)}.")
    return pd.DataFrame([{name: values[name] for name in feature_order}], columns=feature_order)


def _feature_applicability(
    features: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[list[str], list[str]]:
    missing = [name for name in metadata.get("required_features", []) if pd.isna(features.iloc[0][name])]
    warnings: list[str] = []
    for name, limits in metadata.get("training_feature_ranges", {}).items():
        value = features.iloc[0].get(name)
        if value is not None and not pd.isna(value):
            if float(value) < float(limits["minimum"]) or float(value) > float(limits["maximum"]):
                warnings.append(f"{name}_outside_training_range")
    for name, categories in metadata.get("training_categories", {}).items():
        value = features.iloc[0].get(name)
        if value is not None and not pd.isna(value) and value not in categories:
            warnings.append(f"{name}_category_not_seen_in_training")
    if missing:
        warnings.append("required_model_inputs_missing")
    return missing, warnings


def _unavailable_model(status: str, reason: str) -> dict[str, Any]:
    return {"status": status, "prediction": None, "reason": reason}


def _classifier_explanation(
    estimator: Any,
    features: pd.DataFrame,
    class_order: list[str],
    probabilities: np.ndarray,
) -> dict[str, Any]:
    contributions = []
    try:
        for feature in features.columns:
            ablated = features.copy()
            ablated.loc[:, feature] = None
            ablated_probabilities = estimator.predict_proba(ablated)[0]
            contributions.append({
                "feature": feature,
                "method": "missingness_ablation",
                "contribution_to_probability": {
                    label: float(actual - ablated)
                    for label, actual, ablated in zip(
                        class_order, probabilities, ablated_probabilities
                    )
                },
                "direction": (
                    "positive values indicate the observed value raises the class probability "
                    "relative to the fitted missing-value representation"
                ),
            })
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
        return {
            "status": "unsupported",
            "method": "missingness_ablation",
            "failure_reason": type(exc).__name__,
            "feature_contributions": [],
            "limitations": [
                "SHAP is not applied to the calibrated composite estimator; this is not a causal attribution."
            ],
        }
    return {
        "status": "available",
        "method": "missingness_ablation",
        "target": "LUAD/LUSC conditional probability",
        "features_used": list(features.columns),
        "feature_contributions": contributions,
        "limitations": [
            "This is a local model ablation, not a causal explanation.",
            "SHAP is not applied to the calibrated composite estimator.",
        ],
    }


def _survival_explanation(
    estimator: Any,
    features: pd.DataFrame,
    horizons: list[int],
    base_probabilities: dict[str, float],
) -> dict[str, Any]:
    contributions = []
    try:
        for feature in features.columns:
            ablated = features.copy()
            ablated.loc[:, feature] = None
            function = estimator.predict_survival_function(ablated)[0]
            horizon_contributions = {}
            for horizon in horizons:
                if function.x[0] <= horizon <= function.x[-1]:
                    ablated_value = float(function(horizon))
                    horizon_contributions[str(horizon)] = {
                        "observed_minus_ablated": float(
                            base_probabilities[str(horizon)] - ablated_value
                        ),
                        "observed_survival_probability": base_probabilities[str(horizon)],
                        "ablated_survival_probability": ablated_value,
                    }
            contributions.append({
                "feature": feature,
                "method": "missingness_ablation",
                "horizons_days": horizon_contributions,
            })
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
        return {
            "status": "unsupported",
            "method": "missingness_ablation",
            "failure_reason": type(exc).__name__,
            "feature_contributions": [],
            "limitations": [
                "Survival explanation is unavailable for the fitted estimator/input combination."
            ],
        }
    return {
        "status": "available",
        "method": "missingness_ablation",
        "target": "survival probability at each reported horizon",
        "features_used": list(features.columns),
        "feature_contributions": contributions,
        "limitations": [
            "Ablation differences are not causal effects and do not alter censoring or endpoint definitions.",
            "This does not claim absolute survival calibration.",
        ],
    }


def _run_model_independently(
    predictor: Any,
    artifact: dict[str, Any] | None,
    clinician: dict[str, Any],
) -> dict[str, Any]:
    try:
        return predictor(artifact, clinician)
    except (ClinicalInferenceError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
        return {
            "status": "failed",
            "prediction": None,
            "error_code": "model_inference_failed",
            "reason": type(exc).__name__,
        }


def _predict_cancer_type(
    artifact: dict[str, Any] | None,
    clinician: dict[str, Any],
) -> dict[str, Any]:
    if artifact is None:
        return _unavailable_model(
            "not_trained",
            "No eligible cancer-type model artifact is available; TCGA-only subtype/source labels are not used as a classifier.",
        )
    metadata = artifact["metadata"]
    if metadata.get("prediction_baseline_definition") != clinician.get("prediction_baseline_definition"):
        return {
            "status": "abstained",
            "prediction": None,
            "missing_features": [],
            "warnings": ["prediction_baseline_does_not_match_training_baseline"],
            "class_order": metadata.get("class_order", []),
            "calibration_status": metadata.get("calibration_status", "unknown"),
        }
    order = list(metadata["feature_order"])
    features = _feature_frame(clinician, order)
    missing, warnings = _feature_applicability(features, metadata)
    if missing or warnings:
        return {
            "status": "abstained",
            "prediction": None,
            "missing_features": missing,
            "warnings": warnings,
            "class_order": metadata["class_order"],
            "calibration_status": metadata.get("calibration_status", "unknown"),
        }
    estimator = artifact["estimator"]
    probabilities = estimator.predict_proba(features)[0]
    class_order = list(metadata["class_order"])
    if len(probabilities) != len(class_order) or not np.isfinite(probabilities).all():
        raise ClinicalInferenceError("Cancer-type model returned an invalid probability vector.")
    if np.any(probabilities < 0) or not np.isclose(float(probabilities.sum()), 1.0, atol=1e-6):
        raise ClinicalInferenceError("Cancer-type model probabilities are not normalized.")
    return {
        "status": "research_prediction",
        "target": "histology conditional on the explicitly supported classes; not cancer-versus-benign",
        "probabilities": {label: float(probability) for label, probability in zip(class_order, probabilities)},
        "predicted_class": class_order[int(np.argmax(probabilities))],
        "class_order": class_order,
        "features_used": order,
        "calibration_status": metadata.get("calibration_status", "not_calibrated"),
        "applicability": "within_declared_training_feature_support",
        "model_version": metadata.get("model_version"),
        "explanation": _classifier_explanation(
            estimator,
            features,
            class_order,
            probabilities,
        ),
        "warnings": [],
    }


def _predict_survival(
    artifact: dict[str, Any] | None,
    clinician: dict[str, Any],
) -> dict[str, Any]:
    if artifact is None:
        return _unavailable_model(
            "not_trained",
            "No eligible censoring-aware survival model artifact is available.",
        )
    metadata = artifact["metadata"]
    if metadata.get("prediction_baseline_definition") != clinician.get("prediction_baseline_definition"):
        return {
            "status": "blocked",
            "prediction": None,
            "missing_features": [],
            "warnings": ["prediction_baseline_does_not_match_training_baseline"],
            "endpoint": metadata["endpoint"],
        }
    order = list(metadata["feature_order"])
    features = _feature_frame(clinician, order)
    missing, warnings = _feature_applicability(features, metadata)
    if missing or warnings:
        return {
            "status": "blocked",
            "prediction": None,
            "missing_features": missing,
            "warnings": warnings,
            "endpoint": metadata["endpoint"],
        }
    estimator = artifact["estimator"]
    functions = estimator.predict_survival_function(features)
    function = functions[0]
    supported_horizons = []
    horizon_probabilities = {}
    for horizon in metadata.get("supported_horizons_days", []):
        horizon = int(horizon)
        if function.x[0] <= horizon <= function.x[-1]:
            supported_horizons.append(horizon)
            horizon_probabilities[str(horizon)] = float(function(horizon))
    if not supported_horizons:
        return {
            "status": "blocked",
            "prediction": None,
            "missing_features": [],
            "warnings": ["no_validated_horizon_within_model_support"],
            "endpoint": metadata["endpoint"],
        }
    maximum = max(supported_horizons)
    grid = np.unique(function.x[function.x <= maximum])
    if not len(grid):
        raise ClinicalInferenceError("Survival function has no points within the validated horizon.")
    prepend_zero = grid[0] > 0
    if prepend_zero:
        grid = np.concatenate(([0.0], grid))
    values = np.asarray([
        1.0 if prepend_zero and time == 0 else float(function(time))
        for time in grid
    ], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ClinicalInferenceError("Survival model returned invalid survival probabilities.")
    if np.any(np.diff(values) > 1e-10):
        raise ClinicalInferenceError("Survival model returned a non-monotone survival function.")
    return {
        "status": "research_prediction",
        "prediction": {
            "survival_function": {
                "time_days": [float(value) for value in grid],
                "probability": [float(value) for value in values],
            },
            "probability_at_horizons_days": horizon_probabilities,
        },
        "endpoint": metadata["endpoint"],
        "time_origin": metadata["time_origin"],
        "time_unit": "days",
        "features_used": order,
        "calibration_status": "not_calibrated_for_absolute_survival",
        "uncertainty_status": "not_estimated",
        "evaluation": metadata.get("evaluation", {}),
        "model_version": metadata.get("model_version"),
        "explanation": _survival_explanation(
            estimator,
            features,
            supported_horizons,
            horizon_probabilities,
        ),
        "warnings": [
            "Research-cohort estimates are not clinically validated or calibrated for the target-care population.",
            "Source-specific and cross-cohort performance is heterogeneous; transportability is not established.",
        ],
    }


def run_inference(
    request: dict[str, Any],
    *,
    artifacts: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    normalized = validate_request(request)
    artifacts = artifacts or {}
    clinician = normalized["clinician"]
    clinician["prediction_baseline_definition"] = request["baseline_definition"]
    imaging = normalized["imaging"]
    try:
        prediction_baseline_datetime = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except (OSError, OverflowError, ValueError) as exc:
        raise ClinicalInferenceError("Could not generate the prediction baseline timestamp from the system clock.") from exc
    result = {
        "schema_version": "1.0.0",
        "patient_id": request["patient_id"],
        "study_id": imaging["study_id"],
        "prediction_baseline_datetime": prediction_baseline_datetime,
        "baseline_definition": request["baseline_definition"],
        "requested_malignancy_method": normalized["requested_malignancy_method"],
        "input_validation": {
            "status": "passed",
            "baseline_definition_status": request["baseline_definition"],
            "clinician_fields_supplied": [
                key
                for key, value in request["clinician_inputs"].items()
                if value is not None
            ],
            "performance_status": clinician["performance_status"],
        },
        "imaging_context": imaging,
        "cancer_type": _run_model_independently(
            _predict_cancer_type,
            artifacts.get("cancer_type"),
            clinician,
        ),
        "survival": _run_model_independently(
            _predict_survival,
            artifacts.get("survival"),
            clinician,
        ),
        "provenance": {
            "clinical_pipeline_version": "0.1.0",
            "data_dictionary_version": "1.0.0",
            "preprocessing_version": "1.0.0",
            "imaging_contract_version": imaging["contract_version"],
            "imaging_pipeline_version": imaging["imaging_pipeline_version"],
            "imaging_model_versions": imaging["imaging_model_versions"],
            "scan_datetime": imaging["scan_datetime"],
            "prediction_baseline_datetime": prediction_baseline_datetime,
        },
        "limitations": [
            "Research-only clinical decision support; not a replacement for pathology, clinician judgment, or established prediction instruments.",
            "The active Imaging v1 contract does not itself carry timestamps or gate results; a trusted adapter must provide the required evidence envelope.",
            "The active training cohorts do not contain aligned Imaging measurements; imaging summaries are context-only, not model predictors.",
            "No benign/cancer-negative cohort is available for cancer-presence prediction.",
        ],
    }
    errors = list(
        Draft202012Validator(
            _load_schema(OUTPUT_SCHEMA_PATH),
            format_checker=FormatChecker(),
        ).iter_errors(result)
    )
    if errors:
        raise ClinicalInferenceError("Clinical model output failed its versioned output schema.")
    return result


def load_artifacts(path: str | Path) -> dict[str, dict[str, Any] | None]:
    """Load only trusted local joblib artifacts; never accept an artifact path from an API request."""
    import joblib

    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Clinical artifact bundle is unavailable: {bundle_path}")
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != "1.0.0":
        raise ClinicalInferenceError("Unsupported Clinical artifact bundle.")
    models: dict[str, dict[str, Any] | None] = {}
    for name in ("cancer_type", "survival"):
        artifact = bundle.get(name)
        if artifact is not None and (
            not isinstance(artifact, dict)
            or "estimator" not in artifact
            or "metadata" not in artifact
        ):
            raise ClinicalInferenceError(f"Invalid {name} artifact structure.")
        models[name] = artifact
    return models