from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator, FormatChecker

from .harmonization import ClinicalValidationError, validate_clinician_inputs
from .imaging import ImagingEvidenceError, validate_imaging_evidence

logger = logging.getLogger(__name__)
CLINICAL_DIR = Path(__file__).resolve().parent
INPUT_SCHEMA_PATH = CLINICAL_DIR / "schemas" / "clinical_inference_v1.schema.json"
OUTPUT_SCHEMA_PATH = CLINICAL_DIR / "schemas" / "clinical_prediction_v1.schema.json"
CANCER_TYPE_V2_PATH = CLINICAL_DIR / "models" / "cancer-type-luad-lusc-v2.joblib"
CANCER_TYPE_V2_ADAPTER = "luad_lusc_v2_clinical_inputs"
CANCER_TYPE_V2_TARGET = (
    "histology conditional on LUAD/LUSC (patient-level); not cancer-versus-benign, "
    "not a diagnosis, and not an individual nodule's malignancy"
)
CANCER_TYPE_V2_FEATURES = ["sex_at_baseline", "smoking_history"]
CANCER_TYPE_V2_CLASSES = ["LUAD", "LUSC"]
CANCER_TYPE_V2_CATEGORIES = {
    "sex_at_baseline": ["female", "male"],
    "smoking_history": ["former_or_current", "never"],
}
CANCER_TYPE_V2_LIMITATIONS = [
    "Research-only; not clinically validated.",
    "Predicts histology only among patients already known to have LUAD or LUSC; it does not detect cancer, "
    "distinguish cancer from benign disease, or establish a diagnosis.",
    "Trained on a retrospective research cohort of patients at initial pathologic diagnosis; "
    "this deployed model uses sex and smoking history only.",
    "Probabilities are not calibrated risks; the higher probability is not a diagnosis.",
]


@lru_cache(maxsize=1)
def _cancer_type_shap_background() -> pd.DataFrame:
    """Rebuild the exact training partition used by the active two-feature model."""
    from sklearn.model_selection import train_test_split

    dataset_path = (
        CLINICAL_DIR / "artifacts" / "cancer_type_dataset"
        / "cancer_type_model_ready_candidate.csv"
    )
    if not dataset_path.is_file():
        raise FileNotFoundError("Histology SHAP reference dataset is unavailable.")

    df = pd.read_csv(dataset_path, dtype={"patient_id": str}, low_memory=False)
    required = {"patient_id", "cohort", "cancer_type", *CANCER_TYPE_V2_FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Histology SHAP reference dataset is missing columns: {missing}")

    df = df.loc[df["cohort"].eq("MSK-CHORD-2024")].copy()
    df = df.dropna(subset=["cancer_type"])
    if df["patient_id"].isna().any() or df["patient_id"].duplicated().any():
        raise ValueError("Histology SHAP reference cohort has missing or duplicate patient IDs.")
    if set(df["cancer_type"].astype(str).unique()) != set(CANCER_TYPE_V2_CLASSES):
        raise ValueError("Histology SHAP reference cohort does not match the supported classes.")

    X = df[CANCER_TYPE_V2_FEATURES].copy()
    y = df["cancer_type"].astype(str)
    _, _, _, _, train_indices, _ = train_test_split(
        X, y, df.index, test_size=0.20, random_state=42, stratify=y
    )
    background = df.loc[train_indices, CANCER_TYPE_V2_FEATURES].reset_index(drop=True)
    if len(background) != 5423:
        raise ValueError(
            "Histology SHAP reference split differs from the documented 5,423-row training partition."
        )
    return background


def _cancer_type_shap_explanation(
    estimator: Any,
    features: pd.DataFrame,
    class_order: list[str],
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Exact two-feature interventional Shapley values in predict_proba space.

    For two inputs, enumerate all feature coalitions analytically rather than
    using a sampled approximation. This is the SHAP/Shapley definition; it does
    not rely on the SHAP package or change the underlying prediction.
    """
    try:
        if list(features.columns) != CANCER_TYPE_V2_FEATURES:
            raise ValueError("Exact histology SHAP currently supports the active two-feature model only.")

        background = _cancer_type_shap_background()
        n = len(background)
        if n == 0:
            raise ValueError("Histology SHAP reference population is empty.")

        base = np.asarray(estimator.predict_proba(background), dtype=float).mean(axis=0)
        feature_a, feature_b = CANCER_TYPE_V2_FEATURES
        value_a = features.iloc[0][feature_a]
        value_b = features.iloc[0][feature_b]

        masked_a = background.copy()
        masked_a.loc[:, feature_a] = value_a
        coalition_a = np.asarray(estimator.predict_proba(masked_a), dtype=float).mean(axis=0)

        masked_b = background.copy()
        masked_b.loc[:, feature_b] = value_b
        coalition_b = np.asarray(estimator.predict_proba(masked_b), dtype=float).mean(axis=0)

        final = np.asarray(probabilities, dtype=float)
        phi_a = 0.5 * ((coalition_a - base) + (final - coalition_b))
        phi_b = 0.5 * ((coalition_b - base) + (final - coalition_a))

        # Fail closed if numerical/model behavior does not satisfy local accuracy.
        if not np.allclose(base + phi_a + phi_b, final, rtol=0, atol=1e-6):
            raise ValueError("Histology SHAP values failed the local-accuracy check.")

        return {
            "status": "available",
            "method": "exact_interventional_shapley",
            "output_space": "predict_proba",
            "class_order": list(class_order),
            "base_values": {label: float(v) for label, v in zip(class_order, base)},
            "prediction_values": {label: float(v) for label, v in zip(class_order, final)},
            "feature_contributions": [
                {
                    "feature": feature_a,
                    "feature_value": None if pd.isna(value_a) else str(value_a),
                    "values": {label: float(v) for label, v in zip(class_order, phi_a)},
                },
                {
                    "feature": feature_b,
                    "feature_value": None if pd.isna(value_b) else str(value_b),
                    "values": {label: float(v) for label, v in zip(class_order, phi_b)},
                },
            ],
            "reference_population": {
                "cohort": "MSK-CHORD-2024",
                "partition": "stratified training partition; test_size=0.20; random_state=42",
                "rows": int(n),
                "features": list(CANCER_TYPE_V2_FEATURES),
            },
            "limitations": [
                "Exact Shapley attribution for the model's predicted probabilities using an interventional training-reference background.",
                "Values explain model output, not biological causation, clinical risk, or diagnostic truth.",
                "The reference population is the reconstructed training partition; explanations are unavailable if that reference data cannot be verified.",
            ],
        }
    except Exception as exc:
        logger.warning("histology_shap_unavailable reason=%s", type(exc).__name__)
        return {
            "status": "unavailable",
            "method": "exact_interventional_shapley",
            "output_space": "predict_proba",
            "class_order": list(class_order),
            "base_values": {},
            "prediction_values": {
                label: float(v) for label, v in zip(class_order, probabilities)
            },
            "failure_reason": type(exc).__name__,
            "feature_contributions": [],
            "reference_population": {
                "cohort": "MSK-CHORD-2024",
                "partition": "unavailable; SHAP reference could not be verified",
                "rows": 0,
                "features": list(CANCER_TYPE_V2_FEATURES),
            },
            "limitations": [
                "SHAP explanation is unavailable because its reference population or calculation could not be verified.",
                "Prediction values are the existing model probabilities; no SHAP base values or feature contributions are claimed.",
            ],
        }


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


def _cancer_type_v2_result_base(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": CANCER_TYPE_V2_TARGET,
        "class_order": list(metadata["class_order"]),
        "calibration_status": metadata.get("calibration_status", "not_calibrated"),
        "model_version": metadata.get("model_version"),
        "limitations": list(CANCER_TYPE_V2_LIMITATIONS),
    }


def _cancer_type_v2_abstention(
    metadata: dict[str, Any],
    reason: str,
    *,
    missing_features: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": "abstained",
        "prediction": None,
        "reason": reason,
        "missing_features": missing_features or [],
        "warnings": warnings or [],
        "features_used": [],
        **_cancer_type_v2_result_base(metadata),
    }


def _cancer_type_v2_features(
    clinician: dict[str, Any],
    feature_order: list[str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Map only the loaded model's required inputs; unmappable values are reported missing."""

    def mapped(value: Any, mapping: dict[str, str]) -> str | None:
        return mapping.get(value) if isinstance(value, str) else None

    candidates = {
        "sex_at_baseline": mapped(clinician.get("sex_at_baseline"), {"female": "female", "male": "male"}),
        "smoking_history": mapped(
            clinician.get("smoking_status"),
            {"never": "never", "former": "former_or_current", "current": "former_or_current"},
        ),
        "stage": mapped(
            clinician.get("stage_at_baseline"),
            {"stage_1_3": "stage_1_3", "stage_4": "stage_4"},
        ),
    }
    input_names = {
        "sex_at_baseline": "sex_at_baseline",
        "smoking_history": "smoking_status",
        "stage": "stage_at_baseline",
    }
    required = feature_order if feature_order is not None else CANCER_TYPE_V2_FEATURES
    unknown = set(required) - set(candidates)
    if unknown:
        raise ClinicalInferenceError(f"Model declares unsupported cancer-type features: {sorted(unknown)}.")
    values = {name: candidates[name] for name in required if candidates[name] is not None}
    missing = [input_names[name] for name in required if candidates[name] is None]
    return values, missing


def _predict_cancer_type_v2(
    artifact: dict[str, Any],
    clinician: dict[str, Any],
) -> dict[str, Any]:
    metadata = artifact["metadata"]
    if metadata.get("prediction_baseline_definition") != clinician.get("prediction_baseline_definition"):
        return _cancer_type_v2_abstention(
            metadata,
            "This model was trained on patients at initial pathologic diagnosis; the baseline for this "
            "prediction is not confirmed as initial pathologic diagnosis.",
            warnings=["prediction_baseline_does_not_match_training_baseline"],
        )
    order = list(metadata["feature_order"])
    values, missing = _cancer_type_v2_features(clinician, order)
    if missing:
        return _cancer_type_v2_abstention(
            metadata,
            "Required inputs are missing, unknown, or outside the categories this model was trained on: "
            + ", ".join(missing) + ".",
            missing_features=missing,
            warnings=["required_model_inputs_missing"],
        )
    features = pd.DataFrame([{name: values[name] for name in order}], columns=order)
    estimator = artifact["estimator"]
    probabilities = np.asarray(estimator.predict_proba(features)[0], dtype=float)
    class_order = [str(label) for label in getattr(estimator, "classes_", metadata["class_order"])]
    if len(probabilities) != len(class_order) or not np.isfinite(probabilities).all():
        raise ClinicalInferenceError("Cancer-type model returned an invalid probability vector.")
    if (
        np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(float(probabilities.sum()), 1.0, atol=1e-6)
    ):
        raise ClinicalInferenceError("Cancer-type model probabilities are not normalized.")
    return {
        "status": "research_prediction",
        **_cancer_type_v2_result_base(metadata),
        "class_order": class_order,
        "probabilities": {label: float(p) for label, p in zip(class_order, probabilities)},
        "predicted_class": class_order[int(np.argmax(probabilities))],
        "features_used": order,
        "feature_values": values,
        "explanation": _cancer_type_shap_explanation(
            estimator, features, class_order, probabilities
        ),
        "applicability": "within_declared_training_categories",
        "warnings": [
            "Research-only model output; not clinically validated and not a diagnosis.",
        ],
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
    if artifact.get("load_error"):
        return {
            "status": "failed",
            "prediction": None,
            "error_code": "model_artifact_invalid",
            "reason": artifact["load_error"],
        }
    metadata = artifact["metadata"]
    if metadata.get("feature_adapter") == CANCER_TYPE_V2_ADAPTER:
        return _predict_cancer_type_v2(artifact, clinician)
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
            "The Imaging v2 contract carries study provenance and processing-gate status, but clinical outcome prediction still requires an independently validated baseline and outcome evidence envelope.",
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


DEFAULT_ARTIFACT_PATH = CLINICAL_DIR / "models" / "clinical-research-v1.joblib"


def configured_artifact_path() -> Path | None:
    """Trusted local bundle: LUNGINSIGHT_CLINICAL_ARTIFACT, else the deployed default bundle if present."""
    configured = os.environ.get("LUNGINSIGHT_CLINICAL_ARTIFACT")
    if configured:
        return Path(configured)
    return DEFAULT_ARTIFACT_PATH if DEFAULT_ARTIFACT_PATH.is_file() else None


@lru_cache(maxsize=1)
def _cached_artifacts(path: str, modified_ns: int) -> dict[str, dict[str, Any] | None]:
    return load_artifacts(path)


def adapt_cancer_type_pipeline(pipeline: Any) -> dict[str, Any]:
    """Wrap the standalone LUAD/LUSC sklearn Pipeline in the bundle-style {metadata, estimator} contract.

    The pipeline's own feature names, classes and trained categories are verified against the documented
    contract; nothing is inferred or widened.
    """
    try:
        preprocessing = pipeline.named_steps["preprocessing"]
        encoder = preprocessing.named_transformers_["categorical"].named_steps["onehot"]
        columns = [str(name) for name in preprocessing.transformers_[0][2]]
        classes = [str(label) for label in pipeline.named_steps["classifier"].classes_]
        categories = {
            name: sorted(str(value) for value in values)
            for name, values in zip(columns, encoder.categories_)
        }
        predict_proba = pipeline.predict_proba
    except (AttributeError, KeyError, IndexError, TypeError) as exc:
        raise ClinicalInferenceError("Unsupported standalone cancer-type pipeline structure.") from exc
    supported_contracts = {
        tuple(CANCER_TYPE_V2_FEATURES): CANCER_TYPE_V2_CATEGORIES,
        ("sex_at_baseline", "smoking_history", "stage"): {
            **CANCER_TYPE_V2_CATEGORIES,
            "stage": ["stage_1_3", "stage_4"],
        },
    }
    expected_categories = supported_contracts.get(tuple(columns))
    if (
        expected_categories is None
        or classes != CANCER_TYPE_V2_CLASSES
        or categories != expected_categories
        or not callable(predict_proba)
    ):
        raise ClinicalInferenceError("Standalone cancer-type pipeline does not match a supported documented contract.")
    return {
        "estimator": pipeline,
        "metadata": {
            "model_version": "cancer-type-luad-lusc-v2",
            "model_type": "logistic_regression_class_weight_balanced",
            "artifact_source": "standalone_pipeline",
            "feature_adapter": CANCER_TYPE_V2_ADAPTER,
            "target": CANCER_TYPE_V2_TARGET,
            "prediction_baseline_definition": "initial_pathologic_diagnosis",
            "class_order": classes,
            "feature_order": columns,
            "required_features": columns,
            "training_categories": categories,
            "calibration_status": "not_calibrated",
        },
    }


def load_cancer_type_pipeline(path: str | Path | None = None) -> dict[str, Any] | None:
    """Load the trusted standalone LUAD/LUSC pipeline; None when absent, ClinicalInferenceError when invalid."""
    import joblib

    model_path = Path(path) if path else CANCER_TYPE_V2_PATH
    if not model_path.is_file():
        return None
    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:  # corrupt pickle data raises many unrelated exception types
        raise ClinicalInferenceError(
            f"Standalone cancer-type artifact could not be loaded ({type(exc).__name__})."
        ) from exc
    return adapt_cancer_type_pipeline(pipeline)


@lru_cache(maxsize=1)
def _cached_standalone_cancer_type(path: str, modified_ns: int) -> dict[str, Any] | None:
    try:
        return load_cancer_type_pipeline(path)
    except ClinicalInferenceError as exc:
        # An invalid model is reported as a failed model, without disturbing the survival model.
        logger.error("cancer_type_artifact_invalid path=%s reason=%s", path, exc)
        return {"load_error": str(exc)}


def _standalone_cancer_type() -> dict[str, Any] | None:
    path = CANCER_TYPE_V2_PATH
    try:
        modified_ns = path.stat().st_mtime_ns
    except OSError:
        return None
    return _cached_standalone_cancer_type(str(path), modified_ns)


def with_standalone_cancer_type(
    models: dict[str, dict[str, Any] | None] | None,
    *,
    bundle_configured: bool,
) -> dict[str, dict[str, Any] | None] | None:
    """Use v2 as the sole cancer-type model; preserve the bundle's survival artifact.

    The legacy bundled cancer-type estimator is never used as a fallback. If v2 is
    missing, cancer-type inference is unavailable; if invalid, its load error is reported.
    ``bundle_configured`` remains in the signature for compatibility with existing callers.
    """
    merged = dict(models) if models else {"cancer_type": None, "survival": None}
    merged["cancer_type"] = _standalone_cancer_type()
    return merged if any(value is not None for value in merged.values()) else None


def load_deployed_artifacts() -> dict[str, dict[str, Any] | None] | None:
    path = configured_artifact_path()
    models = _cached_artifacts(str(path), path.stat().st_mtime_ns) if path else None
    return with_standalone_cancer_type(
        models,
        bundle_configured=bool(os.environ.get("LUNGINSIGHT_CLINICAL_ARTIFACT")),
    )


def describe_artifacts(artifacts: dict[str, dict[str, Any] | None] | None) -> dict[str, Any]:
    available = artifacts or {}
    cancer_type = available.get("cancer_type")
    survival = available.get("survival")
    if cancer_type is None:
        cancer_type_status = "not_trained"
    else:
        cancer_type_status = "invalid" if cancer_type.get("load_error") else "loaded"
    description: dict[str, Any] = {
        "cancer_type_model": cancer_type_status,
        "survival_model": "loaded" if survival else "not_trained",
    }
    if cancer_type_status == "loaded":
        metadata = cancer_type["metadata"]
        description["cancer_type_model_version"] = metadata.get("model_version")
        description["cancer_type_model_source"] = metadata.get("artifact_source", "configured_bundle")
    return description
