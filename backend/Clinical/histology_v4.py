from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .histology_v4_features import V4_CATEGORIES, build_histology_features, validate_histology_v4_inputs

logger = logging.getLogger(__name__)
CLINICAL_DIR = Path(__file__).resolve().parent
V4_ARTIFACT_PATH = CLINICAL_DIR / "research_histology_v4" / "cancer-type-luad-lusc-research-v4.joblib"
EXPECTED_VERSION = "v4"
EXPECTED_FEATURES = [
    "age_at_diagnosis", "sex", "race", "ethnicity", "smoking_history_code",
    "pack_years", "smoking_year_started", "smoking_year_stopped",
    "dlco_percent_predicted", "fev1_fvc_post_percent", "fev1_fvc_pre_percent",
    "fev1_percent_predicted_post", "fev1_percent_predicted_pre",
    "pulmonary_function_test_performed", "ecog_preoperative",
    "karnofsky_preoperative", "lung_location", "anatomic_subdivision",
    "other_malignancy_history",
]
LIMITATIONS = [
    "Research-only model; not clinically validated and not a diagnosis.",
    "Labels were derived from TCGA-LUAD versus TCGA-LUSC cohort membership; cohort effects may be learned instead of histology-specific signals.",
    "No TCGA-linked CT attribute-ranker measurements were available; attribute ranks and detector size are display-only, not v4 predictors.",
    "Imaging-derived lung location and anatomic subdivision are populated only when Stage 02 anatomy maps unambiguously to a fitted v4 category.",
    "Smoking history codes were retained categorically and are not mapped from the app's never/former/current smoking categories.",
    "Pulmonary-function measurement timing was not established in the supplied training table.",
    "Only explicitly preoperative performance scores were used in training.",
    "Probabilities are conditional on the TCGA LUAD/LUSC research cohort and must not be interpreted as cancer risk or diagnostic certainty.",
]


class HistologyV4Error(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _load_cached(path: str, modified_ns: int) -> dict[str, Any]:
    import joblib
    import sklearn

    # The artifact was serialized with scikit-learn 1.9.1. Pickle compatibility
    # across sklearn versions is not guaranteed, so fail closed rather than score
    # with an unverified runtime.
    if sklearn.__version__ != "1.9.1":
        raise HistologyV4Error(
            f"scikit-learn version mismatch: artifact requires 1.9.1; runtime is {sklearn.__version__}."
        )
    artifact = joblib.load(path)
    required = {
        "model", "model_name", "version", "target_mapping", "feature_columns",
        "numeric_features", "categorical_features", "training_cohort",
        "training_patient_count", "holdout_patient_count", "research_only",
    }
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise HistologyV4Error("The v4 model artifact has an unsupported structure.")
    if artifact["version"] != EXPECTED_VERSION or artifact["feature_columns"] != EXPECTED_FEATURES:
        raise HistologyV4Error("The v4 model feature contract does not match the application adapter.")
    if artifact["target_mapping"] != {"LUSC": 0, "LUAD": 1}:
        raise HistologyV4Error("The v4 model target mapping is not the documented LUAD/LUSC mapping.")
    model = artifact["model"]
    if not callable(getattr(model, "predict_proba", None)):
        raise HistologyV4Error("The v4 artifact does not expose predict_proba.")
    classes = list(model.named_steps["classifier"].classes_)
    if classes != [0, 1]:
        raise HistologyV4Error("The v4 model classes do not match the documented target mapping.")
    categorical = artifact["model"].named_steps["preprocess"].named_transformers_["categorical"]
    encoder = categorical.named_steps["onehot"]
    for feature, categories in zip(artifact["categorical_features"], encoder.categories_):
        if feature in V4_CATEGORIES and set(str(value) for value in categories) != V4_CATEGORIES[feature]:
            raise HistologyV4Error(f"The fitted encoder categories for {feature} differ from the reviewed v4 contract.")
    return artifact


def load_histology_v4() -> dict[str, Any] | None:
    try:
        stat = V4_ARTIFACT_PATH.stat()
    except FileNotFoundError:
        return None
    try:
        return _load_cached(str(V4_ARTIFACT_PATH), stat.st_mtime_ns)
    except Exception as exc:
        logger.error("histology_v4_artifact_unavailable reason=%s", type(exc).__name__)
        return {"load_error": str(exc)}


def predict_histology_v4(
    artifact: dict[str, Any] | None,
    clinician: dict[str, Any],
    *,
    imaging_contract: dict[str, Any] | None = None,
    form_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = {
        "model_version": "cancer-type-luad-lusc-research-v4",
        "target": "patient-level histology conditional on LUAD/LUSC; not cancer detection or nodule malignancy",
        "training_cohort": "TCGA-LUAD and TCGA-LUSC only",
        "cohort_patient_count": 1026,
        "training_patient_count": 820,
        "holdout_patient_count": 206,
        "model_name": "random_forest",
        "evaluation_summary": {
            "holdout_roc_auc": 0.7978,
            "holdout_balanced_accuracy": 0.7282,
            "training_5fold_roc_auc_mean": 0.7525,
            "training_5fold_roc_auc_sd": 0.0102,
        },
        "limitations": LIMITATIONS,
    }
    try:
        form_values = validate_histology_v4_inputs(form_values)
    except ValueError as exc:
        return {"status": "failed", "prediction": None, "reason": str(exc), **base}
    built = build_histology_features(clinician, form_values, imaging_contract)
    preview = {
        "model_inputs_used": built["values"],
        "input_provenance": built["provenance"],
        "imaging_features_available": built["imaging_features_available"],
        "imaging_derived": built["imaging_derived"],
        "warnings": built["warnings"],
    }
    if artifact is None:
        return {"status": "not_trained", "prediction": None, "reason": "The v4 model artifact is not installed.", **base, **preview}
    if artifact.get("load_error"):
        return {"status": "unavailable", "prediction": None, "reason": artifact["load_error"], **base, **preview}
    if clinician.get("prediction_baseline_definition") != "initial_pathologic_diagnosis":
        return {
            "status": "abstained", "prediction": None,
            "reason": "Confirm that the CT and clinical inputs represent initial pathologic diagnosis before running the v4 histology model.",
            **base, **preview, "warnings": sorted(set(preview["warnings"] + ["prediction_baseline_not_confirmed"])),
        }
    try:
        raw_values = built["values"]
        features = pd.DataFrame([{name: raw_values.get(name) for name in artifact["feature_columns"]}], columns=artifact["feature_columns"])
        for name in artifact["numeric_features"]:
            features[name] = pd.to_numeric(features[name], errors="coerce").astype(float)
        missing = [name for name in artifact["feature_columns"] if raw_values.get(name) is None]
        observed = {name: raw_values[name] for name in artifact["feature_columns"] if raw_values.get(name) is not None}
        probabilities = np.asarray(artifact["model"].predict_proba(features)[0], dtype=float)
        if probabilities.shape != (2,) or not np.isfinite(probabilities).all():
            raise HistologyV4Error("The v4 model returned an invalid probability vector.")
        if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
            raise HistologyV4Error("The v4 model returned non-normalized probabilities.")
        values = {"LUSC": float(probabilities[0]), "LUAD": float(probabilities[1])}
        return {
            "status": "research_prediction",
            **base,
            "probabilities": values,
            "predicted_class": max(values, key=values.get),
            "features_used": list(observed),
            "feature_values": observed,
            "model_inputs_used": raw_values,
            "input_provenance": built["provenance"],
            "imaging_features_available": built["imaging_features_available"],
            "imaging_derived": built["imaging_derived"],
            "imputed_features": missing,
            "missing_feature_count": len(missing),
            "input_completeness": f"{len(observed)}/{len(artifact['feature_columns'])}",
            "warnings": sorted(set(built["warnings"] + [
                "Missing predictors were handled by the model's training-time imputation pipeline.",
                "TCGA cohort-membership labels may encode cohort effects; external validation is absent.",
            ])),
        }
    except Exception as exc:
        logger.exception("histology_v4_inference_failed")
        return {
            "status": "failed", "prediction": None,
            "reason": f"v4 inference failed ({type(exc).__name__}).", **base, **preview,
            "warnings": sorted(set(preview["warnings"] + ["Histology v4 prediction failed; imaging-derived summaries are retained for audit."])),
        }
