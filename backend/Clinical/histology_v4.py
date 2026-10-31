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




def _histology_v4_tree_shap(
    artifact: dict[str, Any],
    features: pd.DataFrame,
    probabilities: np.ndarray,
    class_probabilities: dict[str, float],
) -> dict[str, Any]:
    """Explain V4 random-forest probabilities with tree-path-dependent TreeSHAP.

    The fitted preprocessing pipeline is applied unchanged. Encoded one-hot
    and missingness-indicator contributions are aggregated back to the
    original 19 clinical predictors.
    """
    try:
        import shap

        pipeline = artifact["model"]
        preprocess = pipeline.named_steps["preprocess"]
        classifier = pipeline.named_steps["classifier"]

        transformed = preprocess.transform(features)
        transformed_names = [str(x) for x in preprocess.get_feature_names_out()]

        if transformed.shape[1] != len(transformed_names):
            raise ValueError("Transformed V4 feature names do not match matrix width.")

        # Tree-path-dependent mode uses the fitted trees' node-cover statistics;
        # it does not invent or substitute a patient reference dataset.
        explainer = shap.TreeExplainer(
            classifier,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )
        raw_shap = explainer.shap_values(transformed, check_additivity=False)

        class_order = ["LUSC", "LUAD"]
        if list(getattr(classifier, "classes_", [])) != [0, 1]:
            raise ValueError("V4 classifier class order is not the expected [LUSC, LUAD] mapping.")

        # SHAP 0.54 can return either a class list or a 3-D ndarray.
        if isinstance(raw_shap, list):
            if len(raw_shap) != 2:
                raise ValueError("Unexpected V4 SHAP class output.")
            per_class = [
                np.asarray(raw_shap[i], dtype=float).reshape(-1)
                for i in range(2)
            ]
        else:
            arr = np.asarray(raw_shap, dtype=float)
            if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 2:
                per_class = [arr[0, :, i] for i in range(2)]
            elif arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == 2:
                per_class = [arr[0, i, :] for i in range(2)]
            elif arr.ndim == 3 and arr.shape[0] == 2 and arr.shape[1] == 1:
                per_class = [arr[i, 0, :] for i in range(2)]
            else:
                raise ValueError(f"Unsupported V4 SHAP array shape: {arr.shape}")

        expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
        if expected.size != 2:
            raise ValueError("V4 SHAP expected values do not contain both classes.")

        if any(len(v) != len(transformed_names) for v in per_class):
            raise ValueError("V4 SHAP contribution count does not match transformed features.")

        # Map every transformed column to exactly one original input feature.
        groups = {name: [] for name in artifact["feature_columns"]}
        for index, transformed_name in enumerate(transformed_names):
            name = transformed_name.split("__", 1)[-1]
            matched = []
            for feature_name in artifact["feature_columns"]:
                if (
                    name == feature_name
                    or name.startswith(feature_name + "_")
                    or name == "missingindicator_" + feature_name
                ):
                    matched.append(feature_name)

            if len(matched) != 1:
                raise ValueError(
                    f"Cannot safely map transformed SHAP feature {transformed_name!r}."
                )
            groups[matched[0]].append(index)

        contributions = []
        for feature_name in artifact["feature_columns"]:
            indices = groups[feature_name]
            if not indices:
                raise ValueError(f"No transformed SHAP columns found for {feature_name}.")

            feature_value = features.iloc[0][feature_name]
            if pd.isna(feature_value):
                feature_value = None
            else:
                feature_value = str(feature_value)

            contributions.append({
                "feature": feature_name,
                "feature_value": feature_value,
                "values": {
                    class_order[class_index]: float(
                        np.sum(per_class[class_index][indices])
                    )
                    for class_index in range(2)
                },
            })

        base_values = {
            class_order[i]: float(expected[i]) for i in range(2)
        }
        prediction_values = {
            class_order[i]: float(probabilities[i]) for i in range(2)
        }

        reconstruction_errors = {}
        for i, label in enumerate(class_order):
            reconstructed = base_values[label] + sum(
                item["values"][label] for item in contributions
            )
            reconstruction_errors[label] = abs(
                reconstructed - prediction_values[label]
            )

        max_error = max(reconstruction_errors.values())
        if not np.isfinite(max_error) or max_error > 1e-4:
            raise ValueError(
                f"V4 SHAP local-accuracy check failed (max error={max_error:.6g})."
            )

        return {
            "status": "available",
            "method": "tree_shap_tree_path_dependent",
            "output_space": "predict_proba",
            "class_order": class_order,
            "base_values": base_values,
            "prediction_values": prediction_values,
            "feature_contributions": contributions,
            "reference_population": {
                "cohort": "fitted V4 random-forest tree path statistics",
                "partition": "tree_path_dependent; no external background dataset supplied",
                "rows": None,
                "features": list(artifact["feature_columns"]),
            },
            "reconstruction_error": reconstruction_errors,
            "limitations": [
                "TreeSHAP uses the fitted random forest's tree-path statistics; it does not use a reconstructed patient-level training reference dataset.",
                "One-hot encoded and missingness-indicator contributions are aggregated to their original clinical predictors.",
                "Attributions explain model probability output, not biological causation, clinical risk, or diagnostic truth.",
                "The V4 model is research-only and has no external validation.",
            ],
        }

    except Exception as exc:
        logger.warning("histology_v4_shap_unavailable reason=%s", type(exc).__name__)
        return {
            "status": "unavailable",
            "method": "tree_shap_tree_path_dependent",
            "output_space": "predict_proba",
            "class_order": ["LUSC", "LUAD"],
            "base_values": {},
            "prediction_values": {
                "LUSC": float(class_probabilities["LUSC"]),
                "LUAD": float(class_probabilities["LUAD"]),
            },
            "feature_contributions": [],
            "reason": f"V4 TreeSHAP unavailable ({type(exc).__name__}).",
            "failure_reason": type(exc).__name__,
            "reference_population": {
                "cohort": "fitted V4 random-forest tree path statistics",
                "partition": "unavailable",
                "rows": None,
                "features": list(artifact.get("feature_columns", [])),
            },
            "limitations": [
                "No feature contributions are reported because the V4 explanation could not be verified.",
                "The V4 prediction probabilities remain unchanged.",
            ],
        }


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
            "explanation": _histology_v4_tree_shap(
                artifact, features, probabilities, values
            ),
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
