from __future__ import annotations

import json
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

CANCER_FEATURE_ORDER = [
    "age_at_baseline_years",
    "sex_at_baseline",
    "stage_at_baseline",
]
SURVIVAL_FEATURE_ORDER = list(CANCER_FEATURE_ORDER)
DEFAULT_HORIZONS_DAYS = [365, 1096, 1826]
MODEL_VERSION = "clinical-research-v1"


class ModelEligibilityError(ValueError):
    """Raised when a cohort or target cannot support the requested model."""


def _grouped_three_way_split(
    frame: pd.DataFrame,
    target: pd.Series,
    group_col: str,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if group_col not in frame or frame[group_col].isna().any():
        raise ModelEligibilityError("Every observation requires a patient grouping key.")
    if frame[group_col].duplicated().any():
        raise ModelEligibilityError("Training input must contain one resolved row per patient.")
    if target.isna().any() or target.nunique() < 2:
        raise ModelEligibilityError("At least two non-missing target classes are required.")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    labels = target.astype(str).to_numpy()
    groups = frame[group_col].astype(str).to_numpy()
    try:
        splits = list(splitter.split(frame, labels, groups))
    except ValueError as exc:
        raise ModelEligibilityError("Patient-level data are insufficient for grouped validation.") from exc
    test_indices = splits[0][1]
    development_indices = splits[0][0]
    development = frame.iloc[development_indices]
    development_labels = labels[development_indices]
    development_groups = groups[development_indices]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state + 1)
    try:
        train_relative, validation_relative = next(
            inner.split(development, development_labels, development_groups)
        )
    except ValueError as exc:
        raise ModelEligibilityError("Patient-level train/calibration/validation split is infeasible.") from exc
    train_indices = development_indices[train_relative]
    validation_indices = development_indices[validation_relative]
    for name, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        present = set(labels[indices])
        if present != set(labels):
            raise ModelEligibilityError(f"The patient-grouped {name} split omits a target class.")
    sets = [set(groups[index] for index in indices) for indices in (train_indices, validation_indices, test_indices)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("Patient leakage detected across a grouped split.")
    return train_indices, validation_indices, test_indices


def source_target_shortcut_report(
    frame: pd.DataFrame,
    target_col: str,
    source_col: str,
) -> dict[str, Any]:
    if target_col not in frame or source_col not in frame:
        raise ModelEligibilityError("Target and source fields are required for source shortcut analysis.")
    table = pd.crosstab(frame[source_col].fillna("<missing>"), frame[target_col].fillna("<missing>"))
    source_class_counts = table.to_dict(orient="index")
    sources_with_one_target = int((table.gt(0).sum(axis=1) == 1).sum())
    return {
        "source_class_counts": source_class_counts,
        "number_of_sources": int(len(table)),
        "sources_with_only_one_observed_class": sources_with_one_target,
        "perfect_source_target_shortcut": bool(
            len(table) > 1 and sources_with_one_target == len(table)
        ),
    }


def _clinical_preprocessor(
    feature_order: list[str],
) -> ColumnTransformer:
    numeric = [name for name in feature_order if name == "age_at_baseline_years"]
    categorical = [name for name in feature_order if name != "age_at_baseline_years"]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def _classification_metrics(
    estimator: Any,
    features: pd.DataFrame,
    labels: pd.Series,
    classes: list[str],
) -> dict[str, Any]:
    probabilities = estimator.predict_proba(features)
    prediction = estimator.predict(features)
    one_hot = np.column_stack([labels.to_numpy() == label for label in classes]).astype(float)
    return {
        "n_patients": int(len(labels)),
        "class_counts": {label: int((labels == label).sum()) for label in classes},
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, labels=classes, average="macro", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def fit_cancer_type_model(
    frame: pd.DataFrame,
    *,
    target_col: str = "histology",
    group_col: str = "patient_id",
    source_col: str = "source",
    feature_order: list[str] | None = None,
    prediction_baseline_definition: str | None = None,
    use_authorized: bool = False,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a calibrated, patient-grouped LUAD/LUSC classifier on authorized research data."""
    if not use_authorized:
        raise ModelEligibilityError("Training requires explicit confirmation of source data-use permission.")
    if prediction_baseline_definition != "initial_pathologic_diagnosis":
        raise ModelEligibilityError(
            "Cancer-type training requires a verified, declared prediction baseline definition."
        )
    feature_order = list(feature_order or CANCER_FEATURE_ORDER)
    if target_col not in frame or not set(feature_order).issubset(frame.columns):
        raise ModelEligibilityError("Cancer-type target or required features are unavailable.")
    eligible = frame.loc[frame[target_col].isin(["LUAD", "LUSC"])].copy().reset_index(drop=True)
    if eligible.empty or eligible[target_col].nunique() != 2:
        raise ModelEligibilityError("The classifier requires verified LUAD and LUSC target labels.")
    if source_col not in eligible:
        raise ModelEligibilityError("A source label is required to audit cohort-target shortcuts.")
    shortcut = source_target_shortcut_report(eligible, target_col, source_col)
    if shortcut and shortcut["perfect_source_target_shortcut"]:
        raise ModelEligibilityError(
            "Histology is perfectly confounded with cohort source; do not train a source-shortcut classifier."
        )
    train_idx, calibration_idx, test_idx = _grouped_three_way_split(
        eligible,
        eligible[target_col],
        group_col,
        random_state=random_state,
    )
    x = eligible[feature_order]
    y = eligible[target_col].astype(str)
    classes = sorted(y.unique())
    base = Pipeline(
        [
            ("preprocessor", _clinical_preprocessor(feature_order)),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state,
            )),
        ]
    )
    base.fit(x.iloc[train_idx], y.iloc[train_idx])
    if set(y.iloc[calibration_idx]) != set(classes):
        raise ModelEligibilityError("Natural-prevalence calibration data must contain every class.")
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(base),
        method="sigmoid",
    )
    calibrated.fit(x.iloc[calibration_idx], y.iloc[calibration_idx])
    test_metrics = _classification_metrics(
        calibrated,
        x.iloc[test_idx],
        y.iloc[test_idx],
        classes,
    )
    numeric_ranges = {}
    if "age_at_baseline_years" in feature_order:
        age = pd.to_numeric(x.iloc[train_idx]["age_at_baseline_years"], errors="coerce").dropna()
        if not age.empty:
            numeric_ranges["age_at_baseline_years"] = {
                "minimum": float(age.min()),
                "maximum": float(age.max()),
            }
    categories = {}
    for name in feature_order:
        if name != "age_at_baseline_years":
            categories[name] = sorted(
                str(value) for value in x.iloc[train_idx][name].dropna().unique()
            )
    evaluation = {
        "validation_design": "patient-grouped train/calibration/test; test prevalence unmodified",
        "train_n": int(len(train_idx)),
        "calibration_n": int(len(calibration_idx)),
        "test": test_metrics,
        "class_counts_all_patients": {label: int((y == label).sum()) for label in classes},
        "source_target_shortcut": shortcut,
        "imbalance_strategy": "class_weight=balanced fitted on training data only",
        "calibration": "sigmoid calibration fitted on separate naturally distributed patient groups",
        "test_patient_ids_used_for_fitting": False,
    }
    artifact = {
        "estimator": calibrated,
        "metadata": {
            "model_version": MODEL_VERSION,
            "software_versions": _software_versions(),
            "model_type": "calibrated_regularized_logistic_regression",
            "target": "histology conditional on LUAD/LUSC cancer patients",
            "prediction_baseline_definition": prediction_baseline_definition,
            "class_order": classes,
            "feature_order": feature_order,
            "required_features": list(feature_order),
            "training_feature_ranges": numeric_ranges,
            "training_categories": categories,
            "calibration_status": "sigmoid_calibrated_on_patient_grouped_natural_prevalence_validation",
            "evaluation": evaluation,
            "research_only": True,
        },
    }
    return artifact, evaluation


def _survival_preprocessor() -> ColumnTransformer:
    return _clinical_preprocessor(SURVIVAL_FEATURE_ORDER)


def _c_index(target: np.ndarray, risk: np.ndarray) -> float:
    return float(
        concordance_index_censored(
            target["event"],
            target["time"],
            np.asarray(risk, dtype=float),
        )[0]
    )


def _supported_horizons(target: np.ndarray, minimum_at_risk: int = 30) -> list[int]:
    horizons = []
    for horizon in DEFAULT_HORIZONS_DAYS:
        at_risk = int(np.sum(target["time"] >= horizon))
        if at_risk >= minimum_at_risk:
            horizons.append(horizon)
    return horizons


def _survival_metrics(
    estimator: Any,
    train_target: np.ndarray,
    test_target: np.ndarray,
    x_test: pd.DataFrame,
    horizons: list[int],
) -> dict[str, Any]:
    risk = np.asarray(estimator.predict(x_test), dtype=float)
    result: dict[str, Any] = {
        "n_patients": int(len(test_target)),
        "events": int(test_target["event"].sum()),
        "censored": int((~test_target["event"]).sum()),
        "concordance_index": _c_index(test_target, risk),
        "time_dependent_auc": {},
        "integrated_brier_score": None,
        "integrated_brier_status": "not_computed",
    }
    test_max = float(np.max(test_target["time"]))
    train_max = float(np.max(train_target["time"]))
    metric_horizons = [
        horizon for horizon in horizons
        if float(np.min(test_target["time"])) < horizon < min(test_max, train_max)
    ]
    if metric_horizons:
        try:
            auc, mean_auc = cumulative_dynamic_auc(
                train_target,
                test_target,
                risk,
                np.asarray(metric_horizons, dtype=float),
            )
            result["time_dependent_auc"] = {
                str(horizon): float(value)
                for horizon, value in zip(metric_horizons, auc)
            }
            result["mean_time_dependent_auc"] = float(mean_auc)
        except ValueError as exc:
            result["time_dependent_auc_status"] = f"not_estimable: {exc}"
        try:
            survival_functions = estimator.predict_survival_function(x_test)
            estimates = np.vstack([
                [float(function(time)) for time in metric_horizons]
                for function in survival_functions
            ])
            result["integrated_brier_score"] = float(
                integrated_brier_score(
                    train_target,
                    test_target,
                    estimates,
                    np.asarray(metric_horizons, dtype=float),
                )
            )
            result["integrated_brier_status"] = "computed_with_inverse_probability_of_censoring_weights"
        except ValueError as exc:
            result["integrated_brier_status"] = f"not_estimable: {exc}"
    else:
        result["time_dependent_auc_status"] = "no_predefined_horizon_supported_by_train_and_test_followup"
    return result


def fit_survival_model(
    frame: pd.DataFrame,
    *,
    group_col: str = "patient_id",
    time_col: str = "overall_survival_time_days",
    event_col: str = "overall_survival_event",
    use_authorized: bool = False,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train and evaluate a right-censoring-aware RSF using TCGA-only patient rows."""
    if not use_authorized:
        raise ModelEligibilityError("Training requires explicit confirmation of source data-use permission.")
    required = {
        group_col,
        time_col,
        event_col,
        "source",
        "endpoint_issue",
        *SURVIVAL_FEATURE_ORDER,
    }
    if not required.issubset(frame.columns):
        raise ModelEligibilityError(f"Survival training is missing fields: {sorted(required - set(frame.columns))}.")
    if frame["source"].isna().any():
        raise ModelEligibilityError("Every survival-training record requires an explicit cohort source.")
    sources = set(frame["source"].astype(str))
    if not sources or not sources.issubset({"TCGA-LUAD", "TCGA-LUSC"}):
        raise ModelEligibilityError(
            "This survival target accepts TCGA-LUAD/LUSC only; MSK-CHORD endpoints are not harmonized."
        )
    eligible = frame.loc[
        frame[time_col].notna()
        & frame[event_col].notna()
        & frame["endpoint_issue"].isna()
    ].copy().reset_index(drop=True)
    if eligible.empty or eligible[group_col].duplicated().any():
        raise ModelEligibilityError("Survival training requires resolved, unique patient outcomes.")
    if not eligible[event_col].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ModelEligibilityError("Survival event values must be explicit booleans, not inferred encodings.")
    times = pd.to_numeric(eligible[time_col], errors="coerce").to_numpy(dtype=float)
    event = eligible[event_col].astype(bool).to_numpy()
    if not np.isfinite(times).all() or np.any(times < 0) or event.sum() < 2 or (~event).sum() < 2:
        raise ModelEligibilityError("Survival target needs valid nonnegative times, events, and censored observations.")
    target = Surv.from_arrays(event=event, time=times)
    stratification = pd.Series(
        [f"{source}:{int(is_event)}" for source, is_event in zip(
            eligible["source"].astype(str), event
        )],
        index=eligible.index,
    )
    train_idx, validation_idx, test_idx = _grouped_three_way_split(
        eligible,
        stratification,
        group_col,
        random_state=random_state,
    )
    x = eligible[SURVIVAL_FEATURE_ORDER]
    train_target, validation_target, test_target = (
        target[train_idx],
        target[validation_idx],
        target[test_idx],
    )
    model = Pipeline(
        [
            ("preprocessor", _survival_preprocessor()),
            ("survival_forest", RandomSurvivalForest(
                n_estimators=200,
                min_samples_leaf=15,
                max_features="sqrt",
                n_jobs=1,
                random_state=random_state,
            )),
        ]
    )
    model.fit(x.iloc[train_idx], train_target)
    validation_risk = np.asarray(model.predict(x.iloc[validation_idx]), dtype=float)
    validation_c_index = _c_index(validation_target, validation_risk)
    validation_baseline_c_index = _c_index(
        validation_target,
        np.zeros(len(validation_target), dtype=float),
    )
    selected_indices = np.concatenate((train_idx, validation_idx))
    final_target = target[selected_indices]
    final_features = x.iloc[selected_indices]
    model.fit(final_features, final_target)
    evaluation = _survival_metrics(
        model,
        final_target,
        test_target,
        x.iloc[test_idx],
        _supported_horizons(final_target),
    )
    evaluation.update({
        "validation_design": "patient-grouped train/validation/test; fixed RSF configuration; preprocessing fitted on train+validation before final test evaluation",
        "development_patients": int(len(selected_indices)),
        "test_patients": int(len(test_idx)),
        "validation_concordance_index": validation_c_index,
        "validation_kaplan_meier_tie_baseline_concordance_index": validation_baseline_c_index,
        "test_patient_ids_used_for_fitting_or_selection": False,
        "model": "RandomSurvivalForest with 200 trees and minimum leaf size 15",
        "proportional_hazards_assumption": "not_required_by_selected_random_survival_forest",
        "censoring": "right censoring retained; time-dependent AUC/Brier use censoring-aware scikit-survival metrics",
        "source_performance": {},
    })
    for source in sorted(eligible["source"].astype(str).unique()):
        source_mask = eligible.iloc[test_idx]["source"].astype(str).to_numpy() == source
        source_target = test_target[source_mask]
        if len(source_target) >= 10 and source_target["event"].sum() >= 2:
            evaluation["source_performance"][source] = {
                "n_patients": int(len(source_target)),
                "events": int(source_target["event"].sum()),
                "concordance_index": _c_index(
                    source_target,
                    np.asarray(model.predict(x.iloc[test_idx].loc[source_mask]), dtype=float),
                ),
            }
        else:
            evaluation["source_performance"][source] = {
                "n_patients": int(len(source_target)),
                "status": "too_few_held_out_events_for_stable_source_specific_metric",
            }
    horizons = _supported_horizons(final_target)
    numeric_ranges = {}
    age = pd.to_numeric(final_features["age_at_baseline_years"], errors="coerce").dropna()
    if not age.empty:
        numeric_ranges["age_at_baseline_years"] = {
            "minimum": float(age.min()),
            "maximum": float(age.max()),
        }
    categories = {
        name: sorted(str(value) for value in final_features[name].dropna().unique())
        for name in ("sex_at_baseline", "stage_at_baseline")
    }
    metadata = {
        "model_version": MODEL_VERSION,
        "preprocessing_version": "1.0.0",
        "random_state": random_state,
        "software_versions": _software_versions(),
        "model_type": "random_survival_forest",
        "endpoint": "overall survival; death from any cause",
        "time_origin": "initial pathologic diagnosis",
        "prediction_baseline_definition": "initial_pathologic_diagnosis",
        "time_unit": "days",
        "event_definition": "Dead with valid nonnegative days_to_death; Alive is right censored at days_to_last_followup or, if absent, days_to_last_known_alive",
        "feature_order": SURVIVAL_FEATURE_ORDER,
        "required_features": ["age_at_baseline_years"],
        "training_feature_ranges": numeric_ranges,
        "training_categories": categories,
        "supported_horizons_days": horizons,
        "evaluation": evaluation,
        "research_only": True,
        "population_scope": "TCGA-LUAD/TCGA-LUSC clinical cohort; external transportability not established",
    }
    report = {
        "eligible_patients": int(len(eligible)),
        "excluded_endpoint_records": int(len(frame) - len(eligible)),
        "patient_counts_by_cohort": {
            source: int((eligible["source"] == source).sum())
            for source in sorted(eligible["source"].unique())
        },
        "source_outcome_counts": {
            source: {
                "patients": int((eligible["source"] == source).sum()),
                "events": int((
                    eligible.loc[eligible["source"] == source, event_col].astype(bool)
                ).sum()),
                "censored": int((
                    ~eligible.loc[eligible["source"] == source, event_col].astype(bool)
                ).sum()),
            }
            for source in sorted(eligible["source"].unique())
        },
        "events": int(event.sum()),
        "censored": int((~event).sum()),
        "followup_days_quantiles": {
            str(q): float(value)
            for q, value in eligible[time_col].astype(float).quantile([0.25, 0.5, 0.75, 0.9]).items()
        },
        "evaluation": evaluation,
        "supported_horizons_days": horizons,
    }
    return {"estimator": model, "metadata": metadata}, report


def evaluate_cross_cohort_survival(
    frame: pd.DataFrame,
    *,
    training_source: str,
    test_source: str,
) -> dict[str, Any]:
    """Fit on one cohort and evaluate the other without refitting on test patients."""
    if "source" not in frame:
        raise ModelEligibilityError("Cross-cohort survival evaluation requires source labels.")
    if training_source == test_source or not {
        training_source,
        test_source,
    }.issubset({"TCGA-LUAD", "TCGA-LUSC"}):
        raise ModelEligibilityError("Cross-cohort survival evaluation supports distinct TCGA cohorts only.")
    required = {
        "patient_id",
        "source",
        "overall_survival_time_days",
        "overall_survival_event",
        "endpoint_issue",
        *SURVIVAL_FEATURE_ORDER,
    }
    if not required.issubset(frame.columns):
        raise ModelEligibilityError("Cross-cohort survival evaluation is missing required fields.")
    eligible = frame.loc[
        frame["endpoint_issue"].isna()
        & frame["overall_survival_time_days"].notna()
        & frame["overall_survival_event"].notna()
    ].copy()
    training = eligible.loc[eligible["source"] == training_source].copy()
    testing = eligible.loc[eligible["source"] == test_source].copy()
    if training.empty or testing.empty:
        raise ModelEligibilityError("Two distinct, non-empty cohorts are required.")
    if training["patient_id"].duplicated().any() or testing["patient_id"].duplicated().any():
        raise ModelEligibilityError("Cross-cohort evaluation requires unique patient-level rows.")
    overlap = set(training["patient_id"]) & set(testing["patient_id"])
    if overlap:
        raise ModelEligibilityError("Patient overlap detected between the source cohorts.")
    for label, records in (("training", training), ("test", testing)):
        if not records["overall_survival_event"].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ModelEligibilityError(f"{label} event values must be explicit booleans.")
        durations = pd.to_numeric(
            records["overall_survival_time_days"],
            errors="coerce",
        ).to_numpy(dtype=float)
        if not np.isfinite(durations).all() or np.any(durations < 0):
            raise ModelEligibilityError(f"{label} survival times must be finite and nonnegative.")
    train_target = Surv.from_arrays(
        event=training["overall_survival_event"].astype(bool).to_numpy(),
        time=training["overall_survival_time_days"].astype(float).to_numpy(),
    )
    test_target = Surv.from_arrays(
        event=testing["overall_survival_event"].astype(bool).to_numpy(),
        time=testing["overall_survival_time_days"].astype(float).to_numpy(),
    )
    model = Pipeline(
        [
            ("preprocessor", _survival_preprocessor()),
            ("survival_forest", RandomSurvivalForest(
                n_estimators=200,
                min_samples_leaf=15,
                max_features="sqrt",
                n_jobs=1,
                random_state=42,
            )),
        ]
    )
    x_train = training[SURVIVAL_FEATURE_ORDER]
    x_test = testing[SURVIVAL_FEATURE_ORDER]
    model.fit(x_train, train_target)
    metrics = _survival_metrics(
        model,
        train_target,
        test_target,
        x_test,
        _supported_horizons(train_target),
    )
    return {
        "training_source": training_source,
        "test_source": test_source,
        "training_patients": int(len(training)),
        "test_patients": int(len(testing)),
        "test_events": int(test_target["event"].sum()),
        "test_censored": int((~test_target["event"]).sum()),
        "test_patient_ids_used_for_training_or_preprocessing": False,
        "metrics": metrics,
        "status": "cross-cohort_transport_diagnostic_not_external_clinical_validation",
    }


def save_artifact_bundle(
    output_path: str | Path,
    *,
    cancer_type: dict[str, Any] | None = None,
    survival: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    import joblib

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": "1.0.0",
        "cancer_type": cancer_type,
        "survival": survival,
        "provenance": provenance or {},
    }
    joblib.dump(bundle, path)
    return path


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return destination


def _software_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "scikit-survival", "joblib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions