import os
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = os.environ["LUNGINSIGHT_PROJECT_ROOT"]
OUTPUT_DIR = os.environ["LUNGINSIGHT_V4_OUTPUT"]

DATA_ROOT = os.path.join(
    PROJECT_ROOT, "datasets", "prepared", "tcga"
)

CLINICAL_PATH = os.path.join(
    DATA_ROOT, "tcga_msk_tabular_candidates.csv"
)

LABELS_PATH = os.path.join(
    DATA_ROOT, "tcga_patient_labels.csv"
)

EXPECTED_LUAD = 522
EXPECTED_LUSC = 504
EXPECTED_TOTAL = 1026

RANDOM_STATE = 42


def stop(message):
    raise RuntimeError(message)


def normalize_id(series):
    return series.astype("string").str.strip()


def normalize_category(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if not value:
        return np.nan

    token = value.casefold()

    missing_tokens = {
        "[not available]",
        "[not evaluated]",
        "[unknown]",
        "not available",
        "not evaluated",
        "unknown",
        "na",
        "n/a",
        "nan",
        "none",
        "null"
    }

    if token in missing_tokens:
        return np.nan

    return value


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def bounded_numeric(series, lower, upper):

    values = numeric(series)

    return values.where(
        values.between(lower, upper)
    )


def clean_category(series):

    return series.map(
        normalize_category
    ).astype("object")


def write_json(path, data):

    with open(path, "w", encoding="utf-8") as handle:

        json.dump(
            data,
            handle,
            indent=2,
            default=str,
            allow_nan=False
        )


os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# A. LOAD AND VALIDATE SOURCE DATA
# ============================================================

clinical = pd.read_csv(
    CLINICAL_PATH,
    dtype={"patient_id": "string"}
)

labels = pd.read_csv(
    LABELS_PATH,
    dtype={"patient_id": "string"}
)

required_clinical = {
    "patient_id",
    "target",
    "birth_days_to",
    "gender",
    "race",
    "ethnicity",
    "tobacco_smoking_history_indicator",
    "tobacco_smoking_pack_years_smoked",
    "tobacco_smoking_year_started",
    "tobacco_smoking_year_stopped",
    "ecog_score",
    "karnofsky_score",
    "performance_status_timing",
    "carbon_monoxide_diffusion_dlco",
    "fev1_fvc_ratio_postbroncholiator",
    "fev1_fvc_ratio_prebroncholiator",
    "fev1_percent_ref_postbroncholiator",
    "fev1_percent_ref_prebroncholiator",
    "pulmonary_function_test_indicator",
    "location_lung_parenchyma",
    "anatomic_organ_subdivision",
    "history_other_malignancy"
}

required_labels = {
    "patient_id",
    "target",
    "cohort"
}

missing_clinical = sorted(
    required_clinical - set(clinical.columns)
)

missing_labels = sorted(
    required_labels - set(labels.columns)
)

if missing_clinical:
    stop(
        f"Clinical CSV missing columns: {missing_clinical}"
    )

if missing_labels:
    stop(
        f"Label CSV missing columns: {missing_labels}"
    )

clinical["patient_id"] = normalize_id(
    clinical["patient_id"]
)

labels["patient_id"] = normalize_id(
    labels["patient_id"]
)

# ============================================================
# B. REMOVE ONLY VERIFIED HEADER-AS-DATA RECORDS
# ============================================================

clinical_header_mask = (
    clinical["patient_id"]
    .str.casefold()
    .eq("bcr_patient_barcode")
)

label_header_mask = (
    labels["patient_id"]
    .str.casefold()
    .eq("bcr_patient_barcode")
)

clinical_header_count = int(
    clinical_header_mask.sum()
)

label_header_count = int(
    label_header_mask.sum()
)

if clinical_header_count != 2:
    stop(
        "Expected exactly two clinical header records. "
        f"Found {clinical_header_count}. "
        "No rows removed."
    )

if label_header_count != 2:
    stop(
        "Expected exactly two label header records. "
        f"Found {label_header_count}. "
        "No rows removed."
    )

clinical = clinical.loc[
    ~clinical_header_mask
].copy()

labels = labels.loc[
    ~label_header_mask
].copy()

clinical["target"] = (
    clinical["target"]
    .astype("string")
    .str.strip()
    .str.upper()
)

labels["target"] = (
    labels["target"]
    .astype("string")
    .str.strip()
    .str.upper()
)

if set(
    clinical["target"].dropna().unique()
) != {"LUAD", "LUSC"}:

    stop(
        "Unexpected clinical target values: "
        f"{clinical['target'].value_counts(dropna=False).to_dict()}"
    )

if set(
    labels["target"].dropna().unique()
) != {"LUAD", "LUSC"}:

    stop(
        "Unexpected label target values: "
        f"{labels['target'].value_counts(dropna=False).to_dict()}"
    )

if clinical["patient_id"].isna().any():
    stop("Missing clinical patient IDs remain.")

if labels["patient_id"].isna().any():
    stop("Missing label patient IDs remain.")

# Verify uniqueness independently.
if clinical["patient_id"].duplicated().any():

    duplicates = clinical.loc[
        clinical["patient_id"].duplicated(False),
        "patient_id"
    ].tolist()

    stop(
        f"Duplicate clinical patient IDs remain: {duplicates[:20]}"
    )

if labels["patient_id"].duplicated().any():

    duplicates = labels.loc[
        labels["patient_id"].duplicated(False),
        "patient_id"
    ].tolist()

    stop(
        f"Duplicate label patient IDs remain: {duplicates[:20]}"
    )

clinical_counts = (
    clinical["target"].value_counts().to_dict()
)

label_counts = (
    labels["target"].value_counts().to_dict()
)

if clinical_counts != {
    "LUAD": EXPECTED_LUAD,
    "LUSC": EXPECTED_LUSC
}:

    stop(
        f"Unexpected clinical class counts: {clinical_counts}"
    )

if label_counts != {
    "LUAD": EXPECTED_LUAD,
    "LUSC": EXPECTED_LUSC
}:

    stop(
        f"Unexpected label class counts: {label_counts}"
    )

if len(clinical) != EXPECTED_TOTAL:
    stop(
        f"Expected {EXPECTED_TOTAL} clinical rows; "
        f"found {len(clinical)}"
    )

if len(labels) != EXPECTED_TOTAL:
    stop(
        f"Expected {EXPECTED_TOTAL} label rows; "
        f"found {len(labels)}"
    )

# ============================================================
# C. VERIFY PATIENT-LEVEL LABEL AGREEMENT
# ============================================================

label_check = clinical[
    ["patient_id", "target"]
].merge(
    labels[
        ["patient_id", "target"]
    ].rename(
        columns={"target": "label_target"}
    ),
    on="patient_id",
    how="outer",
    validate="one_to_one",
    indicator=True
)

if not (
    label_check["_merge"] == "both"
).all():

    stop(
        "Clinical and label patient ID sets do not match."
    )

if not (
    label_check["target"] ==
    label_check["label_target"]
).all():

    bad = label_check.loc[
        label_check["target"] !=
        label_check["label_target"]
    ].head(20)

    stop(
        "Clinical and label targets disagree:\n"
        + bad.to_string(index=False)
    )

data = clinical.merge(
    labels[["patient_id", "cohort"]],
    on="patient_id",
    how="inner",
    validate="one_to_one"
)

if (
    len(data) != EXPECTED_TOTAL
    or data["patient_id"].nunique() != EXPECTED_TOTAL
):
    stop(
        "Final merged data does not contain exactly "
        "1,026 unique patients."
    )

print("\nDATA VALIDATION PASSED")
print("----------------------")
print("Clinical header records removed:", clinical_header_count)
print("Label header records removed   :", label_header_count)
print("Unique patients                :", data["patient_id"].nunique())
print("LUAD                            :", clinical_counts["LUAD"])
print("LUSC                            :", clinical_counts["LUSC"])

# ============================================================
# D. FEATURE ENGINEERING
# ============================================================

X = pd.DataFrame(index=data.index)
audit = {}

# ------------------------------------------------------------
# Age at diagnosis
# ------------------------------------------------------------

birth_days = numeric(
    data["birth_days_to"]
)

age = birth_days.abs() / 365.25

X["age_at_diagnosis"] = age.where(
    age.between(18, 100)
)

audit["age_at_diagnosis"] = {
    "source": "birth_days_to",
    "transformation": "Absolute days divided by 365.25",
    "valid_range": "18-100 years"
}

# ------------------------------------------------------------
# Demographics
# ------------------------------------------------------------

X["sex"] = clean_category(
    data["gender"]
)

X["race"] = clean_category(
    data["race"]
)

X["ethnicity"] = clean_category(
    data["ethnicity"]
)

# ------------------------------------------------------------
# Smoking
# ------------------------------------------------------------

X["smoking_history_code"] = clean_category(
    data["tobacco_smoking_history_indicator"]
)

X["pack_years"] = bounded_numeric(
    data["tobacco_smoking_pack_years_smoked"],
    0,
    200
)

X["smoking_year_started"] = bounded_numeric(
    data["tobacco_smoking_year_started"],
    1900,
    2026
)

X["smoking_year_stopped"] = bounded_numeric(
    data["tobacco_smoking_year_stopped"],
    1900,
    2026
)

started = X["smoking_year_started"]
stopped = X["smoking_year_stopped"]

invalid_year_order = (
    started.notna()
    & stopped.notna()
    & (stopped < started)
)

X.loc[
    invalid_year_order,
    "smoking_year_stopped"
] = np.nan

audit["smoking_history_code"] = {
    "note": (
        "Original TCGA codes retained categorically. "
        "No unverified mapping to never/former/current."
    )
}

# ------------------------------------------------------------
# Pulmonary function
# ------------------------------------------------------------

X["dlco_percent_predicted"] = bounded_numeric(
    data["carbon_monoxide_diffusion_dlco"],
    0,
    200
)

X["fev1_fvc_post_percent"] = bounded_numeric(
    data["fev1_fvc_ratio_postbroncholiator"],
    0,
    150
)

X["fev1_fvc_pre_percent"] = bounded_numeric(
    data["fev1_fvc_ratio_prebroncholiator"],
    0,
    150
)

X["fev1_percent_predicted_post"] = bounded_numeric(
    data["fev1_percent_ref_postbroncholiator"],
    0,
    200
)

X["fev1_percent_predicted_pre"] = bounded_numeric(
    data["fev1_percent_ref_prebroncholiator"],
    0,
    200
)

X["pulmonary_function_test_performed"] = clean_category(
    data["pulmonary_function_test_indicator"]
)

audit["pulmonary_function"] = {
    "dlco": "0-200 percent predicted; outside range set to missing",
    "fev1_fvc": "0-150 percent; outside range set to missing",
    "fev1_percent_predicted": "0-200 percent; outside range set to missing",
    "timing": (
        "Measurement timing was not established in the supplied table. "
        "Interpret cautiously."
    )
}

# ------------------------------------------------------------
# Performance status
# ------------------------------------------------------------

timing = clean_category(
    data["performance_status_timing"]
)

preoperative = (
    timing.astype("string")
    .str.strip()
    .str.casefold()
    .eq("preoperative")
)

ecog = bounded_numeric(
    data["ecog_score"],
    0,
    5
)

kps = bounded_numeric(
    data["karnofsky_score"],
    0,
    100
)

X["ecog_preoperative"] = ecog.where(
    preoperative
)

X["karnofsky_preoperative"] = kps.where(
    preoperative
)

audit["performance_status"] = {
    "eligibility_rule": (
        "ECOG and Karnofsky retained only for "
        "explicitly Preoperative records."
    ),
    "preoperative_rows": int(preoperative.sum()),
    "excluded_other_or_missing_timing_rows": int(
        (~preoperative).sum()
    )
}

# ------------------------------------------------------------
# Tumour location and history
# ------------------------------------------------------------

X["lung_location"] = clean_category(
    data["location_lung_parenchyma"]
)

X["anatomic_subdivision"] = clean_category(
    data["anatomic_organ_subdivision"]
)

X["other_malignancy_history"] = clean_category(
    data["history_other_malignancy"]
)

# Excluded intentionally:
#
# - source/cohort
# - submitted_tumor_site
# - tumor_tissue_site
# - ICD10
# - pathological stage/T/N/M
# - treatment and treatment outcomes
# - survival/follow-up/vital status
# - mutation testing/results
# - collection metadata
# - form completion date

if X.columns.duplicated().any():
    stop("Duplicate feature names found.")

all_missing_features = X.columns[
    X.isna().all(axis=0)
].tolist()

if all_missing_features:
    stop(
        "Entirely missing predictors found: "
        f"{all_missing_features}"
    )

# ============================================================
# E. EXPORT PREPARED DATA AND MISSINGNESS AUDIT
# ============================================================

prepared = X.copy()

prepared.insert(
    0,
    "patient_id",
    data["patient_id"].values
)

prepared["target"] = data["target"].values

prepared.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "prepared_v4_dataset.csv"
    ),
    index=False
)

missingness = []

for col in X.columns:

    missingness.append({
        "feature": col,
        "type": (
            "numeric"
            if pd.api.types.is_numeric_dtype(X[col])
            else "categorical"
        ),
        "missing_count": int(X[col].isna().sum()),
        "missing_percent": round(
            float(X[col].isna().mean() * 100),
            2
        ),
        "unique_nonmissing": int(
            X[col].nunique(dropna=True)
        )
    })

pd.DataFrame(missingness).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_missingness_audit.csv"
    ),
    index=False
)

# ============================================================
# F. MODEL PIPELINES
# ============================================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

numeric_lr = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median",
            add_indicator=True
        )
    ),
    (
        "scaler",
        StandardScaler(with_mean=False)
    )
])

numeric_rf = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median",
            add_indicator=True
        )
    )
])

categorical_pipe = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="__MISSING__"
        )
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


def make_preprocessor(numeric_pipe):

    return ColumnTransformer([
        (
            "numeric",
            clone(numeric_pipe),
            numeric_features
        ),
        (
            "categorical",
            clone(categorical_pipe),
            categorical_features
        )
    ], remainder="drop")


models = {
    "logistic_regression": Pipeline([
        (
            "preprocess",
            make_preprocessor(numeric_lr)
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE
            )
        )
    ]),

    "random_forest": Pipeline([
        (
            "preprocess",
            make_preprocessor(numeric_rf)
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=3,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        )
    ])
}

# ============================================================
# G. PATIENT-LEVEL TRAIN / HOLDOUT SPLIT
# ============================================================

X_train, X_test, y_train, y_test, id_train, id_test = (
    train_test_split(
        X,
        data["target"].map({
            "LUSC": 0,
            "LUAD": 1
        }).astype(int),
        data["patient_id"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data["target"]
    )
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

# ============================================================
# H. TRAIN AND EVALUATE
# ============================================================

results = {}
fitted_models = {}

for name, model in models.items():

    print(f"\nTraining {name} ...")

    cv_scores = cross_val_score(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1
    )

    model.fit(
        X_train,
        y_train
    )

    fitted_models[name] = model

    probability = model.predict_proba(
        X_test
    )[:, 1]

    prediction = (
        probability >= 0.5
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        prediction,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity_lusc = (
        float(tn / (tn + fp))
        if (tn + fp)
        else None
    )

    sensitivity_luad = (
        float(tp / (tp + fn))
        if (tp + fn)
        else None
    )

    result = {
        "model": name,
        "holdout_patient_count": int(len(y_test)),
        "holdout_roc_auc": float(
            roc_auc_score(y_test, probability)
        ),
        "holdout_average_precision": float(
            average_precision_score(y_test, probability)
        ),
        "accuracy": float(
            accuracy_score(y_test, prediction)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, prediction)
        ),
        "precision_luad": float(
            precision_score(
                y_test,
                prediction,
                zero_division=0
            )
        ),
        "sensitivity_luad": sensitivity_luad,
        "specificity_lusc": specificity_lusc,
        "f1_luad": float(
            f1_score(
                y_test,
                prediction,
                zero_division=0
            )
        ),
        "confusion_matrix_labels": [
            "LUSC",
            "LUAD"
        ],
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test,
            prediction,
            labels=[0, 1],
            target_names=["LUSC", "LUAD"],
            output_dict=True,
            zero_division=0
        ),
        "training_5fold_roc_auc_mean": float(
            np.mean(cv_scores)
        ),
        "training_5fold_roc_auc_std": float(
            np.std(cv_scores)
        ),
        "training_5fold_roc_auc_scores": [
            float(v) for v in cv_scores
        ]
    }

    results[name] = result

    predictions = pd.DataFrame({
        "patient_id": id_test.values,
        "actual": np.where(
            y_test.values == 1,
            "LUAD",
            "LUSC"
        ),
        "predicted": np.where(
            prediction == 1,
            "LUAD",
            "LUSC"
        ),
        "probability_luad": probability
    })

    predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{name}_holdout_predictions.csv"
        ),
        index=False
    )

    print(
        f"  Holdout ROC-AUC: "
        f"{result['holdout_roc_auc']:.4f}"
    )

    print(
        f"  Balanced accuracy: "
        f"{result['balanced_accuracy']:.4f}"
    )

    print(
        f"  LUAD sensitivity: "
        f"{result['sensitivity_luad']:.4f}"
    )

    print(
        f"  LUSC specificity: "
        f"{result['specificity_lusc']:.4f}"
    )

    print(
        f"  Training 5-fold ROC-AUC: "
        f"{result['training_5fold_roc_auc_mean']:.4f} "
        f"+/- {result['training_5fold_roc_auc_std']:.4f}"
    )

# ============================================================
# I. SELECT MODEL USING TRAINING-ONLY CV
# ============================================================

selected_name = max(
    results,
    key=lambda key: (
        results[key][
            "training_5fold_roc_auc_mean"
        ]
    )
)

selected_model = fitted_models[
    selected_name
]

artifact = {
    "model": selected_model,
    "model_name": selected_name,
    "version": "v4",
    "target_mapping": {
        "LUSC": 0,
        "LUAD": 1
    },
    "feature_columns": X.columns.tolist(),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "training_cohort": (
        "TCGA-LUAD and TCGA-LUSC only"
    ),
    "training_patient_count": int(
        len(X_train)
    ),
    "holdout_patient_count": int(
        len(X_test)
    ),
    "random_state": RANDOM_STATE,
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "research_only": True,
    "limitations": [
        "Labels are derived from TCGA-LUAD versus TCGA-LUSC cohort membership.",
        "Cohort effects may be learned instead of histology-specific signals.",
        "No TCGA-linked CT attribute-ranker measurements were available.",
        "Smoking history codes were retained categorically; their semantics were not mapped.",
        "Pulmonary function measurement timing was not established in the supplied table.",
        "Performance scores were retained only for explicitly Preoperative records.",
        "This artifact is not clinically validated and must not be used for clinical decisions."
    ]
}

artifact_path = os.path.join(
    OUTPUT_DIR,
    "cancer-type-luad-lusc-research-v4.joblib"
)

joblib.dump(
    artifact,
    artifact_path
)

# ============================================================
# J. SAVE METRICS AND AUDIT OUTPUTS
# ============================================================

write_json(
    os.path.join(
        OUTPUT_DIR,
        "model_metrics.json"
    ),
    results
)

write_json(
    os.path.join(
        OUTPUT_DIR,
        "data_quality_audit.json"
    ),
    {
        "clinical_header_records_removed": clinical_header_count,
        "label_header_records_removed": label_header_count,
        "clinical_rows_after_filter": int(len(clinical)),
        "label_rows_after_filter": int(len(labels)),
        "unique_patient_ids": int(
            data["patient_id"].nunique()
        ),
        "clinical_target_counts": clinical_counts,
        "label_target_counts": label_counts,
        "patient_ids_match": True,
        "targets_match": True,
        "duplicate_patient_ids_after_filter": 0,
        "feature_count": int(X.shape[1]),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "feature_audit": audit,
        "selected_model": selected_name,
        "selection_rule": (
            "Highest mean 5-fold ROC-AUC "
            "on training partition only."
        )
    }
)

# ============================================================
# K. MODEL CARD
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "MODEL_CARD.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "LungInsight Histology Research Model v4\n"
    )

    f.write("=" * 52 + "\n\n")

    f.write(
        "Purpose: Research classification of TCGA LUAD vs TCGA LUSC.\n"
    )

    f.write(
        f"Selected model: {selected_name}\n"
    )

    f.write(
        "Patients: 1,026 unique TCGA patients.\n"
    )

    f.write(
        "Features: clinical/demographic tabular predictors; race included.\n"
    )

    f.write(
        "CT-derived ranker features: unavailable in current TCGA-linked table.\n"
    )

    f.write(
        "LIDC data: not used.\n"
    )

    f.write(
        "MSK-CHORD data: not used.\n\n"
    )

    f.write("CRITICAL SCIENTIFIC LIMITATION\n")

    f.write(
        "Labels are derived from TCGA-LUAD versus TCGA-LUSC "
        "cohort membership.\n"
    )

    f.write(
        "Model performance may therefore reflect cohort-specific "
        "data collection and missingness patterns rather than "
        "histology-specific signals.\n"
    )

    f.write(
        "This model is research-only and is not clinically validated.\n"
    )

    f.write(
        "Do not use for diagnosis, screening, treatment, "
        "or patient management.\n\n"
    )

    f.write(
        "Metrics: model_metrics.json\n"
    )

    f.write(
        "Data audit: data_quality_audit.json\n"
    )

print("\n" + "=" * 60)
print("HISTOLOGY V4 TRAINING COMPLETE")
print("=" * 60)

print("Selected model:", selected_name)
print("Artifact:", artifact_path)
print("Output directory:", OUTPUT_DIR)

print("Prepared data: prepared_v4_dataset.csv")
print("Metrics: model_metrics.json")
print("Missingness: feature_missingness_audit.csv")
print("Data audit: data_quality_audit.json")
print("Model card: MODEL_CARD.txt")
