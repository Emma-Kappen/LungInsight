from pathlib import Path
from itertools import combinations
import json
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path.cwd()
CLINICAL = ROOT / "backend" / "Clinical"
INPUT = CLINICAL / "artifacts" / "cancer_type_dataset" / "cancer_type_model_ready_candidate.csv"
OUT = CLINICAL / "artifacts" / "cancer_type_sex_smoking_baseline"
SEED = 42
COHORT = "MSK-CHORD-2024"
TARGET = "cancer_type"
FEATURES = ["sex_at_baseline", "smoking_history"]
CLASSES = ["LUAD", "LUSC"]


def build_model():
    preprocessing = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), FEATURES),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )),
    ])


def metrics(model, x, y):
    pred = model.predict(x)
    classes = list(model.classes_)
    p = model.predict_proba(x)[:, classes.index("LUSC")]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "lusc_precision": float(precision_score(y, pred, pos_label="LUSC", zero_division=0)),
        "lusc_recall": float(recall_score(y, pred, pos_label="LUSC", zero_division=0)),
        "lusc_f1": float(f1_score(y, pred, pos_label="LUSC", zero_division=0)),
        "roc_auc_lusc_positive": float(roc_auc_score((y == "LUSC").astype(int), p)),
        "confusion_matrix_labels": CLASSES,
        "confusion_matrix": confusion_matrix(y, pred, labels=CLASSES).tolist(),
        "classification_report": classification_report(
            y, pred, labels=CLASSES, output_dict=True, zero_division=0
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT, dtype={"patient_id": str}, low_memory=False)
    required = ["patient_id", "cohort", TARGET, *FEATURES]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.loc[df["cohort"].eq(COHORT)].copy()
    df = df.dropna(subset=[TARGET])
    if df["patient_id"].isna().any() or df["patient_id"].duplicated().any():
        raise ValueError("Missing or duplicate patient IDs; refusing to train.")
    if set(df[TARGET].astype(str).unique()) != set(CLASSES):
        raise ValueError(f"Expected labels {CLASSES}; got {sorted(df[TARGET].astype(str).unique())}")

    df[TARGET] = df[TARGET].astype(str)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.20, random_state=SEED, stratify=y
    )

    model = build_model()
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    model.fit(X_train, y_train)

    dummy_test = metrics(dummy, X_test, y_test)
    test_metrics = metrics(model, X_test, y_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_result = cross_validate(
        build_model(),
        X_train,
        y_train,
        cv=cv,
        scoring={
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "f1_macro",
            "roc_auc": "roc_auc",
        },
        return_train_score=False,
        n_jobs=1,
        error_score="raise",
    )
    cv_metrics = {}
    for metric in ("balanced_accuracy", "macro_f1", "roc_auc"):
        vals = cv_result[f"test_{metric}"]
        cv_metrics[metric] = {
            "fold_scores": [float(v) for v in vals],
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
        }

    holdout_prob = model.predict_proba(X_test)[:, list(model.classes_).index("LUSC")]
    predictions = pd.DataFrame({
        "dataset_row_index": idx_test,
        "patient_id": df.loc[idx_test, "patient_id"].to_numpy(),
        "true_label": y_test.to_numpy(),
        "predicted_label": model.predict(X_test),
        "probability_lusc": holdout_prob,
    })
    predictions.to_csv(OUT / "holdout_predictions.csv", index=False)

    report = {
        "purpose": "Exploratory sex-and-smoking LUAD/LUSC baseline; not clinical validation.",
        "input_dataset": str(INPUT),
        "cohort": COHORT,
        "target": TARGET,
        "features": FEATURES,
        "excluded_features": [
            "stage", "patient_id", "cohort", "label_provenance",
            "sample_count", "source_record_id", "histology",
            "OncoTree code", "survival", "vital status",
            "treatment outcome", "smoking_pack_years", "ecog_score",
        ],
        "scikit_learn_version": sklearn.__version__,
        "random_seed": SEED,
        "test_fraction": 0.20,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "training_class_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "missing_feature_values": {
            f: int(df[f].isna().sum()) for f in FEATURES
        },
        "preprocessing": "Most-frequent categorical imputation and one-hot encoding fitted within pipeline.",
        "model": "LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)",
        "dummy_holdout_metrics": dummy_test,
        "model_holdout_metrics": test_metrics,
        "training_only_5fold_cv": cv_metrics,
        "comparison_reference_stage_model": {
            "holdout_roc_auc": 0.7089872524144705,
            "holdout_balanced_accuracy": 0.6371132754951712,
            "holdout_macro_f1": 0.5629899963039929,
            "training_cv_roc_auc_mean": 0.7181512403821625,
            "training_cv_balanced_accuracy_mean": 0.6570625614863177,
        },
        "deployment_decision": "candidate_only; production artifact not overwritten",
        "limitations": [
            "Single-cohort random holdout, not external validation.",
            "Smoking history is NLP-derived and may be missing.",
            "Predicted probabilities are not necessarily calibrated.",
            "Not clinically validated; not a diagnosis.",
            "This script intentionally does not change the production model or frontend contract.",
        ],
    }
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(model, OUT / "sex-smoking-baseline.joblib")

    reloaded = joblib.load(OUT / "sex-smoking-baseline.joblib")
    if list(reloaded.feature_names_in_) != FEATURES:
        raise RuntimeError(f"Saved model features mismatch: {reloaded.feature_names_in_}")
    if list(reloaded.classes_) != CLASSES:
        raise RuntimeError(f"Saved model classes mismatch: {reloaded.classes_}")
    if not np.allclose(model.predict_proba(X_test), reloaded.predict_proba(X_test), rtol=0, atol=0):
        raise RuntimeError("Reloaded candidate predictions differ from in-memory predictions.")

    print("Candidate saved to:", OUT)
    print("Production artifact was NOT changed.")
    print("\nHoldout metrics:")
    print(json.dumps({k: test_metrics[k] for k in (
        "accuracy", "balanced_accuracy", "macro_f1",
        "lusc_precision", "lusc_recall", "lusc_f1",
        "roc_auc_lusc_positive", "confusion_matrix"
    )}, indent=2))
    print("\nTraining-only 5-fold CV:")
    print(json.dumps(cv_metrics, indent=2))
    print("\nReference stage-model results:")
    print(json.dumps(report["comparison_reference_stage_model"], indent=2))
    print("\nCandidate features:", list(reloaded.feature_names_in_))
    print("Reload round-trip validation: PASS")


if __name__ == "__main__":
    main()
