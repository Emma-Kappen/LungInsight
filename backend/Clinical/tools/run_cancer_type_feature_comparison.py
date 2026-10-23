from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "cancer_type_dataset" / "cancer_type_model_ready_candidate.csv"
OUT = ROOT / "artifacts" / "cancer_type_model_comparison_20261019"
SEED = 42
TARGET = "cancer_type"
COHORT = "MSK-CHORD-2024"
CLASSES = ["LUAD", "LUSC"]

FEATURE_SETS = {
    "A_sex_only": ["sex_at_baseline"],
    "B_sex_smoking": ["sex_at_baseline", "smoking_history"],
    "C_sex_smoking_stage": ["sex_at_baseline", "smoking_history", "stage"],
}

def make_model(features):
    prep = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), features),
    ])
    return Pipeline([
        ("preprocessing", prep),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED
        )),
    ])

def evaluate(model, X, y):
    pred = model.predict(X)
    classes = list(model.classes_)
    p_lusc = model.predict_proba(X)[:, classes.index("LUSC")]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "lusc_precision": float(precision_score(y, pred, pos_label="LUSC", zero_division=0)),
        "lusc_recall": float(recall_score(y, pred, pos_label="LUSC", zero_division=0)),
        "lusc_f1": float(f1_score(y, pred, pos_label="LUSC", zero_division=0)),
        "roc_auc_lusc_positive": float(roc_auc_score((y == "LUSC").astype(int), p_lusc)),
        "confusion_matrix_labels": CLASSES,
        "confusion_matrix": confusion_matrix(y, pred, labels=CLASSES).tolist(),
        "classification_report": classification_report(
            y, pred, labels=CLASSES, output_dict=True, zero_division=0
        ),
    }

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA, dtype={"patient_id": str})
    required = {"patient_id", "cohort", TARGET, *FEATURE_SETS["C_sex_smoking_stage"]}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.loc[df["cohort"].eq(COHORT)].dropna(subset=[TARGET]).copy()
    if df["patient_id"].isna().any() or df["patient_id"].duplicated().any():
        raise ValueError("Missing or duplicate patient IDs: refusing to split.")
    if set(df[TARGET].unique()) != set(CLASSES):
        raise ValueError(f"Unexpected target classes: {sorted(df[TARGET].unique())}")

    # One fixed split shared by all feature sets.
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.20, random_state=SEED, stratify=df[TARGET]
    )
    y_train = df.loc[train_idx, TARGET].astype(str)
    y_test = df.loc[test_idx, TARGET].astype(str)

    results = {
        "purpose": "Exploratory feature-set comparison; not clinically validated.",
        "input_dataset": str(DATA),
        "cohort": COHORT,
        "rows": len(df),
        "unique_patients": int(df["patient_id"].nunique()),
        "random_seed": SEED,
        "test_fraction": 0.20,
        "training_rows": len(train_idx),
        "test_rows": len(test_idx),
        "split": "One fixed stratified patient-level split shared across feature sets.",
        "preprocessing": "Most-frequent imputation and one-hot encoding fitted within each training pipeline/fold.",
        "feature_sets": {},
        "limitations": [
            "Single-cohort random holdout; not external validation.",
            "Sex-only is a minimal demographic baseline, not expected to be clinically sufficient.",
            "Smoking-history timing is not documented.",
            "Highest-recorded stage may be post-diagnostic and is comparator-only.",
            "Results do not establish clinical utility.",
        ],
        "scikit_learn_version": sklearn.__version__,
    }

    summary = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, features in FEATURE_SETS.items():
        model = make_model(features)
        X_train = df.loc[train_idx, features]
        X_test = df.loc[test_idx, features]

        model.fit(X_train, y_train)
        holdout = evaluate(model, X_test, y_test)

        cv_raw = cross_validate(
            make_model(features), X_train, y_train,
            cv=cv,
            scoring={
                "balanced_accuracy": "balanced_accuracy",
                "macro_f1": "f1_macro",
                "roc_auc": "roc_auc",
            },
            n_jobs=1, error_score="raise",
        )
        cv_metrics = {}
        for metric in ("balanced_accuracy", "macro_f1", "roc_auc"):
            vals = cv_raw[f"test_{metric}"]
            cv_metrics[metric] = {
                "fold_scores": [float(x) for x in vals],
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
            }

        joblib.dump(model, OUT / f"{name}.joblib")
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, list(model.classes_).index("LUSC")]
        pd.DataFrame({
            "patient_id": df.loc[test_idx, "patient_id"].to_numpy(),
            "true_label": y_test.to_numpy(),
            "predicted_label": pred,
            "probability_lusc": prob,
        }).to_csv(OUT / f"{name}_holdout_predictions.csv", index=False)

        results["feature_sets"][name] = {
            "features": features,
            "holdout": holdout,
            "training_only_5fold_cv": cv_metrics,
        }
        summary.append({
            "model": name,
            "features": ", ".join(features),
            "balanced_accuracy": holdout["balanced_accuracy"],
            "macro_f1": holdout["macro_f1"],
            "lusc_precision": holdout["lusc_precision"],
            "lusc_recall": holdout["lusc_recall"],
            "holdout_roc_auc": holdout["roc_auc_lusc_positive"],
            "cv_balanced_accuracy_mean": cv_metrics["balanced_accuracy"]["mean"],
            "cv_roc_auc_mean": cv_metrics["roc_auc"]["mean"],
        })

    (OUT / "comparison_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summary).to_csv(OUT / "comparison_summary.csv", index=False)
    print("\n=== FEATURE COMPARISON ===")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nSaved comparison report: {OUT / 'comparison_metrics.json'}")
    print(f"Saved summary: {OUT / 'comparison_summary.csv'}")
    print("Existing model artifacts were not modified.")

if __name__ == "__main__":
    main()
