from pathlib import Path
import json
from itertools import combinations

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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SEED = 42
COHORT = "MSK-CHORD-2024"
TARGET = "cancer_type"
ALL_FEATURES = [
    "sex_at_baseline",
    "smoking_history",
    "stage",
]
EXPECTED_CLASSES = {"LUAD", "LUSC"}


def build_model(features):
    preprocessing = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), list(features)),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )),
    ])


def calculate_metrics(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    lusc_index = classes.index("LUSC")

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(
            y, predictions, average="macro", zero_division=0
        )),
        "lusc_precision": float(precision_score(
            y, predictions, pos_label="LUSC", zero_division=0
        )),
        "lusc_recall": float(recall_score(
            y, predictions, pos_label="LUSC", zero_division=0
        )),
        "lusc_f1": float(f1_score(
            y, predictions, pos_label="LUSC", zero_division=0
        )),
        "roc_auc": float(roc_auc_score(
            (y == "LUSC").astype(int), probabilities[:, lusc_index]
        )),
    }


def main():
    clinical_dir = Path(__file__).resolve().parents[1]
    input_path = (
        clinical_dir / "artifacts" / "cancer_type_dataset"
        / "cancer_type_model_ready_candidate.csv"
    )
    output_dir = (
        clinical_dir / "artifacts" / "cancer_type_feature_ablation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, dtype={"patient_id": str})
    df = df.loc[df["cohort"].eq(COHORT)].dropna(subset=[TARGET]).copy()

    required = ["patient_id", TARGET] + ALL_FEATURES
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs found.")
    if set(df[TARGET].unique()) != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected target classes: {df[TARGET].unique()}")

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype(str).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
        "roc_auc": "roc_auc",
    }

    # Majority-class dummy baseline for reference.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_test = calculate_metrics(dummy, X_test, y_test)

    results = []
    models = {}

    for size in range(1, len(ALL_FEATURES) + 1):
        for features in combinations(ALL_FEATURES, size):
            name = " + ".join(features)
            model = build_model(features)
            model.fit(X_train[list(features)], y_train)

            test_metrics = calculate_metrics(
                model, X_test[list(features)], y_test
            )
            cv_result = cross_validate(
                build_model(features),
                X_train[list(features)],
                y_train,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=1,
                error_score="raise",
            )

            row = {
                "features": name,
                "n_features": len(features),
                "cv_balanced_accuracy_mean": float(
                    np.mean(cv_result["test_balanced_accuracy"])
                ),
                "cv_balanced_accuracy_std": float(
                    np.std(cv_result["test_balanced_accuracy"], ddof=1)
                ),
                "cv_macro_f1_mean": float(
                    np.mean(cv_result["test_macro_f1"])
                ),
                "cv_macro_f1_std": float(
                    np.std(cv_result["test_macro_f1"], ddof=1)
                ),
                "cv_roc_auc_mean": float(
                    np.mean(cv_result["test_roc_auc"])
                ),
                "cv_roc_auc_std": float(
                    np.std(cv_result["test_roc_auc"], ddof=1)
                ),
            }
            row.update({
                f"test_{key}": value
                for key, value in test_metrics.items()
            })
            results.append(row)
            models[name] = (model, list(features))

    result_df = pd.DataFrame(results).sort_values(
        ["cv_balanced_accuracy_mean", "cv_roc_auc_mean"],
        ascending=False,
    )
    result_df.to_csv(output_dir / "feature_ablation_results.csv", index=False)

    # Save the best feature set selected by training-only CV balanced accuracy.
    best_name = str(result_df.iloc[0]["features"])
    best_model, best_features = models[best_name]
    joblib.dump(best_model, output_dir / "best_ablation_pipeline.joblib")

    # Save all holdout and CV metrics, plus the dummy baseline.
    report = {
        "purpose": "Exploratory feature ablation; not clinical validation.",
        "cohort": COHORT,
        "target": TARGET,
        "random_seed": SEED,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_class_counts": {
            str(k): int(v) for k, v in y_train.value_counts().items()
        },
        "test_class_counts": {
            str(k): int(v) for k, v in y_test.value_counts().items()
        },
        "scikit_learn_version": sklearn.__version__,
        "dummy_holdout_metrics": dummy_test,
        "selection_rule": (
            "Highest mean 5-fold training-only CV balanced accuracy; "
            "ROC-AUC used as secondary sort key."
        ),
        "best_feature_set_by_training_cv": best_name,
        "best_features": best_features,
        "all_results": result_df.to_dict(orient="records"),
        "caution": (
            "Holdout metrics are diagnostic comparisons across feature sets. "
            "Because multiple sets are compared, do not treat the highest "
            "holdout score as an unbiased final performance estimate."
        ),
    }
    (output_dir / "feature_ablation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("Output:", output_dir)
    print("\nDummy holdout baseline:")
    print(json.dumps(dummy_test, indent=2))

    display_columns = [
        "features",
        "cv_balanced_accuracy_mean",
        "cv_balanced_accuracy_std",
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
        "test_balanced_accuracy",
        "test_roc_auc",
        "test_lusc_precision",
        "test_lusc_recall",
        "test_lusc_f1",
    ]
    print("\nFeature ablation results (sorted by training CV):")
    print(result_df[display_columns].round(4).to_string(index=False))

    print("\nBest feature set by training CV:", best_name)
    print("Saved CSV: feature_ablation_results.csv")
    print("Saved JSON: feature_ablation_report.json")
    print("Saved pipeline: best_ablation_pipeline.joblib")


if __name__ == "__main__":
    main()
