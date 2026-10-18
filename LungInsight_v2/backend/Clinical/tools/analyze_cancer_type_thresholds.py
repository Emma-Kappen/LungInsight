from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


SEED = 42
COHORT = "MSK-CHORD-2024"
FEATURES = ["sex_at_baseline", "smoking_history", "stage"]
TARGET = "cancer_type"
LABELS = ["LUAD", "LUSC"]


def build_model():
    prep = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), FEATURES),
    ])
    return Pipeline([
        ("preprocessing", prep),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )),
    ])


def metrics_at_threshold(y_true, probabilities, threshold):
    predictions = np.where(probabilities >= threshold, "LUSC", "LUAD")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "lusc_precision": float(precision_score(
            y_true, predictions, pos_label="LUSC", zero_division=0
        )),
        "lusc_recall": float(recall_score(
            y_true, predictions, pos_label="LUSC", zero_division=0
        )),
        "lusc_f1": float(f1_score(
            y_true, predictions, pos_label="LUSC", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y_true, predictions, average="macro", zero_division=0
        )),
        "roc_auc": float(roc_auc_score(
            (y_true == "LUSC").astype(int), probabilities
        )),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=LABELS
        ).tolist(),
    }


def choose_threshold(y_true, probabilities, objective):
    # Threshold selection uses only out-of-fold training predictions.
    candidates = np.linspace(0.05, 0.95, 181)
    scored = []

    for threshold in candidates:
        result = metrics_at_threshold(y_true, probabilities, threshold)
        score = result[objective]
        # Prefer the threshold closest to 0.5 if scores tie.
        scored.append((score, -abs(threshold - 0.5), threshold))

    scored.sort(reverse=True)
    return float(scored[0][2])


def main():
    clinical_dir = Path(__file__).resolve().parents[1]
    input_path = (
        clinical_dir / "artifacts" / "cancer_type_dataset"
        / "cancer_type_model_ready_candidate.csv"
    )
    output_dir = (
        clinical_dir / "artifacts" / "cancer_type_threshold_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, dtype={"patient_id": str})
    df = df.loc[df["cohort"].eq(COHORT)].dropna(subset=[TARGET]).copy()

    if df["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs detected.")
    if set(df[TARGET].unique()) != set(LABELS):
        raise ValueError(f"Unexpected target labels: {df[TARGET].unique()}")

    X = df[FEATURES].copy()
    y = df[TARGET].astype(str).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    base_model = build_model()

    # Each training row gets a probability from a model that did not train
    # on that row. These probabilities select thresholds without using test.
    oof_probabilities = cross_val_predict(
        clone(base_model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )
    classes = sorted(y_train.unique())
    lusc_index = classes.index("LUSC")
    oof_lusc_probability = oof_probabilities[:, lusc_index]

    balanced_threshold = choose_threshold(
        y_train, oof_lusc_probability, "balanced_accuracy"
    )
    f1_threshold = choose_threshold(
        y_train, oof_lusc_probability, "lusc_f1"
    )

    # Refit on the full training partition, then evaluate holdout only.
    final_model = build_model()
    final_model.fit(X_train, y_train)
    model_classes = list(final_model.classes_)
    test_lusc_probability = final_model.predict_proba(X_test)[
        :, model_classes.index("LUSC")
    ]

    threshold_map = {
        "default_0.50": 0.50,
        "training_oof_max_balanced_accuracy": balanced_threshold,
        "training_oof_max_lusc_f1": f1_threshold,
    }

    results = {}
    for name, threshold in threshold_map.items():
        results[name] = {
            "threshold_selection": (
                "fixed default" if name == "default_0.50"
                else "training-only out-of-fold predictions"
            ),
            "holdout_metrics": metrics_at_threshold(
                y_test, test_lusc_probability, threshold
            ),
        }

    # Save holdout predictions at all three operating thresholds.
    prediction_df = pd.DataFrame({
        "true_label": y_test.to_numpy(),
        "probability_lusc": test_lusc_probability,
    })
    for name, threshold in threshold_map.items():
        prediction_df[f"predicted_{name}"] = np.where(
            test_lusc_probability >= threshold, "LUSC", "LUAD"
        )
    prediction_df.to_csv(output_dir / "holdout_threshold_predictions.csv",
                         index=False)

    report = {
        "purpose": "Exploratory threshold analysis; not clinical validation.",
        "cohort": COHORT,
        "features": FEATURES,
        "target": TARGET,
        "random_seed": SEED,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "training_only_oof_thresholds": {
            "max_balanced_accuracy": balanced_threshold,
            "max_lusc_f1": f1_threshold,
        },
        "holdout_results": results,
        "limitations": [
            "Thresholds are selected on training-only out-of-fold predictions.",
            "The holdout is used for evaluation, not threshold selection.",
            "The selected operating threshold may vary in another population.",
            "Predicted probabilities are not necessarily calibrated risks.",
            "This is not a clinically validated diagnostic model.",
        ],
    }
    (output_dir / "threshold_analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    joblib.dump(final_model, output_dir / "logistic_regression_pipeline.joblib")

    print("Output directory:", output_dir)
    print("\nThresholds selected using training-only OOF predictions:")
    print(json.dumps(report["training_only_oof_thresholds"], indent=2))

    print("\nHoldout comparison:")
    rows = []
    for name, entry in results.items():
        m = entry["holdout_metrics"]
        rows.append({
            "setting": name,
            "threshold": m["threshold"],
            "balanced_accuracy": m["balanced_accuracy"],
            "lusc_precision": m["lusc_precision"],
            "lusc_recall": m["lusc_recall"],
            "lusc_f1": m["lusc_f1"],
            "macro_f1": m["macro_f1"],
            "roc_auc": m["roc_auc"],
            "confusion_matrix": m["confusion_matrix"],
        })
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nSaved: threshold_analysis_report.json")
    print("Saved: holdout_threshold_predictions.csv")
    print("Saved: logistic_regression_pipeline.joblib")


if __name__ == "__main__":
    main()
