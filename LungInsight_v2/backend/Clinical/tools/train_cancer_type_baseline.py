from pathlib import Path
import json
import warnings

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
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SEED = 42
FEATURES = ["sex_at_baseline", "smoking_history", "stage"]
TARGET = "cancer_type"
COHORT = "MSK-CHORD-2024"
CLASSES = ["LUAD", "LUSC"]


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    result = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(
            y_test, predictions, average="macro", zero_division=0
        )),
        "lusc_precision": float(precision_score(
            y_test, predictions, pos_label="LUSC", zero_division=0
        )),
        "lusc_recall": float(recall_score(
            y_test, predictions, pos_label="LUSC", zero_division=0
        )),
        "lusc_f1": float(f1_score(
            y_test, predictions, pos_label="LUSC", zero_division=0
        )),
        "confusion_matrix_labels": CLASSES,
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=CLASSES
        ).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=CLASSES,
            output_dict=True, zero_division=0
        ),
    }

    if hasattr(model, "predict_proba") and len(set(y_test)) == 2:
        classes = list(model.classes_)
        lusc_index = classes.index("LUSC")
        probabilities = model.predict_proba(X_test)[:, lusc_index]
        result["roc_auc_lusc_positive"] = float(
            roc_auc_score((y_test == "LUSC").astype(int), probabilities)
        )
    else:
        result["roc_auc_lusc_positive"] = None

    return result


def main():
    clinical_dir = Path(__file__).resolve().parents[1]
    input_path = (
        clinical_dir / "artifacts" / "cancer_type_dataset"
        / "cancer_type_model_ready_candidate.csv"
    )
    output_dir = clinical_dir / "artifacts" / "cancer_type_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_csv(input_path, dtype={"patient_id": str})
    required = ["patient_id", "cohort", TARGET] + FEATURES
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # This first model deliberately uses MSK only. TCGA cohort membership
    # is confounded with cancer type and must not be pooled into training.
    df = df.loc[df["cohort"].eq(COHORT)].copy()
    df = df.dropna(subset=[TARGET])

    if df["patient_id"].duplicated().any():
        raise ValueError("Duplicate MSK patient IDs detected.")
    if set(df[TARGET].unique()) != set(CLASSES):
        raise ValueError(
            f"Expected target classes {CLASSES}; got {sorted(df[TARGET].unique())}"
        )

    X = df[FEATURES].copy()
    y = df[TARGET].astype(str).copy()

    # Split before fitting any learned preprocessing.
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("categorical", categorical_pipeline, FEATURES),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )),
    ])

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    model.fit(X_train, y_train)

    dummy_metrics = evaluate(dummy, X_test, y_test)
    test_metrics = evaluate(model, X_test, y_test)

    # Cross-validation uses training data only. Each fold refits the
    # imputer, encoder, and classifier independently.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_result = cross_validate(
        model,
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
        values = cv_result[f"test_{metric}"]
        cv_metrics[metric] = {
            "fold_scores": [float(v) for v in values],
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }

    # Save a self-contained pipeline, not just the classifier.
    joblib.dump(model, output_dir / "logistic_regression_pipeline.joblib")

    test_probabilities = model.predict_proba(X_test)
    lusc_index = list(model.classes_).index("LUSC")
    predictions = pd.DataFrame({
        "dataset_row_index": idx_test,
        "true_label": y_test.to_numpy(),
        "predicted_label": model.predict(X_test),
        "probability_lusc": test_probabilities[:, lusc_index],
    })
    predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)

    metrics = {
        "purpose": "Exploratory research baseline; not clinically validated.",
        "input_dataset": str(input_path),
        "cohort": COHORT,
        "target": TARGET,
        "features": FEATURES,
        "excluded_features": [
            "patient_id", "cohort", "label_provenance", "sample_count",
            "source_record_id", "histology", "OncoTree code",
            "survival", "vital status", "treatment outcome",
        ],
        "scikit_learn_version": sklearn.__version__,
        "random_seed": SEED,
        "test_fraction": 0.20,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "training_class_counts": {
            str(k): int(v) for k, v in y_train.value_counts().items()
        },
        "test_class_counts": {
            str(k): int(v) for k, v in y_test.value_counts().items()
        },
        "preprocessing": (
            "Most-frequent categorical imputation and one-hot encoding, "
            "fitted inside the pipeline and separately in each CV fold."
        ),
        "model": "LogisticRegression(class_weight='balanced')",
        "dummy_holdout_metrics": dummy_metrics,
        "model_holdout_metrics": test_metrics,
        "training_only_5fold_cv": cv_metrics,
        "limitations": [
            "Single-cohort random holdout; not external validation.",
            "MSK current age was excluded because it is not age at diagnosis.",
            "MSK labels are imbalanced; interpret per-class metrics.",
            "Smoking history is NLP-derived and has missing values.",
            "Stage definitions and timing may vary.",
            "No clinical utility or prospective validation established.",
        ],
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Output directory:", output_dir)
    print("Training rows:", len(X_train), "| Test rows:", len(X_test))
    print("\nTest-set class counts:")
    print(y_test.value_counts().to_string())

    print("\nDummy classifier — holdout")
    print(json.dumps({
        k: dummy_metrics[k] for k in (
            "accuracy", "balanced_accuracy", "macro_f1",
            "lusc_precision", "lusc_recall", "roc_auc_lusc_positive"
        )
    }, indent=2))

    print("\nLogistic regression — holdout")
    print(json.dumps({
        k: test_metrics[k] for k in (
            "accuracy", "balanced_accuracy", "macro_f1",
            "lusc_precision", "lusc_recall", "lusc_f1",
            "roc_auc_lusc_positive", "confusion_matrix"
        )
    }, indent=2))

    print("\nTraining-only 5-fold CV:")
    print(json.dumps(cv_metrics, indent=2))
    print("\nFull per-class report:")
    print(json.dumps(test_metrics["classification_report"], indent=2))
    print("\nSaved model: logistic_regression_pipeline.joblib")
    print("Saved metrics: metrics.json")
    print("Saved predictions: holdout_predictions.csv")


if __name__ == "__main__":
    main()
