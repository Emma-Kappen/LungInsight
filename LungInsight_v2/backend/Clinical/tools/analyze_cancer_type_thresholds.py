from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "cancer_type_dataset" / "cancer_type_model_ready_candidate.csv"
OUT = ROOT / "artifacts" / "cancer_type_threshold_analysis_20261019"
FEATURES = ["sex_at_baseline", "smoking_history"]
SEED = 42

def make_model():
    prep = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), FEATURES),
    ])
    return Pipeline([
        ("preprocessing", prep),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED
        )),
    ])

def metrics(y, prob, threshold):
    pred = np.where(prob >= threshold, "LUSC", "LUAD")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "lusc_precision": float(precision_score(y, pred, pos_label="LUSC", zero_division=0)),
        "lusc_recall": float(recall_score(y, pred, pos_label="LUSC", zero_division=0)),
        "lusc_f1": float(f1_score(y, pred, pos_label="LUSC", zero_division=0)),
        "confusion_matrix_labels": ["LUAD", "LUSC"],
        "confusion_matrix": confusion_matrix(
            y, pred, labels=["LUAD", "LUSC"]
        ).tolist(),
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, dtype={"patient_id": str})
    df = df.loc[df["cohort"].eq("MSK-CHORD-2024")].dropna(subset=["cancer_type"]).copy()
    if df["patient_id"].isna().any() or df["patient_id"].duplicated().any():
        raise ValueError("Missing or duplicate patient IDs; refusing to proceed.")

    X = df[FEATURES]
    y = df["cancer_type"].astype(str)
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    # Out-of-fold training probabilities only: holdout is not used for selection.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(
        make_model(), X_train, y_train, cv=cv,
        method="predict_proba", n_jobs=1
    )
    classes = sorted(y_train.unique())
    lusc_col = classes.index("LUSC")
    oof_prob = oof[:, lusc_col]

    threshold_rows = []
    for threshold in np.round(np.arange(0.10, 0.901, 0.01), 2):
        row = metrics(y_train.to_numpy(), oof_prob, threshold)
        threshold_rows.append(row)

    # Primary criterion is OOF balanced accuracy; break ties by macro-F1,
    # then prefer a threshold closer to 0.50.
    best = sorted(
        threshold_rows,
        key=lambda r: (
            -r["balanced_accuracy"],
            -r["macro_f1"],
            abs(r["threshold"] - 0.50),
        ),
    )[0]
    chosen_threshold = best["threshold"]

    model = make_model()
    model.fit(X_train, y_train)
    joblib.dump(model, OUT / "sex_smoking_pipeline.joblib")
    holdout_prob = model.predict_proba(X_test)[:, list(model.classes_).index("LUSC")]

    holdout_default = metrics(y_test.to_numpy(), holdout_prob, 0.50)
    holdout_tuned = metrics(y_test.to_numpy(), holdout_prob, chosen_threshold)

    pd.DataFrame(threshold_rows).to_csv(OUT / "training_oof_thresholds.csv", index=False)
    pd.DataFrame({
        "patient_id": df.loc[test_idx, "patient_id"].to_numpy(),
        "true_label": y_test.to_numpy(),
        "probability_lusc": holdout_prob,
        "prediction_threshold_050": np.where(holdout_prob >= 0.50, "LUSC", "LUAD"),
        "prediction_training_selected_threshold": np.where(
            holdout_prob >= chosen_threshold, "LUSC", "LUAD"
        ),
    }).to_csv(OUT / "holdout_predictions.csv", index=False)

    report = {
        "purpose": "Exploratory threshold analysis; not clinically validated.",
        "features": FEATURES,
        "threshold_selection": "Maximize training out-of-fold balanced accuracy; ties resolved by macro-F1, then closeness to 0.50.",
        "selected_threshold": chosen_threshold,
        "training_oof_metrics_at_selected_threshold": best,
        "holdout_at_default_threshold_0_50": holdout_default,
        "holdout_at_training_selected_threshold": holdout_tuned,
        "holdout_used_for_threshold_selection": False,
        "limitations": [
            "Single-cohort random holdout; not external validation.",
            "Smoking-history timing is uncertain.",
            "Threshold is exploratory and may not generalize.",
            "Not for clinical decision-making.",
        ],
    }
    (OUT / "threshold_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Selected threshold from training OOF only:", chosen_threshold)
    print("\nHOLDOUT — default threshold 0.50")
    print(json.dumps(holdout_default, indent=2))
    print("\nHOLDOUT — training-selected threshold")
    print(json.dumps(holdout_tuned, indent=2))
    print("\nSaved report:", OUT / "threshold_report.json")
    print("Saved OOF threshold grid:", OUT / "training_oof_thresholds.csv")
    print("Saved holdout predictions:", OUT / "holdout_predictions.csv")

if __name__ == "__main__":
    main()
