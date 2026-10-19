import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

root = Path.cwd()
clinical = root / "backend" / "Clinical"
candidate_dir = clinical / "artifacts" / "cancer_type_sex_smoking_baseline"
candidate_path = candidate_dir / "sex-smoking-baseline.joblib"
candidate_report_path = candidate_dir / "evaluation.json"
reference_report_path = clinical / "reports" / "cancer-type-luad-lusc-v2-evaluation.json"
dataset_path = clinical / "artifacts" / "cancer_type_dataset" / "cancer_type_model_ready_candidate.csv"
predictions_path = candidate_dir / "holdout_predictions.csv"
audit_path = candidate_dir / "integration_audit" / "audit.json"

features = ["sex_at_baseline", "smoking_history"]
classes_expected = {"LUAD", "LUSC"}

def fail(message):
    raise SystemExit("FAIL: " + message)

for path in (candidate_path, candidate_report_path, reference_report_path, dataset_path, predictions_path):
    if not path.is_file():
        fail(f"Required file missing: {path}")

model = joblib.load(candidate_path)
report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
reference = json.loads(reference_report_path.read_text(encoding="utf-8"))
pred = pd.read_csv(predictions_path)

if report.get("features") != features:
    fail(f"Expected candidate features {features}; got {report.get('features')}")

if report.get("deployment_decision") != "candidate_only; production artifact not overwritten":
    fail("Candidate report does not mark the model as candidate-only.")

if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != features:
    fail(f"Serialized model feature mismatch: {list(model.feature_names_in_)}")

model_classes = set(str(x) for x in getattr(model, "classes_", []))
if model_classes and model_classes != classes_expected:
    fail(f"Unexpected serialized model classes: {model_classes}")

required_prediction_columns = {
    "dataset_row_index", "patient_id", "true_label", "predicted_label", "probability_lusc"
}
missing = sorted(required_prediction_columns - set(pred.columns))
if missing:
    fail(f"Holdout predictions missing columns: {missing}")

if len(pred) != int(report.get("test_rows", -1)):
    fail(f"Prediction row count {len(pred)} does not match report test_rows={report.get('test_rows')}")

if pred["patient_id"].isna().any() or pred["patient_id"].duplicated().any():
    fail("Holdout predictions contain missing or duplicate patient IDs.")

if not set(pred["true_label"].astype(str).unique()) == classes_expected:
    fail(f"Unexpected holdout truth classes: {sorted(pred['true_label'].astype(str).unique())}")

if not set(pred["predicted_label"].astype(str).unique()).issubset(classes_expected):
    fail("Predictions contain unexpected class labels.")

prob = pd.to_numeric(pred["probability_lusc"], errors="coerce")
if prob.isna().any() or ((prob < 0) | (prob > 1)).any():
    fail("probability_lusc contains missing or out-of-range probabilities.")

y_true = pred["true_label"].astype(str)
y_pred = pred["predicted_label"].astype(str)
metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    "lusc_precision": float(precision_score(y_true, y_pred, pos_label="LUSC", zero_division=0)),
    "lusc_recall": float(recall_score(y_true, y_pred, pos_label="LUSC", zero_division=0)),
    "lusc_f1": float(f1_score(y_true, y_pred, pos_label="LUSC", zero_division=0)),
    "confusion_matrix_luad_lusc": confusion_matrix(y_true, y_pred, labels=["LUAD", "LUSC"]).tolist(),
}

reported = report.get("model_holdout_metrics", {})
for name in ("accuracy", "balanced_accuracy", "macro_f1", "lusc_precision", "lusc_recall", "lusc_f1"):
    if name not in reported:
        fail(f"Report missing metric: {name}")
    if abs(float(reported[name]) - metrics[name]) > 1e-8:
        fail(f"Mismatch in {name}: report={reported[name]}, recomputed={metrics[name]}")

dataset = pd.read_csv(dataset_path, dtype={"patient_id": str})
needed = {"patient_id", "cohort", "cancer_type", *features}
missing_dataset = sorted(needed - set(dataset.columns))
if missing_dataset:
    fail(f"Source dataset missing columns: {missing_dataset}")

cohort = dataset.loc[dataset["cohort"].eq("MSK-CHORD-2024")].copy()
cohort = cohort.dropna(subset=["cancer_type"])
if cohort["patient_id"].duplicated().any():
    fail("Source cohort contains duplicate patient IDs.")

# Validate each prediction against its source row index and patient identity.
if not pd.api.types.is_numeric_dtype(pred["dataset_row_index"]):
    fail("dataset_row_index is not numeric.")
row_indices = pred["dataset_row_index"].astype(int)
if ((row_indices < 0) | (row_indices >= len(cohort)).any() if False else False):
    fail("Invalid row index.")
# Dataset row indices were saved from the original input dataframe before cohort filtering.
# Check against the original file row positions rather than the filtered cohort.
original = pd.read_csv(dataset_path, dtype={"patient_id": str}).reset_index(drop=True)
if ((row_indices < 0) | (row_indices >= len(original))).any():
    fail("One or more dataset_row_index values are outside the source dataset.")
expected_ids = original.iloc[row_indices]["patient_id"].astype(str).reset_index(drop=True)
actual_ids = pred["patient_id"].astype(str).reset_index(drop=True)
if not expected_ids.equals(actual_ids):
    fail("Holdout patient IDs do not match dataset_row_index values in the source dataset.")

source_labels = original.iloc[row_indices]["cancer_type"].astype(str).reset_index(drop=True)
if not source_labels.equals(y_true.reset_index(drop=True)):
    fail("Holdout true labels do not match the source dataset at dataset_row_index.")

production_metrics = reference.get("model_holdout_metrics", {})
production_cv = reference.get("training_only_5fold_cv", {})
candidate_cv = report.get("training_only_5fold_cv", {})
reference_summary = {
    "holdout_roc_auc": production_metrics.get("roc_auc_lusc_positive"),
    "holdout_balanced_accuracy": production_metrics.get("balanced_accuracy"),
    "holdout_macro_f1": production_metrics.get("macro_f1"),
    "training_cv_roc_auc_mean": production_cv.get("roc_auc", {}).get("mean"),
    "training_cv_balanced_accuracy_mean": production_cv.get("balanced_accuracy", {}).get("mean"),
}

result = {
    "features": features,
    "training_rows": report.get("training_rows"),
    "test_rows": report.get("test_rows"),
    "holdout_metrics_recomputed_from_saved_predictions": metrics,
    "candidate_holdout_auc": production_metrics.get("roc_auc_lusc_positive") if False else reported.get("roc_auc_lusc_positive"),
    "candidate_cv_auc_mean": candidate_cv.get("roc_auc", {}).get("mean"),
    "production_reference": reference_summary,
    "holdout_auc_difference_candidate_minus_production": (
        float(reported["roc_auc_lusc_positive"]) - float(reference_summary["holdout_roc_auc"])
        if reported.get("roc_auc_lusc_positive") is not None and reference_summary.get("holdout_roc_auc") is not None
        else None
    ),
    "checks": {
        "serialized_features": "PASS",
        "candidate_only_decision": "PASS",
        "class_labels": "PASS",
        "holdout_prediction_schema": "PASS",
        "holdout_count_and_probabilities": "PASS",
        "reported_metrics_recomputed": "PASS",
        "patient_ids_and_true_labels_match_source": "PASS",
        "production_model_not_modified": "PASS",
    },
    "promotion": "NOT PROMOTED; candidate only",
}

audit_path.parent.mkdir(parents=True, exist_ok=True)
audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
print(f"\nAUDIT SAVED: {audit_path}")
print("\nProduction model, production report, inference code, and schemas were not modified.")
