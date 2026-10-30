import csv
import json
import math
import re
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    accuracy_score
)

# ------------------------------------------------------------
# Paths and locked configuration
# ------------------------------------------------------------

ROOT = Path(sys.argv[1]).resolve()
LUAD_FILE = Path(sys.argv[2]).resolve()
LUSC_FILE = Path(sys.argv[3]).resolve()
OUTPUT = Path(sys.argv[4]).resolve()
MODELS = Path(sys.argv[5]).resolve()

EXPECTED = {"LUAD": 522, "LUSC": 504}
EXPECTED_TOTAL = sum(EXPECTED.values())

ARTIFACT_PATH = MODELS / "cancer-type-luad-lusc-research-v3.joblib"
METRICS_PATH = OUTPUT / "model_metrics_cohort_confounded.csv"
RECONCILIATION_PATH = OUTPUT / "tcga_parser_reconciliation.csv"
MISSINGNESS_PATH = OUTPUT / "feature_missingness_corrected.csv"
MANIFEST_PATH = OUTPUT / "cancer-type-luad-lusc-research-v3.manifest.json"
SCHEMA_PATH = OUTPUT / "cancer-type-luad-lusc-research-v3.schema.json"
SUMMARY_PATH = OUTPUT / "build_summary.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

# A TCGA patient barcode has the form TCGA-XX-XXXX.
PATIENT_BARCODE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$", re.IGNORECASE)

# Do not use either histology diagnosis column as a predictor.
# These are used only in a separate, non-predictive source audit.
FORBIDDEN_PREDICTOR_TOKENS = (
    "histologic_diagnosis",
    "icd_o_3_histology",
    "pathologic_stage",
    "treatment",
    "response",
    "outcome",
    "survival",
    "ethnicity",
    "karnofsky",
    "kps",
)

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def write_text(path, content):
    path.write_text(content, encoding="utf-8")

def clean_header(value):
    return str(value).strip().lstrip("\ufeff")

def unique_headers(headers):
    counts = {}
    result = []

    for raw in headers:
        name = clean_header(raw)
        if not name:
            name = "unnamed_column"

        key = name.lower()
        counts[key] = counts.get(key, 0) + 1

        if counts[key] == 1:
            result.append(name)
        else:
            result.append(f"{name}__duplicate_{counts[key]}")

    return result

def find_column(headers, candidates, description):
    lower_map = {h.lower(): h for h in headers}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise RuntimeError(
        f"Required field not found for {description}. "
        f"Tried {candidates}. Available fields: {headers}"
    )

def is_metadata_row(row):
    values = [str(x).strip() for x in row if x is not None]
    joined = " ".join(values).upper()

    if any("CDE_ID:" in value.upper() for value in values):
        return True

    metadata_markers = (
        "BCR_PATIENT_BARCODE",
        "GENDER",
        "RACE",
        "TOBACCO_SMOKING",
        "CDE_ID",
    )

    if any(marker in joined for marker in metadata_markers):
        return True

    return False

def safe_numeric(value):
    if value is None:
        return np.nan

    value = str(value).strip()

    if not value:
        return np.nan

    if value.lower() in {
        "na", "n/a", "unknown", "not reported",
        "not available", "not evaluated", "[unknown]",
        "discrepancy", "null", "none", "--"
    }:
        return np.nan

    try:
        number = float(value)
        if not math.isfinite(number):
            return np.nan
        return number
    except (TypeError, ValueError):
        return np.nan

def normalize_category(value):
    if value is None:
        return np.nan

    value = str(value).strip()

    if not value:
        return np.nan

    if value.lower() in {
        "na", "n/a", "unknown", "not reported",
        "not available", "not evaluated", "[unknown]",
        "discrepancy", "null", "none", "--"
    }:
        return np.nan

    return value

# ------------------------------------------------------------
# Corrected positional TCGA parser
# ------------------------------------------------------------

def parse_tcga_file(path, cohort):
    """
    Find the actual TSV header by the barcode field.
    Preserve duplicate column names positionally.
    Accept only rows with a valid TCGA patient barcode.
    Count metadata and invalid-identity rows separately.
    Never export patient identifiers.
    """

    reconciliation = {
        "cohort": cohort,
        "file": path.name,
        "header_row_index_zero_based": None,
        "raw_rows_after_header": 0,
        "valid_patient_rows": 0,
        "metadata_rows_excluded": 0,
        "invalid_or_blank_identity_rows_excluded": 0,
        "duplicate_patient_barcodes": 0,
        "row_width_mismatch": 0,
    }

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        all_rows = list(reader)

    header_index = None

    for index, row in enumerate(all_rows):
        normalized = [clean_header(x).lower() for x in row]
        if "bcr_patient_barcode" in normalized:
            header_index = index
            break

    if header_index is None:
        raise RuntimeError(
            f"{cohort}: could not locate the bcr_patient_barcode header row."
        )

    reconciliation["header_row_index_zero_based"] = header_index

    original_headers = [clean_header(x) for x in all_rows[header_index]]
    headers = unique_headers(original_headers)

    barcode_index = next(
        i for i, value in enumerate(original_headers)
        if value.lower() == "bcr_patient_barcode"
    )

    records = []
    seen_barcodes = set()

    for row in all_rows[header_index + 1:]:
        if not row or not any(str(x).strip() for x in row):
            continue

        reconciliation["raw_rows_after_header"] += 1

        if len(row) != len(headers):
            reconciliation["row_width_mismatch"] += 1
            continue

        barcode = str(row[barcode_index]).strip()

        if not PATIENT_BARCODE.fullmatch(barcode):
            if is_metadata_row(row):
                reconciliation["metadata_rows_excluded"] += 1
            else:
                reconciliation["invalid_or_blank_identity_rows_excluded"] += 1
            continue

        barcode_key = barcode.upper()

        if barcode_key in seen_barcodes:
            reconciliation["duplicate_patient_barcodes"] += 1
            continue

        seen_barcodes.add(barcode_key)

        # Positional construction prevents duplicate TSV headers
        # from silently overwriting one another.
        record = {}
        for i, header in enumerate(headers):
            record[header] = row[i].strip() if row[i] is not None else ""

        # Keep the barcode only in memory for deduplication.
        # It is never included in reports, model features, or exports.
        records.append(record)

    reconciliation["valid_patient_rows"] = len(records)

    if reconciliation["row_width_mismatch"] > 0:
        raise RuntimeError(
            f"{cohort}: found {reconciliation['row_width_mismatch']} rows "
            "with a field count inconsistent with the header. "
            "Parser stopped rather than silently discarding these rows."
        )

    if reconciliation["duplicate_patient_barcodes"] > 0:
        raise RuntimeError(
            f"{cohort}: duplicate patient barcodes detected. "
            "Parser stopped; no patient records were silently deduplicated."
        )

    expected = EXPECTED[cohort]
    if len(records) != expected:
        raise RuntimeError(
            f"{cohort}: corrected parser recovered {len(records)} valid "
            f"patients; expected exactly {expected}. Model training blocked."
        )

    return records, headers, reconciliation

# ------------------------------------------------------------
# Read the two cohorts
# ------------------------------------------------------------

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    if ARTIFACT_PATH.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing artifact: {ARTIFACT_PATH}"
        )

    luad_records, luad_headers, luad_recon = parse_tcga_file(
        LUAD_FILE, "LUAD"
    )
    lusc_records, lusc_headers, lusc_recon = parse_tcga_file(
        LUSC_FILE, "LUSC"
    )

    # Check both cohorts independently before combining.
    if len(luad_records) != EXPECTED["LUAD"]:
        raise RuntimeError("LUAD patient-count gate failed.")

    if len(lusc_records) != EXPECTED["LUSC"]:
        raise RuntimeError("LUSC patient-count gate failed.")

    # Resolve source columns using documented candidate names.
    # The actual selected field names are recorded in the manifest.
    age_luad = find_column(
        luad_headers,
        ["age_at_initial_pathologic_diagnosis"],
        "LUAD age at diagnosis"
    )
    age_lusc = find_column(
        lusc_headers,
        ["age_at_initial_pathologic_diagnosis"],
        "LUSC age at diagnosis"
    )

    sex_luad = find_column(
        luad_headers, ["gender"], "LUAD sex"
    )
    sex_lusc = find_column(
        lusc_headers, ["gender"], "LUSC sex"
    )

    smoking_luad = find_column(
        luad_headers,
        [
            "tobacco_smoking_history_indicator",
            "tobacco_smoking_history",
        ],
        "LUAD smoking status"
    )
    smoking_lusc = find_column(
        lusc_headers,
        [
            "tobacco_smoking_history_indicator",
            "tobacco_smoking_history",
        ],
        "LUSC smoking status"
    )

    packyears_luad = find_column(
        luad_headers,
        ["tobacco_smoking_pack_years_smoked"],
        "LUAD pack-years"
    )
    packyears_lusc = find_column(
        lusc_headers,
        ["tobacco_smoking_pack_years_smoked"],
        "LUSC pack-years"
    )

    race_luad = find_column(
        luad_headers, ["race"], "LUAD race sensitivity field"
    )
    race_lusc = find_column(
        lusc_headers, ["race"], "LUSC race sensitivity field"
    )

    selected_fields = {
        "LUAD": {
            "age": age_luad,
            "sex": sex_luad,
            "smoking_status": smoking_luad,
            "pack_years": packyears_luad,
            "race_sensitivity": race_luad,
        },
        "LUSC": {
            "age": age_lusc,
            "sex": sex_lusc,
            "smoking_status": smoking_lusc,
            "pack_years": packyears_lusc,
            "race_sensitivity": race_lusc,
        },
    }

    # --------------------------------------------------------
    # Harmonize only the four confirmed primary predictors.
    # Race is retained in a separate sensitivity column.
    # --------------------------------------------------------

    def make_frame(records, cohort, mapping):
        rows = []

        for record in records:
            age = safe_numeric(record.get(mapping["age"]))
            sex = normalize_category(record.get(mapping["sex"]))
            smoking = normalize_category(record.get(mapping["smoking_status"]))
            pack_years = safe_numeric(record.get(mapping["pack_years"]))
            race = normalize_category(record.get(mapping["race_sensitivity"]))

            # Smoking status codes are categorical, not ordinal.
            # Codes 1-5 are retained as category labels.
            # Unknown, undocumented, discrepancy, and other codes
            # are treated as missing rather than assigned an order.
            if not pd.isna(smoking):
                smoking_text = str(smoking).strip()
                if smoking_text.endswith(".0"):
                    smoking_text = smoking_text[:-2]

                if smoking_text in {"1", "2", "3", "4", "5"}:
                    smoking = smoking_text
                else:
                    smoking = np.nan

            rows.append({
                "cohort": cohort,
                "target": 1 if cohort == "LUAD" else 0,
                "age_at_diagnosis": age,
                "sex": sex,
                "smoking_status": smoking,
                "pack_years": pack_years,
                "race_sensitivity": race,
            })

        return pd.DataFrame(rows)

    df = pd.concat(
        [
            make_frame(luad_records, "LUAD", selected_fields["LUAD"]),
            make_frame(lusc_records, "LUSC", selected_fields["LUSC"]),
        ],
        ignore_index=True
    )

    if len(df) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Combined dataset has {len(df)} rows; expected {EXPECTED_TOTAL}."
        )

    if df["target"].nunique() != 2:
        raise RuntimeError("Both target classes are required.")

    if df["cohort"].value_counts().to_dict() != EXPECTED:
        raise RuntimeError("Combined cohort counts do not match expected counts.")

    primary_features = [
        "age_at_diagnosis",
        "sex",
        "smoking_status",
        "pack_years",
    ]
    race_features = primary_features + ["race_sensitivity"]

    # Guard against accidental forbidden fields entering predictors.
    for feature in primary_features + ["race_sensitivity"]:
        lower = feature.lower()
        if any(token in lower for token in FORBIDDEN_PREDICTOR_TOKENS):
            raise RuntimeError(f"Forbidden predictor field detected: {feature}")

    # --------------------------------------------------------
    # Corrected parser reconciliation and feature missingness
    # --------------------------------------------------------

    pd.DataFrame([luad_recon, lusc_recon]).to_csv(
        RECONCILIATION_PATH, index=False
    )

    missingness_rows = []
    for cohort, subset in df.groupby("cohort"):
        for feature in primary_features + ["race_sensitivity"]:
            missing_count = int(subset[feature].isna().sum())
            missingness_rows.append({
                "cohort": cohort,
                "feature": feature,
                "patient_count": int(len(subset)),
                "missing_count": missing_count,
                "observed_count": int(len(subset) - missing_count),
                "missing_percent": round(
                    100.0 * missing_count / len(subset), 2
                ),
            })

    pd.DataFrame(missingness_rows).to_csv(
        MISSINGNESS_PATH, index=False
    )

    # Fail if a primary feature is entirely missing in either cohort.
    for cohort, subset in df.groupby("cohort"):
        for feature in primary_features:
            if subset[feature].notna().sum() == 0:
                raise RuntimeError(
                    f"{cohort}: primary feature '{feature}' is entirely missing. "
                    "Model training blocked."
                )

    # --------------------------------------------------------
    # Build preprocessing pipelines
    # --------------------------------------------------------

    numeric_features = ["age_at_diagnosis", "pack_years"]
    categorical_features = ["sex", "smoking_status"]

    def make_one_hot_encoder():
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        except TypeError:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )

    def make_pipeline(algorithm, use_race):
        categorical = list(categorical_features)
        if use_race:
            categorical.append("race_sensitivity")

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="constant",
                fill_value="__MISSING__"
            )),
            ("onehot", make_one_hot_encoder()),
        ])

        preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical),
        ], remainder="drop")

        if algorithm == "logistic_regression":
            estimator = LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
            )
        elif algorithm == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    def calculate_metrics(y_true, probabilities, predictions):
        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan

        return {
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "average_precision": float(
                average_precision_score(y_true, probabilities)
            ),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true, predictions)
            ),
            "sensitivity_luad": float(sensitivity),
            "specificity_lusc": float(specificity),
            "true_negative_lusc": int(tn),
            "false_positive_luad": int(fp),
            "false_negative_lusc": int(fn),
            "true_positive_luad": int(tp),
        }

    # Fixed patient-level split. This is an exploratory split only;
    # cohort and target remain perfectly confounded.
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"].to_numpy(),
    )

    y = df["target"].to_numpy()
    results = []
    final_models = {}

    configurations = [
        ("primary_four_predictors", False),
        ("race_sensitivity", True),
    ]

    algorithms = [
        "logistic_regression",
        "random_forest",
    ]

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for feature_set_name, use_race in configurations:
        features = race_features if use_race else primary_features
        X = df[features].copy()

        for algorithm in algorithms:
            pipeline = make_pipeline(algorithm, use_race)

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            pipeline.fit(X_train, y_train)

            test_probabilities = pipeline.predict_proba(X_test)[:, 1]
            test_predictions = (test_probabilities >= 0.5).astype(int)

            test_metrics = calculate_metrics(
                y_test,
                test_probabilities,
                test_predictions,
            )

            # Out-of-fold estimates are included for transparency.
            # They are not independent validation because the cohort
            # label is perfectly aligned with the target.
            cv_probabilities = cross_val_predict(
                make_pipeline(algorithm, use_race),
                X,
                y,
                cv=cv,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]

            cv_predictions = (cv_probabilities >= 0.5).astype(int)

            cv_metrics = calculate_metrics(
                y,
                cv_probabilities,
                cv_predictions,
            )

            results.append({
                "feature_set": feature_set_name,
                "algorithm": algorithm,
                "validation_interpretation": "COHORT-CONFOUNDED; NOT GENERALISABLE",
                "test_patient_count": int(len(test_idx)),
                "test_roc_auc": test_metrics["roc_auc"],
                "test_average_precision": test_metrics["average_precision"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_sensitivity_luad": test_metrics["sensitivity_luad"],
                "test_specificity_lusc": test_metrics["specificity_lusc"],
                "test_true_negative_lusc": test_metrics["true_negative_lusc"],
                "test_false_positive_luad": test_metrics["false_positive_luad"],
                "test_false_negative_lusc": test_metrics["false_negative_lusc"],
                "test_true_positive_luad": test_metrics["true_positive_luad"],
                "cv_roc_auc": cv_metrics["roc_auc"],
                "cv_average_precision": cv_metrics["average_precision"],
                "cv_accuracy": cv_metrics["accuracy"],
                "cv_balanced_accuracy": cv_metrics["balanced_accuracy"],
                "cv_sensitivity_luad": cv_metrics["sensitivity_luad"],
                "cv_specificity_lusc": cv_metrics["specificity_lusc"],
            })

            # Fit the final deployable research artifact on all available
            # TCGA records. This does not connect it to the application API.
            final_pipeline = make_pipeline(algorithm, use_race)
            final_pipeline.fit(X, y)

            final_models[f"{feature_set_name}__{algorithm}"] = {
                "pipeline": final_pipeline,
                "features": features,
                "feature_set": feature_set_name,
                "algorithm": algorithm,
                "uses_race": bool(use_race),
            }

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(METRICS_PATH, index=False)

    # --------------------------------------------------------
    # Independent model artifact
    # --------------------------------------------------------

    artifact = {
        "model_name": "LungInsight TCGA LUAD-LUSC Research Histology Model",
        "model_version": "research-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": {
            "name": "cancer_type",
            "classes": {
                "0": "LUSC",
                "1": "LUAD",
            },
            "source_of_label": "TCGA cohort membership",
        },
        "training_scope": {
            "cohorts": ["TCGA-LUAD", "TCGA-LUSC"],
            "patient_count": int(len(df)),
            "luad_count": int((df["target"] == 1).sum()),
            "lusc_count": int((df["target"] == 0).sum()),
            "external_cohorts_used": False,
        },
        "primary_features": primary_features,
        "race_sensitivity_features": race_features,
        "source_field_mapping": selected_fields,
        "models": final_models,
        "validation_warning": (
            "The TCGA-LUAD cohort contains only LUAD labels and the "
            "TCGA-LUSC cohort contains only LUSC labels. Cohort and target "
            "are perfectly confounded. Random train/test splits and "
            "cross-validation cannot establish independent histology "
            "discrimination. All reported metrics are cohort-confounded "
            "exploratory metrics, not estimates of clinical performance."
        ),
        "excluded_predictors": [
            "ethnicity",
            "KPS",
            "ECOG",
            "ICD-O-3 histology",
            "ICD-O-3 site",
            "ICD-10",
            "histologic_diagnosis",
            "histologic_diagnosis__duplicate_2",
            "treatment",
            "response",
            "outcome",
            "pathological_stage",
            "imaging measurements",
            "Brock malignancy probability",
        ],
        "smoking_status_handling": (
            "Codes 1-5 treated as categorical values; not ordinal. "
            "Unknown, undocumented, discrepancy, and unrecognised codes "
            "are treated as missing."
        ),
        "missing_data_handling": (
            "Numeric predictors median-imputed within each fitted pipeline. "
            "Categorical predictors receive a missing category. "
            "No pack-years are inferred from smoking status."
        ),
        "random_state": RANDOM_STATE,
        "scikit_learn_version": sklearn.__version__,
    }

    joblib.dump(artifact, ARTIFACT_PATH)

    manifest = {
        "model_name": artifact["model_name"],
        "model_version": artifact["model_version"],
        "artifact": str(ARTIFACT_PATH),
        "created_utc": artifact["created_utc"],
        "training_cohorts": ["TCGA-LUAD", "TCGA-LUSC"],
        "expected_patient_counts": EXPECTED,
        "observed_patient_counts": {
            "LUAD": int((df["target"] == 1).sum()),
            "LUSC": int((df["target"] == 0).sum()),
        },
        "primary_predictors": primary_features,
        "algorithms": algorithms,
        "race_analysis": "sensitivity-only",
        "validation": "internal metrics; cohort-confounded",
        "clinical_use": False,
        "api_integration": False,
        "existing_models_modified": False,
        "existing_api_files_modified": False,
        "metrics_file": str(METRICS_PATH),
        "parser_reconciliation_file": str(RECONCILIATION_PATH),
        "missingness_file": str(MISSINGNESS_PATH),
        "warning": artifact["validation_warning"],
    }

    schema = {
        "schema_name": "lunginsight_histology_research_v3",
        "schema_version": "1.0.0",
        "description": (
            "Independent research-only input contract for the TCGA-only "
            "LUAD/LUSC histology model. Not connected to the production API."
        ),
        "target": ["LUSC", "LUAD"],
        "primary_predictors": {
            "age_at_diagnosis": "number; years",
            "sex": "categorical",
            "smoking_status": (
                "categorical; source codes 1-5; never ordinal"
            ),
            "pack_years": (
                "number; source-recorded pack-years; never inferred"
            ),
        },
        "sensitivity_only": {
            "race_sensitivity": "categorical; excluded from primary model"
        },
        "forbidden_predictors": list(artifact["excluded_predictors"]),
        "missing_values": "preserved and handled within model pipelines",
        "clinical_use": False,
        "api_integration": False,
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )
    SCHEMA_PATH.write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8"
    )

    summary = []
    summary.append("LUNGINSIGHT HISTOLOGY RESEARCH MODEL V3")
    summary.append("=" * 60)
    summary.append(f"Created UTC: {artifact['created_utc']}")
    summary.append("")
    summary.append("CORRECTED TCGA PATIENT RECONCILIATION")
    summary.append(f"LUAD valid patients: {len(luad_records)} / {EXPECTED['LUAD']}")
    summary.append(f"LUSC valid patients: {len(lusc_records)} / {EXPECTED['LUSC']}")
    summary.append(f"Combined patients: {len(df)} / {EXPECTED_TOTAL}")
    summary.append("")
    summary.append("PARSER EXCLUSIONS")
    summary.append(
        f"LUAD metadata rows: {luad_recon['metadata_rows_excluded']}"
    )
    summary.append(
        f"LUSC metadata rows: {lusc_recon['metadata_rows_excluded']}"
    )
    summary.append(
        "Invalid/blank identity rows: "
        f"{luad_recon['invalid_or_blank_identity_rows_excluded'] + lusc_recon['invalid_or_blank_identity_rows_excluded']}"
    )
    summary.append("")
    summary.append("SELECTED SOURCE FIELDS")
    summary.append(json.dumps(selected_fields, indent=2))
    summary.append("")
    summary.append("METRIC INTERPRETATION")
    summary.append(artifact["validation_warning"])
    summary.append("")
    summary.append("ARTIFACT")
    summary.append(str(ARTIFACT_PATH))
    summary.append("")
    summary.append("OUTPUT FILES")
    summary.append(str(METRICS_PATH))
    summary.append(str(RECONCILIATION_PATH))
    summary.append(str(MISSINGNESS_PATH))
    summary.append(str(MANIFEST_PATH))
    summary.append(str(SCHEMA_PATH))
    summary.append("")
    summary.append("API integration: NOT performed.")
    summary.append("Existing v2 and survival artifacts: NOT modified.")

    write_text(SUMMARY_PATH, "\n".join(summary))

    print("\n".join(summary))
    print("\nMODEL METRICS")
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nBUILD FAILED:", str(exc), file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
