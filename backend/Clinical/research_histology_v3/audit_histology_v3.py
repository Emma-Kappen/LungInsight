import csv
import json
import re
import sys
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

ROOT = Path(sys.argv[1])
ARTIFACT_PATH = Path(sys.argv[2])
LUAD_FILE = Path(sys.argv[3])
LUSC_FILE = Path(sys.argv[4])
OUTPUT = Path(sys.argv[5])

EXPECTED = {"LUAD": 522, "LUSC": 504}
PATIENT_BARCODE = re.compile(
    r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$",
    re.IGNORECASE
)

PRIMARY_FEATURES = [
    "age_at_diagnosis",
    "sex",
    "smoking_status",
    "pack_years",
]

RACE_FEATURE = "race_sensitivity"

lines = []
distribution_rows = []
harmonization_rows = []
importance_rows = []
missing_test_rows = []
prediction_rows = []

failures = []
warnings_found = []

def emit(message=""):
    message = str(message)
    print(message)
    lines.append(message)

def section(title):
    emit()
    emit("=" * 76)
    emit(title)
    emit("=" * 76)

def record_failure(message):
    failures.append(message)
    emit("[FAIL] " + message)

def record_warning(message):
    warnings_found.append(message)
    emit("[WARNING] " + message)

def normalize_missing(value):
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

def numeric(value):
    value = normalize_missing(value)
    if pd.isna(value):
        return np.nan

    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (ValueError, TypeError):
        return np.nan

def unique_headers(headers):
    counts = {}
    result = []

    for raw in headers:
        name = str(raw).strip().lstrip("\ufeff") or "unnamed_column"
        key = name.lower()
        counts[key] = counts.get(key, 0) + 1

        if counts[key] == 1:
            result.append(name)
        else:
            result.append(f"{name}__duplicate_{counts[key]}")

    return result

def parse_source(path, cohort):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))

    header_index = None
    for i, row in enumerate(rows):
        if "bcr_patient_barcode" in [
            str(x).strip().lower() for x in row
        ]:
            header_index = i
            break

    if header_index is None:
        raise RuntimeError(f"{cohort}: barcode header not found")

    original_headers = [
        str(x).strip().lstrip("\ufeff")
        for x in rows[header_index]
    ]
    headers = unique_headers(original_headers)

    barcode_index = next(
        i for i, value in enumerate(original_headers)
        if value.lower() == "bcr_patient_barcode"
    )

    records = []
    seen = set()
    metadata = 0
    invalid = 0
    malformed = 0

    for row in rows[header_index + 1:]:
        if not row or not any(str(x).strip() for x in row):
            continue

        if len(row) != len(headers):
            malformed += 1
            continue

        barcode = str(row[barcode_index]).strip()

        if not PATIENT_BARCODE.fullmatch(barcode):
            joined = " ".join(str(x) for x in row).upper()
            if "CDE_ID:" in joined or "BCR_PATIENT_BARCODE" in joined:
                metadata += 1
            else:
                invalid += 1
            continue

        key = barcode.upper()
        if key in seen:
            raise RuntimeError(f"{cohort}: duplicate patient barcode detected")
        seen.add(key)

        records.append({
            headers[i]: row[i].strip()
            for i in range(len(headers))
        })

    if malformed:
        raise RuntimeError(
            f"{cohort}: {malformed} rows have an unexpected field count"
        )

    if invalid:
        record_warning(
            f"{cohort}: {invalid} rows with invalid/blank barcode excluded"
        )

    if len(records) != EXPECTED[cohort]:
        raise RuntimeError(
            f"{cohort}: expected {EXPECTED[cohort]} patients; "
            f"recovered {len(records)}"
        )

    emit(
        f"{cohort}: {len(records)} valid patients; "
        f"{metadata} metadata rows excluded; "
        f"{invalid} invalid identities"
    )

    return records, headers

def find_field(headers, candidates, cohort, description):
    lookup = {h.lower(): h for h in headers}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise RuntimeError(
        f"{cohort}: cannot locate {description}; candidates={candidates}"
    )

def harmonize(records, cohort, headers):
    mapping = {
        "age": find_field(
            headers,
            ["age_at_initial_pathologic_diagnosis"],
            cohort, "age"
        ),
        "sex": find_field(
            headers, ["gender"], cohort, "sex"
        ),
        "smoking": find_field(
            headers,
            ["tobacco_smoking_history_indicator", "tobacco_smoking_history"],
            cohort, "smoking status"
        ),
        "pack_years": find_field(
            headers,
            ["tobacco_smoking_pack_years_smoked"],
            cohort, "pack-years"
        ),
        "race": find_field(
            headers, ["race"], cohort, "race"
        ),
    }

    rows = []

    for record in records:
        age = numeric(record.get(mapping["age"]))
        sex = normalize_missing(record.get(mapping["sex"]))
        smoking = normalize_missing(record.get(mapping["smoking"]))
        pack_years = numeric(record.get(mapping["pack_years"]))
        race = normalize_missing(record.get(mapping["race"]))

        # Match the model-building code's treatment of smoking codes.
        if not pd.isna(smoking):
            smoking = str(smoking).strip()
            if smoking.endswith(".0"):
                smoking = smoking[:-2]
            if smoking not in {"1", "2", "3", "4", "5"}:
                smoking = np.nan

        rows.append({
            "cohort": cohort,
            "age_at_diagnosis": age,
            "sex": sex,
            "smoking_status": smoking,
            "pack_years": pack_years,
            "race_sensitivity": race,
        })

    return pd.DataFrame(rows), mapping

def distribution_report(df):
    section("1. CORRECTED FEATURE DISTRIBUTIONS AND MISSINGNESS")

    features = PRIMARY_FEATURES + [RACE_FEATURE]

    for cohort in ["LUAD", "LUSC"]:
        subset = df[df["cohort"] == cohort]
        emit(f"\n{cohort} — n={len(subset)}")

        for feature in features:
            series = subset[feature]
            missing = int(series.isna().sum())
            observed = int(series.notna().sum())
            missing_pct = 100 * missing / len(subset)

            emit(
                f"  {feature}: observed={observed}, "
                f"missing={missing} ({missing_pct:.1f}%)"
            )

            distribution_rows.append({
                "cohort": cohort,
                "feature": feature,
                "n": len(subset),
                "observed": observed,
                "missing": missing,
                "missing_percent": round(missing_pct, 2),
                "unique_observed_values": int(series.nunique(dropna=True)),
            })

            if feature in ["age_at_diagnosis", "pack_years"]:
                if observed:
                    emit(
                        f"    min={series.min():.3f}, "
                        f"median={series.median():.3f}, "
                        f"mean={series.mean():.3f}, "
                        f"max={series.max():.3f}"
                    )
            else:
                counts = series.value_counts(dropna=False)
                for value, count in counts.items():
                    label = "[MISSING]" if pd.isna(value) else str(value)
                    emit(f"    {label}: {int(count)}")

    pd.DataFrame(distribution_rows).to_csv(
        OUTPUT / "audit_corrected_distributions.csv",
        index=False
    )

def compare_harmonization(df, mappings):
    section("2. CROSS-COHORT FEATURE HARMONIZATION")

    for feature in PRIMARY_FEATURES + [RACE_FEATURE]:
        luad = df.loc[df.cohort == "LUAD", feature].dropna()
        lusc = df.loc[df.cohort == "LUSC", feature].dropna()

        if feature in ["age_at_diagnosis", "pack_years"]:
            luad_values = set(np.round(luad.astype(float), 4))
            lusc_values = set(np.round(lusc.astype(float), 4))
        else:
            luad_values = set(luad.astype(str))
            lusc_values = set(lusc.astype(str))

        shared = luad_values.intersection(lusc_values)
        only_luad = luad_values - lusc_values
        only_lusc = lusc_values - luad_values

        emit(f"\n{feature}")
        emit(f"  LUAD source field: {mappings['LUAD'][feature_key(feature)]}")
        emit(f"  LUSC source field: {mappings['LUSC'][feature_key(feature)]}")
        emit(f"  Shared observed values: {len(shared)}")
        emit(f"  Values only in LUAD: {len(only_luad)}")
        emit(f"  Values only in LUSC: {len(only_lusc)}")

        if feature in ["sex", "smoking_status", RACE_FEATURE]:
            emit(f"  LUAD categories: {sorted(map(str, luad_values))}")
            emit(f"  LUSC categories: {sorted(map(str, lusc_values))}")

        harmonization_rows.append({
            "feature": feature,
            "luad_source_field": mappings["LUAD"][feature_key(feature)],
            "lusc_source_field": mappings["LUSC"][feature_key(feature)],
            "shared_observed_values": len(shared),
            "values_only_luad": len(only_luad),
            "values_only_lusc": len(only_lusc),
            "luad_unique_values": len(luad_values),
            "lusc_unique_values": len(lusc_values),
        })

        if feature in ["sex", "smoking_status"] and only_luad.union(only_lusc):
            record_warning(
                f"{feature}: category sets differ between cohorts; "
                "review source definitions before interpreting the model."
            )

    pd.DataFrame(harmonization_rows).to_csv(
        OUTPUT / "audit_cross_cohort_harmonization.csv",
        index=False
    )

def feature_key(feature):
    return {
        "age_at_diagnosis": "age",
        "sex": "sex",
        "smoking_status": "smoking",
        "pack_years": "pack_years",
        "race_sensitivity": "race",
    }[feature]

def get_transformed_names(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []

def audit_importance(artifact):
    section("3. MODEL COEFFICIENTS AND FEATURE IMPORTANCE")

    models = artifact.get("models", {})
    if not models:
        record_failure("Artifact contains no models dictionary.")
        return

    for model_name, entry in models.items():
        emit(f"\n{model_name}")

        pipeline = entry.get("pipeline")
        if not isinstance(pipeline, Pipeline):
            record_failure(f"{model_name}: pipeline object is invalid.")
            continue

        classifier = pipeline.named_steps.get("classifier")
        transformed_names = get_transformed_names(pipeline)

        if classifier is None:
            record_failure(f"{model_name}: classifier step missing.")
            continue

        if hasattr(classifier, "coef_"):
            coefficients = np.asarray(classifier.coef_)
            if coefficients.ndim == 2:
                coefficients = coefficients[0]

            if len(transformed_names) != len(coefficients):
                record_warning(
                    f"{model_name}: coefficient names do not match "
                    "transformed feature count."
                )

            emit("  Logistic Regression coefficients:")
            order = np.argsort(np.abs(coefficients))[::-1]

            for index in order:
                name = (
                    transformed_names[index]
                    if index < len(transformed_names)
                    else f"transformed_feature_{index}"
                )
                value = float(coefficients[index])
                emit(f"    {name}: {value:+.6f}")

                importance_rows.append({
                    "model": model_name,
                    "importance_type": "logistic_regression_coefficient",
                    "feature": name,
                    "value": value,
                    "absolute_value": abs(value),
                })

        elif hasattr(classifier, "feature_importances_"):
            importances = np.asarray(classifier.feature_importances_)

            emit("  Random Forest feature importances:")
            order = np.argsort(importances)[::-1]

            for index in order:
                name = (
                    transformed_names[index]
                    if index < len(transformed_names)
                    else f"transformed_feature_{index}"
                )
                value = float(importances[index])
                emit(f"    {name}: {value:.6f}")

                importance_rows.append({
                    "model": model_name,
                    "importance_type": "random_forest_impurity_importance",
                    "feature": name,
                    "value": value,
                    "absolute_value": abs(value),
                })
        else:
            record_warning(
                f"{model_name}: estimator does not expose coefficients "
                "or feature_importances_."
            )

    pd.DataFrame(importance_rows).to_csv(
        OUTPUT / "audit_model_importance.csv",
        index=False
    )

def make_sample(df, feature_set):
    features = list(feature_set)
    sample = df[features].copy()

    # Ensure representative rows from both source cohorts.
    sample = pd.concat([
        sample[df["cohort"].values == "LUAD"].head(8),
        sample[df["cohort"].values == "LUSC"].head(8),
    ], ignore_index=True)

    return sample

def finite_probabilities(pipeline, X):
    probabilities = pipeline.predict_proba(X)
    predictions = pipeline.predict(X)

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise RuntimeError(
            f"Expected binary probability matrix; got {probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Predicted probabilities contain NaN or infinity.")

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Predicted class probabilities do not sum to one.")

    if not set(np.unique(predictions)).issubset({0, 1}):
        raise RuntimeError("Predicted labels are outside the expected class set.")

    return probabilities, predictions

def audit_missingness(artifact, df):
    section("4. MISSING-VALUE IMPUTATION AND PREPROCESSING TESTS")

    models = artifact.get("models", {})

    for model_name, entry in models.items():
        pipeline = entry["pipeline"]
        features = list(entry["features"])

        sample = make_sample(df, features)

        emit(f"\n{model_name}")

        try:
            baseline_prob, baseline_pred = finite_probabilities(
                pipeline, sample
            )
            emit("  Baseline prediction: PASS")
        except Exception as exc:
            record_failure(f"{model_name}: baseline prediction failed: {exc}")
            continue

        # Test each predictor missing in one row.
        for feature in features:
            test = sample.copy()
            test.loc[test.index[0], feature] = np.nan

            try:
                probabilities, predictions = finite_probabilities(
                    pipeline, test
                )
                status = "PASS"
                detail = "one-row missing value handled"
            except Exception as exc:
                status = "FAIL"
                detail = str(exc)
                record_failure(
                    f"{model_name}: single missing {feature} failed: {exc}"
                )

            emit(f"  Single missing {feature}: {status} — {detail}")

            missing_test_rows.append({
                "model": model_name,
                "test": "single_row_missing",
                "feature": feature,
                "status": status,
                "detail": detail,
            })

        # Test an entire feature column missing in the inference batch.
        # The fitted imputer must use training-time statistics/categories.
        for feature in features:
            test = sample.copy()
            test[feature] = np.nan

            try:
                probabilities, predictions = finite_probabilities(
                    pipeline, test
                )
                status = "PASS"
                detail = "all-missing inference column handled"
            except Exception as exc:
                status = "FAIL"
                detail = str(exc)
                record_failure(
                    f"{model_name}: all-missing {feature} failed: {exc}"
                )

            emit(f"  Entire batch missing {feature}: {status} — {detail}")

            missing_test_rows.append({
                "model": model_name,
                "test": "all_batch_missing",
                "feature": feature,
                "status": status,
                "detail": detail,
            })

        # Verify the pipeline did not mutate the caller's input.
        original = sample.copy(deep=True)
        check = sample.copy(deep=True)

        try:
            pipeline.predict_proba(check)
            if not check.equals(original):
                record_failure(
                    f"{model_name}: inference modified the input dataframe."
                )
            else:
                emit("  Input immutability: PASS")
        except Exception as exc:
            record_failure(
                f"{model_name}: input immutability check failed: {exc}"
            )

    pd.DataFrame(missing_test_rows).to_csv(
        OUTPUT / "audit_missing_value_tests.csv",
        index=False
    )

def audit_artifact(artifact, df):
    section("5. STANDALONE ARTIFACT LOAD AND PREDICTION TEST")

    emit(f"Artifact: {ARTIFACT_PATH}")
    emit(f"scikit-learn runtime: {sklearn.__version__}")
    emit(f"Artifact version: {artifact.get('model_version')}")
    emit(f"Artifact name: {artifact.get('model_name')}")

    expected_models = {
        "primary_four_predictors__logistic_regression",
        "primary_four_predictors__random_forest",
        "race_sensitivity__logistic_regression",
        "race_sensitivity__random_forest",
    }

    actual_models = set(artifact.get("models", {}).keys())

    emit(f"Expected model entries: {len(expected_models)}")
    emit(f"Found model entries: {len(actual_models)}")

    if expected_models != actual_models:
        record_failure(
            "Artifact model keys do not match the four expected model variants."
        )

    for model_name, entry in artifact.get("models", {}).items():
        pipeline = entry.get("pipeline")
        features = entry.get("features", [])

        emit(f"\n{model_name}")
        emit(f"  Features: {features}")

        if not isinstance(pipeline, Pipeline):
            record_failure(f"{model_name}: loaded pipeline is not sklearn Pipeline.")
            continue

        if not features:
            record_failure(f"{model_name}: no input feature list.")
            continue

        sample = make_sample(df, features)

        try:
            probabilities, predictions = finite_probabilities(
                pipeline, sample
            )

            emit(f"  Loaded pipeline: PASS")
            emit(f"  Input rows tested: {len(sample)}")
            emit(f"  Output shape: {probabilities.shape}")
            emit(
                "  First probability vector: "
                + np.array2string(probabilities[0], precision=5)
            )
            emit(
                "  First predicted class: "
                + str(predictions[0])
            )

            prediction_rows.append({
                "model": model_name,
                "status": "PASS",
                "rows_tested": len(sample),
                "output_columns": probabilities.shape[1],
                "unique_predictions": ",".join(
                    map(str, sorted(np.unique(predictions)))
                ),
            })

        except Exception as exc:
            record_failure(
                f"{model_name}: standalone prediction failed: {exc}"
            )
            prediction_rows.append({
                "model": model_name,
                "status": "FAIL",
                "rows_tested": len(sample),
                "output_columns": None,
                "unique_predictions": str(exc),
            })

    pd.DataFrame(prediction_rows).to_csv(
        OUTPUT / "audit_standalone_predictions.csv",
        index=False
    )

def main():
    section("LUNGINSIGHT HISTOLOGY V3 — READ-ONLY AUDIT")
    emit(f"Artifact: {ARTIFACT_PATH}")
    emit(f"Output directory: {OUTPUT}")
    emit("No model training or application integration will be performed.")

    section("0. LOAD ARTIFACT AND VERIFY SOURCE ROWS")

    # Load only the independent v3 artifact.
    artifact = joblib.load(ARTIFACT_PATH)

    if not isinstance(artifact, dict):
        raise RuntimeError("Loaded artifact is not the expected dictionary.")

    luad_records, luad_headers = parse_source(LUAD_FILE, "LUAD")
    lusc_records, lusc_headers = parse_source(LUSC_FILE, "LUSC")

    luad_df, luad_mapping = harmonize(
        luad_records, "LUAD", luad_headers
    )
    lusc_df, lusc_mapping = harmonize(
        lusc_records, "LUSC", lusc_headers
    )

    df = pd.concat([luad_df, lusc_df], ignore_index=True)

    mappings = {
        "LUAD": luad_mapping,
        "LUSC": lusc_mapping,
    }

    emit(f"Combined source patients: {len(df)}")
    if len(df) != sum(EXPECTED.values()):
        raise RuntimeError("Combined patient count failed.")

    distribution_report(df)
    compare_harmonization(df, mappings)
    audit_importance(artifact)
    audit_missingness(artifact, df)
    audit_artifact(artifact, df)

    section("FINAL AUDIT SUMMARY")

    if failures:
        emit(f"FAILURES: {len(failures)}")
        for failure in failures:
            emit("  - " + failure)
    else:
        emit("FAILURES: 0")

    if warnings_found:
        emit(f"WARNINGS: {len(warnings_found)}")
        for warning in warnings_found:
            emit("  - " + warning)
    else:
        emit("WARNINGS: 0")

    emit("")
    emit("Interpretation:")
    emit(
        "All prediction tests are technical artifact/input-handling checks. "
        "They do not establish clinical validity or independent histology "
        "discrimination."
    )

    if failures:
        emit("OVERALL RESULT: FAIL — investigate the reported failures.")
        exit_code = 1
    else:
        emit("OVERALL RESULT: PASS — no technical audit failures detected.")
        exit_code = 0

    (OUTPUT / "histology_v3_standalone_audit.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("\nAudit reports written to:", OUTPUT)
    sys.exit(exit_code)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nAUDIT ABORTED:", str(exc), file=sys.stderr)
        traceback.print_exc()
        try:
            (OUTPUT / "histology_v3_standalone_audit.txt").write_text(
                "\n".join(lines + ["", "AUDIT ABORTED: " + str(exc)]),
                encoding="utf-8"
            )
        except Exception:
            pass
        sys.exit(2)
