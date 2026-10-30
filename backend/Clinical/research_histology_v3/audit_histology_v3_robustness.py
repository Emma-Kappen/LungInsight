import csv
import json
import math
import re
import sys
import traceback
import warnings
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
)

# ============================================================
# Configuration
# ============================================================

ARTIFACT_PATH = Path(sys.argv[1])
LUAD_FILE = Path(sys.argv[2])
LUSC_FILE = Path(sys.argv[3])
OUTPUT = Path(sys.argv[4])

EXPECTED = {"LUAD": 522, "LUSC": 504}
EXPECTED_TOTAL = 1026

BARCODE_RE = re.compile(
    r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$",
    re.IGNORECASE
)

RANDOM_SEEDS = [17, 29, 42, 73, 101]
TEST_SIZE = 0.25

NUMERIC_FEATURES = ["age_at_diagnosis", "pack_years"]
CATEGORICAL_FEATURES = ["sex", "smoking_status"]

PRIMARY_FEATURES = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)

RACE_FEATURE = "race_sensitivity"

lines = []
failures = []
warnings_found = []

packyears_rows = []
smoking_metadata_rows = []
race_rows = []
robustness_rows = []

def emit(message=""):
    message = str(message)
    print(message)
    lines.append(message)

def section(title):
    emit()
    emit("=" * 78)
    emit(title)
    emit("=" * 78)

def warn(message):
    warnings_found.append(message)
    emit("[WARNING] " + message)

def fail(message):
    failures.append(message)
    emit("[FAIL] " + message)

def missing(value):
    if value is None:
        return True

    value = str(value).strip()

    return (
        value == ""
        or value.lower() in {
            "na", "n/a", "unknown", "not reported",
            "not available", "not evaluated",
            "discrepancy", "null", "none", "--"
        }
    )

def numeric(value):
    if missing(value):
        return np.nan

    try:
        result = float(str(value).strip())
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan

def category(value):
    if missing(value):
        return np.nan

    return str(value).strip()

def normalize_smoking(value):
    value = category(value)

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value if value in {"1", "2", "3", "4", "5"} else np.nan

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

def find_header(rows):
    for i, row in enumerate(rows):
        values = [str(x).strip().lower() for x in row]
        if "bcr_patient_barcode" in values:
            return i
    return None

def find_field(headers, candidates, cohort, label):
    lookup = {h.lower(): h for h in headers}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise RuntimeError(
        f"{cohort}: field not found for {label}: {candidates}"
    )

# ============================================================
# Source parser
# ============================================================

def parse_source(path, cohort):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))

    header_index = find_header(rows)

    if header_index is None:
        raise RuntimeError(f"{cohort}: TCGA barcode header not found.")

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
    metadata_rows = []
    invalid = 0
    malformed = 0

    for row in rows[header_index + 1:]:
        if not row or not any(str(x).strip() for x in row):
            continue

        if len(row) != len(headers):
            malformed += 1
            continue

        barcode = str(row[barcode_index]).strip()

        if not BARCODE_RE.fullmatch(barcode):
            joined = " ".join(str(x) for x in row).upper()

            if "CDE_ID:" in joined or "BCR_PATIENT_BARCODE" in joined:
                metadata_rows.append(row)
            else:
                invalid += 1
            continue

        key = barcode.upper()

        if key in seen:
            raise RuntimeError(
                f"{cohort}: duplicate patient barcode encountered."
            )

        seen.add(key)

        records.append({
            headers[i]: row[i].strip()
            for i in range(len(headers))
        })

    if malformed:
        raise RuntimeError(
            f"{cohort}: {malformed} malformed-width rows detected."
        )

    if invalid:
        warn(
            f"{cohort}: {invalid} invalid identity rows excluded."
        )

    if len(records) != EXPECTED[cohort]:
        raise RuntimeError(
            f"{cohort}: expected {EXPECTED[cohort]} valid patients; "
            f"found {len(records)}. Audit stopped."
        )

    emit(
        f"{cohort}: {len(records)} valid patients; "
        f"{len(metadata_rows)} metadata rows excluded."
    )

    return records, headers, metadata_rows

# ============================================================
# Build harmonized audit frame
# ============================================================

def build_frame(records, headers, cohort):
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
            [
                "tobacco_smoking_history_indicator",
                "tobacco_smoking_history"
            ],
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

    output = []

    for record in records:
        output.append({
            "cohort": cohort,
            "age_at_diagnosis": numeric(record.get(mapping["age"])),
            "sex": category(record.get(mapping["sex"])),
            "smoking_status": normalize_smoking(
                record.get(mapping["smoking"])
            ),
            "pack_years": numeric(record.get(mapping["pack_years"])),
            "race_sensitivity": category(record.get(mapping["race"])),
        })

    return pd.DataFrame(output), mapping

# ============================================================
# 1. Pack-years outliers and missingness
# ============================================================

def audit_packyears(df):
    section("1. PACK-YEARS OUTLIERS AND MISSINGNESS BY SMOKING CATEGORY")

    emit(
        "Smoking code interpretation used for grouping:"
    )
    emit("  1 = Never smoker")
    emit("  2 = Current smoker")
    emit("  3 = Former smoker; quit >15 years")
    emit("  4 = Former smoker; quit <=15 years")
    emit("  5 = Former smoker; cessation interval unspecified")
    emit(
        "  These descriptions are treated as supplied working definitions; "
        "source codebook verification is addressed separately."
    )

    for cohort in ["LUAD", "LUSC"]:
        subset = df[df.cohort == cohort]
        emit(f"\n{cohort}")

        for smoking_code in ["1", "2", "3", "4", "5", "[MISSING]"]:
            if smoking_code == "[MISSING]":
                group = subset[subset.smoking_status.isna()]
            else:
                group = subset[
                    subset.smoking_status.astype(str) == smoking_code
                ]

            if len(group) == 0:
                continue

            py = group.pack_years
            observed = py.dropna()

            missing_count = int(py.isna().sum())
            positive = int((observed > 0).sum())
            zero = int((observed == 0).sum())
            negative = int((observed < 0).sum())
            over_100 = int((observed > 100).sum())
            over_150 = int((observed > 150).sum())

            median = float(observed.median()) if len(observed) else np.nan
            mean = float(observed.mean()) if len(observed) else np.nan
            maximum = float(observed.max()) if len(observed) else np.nan

            emit(
                f"  Smoking {smoking_code}: n={len(group)}, "
                f"pack-years missing={missing_count} "
                f"({100*missing_count/len(group):.1f}%), "
                f"observed={len(observed)}"
            )
            emit(
                f"    zero={zero}, negative={negative}, "
                f">100={over_100}, >150={over_150}"
            )

            if len(observed):
                emit(
                    f"    median={median:.2f}, mean={mean:.2f}, "
                    f"max={maximum:.2f}"
                )

            packyears_rows.append({
                "cohort": cohort,
                "smoking_status": smoking_code,
                "patient_count": len(group),
                "pack_years_observed": len(observed),
                "pack_years_missing": missing_count,
                "missing_percent": round(
                    100*missing_count/len(group), 2
                ),
                "zero_count": zero,
                "negative_count": negative,
                "over_100_count": over_100,
                "over_150_count": over_150,
                "median": median,
                "mean": mean,
                "maximum": maximum,
            })

            if smoking_code == "1" and positive:
                warn(
                    f"{cohort}: never-smoker group contains "
                    f"{positive} records with positive pack-years. "
                    "Review source consistency; values were not changed."
                )

            if negative:
                warn(
                    f"{cohort}: negative pack-years detected. "
                    "Values were not clipped or removed."
                )

        # Cohort-level quantile outlier screening.
        values = subset.pack_years.dropna()

        if len(values) >= 4:
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = q3 - q1
            upper_fence = q3 + 1.5 * iqr
            lower_fence = q1 - 1.5 * iqr

            flagged = values[
                (values < lower_fence) | (values > upper_fence)
            ]

            emit(
                f"  IQR screening: lower fence={lower_fence:.2f}, "
                f"upper fence={upper_fence:.2f}, "
                f"flagged={len(flagged)}"
            )

            emit(
                "  IQR flags are review candidates, not automatic errors."
            )

    pd.DataFrame(packyears_rows).to_csv(
        OUTPUT / "robustness_packyears_by_smoking.csv",
        index=False
    )

# ============================================================
# 2. Smoking code metadata comparison
# ============================================================

def audit_smoking_metadata(metadata_by_cohort, mappings):
    section("2. SMOKING-STATUS SOURCE METADATA COMPARISON")

    metadata_descriptions = {}

    for cohort in ["LUAD", "LUSC"]:
        rows = metadata_by_cohort[cohort]
        source_field = mappings[cohort]["smoking"]

        # Find the original column position by matching its unique
        # name against the positional header mapping.
        source_position = mappings[cohort]["smoking_position"]

        emit(f"\n{cohort}")
        emit(f"  Source field: {source_field}")
        emit(f"  Metadata rows available: {len(rows)}")

        values = []

        for index, row in enumerate(rows, start=1):
            value = (
                row[source_position].strip()
                if source_position < len(row)
                else ""
            )

            values.append(value)

            emit(
                f"  Metadata row {index}: "
                f"{value if value else '[blank]'}"
            )

        metadata_descriptions[cohort] = values

    emit("\nCross-cohort metadata comparison:")

    luad_values = metadata_descriptions["LUAD"]
    lusc_values = metadata_descriptions["LUSC"]

    same = (
        len(luad_values) == len(lusc_values)
        and [x.strip().upper() for x in luad_values]
        == [x.strip().upper() for x in lusc_values]
    )

    if same:
        emit(
            "  Metadata values at the smoking-status field position match."
        )
        emit(
            "  This supports positional/source-field consistency, "
            "but does not independently verify the meaning of codes 1-5."
        )
    else:
        warn(
            "Smoking-status metadata values differ between cohorts. "
            "Do not assume code semantics are equivalent without the "
            "matching TCGA data dictionaries."
        )

    emit(
        "\nImportant: if these rows contain CDE identifiers rather than "
        "human-readable value descriptions, this check cannot verify "
        "the semantic definitions of codes 1-5."
    )

    for cohort in ["LUAD", "LUSC"]:
        for index, value in enumerate(metadata_descriptions[cohort], start=1):
            smoking_metadata_rows.append({
                "cohort": cohort,
                "metadata_row": index,
                "smoking_field_metadata_value": value,
            })

    pd.DataFrame(smoking_metadata_rows).to_csv(
        OUTPUT / "robustness_smoking_metadata.csv",
        index=False
    )

# ============================================================
# 3. Race documentation categories
# ============================================================

def audit_race(artifact, df):
    section("3. RACE DOCUMENTATION-CATEGORY SENSITIVITY")

    for cohort in ["LUAD", "LUSC"]:
        subset = df[df.cohort == cohort]

        emit(f"\n{cohort}")

        for value, count in subset.race_sensitivity.value_counts(
            dropna=False
        ).items():
            label = "[MISSING]" if pd.isna(value) else str(value)
            emit(f"  {label}: {int(count)}")

            race_rows.append({
                "cohort": cohort,
                "race_category": label,
                "count": int(count),
                "is_documentation_state": (
                    label.lower() in {
                        "[not available]",
                        "[not evaluated]",
                        "[missing]"
                    }
                )
            })

    model_name = "race_sensitivity__logistic_regression"
    entry = artifact.get("models", {}).get(model_name)

    if not entry:
        warn("Race sensitivity Logistic Regression not found in artifact.")
    else:
        pipeline = entry["pipeline"]
        features = entry["features"]

        source = df[features].copy()

        # Evaluate the fitted race model before and after converting
        # documentation states into true missing values.
        baseline = pipeline.predict_proba(source)[:, 1]

        modified = source.copy()
        race = modified[RACE_FEATURE].astype("object")

        documentation_states = {
            "[not available]",
            "[not evaluated]",
        }

        mask = race.apply(
            lambda x: (
                isinstance(x, str)
                and x.strip().lower() in documentation_states
            )
        )

        affected = int(mask.sum())
        modified.loc[mask, RACE_FEATURE] = np.nan

        altered = pipeline.predict_proba(modified)[:, 1]
        delta = np.abs(altered - baseline)

        emit("\nRace model documentation-state perturbation:")
        emit(f"  Rows with documentation-state labels: {affected}")
        emit(
            f"  Mean absolute probability change: {float(delta.mean()):.6f}"
        )
        emit(
            f"  Maximum absolute probability change: {float(delta.max()):.6f}"
        )
        emit(
            f"  Rows whose predicted class changed: "
            f"{int(((baseline >= 0.5) != (altered >= 0.5)).sum())}"
        )

        emit(
            "  This is a sensitivity diagnostic, not an estimate of "
            "the causal or clinical contribution of race."
        )

        race_rows.append({
            "cohort": "COMBINED",
            "race_category": "documentation_state_perturbation",
            "count": affected,
            "is_documentation_state": True,
            "mean_absolute_probability_change": float(delta.mean()),
            "maximum_absolute_probability_change": float(delta.max()),
            "class_changes": int(
                ((baseline >= 0.5) != (altered >= 0.5)).sum()
            ),
        })

    pd.DataFrame(race_rows).to_csv(
        OUTPUT / "robustness_race_documentation.csv",
        index=False
    )

# ============================================================
# 4. Alternative imputation and missingness-indicator stability
# ============================================================

def make_encoder():
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

def build_pipeline(algorithm, variant):
    """
    Variants:
      baseline:
        Numeric median imputation + scaling.
        Categorical explicit missing category.

      numeric_missing_indicators:
        Numeric median imputation + missingness indicators.
        Categorical explicit missing category.

      categorical_mode:
        Numeric median imputation.
        Categorical most-frequent imputation.

      all_missing_indicators:
        Numeric median + indicators.
        Categorical explicit missing category + indicators.
    """

    numeric_indicator = variant in {
        "numeric_missing_indicators",
        "all_missing_indicators",
    }

    categorical_indicator = variant == "all_missing_indicators"

    numeric_steps = [
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=numeric_indicator
        )),
        ("scaler", StandardScaler()),
    ]

    if variant == "categorical_mode":
        categorical_imputer = SimpleImputer(
            strategy="most_frequent"
        )
    else:
        categorical_imputer = SimpleImputer(
            strategy="constant",
            fill_value="__MISSING__",
            add_indicator=categorical_indicator
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", categorical_imputer),
        ("onehot", make_encoder()),
    ])

    transformer = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    if algorithm == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline([
        ("preprocessor", transformer),
        ("classifier", estimator),
    ])

def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
    }

def audit_robustness(df):
    section("4. ALTERNATIVE IMPUTATION AND MISSINGNESS-INDICATOR STABILITY")

    emit(
        "All models below are fitted in memory for audit comparison only."
    )
    emit(
        "They do not replace or overwrite the saved v3 artifact."
    )
    emit(
        "All metrics remain cohort-confounded exploratory metrics."
    )

    variants = [
        "baseline",
        "numeric_missing_indicators",
        "categorical_mode",
        "all_missing_indicators",
    ]

    algorithms = [
        "logistic_regression",
        "random_forest",
    ]

    X = df[PRIMARY_FEATURES].copy()
    y = (df.cohort == "LUAD").astype(int).to_numpy()

    for variant in variants:
        for algorithm in algorithms:
            fold_metrics = []

            for seed in RANDOM_SEEDS:
                train_idx, test_idx = train_test_split(
                    np.arange(len(df)),
                    test_size=TEST_SIZE,
                    random_state=seed,
                    stratify=y,
                )

                model = build_pipeline(algorithm, variant)

                model.fit(
                    X.iloc[train_idx],
                    y[train_idx]
                )

                probabilities = model.predict_proba(
                    X.iloc[test_idx]
                )[:, 1]

                metrics = calculate_metrics(
                    y[test_idx],
                    probabilities
                )

                fold_metrics.append(metrics)

            row = {
                "variant": variant,
                "algorithm": algorithm,
                "repetitions": len(fold_metrics),
            }

            for metric in [
                "roc_auc",
                "average_precision",
                "balanced_accuracy",
                "accuracy",
            ]:
                values = [x[metric] for x in fold_metrics]
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1))

            robustness_rows.append(row)

            emit(
                f"\n{algorithm} | {variant} | "
                f"repetitions={len(fold_metrics)}"
            )

            emit(
                f"  ROC-AUC: "
                f"{row['roc_auc_mean']:.4f} "
                f"+/- {row['roc_auc_std']:.4f}"
            )
            emit(
                f"  Average precision: "
                f"{row['average_precision_mean']:.4f} "
                f"+/- {row['average_precision_std']:.4f}"
            )
            emit(
                f"  Balanced accuracy: "
                f"{row['balanced_accuracy_mean']:.4f} "
                f"+/- {row['balanced_accuracy_std']:.4f}"
            )
            emit(
                f"  Accuracy: "
                f"{row['accuracy_mean']:.4f} "
                f"+/- {row['accuracy_std']:.4f}"
            )

    result_df = pd.DataFrame(robustness_rows)
    result_df.to_csv(
        OUTPUT / "robustness_imputation_comparison.csv",
        index=False
    )

    # Compare baseline with alternative handling.
    section("4B. IMPUTATION SENSITIVITY SUMMARY")

    for algorithm in algorithms:
        subset = result_df[result_df.algorithm == algorithm]
        baseline = subset[subset.variant == "baseline"]

        if baseline.empty:
            continue

        baseline_auc = float(baseline.iloc[0].roc_auc_mean)

        emit(f"\n{algorithm}")

        for _, row in subset.iterrows():
            difference = float(row.roc_auc_mean) - baseline_auc

            emit(
                f"  {row.variant}: ROC-AUC delta from baseline "
                f"{difference:+.4f}"
            )

# ============================================================
# 5. Model card
# ============================================================

def write_model_card(artifact, df, mappings):
    section("5. RESEARCH MODEL CARD")

    created = datetime.now(timezone.utc).isoformat()

    card = f"""# LungInsight Histology Research Model v3

## Model identification

- Model: {artifact.get('model_name')}
- Version: research-v3
- Audit generated: {created}
- Artifact: {ARTIFACT_PATH.name}
- Framework: scikit-learn {sklearn.__version__}
- Intended status: research-only exploratory model

## Intended task

Exploratory binary classification of the TCGA-LUAD and TCGA-LUSC
source cohorts.

The model target was assigned from cohort membership:
- LUAD cohort -> LUAD
- LUSC cohort -> LUSC

## Training population

- TCGA-LUAD: {int((df.cohort == 'LUAD').sum())}
- TCGA-LUSC: {int((df.cohort == 'LUSC').sum())}
- Total: {len(df)}

No MSK-CHORD, LIDC-IDRI, or other cohort was used for this model.

## Primary predictors

- Age at initial pathologic diagnosis
- Sex (`gender`)
- Tobacco smoking history indicator
- Tobacco smoking pack-years

Race is excluded from the primary model and was evaluated only as a
sensitivity feature.

## Excluded variables

- Ethnicity
- KPS
- ECOG
- ICD-O-3 histology
- ICD-O-3 site
- ICD-10
- Histologic diagnosis fields
- Treatment, response, and outcome variables
- Pathological stage
- Imaging measurements
- Brock malignancy probability

## Missing data

The original fitted pipelines use median imputation for numeric features
and an explicit missing category for categorical features.

Pack-years are not inferred from smoking status. Missing values remain
missing until handled by the fitted pipeline.

## Major data limitations

The two source cohorts are single-class:
- TCGA-LUAD contains LUAD labels only.
- TCGA-LUSC contains LUSC labels only.

Cohort identity and target are therefore perfectly confounded.

Random train/test splits and cross-validation cannot establish
independent histology discrimination. Metrics describe separation of
these supplied cohorts and must not be interpreted as estimates of
generalisation to an independent clinical population.

Pack-years have differential missingness between the cohorts. This
requires consideration when interpreting model behaviour.

Race documentation categories such as Not Available and Not Evaluated
are data documentation states, not biological or demographic categories.
Race is not part of the primary model.

## Intended use

Permitted:
- Technical research
- Exploratory analysis of the supplied TCGA cohorts
- Software pipeline testing
- Investigation of feature and missing-data sensitivity

Not permitted:
- Clinical diagnosis
- Clinical decision support
- Patient-level treatment decisions
- Clinical risk stratification
- Claims of independent histology discrimination
- Claims of validated performance in external populations

## Deployment status

This model is not integrated into the LungInsight API.

The existing v2 histology model and survival model remain separate and
unchanged.

## Validation status

Technical artifact loading and input handling were previously audited.
This robustness audit evaluates:
- Pack-years distribution and missingness
- Smoking source metadata consistency
- Race documentation-state sensitivity
- Alternative imputation configurations

These checks do not resolve cohort-target confounding.

## Required future evidence

Before any clinical use could be considered, an independent validation
design with both histology classes represented across the relevant
source populations would be required. Source harmonization, temporal
eligibility, calibration, and clinical applicability would also need
separate evaluation.

## Responsible interpretation

This is an exploratory cohort-discrimination research artifact, not a
clinically validated histology classifier.
"""

    path = OUTPUT / "MODEL_CARD_HISTOLOGY_RESEARCH_V3.md"
    path.write_text(card, encoding="utf-8")

    emit(f"Model card written: {path}")

# ============================================================
# Main
# ============================================================

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    section("LUNGINSIGHT HISTOLOGY V3 - FINAL ROBUSTNESS AUDIT")
    emit(f"Artifact: {ARTIFACT_PATH}")
    emit(f"Output: {OUTPUT}")
    emit("Saved model artifact will not be modified.")
    emit("Existing v2 and survival artifacts will not be modified.")

    section("0. LOAD EXISTING ARTIFACT AND SOURCE DATA")

    artifact = joblib.load(ARTIFACT_PATH)

    if not isinstance(artifact, dict):
        raise RuntimeError("Unexpected v3 artifact structure.")

    luad_records, luad_headers, luad_metadata = parse_source(
        LUAD_FILE, "LUAD"
    )
    lusc_records, lusc_headers, lusc_metadata = parse_source(
        LUSC_FILE, "LUSC"
    )

    luad_df, luad_mapping = build_frame(
        luad_records, luad_headers, "LUAD"
    )
    lusc_df, lusc_mapping = build_frame(
        lusc_records, lusc_headers, "LUSC"
    )

    df = pd.concat([luad_df, lusc_df], ignore_index=True)

    if len(df) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} records; found {len(df)}."
        )

    metadata_by_cohort = {
        "LUAD": luad_metadata,
        "LUSC": lusc_metadata,
    }

    # Find the original source-column positions for metadata comparison.
    def original_position(headers, field):
        # Headers are unique positional names. Resolve the original
        # field name without changing the underlying TSV positions.
        for index, header in enumerate(headers):
            if header == field:
                return index
        raise RuntimeError(f"Cannot resolve positional field {field}")

    mappings = {
        "LUAD": luad_mapping,
        "LUSC": lusc_mapping,
    }

    for cohort, headers in [
        ("LUAD", luad_headers),
        ("LUSC", lusc_headers),
    ]:
        mappings[cohort]["smoking_position"] = original_position(
            headers, mappings[cohort]["smoking"]
        )

    emit(f"Combined valid patients: {len(df)}")
    emit("Cohort and target are perfectly confounded.")

    audit_packyears(df)
    audit_smoking_metadata(metadata_by_cohort, mappings)
    audit_race(artifact, df)
    audit_robustness(df)
    write_model_card(artifact, df, mappings)

    section("FINAL ROBUSTNESS AUDIT SUMMARY")

    if failures:
        emit(f"FAILURES: {len(failures)}")
        for item in failures:
            emit("  - " + item)
    else:
        emit("FAILURES: 0")

    if warnings_found:
        emit(f"WARNINGS: {len(warnings_found)}")
        for item in warnings_found:
            emit("  - " + item)
    else:
        emit("WARNINGS: 0")

    emit("")
    emit("IMPORTANT LIMITATION:")
    emit(
        "All model comparison metrics are cohort-confounded exploratory "
        "metrics. They do not establish independent histology discrimination."
    )

    emit("")
    emit("OUTPUT FILES:")
    for name in [
        "robustness_packyears_by_smoking.csv",
        "robustness_smoking_metadata.csv",
        "robustness_race_documentation.csv",
        "robustness_imputation_comparison.csv",
        "MODEL_CARD_HISTOLOGY_RESEARCH_V3.md",
        "histology_v3_robustness_audit.txt",
    ]:
        emit(f"  {name}")

    if failures:
        emit("\nOVERALL RESULT: FAIL")
        exit_code = 1
    else:
        emit("\nOVERALL RESULT: COMPLETED")
        exit_code = 0

    (OUTPUT / "histology_v3_robustness_audit.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    sys.exit(exit_code)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nROBUSTNESS AUDIT ABORTED:", str(exc), file=sys.stderr)
        traceback.print_exc()

        try:
            (OUTPUT / "histology_v3_robustness_audit.txt").write_text(
                "\n".join(
                    lines + ["", "ROBUSTNESS AUDIT ABORTED: " + str(exc)]
                ),
                encoding="utf-8"
            )
        except Exception:
            pass

        sys.exit(2)
