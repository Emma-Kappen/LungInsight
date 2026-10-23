from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "compiled_source_dataset.csv", dtype=str, keep_default_na=False)

def show_counts(column):
    print(f"\n=== {column}: by cohort ===")
    if column not in df.columns:
        print("COLUMN MISSING")
        return
    print(pd.crosstab(
        df["COHORT"],
        df[column].replace("", "<EMPTY>"),
        margins=True
    ).to_string())

print("=== ROW COUNTS ===")
print(df.groupby(["COHORT", "LABEL"]).size().to_string())

for col in [
    "smoking_status",
    "smoking_status_source_raw",
    "sex",
    "performance_status_source",
    "performance_status_ecog",
    "performance_status_kps",
    "vital_status",
    "age_quality_flag",
]:
    show_counts(col)

print("\n=== NUMERIC FIELD COMPLETENESS BY COHORT ===")
numeric_cols = [
    "age_at_diagnosis_years",
    "smoking_duration_years",
    "years_since_smoking_cessation",
    "pack_years",
    "packs_per_day",
    "performance_status_ecog",
    "performance_status_kps",
    "days_to_death",
]
for col in numeric_cols:
    if col not in df:
        continue
    vals = pd.to_numeric(df[col].replace("", pd.NA), errors="coerce")
    tmp = df[["COHORT", "LABEL"]].copy()
    tmp["value"] = vals
    print(f"\n{col}")
    print(tmp.groupby(["COHORT", "LABEL"])["value"].agg(
        total="size", observed="count", median="median", minimum="min", maximum="max"
    ).to_string())

print("\n=== TCGA RAW SOURCE CATEGORY SAMPLE ===")
tcga = df[df["COHORT"].isin(["TCGA-LUAD", "TCGA-LUSC"])]
print(tcga[[
    "COHORT", "PATIENT_ID", "LABEL", "age_at_diagnosis_years",
    "sex", "smoking_status_source_raw", "smoking_status",
    "performance_status_ecog", "performance_status_kps",
    "vital_status", "days_to_death"
]].head(12).to_string(index=False))

print("\nAudit only. No source data or model artifacts modified.")
