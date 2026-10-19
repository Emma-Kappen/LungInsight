from pathlib import Path
import pandas as pd

root = Path(__import__("os").environ["LI_BUILD_DIR"])
src = pd.read_csv(root / "compiled_source_dataset.csv", dtype=str, keep_default_na=False)
matrix = pd.read_csv(root / "diagnosis_time_model_matrix.csv", dtype=str, keep_default_na=False)

print("=== ROWS BY COHORT AND LABEL ===")
print(src.groupby(["COHORT", "LABEL"]).size().to_string())

print("\n=== DETAILED SMOKING STATUS BY COHORT ===")
print(pd.crosstab(src["COHORT"], src["smoking_status_detailed"], margins=True).to_string())

print("\n=== RAW VALUES BY COHORT ===")
print(pd.crosstab(src["COHORT"], src["smoking_status_source_raw"], margins=True).to_string())

print("\n=== SOURCE CODES BY COHORT ===")
print(pd.crosstab(src["COHORT"], src["smoking_status_source_code"], margins=True).to_string())

print("\n=== MODEL MATRIX SMOKING COLUMNS ===")
print([c for c in matrix.columns if "smoking" in c.lower()])

assert len(src) == 5345, f"Unexpected source row count: {len(src)}"
assert "smoking_status_detailed" in src.columns
assert "smoking_status_detailed" in matrix.columns
assert "smoking_status_source_raw" in src.columns
assert (src.loc[src["COHORT"].str.startswith("TCGA-"), "smoking_status_detailed"] != "Unknown").any(), \
    "No TCGA smoking codes were decoded."
assert not src["smoking_status_source_raw"].isna().any()

print("\nVerification passed. No model was trained.")
