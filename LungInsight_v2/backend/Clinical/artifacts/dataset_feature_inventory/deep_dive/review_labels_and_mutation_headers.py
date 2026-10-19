from pathlib import Path
import pandas as pd
import json
import re

root = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2")
deep = root / "backend" / "Clinical" / "artifacts" / "dataset_feature_inventory" / "deep_dive"
msk = root / "datasets" / "msk_chord_2024"
out = deep / "label_and_mutation_header_review.txt"
lines = []

def emit(x=""):
    lines.append(str(x))

def norm(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")

def read_csv_safe(path):
    try:
        return pd.read_csv(path, low_memory=False, dtype=str, sep=None, engine="python")
    except Exception as e:
        emit(f"READ ERROR: {path}: {e}")
        return None

emit("LUNGINSIGHT — LABEL / IDENTIFIER / MUTATION HEADER REVIEW")
emit("=" * 72)

# 1. Existing label and predictor distributions
dist_path = deep / "candidate_label_predictor_distributions.csv"
emit("\n[1] LABEL AND PREDICTOR DISTRIBUTIONS")
if dist_path.exists():
    df = read_csv_safe(dist_path)
    if df is not None:
        emit(f"Columns: {list(df.columns)}")
        for c in df.columns:
            emit(f"  {c}: {df[c].nunique(dropna=False)} distinct values")
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        mask = pd.Series(False, index=df.index)
        for c in text_cols:
            mask |= df[c].fillna("").str.contains(
                r"histolog|cancer_type|diagnos|icd.?o|gender|sex|smok|age|race|ethnic|patient_id|sample_id|label|cohort",
                case=False, regex=True
            )
        subset = df.loc[mask]
        emit(f"Relevant rows: {len(subset)} of {len(df)}")
        emit(subset.head(250).to_string(index=False, max_colwidth=100))
else:
    emit(f"Missing: {dist_path}")

# 2. Identifier uniqueness / duplicate audit
emit("\n[2] IDENTIFIER AUDIT")
audit_path = deep / "source_table_identifier_audit.csv"
if audit_path.exists():
    audit = read_csv_safe(audit_path)
    if audit is not None:
        emit(f"Columns: {list(audit.columns)}")
        emit(audit.to_string(index=False, max_colwidth=120))
else:
    emit(f"Missing: {audit_path}")

# 3. Column dictionary: exact relevant fields and source tables
emit("\n[3] SOURCE COLUMN DICTIONARY — LABELS / PREDICTORS / IDS")
dict_path = deep / "source_table_column_dictionary.csv"
if dict_path.exists():
    d = read_csv_safe(dict_path)
    if d is not None:
        emit(f"Columns: {list(d.columns)}")
        mask = pd.Series(False, index=d.index)
        for c in d.columns:
            s = d[c].fillna("").astype(str)
            mask |= s.str.contains(
                r"histolog|cancer_type|diagnos|icd.?o|gender|sex|smok|age|race|ethnic|patient_id|sample_id|barcode|uuid|tumor_nuclei|tumor_necrosis|purity|tmb",
                case=False, regex=True
            )
        emit(d.loc[mask].head(350).to_string(index=False, max_colwidth=100))
else:
    emit(f"Missing: {dict_path}")

# 4. Inspect raw mutation file starts. Do not parse as a table here.
emit("\n[4] RAW MUTATION FILE FIRST 55 LINES")
mutation_files = [
    msk / "data_mutations.txt",
    msk / "msk_chord_out" / "nsclc_mutations.csv",
]
for path in mutation_files:
    emit(f"\n--- {path} ---")
    if not path.exists():
        emit("FILE NOT FOUND")
        continue
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 55:
                    break
                emit(f"{i+1:03d}: {line.rstrip()[:500]}")
    except Exception as e:
        emit(f"RAW READ ERROR: {e}")

# 5. Check likely clinical sources directly, using detected header line where known.
emit("\n[5] DIRECT LABEL / CLINICAL FIELD VALUE COUNTS")
sources = [
    ("MSK patient", msk / "data_clinical_patient.txt", 4,
     ["PATIENT_ID", "GENDER", "RACE", "ETHNICITY", "CURRENT_AGE_DEID",
      "SMOKING_PREDICTIONS_3_CLASSES", "STAGE_HIGHEST_RECORDED"]),
    ("MSK sample", msk / "data_clinical_sample.txt", 4,
     ["SAMPLE_ID", "PATIENT_ID", "CANCER_TYPE", "CANCER_TYPE_DETAILED",
      "SAMPLE_TYPE", "SAMPLE_CLASS", "PRIMARY_SITE", "ONCOTREE_CODE",
      "ICD_O_HISTOLOGY_DESCRIPTION", "DIAGNOSIS_DESCRIPTION", "TUMOR_PURITY"]),
]
for label, path, skip, wanted in sources:
    emit(f"\n--- {label}: {path.name} ---")
    if not path.exists():
        emit("FILE NOT FOUND")
        continue
    try:
        df = pd.read_csv(path, sep="\t", skiprows=skip, dtype=str, low_memory=False)
        emit(f"Rows={len(df)}; columns={len(df.columns)}")
        emit(f"Columns={list(df.columns)}")
        for col in wanted:
            if col not in df.columns:
                emit(f"{col}: MISSING")
                continue
            s = df[col].fillna("").str.strip().replace("", "<BLANK>")
            emit(f"\n{col}: unique={s.nunique()}; blank={(s == '<BLANK>').sum()}")
            emit(s.value_counts(dropna=False).head(35).to_string())
        idcol = "PATIENT_ID" if "PATIENT_ID" in df.columns else None
        if idcol:
            emit(f"Unique PATIENT_ID={df[idcol].nunique(dropna=True)}; duplicated rows on ID={df[idcol].duplicated(keep=False).sum()}")
    except Exception as e:
        emit(f"TABLE READ ERROR: {e}")

# 6. Save report and print it.
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nREPORT SAVED: {out}")
