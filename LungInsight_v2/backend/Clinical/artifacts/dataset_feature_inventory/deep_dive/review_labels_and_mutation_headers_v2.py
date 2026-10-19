from pathlib import Path
import pandas as pd
import re

root = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2")
deep = root / "backend" / "Clinical" / "artifacts" / "dataset_feature_inventory" / "deep_dive"
inventory_root = root / "backend" / "Clinical" / "artifacts" / "dataset_feature_inventory"
report_path = deep / "label_and_mutation_header_review_v2.txt"
lines = []

def emit(s=""):
    lines.append(str(s))

def read_auto(path, **kwargs):
    # Keep pandas' C parser for CSV/TSV. Detect the delimiter from the extension.
    suffix = path.suffix.lower()
    sep = "\t" if suffix in (".txt", ".tsv") else ","
    kwargs.pop("low_memory", None)
    return pd.read_csv(path, sep=sep, low_memory=False, **kwargs)

def find_files(filename):
    matches = []
    for base in (root / "datasets", inventory_root):
        if base.exists():
            matches.extend(p for p in base.rglob(filename) if p.is_file())
    # preserve order, remove duplicate paths
    return list(dict.fromkeys(matches))

def show_file_search(names):
    for name in names:
        found = find_files(name)
        emit(f"{name}:")
        if found:
            for p in found[:15]:
                emit(f"  {p}")
        else:
            emit("  NOT FOUND under datasets/ or dataset_feature_inventory/")

emit("LUNGINSIGHT — LABEL / IDENTIFIER / MUTATION HEADER REVIEW V2")
emit("=" * 76)

emit("\n[1] VERIFY DATASET ROOTS")
for p in (root / "datasets", inventory_root, deep):
    emit(f"{p}: exists={p.exists()}")
show_file_search([
    "candidate_label_predictor_distributions.csv",
    "source_table_identifier_audit.csv",
    "source_table_column_dictionary.csv",
    "data_clinical_patient.txt",
    "data_clinical_sample.txt",
    "data_mutations.txt",
    "nsclc_mutations.csv",
])

emit("\n[2] LABEL / PREDICTOR DISTRIBUTIONS")
p = deep / "candidate_label_predictor_distributions.csv"
if p.exists():
    try:
        df = read_auto(p, dtype=str)
        emit(f"Rows={len(df)}; columns={list(df.columns)}")
        emit(df.head(100).to_string(index=False, max_colwidth=100))
    except Exception as e:
        emit(f"READ ERROR: {e}")

emit("\n[3] IDENTIFIER AUDIT")
p = deep / "source_table_identifier_audit.csv"
if p.exists():
    try:
        df = read_auto(p, dtype=str)
        emit(f"Rows={len(df)}; columns={list(df.columns)}")
        emit(df.to_string(index=False, max_colwidth=120))
    except Exception as e:
        emit(f"READ ERROR: {e}")

emit("\n[4] COLUMN DICTIONARY — LABEL / PREDICTOR / ID FIELDS")
p = deep / "source_table_column_dictionary.csv"
if p.exists():
    try:
        df = read_auto(p, dtype=str)
        emit(f"Rows={len(df)}; columns={list(df.columns)}")
        pattern = re.compile(
            r"histolog|cancer_type|diagnos|icd.?o|gender|sex|smok|age|race|ethnic|patient_id|sample_id|barcode|uuid|tumor_nuclei|tumor_necrosis|purity|tmb",
            re.I
        )
        mask = pd.Series(False, index=df.index)
        for c in df.columns:
            mask |= df[c].fillna("").astype(str).str.contains(pattern, na=False)
        emit(df.loc[mask].head(250).to_string(index=False, max_colwidth=100))
    except Exception as e:
        emit(f"READ ERROR: {e}")

emit("\n[5] DISCOVER AND INSPECT MSK CLINICAL TABLES")
clinical_specs = [
    ("data_clinical_patient.txt", 4,
     ["PATIENT_ID", "GENDER", "RACE", "ETHNICITY", "CURRENT_AGE_DEID",
      "SMOKING_PREDICTIONS_3_CLASSES", "STAGE_HIGHEST_RECORDED"]),
    ("data_clinical_sample.txt", 4,
     ["SAMPLE_ID", "PATIENT_ID", "CANCER_TYPE", "CANCER_TYPE_DETAILED",
      "SAMPLE_TYPE", "SAMPLE_CLASS", "PRIMARY_SITE", "ONCOTREE_CODE",
      "ICD_O_HISTOLOGY_DESCRIPTION", "DIAGNOSIS_DESCRIPTION", "TUMOR_PURITY"]),
]
for filename, skip, wanted in clinical_specs:
    emit(f"\n--- {filename} ---")
    paths = find_files(filename)
    if not paths:
        emit("NOT FOUND")
        continue
    for path in paths[:5]:
        emit(f"PATH: {path}")
        try:
            df = read_auto(path, skiprows=skip, dtype=str)
            emit(f"Rows={len(df)}; columns={len(df.columns)}")
            emit(f"Columns={list(df.columns)}")
            for col in wanted:
                if col not in df.columns:
                    emit(f"{col}: missing")
                    continue
                s = df[col].fillna("").str.strip().replace("", "<BLANK>")
                emit(f"\n{col}: unique={s.nunique()}; blank={(s == '<BLANK>').sum()}")
                emit(s.value_counts(dropna=False).head(25).to_string())
            if "PATIENT_ID" in df.columns:
                emit(f"Unique patients={df['PATIENT_ID'].nunique(dropna=True)}")
                emit(f"Rows with duplicated patient ID={df['PATIENT_ID'].duplicated(keep=False).sum()}")
            if "SAMPLE_ID" in df.columns:
                emit(f"Unique samples={df['SAMPLE_ID'].nunique(dropna=True)}")
                emit(f"Rows with duplicated sample ID={df['SAMPLE_ID'].duplicated(keep=False).sum()}")
        except Exception as e:
            emit(f"READ ERROR: {e}")

emit("\n[6] MUTATION FILE RAW HEADER DISCOVERY")
for filename in ("data_mutations.txt", "nsclc_mutations.csv"):
    emit(f"\n--- {filename} ---")
    paths = find_files(filename)
    if not paths:
        emit("NOT FOUND")
        continue
    for path in paths[:5]:
        emit(f"PATH: {path}")
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= 45:
                        break
                    emit(f"{i+1:03d}: {line.rstrip()[:350]}")
        except Exception as e:
            emit(f"RAW READ ERROR: {e}")

report_path.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nREPORT SAVED: {report_path}")
