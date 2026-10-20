from pathlib import Path
import pandas as pd

root = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\datasets")
out = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2\backend\Clinical\artifacts\cancer_type_dataset")

# Find likely raw MSK clinical/sample TSVs; exclude generated dataset outputs.
files = [
    p for p in root.rglob("*.tsv")
    if "msk" in str(p).lower() or "chord" in str(p).lower()
]
files += [
    p for p in root.rglob("*.txt")
    if "msk" in str(p).lower() or "chord" in str(p).lower()
]
files = sorted(set(files))

wanted = {
    "Cancer Type", "Cancer Type Detailed", "Oncotree Code",
    "ICD-O Histology", "Sample ID", "Patient ID"
}
matches = []

for path in files:
    try:
        header = pd.read_csv(path, sep="\t", nrows=0, dtype=str)
        cols = set(map(str, header.columns))
        useful = cols & wanted
        if {"Cancer Type Detailed", "Oncotree Code"} & useful:
            matches.append((path, list(header.columns)))
    except Exception:
        continue

if not matches:
    raise SystemExit(
        "No matching raw MSK TSV found. Locate the original MSK clinical/sample "
        "TSV and update the script's root or file filter."
    )

for path, columns in matches:
    relevant = [c for c in columns if c in wanted]
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, usecols=relevant, low_memory=False)
    except Exception:
        continue

    if not ({"Cancer Type Detailed", "Oncotree Code"} & set(df.columns)):
        continue

    print(f"\n===== FILE: {path.name} =====")
    print(f"Rows: {len(df):,}")

    for col in ("Cancer Type", "Cancer Type Detailed", "Oncotree Code", "ICD-O Histology"):
        if col not in df.columns:
            continue
        print(f"\n--- {col}: top 60 distinct values ---")
        counts = (
            df[col].fillna("").astype(str).str.strip()
            .replace("", "<blank>")
            .value_counts()
            .head(60)
        )
        for value, count in counts.items():
            print(f"{count:8,d}  {value}")

