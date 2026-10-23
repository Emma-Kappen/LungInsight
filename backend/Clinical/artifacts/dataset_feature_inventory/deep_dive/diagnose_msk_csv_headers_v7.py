from pathlib import Path
import pandas as pd

root = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\datasets\msk_chord_2024\msk_chord_out")
report = Path(__file__).resolve().parent

files = [
    root / "nsclc_clinical_sample.csv",
    root / "nsclc_clinical_patient.csv",
    root / "nsclc_patient_level_features.csv",
    report / "msk_luad_lusc_primary_sample_manifest_v4.csv",
]

for path in files:
    print(f"\n{'='*80}\nFILE: {path}\nEXISTS: {path.exists()}")
    if not path.exists():
        continue

    # Show the first physical lines to establish the true delimiter/header.
    print("--- FIRST 6 LINES (truncated to 500 chars each) ---")
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for i in range(6):
            line = f.readline()
            if not line:
                break
            print(f"{i+1}: {line[:500].rstrip()}")

    # Try ordinary CSV parsing first, without guessing or skipping preamble.
    try:
        df = pd.read_csv(path, low_memory=False, nrows=8)
        print("--- HEADER AS PARSED WITH DEFAULT CSV READER ---")
        print(repr(list(df.columns)))
    except Exception as e:
        print(f"Default CSV parse error: {type(e).__name__}: {e}")

    # Parse the entire file using standard comma-separated CSV semantics.
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"FULL SHAPE: {df.shape}")
        print("--- RELEVANT COLUMN NAMES ---")
        relevant = [
            str(c) for c in df.columns
            if any(term in str(c).lower() for term in (
                "patient", "sample", "oncotree", "histology", "subtype",
                "diagnosis", "cancer_type", "tumor_type", "primary_site"
            ))
        ]
        print(relevant[:100] if relevant else "(none matched)")
        for c in relevant:
            if any(term in c.lower() for term in (
                "oncotree", "histology", "subtype", "diagnosis", "cancer_type"
            )):
                vals = df[c].dropna().astype(str).str.strip()
                print(f"\n[{c}] nonmissing={len(vals):,}, unique={vals.nunique():,}")
                print(vals.value_counts().head(12).to_string())
    except Exception as e:
        print(f"Full CSV parse error: {type(e).__name__}: {e}")
