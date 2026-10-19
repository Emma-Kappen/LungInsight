from pathlib import Path
import pandas as pd

data_root = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\datasets\msk_chord_2024\msk_chord_out")
report_dir = Path(__file__).resolve().parent
sample_path = data_root / "nsclc_clinical_sample.csv"
patient_path = data_root / "nsclc_clinical_patient.csv"
manifest_path = report_dir / "msk_luad_lusc_primary_sample_manifest_v4.csv"

def read_msk(path):
    # MSK clinical TXT/CSV exports can include four preamble lines.
    for skip in (4, 0):
        try:
            df = pd.read_csv(path, sep=None, engine="python", skiprows=skip, low_memory=False)
            normalized = {str(c).strip().lower() for c in df.columns}
            if any("histology" in c or "oncotree" in c for c in normalized):
                return df
        except Exception:
            pass
    raise RuntimeError(f"Could not parse expected MSK columns in {path}")

def colname(df, fragments):
    for c in df.columns:
        cl = str(c).strip().lower()
        if any(fragment in cl for fragment in fragments):
            return c
    return None

print("=== SOURCE FILES ===")
for p in (sample_path, patient_path, manifest_path):
    print(f"{p}: exists={p.exists()}")
    if p.exists():
        print(f"  bytes={p.stat().st_size:,}")

sample = read_msk(sample_path)
print("\n=== SAMPLE SOURCE: RELEVANT COLUMNS ===")
for c in sample.columns:
    if any(x in str(c).lower() for x in ("patient", "sample", "histology", "oncotree", "subtype", "site")):
        print(f"{c}: {sample[c].nunique(dropna=True)} nonmissing unique values")

label_candidates = [
    c for c in sample.columns
    if any(x in str(c).lower() for x in ("oncotree", "histology", "subtype"))
]
for c in label_candidates:
    vals = sample[c].fillna("<MISSING>").astype(str).str.strip()
    print(f"\n[{c}] top values")
    print(vals.value_counts().head(15).to_string())

sample_type = colname(sample, ["sample_type"])
patient_id = colname(sample, ["patient_id", "patient id", "patient"])
print("\n=== PRIMARY SAMPLE ROWS ===")
print(f"sample_type column: {sample_type}")
print(f"patient ID column: {patient_id}")
if sample_type:
    st = sample[sample_type].fillna("").astype(str).str.strip().str.lower()
    primary = sample.loc[st.eq("primary")].copy()
    print(f"Primary rows in source: {len(primary):,}")
else:
    primary = None
    print("Could not identify sample-type column; primary filter not applied.")

if primary is not None:
    for c in label_candidates:
        s = primary[c].fillna("<MISSING>").astype(str).str.strip()
        print(f"\n=== PRIMARY LABEL DISTRIBUTION: {c} ===")
        print(s.value_counts().head(25).to_string())

    print("\n=== PRIMARY LABEL AGREEMENT ===")
    onco = colname(primary, ["oncotree_code"])
    subtype = colname(primary, ["detailed_subtype", "histology", "subtype"])
    if onco and subtype:
        a = primary[onco].fillna("").astype(str).str.strip().str.upper()
        b = primary[subtype].fillna("").astype(str).str.strip().str.upper()
        print(f"OncoTree field: {onco}")
        print(f"Subtype field: {subtype}")
        print("OncoTree values:")
        print(a.value_counts().head(15).to_string())
        print("Subtype values:")
        print(b.value_counts().head(15).to_string())
        print("Cross-tab (top values):")
        print(pd.crosstab(a, b).to_string(max_rows=15, max_cols=15))
    else:
        print(f"Could not confidently identify both fields: oncotree={onco}, subtype={subtype}")

print("\n=== PRIMARY MANIFEST COLUMNS ===")
if manifest_path.exists():
    manifest = pd.read_csv(manifest_path, low_memory=False)
    print(", ".join(map(str, manifest.columns)))
    for c in manifest.columns:
        if any(x in str(c).lower() for x in ("label", "histology", "oncotree", "subtype", "patient")):
            print(f"{c}: {manifest[c].nunique(dropna=True)} nonmissing unique values")
