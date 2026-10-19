from pathlib import Path
import pandas as pd

root = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\datasets")
report = Path(__file__).resolve().parent

primary_path = report / "msk_luad_lusc_primary_sample_manifest_v4.csv"
patient_path = root / "msk_chord_2024" / "msk_chord_out" / "nsclc_clinical_patient.csv"

def norm(s):
    return s.astype("string").str.strip().replace("", pd.NA)

print("=== PRIMARY-ONLY TARGET VALIDATION ===")
primary = pd.read_csv(primary_path, dtype="string", low_memory=False)
print(f"Rows: {len(primary):,}")
for c in ["SAMPLE_ID", "PATIENT_ID", "AUDIT_LABEL", "SAMPLE_TYPE",
          "ONCOTREE_CODE", "CANCER_TYPE_DETAILED"]:
    if c in primary.columns:
        print(f"\n{c}:")
        print(norm(primary[c]).fillna("<MISSING>").value_counts().head(10).to_string())

if "AUDIT_LABEL" not in primary.columns:
    raise SystemExit("AUDIT_LABEL missing from primary manifest.")

labels = norm(primary["AUDIT_LABEL"]).str.upper()
valid = labels.isin(["LUAD", "LUSC"])
print("\nTarget counts (AUDIT_LABEL):")
print(labels[valid].value_counts().to_string())
print(f"Rows with missing/unexpected target: {int((~valid).sum()):,}")

# Ensure the target agrees with OncoTree code and detailed subtype.
if "ONCOTREE_CODE" in primary.columns:
    onco = norm(primary["ONCOTREE_CODE"]).str.upper()
    mismatch = valid & onco.notna() & labels.ne(onco)
    print(f"AUDIT_LABEL vs ONCOTREE_CODE mismatches: {int(mismatch.sum()):,}")
    if mismatch.any():
        print(primary.loc[mismatch, ["SAMPLE_ID", "PATIENT_ID", "AUDIT_LABEL", "ONCOTREE_CODE"]].head(20).to_string(index=False))

if "CANCER_TYPE_DETAILED" in primary.columns:
    subtype = norm(primary["CANCER_TYPE_DETAILED"]).str.lower()
    subtype_ok = (
        (labels.eq("LUAD") & subtype.eq("lung adenocarcinoma")) |
        (labels.eq("LUSC") & subtype.eq("lung squamous cell carcinoma"))
    )
    print(f"AUDIT_LABEL vs detailed subtype mismatches: {int((valid & ~subtype_ok).sum()):,}")
    if (valid & ~subtype_ok).any():
        print(primary.loc[valid & ~subtype_ok, ["SAMPLE_ID", "PATIENT_ID", "AUDIT_LABEL", "CANCER_TYPE_DETAILED"]].head(20).to_string(index=False))

# Check patient-level clinical predictor completeness joined by PATIENT_ID.
patients = pd.read_csv(patient_path, dtype="string", low_memory=False)
if "PATIENT_ID" not in patients.columns:
    raise SystemExit("PATIENT_ID missing from patient clinical file.")
if patients["PATIENT_ID"].duplicated().any():
    dup = patients.loc[patients["PATIENT_ID"].duplicated(False), "PATIENT_ID"].head(20).tolist()
    print(f"WARNING: repeated patient IDs in patient table: {dup}")
else:
    print("\nPatient clinical table: PATIENT_ID unique.")

joined = primary.merge(patients, on="PATIENT_ID", how="left", validate="many_to_one", indicator=True)
print("\nPatient-table join status:")
print(joined["_merge"].value_counts().to_string())

predictors = ["CURRENT_AGE_DEID", "GENDER", "SMOKING_PREDICTIONS_3_CLASSES"]
fairness = ["RACE", "ETHNICITY"]
print("\n=== PREDICTOR MISSINGNESS BY TARGET ===")
for c in predictors + fairness:
    if c not in joined.columns:
        print(f"{c}: COLUMN MISSING")
        continue
    v = norm(joined[c])
    tmp = pd.DataFrame({"label": labels, "value": v})
    print(f"\n{c}")
    rows = []
    for lab in ["LUAD", "LUSC"]:
        sub = tmp.loc[tmp["label"].eq(lab), "value"]
        rows.append({
            "label": lab,
            "n": len(sub),
            "missing_n": int(sub.isna().sum()),
            "missing_pct": round(100 * sub.isna().mean(), 2) if len(sub) else None,
            "unique_nonmissing": int(sub.nunique())
        })
    print(pd.DataFrame(rows).to_string(index=False))
    if c in fairness + ["GENDER", "SMOKING_PREDICTIONS_3_CLASSES"]:
        print(pd.crosstab(labels, v.fillna("<MISSING>")).to_string())

print("\n=== TCGA OVERLAP SEARCH ===")
tcga_root = root / "tcga"
if not tcga_root.exists():
    # Find likely TCGA cohort folders under the supplied datasets directory.
    candidates = [p for p in root.iterdir() if p.is_dir() and "tcga" in p.name.lower()]
    print("datasets/tcga not found.")
    print("Top-level TCGA-like folders:", [str(p) for p in candidates[:20]])
else:
    print(f"TCGA root: {tcga_root}")
    files = list(tcga_root.rglob("*"))
    csvs = [p for p in files if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt"}]
    print(f"Delimited files found: {len(csvs):,}")
    for p in csvs[:100]:
        print(p)

print("\nNo model training performed. No production artifacts modified.")
