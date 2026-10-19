import json
from pathlib import Path

report_dir = Path(__file__).resolve().parent
details_path = report_dir / "luad_lusc_baseline_readiness_details_v4.json"
summary_path = report_dir / "luad_lusc_baseline_readiness_summary_v4.csv"
manifest_path = report_dir / "msk_luad_lusc_eligible_sample_manifest_v4.csv"
primary_path = report_dir / "msk_luad_lusc_primary_sample_manifest_v4.csv"

if not details_path.exists():
    raise SystemExit(f"Missing report: {details_path}")

data = json.loads(details_path.read_text(encoding="utf-8"))

def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                yield from walk(v, p)
            else:
                yield p, v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{prefix}[{i}]")

print("=== EXISTING AUDIT SUMMARY ===")
for key, value in data.items():
    if isinstance(value, dict) and len(value) <= 15:
        print(f"\n[{key}]")
        for k, v in value.items():
            if isinstance(v, dict) and len(v) > 12:
                print(f"  {k}: {len(v)} entries (omitted)")
            elif isinstance(v, list) and len(v) > 12:
                print(f"  {k}: {len(v)} entries (omitted)")
            else:
                print(f"  {k}: {v}")
    elif not isinstance(value, (dict, list)):
        print(f"{key}: {value}")

# Avoid PowerShell ConvertFrom-Json: it can collapse case-distinct keys.
try:
    import pandas as pd
except ImportError:
    print("\nPandas unavailable; CSV manifest analysis skipped.")
    raise SystemExit(0)

def load_csv(path):
    if not path.exists():
        print(f"Missing CSV: {path}")
        return None
    return pd.read_csv(path, low_memory=False)

def find_col(df, names):
    if df is None:
        return None
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None

def label_col(df):
    return find_col(df, [
        "strict_label", "histology_label", "cancer_type",
        "label", "oncotree_code", "oncotree_lineage"
    ])

def summarize_cohort(df, name):
    if df is None:
        return
    print(f"\n=== {name.upper()} MANIFEST ===")
    print(f"Rows: {len(df):,}; columns: {len(df.columns):,}")
    lab = label_col(df)
    if lab:
        print("Label counts:")
        print(df[lab].fillna("<MISSING>").astype(str).value_counts().head(12).to_string())
    else:
        print("Label column not identified automatically.")
        print("Column names:", ", ".join(map(str, df.columns[:50])))

    patient = find_col(df, [
        "PATIENT_ID", "PATIENT_ID_DEID", "PATIENT", "PATIENT_BARCODE",
        "bcr_patient_barcode", "patientid", "patient_id"
    ])
    if patient:
        ids = df[patient].astype("string").str.strip()
        print(f"Patient ID column: {patient}")
        print(f"Missing patient IDs: {int(ids.isna().sum() + ids.eq('').sum()):,}")
        print(f"Repeated nonempty patient IDs: {int(ids[ids.notna() & ids.ne('')].duplicated().sum()):,}")
    else:
        print("Patient ID column not identified automatically.")

    sample_type = find_col(df, ["SAMPLE_TYPE", "sample_type", "sample_type_name"])
    if sample_type:
        print("Sample types:")
        print(df[sample_type].fillna("<MISSING>").astype(str).value_counts().head(10).to_string())

    for wanted in [
        "GENDER", "CURRENT_AGE_DEID", "SMOKING_PREDICTIONS_3_CLASSES",
        "RACE", "ETHNICITY", "STAGE_HIGHEST_RECORDED", "OS_MONTHS",
        "OS_STATUS", "TUMOR_PURITY", "SAMPLE_COVERAGE", "TMB_NONSYNONYMOUS",
        "GENE_PANEL", "PRIMARY_SITE"
    ]:
        col = find_col(df, [wanted])
        if not col:
            continue
        s = df[col]
        missing = int(s.isna().sum() + s.astype("string").str.strip().eq("").sum())
        print(f"{col}: missing={missing:,}/{len(df):,} ({100*missing/max(1,len(df)):.2f}%)")
        if wanted in {"CURRENT_AGE_DEID", "OS_MONTHS", "TUMOR_PURITY",
                      "SAMPLE_COVERAGE", "TMB_NONSYNONYMOUS"}:
            x = pd.to_numeric(s, errors="coerce")
            x = x[x.notna()]
            if len(x):
                print(
                    f"  numeric n={len(x):,}; min={x.min():.3g}; "
                    f"median={x.median():.3g}; p95={x.quantile(.95):.3g}; "
                    f"max={x.max():.3g}"
                )
        elif wanted in {"GENDER", "SMOKING_PREDICTIONS_3_CLASSES",
                        "RACE", "ETHNICITY", "PRIMARY_SITE"}:
            counts = s.fillna("<MISSING>").astype(str).value_counts().head(8)
            print("  top categories: " + "; ".join(f"{k}={v}" for k,v in counts.items()))

eligible = load_csv(manifest_path)
primary = load_csv(primary_path)
summarize_cohort(eligible, "eligible all-sample")
summarize_cohort(primary, "primary-only")

if eligible is not None and primary is not None:
    print("\n=== PRIMARY-ONLY LABEL COUNTS ===")
    lab = label_col(primary)
    if lab:
        counts = primary[lab].fillna("<MISSING>").astype(str).value_counts()
        print(counts.to_string())
        if "LUAD" in counts and "LUSC" in counts and counts["LUSC"]:
            print(f"LUAD:LUSC ratio = {counts['LUAD']/counts['LUSC']:.2f}:1")
    else:
        print("Could not identify primary manifest label column.")

print("\n=== SAFETY STATUS ===")
print("This review script did not train a model or write to production artifacts.")
