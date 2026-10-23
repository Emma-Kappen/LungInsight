from pathlib import Path
import csv
import json
import re
import sys
from collections import Counter

import pandas as pd

DATASETS = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\datasets")
OUT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2\backend\Clinical\artifacts\dataset_feature_inventory\deep_dive")
REPORT = OUT / "label_cohort_integrity_audit.txt"
CSV_OUT = OUT / "label_cohort_integrity_summary.csv"
DETAIL_OUT = OUT / "label_cohort_integrity_details.json"

def emit(f, title, value):
    f.write("\n" + "=" * 100 + "\n" + title + "\n" + "=" * 100 + "\n")
    if isinstance(value, (dict, list)):
        f.write(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    else:
        f.write(str(value) + "\n")

def read_msk_table(path):
    # cBioPortal-style clinical tables contain four metadata rows before the real header.
    return pd.read_csv(path, sep="\t", skiprows=4, dtype=str, low_memory=False)

def read_csv_table(path):
    return pd.read_csv(path, dtype=str, low_memory=False)

def norm(s):
    return re.sub(r"\s+", " ", str(s).strip()).upper()

def histology_label(row, columns):
    # Use explicit pathology subtype fields only to derive the reference label.
    for col in columns:
        if col not in row.index:
            continue
        v = norm(row[col])
        if not v or v in {"NAN", "NA", "N/A", "NOT AVAILABLE", "UNKNOWN", "[NOT AVAILABLE]"}:
            continue
        if v in {"LUAD", "LUNG ADENOCARCINOMA", "ADENOCARCINOMA, NOS"}:
            return "LUAD"
        if v in {"LUSC", "LUNG SQUAMOUS CELL CARCINOMA", "SQUAMOUS CELL CARCINOMA, NOS"}:
            return "LUSC"
        if "SQUAMOUS" in v or "SQ CELL" in v:
            return "LUSC"
        if "ADENOCARCINOMA" in v or v in {"8140/3", "8255/3", "8480/3", "8550/3"}:
            return "LUAD"
    return None

def safe_counts(series):
    return {str(k): int(v) for k, v in series.fillna("<MISSING>").value_counts(dropna=False).head(40).items()}

def table_summary(name, df, id_col=None, label_col=None):
    d = {
        "table": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(map(str, df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_pct_top": {
            str(c): round(float(df[c].isna().mean() * 100), 2)
            for c in df.columns
        },
    }
    if id_col and id_col in df.columns:
        ids = df[id_col].astype("string").str.strip()
        d["id_column"] = id_col
        d["id_nonempty"] = int(ids.notna().sum())
        d["id_unique_nonempty"] = int(ids[ids.notna() & (ids != "")].nunique())
        d["duplicate_id_rows"] = int(ids[ids.notna() & (ids != "")].duplicated().sum())
    if label_col and label_col in df.columns:
        d["label_column"] = label_col
        d["label_counts"] = safe_counts(df[label_col])
    return d

def find_one(root, exact_name):
    hits = list(root.rglob(exact_name))
    hits = [p for p in hits if "logs" not in p.parts and p.is_file()]
    return hits[0] if hits else None

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    details = []
    rows = []

    with REPORT.open("w", encoding="utf-8") as f:
        emit(f, "PURPOSE", [
            "Confirm real source paths, parse source tables correctly, inspect subtype labels and IDs.",
            "No model training, no artifact promotion, no mutation of source data.",
            "No clinical or molecular predictor selection is finalized by this audit.",
        ])

        msk_root = DATASETS / "msk_chord_2024"
        msk_patient_path = msk_root / "data_clinical_patient.txt"
        msk_sample_path = msk_root / "data_clinical_sample.txt"
        msk_timeline_path = msk_root / "data_timeline_diagnosis.txt"
        msk_mut_path = msk_root / "data_mutations.txt"
        msk_nsclc_patient_path = msk_root / "msk_chord_out" / "nsclc_clinical_patient.csv"
        msk_nsclc_sample_path = msk_root / "msk_chord_out" / "nsclc_clinical_sample.csv"
        msk_nsclc_features_path = msk_root / "msk_chord_out" / "nsclc_patient_level_features.csv"
        msk_nsclc_mut_path = msk_root / "msk_chord_out" / "nsclc_mutations.csv"

        paths = {
            "MSK patient clinical": msk_patient_path,
            "MSK sample clinical": msk_sample_path,
            "MSK diagnosis timeline": msk_timeline_path,
            "MSK mutations": msk_mut_path,
            "MSK NSCLC patient": msk_nsclc_patient_path,
            "MSK NSCLC sample": msk_nsclc_sample_path,
            "MSK NSCLC patient-level features": msk_nsclc_features_path,
            "MSK NSCLC mutations": msk_nsclc_mut_path,
        }
        emit(f, "1. SOURCE PATH CHECK", {k: {"path": str(v), "exists": v.exists()} for k, v in paths.items()})

        # Confirm the mutation files are MAF-like tables with a real header.
        for name, path in [("MSK mutations", msk_mut_path), ("MSK NSCLC mutations", msk_nsclc_mut_path)]:
            if path.exists():
                with path.open("r", encoding="utf-8-sig", errors="replace") as h:
                    first = h.readline().strip()
                emit(f, f"2. MUTATION HEADER: {name}", first.split("\t")[:18] if "\t" in first else first.split(",")[:18])

        # Read MSK tables with the proper clinical header offset.
        msk_patient = read_msk_table(msk_patient_path) if msk_patient_path.exists() else pd.DataFrame()
        msk_sample = read_msk_table(msk_sample_path) if msk_sample_path.exists() else pd.DataFrame()
        msk_timeline = pd.read_csv(msk_timeline_path, sep="\t", dtype=str, low_memory=False) if msk_timeline_path.exists() else pd.DataFrame()

        emit(f, "3. MSK PATIENT TABLE PARSED", {
            "shape": list(msk_patient.shape),
            "columns": list(msk_patient.columns),
            "gender": safe_counts(msk_patient["GENDER"]) if "GENDER" in msk_patient else {},
            "smoking": safe_counts(msk_patient["SMOKING_PREDICTIONS_3_CLASSES"]) if "SMOKING_PREDICTIONS_3_CLASSES" in msk_patient else {},
            "age_summary": pd.to_numeric(msk_patient.get("CURRENT_AGE_DEID", pd.Series(dtype=str)), errors="coerce").describe().to_dict(),
        })
        emit(f, "4. MSK SAMPLE TABLE PARSED", {
            "shape": list(msk_sample.shape),
            "columns": list(msk_sample.columns),
            "oncotree": safe_counts(msk_sample["ONCOTREE_CODE"]) if "ONCOTREE_CODE" in msk_sample else {},
            "detailed_subtype": safe_counts(msk_sample["CANCER_TYPE_DETAILED"]) if "CANCER_TYPE_DETAILED" in msk_sample else {},
            "primary_site": safe_counts(msk_sample["PRIMARY_SITE"]) if "PRIMARY_SITE" in msk_sample else {},
            "sample_type": safe_counts(msk_sample["SAMPLE_TYPE"]) if "SAMPLE_TYPE" in msk_sample else {},
        })

        if not msk_sample.empty:
            for col in ["PATIENT_ID", "SAMPLE_ID", "ONCOTREE_CODE", "CANCER_TYPE_DETAILED", "PRIMARY_SITE", "SAMPLE_TYPE"]:
                if col in msk_sample:
                    emit(f, f"5. MSK SAMPLE FIELD COUNTS: {col}", safe_counts(msk_sample[col]))
            if "PATIENT_ID" in msk_sample and "SAMPLE_ID" in msk_sample:
                emit(f, "6. MSK SAMPLE/PATIENT CARDINALITY", {
                    "rows": len(msk_sample),
                    "unique_patients": int(msk_sample["PATIENT_ID"].nunique(dropna=True)),
                    "unique_samples": int(msk_sample["SAMPLE_ID"].nunique(dropna=True)),
                    "duplicate_sample_ids": int(msk_sample["SAMPLE_ID"].duplicated().sum()),
                    "patients_with_multiple_samples": int((msk_sample.groupby("PATIENT_ID")["SAMPLE_ID"].nunique() > 1).sum()),
                })

        # Inspect the exported NSCLC table and establish whether it is one sample/patient.
        msk_nsclc_patient = read_csv_table(msk_nsclc_patient_path) if msk_nsclc_patient_path.exists() else pd.DataFrame()
        msk_nsclc_sample = read_csv_table(msk_nsclc_sample_path) if msk_nsclc_sample_path.exists() else pd.DataFrame()
        msk_nsclc_features = read_csv_table(msk_nsclc_features_path) if msk_nsclc_features_path.exists() else pd.DataFrame()

        emit(f, "7. MSK NSCLC EXPORT LABELS", {
            "patient_shape": list(msk_nsclc_patient.shape),
            "sample_shape": list(msk_nsclc_sample.shape),
            "features_shape": list(msk_nsclc_features.shape),
            "sample_label_counts": {
                c: safe_counts(msk_nsclc_sample[c])
                for c in ["CANCER_TYPE", "CANCER_TYPE_DETAILED", "ONCOTREE_CODE", "ICD_O_HISTOLOGY_DESCRIPTION", "DIAGNOSIS_DESCRIPTION", "PRIMARY_SITE", "SAMPLE_TYPE"]
                if c in msk_nsclc_sample
            },
            "features_duplicate_patient_ids": int(msk_nsclc_features["PATIENT_ID"].duplicated().sum()) if "PATIENT_ID" in msk_nsclc_features else None,
            "patient_export_duplicate_ids": int(msk_nsclc_patient["PATIENT_ID"].duplicated().sum()) if "PATIENT_ID" in msk_nsclc_patient else None,
            "sample_export_duplicate_patient_ids": int(msk_nsclc_sample["PATIENT_ID"].duplicated().sum()) if "PATIENT_ID" in msk_nsclc_sample else None,
        })

        # Label count only; do not yet build or train a model.
        if not msk_nsclc_sample.empty and "ONCOTREE_CODE" in msk_nsclc_sample:
            labels = msk_nsclc_sample["ONCOTREE_CODE"].astype("string").str.upper().str.strip()
            emit(f, "8. MSK EXACT LUAD/LUSC LABEL CANDIDATES", {
                "all_rows": len(labels),
                "LUAD": int((labels == "LUAD").sum()),
                "LUSC": int((labels == "LUSC").sum()),
                "other_codes": safe_counts(labels[~labels.isin(["LUAD", "LUSC"])]),
                "missing": int(labels.isna().sum()),
            })

        # TCGA table paths are nested in UUID directories; find exact expected filenames.
        tcga_specs = [
            ("TCGA-LUAD", "nationwidechildrens.org_clinical_patient_luad.txt", "patient"),
            ("TCGA-LUAD", "nationwidechildrens.org_ssf_tumor_samples_luad.txt", "sample"),
            ("TCGA-LUSC", "nationwidechildrens.org_clinical_patient_lusc.txt", "patient"),
            ("TCGA-LUSC", "nationwidechildrens.org_ssf_tumor_samples_lusc.txt", "sample"),
        ]
        tcga_tables = {}
        for cohort, filename, kind in tcga_specs:
            root = DATASETS / cohort
            path = find_one(root, filename) if root.exists() else None
            entry = {"cohort": cohort, "kind": kind, "path": str(path) if path else None, "exists": bool(path)}
            if path:
                # TCGA legacy clinical files have metadata/comment rows; locate the row containing field names.
                lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
                header_idx = None
                for i, line in enumerate(lines[:30]):
                    low = line.lower()
                    if ("\t" in line and ("bcr_patient_barcode" in low or "histological_type" in low or "icd_o_3_histology" in low)):
                        header_idx = i
                        break
                if header_idx is None:
                    entry["header_preview"] = lines[:8]
                else:
                    df = pd.read_csv(path, sep="\t", skiprows=header_idx, dtype=str, low_memory=False)
                    df.columns = [str(c).strip().lstrip("#") for c in df.columns]
                    tcga_tables[(cohort, kind)] = df
                    entry["shape"] = list(df.shape)
                    entry["columns"] = list(df.columns)
                    id_candidates = [c for c in df.columns if c.lower() in {"bcr_patient_barcode", "bcr_patient_uuid", "patient_id", "bcr_sample_barcode"}]
                    entry["id_column"] = id_candidates[0] if id_candidates else None
                    entry["label_field_counts"] = {
                        c: safe_counts(df[c])
                        for c in df.columns
                        if any(term in c.lower() for term in ["histolog", "diagnosis", "icd_o_3", "path_confirm"])
                    }
                    entry["missing_pct"] = {
                        c: round(float(df[c].isna().mean() * 100), 2)
                        for c in df.columns
                        if any(term in c.lower() for term in ["age_at_initial", "gender", "smok", "pack_year", "histolog", "diagnosis", "icd_o_3"])
                    }
            details.append(entry)
            emit(f, f"9. {cohort} {kind.upper()} SOURCE", entry)

        # High-level row summaries to CSV.
        for name, df, id_col in [
            ("MSK patient", msk_patient, "PATIENT_ID"),
            ("MSK sample", msk_sample, "PATIENT_ID"),
            ("MSK timeline", msk_timeline, "PATIENT_ID"),
            ("MSK NSCLC patient export", msk_nsclc_patient, "PATIENT_ID"),
            ("MSK NSCLC sample export", msk_nsclc_sample, "PATIENT_ID"),
            ("MSK NSCLC combined features", msk_nsclc_features, "PATIENT_ID"),
        ]:
            if not df.empty:
                s = table_summary(name, df, id_col=id_col)
                rows.append({
                    "table": name,
                    "rows": s["rows"],
                    "columns": s["columns"],
                    "duplicate_rows": s["duplicate_rows"],
                    "id_column": s.get("id_column"),
                    "id_unique_nonempty": s.get("id_unique_nonempty"),
                    "duplicate_id_rows": s.get("duplicate_id_rows"),
                })
        for (cohort, kind), df in tcga_tables.items():
            id_candidates = [c for c in df.columns if c.lower() in {"bcr_patient_barcode", "bcr_sample_barcode"}]
            id_col = id_candidates[0] if id_candidates else None
            s = table_summary(f"{cohort} {kind}", df, id_col=id_col)
            rows.append({
                "table": f"{cohort} {kind}",
                "rows": s["rows"],
                "columns": s["columns"],
                "duplicate_rows": s["duplicate_rows"],
                "id_column": s.get("id_column"),
                "id_unique_nonempty": s.get("id_unique_nonempty"),
                "duplicate_id_rows": s.get("duplicate_id_rows"),
            })

        pd.DataFrame(rows).to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
        DETAIL_OUT.write_text(json.dumps(details, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        emit(f, "10. OUTPUTS", {"report": str(REPORT), "summary_csv": str(CSV_OUT), "tcga_details_json": str(DETAIL_OUT)})
        emit(f, "11. INTERPRETATION GUARDRAILS", [
            "MSK patient clinical table includes all cancers; it is not a lung-only cohort.",
            "MSK sample subtype fields are labels/reference targets, not predictors.",
            "MSK sample rows can repeat a patient; all train/test partitions must be grouped by patient if modeling is later approved.",
            "The combined MSK patient-level export contains outcome-, stage-, molecular-, treatment-, and post-diagnosis-derived fields; do not use it wholesale as predictors.",
            "Mutation tables are sample-level molecular data and are excluded from a demographics/smoking baseline.",
            "TCGA cohort membership alone supplies a label but is not an independent predictor. Histology fields are reference labels only.",
            "TCGA follow-up records are longitudinal and must not be treated as independent patients.",
            "This script performs no training and does not write to production model/evaluation artifacts.",
        ])

    print(f"Audit complete: {REPORT}")
    print(f"Summary CSV: {CSV_OUT}")
    print(f"TCGA details JSON: {DETAIL_OUT}")

if __name__ == "__main__":
    main()
