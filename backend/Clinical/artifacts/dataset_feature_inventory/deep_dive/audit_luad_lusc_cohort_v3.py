import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight")
DATA = ROOT / "datasets"
OUT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2\backend\Clinical\artifacts\dataset_feature_inventory\deep_dive")
OUT.mkdir(parents=True, exist_ok=True)

REPORT = OUT / "luad_lusc_cohort_decision_audit_v3.txt"
SUMMARY = OUT / "luad_lusc_cohort_decision_summary_v3.csv"
DETAILS = OUT / "luad_lusc_cohort_decision_details_v3.json"

def clean(s):
    return s.astype("string").str.strip().replace(
        {"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "[Not Available]": pd.NA,
         "Not Available": pd.NA, "Unknown": pd.NA, "unknown": pd.NA}
    )

def read_msk_table(path, skiprows=0, sep="\t"):
    return pd.read_csv(path, sep=sep, skiprows=skiprows, dtype="string",
                       low_memory=False, on_bad_lines="warn")

def read_tcga(path):
    # Detect the actual header instead of assuming a fixed preamble length.
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_i = None
    for i, line in enumerate(lines[:40]):
        low = line.lower()
        if ("bcr_patient_barcode" in low or "patient_barcode" in low) and "\t" in line:
            header_i = i
            break
    if header_i is None:
        raise ValueError(f"Could not detect TCGA tabular header: {path}")
    return pd.read_csv(path, sep="\t", skiprows=header_i, dtype="string",
                       low_memory=False, on_bad_lines="warn")

def val_counts(series):
    return {str(k): int(v) for k, v in series.fillna("<MISSING>").value_counts().items()}

def missing_pct(df, cols):
    result = {}
    for c in cols:
        if c in df.columns:
            x = clean(df[c])
            result[c] = {
                "missing_n": int(x.isna().sum()),
                "missing_pct": round(float(x.isna().mean() * 100), 2),
                "unique_nonmissing": int(x.nunique(dropna=True)),
                "top_values": val_counts(x).copy()
            }
    return result

def id_audit(df, col):
    if col not in df:
        return {"column_missing": True}
    x = clean(df[col])
    dup = x[x.notna() & x.duplicated(keep=False)]
    return {
        "rows": int(len(df)),
        "missing_ids": int(x.isna().sum()),
        "unique_ids": int(x.nunique(dropna=True)),
        "rows_with_repeated_id": int(dup.shape[0]),
        "repeated_id_examples": [str(v) for v in dup.drop_duplicates().head(15).tolist()]
    }

def norm_label(v):
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    if s == "LUAD" or "LUNG ADENOCARCINOMA" in s:
        return "LUAD"
    if s == "LUSC" or "LUNG SQUAMOUS CELL CARCINOMA" in s:
        return "LUSC"
    return None

def pick_col(df, choices):
    lower = {str(c).lower(): c for c in df.columns}
    for c in choices:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def tcga_file(cohort, pattern):
    matches = list((DATA / cohort).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} under {DATA / cohort}")
    return matches[0]

def summarize_tcga(cohort, patient_pattern, sample_pattern):
    patient_path = tcga_file(cohort, patient_pattern)
    sample_path = tcga_file(cohort, sample_pattern)
    patient = read_tcga(patient_path)
    sample = read_tcga(sample_path)

    pid = pick_col(patient, ["bcr_patient_barcode", "patient_barcode"])
    sid = pick_col(sample, ["bcr_patient_barcode", "patient_barcode"])
    hist = pick_col(patient, ["icd_o_3_histology", "histological_type", "diagnosis"])
    gender = pick_col(patient, ["gender", "sex"])
    age = pick_col(patient, ["age_at_initial_pathologic_diagnosis", "age_at_diagnosis"])
    packyears = pick_col(patient, ["number_pack_years_smoked"])
    smoking = pick_col(patient, ["tobacco_smoking_history"])
    confirm = pick_col(sample, ["path_confirm_diagnosis_matching"])

    return {
        "cohort": cohort,
        "patient_path": str(patient_path),
        "sample_path": str(sample_path),
        "patient_shape": list(patient.shape),
        "sample_shape": list(sample.shape),
        "patient_id_column": pid,
        "sample_patient_id_column": sid,
        "patient_id_audit": id_audit(patient, pid) if pid else None,
        "sample_patient_id_audit": id_audit(sample, sid) if sid else None,
        "histology_column": hist,
        "histology_distribution": val_counts(patient[hist]) if hist else {},
        "gender_column": gender,
        "gender_distribution": val_counts(patient[gender]) if gender else {},
        "age_column": age,
        "age_missing": int(clean(patient[age]).isna().sum()) if age else None,
        "pack_years_column": packyears,
        "pack_years_distribution": val_counts(patient[packyears]) if packyears else {},
        "smoking_column": smoking,
        "smoking_distribution": val_counts(patient[smoking]) if smoking else {},
        "sample_pathology_confirmation_column": confirm,
        "sample_pathology_confirmation": val_counts(sample[confirm]) if confirm else {},
        "patient_sample_id_overlap": None
    }

def main():
    details = {
        "scope": "Read-only label, cohort, ID and candidate baseline audit; no training.",
        "data_root": str(DATA),
        "warnings": [],
        "msk": {},
        "tcga": {},
        "decision_checks": {}
    }
    rows = []

    msk = DATA / "msk_chord_2024"
    patient = read_msk_table(msk / "data_clinical_patient.txt", skiprows=4)
    sample = read_msk_table(msk / "data_clinical_sample.txt", skiprows=4)
    ns_patient = pd.read_csv(msk / "msk_chord_out" / "nsclc_clinical_patient.csv", dtype="string", low_memory=False)
    ns_sample = pd.read_csv(msk / "msk_chord_out" / "nsclc_clinical_sample.csv", dtype="string", low_memory=False)
    combined = pd.read_csv(msk / "msk_chord_out" / "nsclc_patient_level_features.csv", dtype="string", low_memory=False)

    msk_detail = {
        "patient_shape": list(patient.shape),
        "sample_shape": list(sample.shape),
        "nsclc_patient_shape": list(ns_patient.shape),
        "nsclc_sample_shape": list(ns_sample.shape),
        "combined_features_shape": list(combined.shape),
        "patient_id_audit": id_audit(patient, "PATIENT_ID"),
        "sample_patient_id_audit": id_audit(sample, "PATIENT_ID"),
        "sample_id_audit": id_audit(sample, "SAMPLE_ID"),
        "nsclc_patient_id_audit": id_audit(ns_patient, "PATIENT_ID"),
        "nsclc_sample_patient_id_audit": id_audit(ns_sample, "PATIENT_ID"),
        "combined_patient_id_audit": id_audit(combined, "PATIENT_ID"),
        "sample_type_distribution": val_counts(sample["SAMPLE_TYPE"]) if "SAMPLE_TYPE" in sample else {},
        "sample_class_distribution": val_counts(sample["SAMPLE_CLASS"]) if "SAMPLE_CLASS" in sample else {},
        "primary_site_distribution": val_counts(sample["PRIMARY_SITE"]) if "PRIMARY_SITE" in sample else {},
        "oncotree_distribution": val_counts(sample["ONCOTREE_CODE"]) if "ONCOTREE_CODE" in sample else {},
        "detailed_subtype_distribution": val_counts(sample["CANCER_TYPE_DETAILED"]) if "CANCER_TYPE_DETAILED" in sample else {},
        "icd_histology_distribution": val_counts(sample["ICD_O_HISTOLOGY_DESCRIPTION"]) if "ICD_O_HISTOLOGY_DESCRIPTION" in sample else {},
        "diagnosis_description_distribution": val_counts(sample["DIAGNOSIS_DESCRIPTION"]) if "DIAGNOSIS_DESCRIPTION" in sample else {},
        "patient_baseline_missingness": missing_pct(patient, ["GENDER", "CURRENT_AGE_DEID", "SMOKING_PREDICTIONS_3_CLASSES"]),
        "nsclc_patient_baseline_missingness": missing_pct(ns_patient, ["GENDER", "CURRENT_AGE_DEID", "SMOKING_PREDICTIONS_3_CLASSES"]),
        "combined_baseline_missingness": missing_pct(combined, ["GENDER", "CURRENT_AGE_DEID", "SMOKING_PREDICTIONS_3_CLASSES", "latest_ecog_status"])
    }

    # Validate labels from two independent subtype fields on the filtered NSCLC sample export.
    code_col = pick_col(ns_sample, ["ONCOTREE_CODE"])
    detail_col = pick_col(ns_sample, ["CANCER_TYPE_DETAILED"])
    sample_patient_col = pick_col(ns_sample, ["PATIENT_ID"])
    sample_id_col = pick_col(ns_sample, ["SAMPLE_ID"])
    sample_type_col = pick_col(ns_sample, ["SAMPLE_TYPE"])
    primary_site_col = pick_col(ns_sample, ["PRIMARY_SITE"])

    if code_col and detail_col:
        code_label = ns_sample[code_col].map(norm_label)
        detail_label = ns_sample[detail_col].map(norm_label)
        eligible = code_label.isin(["LUAD", "LUSC"]) & detail_label.isin(["LUAD", "LUSC"])
        conflicts = (code_label.notna() & detail_label.notna() & (code_label != detail_label))
        msk_detail["nsclc_code_label_distribution"] = val_counts(ns_sample[code_col])
        msk_detail["nsclc_detailed_label_distribution"] = val_counts(ns_sample[detail_col])
        msk_detail["code_vs_detail_label_conflict_rows"] = int(conflicts.sum())
        msk_detail["exact_luad_lusc_agreement_rows"] = int(
            ((code_label == detail_label) & code_label.isin(["LUAD", "LUSC"])).sum()
        )
        msk_detail["eligible_luad_lusc_rows_agreeing_fields"] = int(eligible.sum())
        msk_detail["eligible_luad_lusc_patient_ids"] = int(
            clean(ns_sample.loc[eligible, sample_patient_col]).nunique()
        ) if sample_patient_col else None
        msk_detail["eligible_sample_type_distribution"] = val_counts(
            ns_sample.loc[eligible, sample_type_col]
        ) if sample_type_col else {}
        msk_detail["eligible_primary_site_distribution"] = val_counts(
            ns_sample.loc[eligible, primary_site_col]
        ) if primary_site_col else {}
        msk_detail["eligible_label_distribution"] = val_counts(code_label[eligible])
        if sample_patient_col:
            dup_eligible = clean(ns_sample.loc[eligible, sample_patient_col]).duplicated(keep=False)
            msk_detail["eligible_rows_with_repeated_patient_id"] = int(dup_eligible.sum())
        if sample_id_col:
            msk_detail["eligible_rows_with_repeated_sample_id"] = int(
                clean(ns_sample.loc[eligible, sample_id_col]).duplicated(keep=False).sum()
            )
        for label, count in val_counts(code_label[eligible]).items():
            rows.append({"source": "MSK_NSCLC_sample", "metric": f"eligible_{label}", "value": count})
        rows.extend([
            {"source": "MSK_NSCLC_sample", "metric": "rows", "value": len(ns_sample)},
            {"source": "MSK_NSCLC_sample", "metric": "label_conflicts", "value": int(conflicts.sum())},
            {"source": "MSK_NSCLC_sample", "metric": "eligible_luad_lusc_agreeing_rows", "value": int(eligible.sum())}
        ])

    # Patient-level merge consistency and patient overlap checks.
    if "PATIENT_ID" in ns_patient and "PATIENT_ID" in ns_sample:
        pids = set(clean(ns_patient["PATIENT_ID"]).dropna())
        sids = set(clean(ns_sample["PATIENT_ID"]).dropna())
        msk_detail["nsclc_patient_sample_patient_id_set_equal"] = pids == sids
        msk_detail["nsclc_patient_ids_without_sample"] = len(pids - sids)
        msk_detail["nsclc_sample_patient_ids_without_patient_row"] = len(sids - pids)
    if "PATIENT_ID" in combined and "PATIENT_ID" in ns_sample:
        cp = set(clean(combined["PATIENT_ID"]).dropna())
        sp = set(clean(ns_sample["PATIENT_ID"]).dropna())
        msk_detail["combined_vs_sample_patient_id_set_equal"] = cp == sp
        msk_detail["combined_patient_ids_without_sample_row"] = len(cp - sp)
        msk_detail["sample_patient_ids_without_combined_row"] = len(sp - cp)

    details["msk"] = msk_detail

    details["tcga"]["TCGA-LUAD"] = summarize_tcga(
        "TCGA-LUAD",
        "nationwidechildrens.org_clinical_patient_luad.txt",
        "nationwidechildrens.org_ssf_tumor_samples_luad.txt"
    )
    details["tcga"]["TCGA-LUSC"] = summarize_tcga(
        "TCGA-LUSC",
        "nationwidechildrens.org_clinical_patient_lusc.txt",
        "nationwidechildrens.org_ssf_tumor_samples_lusc.txt"
    )

    # TCGA patient barcodes usually encode a shared TCGA participant prefix.
    tcga_sets = {}
    for cohort, item in details["tcga"].items():
        pth = Path(item["patient_path"])
        df = read_tcga(pth)
        pid = item["patient_id_column"]
        tcga_sets[cohort] = set(clean(df[pid]).dropna()) if pid else set()
    overlap = tcga_sets["TCGA-LUAD"] & tcga_sets["TCGA-LUSC"]
    details["decision_checks"]["tcga_luad_lusc_patient_barcode_overlap_n"] = len(overlap)
    details["decision_checks"]["tcga_luad_lusc_patient_barcode_overlap_examples"] = sorted(overlap)[:20]

    # Check whether cohort labels are represented in obvious fields that must never be predictors.
    details["decision_checks"]["label_columns_never_predictors"] = [
        "ONCOTREE_CODE", "CANCER_TYPE_DETAILED", "CANCER_TYPE",
        "ICD_O_HISTOLOGY_DESCRIPTION", "DIAGNOSIS_DESCRIPTION",
        "diagnosis", "histological_type", "icd_o_3_histology"
    ]
    details["decision_checks"]["potential_post_diagnosis_or_leakage_fields_to_exclude_from_baseline"] = [
        "STAGE_HIGHEST_RECORDED", "OS_MONTHS", "OS_STATUS",
        "pathologic/clinical T, N, M stage", "treatments and treatment-event summaries",
        "mutation/gene-panel/variant-derived fields unless explicitly defining a separate molecular model"
    ]
    details["decision_checks"]["do_not_merge_cohorts_without_schema_and_domain_review"] = True
    details["decision_checks"]["no_training_performed"] = True
    details["decision_checks"]["production_artifacts_modified"] = False

    # Make compact decision summary.
    summary_rows = [
        {"source": "MSK_all_samples", "metric": "rows", "value": len(sample)},
        {"source": "MSK_all_samples", "metric": "unique_patients", "value": clean(sample["PATIENT_ID"]).nunique()},
        {"source": "MSK_NSCLC_sample", "metric": "rows", "value": len(ns_sample)},
        {"source": "MSK_NSCLC_patient", "metric": "rows", "value": len(ns_patient)},
        {"source": "MSK_combined_features", "metric": "rows", "value": len(combined)},
        {"source": "TCGA-LUAD_patient", "metric": "rows", "value": details["tcga"]["TCGA-LUAD"]["patient_shape"][0]},
        {"source": "TCGA-LUSC_patient", "metric": "rows", "value": details["tcga"]["TCGA-LUSC"]["patient_shape"][0]},
        {"source": "TCGA", "metric": "LUAD_LUSC_barcode_overlap", "value": len(overlap)}
    ]
    summary_rows.extend(rows)
    pd.DataFrame(summary_rows).to_csv(SUMMARY, index=False)

    with DETAILS.open("w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False, default=str)

    lines = []
    lines += [
        "LUAD/LUSC COHORT DECISION AUDIT V3",
        "=" * 80,
        "Read-only. No training, no source mutation, no production artifact changes.",
        "",
        "MSK SUMMARY",
        json.dumps(msk_detail, indent=2, ensure_ascii=False, default=str),
        "",
        "TCGA SUMMARY",
        json.dumps(details["tcga"], indent=2, ensure_ascii=False, default=str),
        "",
        "DECISION CHECKS",
        json.dumps(details["decision_checks"], indent=2, ensure_ascii=False, default=str),
        "",
        "DECISION GUARDRAILS",
        "- Do not use diagnosis/histology/cohort membership fields as predictors.",
        "- Do not count metastasis and primary samples as independent patients.",
        "- Do not use stage, survival, treatment, or molecular features in a diagnosis-time clinical-only baseline.",
        "- Review source label conflicts and sample selection before defining the training cohort.",
        "- TCGA and MSK must remain separate until feature definitions and population differences are assessed.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"Report: {REPORT}")
    print(f"Summary CSV: {SUMMARY}")
    print(f"Details JSON: {DETAILS}")
    print("\n=== DECISION-CRITICAL FINDINGS ===")
    print(json.dumps({
        "msk_exact_label_candidates": msk_detail.get("eligible_label_distribution"),
        "msk_label_conflicts": msk_detail.get("code_vs_detail_label_conflict_rows"),
        "msk_eligible_rows_repeated_patient_id": msk_detail.get("eligible_rows_with_repeated_patient_id"),
        "msk_eligible_sample_type_distribution": msk_detail.get("eligible_sample_type_distribution"),
        "msk_eligible_primary_site_distribution": msk_detail.get("eligible_primary_site_distribution"),
        "msk_patient_sample_ids_equal": msk_detail.get("nsclc_patient_sample_patient_id_set_equal"),
        "combined_sample_ids_equal": msk_detail.get("combined_vs_sample_patient_id_set_equal"),
        "tcga_luad_lusc_overlap": len(overlap),
        "tcga_luad_histology": details["tcga"]["TCGA-LUAD"]["histology_distribution"],
        "tcga_lusc_histology": details["tcga"]["TCGA-LUSC"]["histology_distribution"]
    }, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()
