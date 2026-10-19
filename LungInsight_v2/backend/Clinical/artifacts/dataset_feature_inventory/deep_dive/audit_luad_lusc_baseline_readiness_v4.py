import json
from pathlib import Path

import pandas as pd

DATA = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\datasets")
OUT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2\backend\Clinical\artifacts\dataset_feature_inventory\deep_dive")
MSK = DATA / "msk_chord_2024" / "msk_chord_out"
DETAILS_IN = OUT / "luad_lusc_cohort_decision_details_v3.json"

REPORT = OUT / "luad_lusc_baseline_readiness_v4.txt"
SUMMARY = OUT / "luad_lusc_baseline_readiness_summary_v4.csv"
DETAILS = OUT / "luad_lusc_baseline_readiness_details_v4.json"

def read_csv(name):
    return pd.read_csv(MSK / name, dtype="string", low_memory=False)

def clean(s):
    return s.astype("string").str.strip().replace({
        "": pd.NA, "NA": pd.NA, "N/A": pd.NA, "Unknown": pd.NA,
        "[Not Available]": pd.NA, "Not Available": pd.NA, "null": pd.NA
    })

def missing_stats(df, columns):
    out = {}
    for c in columns:
        if c not in df.columns:
            out[c] = {"present": False}
            continue
        s = clean(df[c])
        out[c] = {
            "present": True,
            "missing_n": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique_nonmissing": int(s.nunique(dropna=True)),
            "value_counts": {
                str(k): int(v) for k, v in s.fillna("<MISSING>").value_counts().items()
            }
        }
    return out

def numeric_stats(df, columns):
    out = {}
    for c in columns:
        if c not in df.columns:
            continue
        raw = clean(df[c])
        n = pd.to_numeric(raw, errors="coerce")
        out[c] = {
            "numeric_nonmissing": int(n.notna().sum()),
            "unparseable_nonempty": int((raw.notna() & n.isna()).sum()),
            "min": float(n.min()) if n.notna().any() else None,
            "median": float(n.median()) if n.notna().any() else None,
            "max": float(n.max()) if n.notna().any() else None,
            "out_of_range_examples": []
        }
    return out

def label_from(row):
    code = str(row.get("ONCOTREE_CODE", "")).strip().upper()
    detail = str(row.get("CANCER_TYPE_DETAILED", "")).strip().lower()
    if code == "LUAD" and detail == "lung adenocarcinoma":
        return "LUAD"
    if code == "LUSC" and detail == "lung squamous cell carcinoma":
        return "LUSC"
    return None

def main():
    sample = read_csv("nsclc_clinical_sample.csv")
    patient = read_csv("nsclc_clinical_patient.csv")
    features = read_csv("nsclc_patient_level_features.csv")

    # Use only agreed labels; source sample export has one row per patient,
    # but retain strict uniqueness checks and emit excluded rows for review.
    sample["AUDIT_LABEL"] = sample.apply(label_from, axis=1)
    eligible = sample[sample["AUDIT_LABEL"].isin(["LUAD", "LUSC"])].copy()

    if "SAMPLE_TYPE" in eligible:
        eligible["SAMPLE_TYPE_NORM"] = clean(eligible["SAMPLE_TYPE"]).str.lower()
    else:
        eligible["SAMPLE_TYPE_NORM"] = pd.NA

    if "PRIMARY_SITE" in eligible:
        eligible["PRIMARY_SITE_NORM"] = clean(eligible["PRIMARY_SITE"]).str.lower()
    else:
        eligible["PRIMARY_SITE_NORM"] = pd.NA

    patient_id = "PATIENT_ID"
    ids_sample = clean(eligible[patient_id])
    duplicate_ids = ids_sample[ids_sample.notna() & ids_sample.duplicated(keep=False)]
    primary = eligible[eligible["SAMPLE_TYPE_NORM"] == "primary"].copy()
    nonprimary = eligible[eligible["SAMPLE_TYPE_NORM"] != "primary"].copy()

    details = {
        "scope": "Read-only; no training; no model or production artifact modification.",
        "source_paths": {
            "sample": str(MSK / "nsclc_clinical_sample.csv"),
            "patient": str(MSK / "nsclc_clinical_patient.csv"),
            "combined_features": str(MSK / "nsclc_patient_level_features.csv")
        },
        "row_counts": {
            "sample_export": int(len(sample)),
            "patient_export": int(len(patient)),
            "combined_features": int(len(features)),
            "strict_luad_lusc_sample_rows": int(len(eligible)),
            "strict_primary_sample_rows": int(len(primary)),
            "strict_nonprimary_or_unknown_sample_rows": int(len(nonprimary))
        },
        "strict_label_distribution": {
            str(k): int(v) for k, v in eligible["AUDIT_LABEL"].value_counts().items()
        },
        "sample_type_distribution": {
            str(k): int(v) for k, v in clean(eligible["SAMPLE_TYPE"]).fillna("<MISSING>").value_counts().items()
        } if "SAMPLE_TYPE" in eligible else {},
        "primary_site_distribution": {
            str(k): int(v) for k, v in clean(eligible["PRIMARY_SITE"]).fillna("<MISSING>").value_counts().items()
        } if "PRIMARY_SITE" in eligible else {},
        "ids": {
            "eligible_unique_patients": int(ids_sample.nunique(dropna=True)),
            "eligible_missing_patient_ids": int(ids_sample.isna().sum()),
            "eligible_rows_with_repeated_patient_id": int(len(duplicate_ids)),
            "repeated_patient_id_examples": [str(x) for x in duplicate_ids.drop_duplicates().head(20).tolist()],
            "patient_sample_id_sets_equal": set(clean(patient[patient_id]).dropna()) == set(clean(sample[patient_id]).dropna()),
            "feature_sample_id_sets_equal": set(clean(features[patient_id]).dropna()) == set(clean(sample[patient_id]).dropna())
        },
        "candidate_predictor_missingness": {
            "sample": missing_stats(eligible, [
                "GENDER", "CURRENT_AGE_DEID", "SMOKING_PREDICTIONS_3_CLASSES",
                "RACE", "ETHNICITY", "STAGE_HIGHEST_RECORDED", "OS_MONTHS", "OS_STATUS",
                "SAMPLE_TYPE", "PRIMARY_SITE", "TUMOR_PURITY", "SAMPLE_COVERAGE",
                "TMB_NONSYNONYMOUS", "GENE_PANEL"
            ]),
            "patient": missing_stats(patient, [
                "GENDER", "CURRENT_AGE_DEID", "SMOKING_PREDICTIONS_3_CLASSES",
                "RACE", "ETHNICITY", "STAGE_HIGHEST_RECORDED", "OS_MONTHS", "OS_STATUS"
            ]),
            "combined_features": missing_stats(features, [
                "GENDER", "CURRENT_AGE_DEID", "SMOKING_PREDICTIONS_3_CLASSES",
                "RACE", "ETHNICITY", "STAGE_HIGHEST_RECORDED", "OS_MONTHS", "OS_STATUS",
                "latest_ecog_status"
            ])
        },
        "candidate_numeric_checks": {
            "sample": numeric_stats(eligible, ["CURRENT_AGE_DEID", "TUMOR_PURITY", "SAMPLE_COVERAGE", "TMB_NONSYNONYMOUS"]),
            "patient": numeric_stats(patient, ["CURRENT_AGE_DEID", "OS_MONTHS"]),
            "features": numeric_stats(features, ["CURRENT_AGE_DEID", "OS_MONTHS"])
        },
        "label_leakage_columns": [
            "AUDIT_LABEL", "ONCOTREE_CODE", "CANCER_TYPE", "CANCER_TYPE_DETAILED",
            "ICD_O_HISTOLOGY_DESCRIPTION", "DIAGNOSIS_DESCRIPTION"
        ],
        "exclude_from_diagnosis_time_clinical_baseline": [
            "STAGE_HIGHEST_RECORDED", "OS_MONTHS", "OS_STATUS",
            "treatment and treatment-event summaries", "mutation-derived features",
            "TMB_NONSYNONYMOUS", "GENE_PANEL", "SAMPLE_COVERAGE", "TUMOR_PURITY"
        ],
        "critical_cohort_warning": (
            "The MSK eligible rows include 2,339 metastasis, 34 local recurrence, and 87 unknown sample types. "
            "Do not silently mix them with primary diagnostic cases. Primary-only and all-sample cohorts answer "
            "different questions and should be separately reported."
        ),
        "model_training_performed": False,
        "production_artifacts_modified": False
    }

    # Export explicit eligible-row manifest for audit/review only.
    manifest_cols = [
        c for c in [
            "SAMPLE_ID", "PATIENT_ID", "AUDIT_LABEL", "SAMPLE_TYPE",
            "SAMPLE_CLASS", "PRIMARY_SITE", "METASTATIC_SITE",
            "CANCER_TYPE", "CANCER_TYPE_DETAILED", "ONCOTREE_CODE",
            "ICD_O_HISTOLOGY_DESCRIPTION", "DIAGNOSIS_DESCRIPTION"
        ] if c in eligible.columns
    ]
    manifest_path = OUT / "msk_luad_lusc_eligible_sample_manifest_v4.csv"
    eligible[manifest_cols].to_csv(manifest_path, index=False)

    # Save a separate primary-only manifest; this does not declare it the final training set.
    primary_path = OUT / "msk_luad_lusc_primary_sample_manifest_v4.csv"
    primary[manifest_cols].to_csv(primary_path, index=False)

    with DETAILS.open("w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False, default=str)

    summary_rows = [
        {"metric": "sample_export_rows", "value": len(sample)},
        {"metric": "strict_LUAD_LUSC_rows", "value": len(eligible)},
        {"metric": "strict_LUAD_rows", "value": int((eligible["AUDIT_LABEL"] == "LUAD").sum())},
        {"metric": "strict_LUSC_rows", "value": int((eligible["AUDIT_LABEL"] == "LUSC").sum())},
        {"metric": "strict_primary_rows", "value": len(primary)},
        {"metric": "strict_nonprimary_or_unknown_rows", "value": len(nonprimary)},
        {"metric": "strict_unique_patients", "value": ids_sample.nunique(dropna=True)},
        {"metric": "repeated_patient_rows", "value": len(duplicate_ids)},
        {"metric": "patient_sample_id_sets_equal", "value": details["ids"]["patient_sample_id_sets_equal"]},
        {"metric": "feature_sample_id_sets_equal", "value": details["ids"]["feature_sample_id_sets_equal"]}
    ]
    pd.DataFrame(summary_rows).to_csv(SUMMARY, index=False)

    report_lines = [
        "LUAD/LUSC BASELINE READINESS AUDIT V4",
        "=" * 80,
        json.dumps(details, indent=2, ensure_ascii=False, default=str),
        "",
        "COHORT DECISION RULES",
        "1. Primary-only analysis should be the default for a diagnosis-time subtype baseline.",
        "2. Keep metastasis and recurrence in separate sensitivity cohorts unless the intended use explicitly includes them.",
        "3. Do not use label fields, stage, survival, treatment, mutation/TMB, gene-panel, purity, or coverage as clinical baseline predictors.",
        "4. Choose and document a prediction timepoint before feature selection.",
        "5. Check missingness and category harmonization before any cross-cohort combination.",
        "6. The two overlapping TCGA participant barcodes reported in V3 require direct inspection before combining TCGA cohorts.",
        "7. This audit does not train a model or alter existing model/evaluation artifacts."
    ]
    REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print("Report:", REPORT)
    print("Summary:", SUMMARY)
    print("Details:", DETAILS)
    print("Eligible manifest:", manifest_path)
    print("Primary-only manifest:", primary_path)
    print("\nKEY FINDINGS")
    print(json.dumps({
        "strict_labels": details["strict_label_distribution"],
        "sample_types": details["sample_type_distribution"],
        "primary_only_rows": len(primary),
        "nonprimary_or_unknown_rows": len(nonprimary),
        "unique_patients": details["ids"]["eligible_unique_patients"],
        "repeated_patient_rows": details["ids"]["eligible_rows_with_repeated_patient_id"],
        "id_sets_equal": details["ids"]
    }, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()
