import csv
import json
import re
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2")
REPORT = PROJECT / "backend" / "Clinical" / "artifacts" / "dataset_feature_inventory" / "deep_dive"
DATA = PROJECT.parent / "datasets"

MSK_MANIFEST = REPORT / "msk_luad_lusc_primary_sample_manifest_v4.csv"
MSK_PATIENT = DATA / "msk_chord_2024" / "msk_chord_out" / "nsclc_clinical_patient.csv"
TCGA_FILES = {
    "TCGA-LUAD": DATA / "TCGA-LUAD" / "42bf5eb2-bc49-45be-b18a-290f712b006c" / "nationwidechildrens.org_clinical_patient_luad.txt",
    "TCGA-LUSC": DATA / "TCGA-LUSC" / "7aaee8d4-3bea-4dfc-b392-c7aaa3e9a550" / "nationwidechildrens.org_clinical_patient_lusc.txt",
}

MISSING = {"", "NA", "N/A", "NAN", "NONE", "NULL", "[NOT AVAILABLE]",
           "[NOT APPLICABLE]", "NOT AVAILABLE", "NOT APPLICABLE", "UNKNOWN"}

def clean(v):
    if v is None or pd.isna(v):
        return None
    s = str(v).strip()
    if s.upper() in MISSING:
        return None
    return s

def number(v):
    s = clean(v)
    if s is None:
        return None
    s = s.replace(",", "")
    try:
        n = float(s)
        return n if n == n and abs(n) != float("inf") else None
    except (ValueError, TypeError):
        return None

def first(row, *names):
    for name in names:
        if name in row:
            v = clean(row[name])
            if v is not None:
                return v
    return None

def canonical_sex(v):
    if not v:
        return "Unknown"
    x = v.strip().lower()
    if x in {"female", "f", "woman"}:
        return "Female"
    if x in {"male", "m", "man"}:
        return "Male"
    if x in {"other", "nonbinary", "non-binary"}:
        return "Other/Not reported"
    return "Unknown"

def canonical_smoking(v):
    if not v:
        return "Unknown"
    x = re.sub(r"[_-]+", " ", v.strip().lower())
    # Never collapse a combined former/current category into either category.
    if ("former" in x and "current" in x) or ("former/current" in x):
        return "Unknown"
    if "never" in x or x in {"non smoker", "nonsmoker"}:
        return "Never"
    if "current" in x or "active smoker" in x:
        return "Current"
    if "former" in x or "ex smoker" in x or "previous smoker" in x:
        return "Former"
    return "Unknown"

def detailed_smoking(v):
    """Preserve detailed source categories without splitting combined MSK labels."""
    if not v:
        return "Unknown"

    raw = v.strip()
    x = re.sub(r"[_-]+", " ", raw.lower()).strip()

    # TCGA CDE 2181650: retain the source-specific quit-duration distinctions.
    tcga_codes = {
        "1": "Never smoker (lifelong)",
        "2": "Current smoker",
        "3": "Former smoker (quit >15 years)",
        "4": "Former smoker (quit <=15 years)",
        "5": "Former smoker (duration unspecified)",
    }
    if raw in tcga_codes:
        return tcga_codes[raw]

    if "former" in x and "current" in x:
        return "Former/Current smoker (combined source category)"
    if "never" in x or x in {"non smoker", "nonsmoker", "lifelong non smoker"}:
        return "Never smoker (lifelong)"
    if "current" in x or "active smoker" in x:
        return "Current smoker"
    if "former" in x or "ex smoker" in x or "previous smoker" in x:
        return "Former smoker (duration unspecified)"
    if x in {"[not available]", "[unknown]", "[discrepancy]", "unknown"}:
        return "Unknown"
    return "Unmapped source category: " + raw

def smoking_source_code(v):
    """Return a TCGA numeric code as text; never coerce non-code source labels."""
    if v is None:
        return ""
    raw = str(v).strip()
    return raw if raw in {"1", "2", "3", "4", "5"} else ""

def canonical_label(v):
    if not v:
        return None
    x = v.strip().upper()
    if x == "LUAD" or "ADENOCARCINOMA" in x:
        return "LUAD"
    if x == "LUSC" or "SQUAMOUS CELL CARCINOMA" in x:
        return "LUSC"
    return None

def canonical_vital(v):
    if not v:
        return "Unknown"
    x = v.strip().lower()
    if x == "alive":
        return "Alive"
    if x == "dead":
        return "Dead"
    return "Unknown"

def kps_from_ecog(ecog):
    # Approximate ordinal mapping, not an exact clinical equivalence.
    mapping = {0: 100, 1: 80, 2: 60, 3: 40, 4: 20, 5: 0}
    n = number(ecog)
    return mapping.get(int(n)) if n is not None and n.is_integer() else None

def normalized_row(cohort, patient_id, sample_id, label, age, sex, smoking_raw,
                    smoking_duration, cessation_years, pack_years, packs_per_day,
                    ecog, kps, days_to_death, vital, source_file,
                    label_source, race=None, ethnicity=None):
    age_n = number(age)
    age_flag = (
        "missing" if age_n is None else
        "implausible_review" if age_n < 18 or age_n > 100 else
        "ok"
    )
    duration_n = number(smoking_duration)
    cessation_n = number(cessation_years)
    pack_n = number(pack_years)
    ppd_n = number(packs_per_day)
    ecog_n = number(ecog)
    kps_n = number(kps)

    # Negative or out-of-range source values are not silently retained as valid features.
    duration_n = duration_n if duration_n is not None and duration_n >= 0 else None
    cessation_n = cessation_n if cessation_n is not None and cessation_n >= 0 else None
    pack_n = pack_n if pack_n is not None and pack_n >= 0 else None
    ppd_n = ppd_n if ppd_n is not None and ppd_n >= 0 else None
    ecog_n = ecog_n if ecog_n is not None and 0 <= ecog_n <= 5 else None
    kps_n = kps_n if kps_n is not None and 0 <= kps_n <= 100 else None

    if kps_n is None:
        kps_n = kps_from_ecog(str(ecog_n)) if ecog_n is not None else None
        kps_source = "ECOG-derived KPS" if kps_n is not None else "Unknown"
    else:
        kps_source = "KPS"

    raw_smoking = clean(smoking_raw)
    return {
        "COHORT": cohort,
        "PATIENT_ID": patient_id,
        "SAMPLE_ID": sample_id,
        "LABEL": label,
        "LABEL_SOURCE": label_source,
        "age_at_diagnosis_years": age_n,
        "age_quality_flag": age_flag,
        "sex": canonical_sex(sex),
        "smoking_status": canonical_smoking(raw_smoking),
        "smoking_status_source_raw": raw_smoking,
        "smoking_status_source_code": smoking_source_code(raw_smoking),
        "smoking_status_detailed": detailed_smoking(raw_smoking),
        "smoking_duration_years": duration_n,
        "years_since_smoking_cessation": cessation_n,
        "pack_years": pack_n,
        "packs_per_day": ppd_n,
        "performance_status_ecog": ecog_n,
        "performance_status_kps": kps_n,
        "performance_status_source": kps_source,
        "days_to_death": number(days_to_death),
        "vital_status": canonical_vital(vital),
        "race": clean(race),
        "ethnicity": clean(ethnicity),
        "CT_timing_assumption": "Initial pathologic diagnosis assumed",
        "SOURCE_FILE": str(source_file),
    }

def read_msk():
    manifest = pd.read_csv(MSK_MANIFEST, dtype=str, keep_default_na=False)
    patient = pd.read_csv(MSK_PATIENT, dtype=str, keep_default_na=False)
    manifest.columns = [str(c).strip().upper() for c in manifest.columns]
    patient.columns = [str(c).strip().upper() for c in patient.columns]

    required = {"PATIENT_ID", "SAMPLE_ID", "AUDIT_LABEL"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"MSK manifest lacks required columns: {sorted(missing)}")
    if "PATIENT_ID" not in patient.columns:
        raise ValueError("MSK patient clinical table lacks PATIENT_ID")

    if manifest["PATIENT_ID"].astype(str).str.strip().eq("").any():
        raise ValueError("Blank PATIENT_ID in MSK manifest")
    if manifest["SAMPLE_ID"].astype(str).str.strip().eq("").any():
        raise ValueError("Blank SAMPLE_ID in MSK manifest")
    if manifest["PATIENT_ID"].duplicated().any() or manifest["SAMPLE_ID"].duplicated().any():
        raise ValueError("Duplicate patient or sample IDs in MSK primary manifest")
    if patient["PATIENT_ID"].duplicated().any():
        raise ValueError("Duplicate PATIENT_ID in MSK patient clinical table")

    patient = patient.rename(columns={c: "CLIN_" + c for c in patient.columns if c != "PATIENT_ID"})
    joined = manifest.merge(patient, on="PATIENT_ID", how="left", validate="one_to_one", indicator=True)
    unmatched = int((joined["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(f"{unmatched} MSK manifest rows did not match the patient table")
    joined = joined.drop(columns=["_merge"])

    out = []
    for _, r in joined.iterrows():
        row = r.to_dict()
        label = canonical_label(first(row, "AUDIT_LABEL"))
        if label not in {"LUAD", "LUSC"}:
            raise ValueError(f"Invalid MSK label for patient {row.get('PATIENT_ID')}: {row.get('AUDIT_LABEL')}")

        smoking_raw = first(row, "CLIN_SMOKING_PREDICTIONS_3_CLASSES",
                            "SMOKING_PREDICTIONS_3_CLASSES")
        out.append(normalized_row(
            cohort="MSK",
            patient_id=first(row, "PATIENT_ID"),
            sample_id=first(row, "SAMPLE_ID"),
            label=label,
            age=first(row, "CLIN_CURRENT_AGE_DEID", "CURRENT_AGE_DEID"),
            sex=first(row, "CLIN_GENDER", "GENDER"),
            smoking_raw=smoking_raw,
            smoking_duration=None,
            cessation_years=None,
            pack_years=None,
            packs_per_day=None,
            ecog=first(row, "CLIN_EASTERN_CANCER_ONCOLOGY_GROUP",
                       "CLIN_ECOG_SCORE", "EASTERN_CANCER_ONCOLOGY_GROUP", "ECOG_SCORE"),
            kps=first(row, "CLIN_KARNOFSKY_PERFORMANCE_SCORE",
                      "CLIN_KARNOFSKY_SCORE", "KARNOFSKY_PERFORMANCE_SCORE", "KARNOFSKY_SCORE"),
            days_to_death=first(row, "CLIN_DAYS_TO_DEATH", "DAYS_TO_DEATH",
                                "CLIN_DEATH_DAYS_TO", "DEATH_DAYS_TO"),
            vital=first(row, "CLIN_VITAL_STATUS", "VITAL_STATUS"),
            source_file=MSK_PATIENT,
            label_source="MSK primary manifest AUDIT_LABEL",
            race=first(row, "CLIN_RACE", "RACE"),
            ethnicity=first(row, "CLIN_ETHNICITY", "ETHNICITY"),
        ))
    return out

def read_tcga(cohort, path):
    # Legacy TCGA tables: row 1 legacy field names, row 2 alternate field names,
    # row 3 CDE metadata, records from row 4. Tab-separated.
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if len(rows) < 4:
        raise ValueError(f"Unexpected TCGA clinical file layout: {path}")

    headers = [h.strip() for h in rows[0]]
    if "bcr_patient_barcode" not in headers:
        raise ValueError(f"bcr_patient_barcode missing from header: {path}")

    def get(fields, *names):
        for name in names:
            if name in headers:
                idx = headers.index(name)
                if idx < len(fields):
                    v = clean(fields[idx])
                    if v is not None:
                        return v
        return None

    out = []
    skipped = []
    for line_number, fields in enumerate(rows[3:], start=4):
        if not fields or not any(str(x).strip() for x in fields):
            continue
        barcode = get(fields, "bcr_patient_barcode")
        if not barcode or not re.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", barcode):
            skipped.append(line_number)
            continue

        label_raw = get(fields, "histologic_diagnosis", "diagnosis", "histological_type")
        label = canonical_label(label_raw)
        if label not in {"LUAD", "LUSC"}:
            raise ValueError(f"Unresolved TCGA histology at {path}:{line_number}: {label_raw}")

        age = number(get(fields, "age_at_initial_pathologic_diagnosis",
                         "age_at_diagnosis", "age"))
        birth_days = number(get(fields, "days_to_birth", "birth_days_to"))
        if age is None and birth_days is not None:
            age = abs(birth_days) / 365.25

        diagnosis_year = number(get(fields, "year_of_initial_pathologic_diagnosis",
                                     "initial_pathologic_dx_year"))
        started_age = number(get(fields, "tobacco_smoking_age_started",
                                  "age_began_smoking_in_years"))
        started_year = number(get(fields, "tobacco_smoking_year_started",
                                  "year_of_tobacco_smoking_onset"))
        stopped_year = number(get(fields, "tobacco_smoking_year_stopped",
                                  "stopped_smoking_year"))
        duration = None
        cessation = None

        if started_age is not None and age is not None and 0 <= started_age <= age:
            duration = age - started_age
        elif started_year is not None and diagnosis_year is not None and 0 <= diagnosis_year - started_year <= 100:
            duration = diagnosis_year - started_year

        if stopped_year is not None and diagnosis_year is not None:
            if stopped_year <= diagnosis_year and stopped_year > 1900:
                cessation = diagnosis_year - stopped_year

        smoking_raw = get(fields, "tobacco_smoking_history", "tobacco_smoking_history_indicator",
                          "tobacco_smoking_status", "smoking_status")
        pack_years = get(fields, "tobacco_smoking_pack_years_smoked",
                         "number_pack_years_smoked", "pack_years")
        ecog = get(fields, "eastern_cancer_oncology_group", "ecog_score")
        kps = get(fields, "karnofsky_performance_score", "karnofsky_score")
        days_to_death = get(fields, "days_to_death", "death_days_to")
        vital = get(fields, "vital_status")
        sex = get(fields, "gender")

        # Year-based derived features are only retained when the source has the
        # necessary dates and the result is plausible. They remain approximate.
        out.append(normalized_row(
            cohort=cohort,
            patient_id=barcode,
            sample_id=None,
            label=label,
            age=age,
            sex=sex,
            smoking_raw=smoking_raw,
            smoking_duration=duration,
            cessation_years=cessation,
            pack_years=pack_years,
            packs_per_day=get(fields, "packs_per_day", "cigarettes_per_day_as_packs"),
            ecog=ecog,
            kps=kps,
            days_to_death=days_to_death,
            vital=vital,
            source_file=path,
            label_source=f"TCGA patient histology: {label_raw}",
            race=get(fields, "race"),
            ethnicity=get(fields, "ethnicity"),
        ))

    ids = [r["PATIENT_ID"] for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate TCGA participant barcode in {cohort}")
    return out, {"physical_lines": len(rows), "parsed_participants": len(out),
                 "nonparticipant_or_unparsed_line_numbers": skipped}

def main():
    msk = read_msk()
    tcga = []
    tcga_parse = {}
    for cohort, path in TCGA_FILES.items():
        records, audit = read_tcga(cohort, path)
        tcga.extend(records)
        tcga_parse[cohort] = audit

    records = msk + tcga
    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No records were compiled")
    if df["PATIENT_ID"].isna().any() or df["LABEL"].isna().any():
        raise ValueError("Missing participant ID or label in compiled data")
    if not set(df["LABEL"].unique()).issubset({"LUAD", "LUSC"}):
        raise ValueError("Unexpected target labels")
    if df.duplicated(["COHORT", "PATIENT_ID"]).any():
        raise ValueError("Duplicate participant within a cohort")

    # Keep source dataset, model matrix and outcome data separate.
    df.to_csv(ROOT / "compiled_source_dataset.csv", index=False)

    predictor_cols = [
        "age_at_diagnosis_years",
        "sex",
        "smoking_status_detailed",
        "smoking_duration_years",
        "years_since_smoking_cessation",
        "pack_years",
        "packs_per_day",
        "performance_status_kps",
    ]
    model_cols = ["COHORT", "PATIENT_ID", "LABEL"] + predictor_cols
    df[model_cols].to_csv(ROOT / "diagnosis_time_model_matrix.csv", index=False)

    outcome_cols = [
        "COHORT", "PATIENT_ID", "LABEL", "days_to_death", "vital_status",
        "performance_status_ecog", "performance_status_kps",
        "performance_status_source", "CT_timing_assumption", "SOURCE_FILE",
    ]
    df[outcome_cols].to_csv(ROOT / "outcome_analysis_dataset.csv", index=False)

    feature_dictionary = [
        {"field": "age_at_diagnosis_years", "type": "numeric", "role": "predictor",
         "notes": "MSK CURRENT_AGE_DEID; TCGA age or abs(days_to_birth)/365.25; implausible values flagged."},
        {"field": "sex", "type": "categorical", "allowed": ["Female", "Male", "Other/Not reported", "Unknown"], "role": "predictor"},
        {"field": "smoking_status", "type": "categorical", "allowed": ["Never", "Former", "Current", "Unknown"], "role": "audit/legacy comparison",
         "notes": "Legacy coarse mapping retained for comparison; do not use in place of smoking_status_detailed."},
        {"field": "smoking_status_source_raw", "type": "string", "role": "source provenance",
         "notes": "Original smoking history value preserved as read from source."},
        {"field": "smoking_status_source_code", "type": "string", "role": "source provenance",
         "notes": "TCGA numeric source code 1-5 preserved as text; blank for non-code source values."},
        {"field": "smoking_status_detailed", "type": "categorical", "role": "predictor",
         "allowed": ["Never smoker (lifelong)", "Current smoker", "Former smoker (quit >15 years)", "Former smoker (quit <=15 years)", "Former smoker (duration unspecified)", "Former/Current smoker (combined source category)", "Unknown"],
         "notes": "Detailed category derived from TCGA CDE 2181650 codes or explicit source text. Combined MSK former/current category remains combined. Unmapped values are labelled and must be reviewed."},
        {"field": "smoking_duration_years", "type": "numeric", "role": "predictor",
         "notes": "Only populated when derivable from source fields; not imputed during compilation."},
        {"field": "years_since_smoking_cessation", "type": "numeric", "role": "predictor",
         "notes": "Only derived where source cessation and diagnosis years are available."},
        {"field": "pack_years", "type": "numeric", "role": "predictor"},
        {"field": "packs_per_day", "type": "numeric", "role": "predictor",
         "notes": "Only populated from an explicitly available source field; not inferred from pack-years."},
        {"field": "performance_status_ecog", "type": "numeric 0-5", "role": "outcome/audit",
         "notes": "Retained in source and outcome tables; KPS is the normalized model feature."},
        {"field": "performance_status_kps", "type": "numeric 0-100", "role": "predictor",
         "notes": "Recorded KPS preferred; otherwise approximate ECOG mapping 0:100, 1:80, 2:60, 3:40, 4:20, 5:0."},
        {"field": "LABEL", "type": "categorical", "allowed": ["LUAD", "LUSC"], "role": "target only"},
        {"field": "days_to_death", "type": "numeric", "role": "outcome only",
         "notes": "Never use as a diagnosis-time predictor."},
        {"field": "vital_status", "type": "categorical", "allowed": ["Alive", "Dead", "Unknown"], "role": "outcome only",
         "notes": "Never use as a diagnosis-time predictor."},
        {"field": "race / ethnicity", "type": "categorical", "role": "fairness audit only"},
        {"field": "CT_timing_assumption", "type": "metadata", "role": "cohort assumption",
         "notes": "Initial pathologic diagnosis assumed; not independently verified."},
    ]
    (ROOT / "feature_dictionary.json").write_text(
        json.dumps(feature_dictionary, indent=2), encoding="utf-8"
    )

    missingness = {}
    for col in df.columns:
        missingness[col] = {
            "missing": int(df[col].isna().sum()),
            "missing_pct": round(float(df[col].isna().mean() * 100), 3),
        }

    counts = {}
    for (cohort, label), group in df.groupby(["COHORT", "LABEL"], dropna=False):
        counts[f"{cohort}|{label}"] = int(len(group))

    overlap = sorted(
        set(df.loc[df["COHORT"] == "TCGA-LUAD", "PATIENT_ID"]) &
        set(df.loc[df["COHORT"] == "TCGA-LUSC", "PATIENT_ID"])
    )

    audit = {
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_source_policy": "Source files read only; outputs written to new build directory.",
        "total_rows": int(len(df)),
        "cohort_label_counts": counts,
        "unique_patients_by_cohort": {
            c: int(df.loc[df["COHORT"] == c, "PATIENT_ID"].nunique())
            for c in sorted(df["COHORT"].unique())
        },
        "tcga_parse_audit": tcga_parse,
        "tcga_luad_lusc_overlap": overlap,
        "missingness": missingness,
        "age_quality_flags": {
            str(k): int(v) for k, v in df["age_quality_flag"].value_counts(dropna=False).items()
        },
        "smoking_status_source_raw_counts": {
            str(k): int(v) for k, v in df["smoking_status_source_raw"].fillna("<MISSING>").value_counts().items()
        },
        "important_limitations": [
            "No KNN imputer was fitted and no model was trained.",
            "Missing values are preserved for a later train-only fitted preprocessing pipeline.",
            "TCGA clinical file participant counts may differ from prior inventory; see tcga_parse_audit.",
            "CT timing is assumed, not verified.",
            "TCGA smoking status is only mapped when source values explicitly support a category.",
            "ECOG-to-KPS mapping is approximate.",
            "No MSK sample-level pack-years, smoking duration, or cessation values are inferred unless explicitly present in the selected patient table."
        ]
    }
    (ROOT / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("BUILD COMPLETE")
    print(f"Output directory: {ROOT}")
    print(f"Total rows: {len(df)}")
    print("\nCohort/label counts:")
    print(df.groupby(["COHORT", "LABEL"]).size().to_string())
    print("\nUnique patients by cohort:")
    print(df.groupby("COHORT")["PATIENT_ID"].nunique().to_string())
    print("\nTCGA parse audit:")
    print(json.dumps(tcga_parse, indent=2))
    print(f"\nTCGA participant overlap: {len(overlap)}")
    if overlap:
        print("OVERLAP BAR CODES:", ", ".join(overlap))
    print("\nModel matrix columns:")
    print(", ".join(model_cols))
    print("\nMissingness (%), model features:")
    print(df[ predictor_cols ].isna().mean().mul(100).round(2).to_string())
    print("\nNo model was trained. No imputer was fitted.")

if __name__ == "__main__":
    main()
