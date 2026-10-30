from pathlib import Path
import json
import re
from datetime import datetime, timezone

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "backend" / "Clinical" / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "LUAD": ROOT / "datasets" / "TCGA-LUAD" /
        "42bf5eb2-bc49-45be-b18a-290f712b006c" /
        "nationwidechildrens.org_clinical_patient_luad.txt",
    "LUSC": ROOT / "datasets" / "TCGA-LUSC" /
        "7aaee8d4-3bea-4dfc-b392-c7aaa3e9a550" /
        "nationwidechildrens.org_clinical_patient_lusc.txt",
}

PATIENT_ID = "bcr_patient_barcode"
START_COL = "tobacco_smoking_year_started"
STOP_COL = "tobacco_smoking_year_stopped"

# Only columns with explicit smoking-status semantics are candidates.
# The numeric "tobacco_smoking_history_indicator" is NOT decoded here:
# its code definitions have not yet been verified.
STATUS_CANDIDATES = [
    "tobacco_smoking_status",
    "tobacco_smoking_history",
    "smoking_status",
    "smoking_history",
    "tobacco_smoking_history_indicator",
]

MISSING_TOKENS = {
    "", "NA", "N/A", "NAN", "NULL", "NONE", "[DISCREPANCY]",
    "[NOT AVAILABLE]", "[NOT REPORTED]", "[UNKNOWN]",
    "[NOT EVALUATED]", "[NOT APPLICABLE]",
}

def norm(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.upper() in MISSING_TOKENS:
        return ""
    if "[Discrepancy]" in text:
        return ""
    return text

def status_kind(value):
    """Classify only unambiguous textual status labels; never guess numeric codes."""
    text = norm(value).lower()
    if not text:
        return "unknown"

    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Explicit labels only. Avoid interpreting arbitrary numbers as status.
    if re.fullmatch(r"(never smoker|never smoked|non smoker|nonsmoker)", text):
        return "never"
    if re.fullmatch(r"(current smoker|current smoking|currently smoking|smokes)", text):
        return "current"
    if re.fullmatch(r"(former smoker|former smoking|ex smoker|ex smoker|previous smoker|previously smoked|quit smoking)", text):
        return "former"
    if text in {"1", "2", "3", "4", "5"}:
        return "unverified_code"
    return "unknown"

def load_clean(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, encoding="latin1",
                     keep_default_na=False, low_memory=False)
    raw_rows = len(df)

    # Remove TCGA CDE_ID metadata rows, including variants with surrounding spaces.
    cde_mask = df.apply(
        lambda col: col.astype(str).str.strip().str.upper().eq("CDE_ID")
    ).any(axis=1)
    cde_rows = int(cde_mask.sum())
    df = df.loc[~cde_mask].copy()

    # Convert discrepancy and common missing tokens to true missing values.
    for col in df.columns:
        df[col] = df[col].map(norm).replace("", pd.NA)

    if PATIENT_ID not in df.columns:
        raise KeyError(f"{path.name}: missing required column {PATIENT_ID}")

    invalid_id = df[PATIENT_ID].isna() | df[PATIENT_ID].astype(str).str.strip().eq("")
    invalid_id_rows = int(invalid_id.sum())
    df = df.loc[~invalid_id].copy()
    duplicate_ids = int(df[PATIENT_ID].duplicated().sum())

    return df, {
        "source": str(path),
        "raw_rows": raw_rows,
        "cde_metadata_rows_removed": cde_rows,
        "invalid_patient_id_rows_removed": invalid_id_rows,
        "rows_after_cleaning": int(len(df)),
        "unique_patient_ids": int(df[PATIENT_ID].nunique()),
        "duplicate_patient_ids": duplicate_ids,
    }

def process_cohort(label, path, report):
    df, audit = load_clean(path)
    cohort_report = {
        **audit,
        "status_column": None,
        "status_values": {},
        "status_interpretation": (
            "Only explicit textual labels are interpreted. Numeric smoking codes "
            "are left unresolved pending a verified data dictionary."
        ),
        "start_year_missing_before": None,
        "stop_year_missing_before": None,
        "start_year_set_to_zero": 0,
        "stop_year_set_to_zero": 0,
        "unresolved_numeric_status_codes": 0,
        "unclassified_status_values": {},
        "warnings": [],
    }

    for col in (START_COL, STOP_COL):
        if col not in df.columns:
            raise KeyError(f"{label}: required column not found: {col}")

    # Find a candidate status field with values. Prefer explicit status columns.
    status_col = None
    for candidate in STATUS_CANDIDATES:
        if candidate in df.columns and df[candidate].notna().any():
            status_col = candidate
            break

    cohort_report["status_column"] = status_col
    cohort_report["start_year_missing_before"] = int(df[START_COL].isna().sum())
    cohort_report["stop_year_missing_before"] = int(df[STOP_COL].isna().sum())

    if status_col is None:
        cohort_report["warnings"].append(
            "No candidate smoking-status column found. No year values changed."
        )
        status_kinds = pd.Series("unknown", index=df.index)
    else:
        status_values = df[status_col].fillna("<MISSING>").value_counts(dropna=False)
        cohort_report["status_values"] = {
            str(k): int(v) for k, v in status_values.items()
        }
        status_kinds = df[status_col].map(status_kind)

        numeric_codes = df.loc[
            status_kinds.eq("unverified_code"), status_col
        ].value_counts()
        cohort_report["unresolved_numeric_status_codes"] = int(numeric_codes.sum())

        unknown_values = df.loc[
            status_kinds.eq("unknown") & df[status_col].notna(), status_col
        ].value_counts()
        cohort_report["unclassified_status_values"] = {
            str(k): int(v) for k, v in unknown_values.items()
        }

    # "0" is a sentinel meaning NOT APPLICABLE, not a calendar year.
    # Never smoker: both start and stop are not applicable.
    never = status_kinds.eq("never")
    start_never = never & df[START_COL].isna()
    stop_never = never & df[STOP_COL].isna()
    df.loc[start_never, START_COL] = "0"
    df.loc[stop_never, STOP_COL] = "0"
    cohort_report["start_year_set_to_zero"] += int(start_never.sum())
    cohort_report["stop_year_set_to_zero"] += int(stop_never.sum())

    # Current smoker: a stop year is not applicable if it is missing.
    current = status_kinds.eq("current")
    stop_current = current & df[STOP_COL].isna()
    df.loc[stop_current, STOP_COL] = "0"
    cohort_report["stop_year_set_to_zero"] += int(stop_current.sum())

    # Former smokers' missing years remain missing. Unknown/unverified status
    # values remain missing. KNN imputation belongs inside the train/test pipeline.
    cohort_report["start_year_missing_after_rule"] = int(df[START_COL].isna().sum())
    cohort_report["stop_year_missing_after_rule"] = int(df[STOP_COL].isna().sum())
    cohort_report["former_smokers_with_missing_start"] = int(
        (status_kinds.eq("former") & df[START_COL].isna()).sum()
    )
    cohort_report["former_smokers_with_missing_stop"] = int(
        (status_kinds.eq("former") & df[STOP_COL].isna()).sum()
    )
    cohort_report["current_smokers_with_missing_start"] = int(
        (status_kinds.eq("current") & df[START_COL].isna()).sum()
    )
    cohort_report["never_smoker_rows"] = int(never.sum())
    cohort_report["current_smoker_rows"] = int(current.sum())
    cohort_report["former_smoker_rows"] = int(status_kinds.eq("former").sum())

    out_path = OUTPUT_DIR / f"tcga_{label.lower()}_smoking_years_prepared.csv"
    df.to_csv(out_path, index=False)
    cohort_report["output_csv"] = str(out_path)
    report["cohorts"][label] = cohort_report

def main():
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Audit and apply conservative smoking-year not-applicable rules.",
        "sentinel_definition": (
            "0 means NOT APPLICABLE for smoking start/stop year; it is not a calendar year."
        ),
        "rules": [
            "Never smoker + missing start year -> 0 sentinel.",
            "Never smoker + missing stop year -> 0 sentinel.",
            "Current smoker + missing stop year -> 0 sentinel.",
            "Former smoker missing years remain missing.",
            "Current smoker missing start year remains missing.",
            "Unknown status and unverified numeric codes are not decoded.",
            "No KNN imputation is performed in this preprocessing step.",
            "KNN imputation must be fitted on training data only and then applied to test data.",
        ],
        "cohorts": {},
    }

    for label, path in SOURCES.items():
        process_cohort(label, path, report)

    report_path = OUTPUT_DIR / "tcga_smoking_years_preparation_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    txt_path = OUTPUT_DIR / "tcga_smoking_years_preparation_audit.txt"
    lines = [
        "TCGA SMOKING START/STOP YEAR PREPARATION AUDIT",
        f"Created UTC: {report['created_utc']}",
        "",
        "Rule: 0 is a NOT-APPLICABLE sentinel, not a calendar year.",
        "No KNN imputation or model training was performed.",
        "",
    ]
    for label, item in report["cohorts"].items():
        lines += [
            f"===== {label} =====",
            f"Cleaned patients: {item['rows_after_cleaning']}",
            f"Status column selected: {item['status_column']}",
            f"Status value counts: {json.dumps(item['status_values'], ensure_ascii=False)}",
            f"Start-year missing before: {item['start_year_missing_before']}",
            f"Stop-year missing before: {item['stop_year_missing_before']}",
            f"Start-year values set to 0: {item['start_year_set_to_zero']}",
            f"Stop-year values set to 0: {item['stop_year_set_to_zero']}",
            f"Start-year missing after rules: {item['start_year_missing_after_rule']}",
            f"Stop-year missing after rules: {item['stop_year_missing_after_rule']}",
            f"Unverified numeric status-code rows: {item['unresolved_numeric_status_codes']}",
            f"Unclassified status values: {json.dumps(item['unclassified_status_values'], ensure_ascii=False)}",
            f"Output: {item['output_csv']}",
            "Warnings: " + ("; ".join(item["warnings"]) if item["warnings"] else "None"),
            "",
        ]

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print("Preparation completed.")
    print(f"Audit report: {txt_path}")
    print(f"Machine-readable report: {report_path}")
    for label, item in report["cohorts"].items():
        print(
            f"{label}: start-year zeroed={item['start_year_set_to_zero']}; "
            f"stop-year zeroed={item['stop_year_set_to_zero']}; "
            f"status column={item['status_column']}"
        )

if __name__ == "__main__":
    main()
