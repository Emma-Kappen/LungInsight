from pathlib import Path
from datetime import datetime, timezone
import json
import re
import sys

import pandas as pd

ROOT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight")
DATASETS = ROOT / "datasets"
OUT_ROOT = ROOT / "backend" / "Clinical" / "artifacts"

SOURCES = {
    "LUAD": DATASETS / "TCGA-LUAD" /
        "42bf5eb2-bc49-45be-b18a-290f712b006c" /
        "nationwidechildrens.org_clinical_patient_luad.txt",
    "LUSC": DATASETS / "TCGA-LUSC" /
        "7aaee8d4-3bea-4dfc-b392-c7aaa3e9a550" /
        "nationwidechildrens.org_clinical_patient_lusc.txt",
}

PATTERNS = {
    "patient_identifier": r"barcode|patient.*id|case.*id",
    "age": r"age|diagnosis.*age|birth.*days",
    "sex": r"gender|sex",
    "race_ethnicity": r"race|ethnic",
    "smoking_status": r"smok|tobacco|cigarette",
    "pack_years": r"pack.?year|pack_year",
    "smoking_duration": r"smok.*duration|years.*smok|duration.*smok",
    "smoking_start": r"smok.*start|start.*smok|year.*smok.*start",
    "smoking_stop": r"smok.*stop|stop.*smok|cessation|quit.*smok",
    "performance_status": r"karnofsky|performance.*status|ecog",
    "diagnosis_date": r"diagnos.*date|initial.*diagnos|date.*diagnos",
    "treatment": r"treatment|therapy|neoadjuvant|adjuvant|radiation|chemotherapy",
    "surgery": r"surg|resection|operation",
    "vital_status": r"vital.*status|death.*status",
    "follow_up": r"follow.?up|last.*contact|days.*last",
}

def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

def read_table(path):
    # Preserve raw values; do not coerce clinical fields into numeric values.
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
        encoding="utf-8-sig",
    )
    df.columns = [str(c).strip() for c in df.columns]

    # TCGA legacy files may contain a CDE-description row immediately
    # beneath the header. Keep it out of patient-level counts.
    id_cols = [
        c for c in df.columns
        if re.search(r"barcode|patient.*id", c, re.I)
    ]
    if id_cols:
        mask = pd.Series(False, index=df.index)
        for col in id_cols:
            values = df[col].astype(str).str.strip()
            mask |= values.str.match(
                r"^(CDE_ID|BCR|.*description.*|.*definition.*)$",
                case=False,
            )
        df = df.loc[~mask].copy()

    return df

def find_id_column(columns):
    preferred = [
        "bcr_patient_barcode",
        "patient_barcode",
        "patient.bcr_patient_barcode",
        "case_submitter_id",
    ]
    by_lower = {c.lower(): c for c in columns}
    for name in preferred:
        if name.lower() in by_lower:
            return by_lower[name.lower()]

    candidates = [
        c for c in columns
        if re.search(r"bcr_patient_barcode|patient.*barcode", c, re.I)
    ]
    return candidates[0] if candidates else None

def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_ROOT / f"tcga_clinical_feature_inventory_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    inventory = []
    distributions = []
    patient_rows = []
    summaries = {}

    for cohort, path in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(f"{cohort} source not found: {path}")

        df = read_table(path)
        id_col = find_id_column(df.columns)

        if id_col:
            ids = df[id_col].astype(str).str.strip()
            valid_ids = ids[ids.ne("")]
            unique_ids = int(valid_ids.nunique())
            duplicate_rows = int(valid_ids.duplicated(keep=False).sum())
        else:
            ids = pd.Series([""] * len(df), index=df.index)
            unique_ids = 0
            duplicate_rows = 0

        summaries[cohort] = {
            "source_file": str(path),
            "rows_after_metadata_filter": int(len(df)),
            "column_count": int(len(df.columns)),
            "patient_id_column": id_col,
            "nonempty_patient_ids": int(ids.ne("").sum()),
            "unique_patient_ids": unique_ids,
            "rows_with_duplicate_patient_id": duplicate_rows,
        }

        for col in df.columns:
            values = df[col].map(clean)
            nonempty = values[values.ne("")]
            inventory.append({
                "cohort": cohort,
                "column": col,
                "nonempty_n": int(nonempty.size),
                "missing_n": int(len(df) - nonempty.size),
                "availability_pct": round(
                    100 * nonempty.size / len(df), 3
                ) if len(df) else 0,
                "unique_nonempty_values": int(nonempty.nunique()),
                "matched_categories": ";".join(
                    key for key, pattern in PATTERNS.items()
                    if re.search(pattern, col, re.I)
                ),
            })

        for category, pattern in PATTERNS.items():
            matched = [
                c for c in df.columns if re.search(pattern, c, re.I)
            ]

            for col in matched:
                values = df[col].map(clean)
                nonempty = values[values.ne("")]
                counts = nonempty.value_counts(dropna=False)

                for value, count in counts.head(40).items():
                    distributions.append({
                        "cohort": cohort,
                        "category": category,
                        "column": col,
                        "value": value,
                        "count": int(count),
                    })

                for idx, value in values.items():
                    if value == "":
                        continue
                    patient_rows.append({
                        "cohort": cohort,
                        "patient_id": (
                            clean(df.at[idx, id_col]) if id_col else ""
                        ),
                        "category": category,
                        "column": col,
                        "value": value,
                    })

    pd.DataFrame(inventory).to_csv(
        out / "all_column_inventory.csv", index=False
    )
    pd.DataFrame(distributions).to_csv(
        out / "candidate_value_distributions.csv", index=False
    )
    pd.DataFrame(patient_rows).to_csv(
        out / "candidate_patient_values.csv", index=False
    )

    (out / "summary.json").write_text(
        json.dumps({
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_summaries": summaries,
            "categories": PATTERNS,
            "limitations": [
                "Column-name matching is exploratory, not semantic validation.",
                "No fields are approved for model training by this audit.",
                "No missing values are imputed.",
                "No patient records or source files are modified.",
                "Timing and treatment leakage require a separate review.",
            ],
        }, indent=2),
        encoding="utf-8",
    )

    lines = [
        "TCGA CLINICAL FEATURE INVENTORY",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    for cohort, summary in summaries.items():
        lines.extend([
            f"[{cohort}]",
            f"Source: {summary['source_file']}",
            f"Rows: {summary['rows_after_metadata_filter']}",
            f"Columns: {summary['column_count']}",
            f"Patient ID column: {summary['patient_id_column']}",
            f"Nonempty patient IDs: {summary['nonempty_patient_ids']}",
            f"Unique patient IDs: {summary['unique_patient_ids']}",
            f"Rows with duplicated patient IDs: {summary['rows_with_duplicate_patient_id']}",
            "",
        ])

    (out / "summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"OUTPUT_DIR={out}")

if __name__ == "__main__":
    main()
