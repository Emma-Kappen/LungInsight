from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight")
OUT_ROOT = ROOT / "backend" / "Clinical" / "artifacts"

SOURCES = {
    "LUAD": ROOT / "datasets" / "TCGA-LUAD" /
        "42bf5eb2-bc49-45be-b18a-290f712b006c" /
        "nationwidechildrens.org_clinical_patient_luad.txt",
    "LUSC": ROOT / "datasets" / "TCGA-LUSC" /
        "7aaee8d4-3bea-4dfc-b392-c7aaa3e9a550" /
        "nationwidechildrens.org_clinical_patient_lusc.txt",
}

FIELDS = [
    "bcr_patient_barcode",
    "tobacco_smoking_history_indicator",
    "tobacco_smoking_pack_years_smoked",
    "tobacco_smoking_year_started",
    "tobacco_smoking_year_stopped",
    "initial_pathologic_dx_year",
    "age_at_initial_pathologic_diagnosis",
]

STATUS = {
    "1": "lifelong_non_smoker",
    "2": "current_smoker",
    "3": "former_smoker_quit_gt_15y",
    "4": "former_smoker_quit_le_15y",
    "5": "former_smoker_duration_unknown",
    "7": "smoking_history_not_documented",
}

MISSING = {
    "", "[Not Available]", "[Not Applicable]", "[Unknown]",
    "[Discrepancy]", "[Not Evaluated]", "[Pending]",
    "[Not Reported]", "[Restricted]",
}

def numeric(series):
    s = series.astype(str).str.strip()
    s = s.where(~s.isin(MISSING), "")
    return pd.to_numeric(s, errors="coerce")

def load_source(cohort, path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = [c for c in FIELDS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{cohort}: required columns absent: {missing_cols}"
        )

    # Only real TCGA patient records; removes CDE and header-description rows.
    df = df[df["bcr_patient_barcode"].str.match(
        r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$",
        case=False,
    )].copy()

    if df["bcr_patient_barcode"].duplicated().any():
        duplicates = df.loc[
            df["bcr_patient_barcode"].duplicated(keep=False),
            "bcr_patient_barcode",
        ].tolist()
        raise ValueError(
            f"{cohort}: duplicate patient barcodes: {duplicates[:20]}"
        )

    df.insert(0, "cohort", cohort)
    return df

def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_ROOT / f"tcga_smoking_consistency_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    patient_frames = []
    summaries = {}
    cross_tabs = []
    pack_flags = []
    chronology_flags = []

    for cohort, path in SOURCES.items():
        df = load_source(cohort, path)

        for field in FIELDS[1:]:
            df[field] = df[field].astype(str).str.strip()

        df["smoking_status_label"] = df[
            "tobacco_smoking_history_indicator"
        ].map(STATUS).fillna("unmapped_or_missing")

        for field in [
            "tobacco_smoking_pack_years_smoked",
            "tobacco_smoking_year_started",
            "tobacco_smoking_year_stopped",
            "initial_pathologic_dx_year",
            "age_at_initial_pathologic_diagnosis",
        ]:
            df[field + "_numeric"] = numeric(df[field])

        pack = df["tobacco_smoking_pack_years_smoked_numeric"]
        start = df["tobacco_smoking_year_started_numeric"]
        stop = df["tobacco_smoking_year_stopped_numeric"]
        dx = df["initial_pathologic_dx_year_numeric"]
        status = df["tobacco_smoking_history_indicator"]

        checks = {
            "patient_count": int(len(df)),
            "unique_patient_count": int(
                df["bcr_patient_barcode"].nunique()
            ),
            "status_code_counts": {
                str(k): int(v)
                for k, v in status.value_counts().items()
            },
            "known_status_count": int(status.isin(STATUS).sum()),
            "pack_year_numeric_count": int(pack.notna().sum()),
            "pack_year_availability_pct": round(
                100 * pack.notna().mean(), 2
            ),
            "smoking_start_year_count": int(start.notna().sum()),
            "smoking_stop_year_count": int(stop.notna().sum()),
            "diagnosis_year_count": int(dx.notna().sum()),
            "pack_year_min": (
                float(pack.min()) if pack.notna().any() else None
            ),
            "pack_year_median": (
                float(pack.median()) if pack.notna().any() else None
            ),
            "pack_year_max": (
                float(pack.max()) if pack.notna().any() else None
            ),
        }

        summaries[cohort] = checks

        # Patient-level contradictions and review flags.
        never_with_pack = (
            status.eq("1") & pack.notna() & pack.gt(0)
        )
        never_with_start = status.eq("1") & start.notna()
        never_with_stop = status.eq("1") & stop.notna()
        negative_pack = pack.notna() & pack.lt(0)
        high_pack = pack.gt(100)

        for idx in df.index:
            reasons = []
            if never_with_pack.loc[idx]:
                reasons.append("never_smoker_with_pack_years")
            if never_with_start.loc[idx]:
                reasons.append("never_smoker_with_start_year")
            if never_with_stop.loc[idx]:
                reasons.append("never_smoker_with_stop_year")
            if negative_pack.loc[idx]:
                reasons.append("negative_pack_years")
            if high_pack.loc[idx]:
                reasons.append("pack_years_gt_100_review")
            if reasons:
                pack_flags.append({
                    "cohort": cohort,
                    "patient_id": df.at[idx, "bcr_patient_barcode"],
                    "smoking_status_code": status.loc[idx],
                    "smoking_status_label":
                        df.at[idx, "smoking_status_label"],
                    "pack_years": pack.loc[idx],
                    "start_year": start.loc[idx],
                    "stop_year": stop.loc[idx],
                    "diagnosis_year": dx.loc[idx],
                    "flags": ";".join(reasons),
                })

        start_after_dx = start.notna() & dx.notna() & start.gt(dx)
        stop_after_dx = stop.notna() & dx.notna() & stop.gt(dx)
        stop_before_start = (
            start.notna() & stop.notna() & stop.lt(start)
        )
        current_with_stop = status.eq("2") & stop.notna()
        former_without_stop = status.isin(["3", "4", "5"]) & stop.isna()
        former_stop_after_dx = (
            status.isin(["3", "4", "5"]) &
            stop.notna() & dx.notna() & stop.gt(dx)
        )

        chronology_conditions = {
            "smoking_start_after_diagnosis": start_after_dx,
            "smoking_stop_after_diagnosis": stop_after_dx,
            "stop_year_before_start_year": stop_before_start,
            "current_smoker_with_stop_year": current_with_stop,
            "former_smoker_without_stop_year": former_without_stop,
            "former_smoker_stop_after_diagnosis":
                former_stop_after_dx,
        }

        for flag, mask in chronology_conditions.items():
            for idx in df.index[mask]:
                chronology_flags.append({
                    "cohort": cohort,
                    "patient_id": df.at[idx, "bcr_patient_barcode"],
                    "smoking_status_code": status.loc[idx],
                    "smoking_status_label":
                        df.at[idx, "smoking_status_label"],
                    "start_year": start.loc[idx],
                    "stop_year": stop.loc[idx],
                    "diagnosis_year": dx.loc[idx],
                    "flag": flag,
                })

        # Status x pack-year availability.
        tab = pd.crosstab(
            df["smoking_status_label"],
            pack.notna(),
        )
        for label, row in tab.iterrows():
            cross_tabs.append({
                "cohort": cohort,
                "status": label,
                "pack_year_missing": int(row.get(False, 0)),
                "pack_year_present": int(row.get(True, 0)),
            })

        df["pack_years_numeric"] = pack
        df["start_year_numeric"] = start
        df["stop_year_numeric"] = stop
        df["diagnosis_year_numeric"] = dx
        patient_frames.append(df)

    patients = pd.concat(patient_frames, ignore_index=True)

    patients.to_csv(out / "patient_level_smoking_audit.csv", index=False)
    pd.DataFrame(cross_tabs).to_csv(
        out / "status_pack_year_cross_tab.csv", index=False
    )
    pd.DataFrame(pack_flags).to_csv(
        out / "pack_year_review_flags.csv", index=False
    )
    pd.DataFrame(chronology_flags).to_csv(
        out / "smoking_chronology_flags.csv", index=False
    )

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_summaries": summaries,
        "status_code_mapping": STATUS,
        "limitations": [
            "Code mapping follows the TCGA enrollment form.",
            "Pack-year values above 100 are review flags, not automatic errors.",
            "A smoking year after diagnosis is flagged, not automatically deleted.",
            "A stop year may reflect post-diagnosis cessation.",
            "No imputation, recoding, clipping, or source modification occurs.",
        ],
    }

    (out / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    lines = [
        "TCGA SMOKING FEATURE CONSISTENCY AUDIT",
        f"Generated UTC: {result['generated_utc']}",
        "",
    ]

    for cohort, s in summaries.items():
        lines.extend([
            f"[{cohort}]",
            f"Patients: {s['patient_count']}",
            f"Known smoking status: {s['known_status_count']}",
            f"Numeric pack-years: {s['pack_year_numeric_count']}",
            f"Pack-year availability: {s['pack_year_availability_pct']}%",
            f"Start year present: {s['smoking_start_year_count']}",
            f"Stop year present: {s['smoking_stop_year_count']}",
            f"Diagnosis year present: {s['diagnosis_year_count']}",
            f"Pack-year median: {s['pack_year_median']}",
            f"Pack-year maximum: {s['pack_year_max']}",
            "",
        ])

    lines.extend([
        f"Pack-year review flags: {len(pack_flags)}",
        f"Chronology flags: {len(chronology_flags)}",
        "",
        f"OUTPUT_DIR={out}",
    ])

    (out / "summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines))

if __name__ == "__main__":
    main()
