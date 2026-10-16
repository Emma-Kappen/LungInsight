from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .ingestion import load_msk_tables, load_tcga_survival_cohort, msk_lung_patient_labels


def _missingness(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {
        name: round(
            float(
                (
                    frame[name].isna()
                    | frame[name].astype("string").str.strip().eq("")
                ).mean()
            ),
            4,
        )
        for name in columns
        if name in frame
    }


def _category_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].fillna("<missing>").value_counts(dropna=False).items()
    }


def _numeric_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").replace(
        [float("inf"), float("-inf")],
        pd.NA,
    ).dropna()
    if values.empty:
        return {
            "valid_count": 0,
            "minimum": None,
            "q25": None,
            "median": None,
            "q75": None,
            "maximum": None,
        }
    return {
        "valid_count": int(len(values)),
        "minimum": float(values.min()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "maximum": float(values.max()),
    }


def _followup_summary(frame: pd.DataFrame) -> dict[str, float | None]:
    usable = pd.to_numeric(
        frame.loc[frame["endpoint_issue"].isna(), "overall_survival_time_days"],
        errors="coerce",
    ).dropna()
    if usable.empty:
        return {"minimum": None, "q25": None, "median": None, "q75": None, "maximum": None}
    return {
        "minimum": float(usable.min()),
        "q25": float(usable.quantile(0.25)),
        "median": float(usable.median()),
        "q75": float(usable.quantile(0.75)),
        "maximum": float(usable.max()),
    }


def _tcga_summary(root: Path, cohort: str) -> dict[str, Any]:
    frame = load_tcga_survival_cohort(root / cohort, cohort)
    complete = frame.loc[
        frame["endpoint_issue"].isna()
        & frame["overall_survival_time_days"].notna()
        & frame["overall_survival_event"].notna()
    ]
    manifest_files = sorted((root / cohort).glob("gdc_manifest*.txt"))
    metadata_files = sorted((root / cohort).glob("metadata.repository*.json"))
    access_counts: dict[str, int] = {}
    metadata_count: int | None = None
    metadata_data_type_counts: dict[str, int] = {}
    metadata_entity_type_counts: dict[str, int] = {}
    unique_case_entities: set[str] = set()
    unique_sample_entities: set[str] = set()
    metadata_file = metadata_files[-1] if metadata_files else None
    if metadata_file:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        metadata_count = len(metadata)
        for item in metadata:
            access = str(item.get("access", "not_recorded"))
            access_counts[access] = access_counts.get(access, 0) + 1
            data_type = str(item.get("data_type", "not_recorded"))
            metadata_data_type_counts[data_type] = metadata_data_type_counts.get(data_type, 0) + 1
            for entity in item.get("associated_entities", []) or []:
                entity_type = str(entity.get("entity_type", "not_recorded"))
                metadata_entity_type_counts[entity_type] = (
                    metadata_entity_type_counts.get(entity_type, 0) + 1
                )
                entity_id = entity.get("entity_id")
                if entity_type.casefold() == "case" and entity_id:
                    unique_case_entities.add(str(entity_id))
                elif entity_type.casefold() == "sample" and entity_id:
                    unique_sample_entities.add(str(entity_id))
    manifest_rows = 0
    manifest_bytes = 0
    manifest_states: dict[str, int] = {}
    if manifest_files:
        manifest = pd.read_csv(manifest_files[-1], sep="\t", dtype=str)
        manifest_rows = int(len(manifest))
        if "size" in manifest:
            manifest_bytes = int(pd.to_numeric(manifest["size"], errors="coerce").fillna(0).sum())
        if "state" in manifest:
            manifest_states = {
                str(key): int(value)
                for key, value in manifest["state"].fillna("<missing>").value_counts().items()
            }
    xml_paths = list((root / cohort).rglob("*.xml"))
    stages = frame["stage_at_baseline"].dropna().value_counts()
    histologies = frame["histological_type_raw"].fillna("<missing>").value_counts()
    return {
        "dataset": cohort,
        "release": metadata_file.name if metadata_file else (
            manifest_files[-1].name if manifest_files else "release_not_identified"
        ),
        "access": {
            "gdc_metadata_access_counts": access_counts,
            "metadata_records": metadata_count,
            "gdc_data_type_file_counts": metadata_data_type_counts,
            "associated_entity_record_counts_by_type": metadata_entity_type_counts,
            "unique_gdc_case_entities": len(unique_case_entities),
            "unique_gdc_sample_entities": len(unique_sample_entities),
            "terms_status": "GDC files marked open where metadata is available; verify current GDC/TCGA terms before reuse or redistribution"
        },
        "patient_summary_records": int(len(frame)),
        "patient_summary_records_before_deduplication": int(
            frame.attrs.get("patient_summary_records_before_deduplication", len(frame))
        ),
        "identical_duplicate_patient_records_dropped": int(
            frame.attrs.get("identical_duplicate_patient_records_dropped", 0)
        ),
        "unique_patients": int(frame["patient_id"].nunique()),
        "histological_type_patient_counts": {str(key): int(value) for key, value in histologies.items()},
        "source_cohort_label_patient_count": {
            str(key): int(value) for key, value in frame["cohort_label"].value_counts().items()
        },
        "event_count": int((complete["overall_survival_event"] == True).sum()),  # noqa: E712
        "right_censored_count": int((complete["overall_survival_event"] == False).sum()),  # noqa: E712
        "endpoint_excluded_count": int(len(frame) - len(complete)),
        "endpoint_exclusion_reasons": {
            str(key): int(value)
            for key, value in frame.loc[frame["endpoint_issue"].notna(), "endpoint_issue"].value_counts().items()
        },
        "endpoint": {
            "definition": "Overall survival; death from any cause",
            "time_origin": "Initial pathologic diagnosis",
            "time_unit": "days",
            "event": "patient/vital_status == Dead with valid patient/days_to_death",
            "censoring": "Alive censored at days_to_last_followup; days_to_last_known_alive used only when last follow-up is absent",
            "followup_days": _followup_summary(frame),
        },
        "baseline_feature_missingness_fraction": _missingness(
            frame,
            [
                "age_at_baseline_years",
                "sex_at_baseline",
                "stage_at_baseline",
                "kps_direct",
                "smoking_history_raw",
            ],
        ),
        "source_endpoint_field_missingness_fraction": _missingness(
            frame,
            [
                "vital_status",
                "days_to_death",
                "days_to_last_followup",
                "days_to_last_known_alive",
            ],
        ),
        "source_field_names_inspected": [
            "bcr_patient_barcode",
            "vital_status",
            "days_to_death",
            "days_to_last_followup",
            "days_to_last_known_alive",
            "age_at_initial_pathologic_diagnosis",
            "gender",
            "stage_event/pathologic_stage",
            "karnofsky_performance_score",
            "tobacco_smoking_history",
            "histological_type",
        ],
        "baseline_stage_counts": {str(key): int(value) for key, value in stages.items()},
        "baseline_sex_counts": _category_counts(frame, "sex_at_baseline"),
        "baseline_age_years": _numeric_summary(frame["age_at_baseline_years"]),
        "direct_kps_counts": _category_counts(frame, "kps_direct"),
        "smoking_source_code_counts_uninterpreted": _category_counts(frame, "smoking_history_raw"),
        "raw_files": {
            "clinical_xml_files": len(xml_paths),
            "xml_files_not_used_as_patient_summaries": int(
                len(xml_paths) - frame.attrs.get("patient_summary_records_before_deduplication", len(frame))
            ),
            "clinical_xml_total_bytes": sum(path.stat().st_size for path in xml_paths),
            "manifest_files": [path.name for path in manifest_files],
            "manifest_file_count": manifest_rows,
            "manifest_total_bytes": manifest_bytes,
            "manifest_state_counts": manifest_states,
        },
        "training_assessment": (
            "Eligible for separately reported TCGA-only OS research development after operator confirms applicable data-use terms."
        ),
    }


def build_baseline_report(datasets_root: str | Path) -> dict[str, Any]:
    root = Path(datasets_root)
    report: dict[str, Any] = {
        "report_schema_version": "1.0.0",
        "pipeline_version": "0.1.0",
        "patient_level_only": True,
        "no_patient_identifiers_or_records_in_report": True,
        "datasets_root": root.name,
        "datasets": {},
    }
    msk_root = root / "msk_chord_2024"
    if msk_root.is_dir():
        patients, samples = load_msk_tables(msk_root)
        labels = msk_lung_patient_labels(samples)
        source_license = (msk_root / "LICENSE").read_text(encoding="utf-8", errors="replace")
        explicit_nd = "NoDerivatives" in source_license or "NoDerivatives" in source_license.replace("-", "")
        patient_endpoint = patients[["PATIENT_ID", "OS_MONTHS", "OS_STATUS"]].copy()
        labels_with_outcomes = labels.merge(
            patient_endpoint,
            left_on="patient_id",
            right_on="PATIENT_ID",
            how="left",
            validate="one_to_one",
        )
        lung_rows = samples.loc[
            samples["CANCER_TYPE"].fillna("").str.casefold().str.contains("lung", regex=False)
        ]
        class_counts = labels.loc[labels["label_status"] == "eligible", "histology"].value_counts()
        source_file_inventory = []
        for path in sorted(msk_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(msk_root).as_posix()
            item: dict[str, Any] = {"path": relative, "bytes": int(path.stat().st_size)}
            if path.parent == msk_root and (
                path.name.startswith("data_clinical_") or path.name.startswith("data_timeline_")
            ):
                header = pd.read_csv(
                    path,
                    sep="\t",
                    comment="#",
                    nrows=0,
                    low_memory=False,
                )
                item["columns"] = [str(column) for column in header.columns]
            source_file_inventory.append(item)
        performance_timeline = pd.read_csv(
            msk_root / "data_timeline_performance_status.txt",
            sep="\t",
            comment="#",
            usecols=["PATIENT_ID", "ECOG"],
            dtype=str,
        )
        report["datasets"]["MSK-CHORD"] = {
            "release": "msk_chord_2024 local cBioPortal-style release; exact snapshot date not present in local Readme",
            "source_files": {
                "patient_table": "data_clinical_patient.txt",
                "sample_table": "data_clinical_sample.txt",
                "timeline_performance_status": "data_timeline_performance_status.txt",
                "license": "LICENSE",
            },
            "file_inventory": source_file_inventory,
            "patient_rows": int(len(patients)),
            "unique_patients": int(patients["PATIENT_ID"].nunique()),
            "sample_rows": int(len(samples)),
            "unique_samples": int(samples["SAMPLE_ID"].nunique()),
            "patients_with_multiple_samples": int((samples.groupby("PATIENT_ID").size() > 1).sum()),
            "lung_sample_rows": int(len(lung_rows)),
            "lung_patients": int(labels["patient_id"].nunique()),
            "lung_histology_patient_counts": {str(key): int(value) for key, value in class_counts.items()},
            "lung_label_status_counts": {
                str(key): int(value) for key, value in labels["label_status"].value_counts().items()
            },
            "patient_field_missingness_fraction": _missingness(
                patients,
                [
                    "GENDER",
                    "RACE",
                    "ETHNICITY",
                    "CURRENT_AGE_DEID",
                    "STAGE_HIGHEST_RECORDED",
                    "SMOKING_PREDICTIONS_3_CLASSES",
                    "OS_MONTHS",
                    "OS_STATUS",
                ],
            ),
            "patient_demographic_distributions": {
                "gender_source_values": _category_counts(patients, "GENDER"),
                "race_source_values": _category_counts(patients, "RACE"),
                "ethnicity_source_values": _category_counts(patients, "ETHNICITY"),
                "current_age_deid_not_baseline": _numeric_summary(patients["CURRENT_AGE_DEID"]),
            },
            "stage_highest_recorded_source_values_not_baseline": _category_counts(
                patients, "STAGE_HIGHEST_RECORDED"
            ),
            "smoking_predictions_source_values_not_observed_history": _category_counts(
                patients, "SMOKING_PREDICTIONS_3_CLASSES"
            ),
            "lung_os_status_counts_source_encoded": {
                str(key): int(value)
                for key, value in labels_with_outcomes["OS_STATUS"].fillna("<missing>").value_counts().items()
            },
            "lung_os_months_source_field_not_harmonized": _numeric_summary(
                labels_with_outcomes["OS_MONTHS"]
            ),
            "survival_target_status": (
                "OS_MONTHS/OS_STATUS columns exist for all patient rows, but origin/status coding has not been established "
                "against an authoritative release data dictionary; not harmonized to TCGA OS."
            ),
            "performance_status": {
                "timeline_rows": int(len(performance_timeline)),
                "unique_patients": int(performance_timeline["PATIENT_ID"].nunique()),
                "ecog_source_value_counts": _category_counts(performance_timeline, "ECOG"),
                "semantics": "Longitudinal ECOG timeline; no diagnosis-baseline selection policy validated."
            },
            "license": {
                "identifier": "CC BY-NC-ND 4.0",
                "local_license_mentions_no_derivatives": explicit_nd,
                "training_status": "blocked pending legal/data-governance review of adaptation and model-training permission",
            },
            "feature_compatibility": {
                "stage": "STAGE_HIGHEST_RECORDED excluded; timing relative to prediction baseline is unknown.",
                "smoking": "SMOKING_PREDICTIONS_3_CLASSES excluded; derived prediction is not an observed, harmonized history.",
                "age": "CURRENT_AGE_DEID excluded; relation to diagnosis baseline is not established.",
                "kps": "ECOG timeline is not baseline-aligned and is not an exact KPS measurement.",
            },
        }
    else:
        report["datasets"]["MSK-CHORD"] = {"status": "not_found"}

    for cohort in ("TCGA-LUAD", "TCGA-LUSC"):
        if (root / cohort).is_dir():
            report["datasets"][cohort] = _tcga_summary(root, cohort)
        else:
            report["datasets"][cohort] = {"status": "not_found"}
    report["cross_cohort_assessment"] = {
        "classifier": "TCGA cohort membership equals the cohort histology label and is therefore a source shortcut; TCGA alone is unsuitable for classifier fitting. MSK labels are richer, but training is blocked by license review and uncertain baseline feature semantics.",
        "survival": "TCGA-only OS endpoint/time origin are sufficiently explicit for research development after data-use confirmation. MSK OS semantics and prediction baseline are not established; do not pool.",
        "imaging": "No patient-linked LungInsight Imaging features occur in these training tables; imaging summaries cannot be learned as predictors from the inspected cohorts.",
    }
    return report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Create a privacy-conscious, aggregate LungInsight Clinical data audit.")
    parser.add_argument("--datasets-root", type=Path, default=repo_root / "datasets")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "datasets" / ".clinical-audit" / "baseline-report.json",
    )
    args = parser.parse_args()
    report = build_baseline_report(args.datasets_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Clinical aggregate audit written: {args.output}")
    for dataset, details in report["datasets"].items():
        print(f"{dataset}: {details.get('unique_patients', 0)} unique patient summaries")


if __name__ == "__main__":
    main()