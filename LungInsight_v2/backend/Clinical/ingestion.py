from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

from .harmonization import normalize_sex

TCGA_COHORT_LABELS = {"TCGA-LUAD": "LUAD", "TCGA-LUSC": "LUSC"}
TCGA_BASELINE_FEATURES = (
    "age_at_baseline_years",
    "sex_at_baseline",
    "pathologic_stage_ground_truth",
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_text(element: ET.Element, name: str) -> str | None:
    for child in list(element):
        if _local_name(child.tag) == name:
            value = (child.text or "").strip()
            return value or None
    return None


def _direct_raw(element: ET.Element, name: str) -> str | None:
    for child in list(element):
        if _local_name(child.tag) == name:
            return child.text
    return None


def _parse_nonnegative_days(value: str | None) -> float | None:
    if value is None or value.casefold() in {"[not available]", "na", "n/a", "unknown"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _pathologic_stage(patient: ET.Element) -> str | None:
    for child in list(patient):
        if _local_name(child.tag) != "stage_event":
            continue
        for stage in list(child):
            if _local_name(stage.tag) == "pathologic_stage":
                value = (stage.text or "").strip()
                return value or None
    return None


def _pathologic_stage_raw(patient: ET.Element) -> str | None:
    for child in list(patient):
        if _local_name(child.tag) != "stage_event":
            continue
        for stage in list(child):
            if _local_name(stage.tag) == "pathologic_stage":
                return stage.text
    return None


def parse_tcga_patient_xml(path: Path, cohort: str) -> dict[str, Any] | None:
    """Parse a patient summary only; follow-up-only XML files are not independent patients."""
    if cohort not in TCGA_COHORT_LABELS:
        raise ValueError(f"Unsupported TCGA cohort {cohort!r}.")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Malformed TCGA XML file {path.name}.") from exc
    patient = next((node for node in list(root) if _local_name(node.tag) == "patient"), None)
    if patient is None:
        return None
    vital_status = _direct_text(patient, "vital_status")
    if vital_status is None:
        return None
    patient_id = _direct_text(patient, "bcr_patient_barcode")
    if not patient_id:
        raise ValueError(f"Patient-summary XML {path.name} has no patient barcode.")

    death_days_raw = _direct_text(patient, "days_to_death")
    followup_days_raw = _direct_text(patient, "days_to_last_followup")
    last_alive_raw = _direct_text(patient, "days_to_last_known_alive")
    status = vital_status.casefold()
    endpoint_issue: str | None = None
    if status == "dead":
        time_days = _parse_nonnegative_days(death_days_raw)
        event = True
        if time_days is None:
            endpoint_issue = "dead_without_valid_days_to_death"
        if _parse_nonnegative_days(death_days_raw) is None and death_days_raw is not None:
            endpoint_issue = "invalid_or_negative_days_to_death"
        if time_days is not None:
            secondary_times = (
                _parse_nonnegative_days(followup_days_raw),
                _parse_nonnegative_days(last_alive_raw),
            )
            if any(value is not None and value > time_days for value in secondary_times):
                endpoint_issue = "followup_after_death_conflict"
    elif status == "alive":
        event = False
        time_source = "days_to_last_followup"
        time_days = _parse_nonnegative_days(followup_days_raw)
        if followup_days_raw is not None and time_days is None:
            endpoint_issue = "invalid_or_negative_days_to_last_followup"
        elif followup_days_raw is None:
            time_days = _parse_nonnegative_days(last_alive_raw)
            if time_days is not None:
                time_source = "days_to_last_known_alive"
            else:
                time_source = None
        if time_days is None and endpoint_issue is None:
            endpoint_issue = "alive_without_valid_followup_time"
        death_numeric = None
        if death_days_raw is not None:
            try:
                death_numeric = float(death_days_raw)
            except ValueError:
                pass
        if death_numeric is not None and math.isfinite(death_numeric):
            endpoint_issue = "alive_with_days_to_death_conflict"
    else:
        event = None
        time_days = None
        endpoint_issue = "unknown_vital_status"
        time_source = None
    if status == "dead":
        time_source = "days_to_death" if time_days is not None else None

    age_raw = _direct_text(patient, "age_at_initial_pathologic_diagnosis")
    try:
        age = float(age_raw) if age_raw is not None else None
    except ValueError:
        age = None
    if age is not None and (not math.isfinite(age) or not 18 <= age <= 120):
        age = None
    raw_kps = _direct_text(patient, "karnofsky_performance_score")
    try:
        kps = int(raw_kps) if raw_kps is not None else None
        if kps is not None and not 0 <= kps <= 100:
            kps = None
    except ValueError:
        kps = None

    recorded_histology = _direct_text(patient, "histological_type")
    histology = classify_tcga_histology(recorded_histology)
    return {
        "bcr_patient_barcode": _direct_raw(patient, "bcr_patient_barcode"),
        "vital_status": _direct_raw(patient, "vital_status"),
        "days_to_death": _direct_raw(patient, "days_to_death"),
        "days_to_last_followup": _direct_raw(patient, "days_to_last_followup"),
        "days_to_last_known_alive": _direct_raw(patient, "days_to_last_known_alive"),
        "age_at_initial_pathologic_diagnosis": _direct_raw(
            patient, "age_at_initial_pathologic_diagnosis"
        ),
        "gender": _direct_raw(patient, "gender"),
        "pathologic_stage": _pathologic_stage_raw(patient),
        "karnofsky_performance_score": _direct_raw(
            patient, "karnofsky_performance_score"
        ),
        "tobacco_smoking_history": _direct_raw(patient, "tobacco_smoking_history"),
        "histological_type": _direct_raw(patient, "histological_type"),
        "patient_id": patient_id,
        "source": cohort,
        "cohort_label": TCGA_COHORT_LABELS[cohort],
        "histological_type_raw": recorded_histology,
        "histology": histology,
        "histology_label_status": "eligible" if histology is not None else "unsupported_or_missing",
        "age_at_baseline_years": age,
        "sex_at_baseline": normalize_sex(_direct_text(patient, "gender")),
        "pathologic_stage_ground_truth": _pathologic_stage(patient),
        "kps_direct": kps,
        "smoking_history_raw": _direct_text(patient, "tobacco_smoking_history"),
        "vital_status_raw": vital_status,
        "overall_survival_time_days": time_days,
        "overall_survival_event": event,
        "time_source": time_source,
        "endpoint_issue": endpoint_issue,
    }


def load_tcga_survival_cohort(root: str | Path, cohort: str) -> pd.DataFrame:
    """Load patient-level TCGA clinical XML without merging patient or sample rows."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"TCGA cohort directory is unavailable: {root_path}")
    records = []
    for path in sorted(root_path.rglob("*.xml")):
        row = parse_tcga_patient_xml(path, cohort)
        if row is not None:
            records.append(row)
    if not records:
        raise ValueError(f"No patient-summary clinical XML records found in {root_path}.")
    frame = pd.DataFrame.from_records(records)
    frame.attrs["patient_summary_records_before_deduplication"] = int(len(frame))
    duplicate_mask = frame.duplicated("patient_id", keep=False)
    dropped_duplicates = 0
    if duplicate_mask.any():
        duplicate_rows = frame.loc[duplicate_mask]
        conflicting = False
        for _, group in duplicate_rows.groupby("patient_id", dropna=False):
            if group.drop(columns="patient_id").nunique(dropna=False).gt(1).any():
                conflicting = True
                break
        if conflicting:
            raise ValueError(f"Conflicting patient-summary records found in {cohort}.")
        dropped_duplicates = int(frame.duplicated("patient_id", keep="first").sum())
        frame = frame.drop_duplicates("patient_id", keep="first").reset_index(drop=True)
    frame.attrs["identical_duplicate_patient_records_dropped"] = dropped_duplicates
    return frame


def load_msk_tables(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read raw MSK-CHORD clinical patient/sample tables as strings; do not mutate source files."""
    root_path = Path(root)
    patient_path = root_path / "data_clinical_patient.txt"
    sample_path = root_path / "data_clinical_sample.txt"
    for path in (patient_path, sample_path):
        if not path.is_file():
            raise FileNotFoundError(f"MSK-CHORD source table is unavailable: {path}")
    patients = pd.read_csv(patient_path, sep="\t", comment="#", dtype=str, low_memory=False)
    samples = pd.read_csv(sample_path, sep="\t", comment="#", dtype=str, low_memory=False)
    for frame, identity in ((patients, "PATIENT_ID"), (samples, "SAMPLE_ID")):
        if identity not in frame or frame[identity].isna().any():
            raise ValueError(f"MSK-CHORD table is missing valid {identity} values.")
        if frame[identity].duplicated().any():
            raise ValueError(f"Duplicate MSK-CHORD {identity} values require source-specific resolution.")
    if "PATIENT_ID" not in samples:
        raise ValueError("MSK-CHORD sample table has no patient link.")
    return patients, samples


def classify_msk_histology(row: pd.Series) -> str | None:
    """Assign only exact LUAD/LUSC diagnoses; broad NSCLC is not a histology label."""
    code = str(row.get("ONCOTREE_CODE", "") or "").strip().upper()
    detailed = str(row.get("CANCER_TYPE_DETAILED", "") or "").strip().casefold()
    code_label = code if code in {"LUAD", "LUSC"} else None
    detail_label = {
        "lung adenocarcinoma": "LUAD",
        "lung squamous cell carcinoma": "LUSC",
    }.get(detailed)
    if code_label and detail_label and code_label != detail_label:
        return None
    return code_label or detail_label


def classify_tcga_histology(recorded_histology: str | None) -> str | None:
    """Map only explicit TCGA recorded histology text to supported classes."""
    if recorded_histology is None:
        return None
    normalized = recorded_histology.strip().casefold()
    if "adenocarcinoma" in normalized or "bronchioloalveolar carcinoma" in normalized:
        return "LUAD"
    if "squamous cell" in normalized:
        return "LUSC"
    return None


def _msk_field_conflict(row: pd.Series) -> bool:
    code = str(row.get("ONCOTREE_CODE", "") or "").strip().upper()
    detailed = str(row.get("CANCER_TYPE_DETAILED", "") or "").strip().casefold()
    detailed_label = {
        "lung adenocarcinoma": "LUAD",
        "lung squamous cell carcinoma": "LUSC",
    }.get(detailed)
    return code in {"LUAD", "LUSC"} and detailed_label is not None and code != detailed_label


def msk_lung_patient_labels(samples: pd.DataFrame) -> pd.DataFrame:
    required = {"PATIENT_ID", "SAMPLE_ID", "CANCER_TYPE", "CANCER_TYPE_DETAILED", "ONCOTREE_CODE"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"MSK-CHORD sample table lacks required columns: {sorted(missing)}.")
    lung = samples.loc[
        samples["CANCER_TYPE"].fillna("").str.casefold().str.contains("lung", regex=False)
    ].copy()
    lung["histology"] = lung.apply(classify_msk_histology, axis=1)
    lung["field_conflict"] = lung.apply(_msk_field_conflict, axis=1)
    records: list[dict[str, Any]] = []
    for patient_id, group in lung.groupby("PATIENT_ID", sort=False, dropna=False):
        assigned = set(group["histology"].dropna())
        if group["field_conflict"].any() or len(assigned) > 1:
            records.append({"patient_id": patient_id, "histology": None, "label_status": "conflicting"})
        elif len(assigned) == 1 and group["histology"].notna().all():
            records.append({"patient_id": patient_id, "histology": assigned.pop(), "label_status": "eligible"})
        else:
            records.append({"patient_id": patient_id, "histology": None, "label_status": "ambiguous_or_other"})
    return pd.DataFrame.from_records(records, columns=["patient_id", "histology", "label_status"])


def load_msk_cancer_type_cohort(root: str | Path) -> pd.DataFrame:
    """Build one explicit-label, patient-level MSK LUAD/LUSC classifier cohort."""
    patients, samples = load_msk_tables(root)
    labels = msk_lung_patient_labels(samples)
    cohort = labels.merge(
        patients[["PATIENT_ID", "GENDER"]],
        left_on="patient_id",
        right_on="PATIENT_ID",
        how="left",
        validate="one_to_one",
    )
    cohort["sex_at_baseline"] = cohort["GENDER"].map(
        lambda value: None if pd.isna(value) else normalize_sex(value)
    )
    cohort["source"] = "MSK-CHORD"
    cohort["label_provenance"] = "sample/ONCOTREE_CODE_and_CANCER_TYPE_DETAILED"
    cohort = cohort[
        [
            "patient_id",
            "source",
            "histology",
            "label_status",
            "label_provenance",
            "sex_at_baseline",
        ]
    ].copy()
    cohort.attrs["source_patient_rows"] = int(len(patients))
    cohort.attrs["source_sample_rows"] = int(len(samples))
    cohort.attrs["patients_with_multiple_samples"] = int(
        (samples.groupby("PATIENT_ID").size() > 1).sum()
    )
    return cohort