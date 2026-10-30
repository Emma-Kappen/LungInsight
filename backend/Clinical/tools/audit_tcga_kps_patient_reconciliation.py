from __future__ import annotations

import csv
import json
import re
import sys
import traceback
import xml.etree.ElementTree as ET

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# CONFIGURATION — FLAT LUNGINSIGHT PROJECT STRUCTURE
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight")
DATASETS = PROJECT_ROOT / "datasets"
ARTIFACTS = PROJECT_ROOT / "backend" / "Clinical" / "artifacts"

SOURCE_CSV = ARTIFACTS / "tcga_kps_xml_structure_20261029T165441Z.csv"

COHORTS = {
    "LUAD": DATASETS / "TCGA-LUAD",
    "LUSC": DATASETS / "TCGA-LUSC",
}

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

BARCODE_RE = re.compile(
    r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}",
    re.IGNORECASE
)

KPS_TAGS = {
    "karnofsky_performance_score",
    "karnofsky_score",
}

ECOG_TAGS = {
    "ecog_performance_status",
    "ecog_performance_score",
    "ecog_score",
    "eastern_cooperative_oncology_group",
    "eastern_cooperative_oncology_group_performance_status",
}

TIMING_TERMS = (
    "time",
    "timing",
    "assessment",
    "therapy",
    "treatment",
    "follow_up",
)

OUTCOME_TERMS = (
    "vital_status",
    "days_to_death",
    "days_to_last_followup",
    "primary_therapy_outcome_success",
    "new_tumor_event_after_initial_treatment",
    "person_neoplasm_cancer_status",
)

TREATMENT_TERMS = (
    "radiation_therapy",
    "additional_radiation_therapy",
    "postoperative_rx_tx",
    "additional_pharmaceutical_therapy",
    "additional_surgery_metastatic_procedure",
)

FUNCTIONAL_TERMS = (
    "fev1",
    "fvc",
    "pulmonary_function",
    "egfr",
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def local_name(tag):
    """Remove XML namespace and normalize a tag name."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].strip().lower()


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def normalized(value):
    return re.sub(r"[^a-z0-9]+", "_", clean(value).lower()).strip("_")


def extract_barcode(text):
    match = BARCODE_RE.search(clean(text).upper())
    return match.group(0).upper() if match else ""


def safe_float(value):
    try:
        text = clean(value)
        if not text:
            return None
        return float(text)
    except (ValueError, TypeError):
        return None


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)

    if fieldnames is None:
        if rows:
            fieldnames = list(dict.fromkeys(
                key for row in rows for key in row.keys()
            ))
        else:
            fieldnames = []

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def read_csv_rows(path):
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        return list(csv.DictReader(handle))


def find_clinical_table(cohort_dir, cohort):
    """
    Locate the clinical patient table without assuming a UUID directory.
    """
    patterns = [
        "*clinical_patient*luad*.txt",
        "*clinical_patient*LUAD*.txt",
    ] if cohort == "LUAD" else [
        "*clinical_patient*lusc*.txt",
        "*clinical_patient*LUSC*.txt",
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(cohort_dir.rglob(pattern))

    candidates = sorted(set(candidates))

    if not candidates:
        candidates = sorted(
            p for p in cohort_dir.rglob("nationwidechildrens.org_clinical_patient_*.txt")
            if p.is_file()
        )

    return candidates[0] if candidates else None


def read_clinical_table(path):
    """
    TCGA clinical tables may contain CDE_ID metadata rows.
    Remove those rows and retain patient-level records.
    """
    if path is None or not path.exists():
        return [], []

    with path.open("r", newline="", encoding="latin1", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        rows = []

        for row in reader:
            if not row:
                continue

            values = [clean(v) for v in row.values()]
            if any(v.upper().startswith("CDE_ID") for v in values):
                continue

            rows.append({
                clean(k): clean(v)
                for k, v in row.items()
                if k is not None
            })

    return fieldnames, rows


def find_barcode_column(fieldnames):
    candidates = [
        "bcr_patient_barcode",
        "patient_barcode",
        "submitter_id",
        "case_submitter_id",
        "case_id",
    ]

    lookup = {normalized(name): name for name in fieldnames}

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    for name in fieldnames:
        n = normalized(name)
        if "barcode" in n and "patient" in n:
            return name

    return None


# ============================================================
# XML EXTRACTION
# ============================================================

def get_element_text(element):
    """
    Capture direct text and nested text without duplicating whitespace.
    """
    return " ".join(
        part.strip()
        for part in element.itertext()
        if part and part.strip()
    )


def ancestor_context(element, parent_map):
    names = []
    current = element

    while current is not None:
        names.append(local_name(current.tag))
        current = parent_map.get(current)

    return names


def classify_xml_context(context):
    """
    Identify whether a KPS value appears under a follow-up structure.
    Root-level patient records are treated as baseline candidates,
    not as confirmed baseline observations.
    """
    followup_terms = (
        "follow_up",
        "follow_ups",
        "followup",
        "followups",
    )

    if any(
        any(term in part for term in followup_terms)
        for part in context
    ):
        return "follow_up"

    if "patient" in context:
        return "patient_root_candidate"

    return "other_or_unclassified"


def nearest_timing_values(element, parent_map):
    """
    Extract nearby timing/assessment values from the KPS element,
    its ancestors, and their immediate children.
    """
    found = []
    current = element
    depth = 0

    while current is not None and depth < 5:
        for child in list(current):
            tag = local_name(child.tag)

            if any(term in tag for term in TIMING_TERMS):
                value = get_element_text(child)
                if value:
                    found.append({
                        "tag": tag,
                        "value": value,
                    })

        current = parent_map.get(current)
        depth += 1

    # Preserve order while removing duplicates.
    unique = []
    seen = set()

    for item in found:
        key = (item["tag"], item["value"])
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def extract_relevant_fields(root):
    """
    Extract selected outcome, treatment, and functional fields.
    Values are retained with their XML tag names.
    """
    extracted = defaultdict(list)

    for element in root.iter():
        tag = local_name(element.tag)

        relevant = (
            any(term in tag for term in OUTCOME_TERMS)
            or any(term in tag for term in TREATMENT_TERMS)
            or any(term in tag for term in FUNCTIONAL_TERMS)
            or tag in ECOG_TAGS
        )

        if relevant:
            value = get_element_text(element)
            if value:
                extracted[tag].append(value)

    return {
        key: " | ".join(dict.fromkeys(values))
        for key, values in extracted.items()
    }


def parse_xml_file(xml_path, cohort):
    """
    Return KPS/ECOG observations and patient-level contextual fields
    from one XML document.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as exc:
        return {
            "error": str(exc),
            "observations": [],
            "patient_barcode": extract_barcode(str(xml_path)),
        }

    parent_map = {
        child: parent
        for parent in root.iter()
        for child in parent
    }

    patient_barcode = extract_barcode(str(xml_path))

    # Search the XML content for a patient barcode if the filename
    # or directory does not contain one.
    if not patient_barcode:
        for element in root.iter():
            tag = local_name(element.tag)
            value = get_element_text(element)

            if "barcode" in tag or "patient" in tag:
                patient_barcode = extract_barcode(value)
                if patient_barcode:
                    break

    observations = []

    for element in root.iter():
        tag = local_name(element.tag)

        if tag not in KPS_TAGS and tag not in ECOG_TAGS:
            continue

        raw_value = get_element_text(element)
        if not raw_value:
            continue

        context = ancestor_context(element, parent_map)
        context_class = classify_xml_context(context)
        timing = nearest_timing_values(element, parent_map)

        numeric_value = safe_float(raw_value)

        if tag in KPS_TAGS:
            measure = "KPS"
            plausible_range = (
                numeric_value is not None
                and 0 <= numeric_value <= 100
            )
            zero_flag = numeric_value == 0
        else:
            measure = "ECOG"
            plausible_range = (
                numeric_value is not None
                and 0 <= numeric_value <= 5
            )
            zero_flag = False

        if numeric_value is None:
            validation_status = "non_numeric"
        elif not plausible_range:
            validation_status = "outside_expected_range"
        elif zero_flag:
            validation_status = "zero_requires_source_validation"
        else:
            validation_status = "numeric_in_expected_range"

        observations.append({
            "cohort": cohort,
            "patient_barcode": patient_barcode,
            "measure": measure,
            "xml_tag": tag,
            "raw_value": raw_value,
            "numeric_value": numeric_value,
            "zero_flag": zero_flag,
            "validation_status": validation_status,
            "xml_context": " > ".join(context),
            "context_class": context_class,
            "timing_fields": json.dumps(timing, ensure_ascii=False),
            "xml_file": str(xml_path),
        })

    relevant_fields = extract_relevant_fields(root)

    return {
        "error": "",
        "observations": observations,
        "patient_barcode": patient_barcode,
        "relevant_fields": relevant_fields,
    }


# ============================================================
# SOURCE CSV RECONCILIATION
# ============================================================

def inspect_source_csv():
    rows = read_csv_rows(SOURCE_CSV)

    if not rows:
        return {
            "row_count": 0,
            "columns": [],
            "rows": [],
            "barcode_column": None,
            "kps_column": None,
        }

    columns = list(rows[0].keys())
    normalized_columns = {
        normalized(column): column
        for column in columns
    }

    barcode_column = None
    kps_column = None

    for key in (
        "patient_barcode",
        "bcr_patient_barcode",
        "barcode",
        "patient",
    ):
        if key in normalized_columns:
            barcode_column = normalized_columns[key]
            break

    for key in (
        "karnofsky_performance_score",
        "karnofsky_score",
        "kps",
        "kps_value",
        "value",
    ):
        if key in normalized_columns:
            kps_column = normalized_columns[key]
            break

    return {
        "row_count": len(rows),
        "columns": columns,
        "rows": rows,
        "barcode_column": barcode_column,
        "kps_column": kps_column,
    }


# ============================================================
# MAIN AUDIT
# ============================================================

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("LUNGINSIGHT — TCGA KPS / ECOG PATIENT-LEVEL RECONCILIATION")
    print("=" * 72)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Source CSV   : {SOURCE_CSV}")
    print(f"Artifacts    : {ARTIFACTS}")
    print()

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"KPS source CSV not found: {SOURCE_CSV}")

    source_info = inspect_source_csv()

    print("[1/7] Source CSV inspection")
    print(f"Rows    : {source_info['row_count']}")
    print(f"Columns : {source_info['columns']}")
    print(f"Barcode column detected: {source_info['barcode_column']}")
    print(f"KPS column detected    : {source_info['kps_column']}")
    print()

    all_observations = []
    all_xml_errors = []
    cohort_summaries = {}
    patient_fields = defaultdict(lambda: defaultdict(list))
    clinical_records = {}

    print("[2/7] Discovering and parsing all cohort XML files")

    for cohort, cohort_dir in COHORTS.items():
        if not cohort_dir.exists():
            print(f"WARNING: cohort directory missing: {cohort_dir}")
            continue

        xml_files = sorted(
            p for p in cohort_dir.rglob("*.xml")
            if p.is_file()
        )

        print(f"{cohort}: discovered {len(xml_files)} XML files")

        cohort_observations = []
        patient_xml_files = defaultdict(set)

        for index, xml_path in enumerate(xml_files, start=1):
            result = parse_xml_file(xml_path, cohort)

            if result["error"]:
                all_xml_errors.append({
                    "cohort": cohort,
                    "xml_file": str(xml_path),
                    "error": result["error"],
                })
                continue

            barcode = result["patient_barcode"]

            if barcode:
                patient_xml_files[barcode].add(str(xml_path))

                for field, value in result.get("relevant_fields", {}).items():
                    patient_fields[(cohort, barcode)][field].append(value)

            for observation in result["observations"]:
                cohort_observations.append(observation)
                all_observations.append(observation)

            if index % 250 == 0:
                print(f"  Parsed {index}/{len(xml_files)} XML files")

        clinical_path = find_clinical_table(cohort_dir, cohort)
        clinical_columns, clinical_rows = read_clinical_table(clinical_path)

        barcode_column = find_barcode_column(clinical_columns)

        clinical_by_patient = {}

        if barcode_column:
            for row in clinical_rows:
                barcode = extract_barcode(row.get(barcode_column, ""))
                if barcode:
                    clinical_by_patient[barcode] = row

        clinical_records[cohort] = {
            "path": str(clinical_path) if clinical_path else "",
            "columns": clinical_columns,
            "barcode_column": barcode_column,
            "rows": clinical_by_patient,
        }

        kps = [
            o for o in cohort_observations
            if o["measure"] == "KPS"
        ]
        ecog = [
            o for o in cohort_observations
            if o["measure"] == "ECOG"
        ]

        kps_nonempty = [o for o in kps if clean(o["raw_value"])]
        kps_patients = {
            o["patient_barcode"]
            for o in kps
            if o["patient_barcode"]
        }
        kps_nonempty_patients = {
            o["patient_barcode"]
            for o in kps_nonempty
            if o["patient_barcode"]
        }

        cohort_summaries[cohort] = {
            "xml_files_discovered": len(xml_files),
            "xml_files_with_parse_errors": sum(
                1 for error in all_xml_errors
                if error["cohort"] == cohort
            ),
            "patients_with_xml_barcodes": len(patient_xml_files),
            "clinical_table": str(clinical_path) if clinical_path else None,
            "clinical_patient_rows": len(clinical_by_patient),
            "clinical_barcode_column": barcode_column,
            "kps_observations": len(kps),
            "kps_nonempty_observations": len(kps_nonempty),
            "kps_patients": len(kps_patients),
            "kps_nonempty_patients": len(kps_nonempty_patients),
            "ecog_observations": len(ecog),
            "ecog_patients": len({
                o["patient_barcode"]
                for o in ecog
                if o["patient_barcode"]
            }),
            "kps_zero_observations": sum(
                1 for o in kps if o["zero_flag"]
            ),
        }

        print(
            f"  {cohort}: KPS observations={len(kps)}, "
            f"nonempty={len(kps_nonempty)}, "
            f"patients={len(kps_nonempty_patients)}, "
            f"ECOG observations={len(ecog)}"
        )

    print()
    print("[3/7] Patient-level clinical reconciliation")

    patient_keys = set()

    for observation in all_observations:
        if observation["patient_barcode"]:
            patient_keys.add((
                observation["cohort"],
                observation["patient_barcode"]
            ))

    for cohort, data in clinical_records.items():
        for barcode in data["rows"]:
            patient_keys.add((cohort, barcode))

    patient_rows = []

    for cohort, barcode in sorted(patient_keys):
        observations = [
            o for o in all_observations
            if o["cohort"] == cohort
            and o["patient_barcode"] == barcode
        ]

        kps = [o for o in observations if o["measure"] == "KPS"]
        ecog = [o for o in observations if o["measure"] == "ECOG"]

        root_kps = [
            o for o in kps
            if o["context_class"] == "patient_root_candidate"
        ]

        followup_kps = [
            o for o in kps
            if o["context_class"] == "follow_up"
        ]

        other_kps = [
            o for o in kps
            if o["context_class"] == "other_or_unclassified"
        ]

        clinical = clinical_records.get(cohort, {}).get(
            "rows", {}
        ).get(barcode, {})

        kps_values = [
            o["numeric_value"]
            for o in kps
            if o["numeric_value"] is not None
        ]

        zero_observations = [
            o for o in kps
            if o["zero_flag"]
        ]

        patient_rows.append({
            "cohort": cohort,
            "patient_barcode": barcode,
            "present_in_clinical_table": bool(clinical),
            "kps_observation_count": len(kps),
            "kps_numeric_count": len(kps_values),
            "kps_distinct_values": "|".join(
                str(v) for v in sorted(set(kps_values))
            ),
            "kps_root_candidate_count": len(root_kps),
            "kps_follow_up_count": len(followup_kps),
            "kps_other_context_count": len(other_kps),
            "kps_zero_count": len(zero_observations),
            "kps_zero_files": "|".join(
                sorted(set(o["xml_file"] for o in zero_observations))
            ),
            "ecog_observation_count": len(ecog),
            "ecog_values": "|".join(
                sorted(set(o["raw_value"] for o in ecog))
            ),
            "xml_contexts": "|".join(
                sorted(set(o["context_class"] for o in kps))
            ),
            "clinical_vital_status": clinical.get("vital_status", ""),
            "clinical_days_to_death": clinical.get("days_to_death", ""),
            "clinical_days_to_last_followup": clinical.get(
                "days_to_last_followup", ""
            ),
            "clinical_primary_therapy_outcome_success": clinical.get(
                "primary_therapy_outcome_success", ""
            ),
            "clinical_person_neoplasm_cancer_status": clinical.get(
                "person_neoplasm_cancer_status", ""
            ),
        })

    print(f"Patient-level rows: {len(patient_rows)}")

    print()
    print("[4/7] KPS zero-value investigation")

    zero_rows = []

    for observation in all_observations:
        if observation["measure"] != "KPS" or not observation["zero_flag"]:
            continue

        zero_rows.append({
            **observation,
            "zero_interpretation": (
                "Requires original-source validation; "
                "not automatically treated as missing or valid"
            ),
        })

    zero_by_cohort = Counter(row["cohort"] for row in zero_rows)
    zero_by_context = Counter(row["context_class"] for row in zero_rows)
    zero_by_timing = Counter()

    for row in zero_rows:
        try:
            timing_items = json.loads(row["timing_fields"])
        except Exception:
            timing_items = []

        for item in timing_items:
            zero_by_timing[
                f"{item.get('tag', '')}={item.get('value', '')}"
            ] += 1

    print(f"Total KPS zero observations: {len(zero_rows)}")
    print(f"By cohort: {dict(zero_by_cohort)}")
    print(f"By XML context: {dict(zero_by_context)}")
    print(f"By timing: {dict(zero_by_timing)}")

    print()
    print("[5/7] Baseline-candidate and follow-up assessment")

    baseline_rows = []
    followup_rows = []

    for observation in all_observations:
        if observation["measure"] != "KPS":
            continue

        row = dict(observation)

        if observation["context_class"] == "patient_root_candidate":
            row["baseline_assessment"] = (
                "ROOT_LEVEL_CANDIDATE_NOT_CONFIRMED_BASELINE"
            )
            baseline_rows.append(row)

        elif observation["context_class"] == "follow_up":
            row["baseline_assessment"] = "LONGITUDINAL_FOLLOW_UP_EXCLUDE"
            followup_rows.append(row)

        else:
            row["baseline_assessment"] = "UNCLASSIFIED_REQUIRES_REVIEW"
            baseline_rows.append(row)

    print(f"Root-level candidate observations: {len(baseline_rows)}")
    print(f"Follow-up observations: {len(followup_rows)}")

    print()
    print("[6/7] Clinical field availability and source CSV reconciliation")

    clinical_field_summary = {}

    for cohort, data in clinical_records.items():
        rows = list(data["rows"].values())
        columns = data["columns"]

        fields_to_check = [
            "age_at_initial_pathologic_diagnosis",
            "gender",
            "race",
            "history_of_smoking",
            "pack_years_smoked",
            "year_of_tobacco_smoking_onset",
            "year_of_tobacco_smoking_quit",
            "vital_status",
            "days_to_death",
            "days_to_last_followup",
            "primary_therapy_outcome_success",
            "person_neoplasm_cancer_status",
        ]

        field_counts = {}

        for field in fields_to_check:
            if field not in columns:
                field_counts[field] = {
                    "column_present": False,
                    "nonempty": 0,
                    "missing": len(rows),
                }
                continue

            nonempty = sum(
                1 for row in rows
                if clean(row.get(field, ""))
            )

            field_counts[field] = {
                "column_present": True,
                "nonempty": nonempty,
                "missing": len(rows) - nonempty,
            }

        clinical_field_summary[cohort] = {
            "clinical_patient_rows": len(rows),
            "fields": field_counts,
        }

    source_barcode_counts = Counter()
    source_kps_counts = Counter()
    source_zero_rows = []

    barcode_column = source_info["barcode_column"]
    kps_column = source_info["kps_column"]

    for row in source_info["rows"]:
        barcode = extract_barcode(
            row.get(barcode_column, "")
        ) if barcode_column else ""

        value = clean(row.get(kps_column, "")) if kps_column else ""

        if barcode:
            source_barcode_counts[barcode] += 1

        if value:
            source_kps_counts[value] += 1

            if safe_float(value) == 0:
                source_zero_rows.append(row)

    source_reconciliation = {
        "source_csv": str(SOURCE_CSV),
        "row_count": source_info["row_count"],
        "columns": source_info["columns"],
        "barcode_column_detected": barcode_column,
        "kps_column_detected": kps_column,
        "unique_barcodes_detected": len(source_barcode_counts),
        "nonempty_kps_value_distribution": dict(source_kps_counts),
        "zero_value_rows": len(source_zero_rows),
    }

    print("Clinical field availability:")
    print(json.dumps(clinical_field_summary, indent=2))

    print()
    print("[7/7] Writing audit outputs")

    prefix = f"tcga_kps_patient_reconciliation_{TIMESTAMP}"

    observations_path = ARTIFACTS / f"{prefix}_observations.csv"
    patients_path = ARTIFACTS / f"{prefix}_patients.csv"
    zeros_path = ARTIFACTS / f"{prefix}_zero_investigation.csv"
    baseline_path = ARTIFACTS / f"{prefix}_baseline_candidates.csv"
    followup_path = ARTIFACTS / f"{prefix}_followup_observations.csv"
    errors_path = ARTIFACTS / f"{prefix}_xml_errors.csv"
    report_json_path = ARTIFACTS / f"{prefix}.json"
    report_txt_path = ARTIFACTS / f"{prefix}.txt"

    write_csv(observations_path, all_observations)
    write_csv(patients_path, patient_rows)
    write_csv(zeros_path, zero_rows)
    write_csv(baseline_path, baseline_rows)
    write_csv(followup_path, followup_rows)
    write_csv(errors_path, all_xml_errors)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "source_csv_reconciliation": source_reconciliation,
        "cohorts": cohort_summaries,
        "patient_level": {
            "patient_rows": len(patient_rows),
            "unique_patient_barcodes": len({
                row["patient_barcode"] for row in patient_rows
            }),
            "patients_present_in_clinical_table": sum(
                1 for row in patient_rows
                if row["present_in_clinical_table"]
            ),
        },
        "kps": {
            "total_observations": sum(
                1 for o in all_observations if o["measure"] == "KPS"
            ),
            "root_level_candidate_observations": len(baseline_rows),
            "follow_up_observations": len(followup_rows),
            "zero_observations": len(zero_rows),
            "zero_by_cohort": dict(zero_by_cohort),
            "zero_by_context": dict(zero_by_context),
            "zero_by_timing": dict(zero_by_timing),
        },
        "ecog": {
            "total_observations": sum(
                1 for o in all_observations if o["measure"] == "ECOG"
            ),
        },
        "clinical_field_availability": clinical_field_summary,
        "xml_parse_errors": len(all_xml_errors),
        "interpretation_rules": {
            "patient_root": (
                "Baseline candidate only; not proof of pre-treatment timing."
            ),
            "follow_up": (
                "Longitudinal follow-up; exclude from baseline feature "
                "unless a separately justified timepoint is established."
            ),
            "post_adjuvant": (
                "Post-treatment measurements must not be used as baseline "
                "predictors."
            ),
            "zero_kps": (
                "Zero values are flagged for source validation. They are "
                "not automatically converted to missing or accepted as "
                "valid baseline performance scores."
            ),
            "missing_kps": (
                "Missing KPS must not be imputed as zero."
            ),
            "training_decision": (
                "This audit does not modify or approve the training dataset. "
                "KPS should remain excluded from model training until "
                "baseline provenance and zero-value validity are resolved."
            ),
        },
        "output_files": {
            "observations": str(observations_path),
            "patients": str(patients_path),
            "zero_investigation": str(zeros_path),
            "baseline_candidates": str(baseline_path),
            "followup_observations": str(followup_path),
            "xml_errors": str(errors_path),
            "json_report": str(report_json_path),
            "text_report": str(report_txt_path),
        },
    }

    with report_json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    lines = []
    lines.append("LUNGINSIGHT — TCGA KPS / ECOG PATIENT-LEVEL RECONCILIATION")
    lines.append("=" * 72)
    lines.append(f"Generated UTC: {report['generated_utc']}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append("")

    lines.append("COHORT SUMMARY")
    lines.append("-" * 72)

    for cohort, summary in cohort_summaries.items():
        lines.append(f"\n{cohort}")
        for key, value in summary.items():
            lines.append(f"  {key}: {value}")

    lines.append("\nPATIENT-LEVEL RECONCILIATION")
    lines.append("-" * 72)
    lines.append(f"Patient rows: {len(patient_rows)}")
    lines.append(
        "Patients present in clinical table: "
        + str(sum(1 for row in patient_rows if row["present_in_clinical_table"]))
    )

    lines.append("\nKPS OBSERVATIONS")
    lines.append("-" * 72)
    lines.append(f"Total KPS observations: {report['kps']['total_observations']}")
    lines.append(f"Root-level candidates: {len(baseline_rows)}")
    lines.append(f"Follow-up observations: {len(followup_rows)}")
    lines.append(f"Zero observations: {len(zero_rows)}")
    lines.append(f"Zero by cohort: {dict(zero_by_cohort)}")
    lines.append(f"Zero by context: {dict(zero_by_context)}")
    lines.append(f"Zero by timing: {dict(zero_by_timing)}")

    lines.append("\nECOG")
    lines.append("-" * 72)
    lines.append(f"Total ECOG observations: {report['ecog']['total_observations']}")

    lines.append("\nZERO-VALUE INTERPRETATION")
    lines.append("-" * 72)
    lines.append(
        "Every KPS value of zero is retained in the zero-investigation CSV "
        "with its XML file, barcode, XML context, and nearby timing fields."
    )
    lines.append(
        "Zero is not automatically interpreted as missing or as a valid "
        "performance score."
    )

    lines.append("\nBASELINE INTERPRETATION")
    lines.append("-" * 72)
    lines.append(
        "Patient-root KPS is a baseline candidate, not confirmed baseline."
    )
    lines.append(
        "Follow-up KPS is longitudinal and should not be used as a baseline "
        "predictor."
    )
    lines.append(
        "A timing field must be interpreted in its clinical context before "
        "baseline eligibility is assigned."
    )

    lines.append("\nTRAINING RECOMMENDATION")
    lines.append("-" * 72)
    lines.append(
        "Keep KPS excluded from model training until patient-level baseline "
        "provenance, treatment timing, and zero-value validity are resolved."
    )
    lines.append(
        "Do not impute missing KPS as zero. This audit does not modify "
        "training inputs or models."
    )

    lines.append("\nOUTPUT FILES")
    lines.append("-" * 72)

    for key, value in report["output_files"].items():
        lines.append(f"{key}: {value}")

    with report_txt_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print()
    print("=" * 72)
    print("AUDIT COMPLETED")
    print("=" * 72)
    print(f"Patients reconciled : {len(patient_rows)}")
    print(f"KPS observations    : {report['kps']['total_observations']}")
    print(f"KPS zero values     : {len(zero_rows)}")
    print(f"ECOG observations   : {report['ecog']['total_observations']}")
    print(f"XML parse errors    : {len(all_xml_errors)}")
    print()
    print("REPORT:")
    print(report_txt_path)
    print()
    print("PATIENT RECONCILIATION:")
    print(patients_path)
    print()
    print("ZERO-VALUE INVESTIGATION:")
    print(zeros_path)
    print()
    print("JSON SUMMARY:")
    print(report_json_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nAUDIT FAILED")
        traceback.print_exc()
        sys.exit(1)
