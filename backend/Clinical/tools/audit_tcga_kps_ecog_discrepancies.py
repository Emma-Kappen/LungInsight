import csv
import json
import re
import sys
import traceback
import xml.etree.ElementTree as ET

from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight")
ARTIFACTS = ROOT / "backend" / "Clinical" / "artifacts"

COHORTS = {
    "LUAD": {
        "root": ROOT / "datasets" / "TCGA-LUAD",
        "table": ROOT / "datasets" / "TCGA-LUAD" /
            "42bf5eb2-bc49-45be-b18a-290f712b006c" /
            "nationwidechildrens.org_clinical_patient_luad.txt",
    },
    "LUSC": {
        "root": ROOT / "datasets" / "TCGA-LUSC",
        "table": ROOT / "datasets" / "TCGA-LUSC" /
            "7aaee8d4-3bea-4dfc-b392-c7aaa3e9a550" /
            "nationwidechildrens.org_clinical_patient_lusc.txt",
    },
}

BARCODE_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.I)
CDE_RE = re.compile(r"^\s*CDE_ID(?:\s*:|$)", re.I)

MISSING = {
    "", "--", "-", "NA", "N/A", "NULL", "NONE",
    "[UNKNOWN]", "[NOT AVAILABLE]", "[NOT APPLICABLE]",
    "[DISCREPANCY]", "NOT AVAILABLE", "NOT REPORTED",
    "UNKNOWN", "NOT KNOWN", "NAN",
}

KPS_TAGS = {
    "karnofskyscore",
    "karnofskyperformancescore",
    "karnofskyperformancestatus",
    "karnofskyperformancestatusscore",
    "kps",
}

ECOG_TAGS = {
    "ecog",
    "ecogscore",
    "ecogperformancestatus",
    "ecogperformancestatusscale",
    "ecogperformancescore",
    "easterncooperativeoncologygroup",
    "easterncooperativeoncologygroupperformancestatus",
    "performancestatusecog",
}

KPS_TABLE_ALIASES = {
    "karnofskyscore",
    "karnofskyperformancescore",
    "karnofskyperformancestatus",
    "karnofskyperformancestatusscore",
    "kps",
}

ECOG_TABLE_ALIASES = ECOG_TAGS | {
    "ecogperformance",
    "ecogstatus",
}


def norm_tag(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def clean_value(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def is_missing(value):
    value = clean_value(value)
    return value.upper() in MISSING or CDE_RE.match(value) is not None


def comparison_key(value):
    value = clean_value(value)
    if is_missing(value):
        return None

    # Normalize numeric values such as 90, 90.0 and 090 to the same key.
    try:
        number = Decimal(value)
        if number.is_finite():
            return "num:" + format(number.normalize(), "f")
    except (InvalidOperation, ValueError):
        pass

    return "txt:" + value.casefold()


def distinct_usable(values):
    by_key = {}
    for value in values:
        value = clean_value(value)
        key = comparison_key(value)
        if key is not None:
            by_key.setdefault(key, value)
    return [by_key[k] for k in sorted(by_key)]


def barcode_from_path(path):
    match = BARCODE_RE.search(str(path).upper())
    return match.group(0).upper() if match else None


def read_clinical_table(path):
    if not path.is_file():
        raise FileNotFoundError(f"Clinical table not found: {path}")

    with path.open("r", encoding="latin1", newline="") as handle:
        lines = handle.readlines()

    # Locate the actual header rather than assuming that the first line
    # is always the header.
    header_index = None
    for i, line in enumerate(lines[:30]):
        fields = next(csv.reader([line], delimiter="\t"))
        normalized = {norm_tag(x) for x in fields}
        if "bcrpatientbarcode" in normalized:
            header_index = i
            break

    if header_index is None:
        raise ValueError(f"Could not locate bcr_patient_barcode header in {path}")

    reader = csv.DictReader(lines[header_index:], delimiter="\t")
    rows = []
    metadata_removed = 0
    invalid_barcode_removed = 0
    duplicates_ignored = 0
    seen = set()

    for raw in reader:
        if not raw:
            continue

        if any(CDE_RE.match(clean_value(v)) for v in raw.values() if v is not None):
            metadata_removed += 1
            continue

        normalized_row = {
            norm_tag(k): clean_value(v)
            for k, v in raw.items()
            if k is not None
        }

        barcode = normalized_row.get("bcrpatientbarcode", "").upper()
        if not BARCODE_RE.fullmatch(barcode):
            invalid_barcode_removed += 1
            continue

        if barcode in seen:
            duplicates_ignored += 1
            continue

        seen.add(barcode)
        rows.append({
            "barcode": barcode,
            "values": normalized_row,
        })

    return rows, {
        "metadata_rows_removed": metadata_removed,
        "invalid_barcode_rows_removed": invalid_barcode_removed,
        "duplicate_rows_ignored": duplicates_ignored,
        "header_columns": list(reader.fieldnames or []),
    }


def table_field(row, aliases):
    values = row["values"]
    for alias in aliases:
        key = norm_tag(alias)
        if key in values:
            return values[key], True
    return "", False


def extract_xml_fields(path):
    """Return all matching tag values and source evidence from one XML file."""
    results = {"kps": [], "ecog": []}

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as exc:
        return results, f"{type(exc).__name__}: {exc}"

    for element in root.iter():
        # Remove an optional XML namespace prefix.
        tag = element.tag.rsplit("}", 1)[-1]
        tag = tag.rsplit(":", 1)[-1]
        normalized_tag = norm_tag(tag)

        if normalized_tag not in KPS_TAGS and normalized_tag not in ECOG_TAGS:
            continue

        # Do not duplicate a parent's text from nested child elements.
        if list(element):
            continue

        value = clean_value(element.text)
        if is_missing(value):
            continue

        field = "kps" if normalized_tag in KPS_TAGS else "ecog"
        results[field].append({
            "value": value,
            "tag": tag,
            "file": path.name,
        })

    return results, None


def status_for(table_value, table_column_present, xml_values):
    values = distinct_usable([x["value"] for x in xml_values])

    if len(values) > 1:
        return "conflicting_XML_values"

    if not values:
        return "XML_missing"

    if not table_column_present or is_missing(table_value):
        return "table_missing"

    if comparison_key(table_value) == comparison_key(values[0]):
        return "match"

    return "mismatch"


def summarize_field(patient_rows, prefix):
    counts = Counter(row[f"{prefix}_status"] for row in patient_rows)
    return {
        "patients": len(patient_rows),
        "status_counts": dict(sorted(counts.items())),
        "matches": counts.get("match", 0),
        "mismatches": counts.get("mismatch", 0),
        "xml_missing": counts.get("XML_missing", 0),
        "table_missing": counts.get("table_missing", 0),
        "conflicting_xml_values": counts.get("conflicting_XML_values", 0),
    }


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    all_cohort_results = {}
    all_patient_rows = []
    errors = []

    for cohort_name, config in COHORTS.items():
        cohort_root = config["root"]
        table_path = config["table"]

        if not cohort_root.is_dir():
            raise FileNotFoundError(f"{cohort_name} dataset directory not found: {cohort_root}")

        table_rows, table_audit = read_clinical_table(table_path)
        table_by_patient = {row["barcode"]: row for row in table_rows}

        # Match participant barcodes from XML paths, retaining every file.
        xml_by_patient = defaultdict(list)
        xml_files = sorted(
            p for p in cohort_root.rglob("*.xml")
            if "clinical" in p.name.lower()
        )

        for xml_path in xml_files:
            barcode = barcode_from_path(xml_path)
            if barcode:
                xml_by_patient[barcode].append(xml_path)

        patient_rows = []

        for barcode, table_row in sorted(table_by_patient.items()):
            kps_table, kps_col_present = table_field(table_row, KPS_TABLE_ALIASES)
            ecog_table, ecog_col_present = table_field(table_row, ECOG_TABLE_ALIASES)

            kps_evidence = []
            ecog_evidence = []
            parse_errors = []
            files = sorted(xml_by_patient.get(barcode, []))

            for xml_path in files:
                extracted, error = extract_xml_fields(xml_path)
                if error:
                    parse_errors.append(f"{xml_path.name}: {error}")
                    continue
                kps_evidence.extend(extracted["kps"])
                ecog_evidence.extend(extracted["ecog"])

            kps_values = distinct_usable([x["value"] for x in kps_evidence])
            ecog_values = distinct_usable([x["value"] for x in ecog_evidence])

            kps_status = status_for(kps_table, kps_col_present, kps_evidence)
            ecog_status = status_for(ecog_table, ecog_col_present, ecog_evidence)

            kps_table_missing = not kps_col_present or is_missing(kps_table)
            ecog_available_anywhere = (
                bool(ecog_values)
                or (ecog_col_present and not is_missing(ecog_table))
            )

            row = {
                "cohort": cohort_name,
                "patient_barcode": barcode,
                "xml_file_count": len(files),
                "xml_filenames": "; ".join(p.name for p in files),
                "kps_table_column_found": kps_col_present,
                "kps_table_raw": kps_table,
                "kps_xml_distinct_values": " | ".join(kps_values),
                "kps_xml_tags": " | ".join(sorted(set(x["tag"] for x in kps_evidence))),
                "kps_xml_evidence": " || ".join(
                    f'{x["file"]}::{x["tag"]}={x["value"]}' for x in kps_evidence
                ),
                "kps_status": kps_status,
                "ecog_table_column_found": ecog_col_present,
                "ecog_table_raw": ecog_table,
                "ecog_xml_distinct_values": " | ".join(ecog_values),
                "ecog_xml_tags": " | ".join(sorted(set(x["tag"] for x in ecog_evidence))),
                "ecog_xml_evidence": " || ".join(
                    f'{x["file"]}::{x["tag"]}={x["value"]}' for x in ecog_evidence
                ),
                "ecog_status": ecog_status,
                "kps_table_missing_ecog_available": (
                    kps_table_missing and ecog_available_anywhere
                ),
                "xml_parse_errors": " || ".join(parse_errors),
            }

            patient_rows.append(row)
            all_patient_rows.append(row)
            errors.extend(
                f"{cohort_name} {barcode}: {error}" for error in parse_errors
            )

        table_patients = set(table_by_patient)
        xml_patients = set(xml_by_patient)
        xml_only = sorted(xml_patients - table_patients)

        all_cohort_results[cohort_name] = {
            "table_path": str(table_path),
            "xml_root": str(cohort_root),
            "table_cleaning": table_audit,
            "clean_table_patients": len(table_rows),
            "clinical_xml_files_found": len(xml_files),
            "xml_participants_by_path": len(xml_patients),
            "patients_with_multiple_xml_files": sum(
                len(xml_by_patient[p]) > 1 for p in table_patients
            ),
            "table_patients_without_xml": sorted(table_patients - xml_patients),
            "xml_participants_absent_from_table": xml_only,
            "kps": summarize_field(patient_rows, "kps"),
            "ecog": summarize_field(patient_rows, "ecog"),
            "kps_missing_table_ecog_available": sum(
                row["kps_table_missing_ecog_available"] for row in patient_rows
            ),
            "kps_mismatch_patients": [
                row["patient_barcode"] for row in patient_rows
                if row["kps_status"] == "mismatch"
            ],
            "kps_conflicting_xml_patients": [
                row["patient_barcode"] for row in patient_rows
                if row["kps_status"] == "conflicting_XML_values"
            ],
            "ecog_mismatch_patients": [
                row["patient_barcode"] for row in patient_rows
                if row["ecog_status"] == "mismatch"
            ],
            "ecog_conflicting_xml_patients": [
                row["patient_barcode"] for row in patient_rows
                if row["ecog_status"] == "conflicting_XML_values"
            ],
            "patient_rows": patient_rows,
        }

    csv_path = ARTIFACTS / f"tcga_kps_ecog_patient_audit_{stamp}.csv"
    txt_path = ARTIFACTS / f"tcga_kps_ecog_patient_audit_{stamp}.txt"
    json_path = ARTIFACTS / f"tcga_kps_ecog_patient_audit_{stamp}.json"

    fieldnames = list(all_patient_rows[0].keys()) if all_patient_rows else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_patient_rows)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "raw_datasets_modified": False,
        "model_training_performed": False,
        "xml_association_method": (
            "Participant barcode extracted from XML file path; all matching "
            "clinical XML files aggregated per participant."
        ),
        "comparison_policy": {
            "one distinct usable XML value": "compare with table value",
            "multiple distinct usable XML values": "conflicting_XML_values",
            "no usable XML value": "XML_missing",
            "XML value exists but table field is missing/unavailable": "table_missing",
            "numeric formatting": "Normalize numeric equivalents such as 90 and 90.0",
            "automatic_ECOG_to_KPS_conversion": False,
            "automatic_source_precedence": False,
        },
        "cohorts": all_cohort_results,
        "xml_parse_errors": errors,
        "outputs": {
            "csv": str(csv_path),
            "txt": str(txt_path),
            "json": str(json_path),
        },
        "limitations": [
            "Barcode association is based on participant barcode in the XML path.",
            "Only explicitly recognized KPS/ECOG XML tag names are extracted.",
            "Different XML values are flagged; no record is selected automatically.",
            "Agreement does not establish clinical correctness or dictionary validity.",
            "No ECOG-to-KPS conversion is applied.",
        ],
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    lines = [
        "TCGA KPS/ECOG PATIENT-LEVEL DISCREPANCY AUDIT",
        f"Generated UTC: {report['generated_utc']}",
        "READ ONLY; raw datasets unchanged; no model training",
        "",
        "METHOD",
        "- Every clinical XML file associated with a participant barcode is inspected.",
        "- Distinct usable XML values are aggregated; the first XML file is never selected by default.",
        "- Numeric formatting differences such as 90 versus 90.0 are normalized.",
        "- Conflicting XML values remain flagged for manual review.",
        "- ECOG is not converted to KPS.",
        "",
    ]

    for cohort_name, result in all_cohort_results.items():
        lines.extend([
            f"=== {cohort_name} ===",
            f"Clean table patients: {result['clean_table_patients']}",
            f"Clinical XML files found: {result['clinical_xml_files_found']}",
            f"Unique XML participants by path: {result['xml_participants_by_path']}",
            f"Patients with multiple XML files: {result['patients_with_multiple_xml_files']}",
            f"Table patients without XML: {len(result['table_patients_without_xml'])}",
            f"XML participants absent from table: {len(result['xml_participants_absent_from_table'])}",
            "",
        ])

        for field in ("kps", "ecog"):
            s = result[field]
            lines.extend([
                f"{field.upper()} comparison:",
                f"  Matches: {s['matches']}",
                f"  Mismatches: {s['mismatches']}",
                f"  XML missing: {s['xml_missing']}",
                f"  Table missing: {s['table_missing']}",
                f"  Conflicting XML values: {s['conflicting_xml_values']}",
                f"  Status counts: {json.dumps(s['status_counts'], sort_keys=True)}",
                "",
            ])

        lines.extend([
            "KPS table-missing patients with ECOG available in table or XML: "
            f"{result['kps_missing_table_ecog_available']}",
            f"KPS mismatch patient barcodes: {', '.join(result['kps_mismatch_patients']) or 'None'}",
            "KPS conflicting-XML patient barcodes: "
            f"{', '.join(result['kps_conflicting_xml_patients']) or 'None'}",
            f"ECOG mismatch patient barcodes: {', '.join(result['ecog_mismatch_patients']) or 'None'}",
            "ECOG conflicting-XML patient barcodes: "
            f"{', '.join(result['ecog_conflicting_xml_patients']) or 'None'}",
            "",
        ])

    lines.extend([
        "XML PARSE ERRORS",
        *(errors if errors else ["None"]),
        "",
        "LIMITATIONS",
        *[f"- {item}" for item in report["limitations"]],
        "",
        f"CSV report: {csv_path}",
        f"JSON report: {json_path}",
        f"TXT report: {txt_path}",
    ])

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("\nAUDIT OUTPUT FILES")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print(f"TXT : {txt_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
