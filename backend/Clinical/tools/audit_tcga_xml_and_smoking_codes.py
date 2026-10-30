import csv, json, re
from collections import Counter, defaultdict
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight")
DATA = ROOT / "datasets"
OUT = ROOT / "backend" / "Clinical" / "artifacts"

COHORTS = {
    "LUAD": (
        DATA / "TCGA-LUAD",
        DATA / "TCGA-LUAD" / "42bf5eb2-bc49-45be-b18a-290f712b006c" / "nationwidechildrens.org_clinical_patient_luad.txt"
    ),
    "LUSC": (
        DATA / "TCGA-LUSC",
        DATA / "TCGA-LUSC" / "7aaee8d4-3bea-4dfc-b392-c7aaa3e9a550" / "nationwidechildrens.org_clinical_patient_lusc.txt"
    ),
}

SMOKING_MAP = {
    "1": "lifelong_non_smoker",
    "2": "current_smoker",
    "3": "former_smoker_quit_more_than_15_years",
    "4": "former_smoker_quit_15_years_or_less",
    "5": "former_smoker_duration_unspecified",
}
UNKNOWN_CODES = {"7", "77"}
PATIENT_COL = "bcr_patient_barcode"
FEATURES = [
    "age_at_initial_pathologic_diagnosis", "gender",
    "tobacco_smoking_history_indicator", "race",
    "tobacco_smoking_pack_years_smoked",
    "tobacco_smoking_year_started", "tobacco_smoking_year_stopped",
    "karnofsky_score", "num_nodules",
]
XML_HINTS = {
    "tobacco_smoking_history_indicator": ("tobacco_smoking_history_indicator", "2181650"),
    "tobacco_smoking_year_started": ("year_of_tobacco_smoking_onset", "2228604"),
    "tobacco_smoking_year_stopped": ("stopped_smoking_year", "2228610"),
    "karnofsky_score": ("karnofsky_score", "karnofsky_performance_score"),
    "ecog": ("ecog_performance_status", "ecog"),
}
MISSING = {
    "", "na", "n/a", "nan", "null", "none", "unknown",
    "[not available]", "[not applicable]", "[unknown]",
    "[discrepancy]", "not reported", "not available", "not applicable"
}

def s(value):
    return "" if value is None else str(value).strip()

def missing(value):
    v = s(value).casefold()
    return v in MISSING or v.startswith("[discrepancy]")

def metadata_row(row):
    return any(re.match(r"^\s*CDE_ID(?:\s*:|$)", s(v), re.I)
               for v in row.values())

def valid_barcode(value):
    return bool(re.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", s(value).upper()))

def get(row, name):
    for k, v in row.items():
        if k.strip().casefold() == name.casefold():
            return s(v)
    return ""

def load_table(path):
    if not path.exists():
        raise FileNotFoundError(f"Clinical table not found: {path}")
    accepted, metadata, invalid, duplicates = [], 0, 0, 0
    seen = set()
    with path.open(encoding="latin1", newline="") as f:
        for raw in csv.DictReader(f, delimiter="\t"):
            row = {str(k).strip(): s(v) for k, v in raw.items() if k is not None}
            if metadata_row(row):
                metadata += 1
                continue
            barcode = s(row.get(PATIENT_COL)).upper()
            if not valid_barcode(barcode):
                invalid += 1
                continue
            row[PATIENT_COL] = barcode
            if barcode in seen:
                duplicates += 1
                continue
            seen.add(barcode)
            accepted.append(row)
    return accepted, {
        "metadata_rows_removed": metadata,
        "invalid_or_blank_barcode_rows_removed": invalid,
        "duplicate_patient_rows_ignored": duplicates,
        "unique_patients": len(accepted),
    }

def barcode_from_path(path):
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", str(path).upper())
    return m.group(1) if m else None

@lru_cache(maxsize=None)
def xml_fields(path):
    result = defaultdict(list)
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:
        return {"__parse_error__": [str(exc)]}
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        value = s(el.text)
        if not value:
            continue
        attrs = {str(k).rsplit("}", 1)[-1].casefold(): s(v)
                 for k, v in el.attrib.items()}
        searchable = " ".join([tag, *attrs.keys(), *attrs.values()]).casefold()
        for field, hints in XML_HINTS.items():
            if any(h.casefold() in searchable for h in hints):
                result[field].append({"tag": tag, "value": value})
    return dict(result)

def norm_compare(v):
    if missing(v):
        return ""
    return re.sub(r"\.0+$", "", s(v)).casefold()

def audit_cohort(name, root, table_path):
    rows, cleaning = load_table(table_path)
    patients = {r[PATIENT_COL] for r in rows}

    missingness = {}
    for field in FEATURES:
        if field == "num_nodules":
            missingness[field] = {
                "missing": len(rows), "nonmissing": 0,
                "note": "No verified patient-level TCGA source identified"
            }
        else:
            values = [get(r, field) for r in rows]
            missingness[field] = {
                "missing": sum(not s(v) or missing(v) for v in values),
                "nonmissing": sum(bool(s(v)) and not missing(v) for v in values)
            }

    smoking_values = [get(r, "tobacco_smoking_history_indicator") for r in rows]
    smoking_freq = dict(Counter(v if v else "<BLANK>" for v in smoking_values))
    smoking_issues = []
    for row in rows:
        value = get(row, "tobacco_smoking_history_indicator")
        if value in UNKNOWN_CODES or (
            value and not missing(value) and value not in SMOKING_MAP
        ):
            smoking_issues.append({
                "patient": row[PATIENT_COL],
                "raw_value": value,
                "classification": (
                    "unknown_code" if value in UNKNOWN_CODES else "unrecognized_value"
                )
            })
        elif value.casefold().startswith("[discrepancy]"):
            smoking_issues.append({
                "patient": row[PATIENT_COL],
                "raw_value": value,
                "classification": "discrepancy"
            })

    xml_files = sorted(p for p in root.rglob("*.xml") if p.is_file())
    xml_by_patient = defaultdict(list)
    unkeyed = []
    for path in xml_files:
        patient = barcode_from_path(path)
        if patient:
            xml_by_patient[patient].append(path)
        else:
            unkeyed.append(str(path))

    xml_patients = set(xml_by_patient)
    matched = sorted(patients & xml_patients)
    comparisons = {}
    for field in XML_HINTS:
        compared = matches = mismatches = ambiguous = no_xml_value = 0
        mismatch_examples = []
        for row in rows:
            patient = row[PATIENT_COL]
            if patient not in xml_by_patient:
                continue
            files = xml_by_patient[patient]
            table_value = get(row, field)
            extracted = []
            for xml_file in files:
                extracted.extend(xml_fields(xml_file).get(field, []))
            values = sorted(set(
                x["value"] for x in extracted
                if not missing(x["value"])
            ))
            if len(values) > 1:
                ambiguous += 1
                continue
            if not values:
                no_xml_value += 1
                continue
            if not s(table_value) or missing(table_value):
                continue
            compared += 1
            if norm_compare(table_value) == norm_compare(values[0]):
                matches += 1
            else:
                mismatches += 1
                if len(mismatch_examples) < 25:
                    mismatch_examples.append({
                        "patient": patient,
                        "table_value": table_value,
                        "xml_value": values[0],
                        "xml_file": str(files[0])
                    })
        comparisons[field] = {
            "comparable_patient_count": compared,
            "matches": matches,
            "mismatches": mismatches,
            "ambiguous_multiple_xml_or_values": ambiguous,
            "xml_field_not_found": no_xml_value,
            "mismatch_examples": mismatch_examples,
        }

    multi_xml = [
        {"patient": p, "count": len(paths), "files": [str(x) for x in paths[:8]]}
        for p, paths in sorted(xml_by_patient.items()) if len(paths) > 1
    ]
    ecog_columns = Counter()
    for row in rows:
        for key, value in row.items():
            if "ecog" in key.casefold() and s(value) and not missing(value):
                ecog_columns[key] += 1
    kps_missing_ecog = sum(
        (not s(get(row, "karnofsky_score")) or missing(get(row, "karnofsky_score")))
        and any("ecog" in k.casefold() and s(v) and not missing(v)
                for k, v in row.items())
        for row in rows
    )

    return {
        "cohort": name,
        "clinical_table": str(table_path),
        "cleaning": cleaning,
        "smoking": {
            "raw_value_frequencies": smoking_freq,
            "user_supplied_mapping_for_review": SMOKING_MAP,
            "unknown_codes": sorted(UNKNOWN_CODES),
            "unknown_discrepancy_unrecognized_records": smoking_issues,
            "note": "Mapping supplied by user; authoritative source is not verified by this script."
        },
        "feature_missingness": missingness,
        "xml_matching": {
            "xml_file_count": len(xml_files),
            "unique_xml_patient_count": len(xml_patients),
            "table_patients_matched_to_xml": len(matched),
            "table_patients_without_xml": len(patients - xml_patients),
            "xml_patients_absent_from_table": len(xml_patients - patients),
            "xml_files_without_patient_barcode": len(unkeyed),
            "patients_with_multiple_xml_files": len(multi_xml),
            "multiple_xml_examples": multi_xml[:100],
            "unkeyed_xml_examples": unkeyed[:25],
        },
        "table_xml_comparison": comparisons,
        "kps_ecog": {
            "kps_nonmissing": sum(bool(s(get(r, "karnofsky_score"))) and not missing(get(r, "karnofsky_score")) for r in rows),
            "kps_missing_but_ecog_column_available": kps_missing_ecog,
            "ecog_columns_nonmissing_counts": dict(ecog_columns),
            "mapping_applied": False
        },
        "num_nodules": {
            "verified_patient_level_source": None,
            "available": False,
            "action": "Do not infer from LIDC without verified patient linkage."
        }
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "model_training_performed": False,
        "raw_data_modified": False,
        "cohorts": {},
        "limitations": [
            "Smoking code mapping is user-supplied; verify against authoritative dictionary before training.",
            "XML barcode association is based on participant barcode in the XML path.",
            "Patients with multiple XML files are not silently reduced to one record for comparison.",
            "No KPS/ECOG conversion is applied.",
            "num_nodules remains unavailable without a verified patient-level source."
        ]
    }
    for name, (root, table) in COHORTS.items():
        report["cohorts"][name] = audit_cohort(name, root, table)

    luad = report["cohorts"]["LUAD"]["cleaning"]["unique_patients"]
    lusc = report["cohorts"]["LUSC"]["cleaning"]["unique_patients"]
    train_each = int(lusc * 0.8)
    report["preliminary_split_arithmetic"] = {
        "basis": "cleaned cohort counts only; not final eligible counts",
        "LUAD_total": luad,
        "LUSC_total": lusc,
        "training_per_class_if_80_percent_of_LUSC": train_each,
        "LUSC_test": lusc - train_each,
        "LUAD_test_if_equal_training_count": luad - train_each,
        "total_training": train_each * 2,
        "total_test": (lusc - train_each) + (luad - train_each),
        "warning": "Recalculate after final eligibility and missingness rules."
    }

    json_path = OUT / f"tcga_xml_smoking_correction_audit_{stamp}.json"
    txt_path = OUT / f"tcga_xml_smoking_correction_audit_{stamp}.txt"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "TCGA XML MATCHING AND SMOKING-CODE CORRECTION AUDIT",
        f"Generated UTC: {report['generated_utc']}",
        "READ ONLY; no model training; raw datasets unchanged",
        "",
        "SMOKING MAPPING SUPPLIED FOR AUDIT",
        "1 = lifelong non-smoker",
        "2 = current smoker",
        "3 = former smoker, quit >15 years",
        "4 = former smoker, quit <=15 years",
        "5 = former smoker, duration unspecified",
        "7 and 77 = unknown/not documented where applicable",
        "Authoritative dictionary verification remains required.",
        ""
    ]
    for name in ("LUAD", "LUSC"):
        c = report["cohorts"][name]
        lines += [
            f"=== {name} ===",
            f"Unique patients: {c['cleaning']['unique_patients']}",
            f"Metadata rows removed: {c['cleaning']['metadata_rows_removed']}",
            f"Invalid barcode rows removed: {c['cleaning']['invalid_or_blank_barcode_rows_removed']}",
            f"Duplicate rows ignored: {c['cleaning']['duplicate_patient_rows_ignored']}",
            "Smoking raw frequencies:"
        ]
        lines.extend(f"  {k}: {v}" for k, v in c["smoking"]["raw_value_frequencies"].items())
        lines.append(f"Smoking unknown/discrepancy/unrecognized records: {len(c['smoking']['unknown_discrepancy_unrecognized_records'])}")
        lines.append("Feature missingness:")
        for field, values in c["feature_missingness"].items():
            lines.append(f"  {field}: {values['missing']} missing / {c['cleaning']['unique_patients']}")
        x = c["xml_matching"]
        lines += [
            f"XML files: {x['xml_file_count']}",
            f"Unique XML participants: {x['unique_xml_patient_count']}",
            f"Table patients matched to XML: {x['table_patients_matched_to_xml']}",
            f"Table patients without XML: {x['table_patients_without_xml']}",
            f"XML participants absent from table: {x['xml_patients_absent_from_table']}",
            f"Patients with multiple XML files: {x['patients_with_multiple_xml_files']}",
            "Table/XML comparisons (ambiguous participants excluded):"
        ]
        for field, comp in c["table_xml_comparison"].items():
            lines.append(
                f"  {field}: {comp['matches']} matches / {comp['comparable_patient_count']} comparisons; "
                f"{comp['mismatches']} mismatches; {comp['ambiguous_multiple_xml_or_values']} ambiguous"
            )
        lines += [
            f"KPS nonmissing: {c['kps_ecog']['kps_nonmissing']}",
            f"KPS missing with an available ECOG field: {c['kps_ecog']['kps_missing_but_ecog_column_available']}",
            "ECOG-to-KPS conversion applied: NO",
            "num_nodules: unavailable; no LIDC cross-cohort assignment",
            ""
        ]
    lines += [
        "=== PRELIMINARY SPLIT ARITHMETIC ===",
        json.dumps(report["preliminary_split_arithmetic"], indent=2),
        "",
        "LIMITATIONS:",
        *[f"- {x}" for x in report["limitations"]],
        "",
        f"JSON report: {json_path}",
        f"TXT report: {txt_path}"
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print("AUDIT COMPLETE")
    print(f"TXT:  {txt_path}")
    print(f"JSON: {json_path}")
    for name in ("LUAD", "LUSC"):
        c = report["cohorts"][name]
        print(
            f"{name}: {c['cleaning']['unique_patients']} patients; "
            f"{c['xml_matching']['xml_file_count']} XML files / "
            f"{c['xml_matching']['unique_xml_patient_count']} XML participants; "
            f"{len(c['smoking']['unknown_discrepancy_unrecognized_records'])} smoking values to review"
        )
    print("No training or raw-data modification performed.")

if __name__ == "__main__":
    main()
