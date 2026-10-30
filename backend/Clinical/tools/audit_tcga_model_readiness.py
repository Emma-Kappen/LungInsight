from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight")
DATA = ROOT / "datasets"
OUT = ROOT / "backend" / "Clinical" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
TXT_PATH = OUT / f"tcga_model_readiness_audit_{STAMP}.txt"
JSON_PATH = OUT / f"tcga_model_readiness_audit_{STAMP}.json"

COHORTS = {
    "LUAD": {
        "directory": DATA / "TCGA-LUAD",
        "table_name": "nationwidechildrens.org_clinical_patient_luad.txt",
    },
    "LUSC": {
        "directory": DATA / "TCGA-LUSC",
        "table_name": "nationwidechildrens.org_clinical_patient_lusc.txt",
    },
}

FEATURES = [
    "age_at_initial_pathologic_diagnosis",
    "gender",
    "tobacco_smoking_history_indicator",
    "race",
    "tobacco_smoking_pack_years_smoked",
    "tobacco_smoking_year_started",
    "tobacco_smoking_year_stopped",
    "karnofsky_score",
    "num_nodules",
]

# Only these explicit names are considered evidence of a human-readable
# smoking-status definition. Numeric values 1-5 are never decoded here.
STATUS_TEXT = {
    "never smoker": "never",
    "never smoked": "never",
    "current smoker": "current",
    "current reformed smoker": "current_or_reformed_ambiguous",
    "former smoker": "former",
    "ex-smoker": "former",
    "past smoker": "former",
}

COLUMN_ALIASES = {
    "age_at_initial_pathologic_diagnosis": [
        "age_at_initial_pathologic_diagnosis",
    ],
    "gender": ["gender"],
    "tobacco_smoking_history_indicator": [
        "tobacco_smoking_history_indicator",
        "tobacco_smoking_history",
    ],
    "race": ["race"],
    "tobacco_smoking_pack_years_smoked": [
        "tobacco_smoking_pack_years_smoked",
        "pack_years_smoked",
    ],
    "tobacco_smoking_year_started": [
        "tobacco_smoking_year_started",
        "year_of_tobacco_smoking_onset",
    ],
    "tobacco_smoking_year_stopped": [
        "tobacco_smoking_year_stopped",
        "stopped_smoking_year",
    ],
    "karnofsky_score": ["karnofsky_score"],
    "num_nodules": ["num_nodules", "nodule_count"],
}

XML_ALIASES = {
    "tobacco_smoking_history_indicator": {
        "tobacco_smoking_history_indicator",
        "tobacco_smoking_history",
    },
    "tobacco_smoking_year_started": {
        "tobacco_smoking_year_started",
        "year_of_tobacco_smoking_onset",
    },
    "tobacco_smoking_year_stopped": {
        "tobacco_smoking_year_stopped",
        "stopped_smoking_year",
    },
    "karnofsky_score": {"karnofsky_score"},
    "age_at_initial_pathologic_diagnosis": {
        "age_at_initial_pathologic_diagnosis",
    },
    "gender": {"gender"},
    "race": {"race"},
    "tobacco_smoking_pack_years_smoked": {
        "tobacco_smoking_pack_years_smoked",
        "pack_years_smoked",
    },
}

def clean(value):
    if value is None:
        return ""
    return str(value).strip()

def norm(value):
    return re.sub(r"\s+", " ", clean(value)).strip().lower()

def is_metadata_row(row):
    return any(
        re.match(r"^\s*CDE_ID(?:\s*:|$)", clean(v), re.IGNORECASE)
        for v in row.values()
    )

def is_patient_barcode(value):
    return bool(re.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-|$)", clean(value), re.I))

def patient_key(value):
    value = clean(value).upper()
    return value[:12] if value.startswith("TCGA-") else value

def missing(value):
    return norm(value) in {
        "", "na", "n/a", "nan", "null", "none", "[not available]",
        "[not applicable]", "[unknown]", "not reported", "unknown",
    }

def locate_table(directory, name):
    matches = list(directory.rglob(name)) if directory.exists() else []
    if not matches:
        return None
    # Deterministic choice; record all candidates so duplicates are visible.
    matches.sort(key=lambda p: (len(p.parts), str(p).lower()))
    return matches[0]

def load_table(path):
    result = {
        "path": str(path) if path else None,
        "candidate_count": 0,
        "headers": [],
        "raw_rows": 0,
        "metadata_rows_removed": 0,
        "invalid_barcode_rows_removed": 0,
        "blank_barcode_rows_removed": 0,
        "duplicate_barcode_rows": 0,
        "duplicate_patient_keys": 0,
        "records": [],
        "errors": [],
    }
    if not path or not path.exists():
        result["errors"].append("Clinical patient table not found.")
        return result

    try:
        with path.open("r", encoding="latin1", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)
    except Exception as e:
        result["errors"].append(f"Read error: {type(e).__name__}: {e}")
        return result

    if not rows:
        result["errors"].append("File is empty.")
        return result

    header_index = None
    for i, row in enumerate(rows[:20]):
        if any(norm(cell) == "bcr_patient_barcode" for cell in row):
            header_index = i
            break
    if header_index is None:
        result["errors"].append("Could not locate bcr_patient_barcode header.")
        return result

    headers = [clean(h) for h in rows[header_index]]
    result["headers"] = headers
    raw_records = []
    for row in rows[header_index + 1:]:
        if not row or not any(clean(v) for v in row):
            continue
        padded = row + [""] * max(0, len(headers) - len(row))
        raw_records.append({headers[i]: clean(padded[i]) for i in range(len(headers))})
    result["raw_rows"] = len(raw_records)

    barcode_col = next(
        (h for h in headers if norm(h) == "bcr_patient_barcode"), None
    )
    cleaned = []
    for rec in raw_records:
        if is_metadata_row(rec):
            result["metadata_rows_removed"] += 1
            continue
        barcode = clean(rec.get(barcode_col, "")) if barcode_col else ""
        if not barcode:
            result["blank_barcode_rows_removed"] += 1
            continue
        if not is_patient_barcode(barcode):
            result["invalid_barcode_rows_removed"] += 1
            continue
        cleaned.append(rec)

    counts = Counter(clean(r.get(barcode_col, "")) for r in cleaned)
    result["duplicate_barcode_rows"] = sum(n - 1 for n in counts.values() if n > 1)
    key_counts = Counter(patient_key(r.get(barcode_col, "")) for r in cleaned)
    result["duplicate_patient_keys"] = sum(n - 1 for n in key_counts.values() if n > 1)

    # Keep one row per full barcode for audit calculations; duplicate details
    # remain reported and should be resolved before training.
    seen = set()
    unique = []
    for rec in cleaned:
        barcode = clean(rec.get(barcode_col, ""))
        if barcode not in seen:
            seen.add(barcode)
            unique.append(rec)

    result["records"] = unique
    result["clean_rows_before_deduplication"] = len(cleaned)
    result["clean_unique_barcode_rows"] = len(unique)
    return result

def find_col(headers, feature):
    names = {norm(h): h for h in headers}
    for alias in COLUMN_ALIASES.get(feature, [feature]):
        if norm(alias) in names:
            return names[norm(alias)]
    return None

def feature_audit(records, headers):
    output = {}
    for feature in FEATURES:
        col = find_col(headers, feature)
        if feature == "num_nodules":
            output[feature] = {
                "column_found": bool(col),
                "source_column": col,
                "nonmissing": sum(not missing(r.get(col, "")) for r in records) if col else 0,
                "missing": sum(missing(r.get(col, "")) for r in records) if col else len(records),
                "note": (
                    "Column found in patient table; semantics and source still require verification."
                    if col else
                    "Not found in this patient table. Do not transfer LIDC counts to TCGA without verified patient linkage."
                ),
            }
            continue
        if not col:
            output[feature] = {
                "column_found": False,
                "source_column": None,
                "nonmissing": 0,
                "missing": len(records),
            }
            continue
        vals = [clean(r.get(col, "")) for r in records]
        output[feature] = {
            "column_found": True,
            "source_column": col,
            "nonmissing": sum(not missing(v) for v in vals),
            "missing": sum(missing(v) for v in vals),
            "unique_nonmissing_values": len({norm(v) for v in vals if not missing(v)}),
        }
        if feature in ("gender", "race", "tobacco_smoking_history_indicator"):
            output[feature]["value_counts"] = dict(
                Counter(v for v in vals if not missing(v)).most_common(50)
            )
    return output

def status_year_cross_tab(records, headers):
    status_col = find_col(headers, "tobacco_smoking_history_indicator")
    start_col = find_col(headers, "tobacco_smoking_year_started")
    stop_col = find_col(headers, "tobacco_smoking_year_stopped")
    result = {
        "status_column": status_col,
        "start_year_column": start_col,
        "stop_year_column": stop_col,
        "by_status_value": {},
        "explicit_text_status_counts": {},
        "hypothetical_zero_conversion": {},
        "interpretation": (
            "This is a missingness audit only. Numeric status codes are not decoded. "
            "No year is converted to zero."
        ),
    }
    if not status_col:
        result["warning"] = "Smoking status column not found."
        return result

    groups = defaultdict(list)
    explicit = Counter()
    for rec in records:
        status = clean(rec.get(status_col, "")) or "[MISSING]"
        groups[status].append(rec)
        if not missing(status):
            st = norm(status)
            matched = next((v for k, v in STATUS_TEXT.items() if k in st), None)
            if matched:
                explicit[matched] += 1

    for status, group in sorted(groups.items(), key=lambda x: x[0]):
        start_missing = sum(missing(r.get(start_col, "")) for r in group) if start_col else len(group)
        stop_missing = sum(missing(r.get(stop_col, "")) for r in group) if stop_col else len(group)
        result["by_status_value"][status] = {
            "patients": len(group),
            "start_year_missing": start_missing,
            "start_year_present": len(group) - start_missing,
            "stop_year_missing": stop_missing,
            "stop_year_present": len(group) - stop_missing,
            "both_years_missing": sum(
                (missing(r.get(start_col, "")) if start_col else True)
                and (missing(r.get(stop_col, "")) if stop_col else True)
                for r in group
            ),
        }

    result["explicit_text_status_counts"] = dict(explicit)
    # Quantify, without applying, the effect of a proposed rule that a given
    # numeric code might mean "never smoker". This is a sensitivity table,
    # not a code mapping or recommendation.
    for candidate in ("1", "2", "3", "4", "5"):
        group = groups.get(candidate, [])
        result["hypothetical_zero_conversion"][candidate] = {
            "patients_with_code": len(group),
            "start_years_missing_that_would_be_zero_if_this_code_meant_never": (
                sum(missing(r.get(start_col, "")) for r in group) if start_col else len(group)
            ),
            "stop_years_missing_that_would_be_zero_if_this_code_meant_never": (
                sum(missing(r.get(stop_col, "")) for r in group) if stop_col else len(group)
            ),
            "applied": False,
        }
    return result

def xml_tag_value_map(path):
    found = defaultdict(list)
    parse_error = None
    try:
        root = ET.parse(path).getroot()
        for elem in root.iter():
            text = clean(elem.text)
            if not text:
                continue
            attrs = {norm(k): clean(v) for k, v in elem.attrib.items()}
            preferred = attrs.get("preferred_name", "")
            cde = attrs.get("cde", "")
            tag = elem.tag.split("}")[-1]
            found[norm(preferred)].append({"value": text, "tag": tag, "cde": cde})
            found[norm(tag)].append({"value": text, "tag": tag, "cde": cde})
    except Exception as e:
        parse_error = f"{type(e).__name__}: {e}"
    return found, parse_error

def xml_audit(cohort_name, cohort_dir, records, headers):
    files = sorted(cohort_dir.rglob("*.xml")) if cohort_dir.exists() else []
    patient_col = next((h for h in headers if norm(h) == "bcr_patient_barcode"), None)
    table_by_key = {
        patient_key(r.get(patient_col, "")): r
        for r in records
    } if patient_col else {}

    matched = 0
    parsed = 0
    parse_errors = []
    xml_by_key = {}
    xml_status_counts = Counter()
    cde_counts = Counter()
    xml_field_counts = Counter()
    compare = {
        "tobacco_smoking_history_indicator": {"compared": 0, "matches": 0, "mismatches": 0, "unavailable": 0, "examples": []},
        "tobacco_smoking_year_started": {"compared": 0, "matches": 0, "mismatches": 0, "unavailable": 0, "examples": []},
        "tobacco_smoking_year_stopped": {"compared": 0, "matches": 0, "mismatches": 0, "unavailable": 0, "examples": []},
    }

    for path in files:
        # Typical TCGA filename contains the participant barcode.
        m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", path.name, re.I)
        if not m:
            m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", str(path), re.I)
        if not m:
            continue
        key = patient_key(m.group(1))
        values, err = xml_tag_value_map(path)
        if err:
            if len(parse_errors) < 20:
                parse_errors.append({"file": str(path), "error": err})
            continue
        parsed += 1
        xml_by_key[key] = values
        if key in table_by_key:
            matched += 1

        for alias in XML_ALIASES["tobacco_smoking_history_indicator"]:
            for item in values.get(norm(alias), []):
                xml_status_counts[item["value"]] += 1
                if item["cde"]:
                    cde_counts[item["cde"]] += 1

        for feature in XML_ALIASES:
            if any(values.get(norm(alias)) for alias in XML_ALIASES[feature]):
                xml_field_counts[feature] += 1

        if key not in table_by_key:
            continue
        rec = table_by_key[key]
        for feature, stats in compare.items():
            tcol = find_col(headers, feature)
            xml_items = []
            for alias in XML_ALIASES[feature]:
                xml_items.extend(values.get(norm(alias), []))
            xml_vals = [clean(x["value"]) for x in xml_items if not missing(x["value"])]
            table_val = clean(rec.get(tcol, "")) if tcol else ""
            if not tcol or missing(table_val) or not xml_vals:
                stats["unavailable"] += 1
                continue
            stats["compared"] += 1
            if norm(table_val) in {norm(x) for x in xml_vals}:
                stats["matches"] += 1
            else:
                stats["mismatches"] += 1
                if len(stats["examples"]) < 8:
                    stats["examples"].append({
                        "patient": key,
                        "table_value": table_val,
                        "xml_values": xml_vals[:4],
                        "xml_file": str(path),
                    })

    return {
        "xml_files_found": len(files),
        "xml_files_parsed": parsed,
        "xml_files_with_patient_barcode": len(xml_by_key),
        "xml_patient_keys_matching_cleaned_table": matched,
        "table_unique_patient_keys": len(table_by_key),
        "table_patients_with_matching_xml": matched,
        "parse_errors_sample": parse_errors,
        "xml_fields_seen_by_patient_count": dict(xml_field_counts),
        "smoking_status_xml_value_counts": dict(xml_status_counts.most_common(100)),
        "smoking_status_cde_counts": dict(cde_counts),
        "table_vs_xml_comparison": compare,
        "caveat": (
            "XML filenames and values may represent multiple record types or repeated fields. "
            "A mismatch is an audit flag, not proof that the table is wrong. XML absence is not proof of missing source data."
        ),
    }

def dictionary_search(cohort_dirs):
    extensions = {".xsd", ".xml", ".json", ".txt", ".tsv", ".csv", ".md", ".html", ".htm", ".yaml", ".yml"}
    terms = [
        re.compile(r"\b2181650\b", re.I),
        re.compile(r"tobacco_smoking_history_indicator", re.I),
        re.compile(r"never smoker|never smoked|current smoker|former smoker|ex-smoker", re.I),
        re.compile(r"(smoking.{0,80}(code|value|enum|permissible|permitted)|"
                   r"(code|value|enum|permissible|permitted).{0,80}smoking)", re.I),
    ]
    matches = []
    scanned = 0
    unreadable = 0
    truncated_files = 0
    max_bytes = 8 * 1024 * 1024
    max_matches = 500

    for directory in cohort_dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            try:
                if path.stat().st_size > max_bytes:
                    truncated_files += 1
                    continue
                content = path.read_text(encoding="latin1", errors="replace")
                scanned += 1
            except Exception:
                unreadable += 1
                continue
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if not any(t.search(line) for t in terms):
                    continue
                # Context around code identifier; preserve evidence, not interpretation.
                excerpt = line.strip()[:500]
                matches.append({
                    "file": str(path),
                    "line": i + 1,
                    "excerpt": excerpt,
                })
                if len(matches) >= max_matches:
                    return {
                        "files_scanned": scanned,
                        "unreadable_files": unreadable,
                        "files_skipped_size_limit": truncated_files,
                        "matches_capped": True,
                        "match_count": len(matches),
                        "matches": matches,
                        "note": "Matches are raw local evidence. A code definition is accepted only if an authoritative definition explicitly maps each code to a meaning.",
                    }
    return {
        "files_scanned": scanned,
        "unreadable_files": unreadable,
        "files_skipped_size_limit": truncated_files,
        "matches_capped": False,
        "match_count": len(matches),
        "matches": matches,
        "note": "Matches are raw local evidence. A code definition is accepted only if an authoritative definition explicitly maps each code to a meaning.",
    }

def count_kps_derivable(records, headers):
    kps_col = find_col(headers, "karnofsky_score")
    ecog_col = next((h for h in headers if norm(h) in {"ecog_score", "ecog_performance_status", "performance_status"}), None)
    if not ecog_col:
        ecog_col = next((h for h in headers if "ecog" in norm(h)), None)
    if not ecog_col:
        return {
            "ecog_column_found": False,
            "ecog_column": None,
            "observed_kps": sum(not missing(r.get(kps_col, "")) for r in records) if kps_col else 0,
            "ecog_present_and_kps_missing": 0,
            "ecog_values": {},
            "mapping_applied": False,
            "note": "No ECOG-like column found; no KPS derivation performed.",
        }
    ecog_values = Counter(clean(r.get(ecog_col, "")) for r in records if not missing(r.get(ecog_col, "")))
    derivable = sum(
        not missing(r.get(ecog_col, ""))
        and (not kps_col or missing(r.get(kps_col, "")))
        for r in records
    )
    return {
        "ecog_column_found": True,
        "ecog_column": ecog_col,
        "observed_kps": sum(not missing(r.get(kps_col, "")) for r in records) if kps_col else 0,
        "ecog_present_and_kps_missing": derivable,
        "ecog_values": dict(ecog_values),
        "mapping_applied": False,
        "candidate_approximate_mapping_for_review_only": {
            "0": "90-100",
            "1": "70-80",
            "2": "50-60",
            "3": "30-40",
            "4": "10-20",
        },
        "note": "Approximate mapping only; do not apply automatically. Any derived KPS must retain ECOG source and derivation provenance.",
    }

def main():
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "read_only": True,
        "training_performed": False,
        "raw_data_modified": False,
        "cohorts": {},
        "dictionary_search": {},
        "model_readiness": {},
        "findings": [],
    }

    cohort_dirs = []
    for name, cfg in COHORTS.items():
        directory = cfg["directory"]
        cohort_dirs.append(directory)
        matches = list(directory.rglob(cfg["table_name"])) if directory.exists() else []
        table_path = locate_table(directory, cfg["table_name"])
        loaded = load_table(table_path)
        records = loaded.pop("records", [])
        headers = loaded.get("headers", [])
        features = feature_audit(records, headers)
        status_years = status_year_cross_tab(records, headers)
        xml = xml_audit(name, directory, records, headers)
        kps = count_kps_derivable(records, headers)

        report["cohorts"][name] = {
            "table_candidates_found": len(matches),
            **loaded,
            "clean_unique_patient_count": len(records),
            "features": features,
            "smoking_status_year_missingness": status_years,
            "xml_audit": xml,
            "kps_ecog_audit": kps,
        }

    report["dictionary_search"] = dictionary_search(cohort_dirs)

    # Cohort-level counts and provisional split arithmetic.
    luad = report["cohorts"].get("LUAD", {})
    lusc = report["cohorts"].get("LUSC", {})
    luad_n = luad.get("clean_unique_patient_count", 0)
    lusc_n = lusc.get("clean_unique_patient_count", 0)

    # Preliminary only: the final eligible set is unknown until missingness,
    # label rules and the num_nodules requirement are resolved.
    lusc_train = int(lusc_n * 0.80)
    luad_train = min(lusc_train, luad_n)
    lusc_test = lusc_n - lusc_train
    luad_test = luad_n - luad_train

    report["model_readiness"] = {
        "proposed_feature_count": len(FEATURES),
        "proposed_features": FEATURES,
        "num_nodules_source_resolved": any(
            c.get("features", {}).get("num_nodules", {}).get("column_found", False)
            for c in report["cohorts"].values()
        ),
        "preliminary_split_arithmetic_only": {
            "rule": "floor(0.80 * cleaned LUSC row count) as provisional LUSC train count; equal number of LUAD training patients; all remaining cleaned patients provisionally test",
            "lusc_clean_rows": lusc_n,
            "luad_clean_rows": luad_n,
            "provisional_lusc_train": lusc_train,
            "provisional_luad_train": luad_train,
            "provisional_lusc_test": lusc_test,
            "provisional_luad_test": luad_test,
            "warning": "Not the final split. Apply patient-level stratification, fixed seed, eligibility filters and leakage checks only after source/label decisions. Counts may change.",
        },
        "imputation_readiness": {
            "ready_for_training": False,
            "recommendation": (
                "Do not run naïve KNN on label-encoded categorical values. "
                "Fit all preprocessing on training data only. Treat gender, race and smoking-status codes as categorical; "
                "do not assign numeric order to smoking codes. Consider a mixed-type pipeline with separate categorical "
                "and numeric imputers, or a justified mixed-distance method. Scale numeric features before any numeric-distance KNN."
            ),
            "missingness_policy_unresolved": True,
        },
        "metrics_required": [
            "balanced_accuracy",
            "macro_f1",
            "per_class_sensitivity",
            "confusion_matrix",
            "roc_auc",
            "pr_auc",
            "calibration",
            "uncertainty_or_confidence_intervals",
        ],
    }

    # Conservative explicit findings.
    report["findings"].append(
        "Smoking code meanings are unresolved unless an authoritative local dictionary explicitly maps the numeric codes."
    )
    report["findings"].append(
        "No year-zero conversion was performed. Missing year is not equivalent to zero without a verified status rule."
    )
    report["findings"].append(
        "num_nodules must not be populated from LIDC for TCGA patients without verified patient-level linkage."
    )
    report["findings"].append(
        "KPS/ECOG approximate mapping was audited but not applied."
    )
    report["findings"].append(
        "Preliminary split counts are arithmetic only and are not final eligible-patient counts."
    )

    # Save JSON first; text report is a concise human-readable companion.
    JSON_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("TCGA LUAD/LUSC MODEL READINESS AUDIT")
    lines.append("=" * 72)
    lines.append(f"Generated UTC: {report['generated_utc']}")
    lines.append("Read-only: yes | Raw data modified: no | Training performed: no")
    lines.append("")
    for name in ("LUAD", "LUSC"):
        c = report["cohorts"].get(name, {})
        lines.append(f"[{name}]")
        lines.append(f"  Table: {c.get('path')}")
        lines.append(f"  Table candidates: {c.get('table_candidates_found', 0)}")
        lines.append(f"  Raw nonblank rows: {c.get('raw_rows', 0)}")
        lines.append(f"  CDE_ID metadata rows removed: {c.get('metadata_rows_removed', 0)}")
        lines.append(f"  Invalid barcode rows removed: {c.get('invalid_barcode_rows_removed', 0)}")
        lines.append(f"  Blank barcode rows removed: {c.get('blank_barcode_rows_removed', 0)}")
        lines.append(f"  Duplicate barcode rows: {c.get('duplicate_barcode_rows', 0)}")
        lines.append(f"  Duplicate patient keys: {c.get('duplicate_patient_keys', 0)}")
        lines.append(f"  Clean unique patients: {c.get('clean_unique_patient_count', 0)}")
        lines.append("  Feature missingness:")
        for feature, stats in c.get("features", {}).items():
            lines.append(
                f"    {feature}: found={stats.get('column_found')} "
                f"nonmissing={stats.get('nonmissing')} missing={stats.get('missing')} "
                f"column={stats.get('source_column')}"
            )
        sy = c.get("smoking_status_year_missingness", {})
        lines.append("  Smoking status by raw value (codes not decoded):")
        for value, stats in sy.get("by_status_value", {}).items():
            lines.append(
                f"    {value}: n={stats['patients']}, "
                f"start_missing={stats['start_year_missing']}, "
                f"stop_missing={stats['stop_year_missing']}, "
                f"both_missing={stats['both_years_missing']}"
            )
        lines.append("  Hypothetical zero conversion counts (NOT applied):")
        for code, stats in sy.get("hypothetical_zero_conversion", {}).items():
            lines.append(
                f"    if code {code} meant never smoker: "
                f"start_year_missing={stats['start_years_missing_that_would_be_zero_if_this_code_meant_never']}, "
                f"stop_year_missing={stats['stop_years_missing_that_would_be_zero_if_this_code_meant_never']}"
            )
        x = c.get("xml_audit", {})
        lines.append(
            f"  XML: found={x.get('xml_files_found', 0)}, parsed={x.get('xml_files_parsed', 0)}, "
            f"matching patient keys={x.get('table_patients_with_matching_xml', 0)}"
        )
        lines.append("  Table/XML comparisons:")
        for feature, stats in x.get("table_vs_xml_comparison", {}).items():
            lines.append(
                f"    {feature}: compared={stats['compared']}, matches={stats['matches']}, "
                f"mismatches={stats['mismatches']}, unavailable={stats['unavailable']}"
            )
            for ex in stats.get("examples", [])[:3]:
                lines.append(
                    f"      example patient={ex['patient']} table={ex['table_value']} xml={ex['xml_values']}"
                )
        k = c.get("kps_ecog_audit", {})
        lines.append(
            f"  KPS/ECOG: KPS observed={k.get('observed_kps', 0)}, "
            f"ECOG column={k.get('ecog_column')}, ECOG present with KPS missing={k.get('ecog_present_and_kps_missing', 0)}"
        )
        lines.append("")

    ds = report["dictionary_search"]
    lines.append("[Local dictionary/code-definition search]")
    lines.append(f"  Files scanned: {ds.get('files_scanned', 0)}")
    lines.append(f"  Matches: {ds.get('match_count', 0)}")
    lines.append(f"  Files skipped for size: {ds.get('files_skipped_size_limit', 0)}")
    lines.append("  Relevant excerpts (first 100):")
    for item in ds.get("matches", [])[:100]:
        lines.append(f"    {item['file']}:{item['line']}: {item['excerpt']}")
    lines.append("")
    lines.append("[Preliminary split arithmetic — not final eligibility]")
    split = report["model_readiness"]["preliminary_split_arithmetic_only"]
    for key, value in split.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("[Imputation recommendation]")
    lines.append("  " + report["model_readiness"]["imputation_readiness"]["recommendation"])
    lines.append("")
    lines.append("[Findings]")
    for finding in report["findings"]:
        lines.append(f"  - {finding}")
    lines.append("")
    lines.append(f"JSON report: {JSON_PATH}")
    lines.append(f"Text report: {TXT_PATH}")
    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("AUDIT COMPLETE")
    print(f"Text report: {TXT_PATH}")
    print(f"JSON report: {JSON_PATH}")
    print(f"LUAD cleaned unique patients: {luad_n}")
    print(f"LUSC cleaned unique patients: {lusc_n}")
    print(f"Local dictionary-search matches: {ds.get('match_count', 0)}")
    print("No training performed. Raw datasets were not modified.")

if __name__ == "__main__":
    main()
