import csv
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight")
ARTIFACTS = ROOT / "backend" / "Clinical" / "artifacts"

STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ARTIFACTS / f"tcga_kps_baseline_chronology_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE_CSV = ARTIFACTS / "tcga_kps_exact_xml_audit_20261030T125004Z" / "all_kps_observations.csv"

# Recover the observation CSV if the audit folder has a different name.
if not SOURCE_CSV.exists():
    matches = list(ARTIFACTS.rglob("all_kps_observations.csv"))
    if not matches:
        raise FileNotFoundError(
            "Could not locate all_kps_observations.csv under " + str(ARTIFACTS)
        )
    SOURCE_CSV = max(matches, key=lambda p: p.stat().st_mtime)

KPS_TAGS = {"karnofsky_performance_score", "karnofsky_score"}
TIMING_TAGS = {"performance_status_scale_timing", "performance_status_timing"}

# Search the whole patient XML for chronology-related fields.
TERMS = (
    "treatment", "therapy", "adjuvant", "neoadjuvant",
    "chemotherapy", "radiation", "radiotherapy",
    "surgery", "surgical", "resection", "operative",
    "diagnosis", "diagnosed", "diagnostic",
    "days_to", "date", "year_of", "month_of",
    "follow_up", "followup", "progression",
    "recurrence", "tumor_status", "tumour_status",
    "clinical_stage", "pathologic_stage",
    "performance_status", "karnofsky"
)

def lname(tag):
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].split(":")[-1].lower()

def text_of(node):
    return " ".join(" ".join(node.itertext()).split())

def make_parent_map(root):
    result = {}
    for parent in root.iter():
        for child in list(parent):
            result[child] = parent
    return result

def path_of(node, parent_map):
    parts = []
    cur = node
    while cur is not None:
        name = lname(cur.tag)
        parent = parent_map.get(cur)
        if parent is not None:
            same = [x for x in list(parent) if lname(x.tag) == name]
            if len(same) > 1:
                name += f"[{same.index(cur) + 1}]"
        parts.append(name)
        cur = parent_map.get(cur)
    return "/" + "/".join(reversed(parts))

def get_context(node, parent_map):
    cur = parent_map.get(node)
    while cur is not None:
        if lname(cur.tag) in {"patient", "follow_up"}:
            return cur, lname(cur.tag)
        cur = parent_map.get(cur)
    return None, "unscoped"

def parse_json(value):
    try:
        return json.loads(value) if value else []
    except Exception:
        return []

def timing_values(row):
    data = parse_json(row.get("same_parent_timing_json", ""))
    return [
        str(item.get("value", "")).strip()
        for item in data
        if item.get("tag", "").lower() in TIMING_TAGS
    ]

def has_preoperative(row):
    return any(v.casefold() == "preoperative" for v in timing_values(row))

def safe_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def classify_candidate(row):
    raw = (row.get("kps_raw_value") or "").strip()
    num = safe_number(raw)
    timings = timing_values(row)
    nonempty = [x for x in timings if x]

    if num is None:
        value_class = "non_numeric_or_empty"
    elif num == 0:
        value_class = "zero_requires_validation"
    elif num < 0 or num > 100:
        value_class = "outside_expected_range"
    else:
        value_class = "nonzero_numeric"

    if any(x.casefold() == "preoperative" for x in nonempty):
        timing_class = "preoperative_same_parent"
    elif nonempty:
        timing_class = "other_same_parent_timing"
    else:
        timing_class = "no_nonempty_same_parent_timing"

    return value_class, timing_class

def write_csv(path, rows):
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

print("Reading observation-level source:", SOURCE_CSV, flush=True)

with SOURCE_CSV.open("r", newline="", encoding="utf-8-sig") as f:
    source_rows = list(csv.DictReader(f))

# Only nonempty patient-root KPS observations are relevant to the
# candidate baseline review. Preserve all of them, including zero values.
root_rows = [
    r for r in source_rows
    if r.get("record_context") == "patient"
    and (r.get("kps_raw_value") or "").strip()
]

for row in root_rows:
    row["value_class"], row["timing_class"] = classify_candidate(row)

candidate_rows = [r for r in root_rows if has_preoperative(r)]
zero_root_rows = [
    r for r in root_rows
    if (r.get("kps_raw_value") or "").strip() == "0"
]

print("Nonempty patient-root KPS observations:", len(root_rows))
print("Same-parent Preoperative candidates:", len(candidate_rows))
print("Patient-root zero observations:", len(zero_root_rows))

# Process every XML file represented by a root-level KPS observation.
# A patient may have multiple observations in a file, so use unique paths.
files = {}
for row in root_rows:
    path = Path(row["xml_file"])
    files[str(path)] = path

candidate_by_file = defaultdict(list)
for row in candidate_rows:
    candidate_by_file[row["xml_file"]].append(row)

chronology_rows = []
candidate_review = []
xml_errors = []
tag_inventory = Counter()

for index, (file_string, xml_path) in enumerate(sorted(files.items()), 1):
    if index % 25 == 0:
        print(f"Reviewing XML {index}/{len(files)}", flush=True)

    file_candidates = candidate_by_file.get(file_string, [])

    try:
        root = ET.parse(xml_path).getroot()
        parent_map = make_parent_map(root)
        nodes = list(root.iter())

        for node in nodes:
            tag_inventory[lname(node.tag)] += 1

        # Record relevant chronology fields across the entire XML file.
        # The path allows later reviewers to distinguish patient-level,
        # follow-up, treatment, and outcome sections.
        relevant_nodes = []
        for node in nodes:
            tag = lname(node.tag)
            if not any(term in tag for term in TERMS):
                continue

            value = text_of(node)
            if not value:
                continue

            relevant_nodes.append(node)

        for candidate in file_candidates:
            candidate_id = (
                candidate["cohort"] + "|" +
                candidate["patient_barcode"] + "|" +
                candidate["kps_element_path"]
            )

            candidate_review.append({
                "candidate_id": candidate_id,
                "cohort": candidate["cohort"],
                "patient_barcode": candidate["patient_barcode"],
                "xml_file": str(xml_path),
                "kps_value": candidate["kps_raw_value"],
                "kps_element_path": candidate["kps_element_path"],
                "timing_values_json": json.dumps(timing_values(candidate)),
                "timing_class": candidate["timing_class"],
                "value_class": candidate["value_class"],
                "chronology_fields_found": len(relevant_nodes),
                "chronology_review_status": "manual_interpretation_required"
            })

            for node in relevant_nodes:
                tag = lname(node.tag)
                value = text_of(node)
                context, context_type = get_context(node, parent_map)
                parent = parent_map.get(node)

                chronology_rows.append({
                    "candidate_id": candidate_id,
                    "cohort": candidate["cohort"],
                    "patient_barcode": candidate["patient_barcode"],
                    "kps_value": candidate["kps_raw_value"],
                    "kps_element_path": candidate["kps_element_path"],
                    "xml_file": str(xml_path),
                    "field_tag": tag,
                    "field_value": value[:5000],
                    "field_xml_path": path_of(node, parent_map),
                    "field_parent_tag": lname(parent.tag) if parent is not None else "",
                    "field_record_context": context_type,
                    "field_record_path": path_of(context, parent_map) if context is not None else "",
                    "same_parent_as_kps": (
                        parent is not None and
                        parent is parent_map.get(next(
                            (
                                n for n in nodes
                                if path_of(n, parent_map) == candidate["kps_element_path"]
                            ),
                            None
                        ))
                    )
                })

    except Exception as exc:
        xml_errors.append({
            "xml_file": str(xml_path),
            "error": repr(exc)
        })

# Patient-level candidate summary.
by_cohort = defaultdict(list)
for row in candidate_review:
    by_cohort[row["cohort"]].append(row)

summary = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "source_observation_csv": str(SOURCE_CSV),
    "root_nonempty_observations": len(root_rows),
    "root_zero_observations": len(zero_root_rows),
    "preoperative_candidate_observations": len(candidate_rows),
    "preoperative_candidate_patients": {
        cohort: len({r["patient_barcode"] for r in rows})
        for cohort, rows in by_cohort.items()
    },
    "preoperative_candidate_kps_distribution": {
        cohort: dict(Counter(r["kps_value"] for r in rows))
        for cohort, rows in by_cohort.items()
    },
    "xml_files_reviewed": len(files),
    "xml_errors": len(xml_errors),
    "chronology_field_rows": len(chronology_rows),
    "important_limitations": [
        "A Preoperative timing label is not by itself proof that KPS preceded all relevant treatment.",
        "Chronology fields are exported with their exact XML paths; automated keyword matches are not interpreted as treatment dates.",
        "The audit does not infer dates from narrative text.",
        "A field elsewhere in the XML is not assumed to apply to the KPS observation.",
        "Manual clinical interpretation is required before baseline eligibility is assigned.",
        "No source data, dataset builder, training pipeline, or model is modified."
    ]
}

write_csv(OUT / "patient_root_kps_observations.csv", root_rows)
write_csv(OUT / "preoperative_candidate_patients.csv", candidate_review)
write_csv(OUT / "candidate_xml_chronology_fields.csv", chronology_rows)
write_csv(OUT / "patient_root_zero_observations.csv", zero_root_rows)
write_csv(OUT / "xml_errors.csv", xml_errors)

inventory = [
    {"tag": tag, "occurrences_in_reviewed_xml": count}
    for tag, count in sorted(tag_inventory.items())
    if any(term in tag for term in TERMS)
]
write_csv(OUT / "chronology_tag_inventory.csv", inventory)

(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

lines = [
    "TCGA KPS BASELINE CHRONOLOGY REVIEW",
    f"Generated UTC: {summary['generated_utc']}",
    f"Source CSV: {SOURCE_CSV}",
    "",
    f"Nonempty patient-root observations: {len(root_rows)}",
    f"Patient-root zero observations: {len(zero_root_rows)}",
    f"Preoperative candidate observations: {len(candidate_rows)}",
    f"XML files reviewed: {len(files)}",
    f"XML errors: {len(xml_errors)}",
    f"Chronology field rows: {len(chronology_rows)}",
    "",
    "PREOPERATIVE CANDIDATES BY COHORT"
]

for cohort in sorted(by_cohort):
    rows = by_cohort[cohort]
    lines.append(
        f"{cohort}: {len(rows)} observations; "
        f"{len({r['patient_barcode'] for r in rows})} patients"
    )
    lines.append(f"  KPS distribution: {dict(Counter(r['kps_value'] for r in rows))}")

lines += ["", "LIMITATIONS"]
lines.extend("- " + x for x in summary["important_limitations"])

(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

zip_path = OUT.with_suffix(".zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for path in OUT.iterdir():
        z.write(path, arcname=path.name)

print("\nAUDIT COMPLETE")
print("Output:", OUT)
print("ZIP:", zip_path)
print("\nSummary:")
print("\n".join(lines))
