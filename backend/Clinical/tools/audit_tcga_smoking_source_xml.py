from pathlib import Path
import csv, json, re, zipfile, hashlib
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

ROOT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight")
DATA = ROOT / "datasets"
AUDIT = ROOT / "backend" / "Clinical" / "artifacts" / "tcga_smoking_consistency_20261030T143357Z"
OUTROOT = ROOT / "backend" / "Clinical" / "artifacts"

PATIENT_CSV = AUDIT / "patient_level_smoking_audit.csv"
PACK_CSV = AUDIT / "pack_year_review_flags.csv"
CHRON_CSV = AUDIT / "smoking_chronology_flags.csv"

SENTINELS = {
    "", "[Not Available]", "[Not Evaluated]", "[Unknown]",
    "[Discrepancy]", "[Not Applicable]", "Not Available",
    "Not Evaluated", "Unknown", "Discrepancy", "NA", "N/A"
}

FIELDS = [
    "bcr_patient_barcode",
    "tobacco_smoking_history_indicator",
    "tobacco_smoking_age_started",
    "tobacco_smoking_year_started",
    "tobacco_smoking_year_stopped",
    "tobacco_smoking_pack_years_smoked",
    "initial_pathologic_dx_year",
    "age_at_initial_pathologic_diagnosis",
    "ecog_score",
    "karnofsky_score",
    "performance_status_timing",
    "history_neoadjuvant_treatment",
    "race",
    "gender",
]

def clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s in SENTINELS else s

def norm_tag(tag):
    return tag.split("}")[-1].split(":")[-1]

def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def barcode_from_row(row):
    for key in ("bcr_patient_barcode", "patient_barcode", "barcode"):
        value = clean(row.get(key))
        if re.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", value):
            return value
    return ""

def find_barcode(row):
    for value in row.values():
        value = clean(value)
        m = re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", value)
        if m:
            return m.group(0)
    return ""

patients = read_csv(PATIENT_CSV)
pack_flags = read_csv(PACK_CSV)
chron_flags = read_csv(CHRON_CSV)

patient_by_barcode = {}
for row in patients:
    b = barcode_from_row(row)
    if b:
        patient_by_barcode[b] = row

flag_reasons = {}

def add_flag(barcode, reason):
    if barcode:
        flag_reasons.setdefault(barcode, set()).add(reason)

# Chronology flags: retain all patient identifiers and let the output
# distinguish post-diagnosis stop years from other chronology conditions.
for row in chron_flags:
    b = find_barcode(row)
    if b:
        add_flag(b, "smoking_chronology_flag")

# Pack-year review flags: retain all flagged patients, then classify from
# patient-level values rather than relying on potentially variable flag labels.
for row in pack_flags:
    b = find_barcode(row)
    if b:
        add_flag(b, "pack_year_review_flag")

for b, row in patient_by_barcode.items():
    status = clean(row.get("smoking_status_label", "")).lower()
    stop = clean(row.get("tobacco_smoking_year_stopped_numeric", ""))
    dx = clean(row.get("initial_pathologic_dx_year_numeric", ""))
    pack = clean(row.get("pack_years_numeric", ""))
    try:
        if status == "current_smoker" and stop:
            add_flag(b, "current_smoker_with_stop_year")
        if status in ("discrepancy", "unknown", "missing") and pack:
            add_flag(b, "discrepant_status_with_pack_years")
        if pack and abs(float(pack) - 0.15) < 1e-8:
            add_flag(b, "0.15_pack_years")
        if pack and float(pack) > 100:
            add_flag(b, "pack_years_over_100")
        if stop and dx and float(stop) > float(dx):
            add_flag(b, "stop_year_after_diagnosis")
    except ValueError:
        pass

targets = sorted(flag_reasons)
if not targets:
    raise SystemExit("No flagged patient barcodes could be extracted. Check the CSV headers and contents.")

# Discover only clinical XMLs under the two target cohorts.
xml_files = []
for cohort in ("TCGA-LUAD", "TCGA-LUSC"):
    base = DATA / cohort
    if base.exists():
        xml_files.extend(
            p for p in base.rglob("*.xml")
            if "clinical" in p.name.lower()
        )

target_set = set(targets)
matched = {b: [] for b in targets}
parse_errors = []
scanned = 0

def element_record(elem, parent_tag="", grandparent_tag=""):
    return {
        "tag": norm_tag(elem.tag),
        "text": (elem.text or "").strip(),
        "attributes": dict(elem.attrib),
        "parent_tag": parent_tag,
        "grandparent_tag": grandparent_tag,
    }

for path in xml_files:
    scanned += 1
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as e:
        parse_errors.append({"path": str(path), "error": str(e)})
        continue

    text_blob = " ".join(
        (e.text or "").strip()
        for e in root.iter()
        if e.text and e.text.strip()
    )
    found = target_set.intersection(
        b for b in targets if b in text_blob or b in path.name
    )
    if not found:
        continue

    for barcode in found:
        # Capture the patient element containing the exact barcode, when possible.
        patient_nodes = []
        for elem in root.iter():
            if barcode in (elem.text or ""):
                node = elem
                # Walk the tree by searching for the smallest enclosing node
                # that contains this barcode and relevant fields.
                patient_nodes.append(node)

        all_records = []
        for parent in root.iter():
            children = list(parent)
            if not children:
                continue
            parent_text = " ".join((x.text or "").strip() for x in parent.iter())
            if barcode not in parent_text:
                continue
            for child in parent.iter():
                tag = norm_tag(child.tag)
                if tag in FIELDS or any(
                    key in tag.lower()
                    for key in ("smoking", "performance_status", "karnofsky", "ecog", "pathologic_dx_year", "neoadjuvant")
                ):
                    par_tag = norm_tag(parent.tag)
                    all_records.append({
                        "tag": tag,
                        "text": (child.text or "").strip(),
                        "attributes": dict(child.attrib),
                        "context_parent": par_tag,
                    })

        # Deduplicate exact field records while retaining parent context.
        seen = set()
        deduped = []
        for rec in all_records:
            key = (rec["tag"], rec["text"], rec["context_parent"])
            if key not in seen:
                seen.add(key)
                deduped.append(rec)

        matched[barcode].append({
            "cohort": "TCGA-LUAD" if "TCGA-LUAD" in str(path) else "TCGA-LUSC",
            "source_xml": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "flag_reasons": sorted(flag_reasons.get(barcode, [])),
            "patient_level_csv_row": patient_by_barcode.get(barcode, {}),
            "xml_field_records": deduped,
        })

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
outdir = OUTROOT / f"tcga_smoking_source_xml_audit_{stamp}"
outdir.mkdir(parents=True, exist_ok=False)

records = []
for barcode in targets:
    entries = matched.get(barcode, [])
    if not entries:
        records.append({
            "barcode": barcode,
            "flag_reasons": sorted(flag_reasons[barcode]),
            "xml_match_status": "NOT_FOUND",
            "matches": [],
        })
    else:
        records.append({
            "barcode": barcode,
            "flag_reasons": sorted(flag_reasons[barcode]),
            "xml_match_status": "MATCHED",
            "matches": entries,
        })

summary = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "target_patient_count": len(targets),
    "clinical_xml_files_scanned": scanned,
    "matched_patient_count": sum(bool(matched[b]) for b in targets),
    "unmatched_patient_count": sum(not matched[b] for b in targets),
    "xml_parse_error_count": len(parse_errors),
    "flag_reason_counts": {
        reason: sum(reason in reasons for reasons in flag_reasons.values())
        for reason in sorted({r for rs in flag_reasons.values() for r in rs})
    },
    "limitations": [
        "XML source fields are extracted with parent context but do not establish measurement date unless present in the XML.",
        "Multiple XML files may match a patient; all matches are retained.",
        "The script does not alter source datasets or model artifacts.",
        "Flag lists are derived from the supplied audit CSVs and patient-level values."
    ]
}

(outdir / "source_xml_records.json").write_text(
    json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
)
(outdir / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
(outdir / "parse_errors.json").write_text(
    json.dumps(parse_errors, indent=2), encoding="utf-8"
)

with (outdir / "patient_match_summary.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["barcode", "flag_reasons", "xml_match_status", "xml_match_count"])
    for rec in records:
        w.writerow([
            rec["barcode"], ";".join(rec["flag_reasons"]),
            rec["xml_match_status"], len(rec["matches"])
        ])

zip_path = outdir.with_suffix(".zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in outdir.iterdir():
        if p.is_file():
            z.write(p, arcname=p.name)

print(json.dumps(summary, indent=2))
print(f"\nOUTPUT_DIRECTORY={outdir}")
print(f"OUTPUT_ZIP={zip_path}")
