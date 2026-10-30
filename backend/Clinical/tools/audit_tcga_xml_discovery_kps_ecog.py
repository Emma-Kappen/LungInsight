import csv
import re
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight")
OUT = ROOT / "backend" / "Clinical" / "artifacts"

COHORTS = {
    "LUAD": ROOT / "datasets" / "TCGA-LUAD",
    "LUSC": ROOT / "datasets" / "TCGA-LUSC",
}
BARCODE_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.I)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def barcode_for(path):
    # Search full path so XMLs in differently named directories are included.
    match = BARCODE_RE.search(str(path).upper())
    return match.group(0).upper() if match else ""


def xml_tag_inventory(path):
    found = []
    try:
        root = ET.parse(path).getroot()
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            tag = el.tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]
            tag_norm = norm(tag)
            if not any(term in tag_norm for term in
                       ("karnofsky", "ecog", "easterncooperativeoncology",
                        "performance", "functionalstatus")):
                continue

            # Leaf elements only, to avoid duplicating nested text.
            if list(el):
                continue
            value = " ".join((el.text or "").split())
            found.append({
                "tag": tag,
                "normalized_tag": tag_norm,
                "value": value[:500],
                "has_value": bool(value),
            })
        return found, ""
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_files = []
    tag_rows = []
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "raw_data_modified": False,
        "cohorts": {},
        "limitations": [
            "Barcode is inferred from the file path.",
            "Tag-name matching is exploratory and does not establish semantic validity.",
            "XML content is inspected but never modified.",
        ],
    }

    for cohort, base in COHORTS.items():
        if not base.is_dir():
            raise FileNotFoundError(f"Missing cohort directory: {base}")

        # Deliberately include every XML file, not just filenames containing 'clinical'.
        files = sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() == ".xml")
        clinical_named = [p for p in files if "clinical" in p.name.lower()]
        grouped = defaultdict(list)
        all_tag_counts = Counter()
        relevant_tag_counts = Counter()
        tag_examples = defaultdict(list)
        parse_errors = []

        for path in files:
            barcode = barcode_for(path)
            if barcode:
                grouped[barcode].append(path)

            tags, error = xml_tag_inventory(path)
            if error:
                parse_errors.append({"file": str(path), "error": error})

            for item in tags:
                all_tag_counts[item["tag"]] += 1
                tag_examples[item["tag"]].append({
                    "file": path.name,
                    "barcode": barcode,
                    "value": item["value"][:120],
                })
                if item["has_value"]:
                    relevant_tag_counts[item["tag"]] += 1

            all_files.append({
                "cohort": cohort,
                "filename": path.name,
                "full_path": str(path),
                "parent_directory": str(path.parent),
                "barcode_from_path": barcode,
                "filename_contains_clinical": "clinical" in path.name.lower(),
                "has_kps_ecog_related_tag": bool(tags),
                "kps_ecog_related_tags": "; ".join(sorted(set(x["tag"] for x in tags))),
                "kps_ecog_values": " || ".join(
                    f'{x["tag"]}={x["value"]}' for x in tags if x["has_value"]
                ),
                "xml_parse_error": error,
            })

        report["cohorts"][cohort] = {
            "directory": str(base),
            "total_xml_files": len(files),
            "filenames_containing_clinical": len(clinical_named),
            "xml_files_excluded_by_previous_clinical_filename_filter":
                len(files) - len(clinical_named),
            "files_with_barcode_in_path": sum(bool(barcode_for(p)) for p in files),
            "unique_path_barcodes": len(grouped),
            "participants_with_multiple_xml_files": sum(len(v) > 1 for v in grouped.values()),
            "maximum_xml_files_for_one_barcode": max((len(v) for v in grouped.values()), default=0),
            "xml_files_per_barcode_distribution": dict(sorted(
                Counter(len(v) for v in grouped.values()).items()
            )),
            "related_tag_occurrences": dict(sorted(all_tag_counts.items())),
            "related_tags_with_nonempty_values": dict(sorted(relevant_tag_counts.items())),
            "tag_examples": {
                tag: examples[:8]
                for tag, examples in sorted(tag_examples.items())
            },
            "parse_error_count": len(parse_errors),
            "parse_error_examples": parse_errors[:20],
            "sample_xml_not_named_clinical": [
                str(p) for p in files
                if "clinical" not in p.name.lower()
            ][:20],
        }

    csv_path = OUT / f"tcga_xml_discovery_files_{stamp}.csv"
    tag_csv_path = OUT / f"tcga_xml_discovery_tag_inventory_{stamp}.csv"
    json_path = OUT / f"tcga_xml_discovery_kps_ecog_{stamp}.json"
    txt_path = OUT / f"tcga_xml_discovery_kps_ecog_{stamp}.txt"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_files[0].keys()) if all_files else [])
        writer.writeheader()
        writer.writerows(all_files)

    with tag_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["cohort", "tag", "occurrences", "nonempty_values", "example_files_and_values"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cohort in COHORTS:
            result = report["cohorts"][cohort]
            for tag, count in result["related_tag_occurrences"].items():
                examples = result["tag_examples"].get(tag, [])
                writer.writerow({
                    "cohort": cohort,
                    "tag": tag,
                    "occurrences": count,
                    "nonempty_values": result["related_tags_with_nonempty_values"].get(tag, 0),
                    "example_files_and_values": json.dumps(examples[:8], ensure_ascii=False),
                })

    report["outputs"] = {
        "all_xml_files_csv": str(csv_path),
        "tag_inventory_csv": str(tag_csv_path),
        "json": str(json_path),
        "txt": str(txt_path),
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "TCGA XML DISCOVERY AND KPS/ECOG TAG AUDIT",
        f"Generated UTC: {report['generated_utc']}",
        "READ ONLY; raw datasets unchanged",
        "All XML files included regardless of filename.",
        "",
    ]
    for cohort, result in report["cohorts"].items():
        lines += [
            f"=== {cohort} ===",
            f"Total XML files: {result['total_xml_files']}",
            f"Filenames containing 'clinical': {result['filenames_containing_clinical']}",
            f"Files excluded by previous filename filter: {result['xml_files_excluded_by_previous_clinical_filename_filter']}",
            f"Unique barcodes in paths: {result['unique_path_barcodes']}",
            f"Participants with multiple XML files: {result['participants_with_multiple_xml_files']}",
            f"Maximum files for one barcode: {result['maximum_xml_files_for_one_barcode']}",
            f"Files per barcode distribution: {json.dumps(result['xml_files_per_barcode_distribution'], sort_keys=True)}",
            f"Related tag occurrences: {json.dumps(result['related_tag_occurrences'], sort_keys=True)}",
            f"Related tags with values: {json.dumps(result['related_tags_with_nonempty_values'], sort_keys=True)}",
            f"XML parse errors: {result['parse_error_count']}",
            "Sample non-clinical-named XML files:",
            *[f"  {p}" for p in result["sample_xml_not_named_clinical"]],
            "Tag examples:",
        ]
        for tag, examples in result["tag_examples"].items():
            lines.append(f"  {tag}: {json.dumps(examples[:3], ensure_ascii=False)}")
        lines.append("")

    lines += [
        "OUTPUTS",
        f"All XML files CSV: {csv_path}",
        f"Tag inventory CSV: {tag_csv_path}",
        f"JSON: {json_path}",
        f"TXT: {txt_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
