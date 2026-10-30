import csv
import json
import re
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


def local_tag(tag):
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def norm(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def value_of(element):
    return " ".join((element.text or "").split())


def barcode_of(path):
    match = BARCODE_RE.search(str(path).upper())
    return match.group(0).upper() if match else ""


def path_for(element, parent_map):
    parts = []
    node = element
    while node is not None:
        tag = local_tag(node.tag)
        # Include identifying attributes when available.
        attrs = []
        for key, val in sorted(node.attrib.items()):
            attrs.append(f"{local_tag(key)}={val}")
        segment = tag + (("[" + ",".join(attrs) + "]") if attrs else "")
        parts.append(segment)
        node = parent_map.get(node)
    return "/".join(reversed(parts))


def ancestor_nodes(element, parent_map, limit=5):
    result = []
    node = parent_map.get(element)
    while node is not None and len(result) < limit:
        result.append(node)
        node = parent_map.get(node)
    return result


def sibling_timing(element, parent_map):
    # Inspect the KPS element's parent and a few ancestor levels for timing
    # elements. This is evidence only; no timing precedence is inferred.
    timing = []
    node = element
    for _ in range(4):
        if node is None:
            break
        for descendant in node.iter():
            tag = local_tag(descendant.tag)
            if "timing" in norm(tag) or "performance_status_scale" in norm(tag):
                val = value_of(descendant)
                if val:
                    timing.append({
                        "tag": tag,
                        "value": val,
                        "path": path_for(descendant, parent_map),
                    })
        node = parent_map.get(node)

    seen = set()
    unique = []
    for item in timing:
        key = (item["tag"], item["value"], item["path"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kps_rows = []
    related_tag_rows = []
    file_rows = []
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "raw_datasets_modified": False,
        "model_training_performed": False,
        "cohorts": {},
    }

    for cohort, base in COHORTS.items():
        if not base.is_dir():
            raise FileNotFoundError(f"Missing directory: {base}")

        xml_files = sorted(p for p in base.rglob("*.xml") if p.is_file())
        clinical_files = [p for p in xml_files if "clinical" in p.name.lower()]

        tag_occurrences = Counter()
        tag_nonempty = Counter()
        tag_value_examples = defaultdict(list)
        parse_errors = []
        files_with_kps_value = set()
        files_with_ecog_like_tag = set()
        kps_occurrences = 0
        kps_nonempty_occurrences = 0

        for path in xml_files:
            barcode = barcode_of(path)
            try:
                root = ET.parse(path).getroot()
            except Exception as exc:
                parse_errors.append({
                    "file": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue

            nodes = list(root.iter())
            parent_map = {
                child: parent
                for parent in nodes
                for child in list(parent)
            }

            is_clinical_named = "clinical" in path.name.lower()
            file_kps_count = 0
            file_kps_values = []
            file_ecog_tags = set()

            for element in nodes:
                tag = local_tag(element.tag)
                normalized = norm(tag)
                value = value_of(element)

                # Inventory all tag names containing relevant terms, across
                # all XML types, without treating every hit as a clinical field.
                if any(term in normalized for term in (
                    "karnofsky", "ecog", "easterncooperativeoncology",
                    "performance", "functionalstatus"
                )):
                    tag_occurrences[tag] += 1
                    if value:
                        tag_nonempty[tag] += 1
                        if len(tag_value_examples[tag]) < 12:
                            tag_value_examples[tag].append({
                                "file": path.name,
                                "barcode": barcode,
                                "value": value[:160],
                            })

                if "ecog" in normalized or "easterncooperativeoncology" in normalized:
                    file_ecog_tags.add(tag)

                if normalized == "karnofskyperformancescore":
                    kps_occurrences += 1
                    file_kps_count += 1
                    if value:
                        kps_nonempty_occurrences += 1
                        file_kps_values.append(value)
                        files_with_kps_value.add(str(path))

                    if is_clinical_named:
                        ancestors = ancestor_nodes(element, parent_map, limit=4)
                        sibling_values = []
                        if ancestors:
                            parent = ancestors[0]
                            for child in list(parent):
                                child_tag = local_tag(child.tag)
                                child_value = value_of(child)
                                if child_value:
                                    sibling_values.append(f"{child_tag}={child_value}")

                        timing = sibling_timing(element, parent_map)

                        kps_rows.append({
                            "cohort": cohort,
                            "barcode_from_path": barcode,
                            "filename": path.name,
                            "full_path": str(path),
                            "kps_value_raw": value,
                            "kps_is_nonempty": bool(value),
                            "kps_element_path": path_for(element, parent_map),
                            "ancestor_context": " || ".join(
                                f"{local_tag(a.tag)}: {value_of(a)[:200]}"
                                for a in ancestors
                            ),
                            "same_parent_nonempty_children": " || ".join(sibling_values),
                            "timing_evidence_json": json.dumps(timing, ensure_ascii=False),
                        })

            if file_ecog_tags:
                files_with_ecog_like_tag.add(str(path))

            file_rows.append({
                "cohort": cohort,
                "filename": path.name,
                "full_path": str(path),
                "barcode_from_path": barcode,
                "filename_contains_clinical": is_clinical_named,
                "kps_element_count": file_kps_count,
                "kps_nonempty_values": " | ".join(file_kps_values),
                "ecog_like_tag_names": " | ".join(sorted(file_ecog_tags)),
                "xml_parse_error": "",
            })

        for tag, occurrences in sorted(tag_occurrences.items()):
            related_tag_rows.append({
                "cohort": cohort,
                "tag": tag,
                "occurrences_all_xml": occurrences,
                "nonempty_occurrences_all_xml": tag_nonempty[tag],
                "examples_json": json.dumps(
                    tag_value_examples[tag], ensure_ascii=False
                ),
            })

        report["cohorts"][cohort] = {
            "xml_files_all_types": len(xml_files),
            "clinical_named_xml_files": len(clinical_files),
            "kps_element_occurrences_all_xml": kps_occurrences,
            "kps_nonempty_element_occurrences_all_xml": kps_nonempty_occurrences,
            "files_with_at_least_one_nonempty_kps": len(files_with_kps_value),
            "files_with_ecog_like_tag": len(files_with_ecog_like_tag),
            "related_tag_occurrences": dict(sorted(tag_occurrences.items())),
            "related_tag_nonempty_occurrences": dict(sorted(tag_nonempty.items())),
            "parse_error_count": len(parse_errors),
            "parse_error_examples": parse_errors[:20],
        }

    kps_csv = OUT / f"tcga_kps_xml_structure_{stamp}.csv"
    tags_csv = OUT / f"tcga_performance_related_tag_inventory_{stamp}.csv"
    files_csv = OUT / f"tcga_performance_xml_file_inventory_{stamp}.csv"
    json_path = OUT / f"tcga_performance_xml_structure_{stamp}.json"
    txt_path = OUT / f"tcga_performance_xml_structure_{stamp}.txt"

    def write_csv(path, rows):
        fields = list(rows[0].keys()) if rows else []
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(kps_csv, kps_rows)
    write_csv(tags_csv, related_tag_rows)
    write_csv(files_csv, file_rows)

    report["outputs"] = {
        "kps_element_context_csv": str(kps_csv),
        "tag_inventory_csv": str(tags_csv),
        "xml_file_inventory_csv": str(files_csv),
        "json": str(json_path),
        "txt": str(txt_path),
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "TCGA PERFORMANCE-STATUS XML STRUCTURE AUDIT",
        f"Generated UTC: {report['generated_utc']}",
        "READ ONLY; no training; raw datasets unchanged",
        "KPS timing is recorded as evidence only; no source is selected automatically.",
        "",
    ]

    for cohort, data in report["cohorts"].items():
        lines += [
            f"=== {cohort} ===",
            f"XML files, all types: {data['xml_files_all_types']}",
            f"Clinical-named XML files: {data['clinical_named_xml_files']}",
            f"KPS element occurrences, all XML: {data['kps_element_occurrences_all_xml']}",
            f"Nonempty KPS element occurrences: {data['kps_nonempty_element_occurrences_all_xml']}",
            f"Files with nonempty KPS: {data['files_with_at_least_one_nonempty_kps']}",
            f"Files with ECOG-like tags: {data['files_with_ecog_like_tag']}",
            f"Parse errors: {data['parse_error_count']}",
            "Related tag names and nonempty counts:",
        ]
        for tag, count in data["related_tag_occurrences"].items():
            lines.append(
                f"  {tag}: {count} occurrences; "
                f"{data['related_tag_nonempty_occurrences'].get(tag, 0)} nonempty"
            )
        lines.append("")

    lines += [
        "OUTPUTS",
        f"KPS element context CSV: {kps_csv}",
        f"Related-tag inventory CSV: {tags_csv}",
        f"XML file inventory CSV: {files_csv}",
        f"JSON: {json_path}",
        f"TXT: {txt_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
