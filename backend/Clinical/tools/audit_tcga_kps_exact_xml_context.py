import csv
import json
import re
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight")
DATASETS = ROOT / "datasets"
ARTIFACTS = ROOT / "backend" / "Clinical" / "artifacts"

STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ARTIFACTS / f"tcga_kps_exact_xml_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

KPS_TAGS = {
    "karnofsky_performance_score",
    "karnofsky_score",
}
TIMING_TAGS = {
    "performance_status_scale_timing",
    "performance_status_timing",
}
CONTEXT_TAGS = {"patient", "follow_up"}

TREATMENT_TERMS = (
    "therapy", "treatment", "radiation", "chemotherapy",
    "surgery", "surgical", "neoadjuvant", "adjuvant",
    "progression", "tumor_event", "days_to"
)

def local_name(tag):
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].split(":")[-1].lower()

def node_text(node):
    return " ".join(" ".join(node.itertext()).split())

def node_path(node, parent_map):
    parts = []
    current = node
    while current is not None:
        name = local_name(current.tag)
        parent = parent_map.get(current)
        if parent is not None:
            same = [c for c in list(parent) if local_name(c.tag) == name]
            if len(same) > 1:
                index = same.index(current) + 1
                name = f"{name}[{index}]"
        parts.append(name)
        current = parent
    return "/" + "/".join(reversed(parts))

def ancestors(node, parent_map):
    result = []
    current = parent_map.get(node)
    while current is not None:
        result.append(current)
        current = parent_map.get(current)
    return result

def nearest_context(node, parent_map):
    chain = ancestors(node, parent_map)
    for wanted in ("follow_up", "patient"):
        for ancestor in chain:
            if local_name(ancestor.tag) == wanted:
                return ancestor, wanted
    return None, "unscoped"

def under_nested_follow_up(node, context, parent_map):
    current = parent_map.get(node)
    while current is not None and current is not context:
        if local_name(current.tag) == "follow_up":
            return True
        current = parent_map.get(current)
    return False

def sequence_number(context, parent_map):
    if context is None or local_name(context.tag) != "follow_up":
        return ""
    parent = parent_map.get(context)
    if parent is None:
        return ""
    siblings = [x for x in list(parent) if local_name(x.tag) == "follow_up"]
    try:
        return str(siblings.index(context) + 1)
    except ValueError:
        return ""

def barcode_from_file(path):
    match = re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", path.name.upper())
    if match:
        return match.group(0)
    match = re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", str(path).upper())
    return match.group(0) if match else ""

def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def value_class(raw):
    raw = (raw or "").strip()
    if not raw:
        return "empty"
    try:
        number = float(raw)
    except ValueError:
        return "non_numeric"
    if number == 0:
        return "zero_requires_validation"
    if number < 0 or number > 100:
        return "outside_expected_0_100_range"
    if number % 10 != 0:
        return "numeric_nonstandard_increment"
    return "numeric_0_100_nonzero"

def main():
    cohorts = {
        "LUAD": DATASETS / "TCGA-LUAD",
        "LUSC": DATASETS / "TCGA-LUSC",
    }

    observations = []
    errors = []
    xml_files = []
    tag_inventory = Counter()
    patient_file_counts = Counter()

    for cohort, folder in cohorts.items():
        if not folder.exists():
            errors.append({
                "cohort": cohort,
                "file": str(folder),
                "error": "Cohort directory missing"
            })
            continue

        files = sorted(folder.rglob("*.xml"))
        if not files:
            files = sorted(folder.rglob("*.XML"))

        xml_files.extend((cohort, f) for f in files)
        print(f"{cohort}: discovered {len(files)} XML files", flush=True)

    for file_index, (cohort, xml_file) in enumerate(xml_files, 1):
        if file_index % 100 == 0:
            print(f"Parsing {file_index}/{len(xml_files)} XML files...", flush=True)

        barcode = barcode_from_file(xml_file)

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            nodes = list(root.iter())
            parent_map = {}
            for parent in nodes:
                for child in list(parent):
                    parent_map[child] = parent

            for node in nodes:
                tag_inventory[local_name(node.tag)] += 1

            patient_file_counts[(cohort, barcode)] += 1

            kps_nodes = [
                node for node in nodes
                if local_name(node.tag) in KPS_TAGS
            ]

            for kps_index, kps in enumerate(kps_nodes, 1):
                context, context_type = nearest_context(kps, parent_map)
                parent = parent_map.get(kps)
                raw = node_text(kps)
                kps_path = node_path(kps, parent_map)

                direct_sibling_timing = []
                record_timing = []
                treatment_context = []

                # Timing tags in the exact immediate parent.
                if parent is not None:
                    for sibling in list(parent):
                        if local_name(sibling.tag) in TIMING_TAGS:
                            direct_sibling_timing.append({
                                "tag": local_name(sibling.tag),
                                "value": node_text(sibling),
                                "path": node_path(sibling, parent_map)
                            })

                # Timing and treatment metadata within the nearest
                # patient/follow_up record only.
                if context is not None:
                    for candidate in context.iter():
                        if candidate is context:
                            continue

                        if context_type == "patient" and under_nested_follow_up(
                            candidate, context, parent_map
                        ):
                            continue

                        tag = local_name(candidate.tag)
                        value = node_text(candidate)

                        if tag in TIMING_TAGS:
                            record_timing.append({
                                "tag": tag,
                                "value": value,
                                "path": node_path(candidate, parent_map),
                                "same_parent_as_kps": parent_map.get(candidate) is parent
                            })

                        if any(term in tag for term in TREATMENT_TERMS):
                            if value:
                                treatment_context.append({
                                    "tag": tag,
                                    "value": value,
                                    "path": node_path(candidate, parent_map)
                                })

                # Preserve immediate-parent structure.
                sibling_values = []
                if parent is not None:
                    for sibling in list(parent):
                        if sibling is kps:
                            continue
                        value = node_text(sibling)
                        if value:
                            sibling_values.append({
                                "tag": local_name(sibling.tag),
                                "value": value[:1000],
                                "path": node_path(sibling, parent_map)
                            })

                direct_values = {
                    item["value"] for item in direct_sibling_timing if item["value"]
                }
                record_values = {
                    item["value"] for item in record_timing if item["value"]
                }

                if direct_values:
                    timing_class = "same_parent_timing"
                elif record_values:
                    timing_class = "same_record_timing_only"
                else:
                    timing_class = "timing_unlinked"

                try:
                    numeric = float(raw) if raw else None
                except ValueError:
                    numeric = None

                observations.append({
                    "cohort": cohort,
                    "patient_barcode": barcode,
                    "xml_filename": xml_file.name,
                    "xml_file": str(xml_file),
                    "kps_occurrence_in_file": kps_index,
                    "kps_xml_tag": local_name(kps.tag),
                    "kps_raw_value": raw,
                    "kps_numeric_value": numeric if numeric is not None else "",
                    "kps_value_class": value_class(raw),
                    "record_context": context_type,
                    "follow_up_sequence": sequence_number(context, parent_map),
                    "kps_element_path": kps_path,
                    "immediate_parent_tag": local_name(parent.tag) if parent is not None else "",
                    "immediate_parent_path": node_path(parent, parent_map) if parent is not None else "",
                    "timing_association_class": timing_class,
                    "same_parent_timing_json": json.dumps(direct_sibling_timing, ensure_ascii=False),
                    "same_record_timing_json": json.dumps(record_timing, ensure_ascii=False),
                    "immediate_sibling_values_json": json.dumps(sibling_values, ensure_ascii=False),
                    "treatment_context_json": json.dumps(treatment_context, ensure_ascii=False),
                })

        except Exception as exc:
            errors.append({
                "cohort": cohort,
                "patient_barcode": barcode,
                "file": str(xml_file),
                "error": repr(exc),
                "traceback": traceback.format_exc()
            })

    # Patient-level aggregation, keeping observation-level evidence intact.
    by_patient = defaultdict(list)
    for row in observations:
        by_patient[(row["cohort"], row["patient_barcode"])].append(row)

    patient_rows = []
    for (cohort, barcode), rows in sorted(by_patient.items()):
        values = [r["kps_numeric_value"] for r in rows if r["kps_numeric_value"] != ""]
        root_rows = [r for r in rows if r["record_context"] == "patient"]
        follow_rows = [r for r in rows if r["record_context"] == "follow_up"]
        patient_rows.append({
            "cohort": cohort,
            "patient_barcode": barcode,
            "kps_observation_count": len(rows),
            "distinct_kps_values": "|".join(sorted(set(str(v) for v in values))),
            "root_observation_count": len(root_rows),
            "follow_up_observation_count": len(follow_rows),
            "zero_observation_count": sum(r["kps_value_class"] == "zero_requires_validation" for r in rows),
            "nonzero_numeric_observation_count": sum(
                isinstance(r["kps_numeric_value"], (int, float)) and r["kps_numeric_value"] != 0
                for r in rows
            ),
            "same_parent_timing_count": sum(r["timing_association_class"] == "same_parent_timing" for r in rows),
            "same_record_timing_only_count": sum(r["timing_association_class"] == "same_record_timing_only" for r in rows),
            "timing_unlinked_count": sum(r["timing_association_class"] == "timing_unlinked" for r in rows),
            "xml_file": rows[0]["xml_file"]
        })

    zero_rows = [r for r in observations if r["kps_value_class"] == "zero_requires_validation"]
    baseline_candidates = [r for r in observations if r["record_context"] == "patient"]
    followup_rows = [r for r in observations if r["record_context"] == "follow_up"]

    # Search all XML tags for ECOG/performance-status-related elements.
    status_inventory = [
        {"tag": tag, "occurrences": count}
        for tag, count in sorted(tag_inventory.items())
        if "ecog" in tag or "performance_status" in tag or "karnofsky" in tag
    ]

    write_csv(OUT / "all_kps_observations.csv", observations)
    write_csv(OUT / "patient_level_reconciliation.csv", patient_rows)
    write_csv(OUT / "zero_value_observations.csv", zero_rows)
    write_csv(OUT / "patient_root_candidates.csv", baseline_candidates)
    write_csv(OUT / "follow_up_observations.csv", followup_rows)
    write_csv(OUT / "xml_errors.csv", errors)
    write_csv(OUT / "performance_status_tag_inventory.csv", status_inventory)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "cohorts": {k: str(v) for k, v in cohorts.items()},
            "xml_files_discovered": {
                cohort: sum(1 for c, _ in xml_files if c == cohort)
                for cohort in cohorts
            },
            "xml_parse_errors": len(errors)
        },
        "observations": {
            "total": len(observations),
            "by_cohort": dict(Counter(r["cohort"] for r in observations)),
            "by_context": dict(Counter(r["record_context"] for r in observations)),
            "by_cohort_context": {
                f"{cohort}_{context}": sum(
                    r["cohort"] == cohort and r["record_context"] == context
                    for r in observations
                )
                for cohort in cohorts for context in ("patient", "follow_up", "unscoped")
            },
            "by_value_class": dict(Counter(r["kps_value_class"] for r in observations)),
            "by_cohort_value_class": {
                f"{cohort}_{value_class_name}": sum(
                    r["cohort"] == cohort and r["kps_value_class"] == value_class_name
                    for r in observations
                )
                for cohort in cohorts
                for value_class_name in sorted(set(r["kps_value_class"] for r in observations))
            },
            "timing_association": dict(Counter(r["timing_association_class"] for r in observations)),
            "timing_association_by_cohort": {
                f"{cohort}_{association}": sum(
                    r["cohort"] == cohort and r["timing_association_class"] == association
                    for r in observations
                )
                for cohort in cohorts
                for association in ("same_parent_timing", "same_record_timing_only", "timing_unlinked")
            }
        },
        "patients_with_kps": {
            cohort: sum(1 for r in patient_rows if r["cohort"] == cohort)
            for cohort in cohorts
        },
        "zero_value_patients": {
            cohort: len({
                r["patient_barcode"] for r in zero_rows if r["cohort"] == cohort
            })
            for cohort in cohorts
        },
        "interpretation_notes": [
            "Same-parent timing means the timing element is a sibling of the KPS element.",
            "Same-record timing only means timing exists within the nearest patient/follow_up record, but not in the KPS immediate parent.",
            "Timing found elsewhere in the same record is not proof that it applies to the KPS observation.",
            "Patient-root context excludes nested follow_up descendants when collecting timing and treatment metadata.",
            "Root-level observations are candidates only; they are not automatically confirmed pretreatment baseline.",
            "KPS zero is preserved as zero and is not converted to missing.",
            "This audit does not modify datasets, dataset builders, training code, or models."
        ]
    }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    lines = []
    lines.append("TCGA KPS EXACT XML CONTEXT AUDIT")
    lines.append(f"Generated UTC: {summary['generated_utc']}")
    lines.append("")
    lines.append("XML FILE COUNTS")
    for cohort in cohorts:
        lines.append(f"{cohort}: {summary['scope']['xml_files_discovered'].get(cohort, 0)}")
    lines.append(f"XML parse errors: {len(errors)}")
    lines.append("")
    lines.append("OBSERVATION COUNTS")
    lines.append(f"Total KPS observations: {len(observations)}")
    for cohort in cohorts:
        cohort_rows = [r for r in observations if r["cohort"] == cohort]
        lines.append(f"\n{cohort}: {len(cohort_rows)} observations")
        lines.append(f"  Patients with KPS: {len({r['patient_barcode'] for r in cohort_rows})}")
        lines.append(f"  Patient-root observations: {sum(r['record_context']=='patient' for r in cohort_rows)}")
        lines.append(f"  Follow-up observations: {sum(r['record_context']=='follow_up' for r in cohort_rows)}")
        lines.append(f"  Zero observations: {sum(r['kps_value_class']=='zero_requires_validation' for r in cohort_rows)}")
        lines.append(f"  Patients with at least one zero: {len({r['patient_barcode'] for r in cohort_rows if r['kps_value_class']=='zero_requires_validation'})}")
        lines.append("  KPS value distribution:")
        for value, count in sorted(Counter(r["kps_raw_value"] for r in cohort_rows).items()):
            lines.append(f"    {value!r}: {count}")
        lines.append("  Timing association:")
        for association, count in Counter(r["timing_association_class"] for r in cohort_rows).items():
            lines.append(f"    {association}: {count}")

    lines.append("\nPERFORMANCE STATUS / ECOG TAG INVENTORY")
    for item in status_inventory:
        lines.append(f"  {item['tag']}: {item['occurrences']}")

    lines.append("\nINTERPRETATION")
    lines.extend(f"- {note}" for note in summary["interpretation_notes"])

    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nAUDIT COMPLETE")
    print(f"Output directory: {OUT}")
    print(f"Observations: {len(observations)}")
    print(f"Patients with KPS: {len(patient_rows)}")
    print(f"Zero observations: {len(zero_rows)}")
    print(f"XML errors: {len(errors)}")
    print("\nFiles:")
    for path in sorted(OUT.iterdir()):
        print(f"  {path.name} ({path.stat().st_size:,} bytes)")

    # ZIP the full output folder.
    import zipfile
    zip_path = OUT.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in OUT.iterdir():
            archive.write(path, arcname=path.name)
    print(f"\nZIP: {zip_path}")

if __name__ == "__main__":
    main()
