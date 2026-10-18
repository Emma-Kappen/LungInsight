#!/usr/bin/env python3
"""Flatten TCGA GDC metadata.repository JSON files into a CSV inventory."""
import argparse, csv, json
from pathlib import Path
from collections import Counter

FIELDS = ["file_name", "file_id", "submitter_id", "data_category", "data_type",
          "data_format", "access", "state", "file_size"]

def records_from_json(path):
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(obj, list): return obj
    if isinstance(obj, dict):
        for key in ("data", "records", "hits"):
            if isinstance(obj.get(key), list): return obj[key]
        return [obj]
    raise ValueError(f"Unsupported JSON root: {type(obj).__name__}")

def collect_entities(obj):
    found = []
    if isinstance(obj, list):
        for item in obj: found.extend(collect_entities(item))
    elif isinstance(obj, dict):
        wanted = ("entity_type", "entity_submitter_id", "entity_id", "case_id",
                  "case_submitter_id", "submitter_id", "sample_id", "sample_submitter_id")
        item = {k: obj[k] for k in wanted if obj.get(k) not in (None, "")}
        if item: found.append(item)
        for value in obj.values():
            if isinstance(value, (list, dict)): found.extend(collect_entities(value))
    return found

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path)
    a = ap.parse_args()
    root = a.datasets_root.resolve()
    out = (a.output_dir or root / "prepared" / "tcga").resolve()
    out.mkdir(parents=True, exist_ok=True)
    inventory, summary = [], {}
    for cohort in ("TCGA-LUAD", "TCGA-LUSC"):
        matches = list((root / cohort).glob("metadata.repository*.json"))
        if not matches:
            summary[cohort] = {"status": "missing_metadata_json", "record_count": 0}
            continue
        source = max(matches, key=lambda p: p.stat().st_mtime)
        records = records_from_json(source)
        counts = {k: Counter() for k in ("data_category", "data_type", "data_format", "access", "state")}
        imaging_names = []
        for rec in records:
            row = {k: rec.get(k, "") for k in FIELDS}
            row.update(cohort=cohort, metadata_source=str(source),
                       associated_entities_json=json.dumps(collect_entities(rec.get("associated_entities", []))))
            inventory.append(row)
            for key in counts: counts[key][str(rec.get(key) or "<missing>")] += 1
            searchable = " ".join(str(rec.get(k, "")) for k in ("file_name", "data_type", "data_category")).lower()
            if any(t in searchable for t in ("radiology", "imaging", "ct scan", "dicom", "nifti", "image")):
                imaging_names.append(rec.get("file_name", ""))
        summary[cohort] = {
            "status": "ok", "source_file": str(source), "record_count": len(records),
            "counts": {k: dict(v.most_common()) for k, v in counts.items()},
            "possible_imaging_filename_or_type_matches": imaging_names
        }
    csv_path = out / "tcga_gdc_file_inventory.csv"
    columns = FIELDS + ["cohort", "metadata_source", "associated_entities_json"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns); w.writeheader(); w.writerows(inventory)
    json_path = out / "tcga_gdc_metadata_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Inventory: {csv_path} ({len(inventory)} records)")
    print(f"Summary: {json_path}")
    for cohort, data in summary.items():
        print(f"{cohort}: {data['status']}, {data.get('record_count', 0)} records")

if __name__ == "__main__": main()
