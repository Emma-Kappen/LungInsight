#!/usr/bin/env python3
"""Inventory local TCGA imaging files and candidate patient barcodes in paths."""
import argparse, csv, json, re
from pathlib import Path

PATIENT_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.I)

def is_volume(path):
    n = path.name.lower()
    return n.endswith((".dcm", ".dicom", ".nii", ".nii.gz", ".mha", ".mhd", ".nrrd"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path)
    a = ap.parse_args()
    root = a.datasets_root.resolve()
    out = (a.output_dir or root / "prepared" / "tcga").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cohort in ("TCGA-LUAD", "TCGA-LUSC"):
        base = root / cohort
        if not base.exists(): continue
        for path in base.rglob("*"):
            if not path.is_file() or not is_volume(path): continue
            match = PATIENT_RE.search(str(path))
            rows.append({
                "cohort": cohort, "path": str(path), "filename": path.name,
                "size_bytes": path.stat().st_size,
                "patient_barcode_candidate": match.group(1).upper() if match else "",
                "linkage_status": "candidate_only_verify_manifest" if match else "no_barcode_in_path"
            })
    csv_path = out / "tcga_local_imaging_inventory.csv"
    fields = ["cohort", "path", "filename", "size_bytes", "patient_barcode_candidate", "linkage_status"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary = {
        "datasets_root": str(root), "files_found": len(rows),
        "by_cohort": {c: sum(r["cohort"] == c for r in rows) for c in ("TCGA-LUAD", "TCGA-LUSC")},
        "barcode_candidates": sum(bool(r["patient_barcode_candidate"]) for r in rows),
        "status": "files_found_needs_verified_linkage" if rows else "no_local_imaging_files_found",
        "warning": "Path/barcode matches are candidates only; verify patient-to-series linkage before training."
    }
    json_path = out / "tcga_local_imaging_audit.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Inventory: {csv_path}\nAudit: {json_path}")

if __name__ == "__main__": main()
