#!/usr/bin/env python3
"""Create verified patient-level LUAD/LUSC labels from TCGA legacy clinical TSVs.

Legacy TCGA patient files commonly contain two metadata rows after the header:
a row beginning with 'diagnosis' and a row containing CDE_ID values. These are
not patient records and are explicitly skipped.
"""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path

COHORTS = {
    "TCGA-LUAD": ("nationwidechildrens.org_clinical_patient_luad.txt", "LUAD"),
    "TCGA-LUSC": ("nationwidechildrens.org_clinical_patient_lusc.txt", "LUSC"),
}
ID_FIELDS = {"bcr_patient_barcode"}
TARGET_FIELDS = {"histologic_diagnosis"}

def read_tsv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        original = next(reader, [])
        counts, headers = Counter(), []
        for raw in original:
            name = raw.strip()
            counts[name] += 1
            headers.append(name if counts[name] == 1 else f"{name}__duplicate{counts[name]}")
        rows, skipped_metadata = [], 0
        for values in reader:
            if not values or not any(v.strip() for v in values):
                continue
            # Pad short rows; do not silently truncate non-empty extra values.
            if len(values) > len(headers) and any(v.strip() for v in values[len(headers):]):
                raise ValueError(f"{path}: row has {len(values)} cells but header has {len(headers)}")
            values = (values + [""] * max(0, len(headers) - len(values)))[:len(headers)]
            first = values[0].strip().lower() if values else ""
            # GDC/legacy clinical TSV metadata rows are not patient records.
            if first == "diagnosis" or first.startswith("cde_id:") or first == "cde_id":
                skipped_metadata += 1
                continue
            rows.append(dict(zip(headers, values)))
    return headers, rows, skipped_metadata

def norm_label(value: str) -> str:
    v = (value or "").strip().lower()
    if not v:
        return ""
    if "adenocarcinoma" in v or v == "luad":
        return "LUAD"
    if "squamous" in v or v == "lusc":
        return "LUSC"
    return "UNEXPECTED"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    root = args.datasets_root.resolve()
    out = (args.output_dir or root / "prepared" / "tcga").resolve()
    out.mkdir(parents=True, exist_ok=True)
    labels, qc = [], {"input_files": {}, "cohorts": {}, "warnings": []}

    for cohort, (filename, expected) in COHORTS.items():
        matches = sorted((root / cohort).rglob(filename)) if (root / cohort).exists() else []
        if not matches:
            qc["cohorts"][cohort] = {"status": "missing_input", "source_rows": 0}
            qc["warnings"].append(f"{cohort}: missing {filename}")
            continue
        path = matches[0]
        if len(matches) > 1:
            qc["warnings"].append(f"{cohort}: multiple matching files; used {path}")
        headers, rows, skipped_metadata = read_tsv(path)
        id_key = next((h for h in headers if h.lower() in ID_FIELDS), None)
        broad_key = next((h for h in headers if h.lower() in TARGET_FIELDS), None)
        detail_key = next((h for h in headers if h.lower().startswith("histologic_diagnosis__duplicate")), None)
        icd_key = next((h for h in headers if "icd" in h.lower() and ("histolog" in h.lower() or "morph" in h.lower())), None)
        seen, n_missing_id, n_missing_label, n_duplicates, n_unexpected = set(), 0, 0, 0, 0
        for row in rows:
            patient = (row.get(id_key, "") if id_key else "").strip()
            raw = (row.get(broad_key, "") if broad_key else "").strip()
            if not patient:
                n_missing_id += 1
                continue
            if patient in seen:
                n_duplicates += 1
                continue
            seen.add(patient)
            normalized = norm_label(raw)
            if not raw:
                n_missing_label += 1
                target = ""
            elif normalized == expected:
                target = expected
            else:
                target = "UNEXPECTED"
                n_unexpected += 1
                qc["warnings"].append(f"{cohort}: unexpected broad label {raw!r} for {patient}")
            labels.append({
                "patient_id": patient,
                "cohort": cohort,
                "target": target,
                "target_source_field": broad_key or "",
                "target_source_value": raw,
                "histology_detail": row.get(detail_key, "").strip() if detail_key else "",
                "icd_o_3_histology": row.get(icd_key, "").strip() if icd_key else "",
            })
        qc["input_files"][cohort] = str(path)
        qc["cohorts"][cohort] = {
            "source_rows_after_metadata_skip": len(rows),
            "metadata_rows_skipped": skipped_metadata,
            "unique_patients_written": len(seen),
            "missing_patient_ids": n_missing_id,
            "duplicate_rows_skipped": n_duplicates,
            "missing_broad_labels": n_missing_label,
            "unexpected_labels": n_unexpected,
            "status": "ok" if seen and n_unexpected == 0 and n_missing_label == 0 else "review_required",
        }

    csv_path = out / "tcga_patient_labels.csv"
    columns = ["patient_id", "cohort", "target", "target_source_field", "target_source_value",
               "histology_detail", "icd_o_3_histology"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns); w.writeheader(); w.writerows(labels)
    qc_path = out / "tcga_label_qc.json"
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(f"Labels: {csv_path}\nQC: {qc_path}")
    for cohort, stats in qc["cohorts"].items():
        print(f"{cohort}: {stats}")
    print("Only rows with target LUAD/LUSC are eligible for later processing.")
    print("Never use patient_id, cohort, target_source_value, histology_detail, or ICD-O-3 as predictors.")

if __name__ == "__main__":
    main()
