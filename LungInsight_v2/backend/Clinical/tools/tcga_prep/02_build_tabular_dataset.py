#!/usr/bin/env python3
"""Build a tabular-only clinical dataset from TCGA clinical TSVs and an optional normalized MSK CSV.

No imaging files are read. This script deliberately excludes identifiers, cohort/source
identity, target-defining fields, ICD-O codes, detailed histology, and free-text fields.
The MSK CSV must be normalized to one row per patient with columns:
patient_id, target (LUAD or LUSC), and shared tabular feature columns.
"""
from __future__ import annotations
import argparse, csv, json, re
from collections import Counter
from pathlib import Path

COHORTS = {
    "TCGA-LUAD": ("nationwidechildrens.org_clinical_patient_luad.txt", "LUAD"),
    "TCGA-LUSC": ("nationwidechildrens.org_clinical_patient_lusc.txt", "LUSC"),
}
LEAK_EXACT = {
    "patient_id", "bcr_patient_barcode", "cohort", "source", "dataset", "study",
    "target", "label", "histologic_diagnosis", "histology_detail",
    "icd_o_3_histology", "target_source_field", "target_source_value",
}
LEAK_PATTERNS = [
    r"(^|_)barcode($|_)", r"(^|_)case_id($|_)", r"(^|_)patient_id($|_)",
    r"histolog", r"icd.?o", r"morpholog", r"diagnos", r"tumor.?type",
    r"cancer.?type", r"cohort", r"dataset", r"source", r"study.?id",
    r"^cde_id", r"^file_?id", r"^sample_?id", r"^submitter_?id",
]
def safe_feature(name: str) -> bool:
    key = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")
    if not key or key in LEAK_EXACT:
        return False
    return not any(re.search(p, key) for p in LEAK_PATTERNS)

def read_patient_tsv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        raw = next(reader, [])
        counts, headers = Counter(), []
        for h in raw:
            h = h.strip(); counts[h] += 1
            headers.append(h if counts[h] == 1 else f"{h}__duplicate{counts[h]}")
        for values in reader:
            if not values or not any(v.strip() for v in values): continue
            if len(values) > len(headers) and any(v.strip() for v in values[len(headers):]):
                raise ValueError(f"{path}: extra non-empty columns in row")
            values = (values + [""] * max(0, len(headers)-len(values)))[:len(headers)]
            first = values[0].strip().lower() if values else ""
            if first == "diagnosis" or first.startswith("cde_id:") or first == "cde_id":
                continue
            yield dict(zip(headers, values)), headers

def clean_value(v):
    v = (v or "").strip()
    if not v or v.lower() in {"na", "n/a", "not reported", "not available", "[not available]", "[not applicable]", "unknown", "null"}:
        return ""
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--msk-csv", type=Path, help="Optional normalized MSK-CHORD patient-level CSV")
    ap.add_argument("--min-feature-coverage", type=float, default=0.05,
                    help="Keep candidate features observed in at least this fraction of rows (default 0.05)")
    args = ap.parse_args()
    root = args.datasets_root.resolve()
    out = (args.output_dir or root / "prepared" / "tcga").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, qc, all_feature_names = [], {"tcga": {}, "msk": {"provided": False}}, set()

    for cohort, (filename, target) in COHORTS.items():
        matches = sorted((root / cohort).rglob(filename)) if (root / cohort).exists() else []
        if not matches:
            qc["tcga"][cohort] = {"status": "missing_input"}
            continue
        path = matches[0]
        count = 0
        for record, headers in read_patient_tsv(path):
            patient = clean_value(record.get("bcr_patient_barcode", ""))
            if not patient: continue
            # Only retain fields that pass a conservative leakage/name filter.
            feats = {}
            for h in headers:
                if safe_feature(h):
                    value = clean_value(record.get(h, ""))
                    if value:
                        feats[h] = value
                        all_feature_names.add(h)
            rows.append({"patient_id": patient, "target": target, "source": cohort, **feats})
            count += 1
        qc["tcga"][cohort] = {"status": "ok", "rows_read": count, "clinical_file": str(path)}

    if args.msk_csv:
        p = args.msk_csv.resolve()
        if not p.exists(): raise FileNotFoundError(f"MSK CSV not found: {p}")
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            for record in csv.DictReader(f):
                target = clean_value(record.get("target", "")).upper()
                patient = clean_value(record.get("patient_id", ""))
                if target not in {"LUAD", "LUSC"}:
                    qc["msk"].setdefault("rows_skipped_invalid_target", 0)
                    qc["msk"]["rows_skipped_invalid_target"] += 1
                    continue
                feats = {}
                for h, v in record.items():
                    if safe_feature(h):
                        value = clean_value(v)
                        if value:
                            feats[h] = value; all_feature_names.add(h)
                rows.append({"patient_id": patient, "target": target, "source": "MSK-CHORD", **feats})
                qc["msk"]["provided"] = True
                qc["msk"]["input_csv"] = str(p)
                qc["msk"]["rows_added"] = qc["msk"].get("rows_added", 0) + 1
    else:
        qc["msk"]["note"] = "No MSK CSV supplied. This output is TCGA-only and must not be used as the final three-source training set."

    # Determine feature coverage and keep only minimally observed candidates.
    feature_stats = {}
    n = max(1, len(rows))
    candidate_names = sorted(all_feature_names)
    kept = []
    for name in candidate_names:
        present = sum(1 for r in rows if clean_value(r.get(name, "")))
        coverage = present / n
        feature_stats[name] = {"non_missing": present, "coverage": round(coverage, 4),
                               "safe_by_name_only": True,
                               "warning": "Requires semantic review; a safe-looking field name does not guarantee absence of leakage."}
        if coverage >= args.min_feature_coverage:
            kept.append(name)

    output_columns = ["patient_id", "target", "source"] + kept
    dataset_path = out / "tcga_msk_tabular_candidates.csv"
    with dataset_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=output_columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in output_columns})
    schema = {
        "schema_version": "1.0.0",
        "dataset_path": str(dataset_path),
        "row_count": len(rows),
        "target_counts": {t: sum(1 for r in rows if r["target"] == t) for t in ("LUAD", "LUSC")},
        "source_counts": {s: sum(1 for r in rows if r["source"] == s) for s in sorted({r["source"] for r in rows})},
        "feature_columns": kept,
        "excluded_by_name_or_policy": sorted({h for h in candidate_names if h not in kept}),
        "feature_coverage": feature_stats,
        "warnings": [
            "Candidate dataset only: feature names are filtered conservatively but still require domain review.",
            "Do not train directly on this output until shared features are semantically reviewed and encoded.",
            "TCGA-LUAD and TCGA-LUSC source membership is perfectly associated with the label in this construction; source is retained for auditing only and must never be a predictor.",
            "MSK-CHORD must be supplied as a normalized patient-level CSV to include it. This script does not guess its schema.",
            "Upstream-derived inference features (e.g. nodule_count) must also be present in training data with compatible definitions, or the model must explicitly support missingness without fabricated values.",
        ],
    }
    (out / "tcga_msk_tabular_schema_qc.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Candidate table: {dataset_path}")
    print(f"Schema/QC: {out / 'tcga_msk_tabular_schema_qc.json'}")
    print(f"Rows: {len(rows)}; candidate features retained: {len(kept)}")
    print("No CT, DICOM, NIfTI, or other raw imaging data was accessed.")

if __name__ == "__main__":
    main()
