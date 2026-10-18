#!/usr/bin/env python3
"""Audit a prepared tabular CSV for target/source leakage and data quality. No imaging access."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path

FORBIDDEN = {
    "patient_id", "cohort", "source", "dataset", "study", "target_source_value",
    "target_source_field", "histology_detail", "icd_o_3_histology", "histologic_diagnosis",
}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    path = args.csv.resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    target_counts = Counter((r.get("target") or "").strip() for r in rows)
    source_counts = Counter((r.get("source") or "").strip() for r in rows)
    duplicate_ids = 0
    ids = [r.get("patient_id", "").strip() for r in rows if r.get("patient_id", "").strip()]
    duplicate_ids = len(ids) - len(set(ids))
    empty = {c: sum(1 for r in rows if not (r.get(c) or "").strip()) for c in cols}
    forbidden_present = sorted(c for c in cols if c.strip().lower() in FORBIDDEN)
    report = {
        "input": str(path), "rows": len(rows), "columns": cols,
        "target_counts": dict(target_counts), "source_counts": dict(source_counts),
        "rows_missing_patient_id": sum(1 for r in rows if not (r.get("patient_id") or "").strip()),
        "duplicate_patient_ids": duplicate_ids,
        "invalid_or_missing_targets": sum(1 for r in rows if (r.get("target") or "").strip() not in {"LUAD", "LUSC"}),
        "columns_with_missing_values": empty,
        "forbidden_predictor_names_present": forbidden_present,
        "status": "review_required" if duplicate_ids or forbidden_present or any(t not in {"LUAD", "LUSC"} for t in target_counts) else "audit_only",
        "warnings": [
            "This is an audit, not an authorization to train.",
            "Exclude patient_id and source from the predictor matrix even though they are present for auditing.",
            "Review values as well as column names for leakage, duplicates, patient overlap, and source confounding.",
        ],
    }
    out = args.output.resolve() if args.output else path.with_name(path.stem + "_audit.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Audit written: {out}")
if __name__ == "__main__":
    main()
