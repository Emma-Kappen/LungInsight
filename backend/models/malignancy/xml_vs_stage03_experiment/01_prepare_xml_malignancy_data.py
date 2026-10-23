#!/usr/bin/env python
"""Prepare patient-disjoint XML attribute -> reader-rated malignancy datasets.

Uses the patient IDs in frozen Stage 03 train.npz / val.npz / test.npz to assign
XML reader-annotation rows to the canonical patient splits. Stage 03 predictions
are NOT used as training inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

FEATURES = ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture")
TARGET = "malignancy"


def first(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def get_rating(record: dict[str, Any], name: str) -> Any:
    value = first(record, name, name.lower())
    if value is None and isinstance(record.get("attributes"), dict):
        value = record["attributes"].get(name, record["attributes"].get(name.lower()))
    if isinstance(value, dict):
        value = value.get("rating", value.get("value", value.get("score")))
    return value


def get_patient_id(record: dict[str, Any]) -> str:
    value = first(record, "patient_id", "patientId", "PatientID", "patient")
    return str(value).strip() if value is not None else ""


def read_archive_patients(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature archive: {path}")
    with np.load(path, allow_pickle=False) as z:
        key = "patient_id" if "patient_id" in z.files else "patient_ids" if "patient_ids" in z.files else None
        if key is None:
            raise ValueError(f"{path} has neither patient_id nor patient_ids; keys={z.files}")
        values = z[key].astype(str).reshape(-1)
        return {str(x).strip() for x in values if str(x).strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path,
                    help="Flattened nodule_manifest JSON (a JSON list of records)")
    ap.add_argument("--feature-dir", required=True, type=Path,
                    help="Directory containing train.npz, val.npz, and test.npz")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--min-rating", type=float, default=1.0)
    ap.add_argument("--max-rating", type=float, default=5.0)
    args = ap.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Manifest must be a JSON list of flattened records.")

    split_patients = {
        split: read_archive_patients(args.feature_dir / f"{split}.npz")
        for split in ("train", "val", "test")
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_patients[a] & split_patients[b]
        if overlap:
            raise ValueError(
                f"Patient leakage in feature archives between {a} and {b}: {sorted(overlap)[:10]}"
            )

    rows = {name: [] for name in split_patients}
    counters = {
        "manifest_records": len(payload),
        "missing_patient_id": 0,
        "patient_not_in_feature_splits": 0,
        "missing_or_invalid_feature_or_target": 0,
        "duplicate_rows_skipped": 0,
        "accepted": 0,
    }
    seen = set()
    for record in payload:
        if not isinstance(record, dict):
            counters["missing_or_invalid_feature_or_target"] += 1
            continue
        pid = get_patient_id(record)
        if not pid:
            counters["missing_patient_id"] += 1
            continue
        split = next((s for s, pids in split_patients.items() if pid in pids), None)
        if split is None:
            counters["patient_not_in_feature_splits"] += 1
            continue

        values = {}
        good = True
        for name in (*FEATURES, TARGET):
            try:
                x = float(get_rating(record, name))
            except (TypeError, ValueError):
                good = False
                break
            if not np.isfinite(x) or x < args.min_rating or x > args.max_rating:
                good = False
                break
            values[name] = x
        if not good:
            counters["missing_or_invalid_feature_or_target"] += 1
            continue

        study = str(first(record, "study_instance_uid", "studyInstanceUID", "study_uid") or "")
        reader = str(first(record, "xml_radiologist_id", "servicing_radiologist_id", "radiologist_id", "reader_id") or "")
        nodule = str(first(record, "nodule_id", "noduleId") or "")
        key = (pid, study, reader, nodule)
        if key in seen:
            counters["duplicate_rows_skipped"] += 1
            continue
        seen.add(key)

        rows[split].append({
            "patient_id": pid,
            "study_instance_uid": study,
            "radiologist_id": reader,
            "nodule_id": nodule,
            **values,
        })
        counters["accepted"] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["patient_id", "study_instance_uid", "radiologist_id", "nodule_id", *FEATURES, TARGET]
    for split, split_rows in rows.items():
        with (args.output_dir / f"{split}_xml.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(split_rows)
        counters[f"{split}_rows"] = len(split_rows)
        counters[f"{split}_patients_with_rows"] = len({r["patient_id"] for r in split_rows})

    counters["features_used"] = list(FEATURES)
    counters["target"] = TARGET
    counters["split_source"] = "patient_id arrays in frozen Stage 03 train.npz / val.npz / test.npz"
    counters["feature_archive_patient_counts"] = {s: len(p) for s, p in split_patients.items()}
    counters["note"] = (
        "One row per reader/nodule annotation. Attribute inputs and malignancy target are from the same XML-derived "
        "manifest record. Stage 03 predictions are reserved for the separate held-out evaluation."
    )
    report_path = args.output_dir / "prepare_report.json"
    report_path.write_text(json.dumps(counters, indent=2), encoding="utf-8")
    print(json.dumps(counters, indent=2))
    print(f"\nWrote CSVs and report to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
