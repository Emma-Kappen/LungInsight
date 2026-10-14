#!/usr/bin/env python3
"""Validate Stage 01's aligned heart-mask and heart-excluded NPY outputs."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient-dir", type=Path, required=True)
    ap.add_argument("--schema", type=Path, default=Path("meta_schema.json"))
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()
    patient = args.patient_dir.expanduser().resolve()
    schema_path = args.schema.expanduser().resolve()
    schema = read_json(schema_path)
    stage01 = schema["stage_01_hu_and_crop"]
    files = stage01["filenames"]
    paths = {
        "ct": patient / files["volume_hu_cropped.npy"],
        "lung": patient / files["lung_mask.npy"],
        "heart": patient / files["heart_mask.npy"],
        "excluded_volume": patient / files["volume_hu_heart_excluded.npy"],
        "meta": patient / files["meta.json"],
        "tracking": patient / files["heart_tracking_diagnostics.json"],
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing Stage 01 artifact(s):\n  " + "\n  ".join(missing))
    meta = read_json(paths["meta"])
    tracking = read_json(paths["tracking"])
    ct = np.load(paths["ct"], mmap_mode="r", allow_pickle=False)
    lung = np.load(paths["lung"], mmap_mode="r", allow_pickle=False)
    heart = np.load(paths["heart"], mmap_mode="r", allow_pickle=False)
    excluded = np.load(paths["excluded_volume"], mmap_mode="r", allow_pickle=False)
    if ct.ndim != 3:
        raise ValueError(f"CT must be 3-D ZYX; got {ct.shape}")
    for name, arr in (("lung_mask", lung), ("heart_mask", heart), ("volume_hu_heart_excluded", excluded)):
        if arr.shape != ct.shape:
            raise ValueError(f"{name} shape {arr.shape} != CT shape {ct.shape}")
    if heart.dtype != np.dtype(bool):
        raise ValueError(f"heart_mask.npy must be bool, got {heart.dtype}")
    if excluded.dtype != np.dtype(np.float32):
        raise ValueError(f"volume_hu_heart_excluded.npy must be float32, got {excluded.dtype}")
    if not np.array_equal(np.asarray(excluded[~heart]), np.asarray(ct[~heart], dtype=np.float32)):
        raise ValueError("volume_hu_heart_excluded.npy differs from volume_hu_cropped.npy outside heart_mask")
    replacement = float(meta.get("heart_excluded_volume_replacement_hu", schema["stage_01_hu_and_crop"]["preprocessing"]["air_replacement_hu"]))
    if heart.any() and not np.all(np.asarray(excluded[heart]) == np.float32(replacement)):
        raise ValueError("Heart-mask voxels in volume_hu_heart_excluded.npy do not equal configured replacement HU")
    overlap = int(np.count_nonzero(np.asarray(lung, dtype=bool) & np.asarray(heart, dtype=bool)))
    status = tracking.get("heart_mask_tracking", {}).get("status")
    report = {
        "status": "validated" if heart.any() and overlap == 0 and status == "tracked_central_3d_component" else "blocked",
        "schema_version": schema.get("schema_version"),
        "patient_id": meta.get("patient_id", patient.name),
        "ct_shape_zyx": list(ct.shape),
        "ct_dtype": str(ct.dtype),
        "lung_mask_dtype": str(lung.dtype),
        "heart_mask_dtype": str(heart.dtype),
        "heart_excluded_volume_dtype": str(excluded.dtype),
        "heart_mask_voxels": int(heart.sum()),
        "lung_heart_overlap_voxels": overlap,
        "heart_excluded_volume_voxels_replaced": int(heart.sum()),
        "replacement_hu": replacement,
        "tracker_status": status,
        "heart_mask_review_status": meta.get("heart_mask_review_status"),
        "heart_mask_exclusion_verified": bool(meta.get("heart_mask_exclusion_verified", False)),
        "outputs": {k: str(v) for k, v in paths.items() if k not in ("meta", "tracking")},
        "note": "Shape/dtype/value consistency is checked here. A nonempty mask still requires visual review; this is not clinical validation.",
    }
    output = args.report.expanduser().resolve() if args.report else patient / "stage01_heart_output_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "validated":
        return 2
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
