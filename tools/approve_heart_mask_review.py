#!/usr/bin/env python3
"""Record explicit human visual review of a Stage 01 central-rod heart mask.

This tool does not validate anatomy automatically. It requires an explicit CLI
attestation and checks only that the current Stage 01 files are internally
consistent and that a QC montage was generated after those masks.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-dir", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path(__file__).resolve().parents[1] / "meta_schema.json")
    parser.add_argument("--confirm-visual-review", action="store_true",
                        help="Attest that you inspected the newly generated QC montage and approve the mask for audit filtering.")
    parser.add_argument("--review-note", required=True,
                        help="Brief note describing what was checked in the montage; stored for audit provenance.")
    args = parser.parse_args()

    if not args.confirm_visual_review:
        parser.error("Refusing to approve automatically. Inspect the new QC montage, then pass --confirm-visual-review.")
    if len(args.review_note.strip()) < 8:
        parser.error("--review-note must contain at least 8 non-whitespace characters.")

    patient_dir = args.patient_dir.resolve()
    schema = read_json(args.schema.resolve())
    meta_path = patient_dir / "meta.json"
    tracking_path = patient_dir / "heart_tracking_diagnostics.json"
    ct_path = patient_dir / "volume_hu_cropped.npy"
    lung_path = patient_dir / "lung_mask.npy"
    heart_path = patient_dir / "heart_mask.npy"
    qc_montage = patient_dir / "01_mask_qc" / "mask_qc_montage.png"
    required = (meta_path, tracking_path, ct_path, lung_path, heart_path, qc_montage)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot approve heart mask; missing current artifacts:\n  " + "\n  ".join(missing))

    meta = read_json(meta_path)
    tracking = read_json(tracking_path)
    tracking_info = tracking.get("heart_mask_tracking", {})
    if str(meta.get("meta_schema_version")) != str(schema.get("schema_version")):
        raise ValueError(
            f"meta.json schema version {meta.get('meta_schema_version')!r} != current schema {schema.get('schema_version')!r}. "
            "Rerun Stage 01 with the installed schema first."
        )
    if tracking_info.get("status") != "tracked_central_3d_component" or not bool(meta.get("heart_mask_tracking_successful")):
        raise ValueError("The expanding-rod tracker did not report success; review approval is not allowed.")

    ct = np.load(ct_path, mmap_mode="r", allow_pickle=False)
    lung = np.asarray(np.load(lung_path, mmap_mode="r", allow_pickle=False), dtype=bool)
    heart = np.asarray(np.load(heart_path, mmap_mode="r", allow_pickle=False), dtype=bool)
    if ct.ndim != 3 or lung.shape != ct.shape or heart.shape != ct.shape:
        raise ValueError(f"CT/mask shape mismatch: ct={ct.shape}, lung={lung.shape}, heart={heart.shape}")
    if not heart.any():
        raise ValueError("heart_mask.npy is empty; approval is not allowed.")
    overlap = int(np.count_nonzero(lung & heart))
    if overlap:
        raise ValueError(f"lung_mask and heart_mask overlap in {overlap} voxels after Stage 01 exclusion; approval is not allowed.")
    latest_mask_mtime = max(lung_path.stat().st_mtime, heart_path.stat().st_mtime, meta_path.stat().st_mtime)
    if qc_montage.stat().st_mtime < latest_mask_mtime:
        raise ValueError("The QC montage predates the current Stage 01 masks. Regenerate diagnostics and inspect the fresh montage first.")

    reviewed_at = datetime.now(timezone.utc).isoformat()
    meta["heart_mask_exclusion_verified"] = True
    meta["heart_mask_review_status"] = "human_visual_review_approved"
    meta["heart_mask_review_attestation"] = {
        "reviewed_at_utc": reviewed_at,
        "review_note": args.review_note.strip(),
        "review_tool": "tools/approve_heart_mask_review.py",
        "attestation": "The operator explicitly confirms the current Stage 01 mask QC montage was inspected and approves it for candidate-filter audit use.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    attestation_path = patient_dir / "heart_mask_review_attestation.json"
    attestation_path.write_text(json.dumps({
        "patient_id": meta.get("patient_id", patient_dir.name),
        "meta_schema_version": schema.get("schema_version"),
        "heart_mask_voxels": int(heart.sum()),
        "lung_mask_voxels": int(lung.sum()),
        "lung_heart_overlap_voxels": overlap,
        "reviewed_at_utc": reviewed_at,
        "review_note": args.review_note.strip(),
        "qc_montage": str(qc_montage),
        "review_status": "human_visual_review_approved",
        "warning": "This is an operator attestation, not an automated or clinical validation of segmentation accuracy.",
    }, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "heart_mask_review_approved_by_operator",
        "patient_dir": str(patient_dir),
        "heart_mask_voxels": int(heart.sum()),
        "lung_heart_overlap_voxels": overlap,
        "attestation_path": str(attestation_path),
        "reviewed_at_utc": reviewed_at,
        "note": "This only records your explicit visual review; it is not clinical certification.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
