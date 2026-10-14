"""Create or verify the Stage 01 heart/lung-mask human-review attestation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from heart_mask_review import (
    create_review_attestation,
    validate_review_attestation,
)
from pipeline_schema import get_required, load_schema


SCHEMA = load_schema(__file__)
STAGE01_FILES = get_required(SCHEMA, "stage_01_hu_and_crop.filenames", expected_type=dict)
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / get_required(SCHEMA, "paths.output_root")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id", required=True, help="Patient folder name under backend/output.")
    parser.add_argument("--reviewer", help="Operator identity recorded in the audit attestation.")
    parser.add_argument(
        "--confirm-review",
        action="store_true",
        help="Attest that every Stage 01 review montage and its slice metrics were visually inspected.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verify an existing attestation without writing one.",
    )
    args = parser.parse_args()
    patient_dir = OUTPUT_ROOT / args.patient_id
    metadata_path = patient_dir / str(STAGE01_FILES["meta.json"])
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Stage 01 metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    volume = np.load(patient_dir / str(STAGE01_FILES["volume_hu_cropped.npy"]), allow_pickle=False)
    lung = np.load(patient_dir / str(STAGE01_FILES["lung_mask.npy"]), allow_pickle=False)
    heart = np.load(patient_dir / str(STAGE01_FILES["heart_mask.npy"]), allow_pickle=False)

    if args.validate_only:
        if args.confirm_review or args.reviewer:
            parser.error("--validate-only cannot be combined with --confirm-review or --reviewer.")
        result = validate_review_attestation(patient_dir, SCHEMA, metadata, volume, lung, heart)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    if not args.reviewer:
        parser.error("--reviewer is required when creating an attestation.")
    destination = create_review_attestation(
        patient_dir,
        SCHEMA,
        reviewer=args.reviewer,
        confirmed_review=args.confirm_review,
    )
    print(f"Created review attestation: {destination}")
    print("This records an operator processing decision; it is not clinical validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
