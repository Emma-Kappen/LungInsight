"""
run_pipeline.py

End-to-end orchestrator for the LungInsight pipeline. Takes a single
LIDC-IDRI PatientID and runs every stage in order, wiring each
script's --out-dir into the next script's input directory, so you
don't have to hand-copy paths between steps.

    00_audit_longitudinal.py    (optional, once per dataset, not per-patient -- skipped by default)
    01_dicom_to_hu.py           -> DICOM -> HU volume
    02_mask_and_crop.py         -> lung segmentation + non-lung blanking + Z-crop
    03_visualize.py             -> save a QA montage PNG (headless, no GUI blocking)
    04_detect_and_patch.py      -> characteristic-based candidate detection + fixed-size patches
    05_shape_filter_and_grow.py -> reject tubular candidates, grow + crop nodules
    06_run_inference_xai.py     -> run trained classifier + Grad-CAM
    07_visualize_gradcam.py     -> save Grad-CAM montage PNG

Step 4 now has a full CLI (masked_dir positional, --out-dir, writes
candidates.json as {"params": ..., "candidates": [...]} with
voxel_z/voxel_y/voxel_x, sigma_mm, diameter_mm per candidate -- see
04's own module docstring for the full interface contract with 05)
so the pipeline runs straight through 01 -> 07 given a checkpoint.

Usage:
    python run_pipeline.py LIDC-IDRI-0141 \
        --lidc-root "Imaging/LIDC/lidc_idri" \
        --out-root output \
        --checkpoint path/to/model.pth

    # Stop after a specific step (useful while iterating):
    python run_pipeline.py LIDC-IDRI-0141 --stop-after 02

    # Skip steps already run for this patient:
    python run_pipeline.py LIDC-IDRI-0141 --start-from 05 --checkpoint model.pth
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

STEP_ORDER = ["01", "02", "03", "04", "05", "06", "07"]

SCRIPT_NAMES = {
    "01": "01_dicom_to_hu.py",
    "02": "02_mask_and_crop.py",
    "03": "03_visualize.py",
    "04": "04_detect_and_patch.py",
    "05": "05_shape_filter_and_grow.py",
    "06": "06_run_inference_xai.py",
    "07": "07_visualize_gradcam.py",
}


def run_step(script_path: Path, args: list, description: str):
    cmd = [sys.executable, str(script_path)] + [str(a) for a in args]
    print(f"\n{'=' * 70}")
    print(f"[pipeline] {description}")
    print(f"[pipeline] $ {' '.join(cmd)}")
    print("=" * 70)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[pipeline] FAILED at: {description}", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full LungInsight pipeline end-to-end for one patient."
    )
    parser.add_argument("patient_id", help="LIDC-IDRI PatientID, e.g. LIDC-IDRI-0141")
    parser.add_argument(
        "--lidc-root", default="Imaging/LIDC/lidc_idri",
        help="Root directory containing per-patient DICOM folders "
        "(default: 'Imaging/LIDC/lidc_idri').",
    )
    parser.add_argument(
        "--out-root", default="output",
        help="Root directory under which all per-patient outputs are "
        "written, one subfolder per patient (default: 'output').",
    )
    parser.add_argument(
        "--scripts-dir", default=None,
        help="Directory containing the numbered pipeline scripts "
        "(default: same directory as this file).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to the trained model checkpoint (.pth), required to "
        "reach step 06.",
    )
    parser.add_argument(
        "--start-from", choices=STEP_ORDER, default="01",
        help="Skip earlier steps and start here (their outputs must "
        "already exist for this patient). Default: '01'.",
    )
    parser.add_argument(
        "--stop-after", choices=STEP_ORDER, default="07",
        help="Stop after this step number (default: '07', the full "
        "pipeline).",
    )
    parser.add_argument(
        "--patch-shape", type=int, default=64,
        help="Cubic patch side length in voxels, passed through to both "
        "step 04 (candidate patches) and step 05 (grown nodule patches) "
        "so they always agree with each other and with the classifier "
        "checkpoint used in step 06 (default: 64).",
    )
    args = parser.parse_args()

    scripts_dir = Path(args.scripts_dir) if args.scripts_dir else Path(__file__).resolve().parent
    patient_dicom_dir = os.path.join(args.lidc_root, args.patient_id)
    patient_out_root = os.path.join(args.out_root, args.patient_id)

    # Per-step output directories, kept flat and predictable rather than
    # relying on each script's own "<input>_suffix" default, so re-runs
    # are easy to locate and clean up.
    dir_01 = os.path.join(patient_out_root, "01_hu")
    dir_02 = os.path.join(patient_out_root, "02_masked")
    dir_03_png = os.path.join(patient_out_root, "03_qa_montage.png")
    dir_04 = os.path.join(patient_out_root, "04_candidates")
    dir_05 = os.path.join(patient_out_root, "05_nodules")
    dir_06 = os.path.join(patient_out_root, "06_xai")
    dir_07_png = os.path.join(patient_out_root, "07_gradcam_montage.png")

    steps_to_run = STEP_ORDER[STEP_ORDER.index(args.start_from):STEP_ORDER.index(args.stop_after) + 1]
    print(f"[pipeline] Patient: {args.patient_id}")
    print(f"[pipeline] Steps to run: {', '.join(steps_to_run)}")
    print(f"[pipeline] Outputs under: {patient_out_root}")

    if "01" in steps_to_run:
        run_step(
            scripts_dir / SCRIPT_NAMES["01"],
            [patient_dicom_dir, "--out-dir", dir_01],
            "Step 1/7: DICOM -> HU volume",
        )

    if "02" in steps_to_run:
        run_step(
            scripts_dir / SCRIPT_NAMES["02"],
            [dir_01, "--out-dir", dir_02],
            "Step 2/7: Lung segmentation + Z-crop",
        )

    if "03" in steps_to_run:
        run_step(
            scripts_dir / SCRIPT_NAMES["03"],
            [dir_02, "--save", dir_03_png],
            "Step 3/7: QA visualization montage",
        )

    if "04" in steps_to_run:
        run_step(
            scripts_dir / SCRIPT_NAMES["04"],
            [dir_02, "--out-dir", dir_04, "--patch-shape", str(args.patch_shape)],
            "Step 4/7: Candidate detection",
        )

    if "05" in steps_to_run:
        run_step(
            scripts_dir / SCRIPT_NAMES["05"],
            [dir_02, dir_04, "--out-dir", dir_05, "--patch-shape", str(args.patch_shape)],
            "Step 5/7: Shape filter + region growing",
        )

    if "06" in steps_to_run:
        if not args.checkpoint:
            print(
                "[pipeline] STOPPING: step 06 requires --checkpoint "
                "path/to/model.pth",
                file=sys.stderr,
            )
            sys.exit(1)
        run_step(
            scripts_dir / SCRIPT_NAMES["06"],
            [dir_02, dir_04, dir_05, "--checkpoint", args.checkpoint, "--out-dir", dir_06],
            "Step 6/7: Classifier inference + Grad-CAM",
        )

    if "07" in steps_to_run:
        run_step(
            scripts_dir / SCRIPT_NAMES["07"],
            [dir_02, dir_06, "--nodules-dir", dir_05, "--save", dir_07_png],
            "Step 7/7: Grad-CAM visualization montage",
        )

    print(f"\n[pipeline] Done. All outputs under: {patient_out_root}")


if __name__ == "__main__":
    main()