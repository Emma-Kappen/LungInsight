#!/usr/bin/env python
"""
run_pipeline.py

LungInsight — full end-to-end pipeline runner.

Runs the LungInsight pipeline for a single LIDC-IDRI patient:

    01_dicom_to_hu.py
        DICOM -> native HU volume

    02_mask_and_crop.py
        Lung segmentation + crop

    03_visualize.py
        Optional QC montage

    04_detect_candidates.py
        Candidate detection (ViTDet3D, optionally + LoG diagnostic)

    05_extract_candidate_patches.py
        Candidate-centered 64^3 classifier patches

    06_classify_candidates.py
        Multi-head classifier inference

    07_visualize_gradcam.py
        Candidate-local Grad-CAM generation (all heads)

    08_full_ct_gradcam.py
        Sole-authority projection of candidate-local Grad-CAM maps back
        onto the full native Stage 02 CT volume, plus overlay/projection
        visualizations.

The pipeline uses a consistent output layout:

    output/
        LIDC-IDRI-XXXX/
            01/
            02/
            04_candidates/
            05_classifier_patches/
            06_classification/
            07_gradcam/
            08_visualization/

Compatibility notes (why this file was rewritten)
--------------------------------------------------
This runner previously called Stages 04, 05, 07, and 08 with flags those
scripts don't accept, which would fail at argparse time before the stage
ever ran:

  * Stage 04 has no `--skip-vitdet` flag -- ViTDet3D is the primary
    detector and isn't optional. Only `--skip-log` (disable the LoG
    diagnostic detector) exists. The old `--skip-vitdet` flag/CLI option
    has been removed accordingly.
  * Stage 05 does not accept `--masked-dir` or `--candidates`. Its real
    overrides are `--stage02-dir`, `--stage04-dir`, and `--output-dir`.
  * Stage 07 does not accept a `--stage07-dir` or `--output-dir` override
    (its output directory is fixed to `07_gradcam/`); it does accept
    `--max-candidates`, `--peak-slices`, and `--gradcam-layer`, which are
    now exposed here.
  * Stage 08 (rewritten to comply with the LungInsight Imaging Pipeline
    Architecture spec) writes to `08_visualization/` (not
    `08_full_ct_gradcam/`), has no GIF/HTML-viewer output, and accepts
    `--stage02-dir`, `--stage05-dir`, `--stage07-dir`, `--heads`,
    `--candidates`, `--threshold`, and `--alpha`. The old
    `--gradcam-fps` / `--save-gradcam-slices` options referred to
    outputs that no longer exist and have been removed; `--heads` /
    `--candidates` passthroughs have been added instead.

Stages 01-03 are invoked unchanged from the original runner (their
source wasn't part of this compatibility pass); only Stages 04-08 were
verified against their actual `parse_args()`.

Usage
-----

Run the complete pipeline:

    python run_pipeline.py 0141

Equivalent:

    python run_pipeline.py LIDC-IDRI-0141

Resume from Stage 7:

    python run_pipeline.py 0141 --from-stage 7

Run only Stage 8:

    python run_pipeline.py 0141 --from-stage 8 --to-stage 8

Everything except patient_id has a default.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


# =====================================================================
# DEFAULTS
# =====================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DICOM_ROOT = "Imaging/LIDC/lidc_idri"

DEFAULT_OUTPUT_ROOT = "output"

DEFAULT_CHECKPOINT = "Imaging/checkpoints/best_model_gpu_v3.pth"

DEFAULT_TARGET = "malignancy"

DEFAULT_DEVICE = "auto"

DEFAULT_GRADCAM_LAYER = "layer3"


# =====================================================================
# PATIENT ID
# =====================================================================

def normalize_patient_id(raw: str) -> str:
    """
    Accept:

        0141
        141
        LIDC-IDRI-0141

    Return:

        LIDC-IDRI-0141
    """

    raw = raw.strip()

    if raw.upper().startswith("LIDC-IDRI-"):
        digits = raw.split("-")[-1]
    else:
        digits = raw

    if not digits.isdigit():
        raise ValueError(
            f"Could not parse patient ID from '{raw}'. "
            f"Expected something like '0141' or 'LIDC-IDRI-0141'."
        )

    return f"LIDC-IDRI-{digits.zfill(4)}"


# =====================================================================
# SUBPROCESS RUNNER
# =====================================================================

def run_step(description: str, cmd: list[str], required: bool = True) -> bool:
    """
    Run one pipeline stage.

    Output is streamed directly to the terminal.

    Required stage failure:
        terminates the pipeline.

    Non-required stage failure:
        prints a warning and continues.
    """

    print()
    print("=" * 76)
    print(description)
    print("=" * 76)

    print("$ " + " ".join(str(part) for part in cmd))
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:

        if required:
            print()
            print("[FATAL] Step failed:")
            print(f"        {description}")
            print(f"        Exit code: {result.returncode}")
            sys.exit(result.returncode)

        print()
        print("[WARN] Non-critical step failed.")
        print(f"       {description}")
        print("       Continuing pipeline.")
        return False

    return True


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Run the complete LungInsight pipeline for one LIDC-IDRI patient."
    )

    # ---------------------------------------------------------------
    # Required
    # ---------------------------------------------------------------

    parser.add_argument(
        "patient_id",
        help="Patient ID, e.g. '0141'. Also accepts 'LIDC-IDRI-0141'.",
    )

    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------

    parser.add_argument(
        "--dicom-root",
        default=DEFAULT_DICOM_ROOT,
        help=f"Root directory containing patient DICOM folders. Default: {DEFAULT_DICOM_ROOT}",
    )

    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Pipeline output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Classifier checkpoint used by Stages 06 and 07. Default: {DEFAULT_CHECKPOINT}",
    )

    # ---------------------------------------------------------------
    # Model options
    # ---------------------------------------------------------------

    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["auto", "cuda", "cpu"],
        help="Compute device for stages that support explicit device selection.",
    )

    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Classifier head explained prominently by Stage 07 Grad-CAM. All heads are still generated.",
    )

    # ---------------------------------------------------------------
    # Optional stages
    # ---------------------------------------------------------------

    parser.add_argument(
        "--skip-qc",
        action="store_true",
        help="Skip Stage 03 QC visualization.",
    )

    parser.add_argument(
        "--skip-log",
        action="store_true",
        help="Stage 04: skip the LoG diagnostic detector (ViTDet3D always runs; it is not optional).",
    )

    parser.add_argument(
        "--skip-full-viz",
        action="store_true",
        help="Skip Stage 08 full-CT Grad-CAM projection.",
    )

    # ---------------------------------------------------------------
    # Stage range
    # ---------------------------------------------------------------

    parser.add_argument(
        "--from-stage",
        type=int,
        default=1,
        choices=range(1, 9),
        help="Resume from this stage. Earlier outputs must already exist. Default: 1.",
    )

    parser.add_argument(
        "--to-stage",
        type=int,
        default=8,
        choices=range(1, 9),
        help="Stop after this stage. Default: 8.",
    )

    # ---------------------------------------------------------------
    # Stage 07 options
    # ---------------------------------------------------------------

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Stage 07: maximum number of candidates to explain. Default: all.",
    )

    parser.add_argument(
        "--gradcam-layer",
        default=DEFAULT_GRADCAM_LAYER,
        choices=["layer1", "layer2", "layer3", "layer4"],
        help=f"Stage 07: backbone stage to hook for Grad-CAM. Default: {DEFAULT_GRADCAM_LAYER}.",
    )

    # ---------------------------------------------------------------
    # Stage 08 options
    # ---------------------------------------------------------------

    parser.add_argument(
        "--gradcam-threshold",
        type=float,
        default=0.4,
        help="Stage 08: per-candidate normalized Grad-CAM display threshold in [0,1]. Default: 0.4.",
    )

    parser.add_argument(
        "--gradcam-alpha",
        type=float,
        default=0.45,
        help="Stage 08: Grad-CAM overlay opacity in [0,1]. Default: 0.45.",
    )

    parser.add_argument(
        "--heads",
        nargs="*",
        default=None,
        help="Stage 08: restrict projection to these classifier heads. Default: all discovered.",
    )

    parser.add_argument(
        "--candidates",
        nargs="*",
        type=int,
        default=None,
        help="Stage 08: restrict projection to these candidate ids. Default: all discovered.",
    )

    return parser.parse_args()


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    args = parse_args()

    # ---------------------------------------------------------------
    # Validate stage range.
    # ---------------------------------------------------------------

    if args.from_stage > args.to_stage:
        raise ValueError("--from-stage cannot be greater than --to-stage.")

    patient_id = normalize_patient_id(args.patient_id)

    python = sys.executable

    def script(name: str) -> str:
        return os.path.join(SCRIPT_DIR, name)

    # ---------------------------------------------------------------
    # Consistent pipeline paths.
    # ---------------------------------------------------------------

    dicom_dir = os.path.join(args.dicom_root, patient_id)

    patient_out = os.path.join(args.output_root, patient_id)

    stage01_dir = os.path.join(patient_out, "01")
    stage02_dir = os.path.join(patient_out, "02")
    stage04_dir = os.path.join(patient_out, "04_candidates")
    stage05_dir = os.path.join(patient_out, "05_classifier_patches")
    stage06_dir = os.path.join(patient_out, "06_classification")
    stage07_dir = os.path.join(patient_out, "07_gradcam")
    stage08_dir = os.path.join(patient_out, "08_visualization")

    qc_montage_path = os.path.join(stage02_dir, "qc_montage.png")

    # ---------------------------------------------------------------
    # Pipeline header.
    # ---------------------------------------------------------------

    print("#" * 76)
    print("# LUNGINSIGHT — FULL PIPELINE")
    print("#" * 76)

    print(f"Patient ID   : {patient_id}")
    print(f"DICOM dir    : {dicom_dir}")
    print(f"Output root  : {args.output_root}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Device       : {args.device}")
    print(f"Stages       : {args.from_stage} -> {args.to_stage}")

    # ===============================================================
    # STAGE 01 — DICOM to HU volume
    # ===============================================================

    if args.from_stage <= 1 <= args.to_stage:
        run_step(
            "STAGE 01 — DICOM to HU volume",
            [
                python,
                script("01_dicom_to_hu.py"),
                dicom_dir,
                "--out-dir",
                stage01_dir,
            ],
        )

    # ===============================================================
    # STAGE 02 — Lung segmentation + crop
    # ===============================================================

    if args.from_stage <= 2 <= args.to_stage:
        run_step(
            "STAGE 02 — Lung segmentation + crop",
            [
                python,
                script("02_mask_and_crop.py"),
                stage01_dir,
                "--out-dir",
                stage02_dir,
            ],
        )

    # ===============================================================
    # STAGE 03 — QC montage (optional)
    # ===============================================================

    if not args.skip_qc and args.from_stage <= 3 <= args.to_stage:
        run_step(
            "STAGE 03 — QC montage",
            [
                python,
                script("03_visualize.py"),
                stage02_dir,
                "--save",
                qc_montage_path,
            ],
            required=False,
        )

    # ===============================================================
    # STAGE 04 — Candidate detection
    #
    # ViTDet3D always runs (there is no --skip-vitdet flag -- it's the
    # primary detector). --skip-log disables the separate LoG
    # diagnostic-only detector.
    # ===============================================================

    if args.from_stage <= 4 <= args.to_stage:

        cmd = [
            python,
            script("04_detect_candidates.py"),
            patient_id,
            "--output-root",
            args.output_root,
            "--device",
            args.device,
        ]
        """
        if args.skip_log:
            cmd.append("--skip-log")
        """
        cmd.append("--skip-log")
        run_step("STAGE 04 — Candidate detection", cmd)

    # ===============================================================
    # STAGE 05 — Extract classifier patches
    #
    # Real overrides are --stage02-dir / --stage04-dir / --output-dir
    # (not --masked-dir / --candidates).
    # ===============================================================

    if args.from_stage <= 5 <= args.to_stage:
        run_step(
            "STAGE 05 — Extract classifier patches",
            [
                python,
                script("05_extract_candidate_patches.py"),
                patient_id,
                "--output-root",
                args.output_root,
                "--stage02-dir",
                stage02_dir,
                "--stage04-dir",
                stage04_dir,
                "--output-dir",
                stage05_dir,
            ],
        )

    # ===============================================================
    # STAGE 06 — Classifier inference
    # ===============================================================

    if args.from_stage <= 6 <= args.to_stage:
        run_step(
            "STAGE 06 — Classifier inference",
            [
                python,
                script("06_classify_candidates.py"),
                patient_id,
                "--checkpoint",
                args.checkpoint,
                "--output-root",
                args.output_root,
                "--stage05-dir",
                stage05_dir,
                "--output-dir",
                stage06_dir,
                "--device",
                args.device,
            ],
        )

    # ===============================================================
    # STAGE 07 — Candidate-local Grad-CAM
    #
    # No --stage07-dir / --output-dir override exists (fixed to
    # 07_gradcam/). --max-candidates and --gradcam-layer are real.
    # ===============================================================

    if args.from_stage <= 7 <= args.to_stage:

        cmd = [
            python,
            script("07_visualize_gradcam.py"),
            patient_id,
            "--output-root",
            args.output_root,
            "--checkpoint",
            args.checkpoint,
            "--target",
            args.target,
            "--gradcam-layer",
            args.gradcam_layer,
        ]

        if args.max_candidates is not None:
            cmd += ["--max-candidates", str(args.max_candidates)]

        run_step("STAGE 07 — Candidate-local Grad-CAM", cmd)

    # ===============================================================
    # STAGE 08 — Full-CT Grad-CAM projection
    #
    # Writes to 08_visualization/ (overlays/, projections/, report.json).
    # Accepts --stage02-dir / --stage05-dir / --stage07-dir overrides,
    # --heads / --candidates filters, --threshold, --alpha. There is no
    # GIF/HTML viewer output (--gradcam-fps / --save-gradcam-slices from
    # the old runner referred to outputs the rewritten Stage 08 no
    # longer produces).
    # ===============================================================

    if not args.skip_full_viz and args.from_stage <= 8 <= args.to_stage:

        cmd = [
            python,
            script("08_full_ct_gradcam.py"),
            patient_id,
            "--output-root",
            args.output_root,
            "--stage02-dir",
            stage02_dir,
            "--stage05-dir",
            stage05_dir,
            "--stage07-dir",
            stage07_dir,
            "--threshold",
            str(max(args.gradcam_threshold, 0.0)),
            "--alpha",
            str(min(max(args.gradcam_alpha, 0.0), 1.0)),
        ]

        if args.heads:
            cmd += ["--heads", *args.heads]

        if args.candidates:
            cmd += ["--candidates", *[str(c) for c in args.candidates]]

        run_step("STAGE 08 — Full CT Grad-CAM projection", cmd)

    # ---------------------------------------------------------------
    # Final report.
    # ---------------------------------------------------------------

    print()
    print("#" * 76)
    print(f"# PIPELINE COMPLETE — {patient_id}")
    print("#" * 76)

    print("Output:")
    print(f"  {os.path.abspath(patient_out)}")

    if args.to_stage >= 8 and not args.skip_full_viz:
        print()
        print("Full CT Grad-CAM outputs:")
        print(f"  {os.path.abspath(stage08_dir)}")
        print("  overlays/")
        print("  projections/")
        print("  report.json")


if __name__ == "__main__":
    main()