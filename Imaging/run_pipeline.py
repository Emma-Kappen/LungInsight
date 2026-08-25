"""
run_pipeline.py

End-to-end orchestrator for the LungInsight pipeline.

Pipeline:

    01_dicom_to_hu.py
        DICOM -> HU volume

    02_mask_and_crop.py
        lung segmentation + non-lung blanking + Z-crop

    03_visualize.py
        QA montage

    04_detect_candidates.py
        ViTDet3D candidate detection

    05_extract_candidate_patches.py
        fixed-size 64^3 classifier patches

    06_xai.py
        multi-head SE-ResNet3D classification +
        Grad-CAM++ / saliency

    07_visualize.py
        full-volume heatmap visualization

IMPORTANT:
    Step 04 and Step 06 use DIFFERENT checkpoints.

    --detector-checkpoint:
        ViTDet3D detector checkpoint used by Step 04.

    --classifier-checkpoint:
        SE-ResNet3D multi-head classifier checkpoint used by Step 06.

The detector checkpoint is NOT interchangeable with the classifier
checkpoint.

Typical usage:

    python run_pipeline.py LIDC-IDRI-0141 ^
        --lidc-root "Imaging/LIDC/lidc_idri" ^
        --out-root output ^
        --detector-checkpoint "Imaging/pytorch_model.bin" ^
        --classifier-checkpoint "Imaging/checkpoints/best_model_gpu_v2.pth"

PowerShell:

    python Imaging/run_pipeline.py LIDC-IDRI-0141 `
        --lidc-root "Imaging/LIDC/lidc_idri" `
        --out-root output `
        --detector-checkpoint "Imaging/pytorch_model.bin" `
        --classifier-checkpoint "Imaging/checkpoints/best_model_gpu_v2.pth"

Stop after a particular stage:

    --stop-after 04
    --stop-after 05
    --stop-after 06

Start from an already completed stage:

    --start-from 04
    --start-from 05
    --start-from 06
    --start-from 07
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ============================================================================
# PIPELINE ORDER
# ============================================================================

STEP_ORDER = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
]


SCRIPT_NAMES = {
    "01": "01_dicom_to_hu.py",
    "02": "02_mask_and_crop.py",
    "03": "03_visualize.py",
    "04": "04_detect_candidates.py",
    "05": "05_extract_candidate_patches.py",
    "06": "06_xai.py",
    "07": "07_visualize.py",
}


# ============================================================================
# HELPERS
# ============================================================================

def run_step(
    script_path: Path,
    args: list,
    description: str,
):
    """
    Execute one pipeline stage.

    The current Python interpreter is used so the pipeline runs inside
    the same virtual environment from which run_pipeline.py was launched.
    """

    if not script_path.exists():
        print(
            f"\n[pipeline] FAILED: script not found:\n"
            f"           {script_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [
        sys.executable,
        str(script_path),
    ] + [str(a) for a in args]

    print()
    print("=" * 70)
    print(f"[pipeline] {description}")
    print("=" * 70)
    print("[pipeline] $ " + " ".join(cmd))

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(
            f"\n[pipeline] FAILED at: {description}",
            file=sys.stderr,
        )
        print(
            f"[pipeline] Exit code: {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)


def print_path_status(label: str, path: str):
    """Print whether an expected path currently exists."""

    exists = os.path.exists(path)

    state = "OK" if exists else "MISSING"

    print(
        f"[pipeline] {label:<24} {state:<8} {path}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete LungInsight imaging pipeline "
            "(steps 01-07) for one LIDC-IDRI patient."
        )
    )

    # ========================================================================
    # GENERAL
    # ========================================================================

    parser.add_argument(
        "patient_id",
        help=(
            "LIDC-IDRI PatientID, e.g. "
            "LIDC-IDRI-0141"
        ),
    )

    parser.add_argument(
        "--lidc-root",
        default="Imaging/LIDC/lidc_idri",
        help=(
            "Root directory containing per-patient DICOM folders. "
            "Default: Imaging/LIDC/lidc_idri"
        ),
    )

    parser.add_argument(
        "--out-root",
        default="output",
        help=(
            "Root directory under which patient outputs are written. "
            "Default: output"
        ),
    )

    parser.add_argument(
        "--scripts-dir",
        default=None,
        help=(
            "Directory containing the numbered pipeline scripts. "
            "Default: same directory as run_pipeline.py"
        ),
    )

    parser.add_argument(
        "--start-from",
        choices=STEP_ORDER,
        default="01",
        help=(
            "First step to execute. Earlier outputs must already exist. "
            "Default: 01"
        ),
    )

    parser.add_argument(
        "--stop-after",
        choices=STEP_ORDER,
        default="07",
        help=(
            "Last step to execute. Default: 07"
        ),
    )

    # ========================================================================
    # STEP 01
    # ========================================================================

    step01 = parser.add_argument_group(
        "step 01 - DICOM -> HU"
    )

    step01.add_argument(
        "--no-multi-window",
        action="store_true",
        help=(
            "Skip generation of volume_windowed.npy in Step 01."
        ),
    )

    # ========================================================================
    # STEP 02
    # ========================================================================

    step02 = parser.add_argument_group(
        "step 02 - lung mask + Z crop"
    )

    step02.add_argument(
        "--lung-threshold",
        type=float,
        default=None,
        help=(
            "HU threshold used by Step 02 for lung segmentation. "
            "Default: Step 02's own default."
        ),
    )

    step02.add_argument(
        "--z-crop-margin-mm",
        type=float,
        default=None,
        help=(
            "Physical Z margin retained around the lung extent. "
            "Default: Step 02's own default."
        ),
    )

    step02.add_argument(
        "--no-z-crop",
        action="store_true",
        help=(
            "Disable Z-cropping in Step 02."
        ),
    )

    # ========================================================================
    # STEP 03
    # ========================================================================

    step03 = parser.add_argument_group(
        "step 03 - QA visualization"
    )

    step03.add_argument(
        "--qa-mask",
        choices=[
            "lung",
            "air",
            "heart",
            "lung+heart",
        ],
        default="lung+heart",
        help=(
            "Mask displayed in the QA montage. "
            "Default: lung+heart."
        ),
    )

    step03.add_argument(
        "--qa-num-slices",
        type=int,
        default=12,
        help=(
            "Number of slices in the QA montage. "
            "Default: 12."
        ),
    )

    # ========================================================================
    # STEP 04
    # ========================================================================

    step04 = parser.add_argument_group(
        "step 04 - candidate detection"
    )

    step04.add_argument(
        "--detector-checkpoint",
        default=None,
        help=(
            "Path to the ViTDet3D detector checkpoint used by Step 04. "
            "This is NOT the classifier checkpoint."
        ),
    )

    step04.add_argument(
        "--logit-threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum detector nodule-presence logit required to keep "
            "a detection. A value of 0 corresponds to sigmoid confidence "
            "0.5. Default: 0.0."
        ),
    )

    step04.add_argument(
        "--merge-distance-mm",
        type=float,
        default=10.0,
        help=(
            "Maximum physical center-to-center distance in mm used by "
            "Step 04 to merge nearby detections belonging to the same "
            "candidate. Default: 10.0 mm."
        ),
    )

    step04.add_argument(
        "--stride-fraction",
        type=float,
        default=0.75,
        help=(
            "Sliding-window stride as a fraction of detector crop size. "
            "Smaller values increase overlap and recall but cost more "
            "CPU time. Default: 0.75."
        ),
    )

    step04.add_argument(
        "--detector-batch-size",
        type=int,
        default=4,
        help=(
            "Batch size for Step 04 detector inference. "
            "Default: 4."
        ),
    )

    step04.add_argument(
        "--detector-device",
        default="cpu",
        choices=[
            "cpu",
            "cuda",
        ],
        help=(
            "Device used by Step 04. Default: cpu."
        ),
    )

    # ========================================================================
    # STEP 05
    # ========================================================================

    step05 = parser.add_argument_group(
        "step 05 - classifier patch extraction"
    )

    step05.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help=(
            "Cubic classifier patch size in voxels. "
            "Default: 64."
        ),
    )

    step05.add_argument(
        "--patch-source",
        default="volume_hu.npy",
        help=(
            "Source volume inside Step 02 output. "
            "Default: volume_hu.npy, i.e. the original HU volume "
            "after Z-cropping but BEFORE non-lung masking. "
            "Use volume_hu_masked.npy only if you intentionally want "
            "the classifier input masked."
        ),
    )

    # ========================================================================
    # STEP 06
    # ========================================================================

    step06 = parser.add_argument_group(
        "step 06 - classification + XAI"
    )

    step06.add_argument(
        "--classifier-checkpoint",
        default=None,
        help=(
            "Path to the SE-ResNet3D multi-head classifier checkpoint. "
            "Example: best_model_gpu_v2.pth"
        ),
    )

    step06.add_argument(
        "--target-layer",
        default="backbone.layer3",
        help=(
            "Layer used for Grad-CAM++ hooks. "
            "Default: backbone.layer3."
        ),
    )

    step06.add_argument(
        "--heads",
        default=None,
        help=(
            "Comma-separated subset of classifier heads to process. "
            "Default: all heads present in the checkpoint."
        ),
    )

    step06.add_argument(
        "--skip-saliency",
        action="store_true",
        help=(
            "Skip vanilla gradient saliency computation."
        ),
    )

    step06.add_argument(
        "--classifier-device",
        default="cpu",
        choices=[
            "cpu",
            "cuda",
        ],
        help=(
            "Device used by Step 06. Default: cpu."
        ),
    )

    # ========================================================================
    # STEP 07
    # ========================================================================

    step07 = parser.add_argument_group(
        "step 07 - heatmap visualization"
    )

    step07.add_argument(
        "--xai-head",
        default="malignancy",
        help=(
            "Classifier head displayed in Step 07. "
            "Default: malignancy."
        ),
    )

    step07.add_argument(
        "--map-type",
        default="gradcam",
        choices=[
            "gradcam",
            "saliency",
        ],
        help=(
            "Heatmap type displayed in Step 07. "
            "Default: gradcam."
        ),
    )

    step07.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help=(
            "Only overlay candidates whose selected XAI head score "
            "is >= this value. Default: 0.0."
        ),
    )

    step07.add_argument(
        "--candidate-slices-only",
        action="store_true",
        help=(
            "For headless GIF output, only include slices containing "
            "a candidate."
        ),
    )

    step07.add_argument(
        "--fps",
        type=int,
        default=4,
        help=(
            "Frames per second for Step 07 GIF output. "
            "Default: 4."
        ),
    )

    step07.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Run Step 07 in interactive mode instead of saving a GIF."
        ),
    )

    # ========================================================================
    # PARSE
    # ========================================================================

    args = parser.parse_args()

    # ========================================================================
    # PATHS
    # ========================================================================

    scripts_dir = (
        Path(args.scripts_dir)
        if args.scripts_dir
        else Path(__file__).resolve().parent
    )

    patient_dicom_dir = os.path.join(
        args.lidc_root,
        args.patient_id,
    )

    patient_out_root = os.path.join(
        args.out_root,
        args.patient_id,
    )

    # Per-step output paths.

    dir_01 = os.path.join(
        patient_out_root,
        "01_hu",
    )

    dir_02 = os.path.join(
        patient_out_root,
        "02_masked",
    )

    dir_03_png = os.path.join(
        patient_out_root,
        "03_qa_montage.png",
    )

    dir_04 = os.path.join(
        patient_out_root,
        "04_candidates",
    )

    dir_05 = os.path.join(
        patient_out_root,
        "05_patches",
    )

    dir_06 = os.path.join(
        patient_out_root,
        "06_xai",
    )

    dir_07_gif = os.path.join(
        patient_out_root,
        "07_overlay.gif",
    )

    # ========================================================================
    # DETERMINE STEPS
    # ========================================================================

    start_index = STEP_ORDER.index(
        args.start_from
    )

    stop_index = STEP_ORDER.index(
        args.stop_after
    )

    if start_index > stop_index:
        print(
            "[pipeline] ERROR: --start-from cannot be later than "
            "--stop-after.",
            file=sys.stderr,
        )
        sys.exit(2)

    steps_to_run = STEP_ORDER[
        start_index:
        stop_index + 1
    ]

    # ========================================================================
    # HEADER
    # ========================================================================

    print()
    print("=" * 70)
    print("LUNGINSIGHT PIPELINE")
    print("=" * 70)

    print(
        f"[pipeline] Patient:      {args.patient_id}"
    )

    print(
        f"[pipeline] Steps:        {', '.join(steps_to_run)}"
    )

    print(
        f"[pipeline] Output root:  {patient_out_root}"
    )

    print(
        f"[pipeline] Scripts dir:  {scripts_dir}"
    )

    print("=" * 70)

    # ========================================================================
    # STEP 01
    # ========================================================================

    if "01" in steps_to_run:

        step01_args = [
            patient_dicom_dir,
            "--out-dir",
            dir_01,
        ]

        if args.no_multi_window:
            step01_args.append(
                "--no-multi-window"
            )

        run_step(
            scripts_dir / SCRIPT_NAMES["01"],
            step01_args,
            "Step 1/7: DICOM -> HU volume",
        )

    # ========================================================================
    # STEP 02
    # ========================================================================

    if "02" in steps_to_run:

        step02_args = [
            dir_01,
            "--out-dir",
            dir_02,
        ]

        if args.lung_threshold is not None:

            step02_args += [
                "--lung-threshold",
                str(args.lung_threshold),
            ]

        if args.z_crop_margin_mm is not None:

            step02_args += [
                "--z-crop-margin-mm",
                str(args.z_crop_margin_mm),
            ]

        if args.no_z_crop:

            step02_args.append(
                "--no-z-crop"
            )

        run_step(
            scripts_dir / SCRIPT_NAMES["02"],
            step02_args,
            "Step 2/7: Lung segmentation + Z-crop",
        )

    # ========================================================================
    # STEP 03
    # ========================================================================

    if "03" in steps_to_run:

        step03_args = [
            dir_02,
            "--save",
            dir_03_png,
            "--mask",
            args.qa_mask,
            "--num-slices",
            str(args.qa_num_slices),
        ]

        run_step(
            scripts_dir / SCRIPT_NAMES["03"],
            step03_args,
            "Step 3/7: QA visualization montage",
        )

    # ========================================================================
    # STEP 04
    #
    # IMPORTANT:
    #
    # This matches the CURRENT Step 04 argparse interface shown in
    # your error:
    #
    #   --source
    #   --stride-fraction
    #   --logit-threshold
    #   --merge-distance-mm
    #   --batch-size
    #   --device
    #
    # It deliberately does NOT pass:
    #
    #   --confidence-threshold
    #   --nms-iou-threshold
    #
    # because those belong to the older Step 04 implementation.
    # ========================================================================

    if "04" in steps_to_run:

        if not args.detector_checkpoint:

            print(
                "\n[pipeline] STOPPING: Step 04 requires "
                "--detector-checkpoint.",
                file=sys.stderr,
            )

            sys.exit(1)

        detector_checkpoint = os.path.abspath(
            args.detector_checkpoint
        )

        if not os.path.exists(
            detector_checkpoint
        ):

            print(
                f"\n[pipeline] ERROR: detector checkpoint not found:\n"
                f"           {detector_checkpoint}",
                file=sys.stderr,
            )

            sys.exit(1)

        step04_args = [
            dir_02,
            "--checkpoint",
            detector_checkpoint,
            "--out-dir",
            dir_04,

            # CURRENT Step 04 interface:
            "--logit-threshold",
            str(args.logit_threshold),

            "--merge-distance-mm",
            str(args.merge_distance_mm),

            "--stride-fraction",
            str(args.stride_fraction),

            "--batch-size",
            str(args.detector_batch_size),

            "--device",
            args.detector_device,
        ]

        run_step(
            scripts_dir / SCRIPT_NAMES["04"],
            step04_args,
            "Step 4/7: Candidate detection",
        )

    # ========================================================================
    # STEP 05
    # ========================================================================

    if "05" in steps_to_run:

        step05_args = [
            dir_04,
            "--volume-dir",
            dir_02,
            "--out-dir",
            dir_05,
            "--patch-size",
            str(args.patch_size),
            "--source",
            args.patch_source,
        ]

        run_step(
            scripts_dir / SCRIPT_NAMES["05"],
            step05_args,
            "Step 5/7: Candidate patch extraction",
        )

    # ========================================================================
    # STEP 06
    # ========================================================================

    if "06" in steps_to_run:

        if not args.classifier_checkpoint:

            print(
                "\n[pipeline] STOPPING: Step 06 requires "
                "--classifier-checkpoint.",
                file=sys.stderr,
            )

            sys.exit(1)

        classifier_checkpoint = os.path.abspath(
            args.classifier_checkpoint
        )

        if not os.path.exists(
            classifier_checkpoint
        ):

            print(
                f"\n[pipeline] ERROR: classifier checkpoint not found:\n"
                f"           {classifier_checkpoint}",
                file=sys.stderr,
            )

            sys.exit(1)

        step06_args = [
            dir_05,
            "--checkpoint",
            classifier_checkpoint,
            "--out-dir",
            dir_06,
            "--target-layer",
            args.target_layer,
            "--device",
            args.classifier_device,
        ]

        if args.heads:

            step06_args += [
                "--heads",
                args.heads,
            ]

        if args.skip_saliency:

            step06_args.append(
                "--skip-saliency"
            )

        run_step(
            scripts_dir / SCRIPT_NAMES["06"],
            step06_args,
            "Step 6/7: Classification + Grad-CAM++/saliency",
        )

    # ========================================================================
    # STEP 07
    # ========================================================================

    if "07" in steps_to_run:

        step07_args = [
            "--volume-dir",
            dir_02,

            "--xai-dir",
            dir_06,

            "--head",
            args.xai_head,

            "--map-type",
            args.map_type,

            "--score-threshold",
            str(args.score_threshold),
        ]

        if args.interactive:

            print(
                "[pipeline] Step 07 running in interactive mode."
            )

        else:

            step07_args += [
                "--save",
                dir_07_gif,
                "--fps",
                str(args.fps),
            ]

            if args.candidate_slices_only:

                step07_args.append(
                    "--candidate-slices-only"
                )

        run_step(
            scripts_dir / SCRIPT_NAMES["07"],
            step07_args,
            "Step 7/7: Overlay visualization",
        )

    # ========================================================================
    # COMPLETION
    # ========================================================================

    print()
    print("=" * 70)
    print("[pipeline] PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"[pipeline] Patient: {args.patient_id}"
    )

    print(
        f"[pipeline] Outputs: {patient_out_root}"
    )

    print()

    print_path_status(
        "01 HU volume",
        dir_01,
    )

    print_path_status(
        "02 masked volume",
        dir_02,
    )

    print_path_status(
        "03 QA montage",
        dir_03_png,
    )

    print_path_status(
        "04 candidates",
        dir_04,
    )

    print_path_status(
        "05 patches",
        dir_05,
    )

    print_path_status(
        "06 XAI",
        dir_06,
    )

    print_path_status(
        "07 overlay",
        dir_07_gif,
    )

    print()

    if "07" not in steps_to_run:

        print(
            "[pipeline] Pipeline intentionally stopped before Step 07."
        )

    print(
        "[pipeline] Done."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()