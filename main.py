"""
main.py

Single entry point for the LungInsight imaging pipeline. Place this file
in the project root (the parent folder of Imaging/) and run it from
there. It prompts for a patient ID and runs Imaging/run_pipeline.py
end-to-end for that patient.

Only one trained checkpoint currently exists on disk:

    Imaging/checkpoints/best_model_gpu_v2.pth

run_pipeline.py's Step 06 wants two separate checkpoints
(--nodule-checkpoint for Stage 06A, --malignancy-checkpoint for Stage
06B) -- since only one file exists, both are pointed at it below. Swap
in two distinct paths independently once separate checkpoints exist.

This checkpoint is NOT Step 06's hardcoded DualScaleBackbone -- it's an
SE-ResNet50 multi-head model (see Imaging/se_resnet3d.py:
MultiHeadSEResNet3D / se_resnet50_3d), trained with nine LIDC
characteristic heads and no nodule/non-nodule head. Step 06 now supports
loading this dynamically (mirroring Step 07's --model-module/
--model-class pattern), which is what STEP06_* below wires up:
    - Stage 06A's false-positive gate is disabled automatically (no
      'nodule' head on this checkpoint) -- every Stage 05 candidate
      passes straight through to Stage 06B, and this is logged by 06
      itself so it's never silently assumed to have happened.
    - Stage 06B reads the 'malignancy' head plus all remaining heads
      (spiculation, lobulation, calcification, margin, texture,
      sphericity, subtlety) as attributes automatically.
    - STEP06_PATCH_SOURCE picks which Stage 05 patch (local 32mm vs.
      context >=64mm) feeds this single-input model. This is NOT
      verified against the actual checkpoint -- 'local' (the
      nodule-centered crop) is the conventional choice for an LIDC
      characteristic model, but change it to 'context' and re-run if
      results look wrong.
    - se_resnet3d.py must sit in Imaging/ next to 06_run_inference_xai.py
      so Step 06's dynamic import can find it.

Step 07 (explainability / CAM) additionally needs the model's Python
class (--xai-model-module / --xai-model-class) and its target conv
layers (--xai-local-target-layer / --xai-context-target-layer) --
none of which are knowable from a checkpoint path alone, and none of
which were given. The XAI_* constants below are therefore left unset,
and this script runs Steps 1-6 and stops there rather than guessing at
values that would just fail inside run_pipeline.py. Fill in the XAI_*
constants (either the four live-model ones, or XAI_ATTRIBUTIONS_DIR
for the precomputed-attributions fallback) once you know them, and
this script will automatically include Step 7.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_PIPELINE = PROJECT_ROOT / "Imaging" / "run_pipeline.py"
LIDC_ROOT = PROJECT_ROOT / "Imaging" / "LIDC" / "lidc_idri"
CHECKPOINT = PROJECT_ROOT / "Imaging" / "checkpoints" / "best_model_gpu_v2.pth"

# Step 06 dynamic model config -- see module docstring above. This
# checkpoint is an SE-ResNet50 multi-head model, not Step 06's hardcoded
# DualScaleBackbone, so both the "nodule" and "malignancy" checkpoint args
# point at the same file AND at the same dynamically-loaded class.
STEP06_MODEL_MODULE = "se_resnet3d"
STEP06_MODEL_CLASS = "MultiHeadSEResNet3D"
STEP06_PATCH_SOURCE = "local"  # or "context" -- see docstring note above
STEP06_NODULE_HEAD_NAME = "nodule"          # absent on this checkpoint -> gate disabled
STEP06_MALIGNANCY_HEAD_NAME = "malignancy"  # must exist on the checkpoint

# Stage 07 (explainability) config. This checkpoint is single-input (one
# trunk, no separate local/context encoders) -- Step 07 now detects that
# automatically from the model's own forward() signature and, per branch,
# runs the SAME trunk on the local patch and on the context patch as two
# independent forward/backward passes, rather than needing two encoders.
# Because there's only one trunk, both target-layer args point at the same
# layer: 'layer4', which se_resnet3d.MultiHeadSEResNet3D exposes directly
# on the model (self.layer4 = self.backbone.layer4) for exactly this.
XAI_ATTRIBUTIONS_DIR = None
XAI_MODEL_MODULE = "se_resnet3d"
XAI_MODEL_CLASS = "MultiHeadSEResNet3D"
XAI_LOCAL_TARGET_LAYER = "layer4"
XAI_CONTEXT_TARGET_LAYER = "layer4"


def normalize_patient_id(raw: str) -> str:
    """Accept either the full 'LIDC-IDRI-0141' form or a bare number like
    '0141' / '141', and always return the full zero-padded form the
    on-disk DICOM folders actually use."""
    if raw.upper().startswith("LIDC-IDRI-"):
        return raw
    if raw.isdigit():
        return f"LIDC-IDRI-{int(raw):04d}"
    return raw


def main():
    typed = input("Patient ID (e.g. LIDC-IDRI-0141, or just 0141): ").strip()
    if not typed:
        print("[main] No patient ID entered, exiting.", file=sys.stderr)
        sys.exit(1)
    patient_id = normalize_patient_id(typed)
    if patient_id != typed:
        print(f"[main] Interpreting '{typed}' as '{patient_id}'.")

    if not RUN_PIPELINE.exists():
        print(f"[main] Could not find '{RUN_PIPELINE}'.", file=sys.stderr)
        sys.exit(1)
    if not CHECKPOINT.exists():
        print(f"[main] Could not find checkpoint '{CHECKPOINT}'.", file=sys.stderr)
        sys.exit(1)

    have_live_xai = all([
        XAI_MODEL_MODULE, XAI_MODEL_CLASS,
        XAI_LOCAL_TARGET_LAYER, XAI_CONTEXT_TARGET_LAYER,
    ])
    have_attributions_xai = bool(XAI_ATTRIBUTIONS_DIR)
    run_step07 = have_live_xai or have_attributions_xai

    cmd = [
        sys.executable, str(RUN_PIPELINE),
        patient_id,
        "--lidc-root", str(LIDC_ROOT),
        "--nodule-checkpoint", str(CHECKPOINT),
        "--malignancy-checkpoint", str(CHECKPOINT),
        "--nodule-model-module", STEP06_MODEL_MODULE,
        "--nodule-model-class", STEP06_MODEL_CLASS,
        "--malignancy-model-module", STEP06_MODEL_MODULE,
        "--malignancy-model-class", STEP06_MODEL_CLASS,
        "--step06-patch-source", STEP06_PATCH_SOURCE,
        "--nodule-head-name", STEP06_NODULE_HEAD_NAME,
        "--malignancy-head-name", STEP06_MALIGNANCY_HEAD_NAME,
    ]

    if run_step07:
        if have_attributions_xai:
            cmd += ["--xai-attributions-dir", str(XAI_ATTRIBUTIONS_DIR)]
        else:
            cmd += [
                "--xai-checkpoint", str(CHECKPOINT),
                "--xai-model-module", XAI_MODEL_MODULE,
                "--xai-model-class", XAI_MODEL_CLASS,
                "--xai-local-target-layer", XAI_LOCAL_TARGET_LAYER,
                "--xai-context-target-layer", XAI_CONTEXT_TARGET_LAYER,
            ]
    else:
        cmd += ["--stop-after", "06"]
        print(
            "[main] Step 07 (explainability) config not set in main.py -- "
            "running Steps 1-6 only. Fill in the XAI_* constants at the "
            "top of this file to also run Step 7."
        )

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()