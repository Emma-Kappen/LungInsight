# Baseline audit

## Scope and method

Audited the active workspace at `LungInsight_v2`, not earlier generated patch bundles. The root `meta_schema.json` is the canonical schema selected by `backend/Imaging/pipeline_schema.py`; the older nested `backend/Imaging/meta_schema.json` is not the active source of truth. Existing uncommitted repository changes and deletions predate this implementation and were preserved.

## Confirmed current workspace facts

- Active Python: project virtual environment `l-env`, Python 3.12.10.
- Active root schema before the implementation: version 2.10, supported versions through 2.10. It declares CT array order ZYX and physical DICOM-LPS millimetres.
- Stage 03 model/checkpoint helper files and the Stage 04 joblib model exist at their schema-declared locations. `validate_pipeline.py --check-artifacts` confirms all five Stage 03 checkpoint groups, the frozen model helpers, and the Stage 04 artifact.
- Stage 04 is an `ExtraTreesRegressor`, receives six Stage 03 scalar ratings in order `subtlety`, `sphericity`, `margin`, `lobulation`, `spiculation`, `texture`, and estimates a 1–5 LIDC reader malignancy/suspicion rating. This is not a binary classifier or cancer probability.
- The Stage 04 training table contains 4,221 rows labeled `train` and an explicit `validation` split. The report records 613 train patients and 131 validation patients with no overlap.
- Required runtime imports were present during the audit: NumPy, PyTorch, pydicom, Transformers, SHAP, SciPy, Matplotlib, Pillow, joblib, and scikit-learn.
- The LIDC input folder for `LIDC-IDRI-0001` exists and contains 135 DICOM files.
- Current existing Stage 01 output at `backend/output/LIDC-IDRI-0001` contains `(133,512,512)` float32 `volume_hu.npy`, `(108,512,512)` float32 `volume_hu_cropped.npy`, and shape-aligned masks. The lung mask contains 3,501,989 voxels and the heart exclusion mask contains 572,165 voxels.
- That existing output was written under metadata schema 2.10 and lacks the new heart review metrics, montages, and attestation. No Stage 02 or Stage 03 output directory currently exists for this patient.
- The measured current output differs from the older sample values stated in `prompt.md` (including crop depth and heart-mask count). Those older observations were not treated as current facts.

## Confirmed defects found in the pre-change code

1. Stage 01's heuristic heart exclusion could exist without an operator review artifact; Stage 02 did not require the human review before detector inference.
2. Stage 02 did not apply the existing canonical heart/lung acceptance filter to ViTDet3D candidates before authoritative patch extraction, and its LoG path likewise needed a separate accepted-only boundary.
3. The orchestrator reran Stage 01 against a fixed patient output directory by default, risking overwrite of existing patient artifacts.
4. Stage 04 used the canonical joblib regressor but emitted no explanation output. The project has a declared SHAP dependency and an explicit train/validation table; a train-only background can be derived reproducibly.
5. Stage 06 projected CAMs but did not write a stable, linked nodule record joining detector geometry, Stage 03 attributes, Stage 04 rating, CAM references, and SHAP.
6. Stage 07 reported candidate ratings and masks, but did not consume Stage 06 records or produce per-nodule CAM/SHAP visual summaries.

## Baseline checks and their limits

- Existing focused synthetic imaging tests before this work: 40 passed across heart-rod tracking and candidate fusion/filtering.
- The active patient output and model files were inspected read-only. Stage 01 was not rerun because that would overwrite the existing patient output and no current operator review attestation exists.
- No real patient detector or Stage 03–07 inference was run in the baseline audit.
- The default ViTDet3D detector reference is remote (`rlsn/DeTr4LungNodule`); its cache/network availability is not asserted by the local artifact validator.
- SHAP training-table presence and split semantics were inspected; no clinical validity claim follows from this software audit.

## Worktree warning

The worktree already contained broad unrelated changes and deletions before this task. The implementation and ZIP do not restore or remove those changes. Use `INSTALL.ps1` from a clean temporary extraction to preserve any target files it replaces.
