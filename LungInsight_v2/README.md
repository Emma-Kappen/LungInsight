# LungInsight v2 Imaging pipeline delivery — schema 2.12

This delivery adds a review-gated Stages 01–07 workflow. The project-root `meta_schema.json` is canonical. The older `backend/Imaging/meta_schema.json` is not used as the active schema.

## Installation and rollback

1. Extract the downloaded ZIP into a **new temporary folder**, not over the working project.
2. From PowerShell, run the checksum-verifying, backup-aware installer:

   ```powershell
   & "$env:TEMP\LungInsight_v2_bundle\LungInsight_v2\INSTALL.ps1" `
       -TargetWorkspaceRoot 'C:\Users\emkap\Documents\Projects\LungInsight'
   ```

   Replace the extraction path if needed. The installer validates the package SHA-256 manifest before copying. Existing destination files are copied to `.lunginsight_backups\LungInsight_v2_<UTC timestamp>` under the target workspace. No datasets, patient outputs, model weights, or checkpoints are bundled.
3. To roll back, copy the backed-up files from that timestamped folder to the same relative paths under the target workspace. For newly created files, consult `NEW_FILES_SHA256.txt` in the backup folder; remove a listed file only if its current SHA-256 still matches the installed hash.

## Dependencies and external artifacts

Use the project virtual environment and the full `LungInsight_v2\requirements.txt` for end-to-end runs; the smaller `backend\Imaging\requirements.txt` is not the complete Stages 01–07 dependency set. SHAP is required and is checked before pipeline execution.

The following data and trained artifacts remain external to the ZIP and must already exist:

- LIDC DICOM data under `datasets\LIDC\lidc_idri\<PatientID>\...`.
- The five existing Stage 03 hybrid checkpoint groups under `backend\models\nodule_ranker_3d_distributional_v1\spatial_v1\checkpoints\hybrid\`.
- The Stage 03 hybrid checkpoints and Stage 04 `backend\models\malignancy\xml_nodule_level_v1\malignancy_from_nodule_level_xml.joblib` plus `nodule_level_xml_table.csv`. The required inference helper and frozen model source are included in the ZIP.
- The default Stage 02 ViTDet3D model `rlsn/DeTr4LungNodule`, which may need network access or a populated Hugging Face cache.

## Schema migration and safety semantics

The root schema is version **2.12**. It adds Stage 01 operator-review artifacts, Stage 04 SHAP configuration/provenance, Stage 05 target/polarity metadata, Stage 06 stable nodule records, and Stage 07 visualization outputs. Existing model weights and their architecture are unchanged. LoG remains diagnostic-only. Stage 04 outputs an estimated LIDC reader rating, not cancer probability or pathology truth.

Stage 02 fails closed until a human has reviewed every current Stage 01 heart/lung montage and created a hash-bound attestation. A saved attestation becomes invalid if any covered array or review artifact changes. The current sample patient output is an older schema 2.10 result without those review artifacts. The orchestrator refuses to overwrite existing patient output unless `-RegenerateStage01` is explicitly set; that option backs up configured Stage 01 files first.

## Tests and commands

The pipeline orchestrator is a PowerShell script. On systems where unsigned local scripts are blocked, allow scripts for the **current PowerShell process only** before invoking it. This does not change the machine or user execution policy. Run all commands from the `LungInsight_v2` directory:

```powershell
Set-Location 'C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2'
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. .\l-env\Scripts\Activate.ps1
python .\validate_pipeline.py --check-artifacts
python -m unittest discover -s tests -v
python -m unittest discover -s backend\Imaging\tests -v
```

If policy is controlled by Group Policy and the process-level setting is refused, launch the orchestrator in a child PowerShell process with `-ExecutionPolicy Bypass`:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    'C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2\run_pipeline.ps1' `
    -PatientId 'LIDC-IDRI-0001' -ResumeFromStage02 -NoGui
```

## Recovering an interrupted run

If a run stops after Stage 02 has started, do not use `-RegenerateStage01` just to retry: that recomputes Stage 01 and creates another review set. After resolving the reported error, resume from Stage 02 instead:

```powershell
.\run_pipeline.ps1 -PatientId 'LIDC-IDRI-0001' -ResumeFromStage02 -NoGui
```

Stage 02 includes a CPU-based multi-scale LoG diagnostic scan, which can take a while on a full CT. Let it finish; a `KeyboardInterrupt` during `gaussian_laplace` means the scan was interrupted, not that it completed. The resume command reruns Stage 02 and all later stages, while leaving Stage 01 untouched.

To create fresh Stage 01 outputs while preserving the existing ones:

```powershell
.\run_pipeline.ps1 -PatientId 'LIDC-IDRI-0001' -RegenerateStage01 -NoGui
```

That invocation is expected to stop after generating the review material until the operator approves it. Inspect every `heart_mask_review_montage_*.png` and `heart_mask_review_metrics.json`, then attest:

```powershell
python .\backend\Imaging\attest_heart_mask_review.py `
    --patient-id 'LIDC-IDRI-0001' --reviewer 'ACTUAL_OPERATOR_ID' --confirm-review
```

Replace `ACTUAL_OPERATOR_ID` with the real reviewer identity; example placeholders are rejected. Run Stages 02–07 without rerunning Stage 01:

```powershell
.\run_pipeline.ps1 -PatientId 'LIDC-IDRI-0001' -ResumeFromStage02 -NoGui
```

Use `-NoGui` for headless execution. Stage 07 still writes one PNG per accepted Stage 06 record and JSON/CSV summaries. `pipeline_run_summary.json` records command status, inputs, expected output existence, per-stage exit codes, model hashes, and available counters.

Expected artifacts include the candidate acceptance/rejection reports and accepted-only patch manifests; Stage 03 attribute predictions; Stage 04 rating/SHAP outputs; per-patch CAMs and full-CT projections; `stage06_nodule_records.json`; Stage 07 accepted/excluded reports and `nodule_visualizations\nodule_<patch>_explanation.png`.

## Files changed and purpose

- `backend\Imaging\01_hu_and_crop.py`, `heart_mask_review.py`, `attest_heart_mask_review.py`: per-slice review QC and hash-bound human gate.
- `backend\Imaging\02_detect_and_patch.py`, `lunginsight_candidate_fusion.py`: accepted-only patch extraction and separate authoritative/diagnostic candidate audits.
- `backend\Imaging\04_malignancy_prediction.py`, `stage04_shap.py`: frozen model inference and deterministic train-only SHAP.
- `backend\Imaging\05_gradcam.py`: exact expected-rating target definition and raw signed-CAM polarity provenance.
- `backend\Imaging\06_full_ct_gradcam.py`, `07_visualize.py`: stable nodule record joins and per-record CT/CAM/SHAP visual summaries.
- `run_pipeline.ps1`, `validate_pipeline.py`, `meta_schema.json`, tests and reports: safety orchestration, validation, schema and delivery documentation.

## Limitations

- No real patient end-to-end run was performed. Stage 01 output must be regenerated and reviewed before inference can proceed.
- Stage 04 has weak discrimination on Stage 03-predicted features in the existing report (437 cases, MAE 0.758 vs 0.729 constant baseline, QWK 0.016).
- CAM overlays show relative model response and are not clinical evidence or voxelwise probabilities. SHAP explains tabular Stage 04 features and is not image localization.
- The outputs are research software, not a medical device or diagnosis. Human review is an operational gate, not clinical validation.

See `BASELINE_AUDIT.md`, `IMPLEMENTATION_REPORT.md`, and `COMPATIBILITY_AND_RUN_REPORT.md` for actual findings, commands, results, and remaining blockers.

---
