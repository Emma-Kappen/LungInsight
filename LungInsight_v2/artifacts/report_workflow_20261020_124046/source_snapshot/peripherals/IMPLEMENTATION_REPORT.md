# Implementation report

## Implemented

- **Stage 01:** Emits per-slice heart-mask review metrics and paginated CT/lung/heart overlays; creates a hash-bound operator review attestation. An automated tracker result is not treated as clinical validation.
- **Stage 02:** Requires a valid attestation before detector inference and filters both ViTDet3D and LoG records before patch extraction. Only accepted candidates enter their respective manifests; rejected candidates remain separate audit artifacts, and LoG stays diagnostic-only.
- **Stage 03:** Preserves the five existing hybrid checkpoint groups and the existing 64³ float32 patch interface; no model weights or architecture were modified.
- **Stage 04:** Keeps the canonical joblib ExtraTrees rating model separate from the optional ordinal classifier. Uses a deterministic 100-row sample from the training split of the model's XML nodule table for interventional `shap.TreeExplainer`, records feature order, preprocessing, background provenance, model hash, and verifies additivity against the raw regressor output. SHAP contributions explain the raw score before rating clipping/rounding.
- **Stage 05:** Records the exact target output index and expected-rating scalar, raw signed Grad-CAM formula, and gradient polarity interpretation. No attribute-specific inversion or checkpoint modification is applied.
- **Stage 06:** Validates the Stage 02 gate before projection, joins only accepted manifest records to matching Stage 03/04 `(candidate_id, patch_id)` identities, and writes schema-required nodule records with geometry, detector provenance, model identity, rating, uncertainty/limitations, and per-patch CAM and SHAP references.
- **Stage 07:** Reads the Stage 06 records and requires exact identity-set agreement with Stage 03; it writes a headless per-accepted-nodule CT/bounding-box/CAM/SHAP image and exposes the visualization path with the report. Outside-lung candidates stay in the separate exclusion audit.
- **Orchestration:** Runs schema/artifact/dependency checks and both test suites; stops on missing review approval; supports validated Stage 02 resumption; by default refuses to overwrite existing patient output. `-RegenerateStage01` makes an explicit timestamped backup of configured Stage 01 files before regeneration.
- **Validation and delivery:** Adds static contract checks, synthetic SHAP/Stage 06/07 tests, user documentation, installation backup/checksum validation, and the companion audit/run reports.

## Schema and semantics

The canonical root `meta_schema.json` is now **2.12**. It adds the SHAP background/explainer contract, Grad-CAM target/polarity metadata, Stage 06 record fields, Stage 07 visualization outputs, and run summary fields. Supported historical versions remain listed; the root schema remains authoritative. The stale nested Imaging schema was not promoted.

The Stage 04 target remains a LIDC reader-assigned suspicion/malignancy rating from 1 to 5. No `malignant`/`benign` label or cancer probability is generated. The training report shows weak Stage 03-output discrimination (437 held-out nodules; MAE 0.758 versus constant baseline MAE 0.729; QWK 0.016), which must be considered before interpreting ratings.

## Source files in this delivery

- `meta_schema.json`, `run_pipeline.ps1`, `validate_pipeline.py`, and `README.md`.
- Stage scripts: `backend/Imaging/01_hu_and_crop.py`, `02_detect_and_patch.py`, `03_nodule_classification.py`, `04_malignancy_prediction.py`, `04_malignancy_classifier.py`, `05_gradcam.py`, `06_full_ct_gradcam.py`, `07_visualize.py`, and `pipeline_schema.py`.
- New/reused helpers: `attest_heart_mask_review.py`, `heart_mask_review.py`, `stage04_shap.py`, `heart_rod_tracking.py`, `lunginsight_candidate_fusion.py`, `frozen_attribute_common.py`, and the existing `spatial_v1/model_objective.py`.
- Tests: all root `tests/` and `backend/Imaging/tests/` Python test files.
- Reports and delivery metadata: `BASELINE_AUDIT.md`, `IMPLEMENTATION_REPORT.md`, `COMPATIBILITY_AND_RUN_REPORT.md`, `MANIFEST.txt`, `SHA256SUMS.txt`, and `INSTALL.ps1`.

## Validation actually performed

- `validate_pipeline.py --check-artifacts`: passed; schema v2.12, local models/helpers/checkpoints, and declared Python dependencies verified.
- Root unittest suite: 35 tests passed.
- Imaging tracker/candidate suite: 40 tests passed.
- Stage 04 synthetic-input CLI smoke test using the actual frozen model: completed; six SHAP contributions emitted; 4,221 train rows available and deterministic 100-row background used; maximum observed SHAP additivity error approximately `1.72e-08`.
- Standalone SHAP unit tests: 2 passed; Stage 06/07 record and image tests: 3 passed.
- Python compilation/AST and schema JSON checks passed; PowerShell parser check passed.
- All 24 bundled Python files parsed successfully, and a headless Stage 05 rendering smoke test wrote the expected PNG, metadata JSON, and raw/scaled CAM arrays.
- A real Stage 06 output failure was reproduced: normalized `crop_start_zyx` NumPy arrays were leaking into JSON records. Stage 06 now converts that geometry field back to a JSON-safe integer list; the targeted test and the existing patient record serialize successfully.
- The run's current attestation names the reviewer `YOUR_NAME`. The review gate now rejects template reviewer placeholders both when creating and validating attestations, so a real operator identity is required before resuming.
- No real patient or full detector-to-visualization run was performed. The existing patient output lacks the current review artifacts and attestation, and Stage 01 was intentionally not rerun.

## Remaining limits

- A qualified operator must inspect every current Stage 01 review montage and create the hash-bound attestation before Stage 02 can run.
- Real ViTDet3D model download/cache, complete patient inference, Stage 05 CAM output quality, and Stage 06/07 real-data image correspondence remain unverified.
- Grad-CAM colors are per-patch display normalizations. Raw signed maps are retained; gradients target expected ordinal attribute ratings. This does not establish clinical polarity or causal localization.
- SHAP is global-feature attribution for the actual Stage 04 regressor, not image localization or causality. The model's train-table features may differ from Stage 03 predictions.
- No pathology-confirmed label, binary probability, diagnostic threshold, or clinical validation is supplied.
