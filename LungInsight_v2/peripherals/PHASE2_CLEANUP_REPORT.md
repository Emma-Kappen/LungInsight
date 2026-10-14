# Phase 2: Safe Cleanup Report

Date: 2026-10-14

## Scope and method

Used [IMAGING_PIPELINE_STRUCTURE.md](./IMAGING_PIPELINE_STRUCTURE.md) and
[CLEANUP_REPORT.md](./CLEANUP_REPORT.md) as the baseline. Protected the dataset
and patient-output roots, orchestrator, canonical schema, schema helper,
numbered stage scripts, and their source-level dependencies. Model architecture,
training, and inference behavior were not changed.

The static scan covered **140 Python source files** in the repository/project
and timestamped source snapshots. It excluded `.git`, the project virtual
environment, dataset storage, and generated `backend/output` content. Reverse
edges below are static Python import relationships; file/path references in
Python, PowerShell, shell, JSON, and Markdown were separately searched for
orphan candidates. Computed/dynamic imports, external callers, and user-run
commands cannot be ruled out by a static repository scan; ambiguous files were
retained.

## Files moved to review archive

| Previous path | Archived path | Evidence |
|---|---|---|
| `main.py` | `archive/review/20261014/main.py` | Zero-byte placeholder; no importers or exact file references in subprocess scripts, pipeline orchestration, schema, or documentation. |
| `patch_gradcam_diagnostic.py` | `archive/review/20261014/patch_gradcam_diagnostic.py` | One-off source-patching script; no importers or exact file references in subprocess scripts, pipeline orchestration, schema, or documentation. Its hard-coded target is the protected Stage 05 file, so it was retained in review rather than deleted. |

No files were permanently deleted or consolidated.

## Generated-artifact cleanup

- Removed **39 `.pyc` files** from 9 reviewed source/test/tool
  `__pycache__` directories under `LungInsight_v2/`, excluding the virtual
  environment, datasets, and generated output. Each removed cache directory
  contained only bytecode files.
- Found no `.ipynb` notebooks or `.tmp` files in the project tree outside
  protected data/output roots.
- No temporary test output files were identified for removal. Existing tests
  use temporary directories and clean them up.
- Did not touch dataset/storage or generated patient output directories.

## Duplicate utilities and consolidation opportunities

Several repeated helper names exist, notably `read_json` in the active imaging
stages and `sha256_file` in imaging/model utilities. No consolidation was made:
the active copies have different validation/error contracts or are used by
different protected stages. Similar hash helpers also occur in separately
invoked model training/validation scripts; changing those would alter training
workflows. Timestamped backup copies were treated as historical snapshots, not
live duplicate implementations. No pair was confirmed to satisfy both
**greater than 90% implementation similarity** and the condition that only one
copy is referenced while preserving the existing stage, training, and inference
contracts.

## Unresolved ambiguities

- Standalone diagnostic, audit, comparison, and model experiment scripts often
  have no in-repository callers but remain directly executable and document
  their own research purpose. They are categorized `EXPERIMENTAL`, not
  `ORPHANED`, and were not moved.
- Training and validation entrypoints may be run manually or by external
  workflows. Their lack of static import callers is not evidence that they are
  abandoned; they remain in place.
- `04_malignancy_classifier.py` is distinct from the primary
  `04_malignancy_prediction.py` pipeline implementation. Both numbered Stage 04
  files are preserved.
- The nested `backend/Imaging/meta_schema.json` and timestamped backups remain
  because the baseline documents fallback and provenance reasons to preserve
  them.
- The static import parser does not prove absence of importlib, plugin,
  reflection, or external command-line use. The two moved files were selected
  only after exact repository reference checks and manual content review.

## Proposed deletions

- **No source, model, checkpoint, dataset, or output deletions proposed.**
- The 39 removed bytecode files were generated cache artifacts and are
  reproducible.
- No abandoned notebooks or persistent temporary test outputs were found.

## File categories and reverse-dependency graph

Each row is one Python file. “Imported by” lists reverse edges: Python files
with a detected static import of that module. `—` means no reverse import edge
was detected. Backup snapshots are included as separate files. Categories are
`ACTIVE`, `SHARED UTILITY`, `TEST`, `EXPERIMENTAL`, `LEGACY`, and `ORPHANED`.
The graph contains 127 detected direct import edges; the scan recognizes
ordinary static import statements and may omit dynamic or multiline imports.

## Inventory totals

| Category | Files |
|---|---:|
| EXPERIMENTAL | 57 |
| LEGACY | 47 |
| ACTIVE | 10 |
| SHARED UTILITY | 7 |
| TEST | 19 |

| Category | Python file | Imported by (reverse dependencies) |
|---|---|---|
| EXPERIMENTAL | `baseline.py` | none |
| EXPERIMENTAL | `diagnose_predictions.py` | none |
| LEGACY | `LungInsight_v2/archive/review/20261014/main.py` | none |
| LEGACY | `LungInsight_v2/archive/review/20261014/patch_gradcam_diagnostic.py` | none |
| EXPERIMENTAL | `LungInsight_v2/audit_gradcam_heads.py` | none |
| EXPERIMENTAL | `LungInsight_v2/audit_signed_gradcam.py` | none |
| ACTIVE | `LungInsight_v2/backend/Imaging/01_hu_and_crop.py` | none |
| ACTIVE | `LungInsight_v2/backend/Imaging/02_detect_and_patch.py` | none |
| ACTIVE | `LungInsight_v2/backend/Imaging/03_nodule_classification.py` | none |
| ACTIVE | `LungInsight_v2/backend/Imaging/04_malignancy_classifier.py` | none |
| ACTIVE | `LungInsight_v2/backend/Imaging/04_malignancy_prediction.py` | none |
| ACTIVE | `LungInsight_v2/backend/Imaging/05_gradcam.py` | none |
| ACTIVE | `LungInsight_v2/backend/Imaging/06_full_ct_gradcam.py` | none |
| ACTIVE | `LungInsight_v2/backend/Imaging/07_visualize.py` | none |
| ACTIVE | `LungInsight_v2/backend/Imaging/attest_heart_mask_review.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/Imaging/diag_cam_structure.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/Imaging/diag_faithfulness.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/Imaging/diag_geometry_check.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/Imaging/diag_sensitivity.py` | none |
| SHARED UTILITY | `LungInsight_v2/backend/Imaging/heart_mask_review.py` | `LungInsight_v2/backend/Imaging/01_hu_and_crop.py`<br>`LungInsight_v2/backend/Imaging/02_detect_and_patch.py`<br>`LungInsight_v2/backend/Imaging/attest_heart_mask_review.py`<br>`LungInsight_v2/tests/test_heart_mask_review.py` |
| SHARED UTILITY | `LungInsight_v2/backend/Imaging/heart_rod_tracking.py` | `LungInsight_v2/backend/Imaging/01_hu_and_crop.py`<br>`LungInsight_v2/backend/Imaging/tests/test_heart_rod_tracking.py`<br>`LungInsight_v2/backups/expanding_rod_v2_20261014_074826/backend/Imaging/tests/test_heart_rod_tracking.py`<br>`LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/backend/Imaging/tests/test_heart_rod_tracking.py` |
| SHARED UTILITY | `LungInsight_v2/backend/Imaging/lunginsight_candidate_fusion.py` | `LungInsight_v2/backend/Imaging/02_detect_and_patch.py`<br>`LungInsight_v2/backend/Imaging/tests/test_lunginsight_candidate_fusion.py`<br>`LungInsight_v2/backups/expanding_rod_20261014_000528/backend/Imaging/tests/test_lunginsight_candidate_fusion.py`<br>`LungInsight_v2/backups/expanding_rod_v2_20261014_074826/backend/Imaging/tests/test_lunginsight_candidate_fusion.py`<br>`LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/backend/Imaging/tests/test_lunginsight_candidate_fusion.py` |
| EXPERIMENTAL | `LungInsight_v2/backend/Imaging/patch_06_display.py` | none |
| SHARED UTILITY | `LungInsight_v2/backend/Imaging/pipeline_schema.py` | `LungInsight_v2/backend/Imaging/01_hu_and_crop.py`<br>`LungInsight_v2/backend/Imaging/02_detect_and_patch.py`<br>`LungInsight_v2/backend/Imaging/03_nodule_classification.py`<br>`LungInsight_v2/backend/Imaging/04_malignancy_classifier.py`<br>`LungInsight_v2/backend/Imaging/04_malignancy_prediction.py`<br>`LungInsight_v2/backend/Imaging/05_gradcam.py`<br>`LungInsight_v2/backend/Imaging/06_full_ct_gradcam.py`<br>`LungInsight_v2/backend/Imaging/07_visualize.py`<br>`LungInsight_v2/backend/Imaging/attest_heart_mask_review.py`<br>`LungInsight_v2/backups/expanding_rod_20261014_000528/backend/Imaging/01_hu_and_crop.py`<br>`LungInsight_v2/backups/expanding_rod_v2_20261014_074826/backend/Imaging/01_hu_and_crop.py`<br>`LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/backend/Imaging/01_hu_and_crop.py`<br>`LungInsight_v2/tests/test_pipeline_schema.py`<br>`LungInsight_v2/validate_pipeline.py`<br>`pipeline_backup_stage05_20261013_144121/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_stage05_20261013_144121/tests/test_pipeline_schema.py`<br>`pipeline_backup_stage05_20261013_144121/validate_pipeline.py`<br>`pipeline_backup_stage05_20261013_144220/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_stage05_20261013_144220/tests/test_pipeline_schema.py`<br>`pipeline_backup_stage05_20261013_144220/validate_pipeline.py`<br>`pipeline_backup_targetfix_20261013_151619/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_targetfix_20261013_151619/tests/test_pipeline_schema.py`<br>`pipeline_backup_targetfix_20261013_151619/validate_pipeline.py` |
| SHARED UTILITY | `LungInsight_v2/backend/Imaging/stage04_shap.py` | `LungInsight_v2/backend/Imaging/04_malignancy_prediction.py`<br>`LungInsight_v2/tests/test_stage04_shap.py` |
| TEST | `LungInsight_v2/backend/Imaging/tests/test_heart_rod_tracking.py` | none |
| TEST | `LungInsight_v2/backend/Imaging/tests/test_lunginsight_candidate_fusion.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/compare_malignancy_models_validation.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/compare_malignancy_models.py` | none |
| SHARED UTILITY | `LungInsight_v2/backend/models/malignancy/frozen_attribute_common.py` | `LungInsight_v2/backend/Imaging/03_nodule_classification.py`<br>`LungInsight_v2/backend/models/malignancy/generate_frozen_attribute_features.py`<br>`LungInsight_v2/backend/models/malignancy/train_mlp_malignancy.py`<br>`LungInsight_v2/backend/models/malignancy/train_xgb_malignancy.py`<br>`pipeline_backup_20261013_142458/backend/Imaging/03_nodule_classification.py`<br>`pipeline_backup_stage07_20261013_160854/backend/Imaging/03_nodule_classification.py` |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/generate_frozen_attribute_features.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/predict_malignancy_from_attributes.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/predict_stage04_xml_calibrated.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/predict_xml_trained_malignancy_from_rankers.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/run_malignancy_rounding_median_validation.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/train_malignancy_from_ranker_outputs.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/train_malignancy_from_xml_ranker_compatible.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/train_mlp_malignancy.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/train_stage04_nodule_level_xml.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/train_stage04_paired_xml_calibration.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/train_xgb_malignancy.py` | `LungInsight_v2/backend/models/malignancy/train_mlp_malignancy.py` |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/validate_stage04_xml_ranker_scale_alignment.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/xml_vs_stage03_experiment/01_prepare_xml_malignancy_data.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/xml_vs_stage03_experiment/02_train_xml_malignancy.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/xml_vs_stage03_experiment/03_evaluate_continuous_vs_rounded.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/malignancy/xml_vs_stage03_experiment/04_run_diagnostics.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/frozen_encoder_probe.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/inspect_distributional_v1_sanity.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/model_3d.py` | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/train_nodule_ranker_3d.py`<br>`LungInsight_v2/scripts/diagnose_per_head.py` |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/one_stop_diagnostic.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/__init__.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/analyze_multitask_vs_singletask.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/compare_runs.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/evaluate_hybrid.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/evaluate_objective.py` | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/inspect_prediction_distribution.py` |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/gradcam_factory.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/infer_hybrid.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/infer_npy.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/inspect_prediction_distribution.py` | none |
| SHARED UTILITY | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/model_objective.py` | `LungInsight_v2/audit_gradcam_heads.py`<br>`LungInsight_v2/backend/Imaging/05_gradcam.py`<br>`LungInsight_v2/backend/models/malignancy/frozen_attribute_common.py`<br>`LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/evaluate_hybrid.py`<br>`LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/evaluate_objective.py`<br>`LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/gradcam_factory.py`<br>`LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/infer_hybrid.py`<br>`LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/infer_npy.py`<br>`LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/inspect_prediction_distribution.py`<br>`LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/train_objective.py`<br>`LungInsight_v2/compare_lobulation_spiculation_cam.py`<br>`LungInsight_v2/diagnose_lobulation_spiculation_cam.py`<br>`LungInsight_v2/inspect_gradcam_modules.py`<br>`pipeline_backup_20261013_142458/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_stage05_20261013_144121/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_stage05_20261013_144220/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_stage07_20261013_160854/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_targetfix_20261013_151619/backend/Imaging/05_gradcam.py` |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/model_spatial.py` | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/train_spatial.py` |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/train_objective.py` | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/evaluate_hybrid.py`<br>`LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/evaluate_objective.py`<br>`LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/inspect_prediction_distribution.py` |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/train_spatial.py` | none |
| TEST | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/test_distributional_v1.py` | none |
| EXPERIMENTAL | `LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/train_nodule_ranker_3d.py` | `LungInsight_v2/scripts/diagnose_per_head.py` |
| EXPERIMENTAL | `LungInsight_v2/backend/models/vitdet3d.py` | none |
| LEGACY | `LungInsight_v2/backups/expanding_rod_20261014_000528/backend/Imaging/01_hu_and_crop.py` | none |
| LEGACY | `LungInsight_v2/backups/expanding_rod_20261014_000528/backend/Imaging/lunginsight_candidate_fusion.py` | `LungInsight_v2/backend/Imaging/tests/test_lunginsight_candidate_fusion.py`<br>`LungInsight_v2/backups/expanding_rod_20261014_000528/backend/Imaging/tests/test_lunginsight_candidate_fusion.py`<br>`LungInsight_v2/backups/expanding_rod_v2_20261014_074826/backend/Imaging/tests/test_lunginsight_candidate_fusion.py`<br>`LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/backend/Imaging/tests/test_lunginsight_candidate_fusion.py` |
| TEST | `LungInsight_v2/backups/expanding_rod_20261014_000528/backend/Imaging/tests/test_lunginsight_candidate_fusion.py` | none |
| LEGACY | `LungInsight_v2/backups/expanding_rod_20261014_000528/tools/audit_lunginsight_compatibility.py` | none |
| LEGACY | `LungInsight_v2/backups/expanding_rod_20261014_000528/tools/diagnose_stage01_masks.py` | none |
| LEGACY | `LungInsight_v2/backups/expanding_rod_20261014_000528/tools/make_mask_review_bundle.py` | none |
| LEGACY | `LungInsight_v2/backups/expanding_rod_v2_20261014_074826/backend/Imaging/01_hu_and_crop.py` | none |
| LEGACY | `LungInsight_v2/backups/expanding_rod_v2_20261014_074826/backend/Imaging/heart_rod_tracking.py` | `LungInsight_v2/backend/Imaging/tests/test_heart_rod_tracking.py`<br>`LungInsight_v2/backups/expanding_rod_v2_20261014_074826/backend/Imaging/01_hu_and_crop.py`<br>`LungInsight_v2/backups/expanding_rod_v2_20261014_074826/backend/Imaging/tests/test_heart_rod_tracking.py`<br>`LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/backend/Imaging/tests/test_heart_rod_tracking.py` |
| LEGACY | `LungInsight_v2/backups/expanding_rod_v2_20261014_074826/backend/Imaging/lunginsight_candidate_fusion.py` | `LungInsight_v2/backend/Imaging/tests/test_lunginsight_candidate_fusion.py`<br>`LungInsight_v2/backups/expanding_rod_20261014_000528/backend/Imaging/tests/test_lunginsight_candidate_fusion.py`<br>`LungInsight_v2/backups/expanding_rod_v2_20261014_074826/backend/Imaging/tests/test_lunginsight_candidate_fusion.py`<br>`LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/backend/Imaging/tests/test_lunginsight_candidate_fusion.py` |
| TEST | `LungInsight_v2/backups/expanding_rod_v2_20261014_074826/backend/Imaging/tests/test_heart_rod_tracking.py` | none |
| TEST | `LungInsight_v2/backups/expanding_rod_v2_20261014_074826/backend/Imaging/tests/test_lunginsight_candidate_fusion.py` | none |
| LEGACY | `LungInsight_v2/backups/expanding_rod_v2_20261014_074826/tools/approve_heart_mask_review.py` | none |
| LEGACY | `LungInsight_v2/backups/expanding_rod_v2_20261014_074826/tools/audit_lunginsight_compatibility.py` | none |
| LEGACY | `LungInsight_v2/backups/expanding_rod_v2_20261014_074826/tools/diagnose_stage01_masks.py` | none |
| LEGACY | `LungInsight_v2/backups/expanding_rod_v2_20261014_074826/tools/make_mask_review_bundle.py` | none |
| LEGACY | `LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/backend/Imaging/01_hu_and_crop.py` | none |
| LEGACY | `LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/backend/Imaging/heart_rod_tracking.py` | `LungInsight_v2/backend/Imaging/tests/test_heart_rod_tracking.py`<br>`LungInsight_v2/backups/expanding_rod_v2_20261014_074826/backend/Imaging/tests/test_heart_rod_tracking.py`<br>`LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/backend/Imaging/01_hu_and_crop.py`<br>`LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/backend/Imaging/tests/test_heart_rod_tracking.py` |
| LEGACY | `LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/backend/Imaging/lunginsight_candidate_fusion.py` | `LungInsight_v2/backend/Imaging/tests/test_lunginsight_candidate_fusion.py`<br>`LungInsight_v2/backups/expanding_rod_20261014_000528/backend/Imaging/tests/test_lunginsight_candidate_fusion.py`<br>`LungInsight_v2/backups/expanding_rod_v2_20261014_074826/backend/Imaging/tests/test_lunginsight_candidate_fusion.py`<br>`LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/backend/Imaging/tests/test_lunginsight_candidate_fusion.py` |
| TEST | `LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/backend/Imaging/tests/test_heart_rod_tracking.py` | none |
| TEST | `LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/backend/Imaging/tests/test_lunginsight_candidate_fusion.py` | none |
| LEGACY | `LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/tools/approve_heart_mask_review.py` | none |
| LEGACY | `LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/tools/audit_lunginsight_compatibility.py` | none |
| LEGACY | `LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/tools/diagnose_stage01_masks.py` | none |
| LEGACY | `LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/tools/make_mask_review_bundle.py` | none |
| LEGACY | `LungInsight_v2/backups/hu_guided_heart_mask_20261014_081432/tools/validate_stage01_heart_outputs.py` | none |
| EXPERIMENTAL | `LungInsight_v2/compare_lobulation_spiculation_cam.py` | none |
| EXPERIMENTAL | `LungInsight_v2/diagnose_lobulation_spiculation_cam.py` | none |
| EXPERIMENTAL | `LungInsight_v2/inspect_gradcam_modules.py` | none |
| EXPERIMENTAL | `LungInsight_v2/scripts/audit_lidc_h5.py` | none |
| EXPERIMENTAL | `LungInsight_v2/scripts/diagnose_per_head.py` | none |
| EXPERIMENTAL | `LungInsight_v2/scripts/visual_audit_lidc.py` | none |
| TEST | `LungInsight_v2/tests/test_heart_mask_review.py` | none |
| TEST | `LungInsight_v2/tests/test_pipeline_schema.py` | none |
| TEST | `LungInsight_v2/tests/test_stage02_gate_wiring.py` | none |
| TEST | `LungInsight_v2/tests/test_stage04_shap.py` | none |
| TEST | `LungInsight_v2/tests/test_stage06_records.py` | none |
| EXPERIMENTAL | `LungInsight_v2/tools/approve_heart_mask_review.py` | none |
| EXPERIMENTAL | `LungInsight_v2/tools/audit_lunginsight_compatibility.py` | none |
| EXPERIMENTAL | `LungInsight_v2/tools/diagnose_stage01_masks.py` | none |
| EXPERIMENTAL | `LungInsight_v2/tools/fix_pipeline_schema_2_10_compat.py` | none |
| EXPERIMENTAL | `LungInsight_v2/tools/make_mask_review_bundle.py` | none |
| TEST | `LungInsight_v2/tools/tests/test_fix_pipeline_schema_2_10_compat.py` | none |
| EXPERIMENTAL | `LungInsight_v2/tools/validate_stage01_heart_outputs.py` | none |
| ACTIVE | `LungInsight_v2/validate_pipeline.py` | none |
| LEGACY | `pipeline_backup_20261013_142458/backend/Imaging/01_hu_and_crop.py` | none |
| LEGACY | `pipeline_backup_20261013_142458/backend/Imaging/02_detect_and_patch.py` | none |
| LEGACY | `pipeline_backup_20261013_142458/backend/Imaging/03_nodule_classification.py` | none |
| LEGACY | `pipeline_backup_20261013_142458/backend/Imaging/04_malignancy_classifier.py` | none |
| LEGACY | `pipeline_backup_20261013_142458/backend/Imaging/04_malignancy_prediction.py` | none |
| LEGACY | `pipeline_backup_20261013_142458/backend/Imaging/05_gradcam.py` | none |
| LEGACY | `pipeline_backup_20261013_142458/backend/Imaging/06_full_ct_gradcam.py` | none |
| LEGACY | `pipeline_backup_20261013_142458/backend/Imaging/pipeline_schema.py` | `pipeline_backup_20261013_142458/backend/Imaging/01_hu_and_crop.py`<br>`pipeline_backup_20261013_142458/backend/Imaging/02_detect_and_patch.py`<br>`pipeline_backup_20261013_142458/backend/Imaging/03_nodule_classification.py`<br>`pipeline_backup_20261013_142458/backend/Imaging/04_malignancy_classifier.py`<br>`pipeline_backup_20261013_142458/backend/Imaging/04_malignancy_prediction.py`<br>`pipeline_backup_20261013_142458/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_20261013_142458/backend/Imaging/06_full_ct_gradcam.py`<br>`pipeline_backup_20261013_142458/tests/test_pipeline_schema.py`<br>`pipeline_backup_20261013_142458/validate_pipeline.py`<br>`pipeline_backup_stage05_20261013_144121/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_stage05_20261013_144121/tests/test_pipeline_schema.py`<br>`pipeline_backup_stage05_20261013_144121/validate_pipeline.py`<br>`pipeline_backup_stage05_20261013_144220/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_stage05_20261013_144220/tests/test_pipeline_schema.py`<br>`pipeline_backup_stage05_20261013_144220/validate_pipeline.py`<br>`pipeline_backup_targetfix_20261013_151619/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_targetfix_20261013_151619/tests/test_pipeline_schema.py`<br>`pipeline_backup_targetfix_20261013_151619/validate_pipeline.py` |
| TEST | `pipeline_backup_20261013_142458/tests/test_pipeline_schema.py` | none |
| LEGACY | `pipeline_backup_20261013_142458/validate_pipeline.py` | none |
| LEGACY | `pipeline_backup_stage05_20261013_144121/backend/Imaging/05_gradcam.py` | none |
| TEST | `pipeline_backup_stage05_20261013_144121/tests/test_pipeline_schema.py` | none |
| LEGACY | `pipeline_backup_stage05_20261013_144121/validate_pipeline.py` | none |
| LEGACY | `pipeline_backup_stage05_20261013_144220/backend/Imaging/05_gradcam.py` | none |
| TEST | `pipeline_backup_stage05_20261013_144220/tests/test_pipeline_schema.py` | none |
| LEGACY | `pipeline_backup_stage05_20261013_144220/validate_pipeline.py` | none |
| LEGACY | `pipeline_backup_stage07_20261013_160854/backend/Imaging/01_hu_and_crop.py` | none |
| LEGACY | `pipeline_backup_stage07_20261013_160854/backend/Imaging/02_detect_and_patch.py` | none |
| LEGACY | `pipeline_backup_stage07_20261013_160854/backend/Imaging/03_nodule_classification.py` | none |
| LEGACY | `pipeline_backup_stage07_20261013_160854/backend/Imaging/04_malignancy_classifier.py` | none |
| LEGACY | `pipeline_backup_stage07_20261013_160854/backend/Imaging/04_malignancy_prediction.py` | none |
| LEGACY | `pipeline_backup_stage07_20261013_160854/backend/Imaging/05_gradcam.py` | none |
| LEGACY | `pipeline_backup_stage07_20261013_160854/backend/Imaging/06_full_ct_gradcam.py` | none |
| LEGACY | `pipeline_backup_stage07_20261013_160854/backend/Imaging/07_visualize.py` | none |
| LEGACY | `pipeline_backup_stage07_20261013_160854/backend/Imaging/pipeline_schema.py` | `pipeline_backup_stage05_20261013_144121/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_stage05_20261013_144121/tests/test_pipeline_schema.py`<br>`pipeline_backup_stage05_20261013_144121/validate_pipeline.py`<br>`pipeline_backup_stage05_20261013_144220/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_stage05_20261013_144220/tests/test_pipeline_schema.py`<br>`pipeline_backup_stage05_20261013_144220/validate_pipeline.py`<br>`pipeline_backup_stage07_20261013_160854/backend/Imaging/01_hu_and_crop.py`<br>`pipeline_backup_stage07_20261013_160854/backend/Imaging/02_detect_and_patch.py`<br>`pipeline_backup_stage07_20261013_160854/backend/Imaging/03_nodule_classification.py`<br>`pipeline_backup_stage07_20261013_160854/backend/Imaging/04_malignancy_classifier.py`<br>`pipeline_backup_stage07_20261013_160854/backend/Imaging/04_malignancy_prediction.py`<br>`pipeline_backup_stage07_20261013_160854/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_stage07_20261013_160854/backend/Imaging/06_full_ct_gradcam.py`<br>`pipeline_backup_stage07_20261013_160854/tests/test_pipeline_schema.py`<br>`pipeline_backup_stage07_20261013_160854/validate_pipeline.py`<br>`pipeline_backup_targetfix_20261013_151619/backend/Imaging/05_gradcam.py`<br>`pipeline_backup_targetfix_20261013_151619/tests/test_pipeline_schema.py`<br>`pipeline_backup_targetfix_20261013_151619/validate_pipeline.py` |
| TEST | `pipeline_backup_stage07_20261013_160854/tests/test_pipeline_schema.py` | none |
| LEGACY | `pipeline_backup_stage07_20261013_160854/validate_pipeline.py` | none |
| LEGACY | `pipeline_backup_targetfix_20261013_151619/backend/Imaging/05_gradcam.py` | none |
| TEST | `pipeline_backup_targetfix_20261013_151619/tests/test_pipeline_schema.py` | none |
| LEGACY | `pipeline_backup_targetfix_20261013_151619/validate_pipeline.py` | none |
