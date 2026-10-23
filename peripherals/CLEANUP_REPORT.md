# Cleanup Report

## Summary

This pass documented the current pipeline and its verified execution/dependency
structure. It did not move or delete source, configuration, model, checkpoint,
dataset, patient-output, or backup files. This conservative outcome follows the
requirement to preserve runtime behavior and avoid deleting files when
references or external use cannot be ruled out.

The repository already had a substantial modified/deleted worktree when this
pass began, including changes to numbered imaging stages and deletions in model
and checkpoint paths. Those changes were left untouched and are not attributed
to this report. Review the pre-existing Git diff separately before accepting or
discarding it.

## Files created

- `IMAGING_PIPELINE_STRUCTURE.md` — directory map, stage purposes, orchestrated
  flow, and source/config/runtime dependency overview.
- `CLEANUP_REPORT.md` — this record of scope, cleanup decisions, and remaining
  risks.

## Moved files

None.

## Deleted files

None by this pass. No deletion report entries are applicable.

Timestamped `backups/` and repository-level `pipeline_backup_*` snapshots were
not removed. Backup-suffixed `.bak_*` files under the malignancy model area
were also preserved. They may be redundant with current source, but also
represent recoverable historical state; this pass did not establish that
removing them is safe or desirable.

## Consolidated modules

None. Existing shared components already include:

- `backend/Imaging/pipeline_schema.py` for schema loading, validation, and
  schema-derived path access.
- `backend/Imaging/heart_mask_review.py` for review/attestation helpers.
- `backend/Imaging/lunginsight_candidate_fusion.py` for candidate filtering
  and related shared logic.
- `backend/Imaging/stage04_shap.py` for Stage 04 explanation helpers.
- `backend/models/malignancy/frozen_attribute_common.py` for frozen Stage 03
  inference helpers.

No helper merge was attempted because equivalent implementations, callers, and
behavioral compatibility were not exhaustively proven. The existing direct
dependencies and purpose are documented in `IMAGING_PIPELINE_STRUCTURE.md`.

## Configuration and entrypoints

- The root `meta_schema.json` is consumed by `run_pipeline.ps1` and preferred by
  `pipeline_schema.py`; it remains the central pipeline configuration.
- `requirements.txt` remains the dependency manifest.
- `run_pipeline.ps1` is the observed end-to-end entrypoint. It calls the
  numbered stages in order and runs schema/artifact checks and test discovery.
- `validate_pipeline.py` is a validation entrypoint, not a patient inference
  pipeline.
- The separate `04_malignancy_classifier.py` is not invoked by the current
  orchestrator; it remains preserved as a separately callable alternative.

No configuration migration or entrypoint behavior changes were made.

## Safety and validation boundaries

- No files under dataset storage, `backend/output/`, model/checkpoint
  directories, or backup directories were changed by this pass.
- The numbered stages and all `meta_schema.json`-configured dependencies were
  treated as protected active pipeline components.
- No source imports were rewritten, no directories were renamed, and no
  generated artifacts were regenerated.
- The dependency graph records direct orchestrator calls and observed imports.
  It does not claim to prove the absence of external callers, manually invoked
  experiments, or dynamically configured runtime paths.
- This documentation-only pass did not run inference or regenerate any
  patient output.

## Remaining technical debt

1. The active runtime is spread across a PowerShell orchestrator, Python stage
   scripts, and one central JSON schema. Keep their relative paths and naming
   contracts synchronized if future restructuring is undertaken.
2. The `backend/models/` trees combine active inference dependencies with
   training, comparison, diagnostic, and experiment programs. Each script needs
   a separate import, subprocess, documentation, and external-workflow
   reference audit before classification as removable.
3. `04_malignancy_classifier.py` and `04_malignancy_prediction.py` are two
   distinct Stage 04 implementations. The primary orchestrated choice is
   `04_malignancy_prediction.py`; confirm ownership and intended support for
   the alternative before deprecating it.
4. The nested `backend/Imaging/meta_schema.json` is not the selected schema for
   the normal `LungInsight_v2` checkout, but fallback logic exists. Validate
   supported renamed/forked installations before removing or synchronizing
   that copy.
5. Backups, backup-suffixed source files, and experimental model outputs take
   space and can resemble stale code. Their recovery/provenance value should be
   decided by the repository owner before any cleanup.
6. Current pre-existing deletions include model/checkpoint paths. This report
   neither restores nor validates those deletions; run the project artifact
   validator and inspect the existing diff before running inference.

## Potentially questionable cleanup candidates requiring human review

No files were deleted, and no questionable deletions were made by this pass.
Before any future cleanup, review these categories with their owners:

- Timestamped `backups/` and repository-level `pipeline_backup_*` source
  snapshots.
- `*.bak_*` files in model directories.
- The alternate `backend/Imaging/04_malignancy_classifier.py`.
- The nested `backend/Imaging/meta_schema.json`.
- Research/training/diagnostic scripts under `backend/models/`, `tools/`,
  `scripts/`, and the repository root.
- All pre-existing deletions visible in `git status`, especially model and
  checkpoint paths.
