# Imaging Pipeline Structure

## Scope and source of truth

This document describes the active `LungInsight_v2` workspace. The supported
end-to-end entry point is `run_pipeline.ps1`. The project-root
`meta_schema.json` defines the stage contracts, paths, model references,
preprocessing parameters, and visualization settings. `backend/Imaging/pipeline_schema.py`
loads and validates that schema for the Python stages.

The configured input root is `datasets/LIDC/lidc_idri`; patient outputs are
under `backend/output/<patient_id>`. Those paths may contain irreplaceable
research inputs and generated artifacts. They are intentionally excluded from
the source reorganization and cleanup scope.

## Directory tree

```text
LungInsight_v2/
├── backend/
│   ├── Imaging/
│   │   ├── 01_hu_and_crop.py
│   │   ├── 02_detect_and_patch.py
│   │   ├── 03_nodule_classification.py
│   │   ├── 04_malignancy_prediction.py
│   │   ├── 04_malignancy_classifier.py
│   │   ├── 05_gradcam.py
│   │   ├── 06_full_ct_gradcam.py
│   │   ├── 07_visualize.py
│   │   ├── attest_heart_mask_review.py
│   │   ├── heart_mask_review.py
│   │   ├── heart_rod_tracking.py
│   │   ├── lunginsight_candidate_fusion.py
│   │   ├── patch_06_display.py
│   │   ├── pipeline_schema.py
│   │   ├── stage04_shap.py
│   │   └── tests/
│   ├── models/
│   │   ├── malignancy/
│   │   ├── nodule_ranker_3d_distributional_v1/
│   │   │   └── spatial_v1/
│   │   └── vitdet3d.py
│   └── output/                         # Generated patient outputs; preserve
├── backups/                            # Timestamped source snapshots; preserve
├── scripts/                            # Data audits and visual diagnostics
├── tests/                              # Pipeline contract and stage tests
├── tools/                              # Compatibility, review, and audit CLIs
├── meta_schema.json                    # Canonical pipeline configuration
├── requirements.txt
├── run_pipeline.ps1                    # End-to-end orchestrator
└── validate_pipeline.py                # Static schema/artifact validation
```

The repository also contains separate `pipeline_backup_*` directories at the
repository level. They are preserved as historical snapshots, not treated as
active source directories.

## Pipeline execution flow

```text
run_pipeline.ps1
  ├─ validate_pipeline.py
  ├─ validate_pipeline.py --check-artifacts
  ├─ tests/ and backend/Imaging/tests/
  ├─ Stage 01: 01_hu_and_crop.py
  ├─ review gate: attest_heart_mask_review.py --validate-only
  ├─ Stage 02: 02_detect_and_patch.py
  ├─ Stage 03: 03_nodule_classification.py
  ├─ Stage 04: 04_malignancy_prediction.py
  ├─ Stage 05: 05_gradcam.py
  ├─ Stage 06: 06_full_ct_gradcam.py
  └─ Stage 07: 07_visualize.py
```

Stages 02–06 also process the separate LoG diagnostic manifest where configured.
Those diagnostic candidates and outputs remain separate from authoritative
predictions. Stage 02 requires a current hash-bound review attestation before
detection. The orchestrator refuses to overwrite an existing patient output
unless explicitly asked to regenerate Stage 01 with backup, or to resume after
Stage 01.

## Stage and major-script purposes

| Stage / area | Main files | Purpose |
|---|---|---|
| Ingestion, preprocessing, volume generation, resampling | `backend/Imaging/01_hu_and_crop.py` | Reads a patient DICOM series, produces HU volumes, masks, and a cropped volume using schema-controlled metadata and paths. The actual data layout and numerical contracts are in the root schema. |
| Mask review | `backend/Imaging/heart_mask_review.py`, `attest_heart_mask_review.py`, `heart_rod_tracking.py` | Generates review materials, records and validates hash-bound operator approval, and contains heart-rod tracking support. The attestation is an execution gate, not clinical validation. |
| Detection, segmentation filters, patch generation | `backend/Imaging/02_detect_and_patch.py`, `lunginsight_candidate_fusion.py` | Runs the authoritative detector and the separate diagnostic detector path, filters candidates against masks, and emits patch manifests/arrays and audit reports. |
| Feature extraction and classification | `backend/Imaging/03_nodule_classification.py`; `backend/models/malignancy/frozen_attribute_common.py`; `backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/` | Runs the frozen spatial-hybrid model to infer the six supported Stage 03 attributes. Checkpoints and model architecture are external runtime artifacts and are not cleanup targets. |
| Malignancy/rating prediction | `backend/Imaging/04_malignancy_prediction.py`, `stage04_shap.py` | Main pipeline uses the frozen XML-trained rating model and its feature explanations. Output is an estimated LIDC reader rating, not a cancer probability or diagnosis. |
| Alternative classifier | `backend/Imaging/04_malignancy_classifier.py` | Separately callable ordinal implementation. It is distinct from the primary model and is not invoked by the current orchestrator. Retain until its independent usage and intended lifecycle are decided. |
| Ranking and model development | `backend/models/nodule_ranker_3d_distributional_v1/`, `backend/models/malignancy/` | Contains model definitions, training/inference utilities, comparisons, validations, and experiments. These are not all stages in the production orchestrator; absence from that script alone does not establish that a research script is unused. |
| Inference explanation | `backend/Imaging/05_gradcam.py`, `06_full_ct_gradcam.py` | Computes patch Grad-CAM and projects it into the CT frame, retaining the per-nodule record/report contracts. |
| Evaluation, metrics, visualization | `backend/Imaging/07_visualize.py`, `validate_pipeline.py`, `tests/`, `backend/Imaging/tests/` | Stage 07 writes mask-filtered reports/visualizations. The validator checks schema and cross-stage contracts; the test trees cover schema, gates, review, attribution, records, and imaging helpers. Model experiment scripts also emit their own metrics. |
| Orchestration | `run_pipeline.ps1` | Runs preflight validation and tests, enforces Stage 01 review and overwrite safety, calls the stages in order, verifies expected outputs, and records run status. |
| Configuration | `meta_schema.json`, `requirements.txt` | The schema centralizes paths, stage contracts, model locations, preprocessing and display constants, and supported versions. Requirements declare Python dependencies. There is no separate `configs/` directory in the current layout. |
| Supporting tools | `tools/`, `scripts/`, repository-root diagnostic scripts | Compatibility checks, review helpers, data audits, and focused diagnostics. Invoke only for their documented or explicit task; they are not implicitly part of every pipeline run. |

## Dependency overview

The direct orchestration edges are the commands in `run_pipeline.ps1`. The
primary Python dependency edges observed in source are:

```text
All active imaging stages
  └─ backend/Imaging/pipeline_schema.py
       └─ ../../meta_schema.json (preferred canonical configuration)

01_hu_and_crop.py
  └─ heart_mask_review.py

02_detect_and_patch.py
  ├─ heart_mask_review.py
  └─ lunginsight_candidate_fusion.py

03_nodule_classification.py
  └─ backend/models/malignancy/frozen_attribute_common.py
       └─ backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/

04_malignancy_prediction.py
  └─ stage04_shap.py

05_gradcam.py
  └─ schema-defined Stage 03 model/checkpoint implementation

06_full_ct_gradcam.py
  └─ schema-defined Stage 01–05 files and stage contracts

07_visualize.py
  └─ schema-defined Stage 01–06 outputs
```

The helper import relationships above are the direct source-level dependencies;
the schema supplies additional runtime file/model dependencies. In particular,
Stage 03 checkpoint groups, Stage 04 model/table paths, Stage 05 model
checkpoints, and the external DICOM input are file dependencies rather than
ordinary Python imports. `validate_pipeline.py` checks selected stage contracts
and required artifacts, but does not establish that every research script is
unused or that a full patient inference succeeds.

## Entrypoints and execution

- End-to-end run: from `LungInsight_v2`, execute `.\run_pipeline.ps1` with the
  desired patient ID and documented resume/regeneration options.
- Static contract/artifact checks: `python .\validate_pipeline.py` and
  `python .\validate_pipeline.py --check-artifacts`.
- Individual stages and tools have their own command-line interfaces; the
  orchestration script is the authoritative order for production runs.
- Test discovery commands and review/resume instructions are in `README.md`.

## Boundaries and known uncertainty

- `backend/Imaging/meta_schema.json` is an older nested schema copy. The
  repository-root schema is selected by the active orchestrator and preferred
  by the helper; do not edit or remove the nested copy without verifying
  compatibility/fallback behavior for supported checkouts.
- `backups/`, repository-level `pipeline_backup_*` snapshots, and timestamped
  `.bak_*` files are retained as recovery/provenance material.
- Training, diagnostic, validation, and experiment scripts can be manually
  invoked or referenced by external workflows. They are not candidates for
  deletion solely because the production orchestrator does not call them.
- No directory moves were made. Existing path conventions are coupled to the
  schema, PowerShell orchestration, Python imports, model artifacts, and user
  documentation.
