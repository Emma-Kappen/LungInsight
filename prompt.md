# Master Engineering Prompt — Build, Debug, and Complete LungInsight v2 Imaging

You are a senior ML systems engineer and medical-imaging software engineer. Your task is to inspect, debug, correct, integrate, test, and deliver the entire LungInsight v2 Imaging pipeline, Stages 01–07, in the existing project.

This is an implementation task, not merely a request for a design proposal. Work directly against the actual available project files. Deliver complete replacement files in a downloadable ZIP. Do not leave the user to assemble code snippets manually.

Prioritize medical-imaging safety, schema compatibility, model compatibility, reproducibility, and evidence-supported correctness over apparent completeness.

## 1. Workspace and observed environment

Intended workspace:

`C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2`

Project root:

`C:\Users\emkap\Documents\Projects\LungInsight`

Existing virtual environment:

`C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2\l-env`

LIDC dataset root previously observed:

`C:\Users\emkap\Documents\Projects\LungInsight\datasets\LIDC\lidc_idri`

Sample patient:

`LIDC-IDRI-0001`

Typical input layout:

`datasets\LIDC\lidc_idri\<PatientID>\<Study>\<Series>\*.dcm`

Verify all paths against the actual workspace before running anything. Do not assume that a file, model, checkpoint, archive, or external repository exists unless you inspect it.

### Observed baseline

A previous Stage 01 run produced:

- Original HU volume: `(133, 512, 512)`.
- Cropped HU volume: `(109, 512, 512)`.
- Spacing: `[2.5, 0.703125, 0.703125]` mm in ZYX order.
- Lung mask: `3,621,428` true voxels.
- Heart mask: `0` true voxels.
- Output directory: `backend\output\LIDC-IDRI-0001`.

Stage 01 originally saved `volume_hu.npy`, `volume_hu_cropped.npy`, `volume_hu_masked.npy`, `volume_windowed.npy`, `air_mask.npy`, `lung_mask.npy`, `heart_mask.npy`, and `meta.json`.

Stage 02 previously ran successfully on CPU:

- ViTDet3D: 2 raw detections before NMS and 2 authoritative candidates after NMS.
- LoG: 4,931 diagnostic proposals after NMS.
- 2 authoritative patches extracted.
- 100 LoG diagnostic patches extracted.

The existing pipeline/schema treats LoG as diagnostic-only. Do not silently promote LoG proposals to authoritative candidates.

A separate candidate-fusion audit reported 4,933 records before deduplication, 4,923 after deduplication, and zero accepted records because the heart mask was empty. This is a safety block, not proof that the candidates represent valid nodules.

The active schema has at one point been `2.10`, while the loader rejected it because `metadata_policy.supported_schema_versions` contained only `2.6`, `2.7`, and `2.8`. The loader reads the version policy from the schema; a prior attempted repair incorrectly looked for a module-level `SUPPORTED_SCHEMA_VERSIONS` variable.

Audit the actual current versions and contracts before changing them. Do not blindly apply previous generated patch bundles.

## 2. Begin with a complete inventory and baseline audit

Before editing:

1. Find the actual stage files, including download-generated duplicates such as `(1)` and `(2)`. Identify which are active in the workspace and which are backups/downloads.
2. Back up every file before replacement.
3. Inspect the complete contents of:
   - `backend\Imaging\01_hu_and_crop.py`
   - `backend\Imaging\02_detect_and_patch.py`
   - `backend\Imaging\03_nodule_classification.py`
   - `backend\Imaging\04_malignancy_prediction.py`
   - `backend\Imaging\04_malignancy_classifier.py`, if present and used
   - `backend\Imaging\05_gradcam.py`
   - `backend\Imaging\06_full_ct_gradcam.py`
   - `backend\Imaging\07_visualize.py`, or the actual Stage 07 filename
   - `backend\Imaging\pipeline_schema.py`
   - `meta_schema.json`
   - Relevant model definitions, inference helpers, tests, entry points, and build instructions.
4. Read the entire active `meta_schema.json`. Record its version, required fields, coordinate conventions, stage paths, attribute names, model/checkpoint groups, patch shapes, thresholds, rating semantics, output definitions, and version compatibility policy.
5. Trace all stage imports to their actual implementations, model artifacts, and checkpoints.
6. Run non-destructive baseline checks: JSON parsing, import and compilation checks, test discovery, dependency inspection, and output-manifest inspection.
7. Create `BASELINE_AUDIT.md` with the actual project layout, confirmed defects, observed errors, missing dependencies, and executed commands/results.

Separate confirmed defects from hypotheses. Never claim a cause is established when the supplied code or test output does not support that conclusion.

## 3. Preserve `meta_schema.json` as the canonical source of truth

The schema governs all stage-owned facts. Avoid duplicating its values as independent constants where the schema can provide them.

The supplied coordinate documentation establishes the following convention unless the active schema explicitly states otherwise:

- CT arrays use `ZYX = [z,y,x]`.
- Physical coordinates use DICOM-LPS in millimetres.
- `origin_mm` represents the physical location of the centre of voxel `[0,0,0]` in the final cropped CT volume.
- Coordinate mapping: `p_LPS_mm = origin_mm + direction_lps @ (index_zyx * spacing_zyx_mm)`.
- Columns of `direction_lps` represent the physical directions of the array Z, Y, and X axes.
- Stage 01 crops along Z without resampling. Cropping must update the physical origin without changing spacing or direction.
- Stage 02 detections use the native final-cropped CT voxel frame.
- Patches are cropped without resampling, and their metadata must preserve the mapping back to the original CT frame.

Validate all saved arrays, candidate coordinates, patch manifests, CAMs, and physical-coordinate metadata against these rules.

### Schema version migration

- Do not downgrade `meta_schema.json` merely to make older stages load.
- Do not add a version to an allowlist just to silence an error.
- The active loader reads `metadata_policy.supported_schema_versions`; inspect and honor that implementation.
- If the schema declares version `2.10`, verify that the schema fields and semantics are actually compatible with the enabled pipeline code.
- Update the policy only when compatibility is demonstrated and the migration is documented.
- If stages are incompatible, port the affected interfaces or ask for the matching project source. Do not change trained checkpoints to bypass a schema error.
- Keep regression tests for supported versions, missing schema fields, malformed policy values, and coordinate-contract mismatches.

## 4. Stage 01 — DICOM, HU conversion, masks, and NPY outputs

Use the schema-defined Stage 01 entry point and paths. Resolve patient IDs beneath the actual LIDC input tree, recursively searching study/series directories. Do not confuse study or series identifiers with patient IDs.

Preserve existing outputs and add a separately named, schema-defined heart-excluded CT volume where appropriate.

Required aligned outputs include:

- `volume_hu.npy`
- `volume_hu_cropped.npy`
- `volume_hu_masked.npy`
- `volume_windowed.npy`
- `air_mask.npy`
- `lung_mask.npy`
- `heart_mask.npy`
- `meta.json`
- A heart-excluded float32 NPY volume, if included in the final contract
- Machine-readable tracker diagnostics
- Slice-level segmentation metrics and review overlays

Do not overwrite `volume_hu_cropped.npy` with the heart-excluded volume. Existing detectors may rely on the original image. Document the intended input volume for each subsequent stage.

### Existing heart-mask failure

In the supplied implementation, these functions were implicated:

- `_fill_small_holes_2d()`: identified large enclosed holes separately from holes that could be filled.
- `compute_heart_mask()`: selected connected components from those holes and could return an empty mask if the component did not meet size/span criteria.
- `compute_lung_mask()`: used the inferred heart mask to exclude central tissue and restore other large-hole regions.

Re-find these functions and their active line numbers before editing, because the source may have changed.

The sample CT had a zero-voxel heart mask while many central-slice lung contours appeared broadly plausible. Earlier contact sheets did not conclusively reveal the exact cause.

### Implement the requested expanding Z-axis rod with HU-guided tracking

The user requests an expanding rod parallel to the CT Z-axis through the CT centre, using the attenuation difference between heart/central soft tissue and aerated lung to guide tracking and growth.

Implement a conservative, explainable 3D tracker that uses **both a spatial prior and attenuation**, without treating either as sufficient on its own.

Requirements:

1. Use a configurable central seed/rod prior, with configurable offsets and diagnostic recording of its path. Do not assume the heart always occupies one exact pixel coordinate.
2. Use HU ranges or a schema-configured soft-tissue likelihood to select growth candidates. Document the range, units, and rationale.
3. Recognize that HU ranges overlap among heart, blood, mediastinum, liver, tumours, and contrast-enhanced structures. HU is evidence, not definitive organ identity.
4. Track across adjacent slices using physical spacing and continuity constraints.
5. Allow provisional overlap with the lung mask during tracking so that a segmentation leak does not hide heart voxels from the tracker.
6. Bound radial growth and slice-to-slice movement. Prevent sudden expansion into the lungs, spine, chest wall, or upper abdominal organs.
7. Use connected components and morphology as supporting regularizers, not as the sole heart-identification mechanism.
8. Preserve valid peripheral lung regions. Confirm that heart-mask subtraction is not removing an excessive or anatomically implausible amount of lung.
9. Emit per-slice records containing seed position, HU-qualified voxels, chosen component, effective radius/area, lung overlap, continuity score, and rejection/failure reason.
10. Create review montages across all slices and machine-readable summary output.

If tracking fails, produces an implausible shape, encounters low confidence, or cannot distinguish adjacent anatomy, mark the result unverified and block downstream inference. Do not manufacture a mask simply to make the gate pass.

Test synthetic cases representing valid central soft tissue, air/fat false paths, inferior liver-like tissue, central mass ambiguity, abrupt geometry changes, no qualifying seed, empty volumes, and mask leakage.

The final masks and derived CT volume must be saved as NPY arrays. Add validation that checks shape, dtype, finite values, coordinate alignment, metadata and output existence.

## 5. Stage 02 — detector policy, merging, filtering and strict gating

The existing contract makes ViTDet3D authoritative and LoG diagnostic-only. Preserve this policy unless the user approves a documented schema migration.

Implement any research fusion as an optional audit mode, separated from production candidate generation.

A reusable merge function must:

- Have deterministic behaviour and unit tests.
- Normalize detector records to a verified common coordinate frame.
- Use physical spacing for millimetre distances, accounting for anisotropic voxels.
- Support configurable overlap and/or centre-distance thresholds.
- Avoid merging distinct nodules solely due to small box overlap.
- Preserve separate detector scores and provenance. Never average unlike scores or compare unlike score scales.
- Retain original candidate IDs, metadata, and representative geometry.
- Document conflict resolution, preferring authoritative ViTDet3D geometry where the existing contract requires it.
- Report input, union, deduplication and rejection counts.

Implement a separate `filter_detections_by_lung_mask()` operation:

- Validate CT and mask dimensions/alignment.
- Apply the canonical centre-in-lung criterion and any optional box occupancy criterion.
- Explicitly reject candidates whose centres are inside the heart exclusion mask.
- Keep rejection reasons and counts.
- Fail closed if heart exclusion is unverified or empty.
- Preserve the LoG diagnostic-only policy.

Most importantly, apply the acceptance gate **before patch extraction and manifest creation**, or make patch extraction consume only the accepted list. Rejected candidates must never reach Stage 03, Stage 04, Grad-CAM, SHAP, Stage 06, or Stage 07 as surviving nodules.

The audit JSON is not a replacement for the Stage 02 patch manifest. Never direct Stage 03 at an incompatible audit file.

Or ask the user if LoG within Lung tissue can be forwarded to later stages.

## 6. Stage 03 — preserve the actual five-checkpoint design

The known checkpoint groups are:

- `subtlety/`
- `sphericity/`
- `margin/`
- `lobulation_spiculation/`
- `texture/`

The metadata references `LungInsightSpatialObjectiveV1`, with eight radiological attribute targets. Five checkpoint groups do not imply only five predictions.

Inspect the actual checkpoint files, their metadata, model implementation, helper modules, target order, patch shape, and preprocessing.

Preserve the existing trained architectures. Do not replace them with a new model or recreate missing model definitions. Verify that expected helpers, including `frozen_attribute_common.py` and the spatial objective implementation, actually exist and match the checkpoint contract.

If a required model definition or checkpoint is missing, report its exact path and block inference. Do not fabricate outputs.

Keep authoritative predictions separate from LoG diagnostic predictions.

## 7. Stage 04 — malignancy inference, rating semantics and SHAP

Inspect both Stage 04 variants and determine which is canonical from the schema, actual model artifacts, entry points and project imports. Do not mix the joblib model interface and the alternative ordinal model interface.

Verify the actual model artifact, feature order, preprocessing, output classes, probability semantics, and prediction interface.

Do not invent a binary malignancy threshold. If the available model only predicts a LIDC-style reader rating, do not automatically relabel it as pathology-confirmed cancer.

If a validated binary classifier or schema/model-defined threshold is missing, ask for clarification before generating `malignant` and `benign` labels.

For every surviving nodule, the Stage 06 record must include the stable nodule ID, geometry, detector provenance/scores, supported attribute predictions, actual malignancy output, model/version identity, uncertainty/limitations, required schema fields, and explanation artifact references.

SHAP must explain the actual selected malignancy model, with its feature order, preprocessing, explainer configuration, and background data documented. Missing SHAP dependencies must be reported explicitly.

## 8. Stages 05–06 — Grad-CAM polarity and spatial projection

Inspect the actual trained model's forward output, score definitions, attribute directions, target indices, checkpoint metadata, and Grad-CAM scalar before changing map polarity.

Lobulation and Spiculation maps have been reported to highlight low-score regions rather than areas supporting high ratings. Determine the reason from the model's actual semantics; do not blindly invert the maps.

Requirements:

- Differentiate the score with the intended semantic meaning, as supported by the trained model.
- Record the target scalar definition and output semantics in metadata.
- Preserve the intended polarity of other attributes.
- Distinguish signed CAMs, ReLU-positive evidence and display normalization.
- Avoid min-max normalization that silently destroys signed meaning.
- Preserve the real checkpoint, target layer, input shape and preprocessing.
- Associate each CAM with the correct nodule/candidate, model and attribute.
- Never reuse patch 000's CAM for another patch.
- Project CAMs into the correct CT frame using patch offsets/origins, spacing and direction metadata.

Add tests for target selection, polarity, unchanged polarity for other attributes, wrong-patch rejection and spatial alignment.

If the relevant model implementation/checkpoint is missing, report the blocker rather than fabricating a polarity correction.

## 9. Stage 06 records and Stage 07 visualization

Every accepted nodule must have one stable identifier linking its detector record, patch manifest, attributes, malignancy output, Grad-CAM maps, SHAP results and visualization.

Stage 07 must read the same accepted records used by Stage 06. For each nodule, show:

- CT slices with the correct bounding box and stable ID.
- Feature-specific CAM overlays with verified polarity.
- The supported malignancy label and confidence/probabilities.
- The matching SHAP explanation.
- Relevant physical/voxel coordinates and links to the record/artifacts.

Rejected records may appear in a clearly separated audit report only. They must not appear as surviving nodules or receive downstream explanation artifacts.

Ensure output generation can run headlessly; an interactive GUI must be optional. Avoid filename fallback logic that associates one nodule's CAM or SHAP result with another nodule.

Validate Stage 06 records against `meta_schema.json` before writing them.

## 10. Build an orchestrator for Stages 01–07

Create or repair a coherent orchestration/entry point that runs the enabled stages in dependency order for a single patient.

It must:

1. Validate dependencies and model artifacts before a stage begins.
2. Use schema-defined paths and filenames.
3. Propagate exit codes and stop on critical failures.
4. Never proceed when the heart/lung gate is unverified.
5. Feed only accepted records into downstream inference and explanations.
6. Write a run summary with times, versions, inputs, outputs, record counts, model/checkpoint identities and failures.
7. Support safe resumption from validated intermediate artifacts, avoiding stale manifests.
8. Prevent any explanation or Stage 06/07 visualization for rejected candidates.
9. Produce predictable, schema-compliant outputs.

PowerShell commands must work from the actual Windows workspace. Verify paths and preserve exit codes. Present complete `if`/`elseif` blocks in one executable block.

## 11. Testing and validation requirements

Write and execute tests for:

**Schema and environment**
- Required schema sections/fields, version policies, legacy compatibility and malformed metadata.
- Coordinate-contract mismatches.
- Missing dependencies and model artifacts produce explicit failures.

**CT and masks**
- DICOM series selection, HU conversion, Z-crop origin updates.
- Shape, dtype, spacing, origin and direction alignment.
- Tracker seed failures, plausible tracking, air/fat false paths, inferior organ confusion, central-mass ambiguity, empty-mask cases and abrupt geometry changes.
- Protection of peripheral lung regions.
- Empty/unverified mask blocks inference.
- NPY volume/mask generation and reload validation.

**Candidates and gating**
- Empty detector lists, duplicates, distinct nearby nodules, anisotropic distances, provenance, stable ordering and score retention.
- LoG remains diagnostic-only.
- Rejected candidates never enter patch extraction or any downstream inference/explanation stage.
- Counts and rejection reasons reconcile exactly.

**Models and explanations**
- Five checkpoint groups and all model output heads preserved.
- Feature/target order and input preprocessing match actual checkpoints.
- Lobulation/Spiculation CAM polarity follows actual score semantics; other attributes remain unchanged.
- CAM projection uses correct geometry and IDs.
- SHAP explains the actual Stage 04 model.
- No binary label is invented when no validated threshold exists.

**Integration**
- Stage 06 records satisfy the schema.
- Stage 07 outputs correspond exactly to the Stage 06 records and matching artifacts.
- A synthetic end-to-end fixture runs through the stages for which real model interfaces or fixtures are available.
- A real CT/model run is attempted only where all required artifacts are present.

Report actual test commands and results. Distinguish synthetic tests from real CT/checkpoint validation. Never claim tests passed unless they were executed.

## 12. Required file deliverables

Create a downloadable ZIP with workspace-relative paths containing:

- Every complete changed source file.
- New helpers and an orchestrator as needed.
- Automated tests and synthetic fixtures.
- The canonical schema only if deliberately updated.
- `BASELINE_AUDIT.md`
- `IMPLEMENTATION_REPORT.md`
- `README.md`
- Compatibility/run reports and SHA-256 checksums.
- A backup-aware installation script or exact PowerShell installation commands.

The README must specify files changed and why, destination paths, dependencies, required external model/data artifacts, schema migration effects, test commands, Stage 01 and end-to-end commands, expected outputs, counters, limitations and rollback steps.

Verify the actual ZIP contents and internal directory structure before linking it. Do not claim an artifact exists without checking it.

## 13. Engineering rules

- Correctness and anatomical safety take priority over apparent completeness.
- Work against the actual active workspace rather than trusting previous patch bundles.
- Back up files before changing them.
- Do not fabricate checkpoints, model outputs, explanations, clinical labels, paths or test results.
- Do not retrain or replace the existing architecture unless explicitly authorized.
- Do not weaken the safety gate just to get downstream stages running.
- Do not alter schema semantics merely to silence an error.
- Report confirmed defects separately from hypotheses and untested assumptions.
- Ask targeted questions if a critical artifact or validated threshold is missing.
- Never claim clinical validation from software or synthetic testing alone.

## 14. Required final response

Return:

1. Diagnosis with actual active-file/function/line references.
2. Changes implemented for Stages 01–07 and the orchestrator.
3. Exact schema version and migration impact.
4. Downloadable ZIP and list of changed/new files.
5. Exact PowerShell commands to back up/install, activate `l-env`, run tests, preprocess LIDC and run the supported pipeline.
6. Actual tests/results, separated into synthetic and real end-to-end tests.
7. Missing dependencies, unresolved assumptions, clinical limitations and remaining blockers.

The goal is one coherent, traceable and tested Imaging pipeline—not a sequence of disconnected patches. Begin by inventorying the actual workspace and auditing the baseline. Then fix and test the interfaces incrementally and deliver the complete replacement bundle.