# LungInsight Clinical research pipeline

This package is a research implementation, not a clinically validated device or
a substitute for pathology, clinician assessment, or established prediction
instruments. It keeps the cancer-type and overall-survival targets separate.

## Repository findings and dataset disposition

The dataset audit is computed from the local release each time; counts are not
hard-coded. In the inspected workspace:

- `datasets/msk_chord_2024` has 24,950 patient rows and 25,040 sample rows. Its
  sample table identifies LUAD and LUSC, but also 499 broad NSCLC labels and
  other lung histologies that are not assigned to either class. Project
  authorization permits local research fitting. `CURRENT_AGE_DEID`,
  `STAGE_HIGHEST_RECORDED`, and predicted smoking fields are not treated as
  baseline observations. The ECOG timeline is longitudinal, not baseline KPS.
- The local TCGA-LUAD and TCGA-LUSC GDC XML trees contain patient and
  follow-up-only records. In the audited snapshot, 522 LUAD and 504 LUSC
  patient summaries have a direct vital-status field; follow-up-only XML files
  are not counted as patients. At initial audit, 35 patient summaries did not
  have a valid event/time pair, including four negative times. The audit
  re-evaluates those values from the files and records current exclusions.
  GDC metadata marks listed files `open`; applicable TCGA/GDC data-use and
  publication terms still require operator confirmation before training.
- TCGA-only LUAD/LUSC classifier fitting is prohibited: its cohort identity is
  target-associated and would be a source shortcut. The authorized MSK-CHORD
  classifier uses only source-recorded sex, the sole feature with a defensible
  shared inference meaning in the inspected release.
- None of the inspected source files has patient-linked LungInsight Imaging
  measurements. Imaging nodule summaries are therefore provenance-bearing
  context only, not cancer-type or survival predictors.
- The TCGA-only random survival forest was fitted locally after verifying the
  GDC metadata access class as `open`; source patient data were not written to
  reports or redistributed. The fixed patient-grouped holdout had C-index
  0.575, time-dependent AUC 0.582/0.681/0.641 at 12/36/60 months, and
  IPCW integrated Brier score 0.209. These are research-cohort internal
  results, not evidence of clinical calibration or utility.
- Cross-cohort transport diagnostics were heterogeneous: training on LUAD and
  testing on LUSC gave C-index 0.524 (12/36-month AUC 0.512/0.608);
  training on LUSC and testing on LUAD gave C-index 0.628
  (12/36/60-month AUC 0.639/0.636/0.625). The second direction's integrated
  Brier score was not estimable under the observed follow-up support. These
  cohort-transfer experiments are diagnostics, not independent health-system
  validation.

The exact field mappings and exclusions are defined in
[`schemas/clinical_data_dictionary_v1.json`](./schemas/clinical_data_dictionary_v1.json).
The audit writes aggregate counts and schema metadata only. It does not write
patient identifiers or patient-level records.

## Audit and source permissions

From the repository root with the project environment:

```powershell
.\LungInsight_v2\l-env\Scripts\python.exe -m LungInsight_v2.backend.Clinical.audit
```

The default output is `datasets/.clinical-audit/baseline-report.json`, which
is under the ignored local datasets tree. To use another local source root or
report destination:

```powershell
.\LungInsight_v2\l-env\Scripts\python.exe -m LungInsight_v2.backend.Clinical.audit `
  --datasets-root C:\approved\data `
  --output C:\approved\reports\clinical-audit.json
```

The authorized classifier workflow records authorization in its aggregate
report. Never put raw clinical files or patient-level manifests into source
control or model reports.

## Survival development

`load_tcga_survival_cohort(path, "TCGA-LUAD")` and
`load_tcga_survival_cohort(path, "TCGA-LUSC")` load one patient-summary row,
not each follow-up XML as an independent patient. Overall survival is measured
in days from initial pathologic diagnosis:

- `Dead` requires a valid `days_to_death`; it is an event.
- `Alive` is right-censored at `days_to_last_followup`, falling back to
  `days_to_last_known_alive` only when the primary field is absent.
- Unknown status, negative/invalid time, contradictory alive/death data, and
  missing event/follow-up time are excluded, not imputed.

The initial predictor set is diagnosis age and recorded sex. Pathologic stage
is retained only as a source ground-truth label and is excluded from inference
to prevent target leakage. Direct KPS is sparse; ECOG is not silently treated as exact KPS;
smoking source codes lack a verified codebook. Treatment and future events are
not predictors. The endpoint compatibility is specific to TCGA; MSK OS is not
pooled.

Example research fit (requires explicit permission review):

```python
from pathlib import Path
import pandas as pd
from LungInsight_v2.backend.Clinical.ingestion import load_tcga_survival_cohort
from LungInsight_v2.backend.Clinical.training import (
    fit_survival_model, save_artifact_bundle, write_report,
)

root = Path("datasets")
rows = pd.concat([
    load_tcga_survival_cohort(root / "TCGA-LUAD", "TCGA-LUAD"),
    load_tcga_survival_cohort(root / "TCGA-LUSC", "TCGA-LUSC"),
], ignore_index=True)
artifact, report = fit_survival_model(rows, use_authorized=True, random_state=42)
save_artifact_bundle(
    "LungInsight_v2/backend/Clinical/models/clinical-research-v1.joblib",
    survival=artifact,
    provenance={"cohorts": ["TCGA-LUAD", "TCGA-LUSC"]},
)
write_report(
    "LungInsight_v2/backend/Clinical/reports/survival-evaluation.json",
    report,
)
```

The survival candidate is a random survival forest with right censoring. It
uses patient-grouped train/validation/test partitions; source and event strata
are kept represented in each partition where feasible. The holdout test set is
not used for fitting, feature preprocessing, or model selection. It reports
Harrell concordance, time-dependent AUC and IPCW Brier score when supported by
follow-up. Horizons are restricted to 12/36/60-month candidates with adequate
development-cohort risk sets and are never extrapolated. No interval estimate
is emitted. Internal performance is not transportability or clinical
validation.

## Cancer-type classifier

`build_cancer_type_eligibility_report` creates an aggregate, identifier-free
eligibility report before fitting. The audit writes the current decision to
`datasets/.clinical-audit/baseline-report.json`; it includes source-specific
class counts, required-feature completeness, repeated-patient counts,
source-target shortcut diagnostics, authorization status, and explicit
blockers.

`train_cancer_type` trains the authorized MSK-scoped classifier reproducibly:

```powershell
.\LungInsight_v2\l-env\Scripts\python.exe -m `
  LungInsight_v2.backend.Clinical.train_cancer_type --datasets-root datasets
```

It maps only patient-level labels resolved from explicit MSK sample
`ONCOTREE_CODE` and `CANCER_TYPE_DETAILED` fields, excludes broad NSCLC,
conflicting, and other lung histologies, and trains one row per patient. The
current compact contract contains only `sex_at_baseline` from MSK `GENDER`;
current de-identified age, highest-recorded stage, NLP smoking prediction, and
longitudinal ECOG are excluded because they are not diagnosis-baseline-aligned.

The workflow compares a regularized logistic-regression baseline and a random
forest in grouped development folds, selects by grouped log loss, calibrates the
selected estimator on a separate natural-prevalence patient partition, and
reports final held-out metrics. It atomically saves the classifier beside the
existing survival artifact, reloads it, and requires identical fixed-example
probabilities before success. The report is written to
`backend/Clinical/reports/cancer-type-evaluation.json`.

The saved probabilities are conditional on LUAD/LUSC cancer cases in the
MSK-CHORD research population; they are not cancer-versus-benign probabilities.
The single-feature model has limited discrimination and is research-only, not
an externally validated or clinically deployable histology classifier.

## Deployed models

The API loads `models/clinical-research-v1.joblib` automatically when it exists
(or the bundle named by `LUNGINSIGHT_CLINICAL_ARTIFACT`), and analysis runs use
it. Retrain either model from the repository root; each command preserves the
other model in the bundle, reloads the saved artifact and checks that fixed
examples score identically before succeeding:

```powershell
.\LungInsight_v2\l-env\Scripts\python.exe -m LungInsight_v2.backend.Clinical.train_cancer_type --datasets-root datasets
.\LungInsight_v2\l-env\Scripts\python.exe -m LungInsight_v2.backend.Clinical.train_survival --datasets-root datasets
```

`train_survival` fits the TCGA-LUAD/LUSC random survival forest on age and sex
(stage is excluded as target leakage), refuses to deploy if the held-out
concordance index is not above 0.5, and rewrites
`reports/survival-evaluation.json` and `reports/cross-cohort-evaluation.json`.
Both models only predict when the run is submitted with the "CT acquired at
initial pathologic diagnosis" confirmation (`baseline_definition` of
`initial_pathologic_diagnosis`); otherwise they report why they abstained.

### LUAD/LUSC v2 histology model

`models/cancer-type-luad-lusc-v2.joblib` is a standalone scikit-learn
`Pipeline` (not a bundle) that is adapted at load time into the bundle-style
`{metadata, estimator}` contract; its features, classes and trained categories
are verified against the documented contract and it is never retrained or
widened. It predicts histology conditional on LUAD/LUSC from three inputs and
is research-only: it does not detect cancer, separate cancer from benign
disease, establish a diagnosis, or assess an individual nodule.

| Model feature | Clinical input | Mapping |
| --- | --- | --- |
| `sex_at_baseline` | `sex_at_baseline` | `female`, `male` only |
| `smoking_history` | `smoking_status` | `never` → `never`; `former`/`current` → `former_or_current` |
| `stage` | `stage_at_baseline` | `stage_1_3` or `stage_4` only (clinician-entered, never inferred from CT, nodule size or Brock) |

`stage_at_baseline` accepts `stage_1_3`, `stage_4`, `unknown`, `not_reported`
or null; anything else (for example `Stage IIA`) is rejected. Unknown or
unreported sex, smoking or stage, other sex values, and any baseline other than
`initial_pathologic_diagnosis` make the model return `status: "abstained"` with
a `reason` and `missing_features`; the estimator is not called and nothing is
imputed. Existing records without a stage still load and simply abstain.

Artifact precedence (`load_deployed_artifacts`, and `create_app(path)`): a
`cancer_type` in a bundle named by `LUNGINSIGHT_CLINICAL_ARTIFACT` always wins;
otherwise (default bundle or no bundle) the v2 model is used when its file
exists and supersedes the default bundle's sex-only `cancer_type`. The survival
artifact is always taken from the bundle untouched. Both load paths are cached by
file modification time, an unreadable or mismatched v2 file yields a `failed`
cancer-type result (status `invalid`) without affecting survival, and
`/api/v1/clinical/status` reports the actual cancer-type and survival state.

## Clinical explainability

Classifier explanations use the fitted feature contract and report
training-fitted missingness-ablation deltas for each LUAD/LUSC probability.
Survival explanations are reported separately at each supported horizon as
observed-minus-ablated survival-probability deltas. These are local
model-sensitivity explanations, not causal effects and not SHAP values; the
calibrated composite classifier is not silently presented as a SHAP-compatible
estimator. Each explanation identifies the prediction target, model version,
features used or missing, and its limitations.

## KPS handling

Direct KPS takes precedence and is retained as an exact recorded integer.
Where direct KPS is absent, optional ECOG conversion is recorded as a
coarse approximate interval: 0→90–100, 1→70–80, 2→50–60, 3→30–40,
4→10–20, 5→0. The converted interval is never represented as an exact KPS
score. If both are supplied and direct KPS falls outside the ECOG interval,
the direct KPS is retained and the discrepancy is flagged. Mapping version:
`ecog-kps-interval-v1`.

## Imaging and inference integration

The active Imaging contract is
[`imaging_clinical_contract_v2.schema.json`](../Imaging/schemas/imaging_clinical_contract_v2.schema.json).
It includes a validated study-status summary, persisted detector/fusion/exclusion
counts, preprocessing and reviewed-mask status, warnings/errors, provenance,
artifact references, and eligible-only stable nodule records with coordinate
transforms, measurements, classification, Brock status/input completeness,
quality, and explanation references. Missing source artifacts remain unknown;
they are not converted to zero counts or successful states. Clinical rejects
contracts from incomplete/failed processing or without the reviewed Stage 01
mask gate.

The interface reports the eligible nodule-record count and largest non-null
detector-derived diameter estimate. It does not infer confirmed diagnosis from
Stage 04 risk scores and reports diagnosis/pathology as unavailable. Other Imaging
feature mappings require paired, validated training examples before use as
predictors.

Clinician inputs and results are versioned in
[`schemas/clinical_inference_v1.schema.json`](./schemas/clinical_inference_v1.schema.json)
and [`schemas/clinical_prediction_v1.schema.json`](./schemas/clinical_prediction_v1.schema.json).
The survival model's time origin and predictor baseline are initial pathologic
diagnosis. A caller must explicitly confirm that the imaging study timestamp
is that same baseline; otherwise the model returns `blocked`. Equal timestamps
alone do not establish baseline semantics.
To start the local API (use any free port; 8010 may already be in use):

```powershell
.\LungInsight_v2\l-env\Scripts\python.exe -m uvicorn `
  LungInsight_v2.backend.Clinical.api:app --host 127.0.0.1 --port 8011
```

The API returns `not_trained` for model components without loaded artifacts and
rejects invalid, stale-by-baseline, incompatible, or blocked Imaging inputs.
Set `LUNGINSIGHT_CLINICAL_ARTIFACT` only to a trusted local joblib bundle.
Joblib artifacts can execute code when loaded; never load artifacts supplied
by a user or remote client. This development API has no authentication,
authorization, audit trail, or production privacy controls; do not expose it to
untrusted networks or use it with identifiable data outside an approved
environment.

The static clinician intake is under [`frontend/`](../../frontend/). It captures
age, sex, race, ethnicity, smoking history, calculated pack-years, cessation,
relevant history and symptoms. Unknown and not-reported values remain distinct;
the local MSK-CHORD inventory includes age, gender, race and ethnicity fields,
but its smoking class is derived and is not used as observed smoking history.
The form posts DICOM folder/file selections to `/api/v1/imaging/dicom/upload`
through the local API, shows explicit preprocessing and model-unavailable/abstention
statuses, and does not fabricate results. The upload endpoint validates one
CT series, applies DICOM rescale slope/intercept to reconstruct an HU volume,
and stores only the minimum hashed study/series provenance. A successful upload only prepares the CT for analysis. `POST /api/v1/analysis/runs`
returns 202 and the existing staged Imaging runner then executes Stages 01-07
and the Clinical stage in the background. Poll the run status endpoint for
progress; preprocessing success is not an analysis result.
Open `http://127.0.0.1:8011/` (or the port you chose) after starting the local API.

Required intake fields (marked `*` in the form): Patient ID, Age (status, and
a value when reported), Sex, Race, Ethnicity, Smoking information, Family
history of lung cancer, Symptoms (use "No symptoms" for none), and KPS (or the
"KPS unknown" checkbox, which stores `null`). Save and Submit both block when a
required field is missing and mark the field with an error. Submission also
requires a confirmed CT upload for the same Patient ID.

Upload workflow: click "Select CT folder" (or "Select files"), review the
folder name and file count, then click "Confirm and upload CT". Progress is
shown during the upload. Select only the CT series folder; multi-series uploads
are rejected.

Diagnostics: the browser records events in the "Activity log" panel
(`<details>` on the page) and in the developer console with the `[LungInsight]`
prefix. The server logs each request, validation failure, upload stage, and
unhandled exception under the `lunginsight` logger hierarchy
(`lunginsight.clinical_api`, `lunginsight.imaging_api.dicom`), written to the
uvicorn console. Each response carries an `X-Request-ID` header for correlation.

The browser starts an analysis through the backend-controlled run resource:

- `POST /api/v1/analysis/runs` with `patient_id`, `upload_job_id`, supported
  `clinician_inputs`, and optional `data_provenance`.
- `GET /api/v1/analysis/runs/{run_id}` for polling.
- `POST /api/v1/analysis/runs/{run_id}/resume` for an interrupted/blocked run
  after correcting the underlying issue and passing automated Stage 01 checks.
- `GET /api/v1/analysis/runs/{run_id}/study` and
  `/api/v1/analysis/runs/{run_id}/nodules/{nodule_id}` for persisted,
  schema-validated study/nodule records.
- `GET /api/v1/analysis/runs/{run_id}/viewer` for validated run-associated CT
  metadata, mask availability, and nodule review geometry.
- `GET /api/v1/analysis/runs/{run_id}/viewer/slices/{index}` for bounded PNG
  slice delivery. The optional `lung_overlay`, `heart_overlay`,
  `gradcam_attribute` (one attribute), repeatable `gradcam_nodule_id`
  (several nodules at once) and `gradcam_alpha` (0 to 1) query values select
  overlays from artifacts belonging to that run; arbitrary filesystem paths are not
  accepted.

The server resolves the upload job to its validated input tree and invokes the
existing runner; the browser never supplies or executes a shell command. Upload
jobs and run states are persisted as JSON under the configured local output
root so polling and audit records survive process restarts. A restart during
active inference is reported as `interrupted`, never as completed. Runner logs,
partial outputs, and failure details are retained. Stage 01 automated checks
remain hard gates for mask presence, alignment, geometry, overlap, and
tracker status. Visual review and operator attestation are optional research
QA; results record when visual review was not performed.

Run status transitions are `analysis_queued` → `stage01_running` → the ordered
`stage02_running` through `stage07_running` states → `imaging_complete` →
`clinical_running` → `completed`. Runner exceptions become `failed`. The API
records only observed stage-complete markers; a blocked run is never presented
as a completed analysis.

When Imaging completes, the frontend renders the validated contract's nodule
records, Brock statuses, and available Clinical model statuses. The CT viewer
uses slice-based PNG transport rather than sending a full volume through JSON.
It supports axial slice navigation, windowed display, lung/heart-mask overlays,
validated full-CT Grad-CAM overlays when present, bounding boxes, and nodule
selection. It is a research viewer and does not provide diagnostic
interpretation.

The upload route does not invoke the staged runner. Its response supplies an
exact `runner_command` using the preserved, validated DICOM input tree. For an
authorized operator workflow, run that command (or the equivalent):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\LungInsight_v2\run_pipeline.ps1 `
  -PatientId "<patient-id>" `
  -InputPatientPath ".\backend\output\<patient-id>\dicom_uploads\<job-id>\stage01_input\<patient-id>" `
  -NoGui
```

The Stage 01 review montage and hash-bound attestation tool remain available
for optional research QA, but neither is required to proceed. Automated
mask-quality and geometry failures still block downstream processing. The
validated DICOM instances are preserved under
`dicom_uploads\<job_id>\stage01_input\<patient-id>`.
The derived `dicom_uploads\<job_id>\volume_hu.npy` artifact remains
preprocessing evidence and is not treated as a completed Stage 01 result.

## Validation

Run the focused synthetic suite without accessing real patient records:

```powershell
.\LungInsight_v2\l-env\Scripts\python.exe -m pytest `
  LungInsight_v2\backend\Clinical\tests -q
```

The tests use synthetic patients and XML. The aggregate dataset audit and
optional research training are separate workflows; test success does not
mean real-data model validation or clinical validation.
