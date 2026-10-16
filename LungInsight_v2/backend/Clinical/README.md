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
[`imaging_clinical_contract_v1.schema.json`](../Imaging/schemas/imaging_clinical_contract_v1.schema.json).
It provides patient/study identifiers, nodule records, reported diameter,
malignancy metadata and per-nodule provenance, but has no top-level
prediction-baseline timestamp or pipeline gate status. Clinical therefore
requires a trusted integration envelope containing the unchanged v1 contract,
pipeline/model versions, a passed upstream gate and source/baseline timestamps.
The timestamps must match exactly until an approved image-freshness policy is
defined. An unwrapped v1 contract is not sufficient.

The interface reports the contract nodule-record count and largest non-null
reported diameter. It does not infer malignant status from Stage 04 scores and
always reports the malignant-nodule count as unavailable. Other Imaging
feature mappings require paired, validated training examples before use as
predictors.

Clinician inputs and results are versioned in
[`schemas/clinical_inference_v1.schema.json`](./schemas/clinical_inference_v1.schema.json)
and [`schemas/clinical_prediction_v1.schema.json`](./schemas/clinical_prediction_v1.schema.json).
The survival model's time origin and predictor baseline are initial pathologic
diagnosis. A caller must explicitly confirm that the imaging study timestamp
is that same baseline; otherwise the model returns `blocked`. Equal timestamps
alone do not establish baseline semantics.
To start the local API:

```powershell
.\LungInsight_v2\l-env\Scripts\python.exe -m uvicorn `
  LungInsight_v2.backend.Clinical.api:app --host 127.0.0.1 --port 8010
```

The API returns `not_trained` for model components without loaded artifacts and
rejects invalid, stale-by-baseline, incompatible, or blocked Imaging inputs.
Set `LUNGINSIGHT_CLINICAL_ARTIFACT` only to a trusted local joblib bundle.
Joblib artifacts can execute code when loaded; never load artifacts supplied
by a user or remote client. This development API has no authentication,
authorization, audit trail, or production privacy controls; do not expose it to
untrusted networks or use it with identifiable data outside an approved
environment.

The static clinician intake is under [`frontend/`](../../frontend/). It posts
raw DICOM folder/file selections to `/api/v1/imaging/dicom/upload` through the
local API, shows explicit preprocessing and model-unavailable/abstention
statuses, and does not fabricate results. The upload endpoint validates one
CT series, applies DICOM rescale slope/intercept to reconstruct an HU volume,
and stores only the minimum hashed study/series provenance. It deliberately
reports analysis as blocked until the existing staged Imaging runner completes
its review and model gates; preprocessing success is not an analysis result.
Open `http://127.0.0.1:8010/` after starting the local API to use the intake.

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

Stage 02 is intentionally blocked until the Stage 01 review montage has been
inspected and a hash-bound attestation has been created with
`backend\Imaging\attest_heart_mask_review.py`. The validated DICOM instances
are preserved under
`dicom_uploads\<job_id>\stage01_input\<patient-id>` for this explicit handoff.
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
