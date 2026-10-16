# LungInsight Clinical research pipeline

This package is a research implementation, not a clinically validated device or
a substitute for pathology, clinician assessment, or established prediction
instruments. It keeps the cancer-type and overall-survival targets separate.

## Repository findings and dataset disposition

The dataset audit is computed from the local release each time; counts are not
hard-coded. In the inspected workspace:

- `datasets/msk_chord_2024` has 24,950 patient rows and 25,040 sample rows. Its
  sample table identifies LUAD and LUSC, but also 499 broad NSCLC labels and
  other lung histologies that are not assigned to either class. The source
  license file states CC BY-NC-ND 4.0. Training is blocked pending a review of
  whether model fitting/adaptation is permitted. `CURRENT_AGE_DEID`,
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
- A TCGA-only LUAD/LUSC classifier is prohibited by the implementation:
  cohort identity reveals its histology label and would be a source shortcut.
  MSK-CHORD has more useful class mixture but is not used unless licensing and
  baseline feature compatibility have been resolved.
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

Review and record source-specific use permissions before fitting. Training
functions require `use_authorized=True`; this is an explicit operator
attestation, not an automated legal determination. Never put raw clinical
files or patient-level manifests into source control or model reports.

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

The initial predictor set is diagnosis age, recorded sex, and pathologic stage
at diagnosis. Direct KPS is sparse; ECOG is not silently treated as exact KPS;
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

`fit_cancer_type_model` is available for a future legally authorized and
baseline-compatible patient-level cohort with explicit LUAD/LUSC diagnoses.
It uses regularized logistic regression with training-only class weighting,
patient-grouped natural-prevalence calibration and a test set held out from all
fitting. It rejects source/target confounding, ambiguous labels, and infeasible
class splits. It returns probabilities conditional on LUAD/LUSC cancer cases,
not cancer-versus-benign probabilities. No cancer-type model is trained from
the currently inspected data.

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
to the local API and shows explicit model-unavailable or abstention statuses;
it does not fabricate results.
Open `http://127.0.0.1:8010/` after starting the local API to use the intake.

## Validation

Run the focused synthetic suite without accessing real patient records:

```powershell
.\LungInsight_v2\l-env\Scripts\python.exe -m pytest `
  LungInsight_v2\backend\Clinical\tests -q
```

The tests use synthetic patients and XML. The aggregate dataset audit and
optional research training are separate workflows; test success does not
mean real-data model validation or clinical validation.
