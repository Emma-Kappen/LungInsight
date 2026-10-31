# LungInsight

**Explainable lung-nodule assessment from CT imaging and clinical data**

LungInsight is a research-oriented software prototype that processes thoracic CT studies, detects and characterises candidate pulmonary nodules, calculates per-nodule Brock/PanCan malignancy-risk estimates, and presents separate patient-level clinical research predictions.

> **Research use only. Not a diagnostic device.** Outputs are not diagnoses, screening recommendations, treatment recommendations, or substitutes for radiologist/pathologist review. Clinical validity and external generalisation have not been established.

## 1. Current scope

The system keeps imaging-derived evidence and patient-level clinical research models separate until study-level reporting.

- **Imaging:** DICOM-to-HU reconstruction, lung and central-exclusion masks, ViTDet3D nodule detection, candidate gating, patch extraction, six reader-style attribute estimates, Brock risk calculation, Grad-CAM artifact generation/registration, and report visualisation.
- **Nodule malignancy risk:** the published Brock/PanCan equation is the only supported malignancy-risk method. It is calculated per nodule and retains input provenance and additive logit terms.
- **Histology:** a patient-level LUAD-versus-LUSC tabular research classifier (Histology V4). It predicts a class conditional on the case belonging to LUAD or LUSC; it is not cancer-versus-benign classification and is not a per-nodule output.
- **Survival:** a separate patient-level clinical research model. It is not part of Brock scoring.
- **Reporting:** browser-based study and per-nodule review, with explicit unavailable/failed/incomplete states and provenance.

The clinical models do not consume CT features unless a separately versioned model contract explicitly says otherwise. Histology V4 is independent of Brock eligibility; the legacy histology display remains separately Brock-gated.

## 2. Histology V4 — latest reported evaluation

The latest recorded Histology V4 experiment is `20261030_235828`. The selected estimator was a **Random Forest**.

| Evaluation item | Result |
|---|---:|
| Cohort size | 1,026 patients |
| LUAD patients | 522 |
| LUSC patients | 504 |
| Training partition | 820 patients |
| Holdout partition | 206 patients |
| Holdout ROC-AUC | **0.7978** |
| Holdout balanced accuracy | **0.7282** |
| LUAD sensitivity / recall | **0.7238** |
| LUSC sensitivity / recall | **0.7327** |
| Cross-validation ROC-AUC | **0.7525 ± 0.0102** |

The reported balanced accuracy is consistent with the mean of the two class recalls (approximately 0.7282).

### Interpretation

The holdout ROC-AUC indicates useful class-ranking signal in this evaluation cohort, while balanced accuracy and class recalls indicate that errors remain for both histology classes. The cross-validation ROC-AUC is lower than the holdout ROC-AUC; this difference should be retained in reporting rather than presenting only the more favourable holdout result.

These are internal research-evaluation results, not external validation. They do not establish calibration, clinical utility, transportability to another hospital/population, or suitability for patient management.

Only the metrics listed above are currently promoted in this README. Holdout accuracy, precision, F1, confusion-matrix counts, confidence intervals, calibration metrics, and class-specific precision should not be quoted unless confirmed from the matching run's saved evaluation artifact.

### Model and data boundary

- Target classes: `LUAD` and `LUSC`.
- Task: two-class histology classification among cases already labelled as one of those classes.
- Model family selected: Random Forest.
- The V4 feature contract contains 19 named columns (11 numeric and 8 categorical in the recorded artifact summary).
- The inference and report layer must preserve missingness/imputation status and feature-source provenance.
- Imaging-derived summaries shown beside the prediction are display-only unless explicitly present in the trained model's feature contract.

### Leakage and cohort safeguards

The histology target must not be reconstructed from predictors. In particular:

- Histologic diagnosis and ICD-O-3 histology are target-verification fields, not predictors.
- Pathological stage, treatment, treatment response, and other post-diagnosis variables must not enter the predictor set.
- Patient identifiers and cohort identifiers must not be used as predictive features.
- ICD-10 fields require value-level review because some codes may directly encode histology.
- ICD-O-3 site requires assessment for diagnosis-time availability, clinical meaning, and cohort-specific proxy effects.
- A field discovered in one class/cohort but not comparably available in the other must not be included merely to increase feature count.
- Imputation and preprocessing must be fitted using training data only and then applied unchanged to holdout/inference data.

The exact feature list, preprocessing configuration, model artifact, and evaluation JSON for the same run are the source of truth. Do not infer feature membership from a UI field or from a candidate-feature audit.

## 3. Imaging pipeline

The imaging pathway is organised into seven stages:

| Stage | Responsibility |
|---|---|
| 01 — CT preparation | DICOM validation, HU reconstruction, thoracic crop, lung mask, and conservative central/heart-exclusion mask |
| 02 — Detection | ViTDet3D is the sole nodule detector; candidate filtering, lung-tissue gating, ranking, and patch extraction |
| 03 — Attributes | Six LIDC-style reader attributes: subtlety, sphericity, margin, lobulation, spiculation, and texture |
| 04 — Brock | Per-nodule published Brock/PanCan risk calculation and auditable logit-term decomposition |
| 05 — Patch explainability | Supported attribute-model Grad-CAM generation |
| 06 — CT-space registration | Registration of per-nodule Grad-CAM maps into the source CT volume |
| 07 — Visualisation/reporting | Stable nodule identifiers, bounding boxes, artifact availability, and report outputs |

### Important implementation boundaries

- **ViTDet3D is the only active detector.** There is no active LoG detector or LoG fusion stream.
- Candidates outside lung tissue must be excluded from downstream processing.
- Patch contract: `64 × 64 × 64`, `ZYX`, `float32`.
- The central/heart mask is a conservative heuristic exclusion mask, not anatomical heart segmentation.
- The lung mask is a preprocessing mask, not a clinically validated anatomical segmentation.
- Every reportable nodule should retain a stable identifier and bounding box.
- Missing or failed artifacts must remain explicitly unavailable; they must not be represented as negative findings.

## 4. Brock/PanCan risk

Brock is a published deterministic equation applied per nodule. It is not trained on LungInsight data and is not locally calibrated.

The output depends on the quality and provenance of its inputs. Several inputs may be derived from detector geometry or Stage 03 model estimates; therefore, correct formula implementation does not establish end-to-end clinical validity.

The Brock explanation is the exact additive decomposition of the equation's logit. **It is not SHAP.** Do not label Brock terms as SHAP values.

Unknown or invalid required inputs must produce an incomplete/invalid state rather than a fabricated probability. Manual corrections must retain source and correction rationale.

## 5. Other model evidence and limitations

- Stage 03 attribute evaluation has previously shown near-zero average test Spearman correlation (approximately 0.025 in the detailed evaluation). These estimates are experimental and should not be treated as radiologist-confirmed findings.
- A previously documented survival evaluation reported a test C-index of 0.526 and time-dependent AUCs of 0.565, 0.551, and 0.507 at 12, 36, and 60 months. Confirm the current survival evaluation artifact before reusing those values.
- Historical detector figures (accuracy 0.93, ROC-AUC 0.985, box IoU 0.276) had incomplete provenance in prior documentation and should not be presented as independently validated detector performance.
- No expert-reference segmentation metrics, robust detector FROC/CPM results, or pathology-confirmed external clinical validation are established by the figures in this README.

## 6. Explainability and reporting

The application presents distinct evidence types:

- **Grad-CAM:** model-sensitivity visualisation for supported Stage 03 attributes. It is not a segmentation mask, causal explanation, or proof of malignancy.
- **Brock logit terms:** exact contributions from the published equation. These are not SHAP.
- **Histology V4 SHAP:** class-specific probability explanations when a valid V4 SHAP artifact is returned. Contributions are model-attribution values, not causal effects.
- **Input provenance:** observed, imputed, manually entered, derived, missing, and unavailable values must remain distinguishable.

Histology V4 is a patient-level research output and is displayed independently of the legacy Brock-gated histology section. It does not modify Brock scoring.

## 7. Local development

The project is developed on Windows and uses PowerShell. The project root used in the current development environment is:

```text
C:\Users\emkap\Documents\Projects\LungInsight
```

Activate the local virtual environment from the project root:

```powershell
. .\l-env\Scripts\Activate.ps1
```

Use the repository's current launcher and test scripts rather than relying on a stale API module path or port copied from an older checkout. Check the active script/configuration before starting the API.

Useful validation commands include:

```powershell
node --check .\frontend\app.js
```

Run the project’s documented unit/static test modes through its current `run_tests.ps1` script. An end-to-end acceptance run executes the imaging pipeline and should be treated separately from static/schema validation.

## 8. Reproducibility checklist for Histology V4

Before publishing or comparing a new result, retain:

1. Training run identifier and timestamp.
2. Cohort sources, inclusion criteria, and patient counts by class.
3. Exact train/validation/holdout split definition.
4. Ordered feature names and numeric/categorical typing.
5. Leakage/exclusion audit and diagnosis-time availability checks.
6. Preprocessing and imputation fitted on training data only.
7. Selected estimator and hyperparameters.
8. Holdout ROC-AUC, balanced accuracy, class-wise precision/recall/F1, confusion matrix, and confidence intervals when available.
9. Cross-validation fold scores and summary.
10. Calibration and external-validation results, if performed.
11. Artifact hash/version and inference-schema compatibility check.

Do not combine metrics from different runs, splits, feature sets, or cohorts into one table.

## 9. Responsible-use statement

LungInsight is a research prototype for technical demonstration and model-evaluation work. It has not been established as clinically safe or effective. All model outputs require expert interpretation and further validation. The next milestone is reproducible evaluation, external validation, reliability testing, and governance review—not a claim of clinical readiness.
