# LungInsight v2

A **research-only** thoracic CT application. Upload a single chest CT series and LungInsight runs a seven-stage imaging pipeline (lung/heart masks, nodule detection, nodule attribute prediction, Brock/PanCan risk, Grad-CAM explanations, report generation), then combines the result with a stored clinical record to produce two patient-level research predictions: **LUAD-vs-LUSC histology** and **overall survival**. Results are shown in a browser viewer.

> ## ⚠️ Not a diagnostic device
> No output of this software is a pathology result or a diagnosis. No model in this repository has been validated against pathology-confirmed outcomes of LungInsight imaging data. "Elevated" / "unlikely" labels in the viewer are display conventions applied to a probability, not malignancy determinations. A low-risk display does not exclude disease, and any positive display requires review of the source images. Intended use, validation scope, and regulatory status must be defined before any clinical use.

All numbers in this README come from files in the repository. Where no evaluation exists, this document says so rather than estimating a value.

---

## Table of contents

- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Component status at a glance](#component-status-at-a-glance)
- [Architecture](#architecture)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [Models](#models)
- [Brock/PanCan](#brockpancan)
- [Measured performance](#measured-performance)
- [Workflow and API](#workflow-and-api)
- [How nodule results are displayed](#how-nodule-results-are-displayed)
- [Limitations and known discrepancies](#limitations-and-known-discrepancies)
- [Recommended validation work](#recommended-validation-work)

---

## Quick start

From the project root in PowerShell:

```powershell
cd LungInsight   # or your project root path
python -m uvicorn LungInsight_v2.backend.api.main:app --host 127.0.0.1 --port 8011
```

Then open <http://127.0.0.1:8011>.

If the entry module differs, locate it first:

```powershell
Get-ChildItem -Recurse -Filter main.py | Select-Object FullName
```

---

## Repository layout

Abridged from a scan of the working tree. Generated run outputs, dataset contents, backup copies (`*.bak_*`, `*_backup_*`), archives, and model weights are summarized rather than listed. Names that appear elsewhere in this README link to the same paths.

```text
LungInsight/                                  project root
├── datasets/                                 training and reference data (not used at run time)
│   ├── LIDC/                                 lidc_patches.h5, nodule_manifest.json, XML annotations, build_lidc_h5.py
│   ├── msk_chord_2024/                       clinical cohort for the LUAD-vs-LUSC model
│   ├── TCGA-LUAD/                            clinical cohorts for the survival model
│   └── TCGA-LUSC/
├── LungInsight_v2/
│   ├── backend/
│   │   ├── Imaging/                          imaging pipeline (Stages 01-07, Brock, API)
│   │   │   ├── api/                          app.py, analysis.py, dicom_upload.py, contract.py, openapi.json
│   │   │   ├── stages/                       01_hu_and_crop.py, 02_detect_and_patch.py,
│   │   │   │                                 03_nodule_classification.py, 04_malignancy_prediction.py
│   │   │   ├── inference/                    05_gradcam.py
│   │   │   ├── visualization/                06_full_ct_gradcam.py, 07_visualize.py
│   │   │   ├── validation/                   validate_pipeline.py, attest_heart_mask_review.py
│   │   │   ├── schemas/                      imaging_clinical_contract_v1 / v2 .schema.json
│   │   │   ├── configs/
│   │   │   ├── tests/                        test_brock.py, test_stage04_inputs.py, test_imaging_contract.py, ... (20 files)
│   │   │   ├── brock.py                      Brock/PanCan formula
│   │   │   ├── brock_inputs.py               derivation of Brock inputs from Stages 02-03 and the clinical record
│   │   │   ├── heart_rod_tracking.py         heart/central exclusion mask
│   │   │   ├── lunginsight_candidate_fusion.py
│   │   │   ├── stage02_eligibility.py        lung-in / heart-out candidate gate
│   │   │   ├── nodule_identity.py            stable_nodule_id
│   │   │   ├── NODULE_ID_SPEC.md
│   │   │   ├── attribute_contract.py
│   │   │   ├── gradcam_projection.py
│   │   │   ├── html_reports.py
│   │   │   ├── pipeline_schema.py
│   │   │   └── ...                           diag_*.py, heart-mask review helpers, logging, README.md
│   │   ├── Clinical/                         LUAD-vs-LUSC and survival models
│   │   │   ├── train_cancer_type.py, train_survival.py, training.py
│   │   │   ├── inference.py, api.py, ingestion.py, harmonization.py, audit.py, imaging.py
│   │   │   ├── models/                       clinical-research-v1.joblib
│   │   │   ├── reports/                      cancer-type-evaluation.json, survival-evaluation.json,
│   │   │   │                                 cross-cohort-evaluation.json, classifier-baseline-report.json
│   │   │   ├── schemas/                      clinical_data_dictionary_v1.json, clinical_inference_v1 / clinical_prediction_v1
│   │   │   ├── tests/
│   │   │   └── README.md
│   │   ├── models/
│   │   │   ├── vitdet3d.py                   Stage 02 detector architecture
│   │   │   ├── nodule_ranker_3d_distributional_v1/
│   │   │   │   ├── spatial_v1/               Stage 03 attribute models
│   │   │   │   │   ├── model_spatial.py, train_spatial.py, infer_hybrid.py, evaluate_hybrid.py, README.md
│   │   │   │   │   └── checkpoints/hybrid/   deployed checkpoints
│   │   │   │   │       ├── subtlety/ sphericity/ margin/ lobulation_spiculation/ texture/
│   │   │   │   │       │                     each: best.pth, history.csv, run.json, baselines.json
│   │   │   │   │       └── evaluation_reports/   validation.json, test.json
│   │   │   │   ├── frozen_encoder_probe/     probe_results.csv and embeddings
│   │   │   │   ├── one_stop_diagnostic/      ONE_STOP_SUMMARY.txt and baselines
│   │   │   │   └── split.json                patient-disjoint train / val / test split
│   │   │   └── malignancy/                   retired rankers (XGBoost, MLP, pairwise, XML-calibrated variants)
│   │   └── output/                           per-patient run artifacts (generated; see below)
│   ├── frontend/                             index.html, app.js, styles.css
│   ├── scripts/                              LIDC audits, per-head diagnostics, training shell scripts
│   ├── tools/                                Stage 01 diagnostics, heart-mask review tooling
│   ├── tests/                                pipeline schema, Stage 02 gate wiring, Stage 06 records, heart-mask review
│   ├── peripherals/                          audit scripts and cleanup / compatibility reports
│   ├── archive/
│   ├── meta_schema.json
│   ├── requirements.txt
│   ├── run_pipeline.ps1
│   ├── run_tests.ps1
│   └── validate_pipeline.py
├── baseline.py, diagnose_predictions.py      standalone analysis scripts
└── Report.md, README.md
```

Per-patient run artifacts under `backend/output/<patient_id>/` follow the stage layout:

```text
backend/output/LIDC-IDRI-0001/
├── 01 outputs (HU volumes, lung/air/heart masks, meta.json, review montages)
├── 02_detect_and_patch/            candidates.json/.csv, candidate_filter_report.json,
│                                   candidates_rejected_audit.json, detector_metadata.json, patches/
├── 03_nodule_classification/       predictions.json/.csv, classification_metadata.json
├── 04_malignancy_prediction/       malignancy_predictions.json
├── 05_explainability/              patch-level Grad-CAM
└── 06_full_ct_gradcam_rainbow/     stage06_nodule_records.json, per-attribute full-CT CAM volumes and montages
```

In the scanned tree, Stage 01 outputs for this patient appear only inside a `pre_stage01_backup_<timestamp>/` folder. Output folders also exist for `LIDC-IDRI-0002`, `0003`, `0005`, `0006`, `0007`, `0047`, and an `ACCEPTANCE-001` case, plus a `reports/` folder.

---

## Component status at a glance

| Component | Type | Deployed | Measured performance |
|---|---|---|---|
| Stage 01 lung and heart masks | Classical HU thresholding and geometry (not learned) | Yes | None (no Dice or other reference comparison) |
| Stage 02 ViTDet3D detector | 3D CNN + ViT, pretrained upstream on LUNA16 | Yes | Test accuracy 0.93, IoU 0.276, AUC 0.985 (supplied by project team; split/definition not documented); no FROC/sensitivity in the repository |
| Stage 03 attribute models | 3D CNN with spatial attention, trained on LIDC-IDRI | Yes | Near chance (mean Spearman 0.04 on test) |
| Stage 04 Brock/PanCan | Published logistic formula | Yes (only malignancy method) | Formula verified against the BTS calculator; no local outcome validation |
| Clinical LUAD-vs-LUSC | Calibrated logistic regression, sex only | Yes | Balanced accuracy 0.50; ROC AUC 0.60 |
| Clinical survival | Random survival forest, age and sex | Yes | Test C-index 0.526 |
| XGBoost / MLP / pairwise malignancy rankers | Trained on LIDC reader ratings | No (retired) | No better than a constant predictor |

Earlier versions of the project documentation described lobe segmentation, longitudinal matching, and a volume/density quantification engine. **None of these exist in the code** and they have been removed.

---

## Architecture

```text
DICOM series (one CT series)
   |
   v
[Upload and validation]  api/dicom_upload.py
   - CT modality only, one study/series/patient, >= 2 slices
   - consistent size, orientation, spacing; HU calibration metadata
   |
   v
[Stage 01] HU conversion, Z-crop, lung mask, air mask, heart/central exclusion mask, automated mask QC
   |
   v
[Stage 02] ViTDet3D sliding-window detection -> NMS -> fusion -> lung-in / heart-out gate -> top 50 -> 64^3 patches
   |
   v
[Stage 03] Six predicted LIDC-style attributes per nodule (expected rating 1-5)
   |
   v
[Stage 04] Brock/PanCan malignancy risk per nodule (or explicit abstention)
   |
   +--> [Stage 05] patch Grad-CAM per attribute
   |
   +--> [Stage 06] full-CT Grad-CAM registration and nodule records
   |
   v
[Stage 07] nodule_report.json / .csv, HTML report, visualizations
   |
   v
[Imaging contract v2]  schemas/imaging_clinical_contract_v2.schema.json
   |
   v
[Clinical inference]  LUAD-vs-LUSC and survival (independent of the Brock result)
   |
   v
[study_result JSON + browser viewer]  frontend/app.js
```

Design conventions:

- Each nodule has a `stable_nodule_id` (SHA-256 of patient, study UID, series UID, and LPS centroid; see [NODULE_ID_SPEC.md](LungInsight_v2/backend/Imaging/NODULE_ID_SPEC.md)). All stages join on this ID, never on array order.
- Coordinates are ZYX voxel indices with DICOM-LPS millimetre equivalents.
- Run states: `not_started`, `running`, `completed`, `incomplete`, `blocked`, `failed`, `interrupted`. Failures are never reported as success.

---

## Inputs

| Input | Requirements | Used by |
|---|---|---|
| CT DICOM series | CT modality with pixel data; exactly one study, series, and patient; at least 2 slices; consistent dimensions, orientation, and slice spacing; valid rescale slope/intercept. Limits: 4,096 files, 512 MiB per file, 2 GiB total | Stages 01–07 |
| `patient_id` | 1–128 characters, `[A-Za-z0-9][A-Za-z0-9._-]*` | All stages |
| Date of birth | ISO `YYYY-MM-DD`, not in the future. Age is derived at use time and never stored | Brock, survival |
| Sex | `female` or `male` for Brock; `unknown` / `other_or_unspecified` is rejected by Brock | Brock, cancer-type, survival |
| Family history of lung cancer | `yes` / `no` (`unknown` leaves Brock incomplete) | Brock |
| Medical history | Status `reported` plus a condition list; emphysema is derived only when history is `reported` | Brock |
| Smoking status, pack-years, performance status | Accepted by the clinical schema; not predictors in the deployed models | Context only |
| Baseline confirmation | `baseline_definition = initial_pathologic_diagnosis` ("CT acquired at initial pathologic diagnosis") | Clinical models abstain without it |
| Optional manual Brock values | Keyed by `stable_nodule_id`; fill unknown inputs only; a differing value needs a written correction reason | Brock |

Nodule-level Brock inputs (size, location, spiculation, type, count) are **not** entered by the user; they are derived from imaging stages (see [Brock/PanCan](#brockpancan)).

---

## Outputs

| Stage | Principal outputs |
|---|---|
| Upload | Reconstructed HU volume, direction matrix, job status |
| 01 | `volume_hu.npy`, `volume_hu_cropped.npy`, `volume_hu_masked.npy`, `volume_windowed.npy` (lung/soft-tissue/bone windows), `lung_mask.npy`, `air_mask.npy`, `heart_mask.npy`, `meta.json`, `mask_quality_report.json`, review montages |
| 02 | `vitdet_raw_detections.json`, `candidates.json/.csv`, `candidate_filter_report.json`, `candidates_rejected_audit.json`, `detector_metadata.json`, patch arrays `patch_###.npy` (64×64×64 float32) with triplanar PNGs and `patches_manifest.json` |
| 03 | `predictions.json/.csv` (six expected ratings with ordinal distribution, spread, entropy), attribute attention arrays, `classification_metadata.json` |
| 04 | Per-nodule Brock record: status, probability, logit, per-term logit contributions, inputs used, missing inputs, limitations |
| 05 | `gradcam_{attribute}_rainbow.png`, raw signed CAM `.npy`, scaled CAM, metadata, acceptance audit |
| 06 | `stage06_nodule_records.json`, CAMs registered to full-CT space |
| 07 | `nodule_report.json`, `nodule_report.csv`, `excluded_outside_lung.csv`, per-nodule explanation PNGs, HTML report |
| Contract | `imaging_clinical_contract_v2.json`: study status and counts (raw detections, fused, excluded, eligible, Brock completed/incomplete/failed/not_run), per-nodule geometry, measurements, classification, Brock, quality flags, model versions, artifact references |
| Clinical | Cancer-type probabilities, survival function and horizon probabilities (or an abstention reason), explanations, provenance, limitations |
| Study | `study_result_<run_id>.json` (`lunginsight-study-result-v1`) |

---

## Models

### Stage 01 — Lung mask and heart/central exclusion mask (not learned)

Code: [01_hu_and_crop.py](LungInsight_v2/backend/Imaging/stages/01_hu_and_crop.py), [heart_rod_tracking.py](LungInsight_v2/backend/Imaging/heart_rod_tracking.py); config in [meta_schema.json](LungInsight_v2/meta_schema.json).

- HU conversion: `HU = stored * RescaleSlope + RescaleIntercept`; padding and values below −2000 HU are set to −1000 HU.
- Lung mask: thresholds (air −950 HU, lung −320 HU), 5 mm morphological closing, 3 mm wall margin, 2,500 mm² hole limit, minimum component fraction 0.0008.
- Heart/central mask: HU-guided expanding central rod (soft tissue −80 to 500 HU, seed radius 2 mm, step 4 mm, max radius 75 mm, bounded growth), restricted to thoracic slices with ≥ 14% lung area. **A heuristic exclusion mask, not a cardiac segmentation.**
- Z-crop with 10 mm margin; three display windows (lung −600/1500, soft tissue 40/400, bone 400/1800).
- Automated gate (`validated_automated` or `blocked`): non-empty masks, zero lung/heart overlap, aligned arrays, orthonormal direction matrix, successful heart tracking. Visual review and operator attestation are optional, not a gate.

No lobes, airways, or per-lobe volumes are produced. "Upper lobe" downstream is a vertical-position proxy.

### Stage 02 — ViTDet3D nodule detector

Code: [vitdet3d.py](LungInsight_v2/backend/models/vitdet3d.py), [02_detect_and_patch.py](LungInsight_v2/backend/Imaging/stages/02_detect_and_patch.py). A dependency-light re-implementation of [rlsn/LungNoduleDetection](https://github.com/rlsn/LungNoduleDetection) (MIT). Default checkpoint `rlsn/DeTr4LungNodule`, pretrained upstream on LUNA16; **not retrained here**.

| Element | Specification |
|---|---|
| Input window | 40 × 128 × 128 voxels (Z, Y, X), one channel; sliding stride 30 × 96 × 96 |
| Normalization | `(HU + 775.657) / 962.321` (LUNA16 statistics; must not be replaced) |
| CNN stem | 3D ResNet: 7×7×7 stride-2 conv, max-pool, three stages of two basic residual blocks (64/128/256 ch; strides 1/2/2); output 256 × 3 × 8 × 8 |
| Tokens | 192 spatial tokens projected to 256-D, learned [CLS] token, learned absolute positions |
| Transformer | 6 pre-norm ViT layers, 8 heads, hidden 256, MLP 1024, GELU |
| Heads | Classification: 1 logit (nodule in window). Box: 6 values (z/y/x low and high as fractions of the window) |

Single-object-per-window detector evaluated by sliding window (not anchor-based).

Post-processing: logit threshold −5.0 (probability ≈ 0.7%); NMS at 10 mm; fusion within 3 mm, or within 8 mm and 3D box IoU ≥ 0.25; candidates kept only if the rounded center is inside the lung mask and outside the heart mask; top 50 retained (equal-weight mean of within-detector rank percentiles); 64³ patches cropped at native spacing (no resampling). Each candidate gets a coarse heuristic anatomical label that is **not** clinical anatomy.

### Stage 03 — Nodule attribute models (spatial hybrid)

Code: [model_spatial.py](LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/model_spatial.py), [spatial_v1 README](LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/README.md). Class `LungInsightSpatialObjectiveV1`.

| Element | Specification |
|---|---|
| Input | 64³ raw-HU patch; `(clip(HU, −1000, 400) + 1000 − 357.88) / 288.48` |
| Encoder | 3×3×3 conv stem (16 ch), residual blocks with GroupNorm + GELU; channels 32/64/128 with 1/2/2 blocks; stride-8 tap → 128 × 8 × 8 × 8 |
| Readout | 512 spatial tokens; task-specific query attention (96-D) with normalized 3D coordinate keys; hidden head 64; dropout 0.2 |
| Objective | `ordinal_reader`: cumulative-link ordinal loss over all valid reader ratings + confidence-weighted pairwise ranking loss (weight 0.5, median target, pair threshold 0.5) |
| Training data | LIDC-IDRI patches (`lidc_patches.h5`), patient-disjoint split: 1,859 train / 437 val / 355 test nodules |
| Deployed checkpoints | Five: `subtlety`, `sphericity`, `margin`, `lobulation_spiculation` (shared by two attributes), `texture` |
| Output | Expected rating (1–5) for subtlety, sphericity, margin, lobulation, spiculation, texture, plus ordinal distribution, spread, entropy |

Internal structure, calcification, and malignancy are unsupported. Outputs are predictions of reader-style ratings, not measurements, and uncertainty is not calibrated.

### Stages 05–06 — Grad-CAM explanations

Code: [05_gradcam.py](LungInsight_v2/backend/Imaging/05_gradcam.py), [06_full_ct_gradcam.py](LungInsight_v2/backend/Imaging/visualization/06_full_ct_gradcam.py). Not a separate model: the gradient of each expected rating with respect to `encoder.stage3.1.conv2` activations is channel-weighted and summed, trilinearly interpolated, and kept signed (no ReLU). Stage 06 registers maps to full-CT space; registration failures display as unavailable. Maps show model sensitivity, **not** causal evidence and **not** malignancy, and explain the attribute models only (Brock is explained exactly by its logit terms).

### Clinical model — LUAD vs LUSC

Code: [train_cancer_type.py](LungInsight_v2/backend/Clinical/train_cancer_type.py), [inference.py](LungInsight_v2/backend/Clinical/inference.py); artifact `models/clinical-research-v1.joblib`. Sigmoid-calibrated regularized logistic regression, selected over a random forest by grouped log loss. Trained on MSK-CHORD 2024 (6,779 patients: 5,957 LUAD, 822 LUSC) with a single feature, `sex_at_baseline`. Output is histology *conditional on LUAD or LUSC* — not cancer-vs-benign and not nodule-specific. Explanations are missingness-ablation deltas, not SHAP.

### Clinical model — overall survival

Code: [train_survival.py](LungInsight_v2/backend/Clinical/train_survival.py). Random survival forest (200 trees, min leaf 15) on age and sex from TCGA-LUAD and TCGA-LUSC (991 eligible patients, 280 events). Pathologic stage is excluded to prevent target leakage. Horizons are limited to 12, 36, and 60 months, never extrapolated, and reported without uncertainty intervals or absolute calibration.

### Retired models

[backend/models/malignancy](LungInsight_v2/backend/models/malignancy) holds XGBoost, MLP, and pairwise-linear rankers that mapped the six attributes to a LIDC reader malignancy rating. Stage 04 now rejects any method other than Brock and rejects legacy reader-rating malignancy fields. Kept for reference only.

---

## Brock/PanCan

Code: [brock.py](LungInsight_v2/backend/Imaging/brock.py), [brock_inputs.py](LungInsight_v2/backend/Imaging/brock_inputs.py), [04_malignancy_prediction.py](LungInsight_v2/backend/Imaging/stages/04_malignancy_prediction.py). Tests: [test_brock.py](LungInsight_v2/backend/Imaging/tests/test_brock.py), [test_stage04_inputs.py](LungInsight_v2/backend/Imaging/tests/test_stage04_inputs.py).

The full Brock (PanCan) model of McWilliams et al., NEJM 2013 ([DOI 10.1056/NEJMoa1214726](https://doi.org/10.1056/NEJMoa1214726)), implemented as a deterministic published-coefficient formula (`Brock_PanCan_full`, version `published-coefficients-verified-v1`). It is **not fitted to LungInsight data**.

```text
logit = -6.7892 + 0.0287(age-62) + 0.6011·F + 0.2961·H + 0.2953·E
        - 5.3854[(s/10)^-0.5 - 1.58113883]
        - 0.1276·G + 0.3770·P + 0.6581·U - 0.0824(n-4) + 0.7729·S

p = 1 / (1 + e^-logit)
```

| Symbol | Input | Encoding |
|---|---|---|
| age | `age_years` | Years, 18–120 |
| F | `sex` | 1 = female, 0 = male |
| H | `family_history_lung_cancer` | 0/1 |
| E | `emphysema` | 0/1 |
| s | `nodule_size_mm` | Longest diameter in mm, 4–30 |
| G | `nodule_type` = pure ground glass | 0/1 |
| P | `nodule_type` = part solid | 0/1 (solid is reference) |
| U | `upper_lobe` | 0/1 |
| n | `nodule_count` | Total nodules in scan, 1–100 |
| S | `spiculation` | 0/1 |

Every term's logit contribution is returned and sums exactly to the logit (an exact additive decomposition — not SHAP). Unknown, boolean-as-number, non-finite, out-of-range, or unsupported inputs **never** produce a probability: status is `incomplete` or `invalid_input`, and nothing is substituted.

### How each input is obtained

| Input | Source | Mapping | Caveat |
|---|---|---|---|
| Age, sex, family history, emphysema | Stored clinical record | Direct; unknown stays unknown | Emphysema is 0 only if medical history is `reported` and omits emphysema |
| Size | Stage 02 detector box | Longest in-plane (y, x) extent in mm | Detector estimate, not a radiologist measurement (`derived_detector_estimate`) |
| Upper lobe | Stage 02 vertical region | superior = 1, inferior = 0; middle/unknown abstains | Vertical-position proxy; no lobe segmentation |
| Spiculation | Stage 03 predicted rating | Expected rating ≥ 3.0 → 1, else 0 | Inherits Stage 03's near-chance accuracy |
| Nodule type | Stage 03 predicted texture | Round half up: 1 = ground glass, 2–3 = part solid, 4–5 = solid | Inherits Stage 03's near-chance accuracy |
| Nodule count | Lung-gated Stage 02 candidates (up to 50) | Count | Detector count; each extra candidate lowers risk by 0.0824 logits |

A manual value may fill a missing source value. A manual value that conflicts with a known source value is rejected unless a non-empty correction reason is supplied; the replaced value is recorded.

### Verification and support range

- Reference case (62-year-old female, no family history or emphysema, 10 mm solid nodule, non-upper-lobe, 4 nodules, no spiculation): official BTS calculator returned **4.49%** (retrieved 2026-10-16); LungInsight returns **4.48547%**. Unit tests also cover the logit decomposition, input validation, abstention, and the Stage 02/03→Brock mappings. This confirms the arithmetic only, not clinical validity.
- Nodules < 4 mm or > 30 mm return `invalid_input` and no risk. Stage 02 keeps at most 50 candidates (the model cap is 100).
- Derived on screening-detected nodules; transportability and calibration to this application's population are not established.
- Five of nine inputs (size, upper lobe, spiculation, type, count) are model-derived or heuristic here, so the risk is only as reliable as those derivations.

---

## Measured performance

### Stage 01 — masks

No Dice/Jaccard/Hausdorff evaluation exists. The only checks are the automated per-study gates described above.

### Stage 02 — ViTDet3D detector

**Recorded test metrics** (supplied by the project team; not found as a report file in the repository):

| Metric | Value |
|---|---:|
| Test accuracy | 0.93 |
| Test IoU | 0.276 |
| Test AUC | 0.985 |

How to read these:

- **Provenance is undocumented.** The dataset, split, window sampling, class balance, decision threshold, and whether these are upstream or LungInsight-run numbers are not recorded. Add the source file or script path here once known.
- **Accuracy and AUC** most plausibly describe the window-level classification head ("is there a nodule in this window"). Accuracy depends on the positive/negative window ratio, so 0.93 is not interpretable without it. AUC 0.985 indicates strong ranking of nodule vs non-nodule windows.
- **IoU 0.276** describes box localization (the 6-value box head) and is modest. For context, the fusion step merges proposals at 3D box IoU ≥ 0.25, so the average overlap sits only just above that setting. Because Brock's nodule size is taken from the detector box, this suggests size estimates may be imprecise (an inference, not a measured size error).
- **Not recorded:** sensitivity, FROC / CPM, false positives per scan, and localization error in millimetres. These are what determine how many false candidates reach the nodule count, and none exist.

The detector's logit threshold (−5.0) is permissive and the top 50 candidates are kept, so false positives flow into the nodule count and into Brock regardless of window-level accuracy or AUC.

**About the name and external numbers.** "ViTDet3D" here is a project-internal name for a *3D ResNet stem + small ViT* (6 layers, hidden size 256) single-object-per-window detector, re-implemented from [rlsn/LungNoduleDetection](https://github.com/rlsn/LungNoduleDetection). It is **not** the plain-ViT-backbone, windowed-attention ViTDet design (He et al., 2022), and it is not a 3D driving-perception model. Benchmark figures for those architectures (COCO box AP, nuScenes NDS/mAP, KITTI AP₃D, GFLOPs/FPS for ViT-B/L/H backbones) therefore **do not describe this detector** and are deliberately not reported here.

For the medical domain, general literature on transformer-based CT nodule detectors quotes LUNA16 CPM values roughly in the 0.86–0.92 range (mean sensitivity at 0.125–8 false positives per scan). Treat this only as a loose, **unverified** orientation: it was not measured on this checkpoint (`rlsn/DeTr4LungNodule`), was not re-verified for this project, and must not be cited as LungInsight performance. Use the upstream repository for its published LUNA16 results, and see [Recommended validation work](#recommended-validation-work) for the evaluation that is still needed.

### Stage 03 attribute models

Sources: [validation.json](LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/checkpoints/hybrid/evaluation_reports/validation.json) (437 nodules) and [test.json](LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/checkpoints/hybrid/evaluation_reports/test.json) (355 nodules). Target: per-nodule mean of LIDC reader ratings. SD ratio = prediction SD / target SD (near 0 = near-constant output).

Validation set: 437 nodules (215 with reader mean ≥ 3). Test set: 355 nodules (180 with reader mean ≥ 3). The "≥ 3" column restricts the Spearman correlation to nodules whose reader-mean rating is at least 3.

| Attribute | Spearman val | Spearman test | Spearman, reader mean ≥ 3 (val / test) | MAE (val / test) | SD ratio (val / test) | Median-baseline MAE (val) |
|---|---:|---:|---:|---:|---:|---:|
| Subtlety | −0.045 | 0.025 | 0.037 / 0.056 | 0.869 / 0.836 | 0.005 / 0.006 | 0.872 |
| Sphericity | 0.063 | 0.089 | 0.124 / 0.065 | 0.718 / 0.686 | 0.006 / 0.006 | 0.662 |
| Margin | −0.049 | −0.022 | −0.001 / −0.002 | 0.869 / 0.847 | 0.100 / 0.094 | 0.830 |
| Lobulation | 0.120 | 0.034 | 0.128 / 0.044 | 0.704 / 0.664 | 0.562 / 0.600 | 0.608 |
| Spiculation | 0.122 | −0.050 | 0.170 / 0.006 | 0.658 / 0.636 | 0.562 / 0.575 | 0.541 |
| Texture | 0.034 | 0.075 | 0.069 / 0.089 | 0.835 / 0.795 | 0.120 / 0.132 | 0.636 |
| **Mean of six (M6)** | **0.041** | **0.025** | **0.088 / 0.043** | | | |

The median-baseline MAE comes from the frozen-encoder probe diagnostic ([probe_results.csv](LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/frozen_encoder_probe/probe_results.csv)) on the same validation split.

**Reading the table**

- All rank correlations are near zero; none is a useful signal. The best single value (spiculation, 0.170 on the ≥ 3 subset, validation) falls to 0.006 on test. No confidence intervals were computed, and validation→test sign changes (e.g. spiculation 0.122 → −0.050) are consistent with noise.
- Subtlety and sphericity predictions are essentially constant (SD ratio ≈ 0.006); margin (≈ 0.10) and texture (≈ 0.12–0.13) have little spread. Only lobulation and spiculation (SD ratio ≈ 0.56–0.60) vary meaningfully, yet they show no rank signal.
- On validation, MAE is worse than a constant median predictor for five of six attributes (subtlety is marginally better: 0.869 vs 0.872). The largest gaps are texture (0.835 vs 0.636) and spiculation (0.658 vs 0.541).
- Because spiculation and texture feed Brock, the Brock risk inherits this unreliability.
- Diagnostics ([one_stop_diagnostic](LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/one_stop_diagnostic/ONE_STOP_SUMMARY.txt)) show reader disagreement on 53.9% of nodules with ≥ 2 readers, a best handcrafted-feature ridge validation Spearman of 0.115, and no gain from a frozen pretrained-encoder probe.

### Brock/PanCan

| Check | Result |
|---|---|
| BTS calculator reference case | 4.49% (BTS) vs 4.48547% (LungInsight) |
| Logit equals sum of contributions | Exact (tested to 12 decimals) |
| Local outcome validation (AUC, calibration) | Not available — no pathology-labelled LungInsight imaging cohort exists |

### Clinical models

**LUAD vs LUSC** ([cancer-type-evaluation.json](LungInsight_v2/backend/Clinical/reports/cancer-type-evaluation.json)); patient-grouped split, 4,067 train / 1,356 calibration / 1,356 test (1,191 LUAD, 165 LUSC).

| Metric | Model | Prevalence-only baseline |
|---|---:|---:|
| Balanced accuracy | 0.500 (95% CI 0.500–0.500) | 0.500 |
| Macro F1 | 0.468 | |
| Log loss | 0.3615 (95% CI 0.325–0.394) | 0.3703 |
| Multiclass Brier | 0.2100 | 0.2138 |
| ROC AUC (one-vs-rest) | 0.602 | |
| PR AUC LUAD / LUSC | 0.901 / 0.153 | 0.878 / 0.122 |
| LUAD precision / recall | 0.878 / 1.000 | |
| LUSC precision / recall | 0.000 / 0.000 | |

Confusion matrix `[[1191, 0], [165, 0]]`: **every test patient was classified LUAD.**

**Overall survival** ([survival-evaluation.json](LungInsight_v2/backend/Clinical/reports/survival-evaluation.json), [cross-cohort-evaluation.json](LungInsight_v2/backend/Clinical/reports/cross-cohort-evaluation.json)); 793 development / 198 test patients (57 events).

| Metric | Value |
|---|---:|
| Test C-index | 0.526 |
| Validation C-index | 0.471 (Kaplan-Meier tie baseline 0.500) |
| Time-dependent AUC at 12 / 36 / 60 months | 0.565 / 0.551 / 0.507 |
| Integrated Brier score (IPCW) | 0.244 |
| Test C-index by cohort | LUAD 0.567 (n = 100); LUSC 0.504 (n = 98) |
| Train LUAD → test LUSC (489 patients) | C-index 0.498 |
| Train LUSC → test LUAD (502 patients) | C-index 0.530 |

The model clears its deployment gate (test C-index > 0.5) by a small margin and has weak discrimination.

### Retired malignancy rankers (LIDC reader rating, not pathology)

| Model | Test MAE | Test accuracy | Test QWK | Test Spearman |
|---|---:|---:|---:|---:|
| XGBoost (depth 2, 10 rounds) | 0.719 | 0.479 | 0.0 | 0.055 |
| MLP (selected epoch 2) | 0.719 | 0.479 | 0.0 | 0.097 |
| Pairwise linear ranker | Validation pairwise accuracy 0.496 (train 0.574) — a coin flip | | | |

These did not beat a constant baseline, which is why Brock is the only malignancy method.

---

## Workflow and API

1. **Case entry** — save a clinical record (date of birth, sex, family history, medical history, smoking, performance status) and confirm the baseline definition.
2. **Upload** — a DICOM folder or files are validated and reconstructed into one HU volume; invalid or multi-series uploads are rejected with an explicit error.
3. **Analysis run** — the browser starts a run (`requested_malignancy_method: brock`) and polls Stage 01–07 progress, then clinical inference. Failed, blocked, or interrupted runs are shown as such; interrupted runs can be resumed.
4. **Review** — the viewer shows axial CT slices with lung overlay, a nodule table (location, size, Brock result), and per-nodule panels (Brock inputs/provenance, term contributions, predicted attributes, Grad-CAM overlays), plus a stage status table.
5. **Export** — `study_result_<run_id>.json`, the contract JSON, the Stage 07 CSV/JSON/HTML report, and artifacts are retrievable through the API.

Principal endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/imaging/dicom/upload` | Upload a DICOM series |
| `PUT/GET /api/v1/clinical/records/{patient_id}` | Save / read clinical record |
| `POST /api/v1/analysis/runs` | Start an analysis run |
| `GET /api/v1/analysis/runs/{run_id}` | Run status |
| `.../viewer`, `.../viewer/slices/{index}` | Viewer data and slices |
| `.../report`, `.../report/html`, `.../study` | Reports and study result |
| `.../nodules/{nodule_id}` | Per-nodule detail |
| `POST .../nodules/{nodule_id}/brock` | Manual Brock inputs |
| `POST .../runs/{run_id}/resume` | Resume an interrupted run |
| `GET /api/v1/clinical/status` | Clinical model status |

Study-level gating: Stage 01 and Stage 02 acceptance gates must pass for downstream stages to run, and each downstream stage re-validates the selected candidate set and its hash. Missing artifacts yield null counts or explicit errors — never zero risk or a successful state. Stage 04 incompleteness does not block attribute inference or Grad-CAM.

---

## How nodule results are displayed

The application **does not classify a nodule as malignant or benign.** There is no benign/malignant classifier and no cancer-negative training cohort. The viewer maps each nodule's Brock probability to one of three display states using a fixed browser-side threshold ([app.js](LungInsight_v2/frontend/app.js#L697): `MALIGNANCY_THRESHOLD = 0.10`, a BTS-style display setting, not a diagnosis). The backend returns only the probability.

| Brock result | Display state | What the user sees |
|---|---|---|
| Probability ≥ 10% | Elevated | Red banner "Elevated malignancy risk" with the percentage and threshold note, plus a "Likely cancer type" panel (LUAD/LUSC probabilities) |
| Probability < 10% | Unlikely | Green banner "Chances of cancer are unlikely" with the percentage and threshold note |
| `incomplete`, `invalid_input`, `unsupported`, or `error` | Not assessed | Amber banner stating Brock is unavailable (not a negative result), missing inputs, and a form to supply them |

- In all states the nodule keeps its attribute predictions, Grad-CAM overlays, geometry, provenance, and quality flags. Nodules are never dropped for low risk. Only detections failing the lung-inclusion/heart-exclusion gate, or outside the top 50, are excluded (listed in `excluded_outside_lung.csv` and the rejection audit).
- The cancer-type panel is **patient-level** — the same LUAD/LUSC result for every elevated nodule — conditional on cancer being present, and depends only on sex.
- Survival is computed from age and sex only. Neither clinical model uses imaging measurements or the Brock result.
- Both clinical models abstain (cancer type) or are blocked (survival) when baseline confirmation is missing or mismatched, or when a required feature is missing or outside training support.
- Absence is never inference: an undetected nodule, a nodule outside 4–30 mm, and a nodule with unavailable Brock are not reported as benign.

---

## Limitations and known discrepancies

- Stage 03 attribute predictions are near chance, yet two feed Brock and all six are displayed. **They should not be used as findings.**
- Nodule size is the detector box, the count is the detector count, and "upper lobe" is a vertical-position proxy. Each can shift Brock risk materially.
- The 10% display threshold is an unvalidated convention; Brock has not been calibrated to this population.
- The cancer-type model uses only sex and predicted LUAD for every test patient; the survival model has weak discrimination (C-index 0.526).
- The lung/heart masks have no accuracy evaluation. The detector has only window-level accuracy/AUC and a modest IoU (0.276) with undocumented provenance, and no FROC, sensitivity, or false-positives-per-scan. Published ViTDet, nuScenes, KITTI, or generic LUNA16 figures do not apply to this re-implemented detector.
- Stage 03 results have no confidence intervals; sign changes between validation and test indicate high variance.
- Clinical training cohorts (MSK-CHORD, TCGA) are not aligned with LungInsight imaging data, so imaging cannot be a predictor.
- A frontend row labelled "SHAP" in the stage table refers to the Brock term decomposition. The pipeline does not generate SHAP values for malignancy.
- **Documented discrepancy:** [Clinical README](LungInsight_v2/backend/Clinical/README.md) quotes a survival C-index of 0.575, while the current report file shows 0.526. The JSON report is current.

---

## Recommended validation work

1. Assemble a pathology- or follow-up-confirmed CT cohort and evaluate Brock (AUC, calibration, threshold sensitivity/specificity) with pipeline-derived inputs versus manually entered inputs.
2. Document the source and split of the reported detector metrics (accuracy 0.93, IoU 0.276, AUC 0.985), then evaluate ViTDet3D on LIDC-IDRI or LUNA16 with a FROC curve and report false positives per scan at the 50-candidate cap.
3. Compare Stage 01 masks with reference segmentations (Dice, Hausdorff).
4. Retrain Stage 03, or remove its outputs from Brock (spiculation, nodule type) until they show measurable signal; add confidence intervals.
5. Replace the one-feature cancer-type model, or hide it until it separates the classes; add clinically recorded predictors with aligned imaging data.
6. Monitor failure rates, abstention rates, and manual overrides after deployment.

---

## Summary

LungInsight v2 is a traceable, research-only CT workflow whose only malignancy estimate is a correctly implemented, published Brock formula. The data fed into that formula are model-derived and, for the attribute-based inputs, currently unreliable. The learned components have no demonstrated accuracy: the attribute models, both clinical models, and the retired rankers perform near chance, and the masks are unevaluated; the detector has only window-level accuracy/AUC (0.93 / 0.985) and a modest IoU (0.276), with no FROC.