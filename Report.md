# LungInsight v2: Technical and Application Report

## 1. Overview and evidence basis

LungInsight v2 is a research-only thoracic CT application. A single chest CT series is uploaded, processed by a seven-stage imaging pipeline (lung/heart masks, nodule detection, nodule attribute prediction, Brock/PanCan risk, Grad-CAM explanations, report generation), and then combined with a stored clinical record to produce two patient-level clinical research predictions (LUAD-vs-LUSC histology and overall survival). Results are shown in a browser viewer.

It is not a diagnostic device. No output is a pathology result, and no model in the repository was validated against pathology-confirmed outcomes of LungInsight imaging data.

Every number below was read from a file in the repository (paths are linked or named in each section). Where no evaluation exists, the report says so instead of estimating a value. Two earlier versions of this report described lobe segmentation, longitudinal matching, and a volume/density quantification engine; none of these exist in the code and they have been removed.

### Summary of findings

| Component | Type | Deployed | Measured performance |
|---|---|---|---|
| Stage 01 lung and heart masks | Classical HU thresholding and geometry (not learned) | Yes | None (no Dice or other reference comparison) |
| Stage 02 ViTDet3D detector | 3D CNN + ViT, pretrained upstream on LUNA16 | Yes | None local; no FROC/sensitivity in the repository |
| Stage 03 attribute models | 3D CNN with spatial attention, trained on LIDC-IDRI | Yes | Near chance (mean Spearman 0.04 on test) |
| Stage 04 Brock/PanCan | Published logistic formula | Yes (only malignancy method) | Formula verified against the BTS calculator; no local outcome validation |
| Clinical LUAD-vs-LUSC | Calibrated logistic regression, sex only | Yes | Balanced accuracy 0.50; ROC AUC 0.60 |
| Clinical survival | Random survival forest, age and sex | Yes | Test C-index 0.526 |
| XGBoost / MLP / pairwise malignancy rankers | Trained on LIDC reader ratings | No (retired) | No better than a constant predictor |

## 2. Application architecture

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

Each nodule carries a `stable_nodule_id` (SHA-256 of patient, study UID, series UID, and LPS centroid; see [NODULE_ID_SPEC.md](LungInsight_v2/backend/Imaging/NODULE_ID_SPEC.md)). All stages join on this ID, never on array order. Coordinates are ZYX voxel indices with DICOM-LPS millimetre equivalents. Run states are `not_started`, `running`, `completed`, `incomplete`, `blocked`, `failed`, and `interrupted`; failures are never reported as success.

## 3. Inputs and outputs

### 3.1 Required inputs

| Input | Requirements | Used by |
|---|---|---|
| CT DICOM series | CT modality with pixel data; exactly one study, series, and patient; at least 2 slices; consistent dimensions, orientation, and slice spacing; valid rescale slope/intercept. Limits: 4,096 files, 512 MiB per file, 2 GiB total | Stages 01-07 |
| `patient_id` | 1-128 characters, `[A-Za-z0-9][A-Za-z0-9._-]*` | All stages |
| Date of birth | ISO `YYYY-MM-DD`, not in the future. Age is derived at use time and never stored | Brock, survival |
| Sex | `female` or `male` for Brock; `unknown` or `other_or_unspecified` is rejected by Brock | Brock, cancer-type, survival |
| Family history of lung cancer | `yes` / `no` (`unknown` leaves Brock incomplete) | Brock |
| Medical history | Status `reported` plus a condition list; emphysema is derived only when history is `reported` | Brock |
| Smoking status, pack-years, performance status | Accepted by the clinical schema; not predictors in the deployed models | Context only |
| Baseline confirmation | `baseline_definition = initial_pathologic_diagnosis` ("CT acquired at initial pathologic diagnosis") | Clinical models abstain without it |
| Optional manual Brock values | Keyed by `stable_nodule_id`; fill unknown inputs only; a differing value needs a written correction reason | Brock |

The nodule-level Brock inputs (size, location, spiculation, type, count) are not entered by the user; they are derived from imaging stages (Section 5.3).

### 3.2 Generated outputs

| Stage | Principal outputs |
|---|---|
| Upload | Reconstructed HU volume, direction matrix, job status |
| 01 | `volume_hu.npy`, `volume_hu_cropped.npy`, `volume_hu_masked.npy`, `volume_windowed.npy` (lung/soft-tissue/bone windows), `lung_mask.npy`, `air_mask.npy`, `heart_mask.npy`, `meta.json`, `mask_quality_report.json`, review montages |
| 02 | `vitdet_raw_detections.json`, `candidates.json/.csv`, `candidate_filter_report.json`, `candidates_rejected_audit.json`, `detector_metadata.json`, patch arrays `patch_###.npy` (64x64x64 float32) with triplanar PNGs and `patches_manifest.json` |
| 03 | `predictions.json/.csv` (six expected ratings with ordinal distribution, spread, entropy), attribute attention arrays, `classification_metadata.json` |
| 04 | Per-nodule Brock record: status, probability, logit, per-term logit contributions, inputs used, missing inputs, limitations |
| 05 | `gradcam_{attribute}_rainbow.png`, raw signed CAM `.npy`, scaled CAM, metadata, acceptance audit |
| 06 | `stage06_nodule_records.json`, CAMs registered to full-CT space |
| 07 | `nodule_report.json`, `nodule_report.csv`, `excluded_outside_lung.csv`, per-nodule explanation PNGs, HTML report |
| Contract | `imaging_clinical_contract_v2.json`: study status and counts (raw detections, fused, excluded, eligible, Brock completed/incomplete/failed/not_run), per-nodule geometry, measurements, classification, Brock, quality flags, model versions, artifact references |
| Clinical | Cancer-type probabilities, survival function and horizon probabilities (or an abstention reason), explanations, provenance, limitations |
| Study | `study_result_<run_id>.json` (`lunginsight-study-result-v1`) |


## 4. Models and architectures

### 4.1 Stage 01: lung mask and heart/central exclusion mask (not a learned model)

Code: [01_hu_and_crop.py](LungInsight_v2/backend/Imaging/stages/01_hu_and_crop.py), [heart_rod_tracking.py](LungInsight_v2/backend/Imaging/heart_rod_tracking.py), configuration in [meta_schema.json](LungInsight_v2/meta_schema.json).

- HU conversion: `HU = stored * RescaleSlope + RescaleIntercept`; padding and values below -2000 HU are set to -1000 HU.
- Lung mask: threshold-based (air at -950 HU, lung at -320 HU), morphological closing (5 mm), wall margin 3 mm, hole limit 2,500 mm2, minimum component fraction 0.0008.
- Heart/central mask: HU-guided expanding central rod (soft tissue -80 to 500 HU, seed radius 2 mm, step 4 mm, maximum radius 75 mm, bounded contour growth 5 mm and adjacent-slice area ratio 1.4), restricted to thoracic slices with at least 14% lung area. It is a heuristic exclusion mask, not a cardiac segmentation.
- Z-crop with 10 mm margin; three display windows (lung -600/1500, soft tissue 40/400, bone 400/1800).
- Automated gate (`validated_automated` or `blocked`): non-empty lung and heart masks, zero lung/heart overlap, aligned arrays, orthonormal direction matrix, successful heart tracking. Visual review and operator attestation are optional and not a gate.

No lobes, airways, or per-lobe volumes are produced. "Upper lobe" later in the pipeline is a vertical-position proxy (Section 5.3).

### 4.2 Stage 02: ViTDet3D nodule detector

Code: [vitdet3d.py](LungInsight_v2/backend/models/vitdet3d.py), [02_detect_and_patch.py](LungInsight_v2/backend/Imaging/stages/02_detect_and_patch.py). A dependency-light re-implementation of rlsn/LungNoduleDetection (MIT). Default checkpoint `rlsn/DeTr4LungNodule`, pretrained upstream on LUNA16; it was not retrained in this project.

| Element | Specification |
|---|---|
| Input window | 40 x 128 x 128 voxels (Z, Y, X), one channel; sliding stride 30 x 96 x 96 |
| Normalization | `(HU + 775.657) / 962.321` (LUNA16 statistics; must not be replaced) |
| CNN stem | 3D ResNet: 7x7x7 stride-2 conv, max-pool, three stages of two basic residual blocks (64, 128, 256 channels; strides 1, 2, 2); output 256 x 3 x 8 x 8 |
| Tokens | 192 spatial tokens projected to 256 dimensions, a learned [CLS] token, learned absolute positions |
| Transformer | 6 pre-norm ViT layers, 8 heads, hidden size 256, MLP size 1024, GELU |
| Heads | Classification: 1 logit (nodule in this window). Box: 6 values (z/y/x low and high as fractions of the window) |

This is a single-object-per-window detector evaluated by sliding window, not an anchor-based detector.

Post-processing: logit threshold -5.0 (probability about 0.7%), NMS at 10 mm; fusion merges proposals within 3 mm, or within 8 mm and 3D box IoU of at least 0.25; candidates are accepted only if the rounded center is inside the lung mask and outside the heart mask; the top 50 are kept (equal-weight mean of within-detector rank percentiles); 64 x 64 x 64 voxel patches are cropped at native spacing without resampling. Each candidate gets a coarse heuristic anatomical label (vertical region, central/peripheral, trachea/bronchus from the air mask) that is flagged as not clinical anatomy.

### 4.3 Stage 03: nodule attribute models (spatial hybrid)

Code: [model_spatial.py](LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/model_spatial.py), [spatial_v1 README](LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/README.md). Class `LungInsightSpatialObjectiveV1`.

| Element | Specification |
|---|---|
| Input | 64 x 64 x 64 raw-HU patch; `(clip(HU, -1000, 400) + 1000 - 357.88) / 288.48` |
| Encoder | 3x3x3 conv stem (16 ch), residual blocks with GroupNorm and GELU; channels 32/64/128 with 1/2/2 blocks; stride-8 tap giving 128 x 8 x 8 x 8 features |
| Readout | 512 spatial tokens; task-specific query attention (96-D) with normalized 3D coordinate keys; hidden head 64; dropout 0.2 |
| Objective | `ordinal_reader`: cumulative-link ordinal loss over all valid individual reader ratings, plus confidence-weighted pairwise ranking loss (weight 0.5, median target, pair threshold 0.5) |
| Training data | LIDC-IDRI patches (`lidc_patches.h5`), patient-disjoint split: 1,859 train / 437 validation / 355 test nodules (0 patient overlap) |
| Deployed checkpoints | Five: `subtlety`, `sphericity`, `margin`, `lobulation_spiculation` (shared by two attributes), `texture` (`hybrid/<group>/best.pth`) |
| Output | Expected rating (1-5) for subtlety, sphericity, margin, lobulation, spiculation, texture, with the ordinal distribution, spread, and entropy |

Internal structure, calcification, and malignancy are explicitly unsupported by the deployed checkpoints. Outputs are model predictions of reader-style ratings, not measurements, and uncertainty is not calibrated.

### 4.4 Stage 04: Brock/PanCan

See Section 5.

### 4.5 Stages 05 and 06: Grad-CAM explanations

Code: [05_gradcam.py](LungInsight_v2/backend/Imaging/05_gradcam.py), [06_full_ct_gradcam.py](LungInsight_v2/backend/Imaging/visualization/06_full_ct_gradcam.py). Not a separate model. For each attribute the gradient of the expected rating with respect to layer `encoder.stage3.1.conv2` activations is channel-weighted and summed, interpolated trilinearly to the patch, and kept signed (no ReLU). Stage 06 registers the maps to full-CT space. The maps indicate model sensitivity, not causal evidence and not malignancy; they explain the attribute models, not Brock (Brock is explained exactly by its logit terms). Registration failures are shown as unavailable.

### 4.6 Clinical model: LUAD-versus-LUSC histology

Code: [train_cancer_type.py](LungInsight_v2/backend/Clinical/train_cancer_type.py), [inference.py](LungInsight_v2/backend/Clinical/inference.py), artifact `models/clinical-research-v1.joblib`. Sigmoid-calibrated regularized logistic regression selected over a random forest by grouped log loss. Trained on MSK-CHORD 2024 patients (6,779: 5,957 LUAD and 822 LUSC) with one feature, `sex_at_baseline`. The output is histology conditional on the patient having LUAD or LUSC. It is not cancer-versus-benign and is not nodule-specific. Explanations are missingness-ablation deltas, not SHAP.

### 4.7 Clinical model: overall survival

Code: [train_survival.py](LungInsight_v2/backend/Clinical/train_survival.py). Random survival forest (200 trees, minimum leaf 15) on age and sex from TCGA-LUAD and TCGA-LUSC (991 eligible patients, 280 events). Pathologic stage is excluded to prevent target leakage. Horizons are limited to 12, 36, and 60 months, are never extrapolated, and are reported without uncertainty intervals or absolute calibration.

### 4.8 Retired models (not part of the pipeline)

[backend/models/malignancy](LungInsight_v2/backend/models/malignancy) contains XGBoost, MLP, and pairwise-linear rankers that mapped the six predicted attributes to a LIDC reader malignancy rating (1-5). Stage 04 now rejects any method other than Brock and rejects legacy reader-rating malignancy fields. They are retained for reference only; their metrics are in Section 6.5.

## 5. Brock/PanCan model

Code: [brock.py](LungInsight_v2/backend/Imaging/brock.py), [brock_inputs.py](LungInsight_v2/backend/Imaging/brock_inputs.py), [04_malignancy_prediction.py](LungInsight_v2/backend/Imaging/stages/04_malignancy_prediction.py), tests in [test_brock.py](LungInsight_v2/backend/Imaging/tests/test_brock.py) and [test_stage04_inputs.py](LungInsight_v2/backend/Imaging/tests/test_stage04_inputs.py).

### 5.1 What it is

The full Brock (PanCan) model of McWilliams et al., NEJM 2013 (DOI 10.1056/NEJMoa1214726), implemented as a deterministic published-coefficient formula (`Brock_PanCan_full`, version `published-coefficients-verified-v1`, formula `mcwilliams-nejmoa1214726-full-v1`). It is not fitted to LungInsight data. The target is the probability that a pulmonary nodule is malignant, as defined by the original screening population.

### 5.2 Formula

$$\text{logit} = -6.7892 + 0.0287(\text{age}-62) + 0.6011\,F + 0.2961\,H + 0.2953\,E - 5.3854\left[\left(\tfrac{s}{10}\right)^{-0.5} - 1.58113883\right] - 0.1276\,G + 0.3770\,P + 0.6581\,U - 0.0824(n-4) + 0.7729\,S$$

$$p = \frac{1}{1+e^{-\text{logit}}}$$

| Symbol | Input | Encoding |
|---|---|---|
| age | `age_years` | Years, 18-120 |
| F | `sex` | 1 = female, 0 = male |
| H | `family_history_lung_cancer` | 0/1 |
| E | `emphysema` | 0/1 |
| s | `nodule_size_mm` | Longest diameter in mm, 4-30 |
| G | `nodule_type` = pure ground glass | 0/1 |
| P | `nodule_type` = part solid | 0/1 (solid is the reference) |
| U | `upper_lobe` | 0/1 |
| n | `nodule_count` | Total nodules in the scan, 1-100 |
| S | `spiculation` | 0/1 |

The output includes every term's logit contribution; contributions sum exactly to the logit. Because the model is additive, this decomposition is exact (not SHAP, not a sampled approximation). Unknown, boolean-as-number, non-finite, out-of-range, or unsupported-category inputs never produce a probability: the status is `incomplete` or `invalid_input`, and nothing is substituted.

### 5.3 How each input is obtained

| Input | Source | Mapping | Caveat |
|---|---|---|---|
| Age, sex, family history, emphysema | Stored clinical record (age from date of birth) | Direct; unknown values stay unknown | Emphysema is 0 only if medical history is `reported` and does not list emphysema |
| Size | Stage 02 detector box | Longest in-plane (y, x) extent in mm | Detector box estimate, not a radiologist measurement; labelled `derived_detector_estimate` |
| Upper lobe | Stage 02 vertical region | superior = 1, inferior = 0; middle or unknown abstains | Vertical-position proxy; there is no lobe segmentation |
| Spiculation | Stage 03 predicted rating | Expected rating >= 3.0 gives 1, otherwise 0 | Inherits Stage 03 accuracy (Section 6.3); raw rating kept as provenance |
| Nodule type | Stage 03 predicted texture rating | Round half up: 1 = ground glass, 2-3 = part solid, 4-5 = solid | Inherits Stage 03 accuracy (Section 6.3) |
| Nodule count | Number of lung-gated Stage 02 candidates (up to 50) | Count | Detector count, not radiologist-confirmed; each extra candidate lowers risk by 0.0824 logits |

A manual value may fill a missing source value. A manual value that conflicts with a known source value is rejected unless a non-empty correction reason is supplied, and the replaced value is recorded.

### 5.4 Verification

For the reference case (62-year-old female, no family history or emphysema, 10 mm solid nodule, non-upper-lobe, 4 nodules, no spiculation) the official BTS Pulmonary Nodule Risk Calculator returned 4.49% (retrieved 2026-10-16). LungInsight returns 4.48547%, which rounds to 4.49%. Unit tests also check the logit decomposition, input validation, abstention, and the Stage 02/03-to-Brock mappings. This confirms the arithmetic only; it is not a clinical validation. The published Brock discrimination figures are not reproduced here because they are not in the repository.

### 5.5 Support range and limitations

- Nodules under 4 mm or over 30 mm return `invalid_input` and no risk. The nodule count is capped at 100 by the model, but Stage 02 keeps at most 50 candidates.
- The model was derived on screening-detected nodules; transportability and calibration to this application's population are not established.
- Five of the nine inputs (size, upper lobe, spiculation, type, and count) are model-derived or heuristic here, not observed, so the reported risk is only as reliable as those derivations.


## 6. Performance metrics

Each metric is quoted from the report file named in its heading. Where a table has no confidence interval, none was computed.

### 6.1 Stage 01 masks

No Dice, Jaccard, Hausdorff, or failure-rate evaluation exists. The only checks are the automated per-study gates listed in Section 4.1.

### 6.2 Stage 02 ViTDet3D detector

No local evaluation exists: no sensitivity, FROC, false positives per scan, or localization error on LIDC-IDRI or any LungInsight data was found in the repository. The upstream checkpoint's published LUNA16 results are not reproduced here and were not re-verified. The detector's logit threshold (-5.0) is permissive, and the top 50 candidates per scan are kept, so detector false positives flow into the nodule count and into Brock.

### 6.3 Stage 03 attribute models (deployed)

Source: [validation.json](LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/checkpoints/hybrid/evaluation_reports/validation.json) (437 nodules; 215 with reader mean >= 3) and [test.json](LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/checkpoints/hybrid/evaluation_reports/test.json) (355 nodules; 180 with reader mean >= 3). Target: per-nodule mean of LIDC reader ratings. Spearman is the rank correlation between predicted expected rating and reader mean; SD ratio is prediction SD divided by target SD (near 0 means near-constant output).

| Attribute | Spearman val | Spearman test | Spearman (reader mean >= 3) val / test | MAE val / test | SD ratio val / test | Median-baseline MAE (val) |
|---|---:|---:|---:|---:|---:|---:|
| Subtlety | -0.045 | 0.025 | 0.037 / 0.056 | 0.869 / 0.836 | 0.005 / 0.006 | 0.872 |
| Sphericity | 0.063 | 0.089 | 0.124 / 0.065 | 0.718 / 0.686 | 0.006 / 0.006 | 0.662 |
| Margin | -0.049 | -0.022 | -0.001 / -0.002 | 0.869 / 0.847 | 0.100 / 0.094 | 0.830 |
| Lobulation | 0.120 | 0.034 | 0.128 / 0.044 | 0.704 / 0.664 | 0.562 / 0.600 | 0.608 |
| Spiculation | 0.122 | -0.050 | 0.170 / 0.006 | 0.658 / 0.636 | 0.562 / 0.575 | 0.541 |
| Texture | 0.034 | 0.075 | 0.069 / 0.089 | 0.835 / 0.795 | 0.120 / 0.132 | 0.636 |
| Mean of six (M6) | 0.041 | 0.025 | 0.088 / 0.043 | | | |

The baseline column comes from the frozen-encoder probe diagnostic ([probe_results.csv](LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/frozen_encoder_probe/probe_results.csv)) on the same validation split.

Interpretation:

- All rank correlations are close to zero; none is a useful signal. There are no confidence intervals, and the validation-to-test sign changes (for example spiculation 0.122 to -0.050) are consistent with noise.
- Subtlety and sphericity predictions are essentially constant (SD ratio about 0.006). Margin and texture have little spread.
- On validation, MAE is worse than a constant median predictor for five of six attributes (subtlety is marginally better).
- Diagnostics in [one_stop_diagnostic](LungInsight_v2/backend/models/nodule_ranker_3d_distributional_v1/one_stop_diagnostic/ONE_STOP_SUMMARY.txt) show reader disagreement on 53.9% of nodules with two or more readers, a best handcrafted-feature ridge validation Spearman of 0.115, and no gain from a frozen pretrained-encoder probe. The multi-task versus single-task comparison ([multitask_vs_singletask.csv](LungInsight_v2/diagnostics/multitask_vs_singletask.csv)) also shows validation Spearman at or below about 0.11.
- Spiculation and texture feed Brock (Section 5.3), so Brock risk must be treated with the same caution.

### 6.4 Brock/PanCan

| Check | Result |
|---|---|
| BTS calculator reference case | 4.49% (BTS) versus 4.48547% (LungInsight) |
| Logit equals sum of contributions | Exact (tested to 12 decimals) |
| Local outcome validation (AUC, calibration) | Not available: no pathology-labelled LungInsight imaging cohort exists |

### 6.5 Clinical models

**LUAD-versus-LUSC** ([cancer-type-evaluation.json](LungInsight_v2/backend/Clinical/reports/cancer-type-evaluation.json)); patient-grouped split, 4,067 train, 1,356 calibration, 1,356 test (1,191 LUAD, 165 LUSC; natural prevalence).

| Metric | Model | Prevalence-only baseline |
|---|---:|---:|
| Balanced accuracy | 0.500 (95% CI 0.500-0.500) | 0.500 |
| Macro F1 | 0.468 | |
| Log loss | 0.3615 (95% CI 0.325-0.394) | 0.3703 |
| Multiclass Brier | 0.2100 | 0.2138 |
| ROC AUC (one-vs-rest) | 0.602 | |
| PR AUC: LUAD / LUSC | 0.901 / 0.153 | 0.878 / 0.122 (prevalence) |
| LUAD precision / recall | 0.878 / 1.000 | |
| LUSC precision / recall | 0.000 / 0.000 | |

The confusion matrix is [[1191, 0], [165, 0]]: every test patient was classified LUAD. With sex as the only feature, the model has weak ranking ability (AUC 0.60) and cannot identify LUSC at the default decision rule.

**Overall survival** ([survival-evaluation.json](LungInsight_v2/backend/Clinical/reports/survival-evaluation.json), [cross-cohort-evaluation.json](LungInsight_v2/backend/Clinical/reports/cross-cohort-evaluation.json)); 793 development and 198 test patients (57 events).

| Metric | Value |
|---|---:|
| Test C-index | 0.526 |
| Validation C-index | 0.471 (Kaplan-Meier tie baseline 0.500) |
| Time-dependent AUC at 12 / 36 / 60 months | 0.565 / 0.551 / 0.507 |
| Integrated Brier score (IPCW) | 0.244 |
| Test C-index by cohort | LUAD 0.567 (n = 100); LUSC 0.504 (n = 98) |
| Train LUAD, test LUSC (489 patients) | C-index 0.498; AUC 12 / 36 months 0.474 / 0.550; IBS 0.247 |
| Train LUSC, test LUAD (502 patients) | C-index 0.530; AUC 12 / 36 / 60 months 0.517 / 0.507 / 0.509; IBS not estimable |

The model clears its deployment gate (test C-index above 0.5) by a small margin and has weak discrimination. [Clinical README](LungInsight_v2/backend/Clinical/README.md) still quotes an earlier fit (C-index 0.575); the JSON report above is the current one.

### 6.6 Retired malignancy rankers (LIDC reader rating, not pathology)

Sources: [xgb_malignancy_model/metrics.json](LungInsight_v2/backend/models/malignancy/xgb_malignancy_model/metrics.json), [mlp_malignancy_model/metrics.json](LungInsight_v2/backend/models/malignancy/mlp_malignancy_model/metrics.json), [pairwise_ranker_v1.metrics.json](LungInsight_v2/backend/models/malignancy/pairwise_ranker_v1.metrics.json), [comparison_report.json](LungInsight_v2/backend/models/malignancy/validation_model_comparison_v1/comparison_report.json). Test set: 355 nodules.

| Model | Test MAE | Test accuracy (rounded) | Test QWK | Test Spearman |
|---|---:|---:|---:|---:|
| XGBoost (depth 2, 10 rounds) | 0.719 | 0.479 | 0.0 | 0.055 |
| MLP (selected epoch 2) | 0.719 | 0.479 | 0.0 | 0.097 |
| Pairwise linear ranker | Validation pairwise accuracy 0.496 (train 0.574), a coin flip | | | |

On validation, the best candidate (gradient boosting) had MAE 0.699 versus 0.703 for a constant mean predictor, and every prediction rounded to rating 3. These models did not beat a constant baseline, which is why Brock is the only malignancy method.

## 7. Application workflow and behavior

### 7.1 End-to-end workflow

1. **Case entry.** The user saves a clinical record (date of birth, sex, family history, medical history, smoking, performance status) and confirms the baseline definition.
2. **Upload.** A DICOM folder or files are validated and reconstructed into one HU volume; invalid or multi-series uploads are rejected with an explicit error.
3. **Analysis run.** The browser starts a run (`requested_malignancy_method: brock`) and polls stage progress: Stage 01 to 07, then clinical inference. Failed, blocked, or interrupted runs are shown as such, and interrupted runs can be resumed.
4. **Review.** The viewer shows axial CT slices with lung overlay, a nodule table (location, size, Brock result), and per-nodule panels: Brock inputs and provenance, Brock term contributions, predicted attributes, and Grad-CAM overlays. A stage status table reports each stage's completed, failed, and unavailable counts.
5. **Export.** `study_result_<run_id>.json`, the contract JSON, the Stage 07 CSV/JSON/HTML report, and artifacts are retrievable through the API.

Principal endpoints: `POST /api/v1/imaging/dicom/upload`, `PUT/GET /api/v1/clinical/records/{patient_id}`, `POST /api/v1/analysis/runs`, `GET /api/v1/analysis/runs/{run_id}` (status), `/viewer`, `/viewer/slices/{index}`, `/report`, `/report/html`, `/study`, `/nodules/{nodule_id}`, `POST /nodules/{nodule_id}/brock` (manual Brock inputs), `POST /runs/{run_id}/resume`, and `GET /api/v1/clinical/status`.

### 7.2 Handling of malignant versus non-malignant nodules

The application does not classify a nodule as malignant or benign. There is no benign/malignant classifier and no cancer-negative training cohort, and no output is a diagnosis. What the user sees is the Brock probability for each nodule, mapped to one of three display states by a fixed threshold in the browser code ([app.js](LungInsight_v2/frontend/app.js#L697): `MALIGNANCY_THRESHOLD = 0.10`, described in the code as a BTS-style display setting, not a diagnosis). The backend itself returns only the probability.

| Brock result for the nodule | Display state | What the user sees |
|---|---|---|
| Probability >= 10% | Elevated | Red banner "Elevated malignancy risk" with the percentage, the threshold note, and a "Likely cancer type" panel (LUAD and LUSC probabilities from the clinical classifier) |
| Probability < 10% | Unlikely | Green banner "Chances of cancer are unlikely" with the percentage and the threshold note; no cancer-type panel |
| Status `incomplete`, `invalid_input`, `unsupported`, or `error` | Not assessed | Amber banner stating Brock is unavailable and that this is not a negative result, a list of missing inputs, and a form to supply them manually |

In all three states the nodule keeps its attribute predictions, Grad-CAM overlays, geometry, provenance, and quality flags. Nodules are never dropped because their risk is low. Only detections that fail the lung-inclusion or heart-exclusion gate, or fall outside the top 50, are excluded, and those are listed in `excluded_outside_lung.csv` and the rejection audit.

Supporting behaviors:

- **Manual Brock inputs.** For a nodule with missing inputs, the user may enter values; Brock is then recomputed with the same formula and labelled "clinician-supplied". Known values cannot be overwritten without a recorded correction reason.
- **Cancer-type panel is patient-level.** It is the same LUAD/LUSC result for every elevated nodule in the study. It is conditional on cancer being present, depends only on sex, and, as measured, classified every test patient as LUAD (Section 6.5).
- **Survival is separate.** The survival estimate is computed from age and sex only. Neither clinical model uses imaging measurements or the Brock result as a predictor; imaging is context only.
- **Abstention.** Both clinical models abstain (cancer type) or are blocked (survival) when the baseline confirmation is missing or does not match training, or when a required feature is missing or outside training support. The study-level Brock counts (completed, incomplete, failed, not run) are always shown.
- **No negative inference from absence.** A nodule not detected, a nodule outside the 4-30 mm Brock range, and a nodule with unavailable Brock are not reported as benign.

### 7.3 Study-level states and failure handling

Study status is one of `not_started`, `running`, `completed`, `incomplete`, `blocked`, `failed`, or `interrupted`. Stage 02 acceptance gates and the Stage 01 mask gate must pass for downstream stages to run; each downstream stage re-validates the selected candidate set and its hash. Missing artifacts yield null counts or explicit errors, never zero risk or a successful state. Stage 04 incompleteness does not block attribute inference or Grad-CAM.

## 8. Reliability, limitations, and discrepancies

- Stage 03 attribute predictions are near chance (Section 6.3), yet two of them feed Brock and all six are displayed. They should not be used as findings.
- Nodule size is the detector box, the count is the detector count, and "upper lobe" is a vertical-position proxy. Each can shift Brock risk materially.
- The 10% display threshold is an unvalidated convention. The Brock model has not been calibrated to this population.
- The cancer-type model uses only sex and predicts LUAD for every test patient; the survival model has weak discrimination (C-index 0.526). Both are research outputs.
- The detector and the lung/heart masks have no local accuracy evaluation.
- Stage 03 results have no confidence intervals; sign changes between validation and test indicate high variance.
- Training cohorts for the clinical models (MSK-CHORD, TCGA) are not aligned with LungInsight imaging data, so imaging cannot be a predictor.
- A frontend row labelled "SHAP" in the stage table refers to the Brock term decomposition. The pipeline does not generate SHAP values for malignancy.
- Documented discrepancy: the Clinical README quotes a survival C-index of 0.575, while the current report file shows 0.526.
- Intended use, validation scope, and regulatory status must be defined before any clinical use. A negative or low-risk display does not exclude disease, and any positive display requires review of the source images.

## 9. Recommended validation work

1. Assemble a pathology- or follow-up-confirmed CT cohort and evaluate Brock (AUC, calibration, threshold sensitivity and specificity) with the pipeline-derived inputs, compared with manually entered inputs.
2. Evaluate ViTDet3D on LIDC-IDRI or LUNA16 with a FROC curve and report false positives per scan at the 50-candidate cap.
3. Compare Stage 01 masks with reference segmentations (Dice, Hausdorff).
4. Either retrain Stage 03 or remove its outputs from Brock (spiculation, nodule type) until they show a measurable signal, and add confidence intervals.
5. Replace the one-feature cancer-type model, or hide it until it separates the classes, and add clinically recorded predictors with aligned imaging data.
6. Monitor failure rates, abstention rates, and manual overrides after deployment.

## 10. Conclusion

LungInsight v2 is a traceable, research-only CT workflow whose only malignancy estimate is a correctly implemented, published Brock formula. The data it feeds into that formula are model-derived and, for the attribute-based inputs, currently unreliable. The learned components have no demonstrated accuracy: the attribute models, both clinical models, and the retired malignancy rankers perform near chance, and the detector and masks are unevaluated locally. Elevated or unlikely labels in the viewer are display conventions applied to a probability, not malignancy determinations.




