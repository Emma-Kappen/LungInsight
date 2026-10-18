# LungInsight tabular-only LUAD/LUSC preparation scripts

These scripts **do not read CT, DICOM, NIfTI, or any raw imaging data**.

## 1. Fix and prepare TCGA labels

From PowerShell, in the LungInsight virtual environment:

```powershell
$datasets = 'C:\Users\emkap\Documents\Projects\LungInsight\datasets'
$dest = 'C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2\backend\Clinical\tools\tcga_prep'

python "$dest\01_prepare_tcga_labels.py" --datasets-root $datasets
```

The parser explicitly skips TCGA clinical TSV metadata rows beginning with `diagnosis` and `CDE_ID`. It writes:
- `prepared\tcga\tcga_patient_labels.csv`
- `prepared\tcga\tcga_label_qc.json`

Review QC; unexpected labels or missing targets should be resolved before training.

## 2. Build tabular candidate dataset

```powershell
python "$dest\02_build_tabular_dataset.py" --datasets-root $datasets
```

This creates a **TCGA-only candidate table** unless a normalized MSK-CHORD CSV is supplied. The MSK CSV must contain one patient per row, `patient_id`, `target` (`LUAD` or `LUSC`), and tabular feature columns:

```powershell
$msk = 'C:\path\to\normalized_msk_chord_luad_lusc.csv'
python "$dest\02_build_tabular_dataset.py" --datasets-root $datasets --msk-csv $msk
```

Do not use the candidate table directly for training. Review the retained feature semantics, harmonize shared fields, and ensure labels are sourced from verified patient-level records. Do not use `patient_id`, `source`, cohort name, ICD-O-3, detailed histology, or diagnosis-derived fields as predictors.

## 3. Audit a candidate table

```powershell
$table = Join-Path $datasets 'prepared\tcga\tcga_msk_tabular_candidates.csv'
python "$dest\03_audit_tabular_dataset.py" --csv $table
```

## Critical modeling constraints

- Training sources are intended to be MSK-CHORD, TCGA-LUAD, and TCGA-LUSC.
- TCGA-LUAD and TCGA-LUSC are source-separated by target in this extraction. That means source membership is confounded with label. Never feed `source` to the model, and do not claim source-independent generalization from a random split alone.
- MSK-CHORD needs to be normalized to a compatible patient-level schema before merging; this script intentionally does not guess its format.
- Clinical variables must be available and consistently defined across training and inference.
- Upstream model outputs (e.g. nodule count) are eligible only if a compatible feature exists in training data or the deployed model has a validated missing-feature strategy. Do not fabricate training values.
- This bundle prepares/audits tabular data; it does not train or clinically validate the classifier.
