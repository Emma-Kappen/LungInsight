# LungInsight Histology v4 integration patch

## What this patch changes
- Adds `backend/Clinical/histology_v4.py` as a separate, independently loaded TCGA-only LUAD/LUSC research model.
- Adds `cancer_type_v4` to each Clinical prediction response. Existing `cancer_type` (v2) and `survival` outputs are retained.
- Displays the v4 result separately in the report UI, behind the existing Brock eligibility display gate.
- Installs the selected `cancer-type-luad-lusc-research-v4.joblib` artifact at the path expected by the loader.
- Pins scikit-learn to 1.9.1 because the v4 joblib was serialized with that version.

## Input mapping safety
Only age, sex, race and ethnicity are mapped from current app inputs, where categories match the training data. The app's calculated pack-years are deliberately not mapped to TCGA's directly recorded pack-years. The app's never/former/current smoking labels are not converted to the undocumented TCGA smoking-history codes. Pulmonary-function, preoperative performance, smoking-year, and tumour-location inputs remain explicitly missing and are handled by the fitted imputation pipeline. The response lists every imputed feature.

## Model interpretation
- Cohort: TCGA-LUAD and TCGA-LUSC, 1,026 patients.
- Selected estimator: Random Forest.
- Holdout ROC-AUC: 0.7978; balanced accuracy: 0.7282.
- Training five-fold ROC-AUC: 0.7525 +/- 0.0102.
- This is an internal research estimate. Labels derive from cohort membership and cohort effects may be learned. It is not clinically validated, not a diagnosis, and not a cancer-versus-benign classifier.

## Apply
Extract this ZIP into the LungInsight project root, then run:
`powershell -ExecutionPolicy Bypass -File .\apply_histology_v4.ps1`

The script creates a timestamped backup before copying. It does not touch v2/v3 model files or survival artifacts.
