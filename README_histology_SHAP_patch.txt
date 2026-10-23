LungInsight v2 — Histology SHAP patch
=====================================

What this patch adds
- Exact interventional Shapley values for the active two-input LUAD/LUSC model.
- Explanation values for both LUAD and LUSC predicted probabilities.
- A histology-specific frontend explanation, separate from Brock formula contributions and survival ablation.
- Non-blocking failure: if the reference data or explanation is unavailable, prediction values are still returned.
- A schema updater and a smoke test.

Important implementation detail
The explanation uses the exact two-feature Shapley formula over the reconstructed 5,423-row MSK-CHORD-2024 training partition (stratified 80/20 split, random_state=42). This is an exact SHAP/Shapley calculation for the model's predict_proba output, not a sampled approximation and not a call to the SHAP Python package. The active model is not retrained or overwritten.

Install
1. Back up your current files first:
   Copy-Item backend\Clinical\inference.py backend\Clinical\inference.py.before_histology_shap
   Copy-Item frontend\app.js frontend\app.js.before_histology_shap
   Copy-Item backend\Clinical\schemas\clinical_prediction_v1.schema.json backend\Clinical\schemas\clinical_prediction_v1.schema.json.before_histology_shap

2. From the LungInsight_v2 project root, extract the zip so its backend/, frontend/, and tools/ paths merge into your project. Confirm the two replacement source files are the ones in the archive.

3. Run the schema updater from the project root:
   python tools\patch_histology_shap_schema.py

   It makes a .schema-backup.json backup before editing. It refuses to edit if it cannot identify exactly one histology result schema.

4. Run:
   python -m py_compile backend\Clinical\inference.py
   node --check frontend\app.js
   python tools\test_histology_shap.py

5. Run the existing Clinical test suite and a normal end-to-end inference request. Confirm the output schema accepts the new optional cancer_type.explanation field.

Safety/interpretation
- SHAP values explain model output, not biological causes or diagnosis.
- The model remains research-only; probabilities are not calibrated clinical risks.
- The SHAP background is reconstructed from the documented candidate training split. The explanation is unavailable if that source dataset is missing, changed, or fails validation; prediction remains non-blocking.
- This patch does not retrain, promote, or replace the active model.
