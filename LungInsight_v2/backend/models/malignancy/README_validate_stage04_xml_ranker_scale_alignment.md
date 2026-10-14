# Stage 04 input-scale alignment validation

This is a validation-only diagnostic. It does not train/retrain Stage 03 or Stage 04. It evaluates the existing XML-trained Stage 04 model twice on the existing frozen Stage 03 validation outputs: (1) raw ranker inputs, and (2) ranker inputs affinely mapped to the XML TRAIN feature distribution using means and standard deviations from `train.npz` and the XML table's `split=train` rows only.

It reads only `train.npz`, `val.npz`, the existing XML nodule table, and the XML-trained Stage 04 model. It never opens `test.npz`, H5, CT pixels, or DICOM.

Run from `C:\Users\emkap\Documents\Projects\LungInsight`:

```powershell
python .\LungInsight_v2\backend\models\malignancy\validate_stage04_xml_ranker_scale_alignment.py `
    --xml-table .\LungInsight_v2\backend\models\malignancy\xml_nodule_level_v1\nodule_level_xml_table.csv `
    --model .\LungInsight_v2\backend\models\malignancy\xml_nodule_level_v1\malignancy_from_nodule_level_xml.joblib `
    --features-dir .\LungInsight_v2\backend\models\malignancy\frozen_features `
    --out-dir .\LungInsight_v2\backend\models\malignancy\xml_nodule_level_v1\scale_alignment_validation_v1
```

The report is `feature_scale_alignment_report.json`. Do not promote the aligned approach unless it clearly beats the constant baseline on the held-out validation set. This test does not fix the physical-nodule count discrepancy; the report explicitly preserves that caveat.
