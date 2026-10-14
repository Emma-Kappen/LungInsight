# XML-trained malignancy regressor compatible with frozen Stage 03 outputs

This package trains Stage 04 from the existing LIDC XML reader annotations and keeps the Stage 03 rankers, checkpoints, H5, and canonical patient split unchanged.

## Training and validation

Run from the `LungInsight_v2` repository root using the existing `l-env`:

```powershell
python backend/models/malignancy/train_malignancy_from_xml_ranker_compatible.py `
    --annotations datasets/LIDC/lidc_all_annotations_match.json `
    --split-json backend/models/nodule_ranker_3d_distributional_v1/split.json `
    --ranker-val-archive backend/models/malignancy/frozen_features/val.npz `
    --out-dir backend/models/malignancy/xml_annotation_ranker_compatible_v1
```

The model fits all valid per-reader XML annotation rows belonging to the existing train patients. It reports validation on the held-out XML annotation rows belonging to existing validation patients. If `frozen_features/val.npz` exists, it also applies the same trained model to those Stage 03 outputs and reports the nodule-level reader-mean target metrics. That second metric is the more relevant check for deployment compatibility, because XML and ranker-output feature distributions differ.

The trainer does not open `frozen_features/test.npz`, produce test metrics, read H5/CT, load any Stage 03 checkpoints, or tune hyperparameters on validation. It does read the existing split file and ignores annotation records whose patient IDs are outside train/validation.

Outputs include:
- `malignancy_from_xml_annotations.joblib`
- `training_report.json`
- `xml_annotation_validation_predictions.csv`
- `stage03_validation_predictions.csv` (when the validation archive is present)

## Run inference on Stage 03 output

```powershell
python backend/models/malignancy/predict_xml_trained_malignancy_from_rankers.py `
    path\to\predictions.json `
    --model backend/models/malignancy/xml_annotation_ranker_compatible_v1/malignancy_from_xml_annotations.joblib
```

By default this writes `malignancy_predictions.json` beside the supplied Stage 03 `predictions.json`. It consumes exactly these scalar attributes in order: `subtlety`, `sphericity`, `margin`, `lobulation`, `spiculation`, and `texture`. It does not accept or read CT/H5 inputs. Outputs represent the LIDC reader-assigned malignancy/suspicion rating, not pathology-confirmed cancer probability.

## Interpretation caution

Validation on XML reader-rated inputs measures the annotation-to-rating relationship. Stage 03 ranker outputs can be much narrower than XML values; use the separate `ranker_output_validation` section of `training_report.json` to assess how the model behaves with those outputs. Accepting the interface is not itself proof of useful deployed accuracy.
