# Stage 04: nodule-level XML-supervised malignancy model

This experiment trains **Stage 04 only**. It does not train or modify the six Stage 03 rankers. The Stage 04 inference contract remains exactly these six scalar inputs, in order:

`subtlety`, `sphericity`, `margin`, `lobulation`, `spiculation`, `texture`

The target is the median LIDC reader-assigned malignancy/suspicion rating (1–5) for a spatially grouped physical nodule. This is a radiologist rating, not a pathology-confirmed cancer probability.

## Why it parses raw XML and DICOM headers

The XML `noduleID` is reader/session-local; it is **not** a safe physical-nodule grouping key. The script uses XML ROI contour centers and DICOM geometry headers to group annotations from different readers, using a deterministic greedy 10 mm centroid rule (the grouping radius used by the existing fixed-H5 generator). DICOM reads use `stop_before_pixels=True`: no CT PixelData is loaded, no H5 is opened, and geometry is not included as a model feature. All annotation bodies parsed for training/validation belong to the existing train/validation patients only.

## Run from the repository parent

Run from:
`C:\Users\emkap\Documents\Projects\LungInsight`

This matters because the source annotations are in the parent repository's `datasets\LIDC`, while the active code lives under `LungInsight_v2`.

```powershell
python .\LungInsight_v2\backend\models\malignancy\train_stage04_nodule_level_xml.py `
    --lidc-root .\datasets\LIDC `
    --split-json .\LungInsight_v2\backend\models\nodule_ranker_3d_distributional_v1\split.json `
    --ranker-val-archive .\LungInsight_v2\backend\models\malignancy\frozen_features\val.npz `
    --out-dir .\LungInsight_v2\backend\models\malignancy\xml_nodule_level_v1
```

If the existing virtual environment does not have `pydicom` installed, activate `LungInsight_v2\l-env` and install it with `python -m pip install pydicom`. NumPy, scikit-learn, and joblib are also required.

## Outputs

- `malignancy_from_nodule_level_xml.joblib` — model bundle compatible with the existing `predict_xml_trained_malignancy_from_rankers.py` inference adapter.
- `training_report.json` — split safety, grouping audit, feature/target counts, XML-nodule validation metrics, and Stage 03 `val.npz` metrics.
- `nodule_level_xml_table.csv` — train/validation physical-nodule medians for audit.
- `xml_nodule_validation_predictions.csv` — held-out XML-nodule validation predictions.
- `stage03_validation_predictions.csv` — only if the existing Stage 03 validation archive is present and valid.

The `ranker_validation` section is the relevant measure of how this trained model behaves when given frozen Stage 03 outputs. The XML nodule-level validation measures the annotation-to-rating relationship on human-rated inputs. Do not interpret those two metrics as interchangeable.

## Safety and split contract

- Uses the existing `split.json` unchanged and asserts train/validation patient separation.
- Trains from XML annotations belonging to train patients only.
- Validates from XML annotations belonging to validation patients only.
- If provided, reads only the existing `frozen_features\val.npz`; it never opens `test.npz`.
- Does not read H5, CT pixel data, Stage 03 checkpoints, or alter Stage 03.
- Only the six scalar attributes are given to the malignancy regressor.

## Important audit before trusting metrics

Review `training_report.json` for `nodule_counts`, `parse_and_aggregation_stats`, and `dicom_geometry_index_stats`. Physical grouping is a spatial heuristic; compare the usable train/validation nodule counts with the established frozen-feature counts (previously 1,859 train and 437 validation nodules). If counts are materially different, inspect XML/DICOM series matching and grouping before treating the model metrics as reliable. Do not tune the grouping threshold against the test split.
