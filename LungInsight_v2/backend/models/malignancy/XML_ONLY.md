# XML-Only Malignancy Suspicion Model

This workflow estimates an LIDC radiologist's ordinal malignancy suspicion rating. It does not estimate confirmed cancer and is not a diagnostic system. It uses structured XML ratings and contour geometry only; CT patches and attribute-model checkpoints are not model inputs.

## Data and linkage

The XML stores per-reading-session `unblindedReadNodule` records, eight non-malignancy characteristics, a separate malignancy rating, and ROI contour coordinates. The source rating scales are subtlety 1-5, internal structure 1-4, calcification 1-6, and the other six characteristics 1-5. Malignancy is 1-5. The source contains no `<diameter>` element; the feature table derives pixel-unit contour extents and an equivalent-diameter measurement from ROI edge points. These are not millimeters because XML does not provide a reliable pixel spacing.

XML nodule IDs are reader-local, so they are not used as physical-nodule identities. An XML reader record is linked only when its exact study/series identity and complete eight-rating non-malignancy signature uniquely identify one H5 reader slot, with a one-to-one reverse match. Malignancy is not used for matching. H5 patches are never read as predictor input. The retained physical-nodule ID and split membership come from the canonical H5 sample index and its existing cluster identity.

The canonical split's sample counts and index hashes are checked against the existing project split validator values, then patient disjointness and exact H5 partition coverage are checked. In the current source, 1,318 XML files parse, 283 have no usable study/series identity, and 20,362 reader nodule records are parseable in files with identity. Of these, 5,519 link uniquely; 1,509 are ambiguous, 13,333 lack a complete non-malignancy signature, and one has no matching H5 signature. These are retained in the unmatched audit rather than force-matched. Coverage for each canonical split is in `split_manifest.json`.

## Models and scoring

The ordinal model uses cumulative logistic regression; the comparison model uses cumulative HistGradientBoosting classifiers. Both fit reader-level 1-5 targets and return a probability distribution over the five ordered ratings. Nodule-level feature summaries are repeated for each available reader rating during training and evaluation. Validation QWK selects the primary model, with validation MAE as a tie-breaker. The test split is blocked until validation evaluation exists and is evaluated once. Evaluation compares reader ratings individually instead of treating a consensus as ground truth, and includes train-only majority and median baselines.

## Commands

Run these from the `LungInsight_v2` directory in PowerShell. The default output directory is `backend/output/malignancy/xml_only_run`; commands refuse to overwrite existing artifacts. Choose a fresh `--out-dir`/`--data-dir` for a new run.

```powershell
.\l-env\Scripts\python.exe -m unittest discover -s backend\models\malignancy -p test_xml_malignancy.py -v
.\l-env\Scripts\python.exe backend\models\malignancy\build_xml_malignancy_dataset.py
.\l-env\Scripts\python.exe backend\models\malignancy\train_xml_malignancy.py
.\l-env\Scripts\python.exe backend\models\malignancy\evaluate_xml_malignancy.py --split val
.\l-env\Scripts\python.exe backend\models\malignancy\evaluate_xml_malignancy.py --split test
```

Inference accepts a JSON object whose keys follow `feature_schema.json`'s `feature_order`. This example derives an input record from the first built nodule row, writes it to a temporary JSON file, and runs the saved primary model:

```powershell
$row = Import-Csv 'backend\output\malignancy\xml_only_run\xml_features.csv' | Select-Object -First 1
$featureNames = @('subtlety_mean','subtlety_std','internalStructure_mean','internalStructure_std','calcification_mean','calcification_std','sphericity_mean','sphericity_std','margin_mean','margin_std','lobulation_mean','lobulation_std','spiculation_mean','spiculation_std','texture_mean','texture_std','roi_count_mean','roi_count_std','contour_points_mean','contour_points_std','contour_bbox_width_mean','contour_bbox_width_std','contour_bbox_height_mean','contour_bbox_height_std','contour_equivalent_diameter_mean','contour_equivalent_diameter_std')
$record = @{}
foreach ($name in $featureNames) { $record[$name] = [double]$row.$name }
$recordPath = Join-Path $env:TEMP 'lunginsight_xml_features.json'
$record | ConvertTo-Json -Compress | Set-Content -Encoding Ascii $recordPath
.\l-env\Scripts\python.exe backend\models\malignancy\predict_xml_malignancy.py --record $recordPath
Remove-Item $recordPath
```

Inference reports missing or out-of-range feature warnings, expected rating, and category probabilities. The prediction must be interpreted as an estimate of radiologist suspicion, not a cancer diagnosis.