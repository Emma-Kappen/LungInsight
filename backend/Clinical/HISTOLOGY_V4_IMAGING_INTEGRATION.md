# Histology v4 imaging-input integration

## Route flow

`POST /api/v1/analysis/runs` accepts optional `histology_v4_inputs` alongside `clinician_inputs`. The analysis runner passes the completed, validated Imaging contract to `run_inference()`. The `POST /api/v1/clinical/predict` route accepts the same optional field for direct Clinical inference. `predict_histology_v4()` calls `build_histology_features()` and passes the unchanged 19-column row to the existing v4 artifact.

## Malignant-nodule eligibility

The selected nodule set is defined by the existing frontend Brock eligibility rule: `selected_prediction.method == "brock"`, `status == "available"`, and probability >= 0.10. This is the existing 10% research display threshold, not a pathology diagnosis or a separate malignancy classifier.

## Imaging-derived v4 inputs

- `lung_location`: `primary == "bronchus"` maps to `Central Lung`; `primary == "lung_wall"` with `wall_relationship == "wall_adjacent"` maps to `Peripheral Lung`. Other anatomy labels abstain. Brock's own `vertical_region` mapping is superior/inferior to an upper-lobe binary; that is not equivalent to TCGA's central/peripheral category and is deliberately not substituted.
- `anatomic_subdivision`: only explicit `primary == "bronchus"` maps to `Bronchial`. Stage 02 superior/middle/inferior thirds do not identify anatomical lobes and are not mapped to L/R lobe categories.
- `other_malignancy_history`: never inferred from CT, number of nodules, bilateral findings, or Brock scores. The existing structured clinical history maps `prior_cancer` to `Yes, History of Prior Malignancy`, and a reported complete history without `prior_cancer` to `No`. An explicit `histology_v4_inputs.other_malignancy_history` form value takes precedence.
- Mode ties: among tied modal categories, prefer the category represented by the nodule with the largest Brock-convention in-plane BBox diameter; then highest Brock probability; then alphabetical category; stable nodule ID is the final deterministic fallback.

## Display-only outputs

The eight Stage 03 attributes use each nodule's `expected_rating` output and are averaged independently across malignant nodules. `malignancy` is excluded from the eight-attribute summary because it is the Brock-selected malignancy score, not one of the eight attribute rankers. Mean detector size is the mean of `max(Y-extent-mm, X-extent-mm)`, matching the shared `bbox_in_plane_diameter_mm()` helper in `brock_inputs.py`; the BBox voxel extents are multiplied by the validated Stage 01 Z/Y/X spacing. Partial values are averaged over available valid outputs and accompanied by warnings.

## Form payload

`histology_v4_inputs` accepts the unchanged v4 feature names and exact training-time categorical values. It is optional; omitted or null values remain missing for the fitted imputation pipeline. Explicit form values override imaging-derived `lung_location` and `anatomic_subdivision`; conflicts are included in warnings. The API currently accepts this object, but the supplied frontend case-entry form does not yet render controls for all 19 v4 inputs. A frontend form change is required before users can enter those additional fields through the browser UI.

## Model preservation

No v4 artifact, estimator, feature order, training data, or fitted imputation pipeline is modified. Attribute-rank means and mean detector size are response/display-only fields and are not passed to `predict_proba()`.
