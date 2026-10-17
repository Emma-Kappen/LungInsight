# LungInsight Imaging risk and explainability

The Imaging pipeline remains research-only. Detection, nodule-attribute
classification, Grad-CAM, and bounding-box behavior are preserved. Brock/PanCan
is the only malignancy-risk model supported by this POC.

## Brock/PanCan full model

`brock.py` implements the deterministic full model from McWilliams et al.,
*Probability of Cancer in Pulmonary Nodules Detected on First Screening CT*,
NEJM 2013, DOI [10.1056/NEJMoa1214726](https://doi.org/10.1056/NEJMoa1214726).
The implementation uses the published-coefficient formula specified by the
project, including the nonlinear size term, spiculation, and solid as the
nodule-type reference category.

Inputs are explicit and must be available: age in years, model sex mapping,
family history, emphysema, longest diameter in millimetres, nodule type,
upper-lobe location, total scan nodule count, and spiculation. Missing or
invalid input produces `incomplete` or `invalid_input` status and no probability;
values are never fabricated from reader ratings, detector scores, or defaults. Contributions are
reported in logit units and sum to the reported logit. The result is a risk
estimate, not a pathology diagnosis, and is conditional on the model's
supported input population. The implementation currently accepts 4--30 mm
nodules and counts from 1--100; these software validation bounds are not a
claim of external clinical validation.

Reference verification: the official [BTS Pulmonary Nodule Risk
Calculator](https://www.brit-thoracic.org.uk/clinical-resources/guidelines/pulmonary-nodules/pn-risk-calculator/)
returned 4.49% for the fixed case age 62, female, no family history, no
emphysema, 10 mm solid nodule, non-upper-lobe, four total nodules, and no
spiculation (retrieved 2026-10-16). This implementation returned 4.48547%,
which rounds to 4.49%. The test keeps the rounded tolerance because the
official calculator displays two decimal percentage places.

Stage 04 accepts an optional JSON envelope with `patient` and per-nodule
`nodules` values via `--brock-inputs`, or as `-BrockInputs` to
`run_pipeline.ps1`. Patient-level predictors are age, sex, family history, and
emphysema; per-nodule predictors are size (millimetres), type, upper-lobe
location, total nodule count, and spiculation. The envelope is keyed by the
selected Stage 02 stable candidate ID. Unknown fields and unselected IDs are
rejected. Detector diameter and Stage 03 reader-rating predictions are not
substitutes for Brock inputs. The only accepted method selector is
`--malignancy-method brock`. The selected method is stored per nodule in
`malignancy.selected_prediction`; an unavailable Brock result has no numeric
probability and is never replaced by another score. Stage 07 reports show the
selected method, Brock status,
probability, and formula contributions when present, or the explicit
abstention status otherwise.

## Explainability status by stage

- Stage 01: preprocessing and mask QC metadata are the explanation surface;
  missing, empty, misaligned, or geometrically invalid masks are retained as
  QC failures. Visual review is optional and is recorded as such.
- Stage 02: detector source, fusion, geometry, and rejection provenance are
  retained. No saliency is fabricated for nondifferentiable fusion.
- Stage 03: only the selected, mask-validated fused Stage 02 manifest is
  classified. The six expected LIDC-style ratings are model predictions, not
  measured current-patient findings or annotation-derived reader labels.
  Existing ordinal head distributions and derived spread/entropy are recorded;
  these are not calibrated confidence. Unsupported attributes, diagnosis,
  pathology, and unavailable measurements remain explicitly unavailable.
- Stage 04: Brock uses exact formula decomposition; SHAP is not used for
  malignancy risk.
- Stage 05: eligible fused nodules receive supported attribute Grad-CAM;
  Brock SHAP is not claimed or generated.
- Stage 06: Grad-CAM status and spatial-registration failures are explicit.
- Stage 07: invalid localization or missing explanation evidence is displayed
  as unavailable rather than as a successful result.

The separate LoG diagnostic manifest was removed. ViTDet3D is the sole nodule
detector. Its detections enter Stages 03–06 only after lung-mask inclusion,
heart-mask exclusion, and top-50 selection. Every downstream stage rechecks the selected
candidate set and mask gate; Stage 04 records incomplete Brock inputs without
blocking otherwise eligible attribute inference or Grad-CAM.

The versioned Imaging-to-Clinical contract is
[`imaging_clinical_contract_v2.schema.json`](schemas/imaging_clinical_contract_v2.schema.json).
It reports per-nodule stable identity, detector provenance, voxel/physical
geometry, estimates, attribute classification, Brock status and completeness,
quality, model versions, explanations, and artifact references. Automated
mask validation remains mandatory; human review/attestation is optional and
is not implied by a successful automated status. Study-level
counts are derived from persisted raw detections, fusion/filter audits, selected
candidate lists, and Brock prediction records. Missing files remain unknown;
failed or blocked processing is never represented as success. Only candidates
that pass the automated Stage 01 mask and geometry gates are emitted as
eligible nodule rows.
Legacy reader-rating malignancy fields are not accepted.
