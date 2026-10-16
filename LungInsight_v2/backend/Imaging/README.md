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
`nodules` values via `--brock-inputs`. The only accepted method selector is
`--malignancy-method brock`. The selected method is stored per nodule in
`malignancy.selected_prediction`; an unavailable Brock result has no numeric
probability and is never replaced by another score. Stage 07 reports show the
selected method, Brock status,
probability, and formula contributions when present, or the explicit
abstention status otherwise.

## Explainability status by stage

- Stage 01: preprocessing and mask QC metadata are the explanation surface;
  missing, empty, or suspicious masks are retained as QC failures.
- Stage 02: detector source, fusion, geometry, and rejection provenance are
  retained. No saliency is fabricated for nondifferentiable fusion.
- Stage 03: existing model-specific attribute evidence is retained.
- Stage 04: Brock uses exact formula decomposition; SHAP is not used for
  malignancy risk.
- Stage 05: per-nodule records carry Brock decomposition and compatible
  attribute Grad-CAM references, each with its own model identity.
- Stage 06: Grad-CAM status and spatial-registration failures are explicit.
- Stage 07: invalid localization or missing explanation evidence is displayed
  as unavailable rather than as a successful result.

The Brock-only malignancy fields are carried by the Imaging-to-Clinical contract
(`imaging_clinical_contract_v1`); legacy reader-rating malignancy fields are not
accepted.
