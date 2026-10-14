# LungInsight v2 pipeline audit/refactor bundle

This bundle contains revised Stage 01–06 inference scripts, a canonical `meta_schema.json`, shared schema utilities, contract validation/tests, and a PowerShell runner. It does **not** contain, retrain, rewrite, or replace any trained checkpoint/model artifact.

## Install

Merge the contents of this ZIP into the existing `LungInsight_v2` repository root. Back up the current scripts and schema first. Keep the existing `backend/models/`, datasets, and `l-env` intact.

Included files:

- `meta_schema.json`
- `backend/Imaging/01_hu_and_crop.py`
- `backend/Imaging/02_detect_and_patch.py`
- `backend/Imaging/03_nodule_classification.py`
- `backend/Imaging/04_malignancy_prediction.py` (primary Stage 04)
- `backend/Imaging/04_malignancy_classifier.py` (optional alternative Stage 04)
- `backend/Imaging/05_gradcam.py`
- `backend/Imaging/06_full_ct_gradcam.py`
- `backend/Imaging/pipeline_schema.py`
- `validate_pipeline.py`
- `tests/test_pipeline_schema.py`
- `run_pipeline.ps1`

## Metadata source of truth

`meta_schema.json` is version **2.5**. `pipeline_schema.py` resolves the canonical schema at the `LungInsight_v2` root ahead of any stale nested copy, validates the schema version and required keys, and provides dotted-key accessors for stage scripts. `LUNGINSIGHT_META_SCHEMA` can explicitly override the file path when needed.

The schema declares stage paths, filenames, input/output contracts, HU conversion/windowing and crop settings, detector settings, coordinate conventions, patch geometry/dtype, canonical six-attribute order, frozen checkpoint groups, Stage 04 rating transforms, and Grad-CAM display configuration. Model architecture/tensor operations and checkpoint state remain in the original model code/artifacts; they are not reconstructed from prose in the schema.

The schema version was incremented to 2.5 to reflect the separate LoG patch-manifest and Stage 03 diagnostic-output contract. Run the pipeline from Stage 01 after installing this bundle so new artifacts consistently carry `meta_schema_version: "2.5"`.

## Changes by script

### Stage 01 — `01_hu_and_crop.py`

- Reads preprocessing, filenames and paths from the canonical schema.
- Resolves the LIDC data path from the schema and derives the legacy parent directory without duplicating an absolute path.
- Writes schema provenance to `meta.json`; preserves the existing CT processing and segmentation algorithms.

### Stage 02 — `02_detect_and_patch.py`

- Reads ViTDet3D settings, normalization, thresholds, coordinate/patch conventions, output names and localization parameters from the schema.
- Retains `candidates.json` and `patches/patches_manifest.json` as the **authoritative ViTDet3D-only** candidate/patch route.
- Converts LoG detections into 64³ raw-HU patches in a **separate** tree: `02_detect_and_patch/log_diagnostic_patches/` with its own `log_patches_manifest.json`. LoG records are not merged into `candidates.json`.
- Disables per-patch PNG visualizations for diagnostic LoG patches by default to reduce disk use; records preserve detector source/ID, scale and LoG response.
- Adds `--log-patch-limit N` as an optional limit for test runs. Default is unlimited, as requested: every LoG candidate is patched for Stage 03.
- Prints the estimated raw `.npy` array size before extraction. Each 64³ float32 patch is about 1 MiB; the observed 4,931 LoG proposals would require about **4.81 GiB** for arrays alone, excluding manifests and other outputs.

### Stage 03 — `03_nodule_classification.py`

- Checks the existing frozen helper's feature order, checkpoint groups and patch shape against the schema; validates patch manifest versions, IDs, shapes and dtypes.
- Runs the existing six-attribute spatial-hybrid checkpoints on the authoritative ViTDet3D patches and writes `predictions.json` / `predictions.csv` as before.
- By default, also runs the same frozen attribute heads on the separate LoG diagnostic patches and writes `log_diagnostic_predictions.json` / `log_diagnostic_predictions.csv`.
- The LoG outputs are explicitly tagged `detection_source: "LoG"`, `purpose: "research_comparison_only"`, and `stage04_eligible: false`. They are **not** merged into `predictions.json`, which remains the primary Stage 04 input.
- Use `--skip-log-diagnostic` to bypass LoG Stage 03 inference deliberately. Do not confuse exploratory ratings for LoG proposals with validated nodule ratings.

### Stage 04 primary — `04_malignancy_prediction.py`

- Loads `STAGE03_PATHS` from `meta_schema.json` (fixes the undefined `STAGE03_PATHS` runtime `NameError`).
- Consumes the authoritative `predictions.json` and existing joblib model, validates the feature order/ranges, and emits the existing scalar LIDC reader malignancy/suspicion rating. This is not a cancer probability or diagnosis.

### Stage 04 alternative — `04_malignancy_classifier.py`

- Remains a distinct optional ordinal-distribution inference path; it is not substituted for the primary joblib model. It requires its existing JSON checkpoint and helper code.

### Stage 05 — `05_gradcam.py`

- Reads the six attributes, checkpoint mapping, target layer, patch shape, palette, outline, alpha and filenames from the schema; fixes the default patch path to the actual Stage 02 `patches/arrays/patch_000.npy` layout.
- Uses existing weights unchanged and checks patch/checkpoint presence and finite input/CAM values.

### Stage 06 — `06_full_ct_gradcam.py`

- Reads the Stage 01 volume, Stage 02 authoritative manifest, Stage 05 CAM roots/patterns, palette, display window and output paths from the schema.
- Validates patch IDs, geometry, CAM arrays and manifest versions. An unqualified patch-000 CAM is not reused for other patches.

### Shared helpers, runner and tests

- `pipeline_schema.py`: canonical schema resolution, key/type/version validation and actionable errors.
- `validate_pipeline.py`: cross-stage path/feature/shape/schema checks plus a specific regression guard for the Stage 04 `STAGE03_PATHS` variable.
- `run_pipeline.ps1`: runs schema validation, local artifact checks, unit tests, then stages 01–06; it verifies required intermediate/final files and stops on the first error.
- `tests/test_pipeline_schema.py`: 11 lightweight tests, including LoG-manifest separation, candidate conversion and both previous path-regression cases.

## Run from Windows PowerShell

```powershell
Set-Location 'C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2'
. .\l-env\Scripts\Activate.ps1

python -m json.tool .\meta_schema.json > $null
python .\validate_pipeline.py
python .\validate_pipeline.py --check-artifacts
python -m unittest discover -s tests -v
```

`--check-artifacts` should be run in the actual repository where the pre-existing model/helper/checkpoint files are present. Those artifacts are not included in the ZIP.

## Individual commands, Stages 01–06

```powershell
$PatientId = 'LIDC-IDRI-0001'

# Stage 01: DICOM -> HU arrays, masks, and meta.json
python .\backend\Imaging\01_hu_and_crop.py $PatientId

# Stage 02: ViTDet3D candidates + all LoG diagnostic patches (CPU)
python .\backend\Imaging\02_detect_and_patch.py $PatientId --device cpu

# Stage 03: predict six attributes for ViTDet3D and the separate LoG proposals (CPU)
python .\backend\Imaging\03_nodule_classification.py --patient-id $PatientId --device cpu

# Stage 04: scalar malignancy-rating estimate for authoritative ViTDet3D candidates only
python .\backend\Imaging\04_malignancy_prediction.py $PatientId

# Stage 05: Grad-CAM for patch_000
python .\backend\Imaging\05_gradcam.py --patient-id $PatientId

# Stage 06: project available CAMs into the processed CT coordinate frame
python .\backend\Imaging\06_full_ct_gradcam.py --patient-id $PatientId
```

### Complete end-to-end command

Run this from the project parent directory. `-ExecutionPolicy Bypass` applies only to this PowerShell child process. By default, all LoG proposals are processed; use `-LogPatchLimit 100` for a smaller trial if disk space or CPU time is limited.

```powershell
Set-Location 'C:\Users\emkap\Documents\Projects\LungInsight'
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File '.\LungInsight_v2\run_pipeline.ps1' `
  -PatientId 'LIDC-IDRI-0001'

# Optional limited LoG trial (reruns the whole pipeline with at most 100 LoG patches):
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File '.\LungInsight_v2\run_pipeline.ps1' `
  -PatientId 'LIDC-IDRI-0001' `
  -LogPatchLimit 100
```

### Optional alternative Stage 04

Use only when the existing ordinal JSON checkpoint and its matching helper files are available. It produces a rating distribution rather than the scalar value used by the default runner.

```powershell
python .\backend\Imaging\04_malignancy_classifier.py `
  --predictions ".\backend\output\$PatientId\03_nodule_classification\predictions.json" `
  --checkpoint '<absolute path to existing ordinal_v1 JSON checkpoint>' `
  --out ".\backend\output\$PatientId\04_malignancy_prediction\malignancy_rating_distribution.json"
```

## Output sanity checks

- Stage 01: `backend\output\<patient>\meta.json`, `volume_hu_cropped.npy`
- Stage 02 authoritative: `02_detect_and_patch\candidates.json`, `patches\patches_manifest.json`, and `patches\arrays\patch_###.npy`
- Stage 02 LoG: `02_detect_and_patch\log_diagnostic_patches\log_patches_manifest.json` and `arrays\patch_###.npy`
- Stage 03 authoritative: `03_nodule_classification\predictions.json` / `.csv`
- Stage 03 LoG: `03_nodule_classification\log_diagnostic_predictions.json` / `.csv`
- Stage 04: `04_malignancy_prediction\malignancy_predictions.json`
- Stage 05: `05_explainability\gradcam_rainbow_final\`
- Stage 06: `06_full_ct_gradcam_rainbow\full_ct_gradcam_report.json`

## Assumptions and limits

1. Stage 03 exposes only six attributes from the supplied frozen spatial-hybrid checkpoints: `subtlety`, `sphericity`, `margin`, `lobulation`, `spiculation`, and `texture`. It does not predict internal structure, calcification or malignancy.
2. LoG proposals are exploratory diagnostic candidates. The attribute model is run on their patches at the user's request, but the ratings are not validated for this proposal source; they are kept separate from the authoritative Stage 03/Stage 04 path.
3. The Stage 04 primary joblib model is used in the wrapper. The alternative ordinal-distribution script remains optional and requires its own existing checkpoint.
4. Stage 05 generates Grad-CAM for patch 000 by default. Stage 06 reports missing maps for other authoritative patches until maps are generated for them; it does not invent CAMs.
5. The trained checkpoints, helper model files and patient dataset are external existing repository assets and remain untouched. The bundle was validated through JSON/schema checks, Python compilation and 11 unit tests; this environment did not run patient CT/model inference. Run the provided pipeline locally to validate actual execution.
