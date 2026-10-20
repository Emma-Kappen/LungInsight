# Compatibility and run report

## Runtime compatibility observed

- OS: Windows.
- Python: 3.12.10 from the project `l-env`.
- `shap`: 0.52.0; `scikit-learn`: 1.9.1.
- The project environment resolved all required pipeline modules during this run: NumPy, PyTorch, pydicom, Transformers, SHAP, SciPy, Matplotlib, Pillow, joblib, and scikit-learn.
- Stage 03 checkpoints and their architecture/helper sources were found by the local artifact check. Stage 04 joblib model and its training table were found and loaded.
- The default detector is remote and may require network access or a pre-populated Hugging Face cache on first Stage 02 execution.

## Compatibility changes

- Canonical metadata schema migrated from 2.10 to 2.12. No trained artifact or model definition was modified.
- Stage 02 manifest schema remains the schema-declared version; the root metadata schema now describes the additional review, filtering, explanation, record, and visualization contracts.
- `backend/Imaging/pipeline_schema.py` continues to choose the `LungInsight_v2/meta_schema.json` root first. The older nested schema is intentionally not synchronized or selected.
- Stage 04 joblib inference is the primary model path. `04_malignancy_classifier.py` remains an optional, distinct ordinal implementation and is not mixed into the primary joblib interface.
- Stage 04 SHAP preprocessing is identity: six Stage 03 expected ratings, in canonical order, are passed directly to the selected model. The background is a deterministic sample of 100 rows from `split=train` in `nodule_level_xml_table.csv` (4,221 training rows; seed 2026). The explanation is for the raw regression output, before schema-defined clipping/rounding.
- Stage 05 Grad-CAM differentiates the existing model's expected-rating scalar at the indexed target. It preserves signed raw maps and does not invert lobulation/spiculation. Review and clinical interpretation remain necessary.

## Commands/results

Executed in the active environment:

```powershell
.\l-env\Scripts\python.exe .\validate_pipeline.py --check-artifacts
.\l-env\Scripts\python.exe -m unittest discover -s tests -v
.\l-env\Scripts\python.exe -m unittest discover -s backend\Imaging\tests -v
```

The static artifact/schema check passed. The root suite passed 35 tests, including two SHAP tests and three Stage 06/07 integration tests. The imaging tracker/candidate suite passed 40 tests. A synthetic Stage 04 CLI invocation using the actual Stage 04 model completed and emitted six SHAP feature contributions with a maximum additivity residual below `2e-8`.

PowerShell parsing, changed-file compilation, and root schema JSON parsing passed.

## Real-data status

No real Stage 01–07 patient inference was run. The available `LIDC-IDRI-0001` output is an older schema 2.10 result with no current review metrics, montages, or attestation; the DICOM input directory has 135 slices/files as counted by the inventory, and Stage 02/03 outputs are absent. The old output also differs from previously supplied sample counts, so it was not overwritten or treated as a current reproducibility baseline.

Use `-RegenerateStage01` only when regeneration is intended; it creates a timestamped backup of schema-declared Stage 01 outputs. Then inspect every montage and create an attestation. Stage 02 onward is blocked until the attestation hashes match the current arrays and QC artifacts.

## Known limits and blockers

- No end-to-end model result, nodule count, or patient report is claimed.
- No binary malignancy classes or cancer probabilities are produced.
- Stage 04's report shows weak performance when driven by Stage 03-predicted features (MAE worse than the constant baseline and QWK approximately 0.016).
- The ViTDet3D remote model's first-run download was not attempted.
- CAM polarity is tied to the expected ordinal-rating gradient and is explicitly recorded, but not independently clinically validated.
- SHAP explains six tabular model inputs relative to the XML training-feature distribution; it does not explain CT voxels.
- The new human gate documents operator review, not clinical validation.
