# LungInsight v2 candidate-fusion utility — schema 2.6 compatibility update

## What happened in the PowerShell transcript

There were two separate errors:

1. From the correct `LungInsight_v2` directory, `lunginsight_candidate_fusion.py` rejected `meta_schema.json` version `2.6` because its loader had a hard-coded `schema_version == "2.3"` check. That was an unnecessary version-only restriction: the utility needs the canonical coordinate contract, not a specific version string.
2. After `cd ..`, the shell was in `C:\Users\emkap\Documents\Projects\LungInsight`, one directory above the active project. The relative path `backend\Imaging\lunginsight_candidate_fusion.py` therefore did not exist. Relative commands must be run from the `LungInsight_v2` root.

The revised helper accepts a schema version when it has the required coordinate definitions and validates them structurally. It still fails closed if array axes, the DICOM physical coordinate system, or the Stage 02 CT frame are unsupported or conflicting. It reads but never edits `meta_schema.json`.

Your posted compatibility-audit output reports schema `2.6`, `full_pipeline_compatible_from_supplied_files: true`, and no interface-presence problems. The audit itself explicitly says this does not verify checkpoint loading, model validity, clinical validity, or GPU inference.

## Files in this replacement bundle

- `backend/Imaging/lunginsight_candidate_fusion.py` — corrected schema loader plus deterministic detector merging and lung/heart filtering.
- `backend/Imaging/tests/test_lunginsight_candidate_fusion.py` — synthetic merge/filter tests plus schema-contract tests for version `2.3`, version `2.6`, and inconsistent coordinate definitions.
- `tools/audit_lunginsight_compatibility.py` — included unchanged for repeatable local interface audit.
- `COMPATIBILITY_FINDINGS.json` — records the specific failure and validation boundaries.

The bundle does not replace the numbered Stage 01–07 scripts and does not modify `meta_schema.json`.

## Install from Downloads

Run in PowerShell. If your downloaded filename differs, change `$zip` to the exact file name in Downloads.

```powershell
$zip = Join-Path $env:USERPROFILE 'Downloads\LungInsight_v2_pipeline_corrections_schema_2_6_compat.zip'
$workspace = 'C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2'

if (-not (Test-Path -LiteralPath $zip)) { throw "ZIP not found: $zip" }
if (-not (Test-Path -LiteralPath $workspace)) { throw "Workspace not found: $workspace" }
Expand-Archive -LiteralPath $zip -DestinationPath $workspace -Force
Set-Location -LiteralPath $workspace
```

## Test and compile

```powershell
$workspace = 'C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2'
Set-Location -LiteralPath $workspace
. .\l-env\Scripts\Activate.ps1

python -m unittest discover -s backend\Imaging\tests -v
python -m py_compile backend\Imaging\lunginsight_candidate_fusion.py tools\audit_lunginsight_compatibility.py
python tools\audit_lunginsight_compatibility.py --root .
```

The unit tests do not need a real CT or trained model. The audit may report success for interface presence, but that alone is not end-to-end model validation.

## Run the merge/filter audit against an existing patient

**Run from `LungInsight_v2`, not its parent `LungInsight`.** This command expects Stage 01 masks/meta and Stage 02 candidate JSONs already to exist. It does not run the numbered stages and does not trigger attribute, malignancy, Grad-CAM, SHAP, Stage 06, or Stage 07 inference.

```powershell
Set-Location -LiteralPath 'C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2'
$workspace = (Get-Location).Path
$patient = 'LIDC-IDRI-0001'
$patientDir = Join-Path $workspace "backend\output\$patient"
$vitdet = Join-Path $patientDir '02_detect_and_patch\candidates.json'
$log = Join-Path $patientDir '02_detect_and_patch\log_candidates_diagnostic.json'
$schema = Join-Path $workspace 'meta_schema.json'

foreach ($path in @(
    $schema,
    (Join-Path $patientDir 'volume_hu_cropped.npy'),
    (Join-Path $patientDir 'lung_mask.npy'),
    (Join-Path $patientDir 'heart_mask.npy'),
    (Join-Path $patientDir 'meta.json'),
    $vitdet,
    $log
)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required input not found: $path" }
}

python backend\Imaging\lunginsight_candidate_fusion.py `
    --patient-dir $patientDir `
    --schema $schema `
    --vitdet-candidates $vitdet `
    --log-candidates $log `
    --output-dir (Join-Path $patientDir '02_candidate_fusion_audit')
```

Expected outputs are `merged_candidates.json`, `filtered_candidates.json`, `rejected_candidates.json`, and `merge_filter_report.json` inside `backend\output\<patient>\02_candidate_fusion_audit`. The report includes LoG and ViTDet3D counts, union count, post-deduplication count, post-filter accepted/rejected counts, and rejection reasons.

## Validation boundary and remaining integration work

The merge/filter utility is an intermediate audit tool, not yet wired into Stage 02 patch extraction or the Stage 03 manifest boundary. Do not treat its filtered JSON as a replacement for the patch manifest. The supplied source/schema alone does not establish the actual model checkpoint load, Grad-CAM scalar semantics for Lobulation/Spiculation, SHAP background/explainer, or a schema-defined benign/malignant threshold. No binary clinical labels or explanations are fabricated by this utility.

The current local schema `2.6` file was not included as a separate attachment in this troubleshooting turn, so the new loader is validated against a synthetic `2.6` schema fixture and fails on contradictory coordinate definitions. It must also pass against your actual workspace schema when you rerun the command. If its coordinate fields intentionally changed, do not bypass validation; reconcile the coordinate frame with the schema first.
