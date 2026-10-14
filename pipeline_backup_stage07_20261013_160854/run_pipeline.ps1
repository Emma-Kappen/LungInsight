[CmdletBinding()]
param(
    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$PatientId = 'LIDC-IDRI-0001',
    [Parameter()]
    [ValidateRange(0, 2147483647)]
    [int]$LogPatchLimit = 0
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonVenv = Join-Path $RepoRoot 'l-env\Scripts\python.exe'
if (Test-Path $PythonVenv) {
    $Python = $PythonVenv
} else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        throw 'Python was not found. Activate the project virtual environment or install Python.'
    }
    $Python = $PythonCommand.Source
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory=$true)][string]$Label,
        [Parameter(Mandatory=$true)][string[]]$Arguments
    )
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE. Fix the reported error before continuing; downstream stages were not run."
    }
}

function Require-File {
    param([Parameter(Mandatory=$true)][string]$RelativePath)
    $Resolved = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $Resolved -PathType Leaf)) {
        throw "Expected output is missing: $Resolved. The upstream stage may have failed or used a different output path."
    }
}

Push-Location $RepoRoot
try {
    Invoke-PythonStep 'Static schema and stage-contract validation' @('.\validate_pipeline.py')
    Invoke-PythonStep 'Check required local model artifacts' @('.\validate_pipeline.py', '--check-artifacts')
    Invoke-PythonStep 'Schema helper unit tests' @('-m', 'unittest', 'discover', '-s', 'tests', '-v')

    $PatientOutput = "backend\output\$PatientId"
    Invoke-PythonStep 'Stage 01 - HU conversion and crop' @('.\backend\Imaging\01_hu_and_crop.py', $PatientId)
    Require-File "$PatientOutput\meta.json"
    Require-File "$PatientOutput\volume_hu_cropped.npy"

    $Stage02Arguments = @('.\backend\Imaging\02_detect_and_patch.py', $PatientId, '--device', 'cpu')
    if ($LogPatchLimit -gt 0) {
        $Stage02Arguments += @('--log-patch-limit', [string]$LogPatchLimit)
    }
    Invoke-PythonStep 'Stage 02 - detection and patch extraction (CPU; LoG patches are separate)' $Stage02Arguments
    Require-File "$PatientOutput\02_detect_and_patch\patches\patches_manifest.json"
    Require-File "$PatientOutput\02_detect_and_patch\log_diagnostic_patches\log_patches_manifest.json"

    Invoke-PythonStep 'Stage 03 - frozen attribute inference (CPU; ViTDet3D + separate LoG diagnostics)' @('.\backend\Imaging\03_nodule_classification.py', '--patient-id', $PatientId, '--device', 'cpu')
    Require-File "$PatientOutput\03_nodule_classification\predictions.json"
    Require-File "$PatientOutput\03_nodule_classification\log_diagnostic_predictions.json"

    Invoke-PythonStep 'Stage 04 - primary frozen malignancy-rating inference' @('.\backend\Imaging\04_malignancy_prediction.py', $PatientId)
    Require-File "$PatientOutput\04_malignancy_prediction\malignancy_predictions.json"

    Invoke-PythonStep 'Stage 05 - Grad-CAM for patch 000 (CPU)' @('.\backend\Imaging\05_gradcam.py', '--patient-id', $PatientId)
    Require-File "$PatientOutput\05_explainability\gradcam_rainbow_final\gradcam_subtlety_rainbow_scaled.npy"

    Invoke-PythonStep 'Stage 06 - full-CT Grad-CAM projection' @('.\backend\Imaging\06_full_ct_gradcam.py', '--patient-id', $PatientId)
    Require-File "$PatientOutput\06_full_ct_gradcam_rainbow\full_ct_gradcam_report.json"

    Write-Host "`nPipeline completed for $PatientId." -ForegroundColor Green
    if ($LogPatchLimit -eq 0) { Write-Host 'All LoG diagnostic candidates were selected for Stage 03.' } else { Write-Host "LoG diagnostic patch limit: $LogPatchLimit" }
    Write-Host "Note: Stage 05 generates Grad-CAM for patch 000 by default. Stage 06 reports other patches as missing CAMs until maps are generated for them."
}
finally {
    Pop-Location
}
