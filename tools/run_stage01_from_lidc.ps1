[CmdletBinding()]
param(
    [Parameter()]
    [string]$PatientId = 'LIDC-IDRI-0001',

    [Parameter()]
    [string]$Workspace = 'C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2',

    # Optional override if the dataset is not at the documented default location.
    [Parameter()]
    [string]$LidcPatientPath = ''
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Workspace -PathType Container)) {
    throw "Workspace not found: $Workspace"
}
$Workspace = (Resolve-Path -LiteralPath $Workspace).Path

if ([string]::IsNullOrWhiteSpace($LidcPatientPath)) {
    # In 01_hu_and_crop.py, PROJECT_ROOT is the parent of LungInsight_v2.
    $projectRoot = Split-Path -Parent $Workspace
    $LidcPatientPath = Join-Path $projectRoot "datasets\LIDC\lidc_idri\$PatientId"
}
if (-not (Test-Path -LiteralPath $LidcPatientPath -PathType Container)) {
    throw @"
LIDC patient directory not found: $LidcPatientPath
The supplied Stage 01 script documents this layout:
  <LungInsight project root>\datasets\LIDC\lidc_idri\<PatientID>\<Study>\<Series>\<DICOM files>
If your dataset is stored elsewhere, rerun with -LidcPatientPath 'D:\full\path\to\LIDC-IDRI-0001'.
"@
}
$LidcPatientPath = (Resolve-Path -LiteralPath $LidcPatientPath).Path

$stage01 = Join-Path $Workspace 'backend\Imaging\01_hu_and_crop.py'
$schema = Join-Path $Workspace 'meta_schema.json'
if (-not (Test-Path -LiteralPath $stage01 -PathType Leaf)) {
    throw "Stage 01 script not found: $stage01"
}
if (-not (Test-Path -LiteralPath $schema -PathType Leaf)) {
    throw "Canonical schema not found: $schema"
}

$venvPython = Join-Path $Workspace 'l-env\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $python = $venvPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $pythonCommand) {
        throw 'Python was not found. Activate the project virtual environment or install Python.'
    }
    $python = $pythonCommand.Source
    Write-Warning "Project venv Python not found at $venvPython; using $python"
}

Write-Host "Workspace:        $Workspace"
Write-Host "Patient ID:       $PatientId"
Write-Host "LIDC input tree:  $LidcPatientPath"
Write-Host "Stage 01 script:  $stage01"
Write-Host ''
Write-Host 'Running Stage 01: DICOM CT -> HU volume -> lung/heart masks -> cropped outputs.'

Push-Location $Workspace
try {
    & $python $stage01 $LidcPatientPath
    if ($LASTEXITCODE -ne 0) {
        throw "Stage 01 exited with code $LASTEXITCODE. See the Python error above."
    }
} finally {
    Pop-Location
}

Write-Host ''
Write-Host 'Stage 01 returned successfully.'
Write-Host 'Use the [done] Output directory printed above as the authoritative output location.'
Write-Host 'Expected outputs include volume_hu_cropped.npy, lung_mask.npy, heart_mask.npy, volume_hu_heart_excluded.npy, heart_tracking_diagnostics.json, and meta.json.'
Write-Host 'Stage 02 must run next to create candidates.json before candidate fusion can run.'
