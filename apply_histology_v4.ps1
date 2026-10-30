[CmdletBinding()]
param(
    [switch]$InstallSklearn
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Required = @(
    "backend\Clinical\inference.py",
    "backend\Clinical\schemas\clinical_prediction_v1.schema.json",
    "frontend\app.js",
    "requirements.txt"
)
foreach ($Relative in $Required) {
    if (-not (Test-Path (Join-Path $Root $Relative))) {
        throw "Run this script from the extracted patch at the LungInsight project root. Missing: $Relative"
    }
}

$PatchFiles = @(
    "backend\Clinical\histology_v4.py",
    "backend\Clinical\inference.py",
    "backend\Clinical\schemas\clinical_prediction_v1.schema.json",
    "backend\Clinical\research_histology_v4\cancer-type-luad-lusc-research-v4.joblib",
    "backend\Clinical\research_histology_v4\MODEL_CARD.txt",
    "backend\Clinical\research_histology_v4\model_metrics.json",
    "backend\Clinical\research_histology_v4\data_quality_audit.json",
    "backend\Clinical\tests\test_histology_v4.py",
    "frontend\app.js",
    "requirements.txt"
)
foreach ($Relative in $PatchFiles) {
    if (-not (Test-Path (Join-Path $PSScriptRoot $Relative))) {
        throw "Patch is incomplete; missing $Relative"
    }
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Backup = Join-Path $Root "diagnostics\histology_v4_integration_backup_$Stamp"
New-Item -ItemType Directory -Path $Backup -Force | Out-Null

foreach ($Relative in $PatchFiles) {
    $Target = Join-Path $Root $Relative
    if (Test-Path $Target) {
        $BackupTarget = Join-Path $Backup $Relative
        New-Item -ItemType Directory -Path (Split-Path $BackupTarget -Parent) -Force | Out-Null
        Copy-Item -LiteralPath $Target -Destination $BackupTarget -Force
    }
}

foreach ($Relative in $PatchFiles) {
    $Source = Join-Path $PSScriptRoot $Relative
    $Target = Join-Path $Root $Relative
    New-Item -ItemType Directory -Path (Split-Path $Target -Parent) -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Target -Force
}

$Python = Join-Path $Root "l-env\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}
$SklearnVersion = & $Python -c "import sklearn; print(sklearn.__version__)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Could not query scikit-learn using $Python."
} elseif ($SklearnVersion.Trim() -ne "1.9.1") {
    if ($InstallSklearn) {
        & $Python -m pip install "scikit-learn==1.9.1"
        if ($LASTEXITCODE -ne 0) { throw "Could not install the artifact's required scikit-learn version." }
    } else {
        Write-Warning "The v4 artifact was serialized with scikit-learn 1.9.1; this environment has $($SklearnVersion.Trim())."
        Write-Warning "The v4 model will safely report unavailable until the runtime matches. Re-run with -InstallSklearn to install the pinned version."
    }
}

Write-Host ""
Write-Host "Histology v4 integration files copied." -ForegroundColor Green
Write-Host "Backup: $Backup"
Write-Host "Existing v2/v3 and survival artifacts were not modified."
Write-Host ""
Write-Host "Next: run focused tests:"
Write-Host "& `"$Python`" -m pytest backend\Clinical\tests\test_histology_v4.py -q"
Write-Host "Then restart the LungInsight API and submit a new case."
