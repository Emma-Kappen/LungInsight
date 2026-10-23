[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Unit', 'Static', 'Acceptance')]
    [string]$Mode,
    [Parameter()]
    [string]$PatientId = 'LIDC-IDRI-0007'
)

$ErrorActionPreference = 'Continue'
Set-StrictMode -Version Latest
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonVenv = Join-Path $RepoRoot 'l-env\Scripts\python.exe'
$NestedPythonVenv = Join-Path $RepoRoot 'LungInsight_v2\l-env\Scripts\python.exe'
if (Test-Path -LiteralPath $PythonVenv -PathType Leaf) {
    $Python = $PythonVenv
} elseif (Test-Path -LiteralPath $NestedPythonVenv -PathType Leaf) {
    $Python = $NestedPythonVenv
} else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        Write-Error 'Python was not found.'; exit 2
    }
    $Python = $PythonCommand.Source
}
Write-Host "Python executable: $Python" -ForegroundColor DarkGray
$TestLogRoot = Join-Path $RepoRoot 'backend\output\.test_runs'
New-Item -ItemType Directory -Path $TestLogRoot -Force | Out-Null
$Stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$LogPath = Join-Path $TestLogRoot "$Mode-$Stamp.log"

Push-Location $RepoRoot
try {
    switch ($Mode) {
        'Unit' {
            Write-Host "Unit tests (no CT processing, no model inference): $LogPath" -ForegroundColor Cyan
            & $Python -m unittest discover -s tests -v 2>&1 | Tee-Object -FilePath $LogPath
            if ($LASTEXITCODE -ne 0) { exit 1 }
            & $Python -m unittest discover -s backend\Imaging\tests -v 2>&1 | Tee-Object -FilePath $LogPath -Append
            exit $LASTEXITCODE
        }
        'Static' {
            Write-Host "Static schema and stage-contract validation (does not execute CT processing or model inference): $LogPath" -ForegroundColor Cyan
            & $Python .\validate_pipeline.py 2>&1 | Tee-Object -FilePath $LogPath
            if ($LASTEXITCODE -ne 0) { exit 1 }
            & $Python .\validate_pipeline.py --check-artifacts 2>&1 | Tee-Object -FilePath $LogPath -Append
            exit $LASTEXITCODE
        }
        'Acceptance' {
            Write-Host "End-to-end acceptance for ${PatientId}: runs the production pipeline, then validates its artifacts. Log: $LogPath" -ForegroundColor Cyan
            & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot 'run_pipeline.ps1') -PatientId $PatientId 2>&1 |
                Tee-Object -FilePath $LogPath
            if ($LASTEXITCODE -ne 0) {
                Write-Host "ACCEPTANCE FAILED: production pipeline exited non-zero. This is a production-stage failure, not a test failure." -ForegroundColor Red
                exit 1
            }
            $Check = @'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("backend/Imaging").resolve()))
from api.contract import validate_contract
from pipeline_schema import load_schema, get_required
schema = load_schema(schema_path=Path("meta_schema.json").resolve())
root = Path(get_required(schema, "paths.output_root", expected_type=str)) / sys.argv[1]
contract_path = root / get_required(schema, "imaging_clinical_contract.filename", expected_type=str)
contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
validate_contract(contract)
print(f"Contract valid: {contract_path}")
'@
            $Check | & $Python - $PatientId 2>&1 | Tee-Object -FilePath $LogPath -Append
            if ($LASTEXITCODE -ne 0) {
                Write-Host 'ACCEPTANCE FAILED: contract validation failed.' -ForegroundColor Red
                exit 1
            }
            Write-Host 'ACCEPTANCE PASSED.' -ForegroundColor Green
            exit 0
        }
    }
} finally {
    Pop-Location
}
