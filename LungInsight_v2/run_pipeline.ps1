[CmdletBinding()]
param(
    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$PatientId = 'LIDC-IDRI-0001',
    [Parameter()]
    [switch]$ResumeFromStage02,
    [Parameter()]
    [switch]$RegenerateStage01,
    [Parameter()]
    [string]$InputPatientPath,
    [Parameter()]
    [string]$BrockInputs,
    [Parameter()]
    [switch]$NoGui
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonVenv = Join-Path $RepoRoot 'l-env\Scripts\python.exe'
if (Test-Path $PythonVenv) {
    $Python = $PythonVenv
} else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) { throw 'Python was not found. Activate the project virtual environment or install Python.' }
    $Python = $PythonCommand.Source
}

$SchemaPath = Join-Path $RepoRoot 'meta_schema.json'
if (-not (Test-Path -LiteralPath $SchemaPath -PathType Leaf)) { throw "Canonical meta_schema.json is missing: $SchemaPath" }
try { $Schema = Get-Content -LiteralPath $SchemaPath -Raw -Encoding UTF8 | ConvertFrom-Json }
catch { throw "Could not parse meta_schema.json ($SchemaPath): $($_.Exception.Message)" }

function Invoke-PythonStep {
    param([Parameter(Mandatory=$true)][string]$Label, [Parameter(Mandatory=$true)][string[]]$Arguments)
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    $StartedAt = [DateTime]::UtcNow
    & $Python @Arguments
    $ExitCode = $LASTEXITCODE
    $RunSteps.Add([ordered]@{
        label = $Label
        arguments = @($Arguments)
        started_at_utc = $StartedAt.ToString('o')
        completed_at_utc = [DateTime]::UtcNow.ToString('o')
        exit_code = $ExitCode
    })
    if ($ExitCode -ne 0) { throw "$Label failed with exit code $ExitCode. Fix the reported error before continuing; downstream stages were not run." }
}
function Require-File {
    param([Parameter(Mandatory=$true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Expected output is missing: $Path. The upstream stage may have failed or used a different output path." }
}
function Resolve-PatientPath {
    param([Parameter(Mandatory=$true)][string]$RelativePath)
    return Join-Path (Join-Path $RepoRoot (Join-Path ([string]$Schema.paths.output_root) $PatientId)) $RelativePath
}

Push-Location $RepoRoot
$RunStartedAt = [DateTime]::UtcNow
$RunSteps = [System.Collections.Generic.List[object]]::new()
$RunFailure = $null
$ContractFinalizationFailure = $null
$RunStatus = 'failed'
if ($ResumeFromStage02 -and $RegenerateStage01) {
    Pop-Location
    throw '-ResumeFromStage02 and -RegenerateStage01 are mutually exclusive.'
}
try {
    Invoke-PythonStep 'Static schema and stage-contract validation' @('.\validate_pipeline.py')
    Invoke-PythonStep 'Check required local model artifacts' @('.\validate_pipeline.py', '--check-artifacts')
    Invoke-PythonStep 'Schema helper unit tests' @('-m', 'unittest', 'discover', '-s', 'tests', '-v')
    Invoke-PythonStep 'Imaging tracker and candidate-gate tests' @('-m', 'unittest', 'discover', '-s', 'backend\Imaging\tests', '-v')

    $PatientOutput = Join-Path ([string]$Schema.paths.output_root) $PatientId
    if (-not $ResumeFromStage02) {
        if (Test-Path -LiteralPath $PatientOutput -PathType Container) {
            if (-not $RegenerateStage01) {
                throw "Refusing to overwrite existing patient outputs at $PatientOutput. Back up the outputs, then explicitly use -RegenerateStage01; or use -ResumeFromStage02 only when existing Stage 01 artifacts pass automated validation."
            }
            $BackupName = 'pre_stage01_backup_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
            $BackupDirectory = Join-Path $PatientOutput $BackupName
            New-Item -ItemType Directory -Path $BackupDirectory | Out-Null
            foreach ($Property in $Schema.stage_01_hu_and_crop.filenames.PSObject.Properties) {
                $ConfiguredName = [string]$Property.Value
                if ($Property.Name -eq 'heart_mask_review_montage_pattern') {
                    $MontageFilter = $ConfiguredName.Replace('{page:03d}', '*')
                    Get-ChildItem -LiteralPath $PatientOutput -Filter $MontageFilter -File -ErrorAction SilentlyContinue |
                        Copy-Item -Destination $BackupDirectory
                } elseif ($ConfiguredName -and -not $ConfiguredName.Contains('{')) {
                    $ExistingStage01Output = Join-Path $PatientOutput $ConfiguredName
                    if (Test-Path -LiteralPath $ExistingStage01Output -PathType Leaf) {
                        Copy-Item -LiteralPath $ExistingStage01Output -Destination $BackupDirectory
                    }
                }
            }
            Write-Host "Backed up existing Stage 01 files to $BackupDirectory" -ForegroundColor Yellow
        }
        $Stage01Input = if ($InputPatientPath) { $InputPatientPath } else { $PatientId }
        Invoke-PythonStep 'Stage 01 - HU conversion, masks, crop, and review overlays' @('.\backend\Imaging\01_hu_and_crop.py', $Stage01Input)
    } else {
        Write-Host 'Resume requested: Stage 01 will not be rerun; existing Stage 01 artifacts must pass automated validation.' -ForegroundColor Yellow
    }
    Require-File (Join-Path $PatientOutput ([string]$Schema.stage_01_hu_and_crop.filenames.'meta.json'))
    Require-File (Join-Path $PatientOutput ([string]$Schema.stage_01_hu_and_crop.filenames.'volume_hu_cropped.npy'))
    Require-File (Join-Path $PatientOutput ([string]$Schema.stage_01_hu_and_crop.filenames.'heart_mask_review_metrics.json'))
    $Stage02Arguments = @('.\backend\Imaging\02_detect_and_patch.py', $PatientId, '--device', 'cpu')
    Invoke-PythonStep 'Stage 02 - fused detection, lung-mask gate, and patch extraction (CPU)' $Stage02Arguments
    Require-File (Join-Path $PatientOutput ([string]$Schema.stage_02_detect_and_patch.patch_outputs.manifest))

    Invoke-PythonStep 'Stage 03 - frozen attribute inference for selected fused candidates (CPU)' @('.\backend\Imaging\03_nodule_classification.py', '--patient-id', $PatientId, '--device', 'cpu')
    $Stage03Dir = Join-Path $PatientOutput ([string]$Schema.paths.stage_03.directory)
    $PrimaryStage03 = Join-Path $Stage03Dir ([string]$Schema.stage_03_nodule_classification.outputs.predictions_json)
    Require-File $PrimaryStage03

    $Stage04Arguments = @('.\backend\Imaging\04_malignancy_prediction.py', $PatientId)
    if ($BrockInputs) {
        if (-not (Test-Path -LiteralPath $BrockInputs -PathType Leaf)) { throw "Brock input envelope does not exist: $BrockInputs" }
        $Stage04Arguments += @('--brock-inputs', (Resolve-Path -LiteralPath $BrockInputs).Path)
    }
    Invoke-PythonStep 'Stage 04 - Brock/PanCan malignancy-risk inference' $Stage04Arguments
    $Stage04Dir = Join-Path $PatientOutput ([string]$Schema.paths.stage_04.directory)
    $PrimaryStage04 = Join-Path $Stage04Dir ([string]$Schema.paths.stage_04.output_json)
    Require-File $PrimaryStage04

    $PrimaryManifest = Join-Path $PatientOutput ([string]$Schema.stage_02_detect_and_patch.patch_outputs.manifest)
    $PrimaryCamRoot = Join-Path $PatientOutput ([string]$Schema.paths.stage_05.directory)
    Invoke-PythonStep 'Stage 05 - Grad-CAM for all eligible fused candidates (CPU)' @('.\backend\Imaging\05_gradcam.py', '--patient-id', $PatientId, '--manifest', $PrimaryManifest, '--output-dir', $PrimaryCamRoot, '--batch-limit', '0')
    Require-File (Join-Path $PrimaryCamRoot ([string]$Schema.stage_05_gradcam.filenames.acceptance_audit))

    Invoke-PythonStep 'Stage 06 - full-CT Grad-CAM projection for authoritative patches' @('.\backend\Imaging\06_full_ct_gradcam.py', '--patient-id', $PatientId)
    $PrimaryStage06 = Join-Path $PatientOutput ([string]$Schema.paths.stage_06.directory)
    Require-File (Join-Path $PrimaryStage06 ([string]$Schema.stage_06_full_ct_gradcam.report_filename))
    $PrimaryNoduleRecords = Join-Path $PrimaryStage06 ([string]$Schema.stage_06_full_ct_gradcam.record_filename)
    Require-File $PrimaryNoduleRecords

    $Stage07Args = @('.\backend\Imaging\07_visualize.py', '--patient-id', $PatientId)
    if ($NoGui) { $Stage07Args += '--no-gui' }
    Invoke-PythonStep 'Stage 07 - mask-filtered nodule report and interactive CT slider' $Stage07Args
    $Stage07Root = Join-Path $PatientOutput ([string]$Schema.paths.stage_07.directory)
    Require-File (Join-Path $Stage07Root ([string]$Schema.paths.stage_07.report_json))
    Require-File (Join-Path $Stage07Root ([string]$Schema.paths.stage_07.report_csv))
    Require-File (Join-Path $Stage07Root ([string]$Schema.paths.stage_07.excluded_csv))

    Write-Host "`nAll imaging stages completed; publishing final run status and contract."
    $RunStatus = 'completed'
}
catch {
    $RunFailure = $_.Exception.Message
    throw
}
finally {
    $SummaryRoot = Join-Path $RepoRoot ([string]$Schema.paths.output_root)
    $SummaryDirectory = Join-Path $SummaryRoot $PatientId
    if (Test-Path -LiteralPath $SummaryDirectory -PathType Container) {
        $SummaryPath = Join-Path $SummaryDirectory ([string]$Schema.paths.pipeline_run_summary_filename)
        $Stage02Root = Join-Path $SummaryDirectory ([string]$Schema.paths.stage_02.directory)
        $Stage03Root = Join-Path $SummaryDirectory ([string]$Schema.paths.stage_03.directory)
        $Stage04Root = Join-Path $SummaryDirectory ([string]$Schema.paths.stage_04.directory)
        $Stage05Root = Join-Path $SummaryDirectory ([string]$Schema.paths.stage_05.directory)
        $Stage06Root = Join-Path $SummaryDirectory ([string]$Schema.paths.stage_06.directory)
        $Stage07Root = Join-Path $SummaryDirectory ([string]$Schema.paths.stage_07.directory)
        $ExpectedOutputs = @(
            (Join-Path $SummaryDirectory ([string]$Schema.stage_01_hu_and_crop.filenames.'meta.json')),
            (Join-Path $SummaryDirectory ([string]$Schema.stage_01_hu_and_crop.filenames.'volume_hu_cropped.npy')),
            (Join-Path $Stage02Root ([string]$Schema.stage_02_detect_and_patch.filenames.candidate_filter_report)),
            (Join-Path $SummaryDirectory ([string]$Schema.stage_02_detect_and_patch.patch_outputs.manifest)),
            (Join-Path $Stage03Root ([string]$Schema.stage_03_nodule_classification.outputs.predictions_json)),
            (Join-Path $Stage04Root ([string]$Schema.paths.stage_04.output_json)),
            (Join-Path $Stage05Root ([string]$Schema.stage_05_gradcam.filenames.acceptance_audit)),
            (Join-Path $Stage06Root ([string]$Schema.stage_06_full_ct_gradcam.report_filename)),
            (Join-Path $Stage06Root ([string]$Schema.stage_06_full_ct_gradcam.record_filename)),
            (Join-Path $SummaryDirectory ([string]$Schema.imaging_clinical_contract.filename)),
            (Join-Path $Stage07Root ([string]$Schema.paths.stage_07.report_json)),
            (Join-Path $Stage07Root ([string]$Schema.paths.stage_07.report_csv))
        )
        $Counters = [ordered]@{}
        $FilterReportPath = Join-Path $Stage02Root ([string]$Schema.stage_02_detect_and_patch.filenames.candidate_filter_report)
        if (Test-Path -LiteralPath $FilterReportPath -PathType Leaf) {
            $FilterReport = Get-Content -LiteralPath $FilterReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $Counters.authoritative_candidates_input = $FilterReport.input_count
            $Counters.authoritative_candidates_accepted = $FilterReport.accepted_count
            $Counters.authoritative_candidates_rejected = $FilterReport.rejected_count
            $Counters.authoritative_rejection_reasons = $FilterReport.rejection_counts
        }
        $PatchManifestPath = Join-Path $SummaryDirectory ([string]$Schema.stage_02_detect_and_patch.patch_outputs.manifest)
        if (Test-Path -LiteralPath $PatchManifestPath -PathType Leaf) {
            $PatchManifest = Get-Content -LiteralPath $PatchManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $Counters.authoritative_patches = $PatchManifest.patch_count
        }
        $Stage03Path = Join-Path $Stage03Root ([string]$Schema.stage_03_nodule_classification.outputs.predictions_json)
        if (Test-Path -LiteralPath $Stage03Path -PathType Leaf) {
            $Stage03Data = Get-Content -LiteralPath $Stage03Path -Raw -Encoding UTF8 | ConvertFrom-Json
            $Counters.authoritative_attribute_predictions = $Stage03Data.n_patches
        }
        $Stage04Path = Join-Path $Stage04Root ([string]$Schema.paths.stage_04.output_json)
        if (Test-Path -LiteralPath $Stage04Path -PathType Leaf) {
            $Stage04Data = Get-Content -LiteralPath $Stage04Path -Raw -Encoding UTF8 | ConvertFrom-Json
            $BrockPredictions = @($Stage04Data.predictions)
            $Counters.authoritative_brock_predictions = $BrockPredictions.Count
            foreach ($Status in @('available', 'incomplete', 'invalid_input', 'error')) {
                $Counters["authoritative_brock_$Status"] = @(
                    $BrockPredictions | Where-Object { $_.brock.status -eq $Status }
                ).Count
            }
        }
        $Stage06RecordPath = Join-Path $Stage06Root ([string]$Schema.stage_06_full_ct_gradcam.record_filename)
        if (Test-Path -LiteralPath $Stage06RecordPath -PathType Leaf) {
            $Stage06Data = Get-Content -LiteralPath $Stage06RecordPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $Counters.authoritative_stage06_nodule_records = $Stage06Data.accepted_record_count
        }
        $Stage07ReportPath = Join-Path $Stage07Root ([string]$Schema.paths.stage_07.report_json)
        if (Test-Path -LiteralPath $Stage07ReportPath -PathType Leaf) {
            $Stage07Data = Get-Content -LiteralPath $Stage07ReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $Counters.stage07_visualizations = $Stage07Data.visualization_count
            $Counters.stage07_in_lung_records = $Stage07Data.nodule_count_center_inside_lung_mask
        }

        $ModelIdentities = [System.Collections.Generic.List[object]]::new()
        $CheckpointRoot = Join-Path $RepoRoot ([string]$Schema.stage_03_nodule_classification.model.checkpoint_root)
        foreach ($Group in $Schema.stage_03_nodule_classification.model.checkpoint_groups.PSObject.Properties) {
            $CheckpointPath = Join-Path (Join-Path $CheckpointRoot $Group.Name) 'best.pth'
            $CheckpointIdentity = [ordered]@{ model_group = $Group.Name; path = $CheckpointPath; sha256 = $null }
            if (Test-Path -LiteralPath $CheckpointPath -PathType Leaf) {
                $CheckpointIdentity.sha256 = (Get-FileHash -LiteralPath $CheckpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
            }
            $ModelIdentities.Add($CheckpointIdentity)
        }
        $ModelIdentities.Add([ordered]@{
            model = [string]$Schema.stage_04_malignancy_prediction.model_name
            model_version = [string]$Schema.stage_04_malignancy_prediction.model_version
            formula_version = [string]$Schema.stage_04_malignancy_prediction.formula_version
            path = $null
            sha256 = $null
        })

        $RunSummary = [ordered]@{
            schema_version = 'pipeline-run-summary-v1'
            meta_schema_version = [string]$Schema.schema_version
            patient_id = $PatientId
            status = $RunStatus
            started_at_utc = $RunStartedAt.ToString('o')
            completed_at_utc = [DateTime]::UtcNow.ToString('o')
            python_executable = $Python
            powershell_version = $PSVersionTable.PSVersion.ToString()
            resumed_from_stage_02 = [bool]$ResumeFromStage02
            regenerated_stage01_with_backup = [bool]$RegenerateStage01
            inputs = [ordered]@{
                dicom_patient_directory = Join-Path (Join-Path $RepoRoot ([string]$Schema.stage_01_hu_and_crop.input_root_from_project_root)) $PatientId
                stage01_output_directory = Join-Path $SummaryRoot $PatientId
                schema = (Join-Path $RepoRoot 'meta_schema.json')
                brock_inputs = if ($BrockInputs) { (Resolve-Path -LiteralPath $BrockInputs).Path } else { $null }
            }
            outputs = @($ExpectedOutputs | ForEach-Object {
                [ordered]@{ path = $_; exists = [bool](Test-Path -LiteralPath $_ -PathType Leaf) }
            })
            counters = $Counters
            model_identities = @($ModelIdentities.ToArray())
            steps = @($RunSteps.ToArray())
            failure = $RunFailure
        }
        $RunSummary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8
        Write-Host "Run summary: $SummaryPath"
        if ($RunStatus -eq 'completed') {
            $ContractFinalizer = 'import sys; from pathlib import Path; from backend.Imaging.api.contract import create_contract_file; root = Path(sys.argv[1]); create_contract_file(root, sys.argv[2], Path(sys.argv[3]), processing_status="completed")'
            try {
                Invoke-PythonStep 'Publish completed Imaging-to-Clinical contract' @(
                    '-c',
                    $ContractFinalizer,
                    $SummaryDirectory,
                    $PatientId,
                    $Stage06RecordPath
                )
            } catch {
                $RunStatus = 'failed'
                $ContractFinalizationFailure = $_.Exception.Message
                $RunFailure = $ContractFinalizationFailure
            }
            $RunSummary.status = $RunStatus
            $RunSummary.failure = $RunFailure
            $RunSummary.steps = @($RunSteps.ToArray())
            $RunSummary.completed_at_utc = [DateTime]::UtcNow.ToString('o')
            $RunSummary.outputs = @($ExpectedOutputs | ForEach-Object {
                [ordered]@{ path = $_; exists = [bool](Test-Path -LiteralPath $_ -PathType Leaf) }
            })
            $RunSummary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8
            if ($RunStatus -eq 'completed') {
                Write-Host "`nPipeline completed for $PatientId." -ForegroundColor Green
                Write-Host "Nodule report: $(Join-Path $Stage07Root ([string]$Schema.paths.stage_07.report_json))"
                Write-Host "Nodule CSV:    $(Join-Path $Stage07Root ([string]$Schema.paths.stage_07.report_csv))"
                Write-Host "Excluded CSV:  $(Join-Path $Stage07Root ([string]$Schema.paths.stage_07.excluded_csv))"
            }
        }
    }
    Pop-Location
    if ($ContractFinalizationFailure) {
        throw "Completed pipeline contract finalization failed: $ContractFinalizationFailure"
    }
}
