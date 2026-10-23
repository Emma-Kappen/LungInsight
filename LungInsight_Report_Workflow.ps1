<#
LungInsight v2 â€” Report workflow audit and acceptance runner
Run from PowerShell. Safe by default: does not modify application source files.
It inventories the frontend/report/model paths, snapshots key files, runs available
syntax/tests, checks for expected report features, and writes a Markdown report.

Usage:
  Set-ExecutionPolicy -Scope Process Bypass
  .\LungInsight_Report_Workflow.ps1 -Mode Audit
  .\LungInsight_Report_Workflow.ps1 -Mode Test
  .\LungInsight_Report_Workflow.ps1 -Mode All

Optional:
  .\LungInsight_Report_Workflow.ps1 -ProjectRoot "C:\...\LungInsight_v2" -Mode All
  .\LungInsight_Report_Workflow.ps1 -Mode All -RunAllPytest
#>
[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [ValidateSet('Audit','Test','All')]
    [string]$Mode = 'All',
    [switch]$RunAllPytest,
    [switch]$SkipBackup
)

$ErrorActionPreference = 'Continue'
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$ArtifactRoot = Join-Path $ProjectRoot "artifacts\report_workflow_$Stamp"
$BackupRoot = Join-Path $ArtifactRoot 'source_snapshot'
$ReportPath = Join-Path $ArtifactRoot 'acceptance_audit.md'
$JsonPath = Join-Path $ArtifactRoot 'acceptance_audit.json'
$script:Findings = [System.Collections.Generic.List[object]]::new()
$script:Commands = [System.Collections.Generic.List[object]]::new()

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "ProjectRoot does not exist: $ProjectRoot"
}
New-Item -ItemType Directory -Path $ArtifactRoot -Force | Out-Null

function Add-Finding {
    param([string]$Area,[string]$Status,[string]$Detail,[string]$Evidence='')
    $script:Findings.Add([pscustomobject]@{
        area=$Area; status=$Status; detail=$Detail; evidence=$Evidence
    })
}
function Run-CapturedCommand {
    param([string]$Name,[string]$File,[string[]]$Arguments,[string]$WorkingDirectory=$ProjectRoot)
    $outFile = Join-Path $ArtifactRoot (($Name -replace '[^A-Za-z0-9_.-]','_') + '.log')
    $started = Get-Date
    Push-Location $WorkingDirectory
    try {
        $output = & $File @Arguments 2>&1
        $exitCode = $LASTEXITCODE
        $output | Out-String -Width 300 | Set-Content -LiteralPath $outFile -Encoding UTF8
        if ($null -eq $exitCode) { $exitCode = 0 }
        $status = if ($exitCode -eq 0) {'PASS'} else {'FAIL'}
        $script:Commands.Add([pscustomobject]@{
            name=$Name; command=("$File " + ($Arguments -join ' ')); exit_code=$exitCode
            status=$status; log=$outFile; started=$started.ToString('s')
        })
        Add-Finding -Area "Command: $Name" -Status $status -Detail "Exit code $exitCode" -Evidence $outFile
    } catch {
        $_ | Out-String | Set-Content -LiteralPath $outFile -Encoding UTF8
        $script:Commands.Add([pscustomobject]@{
            name=$Name; command=("$File " + ($Arguments -join ' ')); exit_code=-999
            status='ERROR' ; log=$outFile; started=$started.ToString('s')
        })
        Add-Finding -Area "Command: $Name" -Status 'ERROR' -Detail $_.Exception.Message -Evidence $outFile
    } finally { Pop-Location }
}
function Find-TextInFiles {
    param([string[]]$Files,[string[]]$Patterns,[string]$Area)
    $found = @()
    foreach ($f in $Files) {
        if (Test-Path -LiteralPath $f -PathType Leaf) {
            foreach ($pattern in $Patterns) {
                $matches = Select-String -LiteralPath $f -Pattern $pattern -AllMatches -ErrorAction SilentlyContinue
                foreach ($m in $matches) {
                    $found += "$($m.Path):$($m.LineNumber): $($m.Line.Trim())"
                }
            }
        }
    }
    if ($found.Count -gt 0) {
        Add-Finding -Area $Area -Status 'REVIEW' -Detail "$($found.Count) matching source line(s) found; inspect evidence." -Evidence ($found -join "`n")
    } else {
        Add-Finding -Area $Area -Status 'NOT FOUND' -Detail 'No matching text found in the inspected files. This is not proof the feature is absent.'
    }
}
function Get-TrackedFiles {
    $exclude = '\\(\\.git|\\.venv|venv|l-env|node_modules|__pycache__|artifacts|dist|build)\\'
    Get-ChildItem -LiteralPath $ProjectRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch $exclude -and $_.Length -lt 20MB }
}
function Write-TextReport {
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('# LungInsight v2 - Report workflow audit')
    $lines.Add('')
    $lines.Add("- Project root: $ProjectRoot")
    $lines.Add("- Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')")
    $lines.Add("- Mode: $Mode")
    $lines.Add("- Artifact directory: $ArtifactRoot")
    $lines.Add('')
    $lines.Add('## Summary')
    $lines.Add('')
    $lines.Add('| Status | Count |')
    $lines.Add('|---|---:|')
    foreach ($s in @('PASS','FAIL','ERROR','REVIEW','NOT FOUND','INFO','BLOCKED')) {
        $count = @($script:Findings | Where-Object { $_.status -eq $s }).Count
        if ($count -gt 0) { $lines.Add("| $s | $count |") }
    }
    $lines.Add('')
    $lines.Add('## Findings')
    $lines.Add('')
    foreach ($f in $script:Findings) {
        $lines.Add("### [$($f.status)] $($f.area)")
        $lines.Add('')
        $lines.Add($f.detail)
        if ($f.evidence) {
            $lines.Add('')
            $lines.Add('```text')
            $lines.Add([string]$f.evidence)
            $lines.Add('```')
        }
        $lines.Add('')
    }
    $lines.Add('## Commands executed')
    $lines.Add('')
    if ($script:Commands.Count -eq 0) {
        $lines.Add('No test commands were run in this mode.')
    }
    foreach ($c in $script:Commands) {
        $lines.Add("- **$($c.status)** $($c.command) - exit $($c.exit_code); log: $($c.log)")
    }
    $lines.Add('')
    $lines.Add('## Interpretation')
    $lines.Add('')
    $lines.Add('- REVIEW means source references were found; it does not prove runtime correctness.')
    $lines.Add('- NOT FOUND means no matching literal was found in inspected files; the feature may be implemented differently.')
    $lines.Add('- Passing syntax/tests does not replace visual inspection of the generated PDF.')
    $lines.Add('- No application source files are changed by this script.')
    $lines | Set-Content -LiteralPath $ReportPath -Encoding UTF8
    [pscustomobject]@{
        generated_at = (Get-Date).ToString('o')
        project_root = $ProjectRoot
        mode = $Mode
        artifact_root = $ArtifactRoot
        report = $ReportPath
        findings = @($script:Findings)
        commands = @($script:Commands)
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $JsonPath -Encoding UTF8
}
Write-Host "Project: $ProjectRoot"
Write-Host "Artifacts: $ArtifactRoot"

# Locate important files.
$relativeTargets = @(
 'frontend\index.html','frontend\styles.css','frontend\app.js',
 'frontend\package.json','package.json','playwright.config.js','playwright.config.ts',
 'pytest.ini','pyproject.toml','setup.cfg','requirements.txt',
 'backend\Imaging\meta_schema.json',
 'backend\Imaging\schemas\imaging_clinical_contract_v1.schema.json',
 'backend\Imaging\stages\04_malignancy_prediction.py',
 'backend\Imaging\stage04_shap.py',
 'backend\Clinical'
)
$targetFiles = @()
foreach ($rel in $relativeTargets) {
    $path = Join-Path $ProjectRoot $rel
    if (Test-Path -LiteralPath $path) {
        Add-Finding -Area "File present: $rel" -Status 'PASS' -Detail 'Path exists.'
        if (Test-Path -LiteralPath $path -PathType Leaf) { $targetFiles += $path }
    } else {
        Add-Finding -Area "File present: $rel" -Status 'NOT FOUND' -Detail 'Path not found at this expected location; inspect the repository layout.'
    }
}

$allFiles = @(Get-TrackedFiles)
Add-Finding -Area 'Repository inventory' -Status 'INFO' -Detail "$($allFiles.Count) files inventoried (excluding common environments, build output and files over 20 MB)."
$reportRelated = @($allFiles | Where-Object {
    $_.FullName -match '(?i)(frontend|report|histolog|survival|brock|shap|print|nodule|gradcam|imaging_contract|acceptance|test)'
})
$inventoryPath = Join-Path $ArtifactRoot 'relevant_file_inventory.txt'
$reportRelated | ForEach-Object { $_.FullName.Substring($ProjectRoot.Length).TrimStart('\') } |
    Set-Content -LiteralPath $inventoryPath -Encoding UTF8
Add-Finding -Area 'Relevant file inventory' -Status 'INFO' -Detail "$($reportRelated.Count) relevant-looking paths listed." -Evidence $inventoryPath

if (-not $SkipBackup) {
    $backupCandidates = @($allFiles | Where-Object {
        $_.FullName -match '(?i)\\frontend\\(index\.html|styles\.css|app\.js)$|histolog|shap|report|print|brock|survival|imaging_clinical_contract_v1\.schema\.json'
    })
    foreach ($f in $backupCandidates) {
        $rel = $f.FullName.Substring($ProjectRoot.Length).TrimStart('\')
        $dest = Join-Path $BackupRoot $rel
        New-Item -ItemType Directory -Path (Split-Path -Parent $dest) -Force | Out-Null
        Copy-Item -LiteralPath $f.FullName -Destination $dest -Force -ErrorAction SilentlyContinue
    }
    Add-Finding -Area 'Source snapshot' -Status 'INFO' -Detail "Copied matching frontend/report/model/schema files to $BackupRoot. No source edits were made."
}

# Capture frontend/report implementation references.
$frontFiles = @($allFiles | Where-Object { $_.FullName -match '(?i)\\frontend\\.*\.(html|css|js|ts|tsx|jsx)$' } | ForEach-Object FullName)
$reportPatterns = @{
 'Multi-nodule report and IDs' = @('noduleId','nodule_id','stable_nodule_id','nodules','candidate_id','nodule.*summary')
 'Nodule CT selection / bounding boxes' = @('bbox','bounding.?box','bbox_voxel','ct.?slice','slice.?index','volume_hu_cropped','candidate.*center','coordinates')
 'Brock assessment / explanation' = @('Brock','brock','malignancy','contribution','shap')
 'Histology window and classes' = @('histology','LUAD','LUSC','class.?probab','probabilit')
 'Survival panel' = @('survival','hazard','time.?to.?event')
 'Print layout / charts' = @('@media\s+print','window\.print','beforeprint','afterprint','print-color-adjust','page-break','break-inside','canvas','svg')
 'Resizable / collapsible windows' = @('resize','resizable','collapse','aria-expanded','details','summary')
}
foreach ($key in $reportPatterns.Keys) {
    Find-TextInFiles -Files $frontFiles -Patterns $reportPatterns[$key] -Area $key
}
$cssFiles = @($allFiles | Where-Object { $_.FullName -match '(?i)\.(css|scss)$' } | ForEach-Object FullName)
Find-TextInFiles -Files $cssFiles -Patterns @('@media\s+print','break-inside','page-break','@media','grid-template-columns','min-height','height:\s*100%') -Area 'Responsive and print CSS'
$backendFiles = @($allFiles | Where-Object { $_.FullName -match '(?i)\.(py|json|yaml|yml)$' } | ForEach-Object FullName)
Find-TextInFiles -Files $backendFiles -Patterns @('shap','TreeExplainer','KernelExplainer','Explainer','LUAD','LUSC','class_order','feature_names','histology') -Area 'Backend histology/SHAP implementation'
Find-TextInFiles -Files $backendFiles -Patterns @('reportable','excluded_reason','exclusion_reason','gate_reason','BLOCKED','candidate_acceptance_filter') -Area 'Pipeline inclusion/exclusion traceability'

# Detect toolchain.
$python = Get-Command python -ErrorAction SilentlyContinue
$node = Get-Command node -ErrorAction SilentlyContinue
$npm = Get-Command npm -ErrorAction SilentlyContinue
Add-Finding -Area 'Python executable' -Status $(if($python){'PASS'}else{'NOT FOUND'}) -Detail $(if($python){$python.Source}else{'python is not on PATH in this shell.'})
Add-Finding -Area 'Node executable' -Status $(if($node){'PASS'}else{'NOT FOUND'}) -Detail $(if($node){$node.Source}else{'Node.js is not installed or not on PATH; frontend syntax/browser tests may be unavailable.'})
Add-Finding -Area 'npm executable' -Status $(if($npm){'PASS'}else{'NOT FOUND'}) -Detail $(if($npm){$npm.Source}else{'npm is not installed or not on PATH.'})

if ($Mode -in @('Audit','All')) {
    # Record git state without modifying it.
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git -and (Test-Path (Join-Path $ProjectRoot '.git'))) {
        Run-CapturedCommand -Name 'git_status' -File $git.Source -Arguments @('status','--short') -WorkingDirectory $ProjectRoot
        Run-CapturedCommand -Name 'git_diff_stat' -File $git.Source -Arguments @('diff','--stat') -WorkingDirectory $ProjectRoot
    } else {
        Add-Finding -Area 'Git status' -Status 'INFO' -Detail 'Git unavailable or project root is not a Git worktree.'
    }
}

if ($Mode -in @('Test','All')) {
    if ($python) {
        # Python syntax compile of relevant backend scripts.
        $pyFiles = @($allFiles | Where-Object {
            $_.Extension -eq '.py' -and $_.FullName -match '(?i)(Imaging|Clinical|histolog|shap|report|test_)' -and
            $_.FullName -notmatch '(\\l-env\\|\\venv\\|\\__pycache__\\)'
        } | Select-Object -First 250)
        if ($pyFiles.Count -gt 0) {
            $compileScript = Join-Path $ArtifactRoot 'compile_relevant_python.py'
            @'
import py_compile, sys
from pathlib import Path
errors=[]
for item in sys.argv[1:]:
    try:
        py_compile.compile(item, doraise=True)
    except Exception as exc:
        errors.append(f"{item}: {exc}")
if errors:
    print("\n".join(errors))
    raise SystemExit(1)
print(f"Compiled {len(sys.argv)-1} Python files successfully.")
'@ | Set-Content -LiteralPath $compileScript -Encoding UTF8
            $args = @($compileScript) + @($pyFiles | ForEach-Object FullName)
            Run-CapturedCommand -Name 'python_compile_relevant' -File $python.Source -Arguments $args -WorkingDirectory $ProjectRoot
        } else {
            Add-Finding -Area 'Python compile' -Status 'NOT FOUND' -Detail 'No relevant Python files found for compilation.'
        }

        $testDirs = @('backend\Imaging\tests','backend\Clinical\tests','tools\tests','tests')
        foreach ($td in $testDirs) {
            $full = Join-Path $ProjectRoot $td
            if (Test-Path -LiteralPath $full -PathType Container) {
                Run-CapturedCommand -Name ("unittest_" + ($td -replace '[\\:]','_')) -File $python.Source -Arguments @('-m','unittest','discover','-s',$full,'-v') -WorkingDirectory $ProjectRoot
            }
        }
        if ($RunAllPytest) {
            Run-CapturedCommand -Name 'pytest_all' -File $python.Source -Arguments @('-m','pytest','-q') -WorkingDirectory $ProjectRoot
        } else {
            $pytestFiles = @($allFiles | Where-Object {
                $_.Name -match '^test_.*\.py$|.*_test\.py$' -and $_.FullName -match '(?i)(frontend|Clinical|Imaging|report|histolog|shap)'
            })
            if ($pytestFiles.Count -gt 0) {
                Run-CapturedCommand -Name 'pytest_report_related' -File $python.Source -Arguments (@('-m','pytest','-q') + @($pytestFiles | ForEach-Object FullName)) -WorkingDirectory $ProjectRoot
            } else {
                Add-Finding -Area 'Pytest report-related tests' -Status 'NOT FOUND' -Detail 'No report-related pytest files identified. Run with -RunAllPytest if appropriate.'
            }
        }
    }

    if ($node) {
        foreach ($js in @('frontend\app.js')) {
            $jsPath = Join-Path $ProjectRoot $js
            if (Test-Path -LiteralPath $jsPath -PathType Leaf) {
                Run-CapturedCommand -Name ('node_check_' + ($js -replace '[\\.]','_')) -File $node.Source -Arguments @('--check',$jsPath) -WorkingDirectory $ProjectRoot
            }
        }
        $pkg = Join-Path $ProjectRoot 'frontend\package.json'
        if (-not (Test-Path $pkg)) { $pkg = Join-Path $ProjectRoot 'package.json' }
        if ((Test-Path $pkg) -and $npm) {
            $pkgObj = Get-Content -LiteralPath $pkg -Raw | ConvertFrom-Json
            if ($pkgObj.scripts -and $pkgObj.scripts.test) {
                Run-CapturedCommand -Name 'npm_test' -File $npm.Source -Arguments @('test','--','--runInBand') -WorkingDirectory (Split-Path -Parent $pkg)
            } else {
                Add-Finding -Area 'npm tests' -Status 'NOT FOUND' -Detail "No scripts.test entry found in $pkg."
            }
        }
    }
}

# Artifact presence checks (these are discovery aids, not model correctness checks).
$artifactPatterns = @(
    @{name='Histology model artifacts'; regex='(?i)(histolog|luad|lusc).*\.(pkl|joblib|pt|pth|onnx|json|yaml|yml)$'},
    @{name='SHAP artifacts'; regex='(?i)(shap|explain).*\.(json|png|svg|html|npy|npz|pkl)$'},
    @{name='Report test artifacts'; regex='(?i)(report|print|histolog|shap).*(test|spec).*\.(py|js|ts)$'}
)
foreach ($ap in $artifactPatterns) {
    $matches = @($allFiles | Where-Object { $_.FullName -match $ap.regex })
    Add-Finding -Area $ap.name -Status $(if($matches.Count){'REVIEW'}else{'NOT FOUND'}) -Detail "$($matches.Count) matching path(s) found." -Evidence (($matches | Select-Object -First 30 | ForEach-Object { $_.FullName.Substring($ProjectRoot.Length).TrimStart('\') }) -join "`n")
}

Write-TextReport
Write-Host ''
Write-Host 'Workflow completed.' -ForegroundColor Green
Write-Host "Markdown report: $ReportPath"
Write-Host "JSON report:     $JsonPath"
Write-Host "Logs/artifacts:  $ArtifactRoot"
Write-Host 'No application source files were changed.'
