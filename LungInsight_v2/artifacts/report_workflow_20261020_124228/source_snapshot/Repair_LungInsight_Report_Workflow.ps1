# Repairs the generated workflow script's report writer, then validates its syntax.
# Run this from the LungInsight_v2 project root after the original script is present.
$ErrorActionPreference = 'Stop'
$target = Join-Path (Get-Location).Path 'LungInsight_Report_Workflow.ps1'
if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
    throw "Cannot find $target. Run this from the LungInsight_v2 project root."
}
$source = Get-Content -LiteralPath $target -Raw
$startMarker = 'function Write-TextReport {'
$endMarker = "`nWrite-Host `"Project: $ProjectRoot`""
$start = $source.IndexOf($startMarker)
if ($start -lt 0) { throw 'Could not find Write-TextReport function in the script.' }

# Find the function end by locating the next known top-level statement.
$nextMarker = "`nWrite-Host `"Project:"
$end = $source.IndexOf($nextMarker, $start)
if ($end -lt 0) {
    # Fallback marker for versions with different whitespace.
    $nextMarker = "`r`nWrite-Host `"Project:"
    $end = $source.IndexOf($nextMarker, $start)
}
if ($end -lt 0) { throw 'Could not find the end marker after Write-TextReport; no changes written.' }

$replacement = @'
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
'@
$patched = $source.Substring(0, $start) + $replacement + $source.Substring($end)
# Save as UTF-8 with BOM for compatibility with Windows PowerShell 5.1.
$utf8Bom = New-Object System.Text.UTF8Encoding($true)
[System.IO.File]::WriteAllText($target, $patched, $utf8Bom)

$tokens = $null
$parseErrors = $null
[System.Management.Automation.Language.Parser]::ParseFile($target, [ref]$tokens, [ref]$parseErrors) | Out-Null
if ($parseErrors.Count -gt 0) {
    $parseErrors | ForEach-Object { Write-Error ("Line {0}, column {1}: {2}" -f $_.Extent.StartLineNumber, $_.Extent.StartColumnNumber, $_.Message) }
    throw "PowerShell parser found $($parseErrors.Count) error(s). The script was patched, but do not run the audit until these are resolved."
}
Write-Host 'Repair complete. PowerShell syntax check passed.' -ForegroundColor Green
Write-Host "Patched: $target"
Write-Host 'No LungInsight application files were modified.'
