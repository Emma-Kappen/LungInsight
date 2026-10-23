Set-Location "C:\Users\emkap\Documents\Projects\LungInsight"
$ErrorActionPreference = "Stop"
$file = ".\frontend\app.js"

if (-not (Test-Path -LiteralPath $file)) {
    throw "File not found: $file"
}

$fullPath = (Resolve-Path -LiteralPath $file).Path
$backup = "$fullPath.encoding-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

# Read and preserve the source as UTF-8.
$content = [System.IO.File]::ReadAllText(
    $fullPath,
    [System.Text.Encoding]::UTF8
)
$original = $content

# Repair confirmed UTF-8/Windows-1252 mojibake sequences.
$replacements = [ordered]@{
    'âˆ’' = '−'
    'Î£'  = 'Σ'
    'â€¦' = '…'
}

foreach ($bad in $replacements.Keys) {
    $count = ([regex]::Matches(
        $content,
        [regex]::Escape($bad)
    )).Count
    if ($count -gt 0) {
        $content = $content.Replace($bad, $replacements[$bad])
        Write-Host "Repaired '$bad' -> '$($replacements[$bad])' ($count occurrence(s))"
    }
}

if ($content -eq $original) {
    Write-Host "No matching corruption sequences found; source file unchanged." -ForegroundColor Yellow
    exit 0
}

# Back up the original before writing any changes.
[System.IO.File]::Copy($fullPath, $backup, $false)

# Write UTF-8 without a BOM.
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($fullPath, $content, $utf8NoBom)

Write-Host "`nBackup: $backup" -ForegroundColor Cyan
Write-Host "Updated: $fullPath" -ForegroundColor Green

# Verify the Brock formula and remaining common corruption markers.
Select-String -LiteralPath $fullPath -Encoding utf8 -Pattern 'Brock formula|LP =|âˆ’|Î£|â€¦|Ã|Â' |
    Select-Object LineNumber, Line

Write-Host "`nReview the formula and any remaining flagged lines above." -ForegroundColor Cyan
