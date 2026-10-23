[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$TargetWorkspaceRoot,
    [Parameter()]
    [string]$BackupRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$PackageRoot = Split-Path -Parent $PSScriptRoot
$TargetWorkspaceRoot = [System.IO.Path]::GetFullPath($TargetWorkspaceRoot)
if (-not (Test-Path -LiteralPath $TargetWorkspaceRoot -PathType Container)) {
    throw "Target workspace does not exist: $TargetWorkspaceRoot"
}
$ChecksumPath = Join-Path $PSScriptRoot 'SHA256SUMS.txt'
if (-not (Test-Path -LiteralPath $ChecksumPath -PathType Leaf)) {
    throw "SHA-256 manifest is missing: $ChecksumPath"
}

$ExpectedHashes = @{}
foreach ($Line in Get-Content -LiteralPath $ChecksumPath -Encoding UTF8) {
    if ([string]::IsNullOrWhiteSpace($Line)) { continue }
    if ($Line -notmatch '^([0-9a-fA-F]{64})\s+\*(.+)$') {
        throw "Invalid SHA-256 manifest line: $Line"
    }
    $RelativePath = $Matches[2]
    if ([System.IO.Path]::IsPathRooted($RelativePath) -or $RelativePath -match '(^|[\\/])\.\.([\\/]|$)') {
        throw "Unsafe relative path in SHA-256 manifest: $RelativePath"
    }
    $SourcePath = Join-Path $PackageRoot $RelativePath
    if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
        throw "Checksum-listed package file is missing: $SourcePath"
    }
    $ActualHash = (Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash
    if ($ActualHash -ine $Matches[1]) {
        throw "Package checksum mismatch for $RelativePath. No destination files were changed."
    }
    $ExpectedHashes[$RelativePath] = $Matches[1].ToLowerInvariant()
}
if ($ExpectedHashes.Count -eq 0) {
    throw "SHA-256 manifest contains no file entries: $ChecksumPath"
}

if ([string]::IsNullOrWhiteSpace($BackupRoot)) {
    $BackupRoot = Join-Path $TargetWorkspaceRoot '.lunginsight_backups'
}
$BackupRoot = [System.IO.Path]::GetFullPath($BackupRoot)
$BackupDirectory = Join-Path $BackupRoot ('LungInsight_v2_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
New-Item -ItemType Directory -Path $BackupDirectory -Force | Out-Null

$NewFiles = [System.Collections.Generic.List[string]]::new()
foreach ($RelativePath in $ExpectedHashes.Keys) {
    $SourcePath = Join-Path $PackageRoot $RelativePath
    $Destination = Join-Path $TargetWorkspaceRoot $RelativePath
    $DestinationDirectory = Split-Path -Parent $Destination
    if (-not (Test-Path -LiteralPath $DestinationDirectory -PathType Container)) {
        New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
    }
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $BackupPath = Join-Path $BackupDirectory $RelativePath
        $BackupParent = Split-Path -Parent $BackupPath
        if (-not (Test-Path -LiteralPath $BackupParent -PathType Container)) {
            New-Item -ItemType Directory -Path $BackupParent -Force | Out-Null
        }
        Copy-Item -LiteralPath $Destination -Destination $BackupPath
    } else {
        $NewFiles.Add($RelativePath)
    }
    Copy-Item -LiteralPath $SourcePath -Destination $Destination -Force
}

$ChecksumRelativePath = 'LungInsight_v2/SHA256SUMS.txt'
$ChecksumDestination = Join-Path $TargetWorkspaceRoot $ChecksumRelativePath
$ChecksumDestinationDirectory = Split-Path -Parent $ChecksumDestination
if (-not (Test-Path -LiteralPath $ChecksumDestinationDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $ChecksumDestinationDirectory -Force | Out-Null
}
if (Test-Path -LiteralPath $ChecksumDestination -PathType Leaf) {
    $ChecksumBackupPath = Join-Path $BackupDirectory $ChecksumRelativePath
    $ChecksumBackupParent = Split-Path -Parent $ChecksumBackupPath
    if (-not (Test-Path -LiteralPath $ChecksumBackupParent -PathType Container)) {
        New-Item -ItemType Directory -Path $ChecksumBackupParent -Force | Out-Null
    }
    Copy-Item -LiteralPath $ChecksumDestination -Destination $ChecksumBackupPath
} else {
    $NewFiles.Add($ChecksumRelativePath)
}
Copy-Item -LiteralPath $ChecksumPath -Destination $ChecksumDestination -Force

$NewFileManifest = Join-Path $BackupDirectory 'NEW_FILES_SHA256.txt'
$NewFiles | ForEach-Object {
    $InstalledPath = Join-Path $TargetWorkspaceRoot $_
    $Hash = (Get-FileHash -LiteralPath $InstalledPath -Algorithm SHA256).Hash.ToLowerInvariant()
    "$Hash *$_"
} | Set-Content -LiteralPath $NewFileManifest -Encoding UTF8

foreach ($RelativePath in $ExpectedHashes.Keys) {
    $Destination = Join-Path $TargetWorkspaceRoot $RelativePath
    if (-not (Test-Path -LiteralPath $Destination -PathType Leaf)) {
        throw "Installed file is missing: $Destination. Existing-file backups are at $BackupDirectory"
    }
    $InstalledHash = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash
    if ($InstalledHash -ine $ExpectedHashes[$RelativePath]) {
        throw "Installed checksum mismatch for $RelativePath. Existing-file backups are at $BackupDirectory"
    }
}

Write-Host "Installation and checksum verification completed." -ForegroundColor Green
Write-Host "Target workspace: $TargetWorkspaceRoot"
Write-Host "Pre-existing files backed up under: $BackupDirectory"
Write-Host "No patient output, dataset, checkpoint, or model-weight files are included in this bundle."
