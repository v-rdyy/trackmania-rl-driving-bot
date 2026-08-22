[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $SourceRoot,
    [Parameter(Mandatory)] [string] $GameRoot,
    [Parameter(Mandatory)] [string] $BackupRoot,
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'

$sourceFull = [IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
$gameFull = [IO.Path]::GetFullPath($GameRoot).TrimEnd('\')
$backupFull = [IO.Path]::GetFullPath($BackupRoot).TrimEnd('\')
$expectedGameRoot = 'C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever'
$expectedPatchedExeSha256 = '4B6A7B31D86766409E94101F1256CD61DFFECB23EA497A40B6617506B2CED2D4'
$documentsRoot = [IO.Path]::GetFullPath((Join-Path $env:USERPROFILE 'Documents')).TrimEnd('\')

$sourceExe = Join-Path $sourceFull 'TmForever.exe'
if (-not (Test-Path -LiteralPath $sourceExe)) {
    throw "Extracted patch payload is invalid: $sourceFull"
}
$sourceExeHash = (Get-FileHash -LiteralPath $sourceExe -Algorithm SHA256).Hash
if ($sourceExeHash -ne $expectedPatchedExeSha256) {
    throw "Unexpected 2.11.26 patch payload. TmForever.exe SHA-256 was $sourceExeHash."
}
if ($gameFull -ne $expectedGameRoot) {
    throw "Refusing unexpected game target: $gameFull"
}
if (-not (Test-Path -LiteralPath (Join-Path $gameFull 'Nadeo.ini'))) {
    throw "Target does not look like a TrackMania Forever installation: $gameFull"
}
if (-not $backupFull.StartsWith($documentsRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "Backup must be a specific directory beneath Documents: $backupFull"
}

$sourceFiles = @(Get-ChildItem -LiteralPath $sourceFull -Recurse -File)
$manifest = foreach ($file in $sourceFiles) {
    if (-not $file.FullName.StartsWith($sourceFull + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Source file escapes the extracted patch directory: $($file.FullName)"
    }
    $relativePath = $file.FullName.Substring($sourceFull.Length + 1)
    $targetPath = [IO.Path]::GetFullPath((Join-Path $gameFull $relativePath))
    if (-not $targetPath.StartsWith($gameFull + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Patch path escapes the game directory: $relativePath"
    }

    $previousExists = Test-Path -LiteralPath $targetPath
    [pscustomobject]@{
        RelativePath = $relativePath
        PreviousExists = $previousExists
        PreviousSha256 = if ($previousExists) { (Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash } else { '' }
        PatchSha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    }
}

$existingCount = @($manifest | Where-Object PreviousExists).Count
$newCount = $manifest.Count - $existingCount
Write-Host "Patch audit: $($manifest.Count) files ($existingCount existing, $newCount new)."

if ($DryRun) {
    Write-Host "Would back up overwritten files to: $backupFull"
    return
}
if (Test-Path -LiteralPath $backupFull) {
    throw "Backup target already exists; refusing to mix runs: $backupFull"
}

New-Item -ItemType Directory -Path $backupFull | Out-Null
$manifestPath = Join-Path $backupFull 'patch-manifest.csv'
$manifest | Export-Csv -LiteralPath $manifestPath -NoTypeInformation

foreach ($entry in $manifest) {
    $sourcePath = Join-Path $sourceFull $entry.RelativePath
    $targetPath = Join-Path $gameFull $entry.RelativePath

    if ($entry.PreviousExists) {
        $backupPath = Join-Path $backupFull $entry.RelativePath
        New-Item -ItemType Directory -Path (Split-Path -Parent $backupPath) -Force | Out-Null
        Copy-Item -LiteralPath $targetPath -Destination $backupPath
    }

    New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force | Out-Null
    Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
}

foreach ($entry in $manifest) {
    $targetPath = Join-Path $gameFull $entry.RelativePath
    $installedHash = (Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash
    if ($installedHash -ne $entry.PatchSha256) {
        throw "Installed patch hash mismatch: $($entry.RelativePath)"
    }
}

$installedExeHash = (Get-FileHash -LiteralPath (Join-Path $gameFull 'TmForever.exe') -Algorithm SHA256).Hash
if ($installedExeHash -ne $expectedPatchedExeSha256) {
    throw "Installed TmForever.exe does not match the verified 2.11.26 payload."
}

Write-Host "Patch files installed and verified. Backup: $backupFull"
