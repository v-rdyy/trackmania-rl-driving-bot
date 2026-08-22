[CmdletBinding()]
param(
    [string] $Destination = (Join-Path $env:USERPROFILE 'Documents\TMInterface\Plugins\python_link.as'),
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$source = Join-Path $workspaceRoot 'vendor\tminterface\python_link.as'
$expectedSha256 = '17FEFF21FEC2E9578AAB59C0C5D2C7EAFBC3313462FCAB2BFC18B29A08408083'

if (-not (Test-Path -LiteralPath $source)) {
    throw "Vendored bridge is missing: $source"
}

$sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
if ($sourceHash -ne $expectedSha256) {
    throw "Vendored bridge hash mismatch. Expected $expectedSha256; got $sourceHash."
}

$destinationDirectory = Split-Path -Parent $Destination
if ($DryRun) {
    Write-Host "Would install verified bridge to: $Destination"
    return
}

New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
Copy-Item -LiteralPath $source -Destination $Destination -Force

if (Test-Path -LiteralPath $Destination) {
    $destinationHash = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash
    if ($destinationHash -ne $expectedSha256) {
        throw "Installed bridge hash mismatch. Expected $expectedSha256; got $destinationHash."
    }
    Write-Host "Installed bridge verified: $Destination"
}
