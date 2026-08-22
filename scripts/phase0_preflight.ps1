[CmdletBinding()]
param()

$ErrorActionPreference = 'SilentlyContinue'

function New-Check {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [bool] $Passed,
        [Parameter(Mandatory)] [string] $Detail
    )

    [pscustomobject]@{
        Check  = $Name
        Status = if ($Passed) { 'PASS' } else { 'PENDING' }
        Detail = $Detail
    }
}

$steamRoot = 'C:\Program Files (x86)\Steam'
$manifestPath = Join-Path $steamRoot 'steamapps\appmanifest_11020.acf'
$gamePath = Join-Path $steamRoot 'steamapps\common\TrackMania Nations Forever'
$tmLoaderPath = Join-Path $env:LOCALAPPDATA 'TMLoader\TMLoader.exe'
$tmLoaderProfile = Join-Path $env:LOCALAPPDATA 'TMLoader\database\TmForever\profiles\default.yaml'
$pluginPath = Join-Path $env:USERPROFILE 'Documents\TMInterface\Plugins\python_link.as'

$manifest = if (Test-Path -LiteralPath $manifestPath) {
    Get-Content -Raw -LiteralPath $manifestPath
} else {
    ''
}

$profile = if (Test-Path -LiteralPath $tmLoaderProfile) {
    Get-Content -Raw -LiteralPath $tmLoaderProfile
} else {
    ''
}

$pythonCommand = Get-Command py -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
}

$pythonIsStoreAlias = $pythonCommand -and $pythonCommand.Source -like '*\Microsoft\WindowsApps\python*.exe'
$pythonReady = [bool]$pythonCommand -and -not $pythonIsStoreAlias
$pythonDetail = if ($pythonReady) {
    & $pythonCommand.Source --version 2>&1 | Out-String
} elseif ($pythonIsStoreAlias) {
    'Only the Microsoft Store execution alias is present.'
} else {
    'No Python command was found.'
}

$checks = @(
    New-Check -Name 'TMNF Steam manifest' -Passed ($manifest -match '"appid"\s+"11020"') -Detail $manifestPath
    New-Check -Name 'TMNF game files' -Passed (Test-Path -LiteralPath (Join-Path $gamePath 'TmForever.exe')) -Detail $gamePath
    New-Check -Name 'TMNF running' -Passed ([bool](Get-Process TmForever -ErrorAction SilentlyContinue)) -Detail 'Launch through the approved TMLoader profile for live checks.'
    New-Check -Name 'TMLoader installed' -Passed (Test-Path -LiteralPath $tmLoaderPath) -Detail $tmLoaderPath
    New-Check -Name 'TMInterface enabled in profile' -Passed ($profile -match '(?im)^\s*-?\s*id:\s*TMInterface\s*$') -Detail $tmLoaderProfile
    New-Check -Name 'python_link.as bridge' -Passed (Test-Path -LiteralPath $pluginPath) -Detail $pluginPath
    New-Check -Name 'Python runtime' -Passed $pythonReady -Detail $pythonDetail.Trim()
)

$checks | Format-Table -AutoSize -Wrap

if ($checks.Status -contains 'PENDING') {
    Write-Host "`nPhase 0 is not ready for live telemetry/control probes."
    exit 1
}

Write-Host "`nStatic prerequisites are present. Continue with the manual live checks in phase0_notes.md."
