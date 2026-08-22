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
$gameExecutable = Join-Path $gamePath 'TmForever.exe'
$expectedPatchedGameSha256 = '4B6A7B31D86766409E94101F1256CD61DFFECB23EA497A40B6617506B2CED2D4'
$tmLoaderPath = Join-Path $env:LOCALAPPDATA 'TMLoader\TMLoader.exe'
$tmLoaderProfile = Join-Path $env:LOCALAPPDATA 'TMLoader\database\TmForever\profiles\default.yaml'
$tmInterfaceDll = Join-Path $env:LOCALAPPDATA 'TMLoader\database\TmForever\products\TMInterface\2.2.1\TMInterface.dll'
$expectedTmInterfaceSha256 = 'C986CA9BC1F8FD208FCD59DA7A1BECE8386BA0ACD7E3FF20D3E2F4F9404D027B'
$pluginPath = Join-Path $env:USERPROFILE 'Documents\TMInterface\Plugins\python_link.as'
$expectedPluginSha256 = '17FEFF21FEC2E9578AAB59C0C5D2C7EAFBC3313462FCAB2BFC18B29A08408083'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'

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
$tmInterfacePinned = $profile -match '(?ms)^\s*-\s*id:\s*TMInterface\s*\r?\n\s*version:\s*2\.2\.1\s*$'
$profileRuntimeSettings = $profile -match 'set unfocused_fps_limit false' -and $profile -match 'set auto_reload_plugins true'
$gameExecutableHash = if (Test-Path -LiteralPath $gameExecutable) {
    (Get-FileHash -LiteralPath $gameExecutable -Algorithm SHA256).Hash
} else {
    ''
}
$tmInterfaceHash = if (Test-Path -LiteralPath $tmInterfaceDll) {
    (Get-FileHash -LiteralPath $tmInterfaceDll -Algorithm SHA256).Hash
} else {
    ''
}
$pluginHash = if (Test-Path -LiteralPath $pluginPath) {
    (Get-FileHash -LiteralPath $pluginPath -Algorithm SHA256).Hash
} else {
    ''
}
$bridgeListener = [Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
    Where-Object { $_.Address.ToString() -eq '127.0.0.1' -and $_.Port -eq 8478 } |
    Select-Object -First 1

$pythonCommand = if (Test-Path -LiteralPath $projectPython) {
    Get-Item -LiteralPath $projectPython
} else {
    Get-Command py -ErrorAction SilentlyContinue
}
if (-not $pythonCommand) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
}

$pythonIsStoreAlias = $pythonCommand -and $pythonCommand.Source -like '*\Microsoft\WindowsApps\python*.exe'
$pythonReady = [bool]$pythonCommand -and -not $pythonIsStoreAlias
$pythonDetail = if ($pythonReady) {
    & $pythonCommand.FullName --version 2>&1 | Out-String
} elseif ($pythonIsStoreAlias) {
    'Only the Microsoft Store execution alias is present.'
} else {
    'No Python command was found.'
}

$checks = @(
    New-Check -Name 'TMNF Steam manifest' -Passed ($manifest -match '"appid"\s+"11020"') -Detail $manifestPath
    New-Check -Name 'TMNF game files' -Passed (Test-Path -LiteralPath $gameExecutable) -Detail $gamePath
    New-Check -Name 'TMNF 2.11.26 compatibility patch' -Passed ($gameExecutableHash -eq $expectedPatchedGameSha256) -Detail "TmForever.exe SHA-256: $gameExecutableHash"
    New-Check -Name 'TMNF running' -Passed ([bool](Get-Process TmForever -ErrorAction SilentlyContinue)) -Detail 'Launch through the approved TMLoader profile for live checks.'
    New-Check -Name 'TMLoader installed' -Passed (Test-Path -LiteralPath $tmLoaderPath) -Detail $tmLoaderPath
    New-Check -Name 'TMInterface 2.2.1 in profile' -Passed $tmInterfacePinned -Detail $tmLoaderProfile
    New-Check -Name 'TMLoader runtime settings' -Passed $profileRuntimeSettings -Detail 'Window-friendly FPS and plugin auto-reload settings are pinned.'
    New-Check -Name 'TMInterface 2.2.1 payload' -Passed ($tmInterfaceHash -eq $expectedTmInterfaceSha256) -Detail "TMInterface.dll SHA-256: $tmInterfaceHash"
    New-Check -Name 'python_link.as bridge' -Passed ($pluginHash -eq $expectedPluginSha256) -Detail "python_link.as SHA-256: $pluginHash"
    New-Check -Name 'python_link.as loopback listener' -Passed ([bool]$bridgeListener) -Detail 'Expected 127.0.0.1:8478 while TrackMania is running.'
    New-Check -Name 'Python runtime' -Passed $pythonReady -Detail $pythonDetail.Trim()
)

$checks | Format-Table -AutoSize -Wrap

if ($checks.Status -contains 'PENDING') {
    Write-Host "`nPhase 0 is not ready for live telemetry/control probes."
    exit 1
}

Write-Host "`nStatic prerequisites are present. Continue with the manual live checks in phase0_notes.md."
