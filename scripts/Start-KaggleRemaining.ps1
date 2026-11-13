<#
Standalone Windows PowerShell entry point. Default: offline review.
Run in a separate PowerShell window after closing VS Code/Codex.
No activation, scheduled task, background service or IDE is needed.
#>
[CmdletBinding(DefaultParameterSetName = 'Review')]
param(
    [Parameter(ParameterSetName = 'Execute')][switch]$Execute,
    [Parameter(ParameterSetName = 'Status')][switch]$Status
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonPath = Join-Path $projectRoot 'artifacts\orchestrator\kaggle-cli-env\Scripts\python.exe'
$entryPath = Join-Path $projectRoot 'scripts\run_autonomous_kaggle.py'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Existing Kaggle Python environment is missing: $pythonPath"
}

Push-Location -LiteralPath $projectRoot
try {
    $env:CUDA_VISIBLE_DEVICES = ''
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONUNBUFFERED = '1'
    $env:OMP_NUM_THREADS = '2'
    $env:MKL_NUM_THREADS = '2'
    $env:OPENBLAS_NUM_THREADS = '2'
    $env:NUMEXPR_NUM_THREADS = '2'
    $env:PYTHONPATH = (Join-Path $projectRoot 'src') + [System.IO.Path]::PathSeparator + (Join-Path $projectRoot 'scripts')
    $runArguments = @($entryPath)
    if ($Execute) { $runArguments += '--execute' }
    if ($Status) { $runArguments += '--status' }
    Write-Host 'GeoFire-X standalone: lots 003-008; start 3 GiB; reserve 2 GiB; tree cap 512 MiB.'
    Write-Host 'Logs: reports\kaggle_remaining_20261108_v1\resume-01\autonomous-01'
    & $pythonPath @runArguments
    $resultCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $resultCode
