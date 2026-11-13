$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentRoot = Join-Path $projectRoot 'artifacts\orchestrator\kaggle-cli-env'
$toolPython = Join-Path $environmentRoot 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $toolPython)) {
    & (Join-Path $projectRoot '.venv\Scripts\python.exe') -m venv $environmentRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create isolated Kaggle tooling environment' }
}
& $toolPython -m pip install --disable-pip-version-check kaggle==2.2.5 kagglehub==1.0.3
if ($LASTEXITCODE -ne 0) { throw 'Could not install official Kaggle tooling' }
Write-Output 'Kaggle tooling ready. Credentials stay in the Windows user profile.'
