param([int]$Port = 8026)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Push-Location -LiteralPath $projectRoot
$previousPythonPath = $env:PYTHONPATH
try {
    if (!(Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Environnement .venv absent.' }
    if (!(Test-Path -LiteralPath 'frontend\node_modules\esbuild')) { throw 'Dependances frontend absentes. Consulter la procedure de demonstration.' }
    Push-Location -LiteralPath (Join-Path $projectRoot 'frontend')
    try { & npm.cmd run build; if ($LASTEXITCODE -ne 0) { throw 'Compilation frontend echouee.' } }
    finally { Pop-Location }
    $env:PYTHONPATH = Join-Path $projectRoot 'src'
    Write-Host "GeoFire-X Studio : http://127.0.0.1:$Port - Ctrl+C pour arreter."
    & .\.venv\Scripts\python.exe scripts/serve_jury.py --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Serveur arrete en erreur. Verifier si le port est deja occupe.' }
}
finally { $env:PYTHONPATH = $previousPythonPath; Pop-Location }
