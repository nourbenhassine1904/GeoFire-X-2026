param([ValidateRange(1024,65535)][int]$Port = 8026, [switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Push-Location -LiteralPath $projectRoot
$previousPythonPath = $env:PYTHONPATH
try {
    if (!(Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Environnement .venv absent.' }
    if (!(Test-Path -LiteralPath 'frontend\node_modules\esbuild')) { throw 'Dependances frontend absentes. Consulter la procedure de demonstration.' }
    if (!$SkipBuild) {
        Push-Location -LiteralPath (Join-Path $projectRoot 'frontend')
        try { & npm.cmd run build; if ($LASTEXITCODE -ne 0) { throw 'Compilation frontend echouee.' } }
        finally { Pop-Location }
    }
    if (!(Test-Path -LiteralPath 'frontend\dist\index.html')) { throw 'Frontend absent : relancer sans -SkipBuild.' }
    $env:PYTHONPATH = Join-Path $projectRoot 'src'
    Write-Host 'Studio V2 : inference CPU sur validation 2020 uniquement. Test 2021 termine, lecture seule.'
    if (!(Test-Path -LiteralPath 'artifacts\studio-v2-deployment\qualification.json')) {
        Write-Warning 'Qualification CPU absente : les modeles V2 seront desactives. Consulter docs/DEPLOYMENT_STUDIO_V2.md.'
    }
    $v2Root = if ($env:GEOFIREX_V2_ROOT) { $env:GEOFIREX_V2_ROOT } else { 'D:\GeoFire-X\GeoFire-X-V2' }
    if (!(Test-Path -LiteralPath $v2Root)) { Write-Warning 'ADATA absent : reconnecter le disque ou definir GEOFIREX_V2_ROOT. Les preuves locales restent consultables.' }
    Write-Host "GeoFire-X Studio : http://127.0.0.1:$Port - Ctrl+C pour arreter."
    & .\.venv\Scripts\python.exe scripts/serve_jury.py --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Serveur arrete en erreur. Verifier si le port est deja occupe.' }
}
finally { $env:PYTHONPATH = $previousPythonPath; Pop-Location }
