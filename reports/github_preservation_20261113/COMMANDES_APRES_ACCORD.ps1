# NE PAS EXECUTER SANS ACCORD EXPLICITE POUR LE COMMIT ET LE PUSH.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:\Users\nourb\Projects\GeoFire-X-2026'
$gitOptions = @('-c', 'safe.directory=C:/Users/nourb/Projects/GeoFire-X-2026')
$branch = & git @gitOptions branch --show-current
if ($LASTEXITCODE -ne 0 -or $branch -ne 'feature/scientific-scaleup-v1') { throw 'Branche incorrecte' }
$head = & git @gitOptions rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $head -ne '7428561f607914eb2f7cd286ee50435d142f8e09') { throw 'HEAD a change : refaire la revue' }
& git @gitOptions fetch origin
if ($LASTEXITCODE -ne 0) { throw 'Fetch echoue' }
$remote = & git @gitOptions ls-remote --heads origin refs/heads/feature/scientific-scaleup-v1
if ($LASTEXITCODE -ne 0) { throw 'Verification distante impossible' }
if ($remote) { throw 'La branche distante existe maintenant : comparer avant publication' }
foreach ($row in (Import-Csv reports/github_preservation_20261113/publication_manifest.csv)) {
    if ((Get-FileHash -LiteralPath $row.path -Algorithm SHA256).Hash.ToLower() -ne $row.sha256) { throw "Contenu modifie : $($row.path)" }
}
& git @gitOptions --literal-pathspecs add --pathspec-from-file=reports/github_preservation_20261113/publication_files.txt
if ($LASTEXITCODE -ne 0) { throw 'Preparation explicite echouee' }
$expected = @(Get-Content reports/github_preservation_20261113/publication_files.txt | Sort-Object)
$actual = @(& git @gitOptions -c core.quotepath=false diff --cached --name-only | Sort-Object)
if ($LASTEXITCODE -ne 0 -or (Compare-Object $expected $actual)) { throw 'Index different de la selection approuvee' }
& git @gitOptions commit -m 'feat: preserve scientific scale-up, verified results and Prediction Lab'
if ($LASTEXITCODE -ne 0) { throw 'Commit echoue : aucun push' }
& git @gitOptions push --set-upstream origin HEAD:refs/heads/feature/scientific-scaleup-v1
if ($LASTEXITCODE -ne 0) { throw 'Push echoue : ne pas forcer' }
