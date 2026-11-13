<#!
Default Review is offline. Previous attempt stopped; a new publication needs fresh explicit authorization.
Publish and Certify require explicit authorization and -Execute, one phase at a time.
Only lot 001 is supported. No code dataset, GPU, training, Git operation or retry.
#>
[CmdletBinding()]
param(
    [ValidateSet('Review', 'Publish', 'Certify')][string]$Phase = 'Review',
    [switch]$Execute,
    [ValidateRange(3,99)][int]$PublicationAttempt = 3
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$planPath = Join-Path $PSScriptRoot 'pilot/pilot-plan.json'
$plan = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($plan.handle -ne 'nnnnbhss/geofirex-wsts-fold0-tv-001-v1' -or $plan.zip_bytes_measured -ne 4829918684 -or
    $plan.zip_sha256_from_completed_staging -ne '464e89146fa6fdabfccdd1eb81d97b08549ca9b14428a32e3993f5996e8f5890') {
    throw 'Pinned single-lot plan changed; review required.'
}
if ($Phase -eq 'Review') {
    $plan | ConvertTo-Json -Depth 10
    return
}
if (-not $Execute) { throw 'No transfer: explicit user authorization and -Execute are required.' }
$ml = Join-Path $projectRoot '.venv/Scripts/python.exe'
$kg = Join-Path $projectRoot 'artifacts/orchestrator/kaggle-cli-env/Scripts/python.exe'
$cred = 'C:/Users/nourb/.kaggle/credentials.json'
$campaign = 'reports/kaggle_pilot_001_20261107_v1'
$journal = "$campaign/http-body.jsonl"
Push-Location -LiteralPath $projectRoot
try {
    if ($Phase -eq 'Publish') {
        $attemptTag = $PublicationAttempt.ToString('00')
        & $ml scripts/run_kaggle_phase_guarded.py --out "$campaign/lot-001-upload-monitor-$attemptTag" --seconds 7200 -- $kg scripts/publish_kaggle_single_attempt.py --staged artifacts/kaggle_publication_20261107_v1/lot-001 --credentials $cred --journal $journal --attempt "$campaign/lot-001-create-attempt-$attemptTag" --max-http-body-bytes 6000000000 --stop-at-cumulative-body-bytes 6000000000 --license CC-BY-4.0 --execute
    } elseif ($Phase -eq 'Certify') {
        & $ml scripts/run_kaggle_phase_guarded.py --out "$campaign/lot-001-cpu-submit-monitor-01" --seconds 120 -- $kg scripts/submit_kaggle_cpu_certification.py --folder reports/kaggle_pilot_001_20261107_v1/prepared-cpu/cpu-certification/lot-001 --credentials $cred --journal $journal --attempt "$campaign/lot-001-cpu-submit-01" --max-http-body-bytes 6000000000 --stop-at-cumulative-body-bytes 6000000000 --execute
    }
    if ($LASTEXITCODE -ne 0) { throw 'Selected phase stopped. Inspect logs and remote state; never retry automatically.' }
} finally {
    Pop-Location
}
