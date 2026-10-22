# Starts the SFA-CA API on this laptop, shares it through a free Cloudflare quick tunnel, and publishes the
# tunnel address to web/api_url.txt on your fork so the GitHub Pages site finds it. Leave this window open.
#
# Quick tunnels are deleted by Cloudflare after a network drop or laptop sleep ("Tunnel not found"). This script
# checks the PUBLIC address every minute and, if it stops answering, opens a new tunnel and republishes it.
#
#   powershell -ExecutionPolicy Bypass -File 04_pipeline_backend_frontend\ops\start_public.ps1
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path   # repository root
$cf   = "C:\Users\sange\cloudflared_bin\cloudflared.exe"
$log  = Join-Path $env:TEMP "sfaca_cf.log"
$urlFile = "04_pipeline_backend_frontend\web\api_url.txt"
Set-Location $root

function Test-Api {
    try { (Invoke-RestMethod "http://localhost:7860/health" -TimeoutSec 3).status -eq "ok" } catch { $false }
}
function Test-Public($u) {
    try { (Invoke-RestMethod "$u/health" -TimeoutSec 20).status -eq "ok" } catch { $false }
}

function Start-Api {
    if (Test-Api) { return }
    Write-Host "Starting API (loading the model, ~30 s)..."
    $env:ADAPTER_DIR = Join-Path $root "results\adapters"
    Start-Process python -ArgumentList "04_pipeline_backend_frontend\backend\api.py" -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $env:TEMP "sfaca_api.out") -RedirectStandardError (Join-Path $env:TEMP "sfaca_api.err")
    for ($i = 0; $i -lt 60 -and -not (Test-Api); $i++) { Start-Sleep 3 }
    if (-not (Test-Api)) { throw "API did not start. See $env:TEMP\sfaca_api.err" }
}

# Cloudflare's tunnel-creation call is slow (10-16 s) and cloudflared gives up at 15 s, so retry.
function Open-Tunnel {
    Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
    # Few attempts only: Cloudflare rate-limits tunnel creation (HTTP 429) if we hammer it.
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Write-Host "Opening tunnel (attempt $attempt)..."
        try { Remove-Item $log -Force -ErrorAction Stop } catch {}
        $p = Start-Process $cf -ArgumentList "tunnel","--no-autoupdate","--edge-ip-version","4","--protocol","http2","--url","http://localhost:7860" `
            -WindowStyle Hidden -PassThru -RedirectStandardError $log
        for ($i = 0; $i -lt 20 -and -not $p.HasExited; $i++) {
            Start-Sleep 3
            if (Test-Path $log) {
                $txt = Get-Content $log -Raw
                if ($txt -match "status 429") { if (-not $p.HasExited) { $p | Stop-Process -Force }; return "RATE_LIMITED" }
                if ($txt -match "Registered tunnel connection" -and $txt -match "https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com") {
                    return $Matches[0]
                }
            }
        }
        if (-not $p.HasExited) { $p | Stop-Process -Force }
        Start-Sleep 10
    }
    return $null
}

function Publish-Url($url) {
    Set-Content -Path (Join-Path $root $urlFile) -Value $url -NoNewline -Encoding ascii
    $ErrorActionPreference = "Continue"   # git writes progress to stderr, which is not an error
    git add ($urlFile -replace "\\", "/")
    git commit -m "chore: update tunnel URL" 2>&1 | Out-Null
    git push fork HEAD:main 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: could not push the new URL to your fork (are you logged in to GitHub?)" }
    $ErrorActionPreference = "Stop"
}

Start-Api
Write-Host "API is healthy."
$url = $null
$failures = 0
while ($true) {
  try {
    if (-not $url) {
        if (-not (Test-Api)) { Start-Api }
        $url = Open-Tunnel
        if ($url -eq "RATE_LIMITED") {
            Write-Host "Cloudflare is rate-limiting new tunnels (429). Waiting 5 minutes before trying again..."
            $url = $null; Start-Sleep 300; continue
        }
        if (-not $url) { Write-Host "Could not open a tunnel; retrying in 2 minutes..."; Start-Sleep 120; continue }
        Publish-Url $url
        $failures = 0
        Write-Host ""
        Write-Host "LIVE: https://sangeeth0301.github.io/multilingual-aa-sfaca/   (the page picks up the new address within 1-2 minutes)"
        Write-Host "API tunnel: $url"
        Write-Host "Keep this laptop on and awake and this window open. Close it to stop sharing."
    }
    Start-Sleep 60
    if (-not (Test-Api)) { Write-Host "API stopped; restarting it."; Start-Api }
    if (Test-Public $url) { $failures = 0 } else { $failures++ }
    if ($failures -ge 2) {
        Write-Host "Public address stopped answering; opening a new tunnel."
        $url = $null
    }
  } catch {
    Write-Host ("Launcher error (continuing): " + $_.Exception.Message)
    $url = $null
    Start-Sleep 30
  }
}
