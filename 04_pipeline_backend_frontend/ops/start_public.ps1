# Starts the SFA-CA API on this laptop, opens a free Cloudflare tunnel, and publishes the tunnel
# URL to web/api_url.txt on your fork so the GitHub Pages site finds it. Leave this window open.
#
#   powershell -ExecutionPolicy Bypass -File 04_pipeline_backend_frontend\ops\start_public.ps1
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path   # repository root
$cf   = "C:\Users\sange\cloudflared_bin\cloudflared.exe"
$log  = Join-Path $env:TEMP "sfaca_cf.log"
Set-Location $root

function Test-Api { try { (Invoke-RestMethod "http://localhost:7860/health" -TimeoutSec 3).status -eq "ok" } catch { $false } }

if (-not (Test-Api)) {
    Write-Host "Starting API (loading the model, ~30 s)..."
    $env:ADAPTER_DIR = Join-Path $root "results\adapters"
    Start-Process python -ArgumentList "04_pipeline_backend_frontend\backend\api.py" -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $env:TEMP "sfaca_api.out") -RedirectStandardError (Join-Path $env:TEMP "sfaca_api.err")
    for ($i = 0; $i -lt 60 -and -not (Test-Api); $i++) { Start-Sleep 3 }
    if (-not (Test-Api)) { throw "API did not start. See $env:TEMP\sfaca_api.err" }
}
Write-Host "API is healthy."

# Cloudflare's quick-tunnel endpoint is slow, so retry until a tunnel registers.
$url = $null
for ($attempt = 1; $attempt -le 8 -and -not $url; $attempt++) {
    Write-Host "Opening tunnel (attempt $attempt)..."
    Remove-Item $log -ErrorAction SilentlyContinue
    $p = Start-Process $cf -ArgumentList "tunnel","--no-autoupdate","--edge-ip-version","4","--protocol","http2","--url","http://localhost:7860" `
        -WindowStyle Hidden -PassThru -RedirectStandardError $log
    for ($i = 0; $i -lt 20 -and -not $url -and -not $p.HasExited; $i++) {
        Start-Sleep 3
        if (Test-Path $log) {
            $txt = Get-Content $log -Raw
            if ($txt -match "Registered tunnel connection" -and $txt -match "https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com") { $url = $Matches[0] }
        }
    }
    if (-not $url -and -not $p.HasExited) { $p | Stop-Process -Force }
}
if (-not $url) { throw "Could not open a tunnel. Try again in a minute." }

Set-Content -Path (Join-Path $root "04_pipeline_backend_frontend\web\api_url.txt") -Value $url -NoNewline -Encoding ascii
$ErrorActionPreference = "Continue"   # git writes progress to stderr, which is not an error
git add 04_pipeline_backend_frontend/web/api_url.txt
git commit -m "chore: update tunnel URL" 2>&1 | Out-Null
git push fork HEAD:main 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: could not push the new URL to your fork (are you logged in to GitHub?)" }
$ErrorActionPreference = "Stop"
Write-Host ""
Write-Host "LIVE: https://sangeeth0301.github.io/multilingual-aa-sfaca/   (the page picks up the new URL within ~1-2 minutes)"
Write-Host "API tunnel: $url"
Write-Host "Keep this laptop on and this window open. Press Ctrl+C or close it to stop sharing."
while ($true) { Start-Sleep 60; if ($p.HasExited -or -not (Test-Api)) { Write-Host "Tunnel or API stopped. Re-run this script."; break } }
