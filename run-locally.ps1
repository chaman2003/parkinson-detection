# Parkinson Detection - Mobile App Startup Script
# Starts backend, frontend proxy, and ngrok for smartphone access

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "  PARKINSON DETECTION - MOBILE SETUP" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Stop existing processes
Write-Host "Cleaning up existing processes..." -ForegroundColor Yellow
$port5000 = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
$port8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue

if ($port5000) {
    Get-Process -Id $port5000.OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

if ($port8000) {
    Get-Process -Id $port8000.OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Get-Process ngrok -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

Write-Host "Done." -ForegroundColor Green
Write-Host ""

# Start Backend
Write-Host "Starting Backend Server (Port 5000)..." -ForegroundColor Cyan
$backendPath = Join-Path $scriptDir "backend"
if (Test-Path $backendPath) {
    $backendCmd = "Set-Location '$backendPath'; Write-Host 'Backend Starting...' -ForegroundColor Green; python app.py"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd -WindowStyle Normal
    Start-Sleep -Seconds 8
    Write-Host "  Backend started" -ForegroundColor Green
} else {
    Write-Host "  ERROR: Backend not found" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Start Frontend Proxy
Write-Host "Starting Frontend Proxy Server (Port 8000)..." -ForegroundColor Cyan
$frontendPath = Join-Path $scriptDir "frontend"
if (Test-Path $frontendPath) {
    $frontendCmd = "Set-Location '$frontendPath'; Write-Host 'Frontend Proxy Starting...' -ForegroundColor Green; python server.py 8000"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd -WindowStyle Normal
    Start-Sleep -Seconds 4
    Write-Host "  Frontend proxy started" -ForegroundColor Green
} else {
    Write-Host "  ERROR: Frontend not found" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Start ngrok
Write-Host "Starting ngrok Tunnel..." -ForegroundColor Cyan
$ngrokPath = Join-Path $scriptDir "ngrok.exe"
if (Test-Path $ngrokPath) {
    $ngrokCmd = "Set-Location '$scriptDir'; Write-Host 'ngrok Starting with custom domain...' -ForegroundColor Green; .\ngrok.exe http --domain=ostensible-unvibrant-clarisa.ngrok-free.dev 8000"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $ngrokCmd -WindowStyle Normal
    Write-Host "  Waiting for ngrok..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    Write-Host "  ngrok started" -ForegroundColor Green
} else {
    Write-Host "  WARNING: ngrok not found" -ForegroundColor Yellow
    Write-Host "  Download from: https://ngrok.com/download" -ForegroundColor Yellow
}
Write-Host ""

# Get ngrok URL
Write-Host "Getting ngrok URL..." -ForegroundColor Cyan
Start-Sleep -Seconds 3

try {
    $ngrokData = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 5 -ErrorAction Stop
    $publicUrl = $ngrokData.tunnels[0].public_url
    
    Write-Host ""
    Write-Host "===============================================================================" -ForegroundColor Green
    Write-Host "  ALL SERVICES RUNNING - MOBILE READY!" -ForegroundColor Green
    Write-Host "===============================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "SMARTPHONE ACCESS:" -ForegroundColor Cyan
    Write-Host "  Open this URL on your phone: https://ostensible-unvibrant-clarisa.ngrok-free.dev" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "SERVICES:" -ForegroundColor Cyan
    Write-Host "  Backend:  http://localhost:5000" -ForegroundColor White
    Write-Host "  Frontend: http://localhost:8000" -ForegroundColor White
    Write-Host "  Mobile:   https://ostensible-unvibrant-clarisa.ngrok-free.dev" -ForegroundColor Yellow
    Write-Host "  Dashboard: http://127.0.0.1:4040" -ForegroundColor White
    Write-Host ""
    Write-Host "FEATURES ENABLED:" -ForegroundColor Cyan
    Write-Host "  - Microphone (voice test)" -ForegroundColor Green
    Write-Host "  - Accelerometer (tremor test)" -ForegroundColor Green
    Write-Host "  - Gyroscope (motion analysis)" -ForegroundColor Green
    Write-Host "  - HTTPS secure connection" -ForegroundColor Green
    Write-Host ""
    Write-Host "NEXT STEPS:" -ForegroundColor Cyan
    Write-Host "  1. Copy the mobile URL above" -ForegroundColor White
    Write-Host "  2. Open it on your smartphone" -ForegroundColor White
    Write-Host "  3. Grant microphone and sensor permissions" -ForegroundColor White
    Write-Host "  4. Start testing!" -ForegroundColor White
    Write-Host ""
    
} catch {
    Write-Host ""
    Write-Host "===============================================================================" -ForegroundColor Yellow
    Write-Host "  SERVICES STARTED" -ForegroundColor Yellow
    Write-Host "===============================================================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "LOCAL ACCESS:" -ForegroundColor Cyan
    Write-Host "  Frontend: http://localhost:8000" -ForegroundColor White
    Write-Host "  Backend:  http://localhost:5000" -ForegroundColor White
    Write-Host ""
    Write-Host "MOBILE ACCESS:" -ForegroundColor Cyan
    Write-Host "  Check the ngrok window for the public URL" -ForegroundColor White
    Write-Host "  Or visit: http://127.0.0.1:4040" -ForegroundColor White
    Write-Host ""
}

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "  KEEP ALL WINDOWS OPEN" -ForegroundColor Yellow
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Backend window   - Flask server" -ForegroundColor White
Write-Host "  Frontend window  - Proxy server" -ForegroundColor White
Write-Host "  ngrok window     - HTTPS tunnel" -ForegroundColor White
Write-Host ""
Write-Host "Ready for mobile testing!" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to exit this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
