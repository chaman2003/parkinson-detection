# Parkinson Detection - Backend & ngrok Only
# Starts backend server and ngrok tunnel (no frontend proxy)

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "  PARKINSON DETECTION - BACKEND & NGROK" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Stop existing processes
Write-Host "Cleaning up existing processes..." -ForegroundColor Yellow
$port5000 = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue

if ($port5000) {
    Write-Host "  Found process on port 5000, stopping..." -ForegroundColor Yellow
    Get-Process -Id $port5000.OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

$ngrokProcess = Get-Process ngrok -ErrorAction SilentlyContinue
if ($ngrokProcess) {
    Write-Host "  Found ngrok process, stopping..." -ForegroundColor Yellow
    $ngrokProcess | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Write-Host "  [OK] Cleanup complete" -ForegroundColor Green
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

# Start ngrok
Write-Host "Starting ngrok Tunnel..." -ForegroundColor Cyan
$ngrokPath = Join-Path $scriptDir "ngrok.exe"
if (Test-Path $ngrokPath) {
    Start-Process $ngrokPath -ArgumentList "http --domain=ostensible-unvibrant-clarisa.ngrok-free.dev 5000" -WindowStyle Normal
    Start-Sleep -Seconds 8
    Write-Host "  ngrok tunnel started" -ForegroundColor Green
} else {
    Write-Host "  WARNING: ngrok.exe not found" -ForegroundColor Yellow
    Write-Host "  Download from: https://ngrok.com/download" -ForegroundColor Gray
}

Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Green
Write-Host "  SERVICES RUNNING" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Green
Write-Host ""

Write-Host "SERVICES:" -ForegroundColor Cyan
Write-Host "  Backend:  http://localhost:5000" -ForegroundColor White
Write-Host "  API URL:  https://elease-unmeaning-mireille.ngrok-free.dev" -ForegroundColor Yellow
Write-Host "  Dashboard: http://127.0.0.1:4040" -ForegroundColor White
Write-Host ""

Write-Host "API ENDPOINTS:" -ForegroundColor Cyan
Write-Host "  Health:   GET  http://localhost:5000/api/health" -ForegroundColor White
Write-Host "  Analyze:  POST http://localhost:5000/api/analyze" -ForegroundColor White
Write-Host ""

Write-Host "MOBILE ACCESS:" -ForegroundColor Cyan
Write-Host "  URL: https://elease-unmeaning-mireille.ngrok-free.dev" -ForegroundColor Yellow
Write-Host "  Share this URL with your mobile device" -ForegroundColor Gray
Write-Host ""

Write-Host "TEST ENDPOINTS:" -ForegroundColor Cyan
Write-Host "  curl http://localhost:5000/api/health" -ForegroundColor Gray
Write-Host "  curl -X POST -F audio=@file.wav http://localhost:5000/api/analyze" -ForegroundColor Gray
Write-Host ""

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "  IMPORTANT: Keep both terminal windows open!" -ForegroundColor Yellow
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "  Backend window (Flask server)" -ForegroundColor White
Write-Host "  ngrok window   (HTTPS tunnel)" -ForegroundColor White
Write-Host ""

Write-Host "  Close either window to stop the service" -ForegroundColor Gray
Write-Host ""
Write-Host "  Ready for mobile testing!" -ForegroundColor Green
Write-Host ""

Write-Host 'Press any key to exit this window...' -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
