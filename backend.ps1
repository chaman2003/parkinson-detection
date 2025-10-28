# Parkinson Detection - Backend & ngrok Only
# Starts backend server and ngrok tunnel (no frontend proxy)
# Uses virtual environment created by install.ps1

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "  PARKINSON DETECTION - BACKEND & NGROK" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $scriptDir "venv"
$venvActivate = Join-Path $venvDir "Scripts\Activate.ps1"

# Check if venv exists
if (-not (Test-Path $venvActivate)) {
    Write-Host "[ERROR] Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run install.ps1 first to create the virtual environment" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Command: .\install.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Activate virtual environment
Write-Host "[ENV] Activating virtual environment..." -ForegroundColor Cyan
. $venvActivate
Write-Host "[OK] Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Stop existing processes
Write-Host "Cleaning up existing processes..." -ForegroundColor Yellow
$port5000 = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue

if ($port5000) {
    Get-Process -Id $port5000.OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
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
    $backendCmd = ". '$venvActivate'; Set-Location '$backendPath'; Write-Host 'Backend Starting...' -ForegroundColor Green; python app.py"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd -WindowStyle Normal
    Start-Sleep -Seconds 8
    Write-Host "  Backend started" -ForegroundColor Green
} else {
    Write-Host "  ERROR: Backend not found" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Start ngrok
Write-Host "Starting ngrok Tunnel..." -ForegroundColor Cyan
$ngrokPath = Join-Path $scriptDir "ngrok.exe"
if (Test-Path $ngrokPath) {
    $ngrokCmd = "Set-Location '$scriptDir'; Write-Host 'ngrok Starting with custom domain...' -ForegroundColor Green; .\ngrok.exe http --domain=ostensible-unvibrant-clarisa.ngrok-free.dev 5000"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $ngrokCmd -WindowStyle Normal
    Write-Host "  Waiting for ngrok..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    Write-Host "  ngrok started" -ForegroundColor Green
} else {
    Write-Host "  WARNING: ngrok not found" -ForegroundColor Yellow
    Write-Host "  Download from: https://ngrok.com/download" -ForegroundColor Yellow
}
Write-Host ""

Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Green
Write-Host "  BACKEND & NGROK RUNNING" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "SERVICES:" -ForegroundColor Cyan
Write-Host "  Backend:  http://localhost:5000" -ForegroundColor White
Write-Host "  API URL:  https://ostensible-unvibrant-clarisa.ngrok-free.dev" -ForegroundColor Yellow
Write-Host "  Dashboard: http://127.0.0.1:4040" -ForegroundColor White
Write-Host ""
Write-Host "API ENDPOINTS:" -ForegroundColor Cyan
Write-Host "  Health:   https://ostensible-unvibrant-clarisa.ngrok-free.dev/api/health" -ForegroundColor White
Write-Host "  Analyze:  https://ostensible-unvibrant-clarisa.ngrok-free.dev/api/analyze" -ForegroundColor White
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "  1. Use this backend with your Vercel-deployed frontend" -ForegroundColor White
Write-Host "  2. Set BACKEND_URL in Vercel to: https://ostensible-unvibrant-clarisa.ngrok-free.dev" -ForegroundColor White
Write-Host "  3. Or test locally by running frontend separately (python server.py 8000)" -ForegroundColor White
Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "  KEEP BOTH WINDOWS OPEN" -ForegroundColor Yellow
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Backend window - Flask server" -ForegroundColor White
Write-Host "  ngrok window   - HTTPS tunnel" -ForegroundColor White
Write-Host ""
Write-Host "Ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to exit this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
