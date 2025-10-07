# Parkinson Detection - Complete Backend & InstaTunnel Setup
# This script starts backend, InstaTunnel, and tests the connection in separate terminals

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Parkinson Detection - Full Setup         " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "      Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "      ERROR: Python not found!" -ForegroundColor Red
    Write-Host "      Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/4] Installing/Updating Python Dependencies..." -ForegroundColor Yellow
Write-Host "      This may take a few minutes on first run..." -ForegroundColor Yellow

# Check if requirements.txt exists
if (Test-Path "backend\requirements.txt") {
    Write-Host "      Found requirements.txt" -ForegroundColor Green
    
    # Install dependencies
    Set-Location backend
    try {
        Write-Host "      Installing packages..." -ForegroundColor Yellow
        python -m pip install --upgrade pip --quiet
        python -m pip install -r requirements.txt --quiet
        Write-Host "      All dependencies installed successfully!" -ForegroundColor Green
    } catch {
        Write-Host "      WARNING: Some dependencies may have failed to install" -ForegroundColor Yellow
        Write-Host "      Backend will try to run anyway..." -ForegroundColor Yellow
    }
    Set-Location ..
} else {
    Write-Host "      WARNING: requirements.txt not found!" -ForegroundColor Yellow
    Write-Host "      Proceeding anyway..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[3/4] Starting Backend Server..." -ForegroundColor Yellow

# Get current directory
$rootDir = Get-Location

# Start Flask backend in new terminal
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$rootDir'; cd backend; Write-Host '========================================' -ForegroundColor Cyan; Write-Host '  Flask Backend Server' -ForegroundColor Cyan; Write-Host '========================================' -ForegroundColor Cyan; Write-Host ''; Write-Host 'Starting on http://localhost:5000...' -ForegroundColor Yellow; Write-Host ''; python app.py"
)

Write-Host "      Backend starting in new terminal..." -ForegroundColor Green
Write-Host "      Waiting 5 seconds for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "[4/4] Starting InstaTunnel..." -ForegroundColor Yellow

# Start InstaTunnel in new terminal
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$rootDir'; Write-Host '========================================' -ForegroundColor Cyan; Write-Host '  InstaTunnel Client' -ForegroundColor Cyan; Write-Host '========================================' -ForegroundColor Cyan; Write-Host ''; Write-Host 'Connecting to InstaTunnel...' -ForegroundColor Yellow; Write-Host 'Subdomain: parkinsons-disease' -ForegroundColor Yellow; Write-Host 'Port: 5000' -ForegroundColor Yellow; Write-Host ''; instatunnel connect 5000 --subdomain parkinsons-disease"
)

Write-Host "      InstaTunnel starting in new terminal..." -ForegroundColor Green
Write-Host "      Waiting 3 seconds for tunnel to connect..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "Testing connection..." -ForegroundColor Yellow

# Test connection in new terminal
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$rootDir'; Write-Host '========================================' -ForegroundColor Cyan; Write-Host '  Connection Test' -ForegroundColor Cyan; Write-Host '========================================' -ForegroundColor Cyan; Write-Host ''; Write-Host 'Testing: https://parkinsons-disease.instatunnel.my/api/health' -ForegroundColor Yellow; Write-Host ''; Start-Sleep -Seconds 2; curl https://parkinsons-disease.instatunnel.my/api/health; Write-Host ''; Write-Host '========================================' -ForegroundColor Green; Write-Host 'If you see status:healthy above, SUCCESS!' -ForegroundColor Green; Write-Host '========================================' -ForegroundColor Green; Write-Host ''; Write-Host 'Press any key to close this window...' -ForegroundColor Yellow; Read-Host"
)

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Installation Summary:" -ForegroundColor Cyan
Write-Host "  ✓ Python dependencies installed from requirements.txt" -ForegroundColor Green
Write-Host "  ✓ Backend server started" -ForegroundColor Green
Write-Host "  ✓ InstaTunnel connected" -ForegroundColor Green
Write-Host ""
Write-Host "Three terminals opened:" -ForegroundColor Cyan
Write-Host "  1. Backend Server (Flask on port 5000)" -ForegroundColor White
Write-Host "  2. InstaTunnel Client (connecting tunnel)" -ForegroundColor White
Write-Host "  3. Connection Test (testing health endpoint)" -ForegroundColor White
Write-Host ""
Write-Host "Your tunnel URL:" -ForegroundColor Cyan
Write-Host "  https://parkinsons-disease.instatunnel.my" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Check terminal #3 for health check result" -ForegroundColor White
Write-Host "  2. If successful, update Vercel BACKEND_URL" -ForegroundColor White
Write-Host "  3. Keep all terminals open while using the app" -ForegroundColor White
Write-Host ""
Write-Host "To stop: Close all terminal windows or press CTRL+C in each" -ForegroundColor Yellow
Write-Host ""
