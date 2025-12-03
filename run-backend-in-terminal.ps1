# Run backend inside the current PowerShell terminal
# Usage: Open VS Code terminal and run: .\run-backend-in-terminal.ps1

Write-Host "Starting Parkinson Detection backend in current terminal..." -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendPath = Join-Path $scriptDir "backend"

if (-not (Test-Path $backendPath)) {
    Write-Host "ERROR: backend directory not found at $backendPath" -ForegroundColor Red
    exit 1
}

# Change to backend directory
Set-Location $backendPath

# Optional: Activate venv if present (commented by default)
# If you use a virtualenv in backend\venv, uncomment the lines below
# $venvActivate = Join-Path $backendPath "venv\Scripts\Activate.ps1"
# if (Test-Path $venvActivate) {
#     Write-Host "Activating virtual environment..." -ForegroundColor Yellow
#     . $venvActivate
# }

Write-Host "Running: python app.py" -ForegroundColor Green
try {
    python app.py
} catch {
    Write-Host "Error launching backend: $_" -ForegroundColor Red
    exit 1
}

Write-Host "Backend process exited." -ForegroundColor Yellow
