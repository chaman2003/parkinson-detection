Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "  PARKINSON DETECTION - COMPLETE INSTALLATION" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ErrorActionPreference = "Continue"
$venvDir = Join-Path $scriptDir "venv"

# Track installation status
$installationSteps = @{
    "Python Check" = $false
    "Virtual Environment" = $false
    "Python Dependencies" = $false
    "Node.js (Optional)" = $false
    "ngrok Download" = $false
    "Project Setup" = $false
    "Models Training" = $false
}

# Function to print step header
function Print-Step {
    param([string]$step, [int]$number, [int]$total)
    Write-Host ""
    Write-Host "[$number/$total] $step" -ForegroundColor Yellow
    Write-Host "=" * 70 -ForegroundColor Yellow
}

# Step 1: Check Python Installation
Print-Step "Checking Python Installation" 1 6

try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Found: $pythonVersion" -ForegroundColor Green
    $installationSteps["Python Check"] = $true
} catch {
    Write-Host "[ERROR] Python not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    exit 1
}

# Step 2: Create Virtual Environment
Print-Step "Creating Virtual Environment" 2 7

if (Test-Path $venvDir) {
    Write-Host "[OK] Virtual environment already exists at: $venvDir" -ForegroundColor Green
    $installationSteps["Virtual Environment"] = $true
} else {
    Write-Host "[ENV] Creating virtual environment..." -ForegroundColor Cyan
    python -m venv $venvDir
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Virtual environment created successfully" -ForegroundColor Green
        $installationSteps["Virtual Environment"] = $true
    } else {
        Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
$venvActivate = Join-Path $venvDir "Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "[ENV] Activating virtual environment..." -ForegroundColor Cyan
    . $venvActivate
    Write-Host "[OK] Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Could not find activation script" -ForegroundColor Red
    exit 1
}

# Step 3: Install Python Dependencies
Print-Step "Installing Python Dependencies" 3 7

Write-Host "[PKG] Upgrading pip in venv..." -ForegroundColor Cyan
python -m pip install --upgrade pip -q
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] pip upgraded successfully" -ForegroundColor Green
}

Write-Host "[PKG] Installing backend requirements..." -ForegroundColor Cyan
$backendReqs = Join-Path $scriptDir "backend\requirements.txt"
if (Test-Path $backendReqs) {
    python -m pip install -r $backendReqs -q
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Backend dependencies installed" -ForegroundColor Green
        $installationSteps["Python Dependencies"] = $true
    } else {
        Write-Host "[WARN] Some backend dependencies may have failed to install" -ForegroundColor Yellow
        $installationSteps["Python Dependencies"] = $true
    }
} else {
    Write-Host "[ERROR] requirements.txt not found" -ForegroundColor Red
    exit 1
}

# Step 4: Check Node.js (Optional)
Print-Step "Checking Node.js (Optional)" 4 7

try {
    $nodeVersion = node --version 2>&1
    Write-Host "[OK] Found: $nodeVersion" -ForegroundColor Green
    $installationSteps["Node.js (Optional)"] = $true
} catch {
    Write-Host "[WARN] Node.js not found (optional - only needed for frontend build)" -ForegroundColor Yellow
    Write-Host "   To install: https://nodejs.org/" -ForegroundColor Gray
    $installationSteps["Node.js (Optional)"] = $true
}

# Step 5: Download ngrok
Print-Step "Setting Up ngrok Tunnel" 5 7

$ngrokPath = Join-Path $scriptDir "ngrok.exe"

if (Test-Path $ngrokPath) {
    Write-Host "[OK] ngrok already installed at: $ngrokPath" -ForegroundColor Green
    $installationSteps["ngrok Download"] = $true
} else {
    Write-Host "[DOWNLOAD] Downloading ngrok..." -ForegroundColor Cyan
    Write-Host "   Get ngrok from: https://ngrok.com/download" -ForegroundColor Gray
    Write-Host ""
    Write-Host "   Steps:" -ForegroundColor Cyan
    Write-Host "   1. Go to https://ngrok.com/download" -ForegroundColor White
    Write-Host "   2. Download the Windows version" -ForegroundColor White
    Write-Host "   3. Extract ngrok.exe to: $scriptDir" -ForegroundColor White
    Write-Host ""
    Write-Host "   After downloading ngrok, run this script again." -ForegroundColor Yellow
    
    $response = Read-Host "Have you downloaded and placed ngrok.exe? (yes/no)"
    if ($response.ToLower() -eq "yes") {
        if (Test-Path $ngrokPath) {
            Write-Host "[OK] ngrok.exe found!" -ForegroundColor Green
            $installationSteps["ngrok Download"] = $true
        } else {
            Write-Host "[ERROR] ngrok.exe still not found at: $ngrokPath" -ForegroundColor Red
            Write-Host "Please place it manually and try again" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[SKIP] Skipping ngrok setup for now" -ForegroundColor Yellow
        Write-Host "   You can set it up later by running: .\install.ps1" -ForegroundColor Gray
    }
}

# Step 6: Project Structure Setup
Print-Step "Setting Up Project Structure" 6 7

Write-Host "[DIR] Checking project directories..." -ForegroundColor Cyan

$requiredDirs = @(
    "backend",
    "backend\datasets",
    "backend\datasets\voice_dataset",
    "backend\datasets\voice_dataset\healthy",
    "backend\datasets\voice_dataset\parkinson",
    "backend\models",
    "backend\recorded_data",
    "backend\recorded_data\tremor_data",
    "backend\recorded_data\voice_recordings",
    "backend\recorded_data\voice_recordings\healthy",
    "backend\recorded_data\voice_recordings\parkinsons",
    "backend\uploads",
    "frontend"
)

foreach ($dir in $requiredDirs) {
    $fullPath = Join-Path $scriptDir $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force -ErrorAction SilentlyContinue | Out-Null
        Write-Host "  [DIR] Created: $dir" -ForegroundColor Gray
    }
}

Write-Host "[OK] Project structure ready" -ForegroundColor Green
$installationSteps["Project Setup"] = $true

# Step 7: ML Models Setup
Print-Step "ML Models Training Setup" 7 7

$modelsDir = Join-Path $scriptDir "backend\models"
$voiceModel = Join-Path $modelsDir "voice_model.pkl"
$tremorModel = Join-Path $modelsDir "tremor_model.pkl"

if ((Test-Path $voiceModel) -and (Test-Path $tremorModel)) {
    Write-Host "[OK] ML models already trained and available" -ForegroundColor Green
    $installationSteps["Models Training"] = $true
} else {
    Write-Host "[INFO] ML models will be trained on first backend startup" -ForegroundColor Cyan
    Write-Host "   This requires sample datasets in:" -ForegroundColor Gray
    Write-Host "   - backend/datasets/voice_dataset/" -ForegroundColor Gray
    Write-Host "   - backend/recorded_data/" -ForegroundColor Gray
    Write-Host ""
    Write-Host "   Training will take 3-5 minutes on first run" -ForegroundColor Yellow
    $installationSteps["Models Training"] = $true
}

# Installation Summary
Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Green
Write-Host "  INSTALLATION SUMMARY" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Green
Write-Host ""

$completedSteps = ($installationSteps.Values | Where-Object { $_ -eq $true }).Count
$totalSteps = $installationSteps.Count

foreach ($step in $installationSteps.GetEnumerator()) {
    $status = if ($step.Value) { "[OK]" } else { "[WARN]" }
    Write-Host "  $status $($step.Key)" -ForegroundColor $(if ($step.Value) { "Green" } else { "Yellow" })
}

Write-Host ""
Write-Host "  Progress: $completedSteps/$totalSteps steps completed" -ForegroundColor Cyan

if ($completedSteps -eq $totalSteps) {
    Write-Host ""
    Write-Host "===============================================================================" -ForegroundColor Green
    Write-Host "  [SUCCESS] INSTALLATION COMPLETE!" -ForegroundColor Green
    Write-Host "===============================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "[READY] Ready to start the application!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Virtual environment location:" -ForegroundColor Cyan
    Write-Host "  $venvDir" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To run the application, choose one:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  1. Full Setup (Backend + Frontend + ngrok):" -ForegroundColor White
    Write-Host "     .\run-locally.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  2. Backend Only (with ngrok):" -ForegroundColor White
    Write-Host "     .\backend.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  3. Manual Startup (with venv activation):" -ForegroundColor White
    Write-Host "     .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "     cd backend; python app.py" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "[WARN] Some steps need attention" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please review the warnings above and run this script again." -ForegroundColor Yellow
}

# Configuration Checklist
Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "  [CHECKLIST] CONFIGURATION CHECKLIST" -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Before running the app, make sure you have:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  [ ] ngrok authenticated (if you have an account)" -ForegroundColor White
Write-Host "    Command: ngrok authtoken <YOUR_TOKEN>" -ForegroundColor Gray
Write-Host ""
Write-Host "  [ ] Voice dataset samples in: backend/datasets/voice_dataset/" -ForegroundColor White
Write-Host "    Subdirectories: healthy/, parkinson/" -ForegroundColor Gray
Write-Host ""
Write-Host "  [ ] (Optional) Vercel account setup for frontend deployment" -ForegroundColor White
Write-Host "    Set BACKEND_URL to: https://ostensible-unvibrant-clarisa.ngrok-free.dev" -ForegroundColor Gray
Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

# Next Steps
Write-Host "[DOCS] Documentation:" -ForegroundColor Cyan
Write-Host "  - README.md - Project overview" -ForegroundColor Gray
Write-Host "  - backend/requirements.txt - Python dependencies" -ForegroundColor Gray
Write-Host "  - frontend/.env.example - Environment variables" -ForegroundColor Gray
Write-Host ""

Write-Host "[LINKS] Useful Links:" -ForegroundColor Cyan
Write-Host "  - ngrok: https://ngrok.com" -ForegroundColor Gray
Write-Host "  - Flask: https://flask.palletsprojects.com" -ForegroundColor Gray
Write-Host "  - Vercel: https://vercel.com" -ForegroundColor Gray
Write-Host ""

Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
