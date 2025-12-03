# Start ngrok from project root (runs in current terminal)
# Usage: .\run-ngrok.ps1 [-Port 5000] [-Domain "your-subdomain.ngrok-free.dev"]
param(
    [int]$Port = 5000,
    [string]$Domain = "elease-unmeaning-mireille.ngrok-free.dev"
)

Write-Host "Starting ngrok (port: $Port, domain: $Domain) in current terminal..." -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ngrokPath = Join-Path $scriptDir "ngrok.exe"

if (-not (Test-Path $ngrokPath)) {
    Write-Host "ERROR: ngrok.exe not found in project root ($ngrokPath)" -ForegroundColor Red
    Write-Host "Download ngrok from https://ngrok.com/download and place ngrok.exe in project root." -ForegroundColor Yellow
    exit 1
}

# Build arglist: prefer --domain if provided, otherwise default behavior
$argList = @("http", "--domain=$Domain", "$Port")

# Run ngrok in the current terminal and stream output
& $ngrokPath $argList

Write-Host "ngrok process exited." -ForegroundColor Yellow
