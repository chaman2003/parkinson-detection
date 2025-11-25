# ngrok-launch.ps1
# Helper script to launch ngrok in a new window

Set-Location "$PSScriptRoot"
Write-Host "Starting ngrok tunnel..." -ForegroundColor Cyan
Start-Process -FilePath "ngrok.exe" -ArgumentList "http --url=https://elease-unmeaning-mireille.ngrok-free.dev 5000" -NoNewWindow -Wait
