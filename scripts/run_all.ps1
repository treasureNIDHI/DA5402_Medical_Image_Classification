$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

wsl bash -lc "cd /home/specialvidhya/nidhi_DA5402/medical_classification && source envp/bin/activate && dvc repro && docker compose up --build -d"

Write-Host 'API running at http://localhost:8001'
Write-Host 'Health check: curl http://localhost:8001/health'
