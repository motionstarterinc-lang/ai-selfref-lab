# Windows setup: run from this folder in PowerShell:  .\setup.ps1
# (If scripts are blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
$ErrorActionPreference = "Stop"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Write-Error "Install Python 3.11+ from python.org first (tick 'Add to PATH')." }
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
if ($args -contains "--interp") { .\.venv\Scripts\pip.exe install -r requirements-interp.txt }
if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "Created .env - open it and paste your OpenRouter key." }
.\.venv\Scripts\python.exe -m pytest -q
Write-Host "`nDone. Start the app with:  .\.venv\Scripts\streamlit.exe run app.py"
