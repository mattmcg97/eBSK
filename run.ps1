# Run the defence smoke-test decomposition from PowerShell.
#   .\run.ps1                      # uses SIS_UAT_CG_CURATED
#   .\run.ps1 -Database SIS_PROD_CG_CURATED
# Connection: .env at the repo root (same as the sis repo; see .env.example).
param([string]$Database = "SIS_UAT_CG_CURATED")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvPy = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $venvPy)) {
    if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }   # half-made venv

    # any installed Python 3 (py launcher first, then python on PATH)
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv .venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv .venv
    } else {
        throw "No Python found. Install Python 3.10-3.12 from python.org (tick 'Add to PATH')."
    }
    if (-not (Test-Path $venvPy)) { throw "Failed to create .venv - see the error above." }

    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Remove-Item -Recurse -Force .venv
        throw "Package install failed - see the error above."
    }
}

& $venvPy --version
& $venvPy decomp\decomp_defence.py $Database
