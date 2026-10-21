# Run the defence smoke-test decomposition from PowerShell.
#   .\run.ps1                      # uses SIS_UAT_CG_CURATED
#   .\run.ps1 -Database SIS_PROD_CG_CURATED
# Connection: set $env:SNOWFLAKE_CONNECTION (named connection in
# ~/.snowflake/connections.toml) or the SNOWFLAKE_* variables in README.md.
param([string]$Database = "SIS_UAT_CG_CURATED")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

.\.venv\Scripts\python.exe decomp\decomp_defence.py $Database
