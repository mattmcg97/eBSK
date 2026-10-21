# Run the decomposition from PowerShell (offline: reads from Snowflake, writes locally).
#   .\run.ps1                              # defence on, latest 7 days, SIS_UAT_CG_CURATED
#   .\run.ps1 -Defence off                 # baseline: same smoke config without defence
#   .\run.ps1 -Defence corr -Days 27
#   .\run.ps1 -AsOf 2026-09-14 -Days 7     # one ISO week (Mon 7th - Sun 13th Sep)
#   .\run.ps1 -Dispersion weak             # data-driven NB dispersion (see README)
#   .\run.ps1 -Chains 4 -Draws 1000        # before trusting any numbers
#   .\run.ps1 -Database SIS_PROD_CG_CURATED
# Connection: .env at the repo root (same as the sis repo; see .env.example).
param(
    [string]$Database = "SIS_UAT_CG_CURATED",
    [ValidateSet("off", "on", "corr")][string]$Defence = "on",
    [ValidateSet("original", "weak")][string]$Dispersion = "original",
    [ValidateSet("pool", "week")][string]$Stream = "pool",
    [int]$Days = 7,
    [string]$AsOf = "latest",
    [int]$Chains = 2,
    [int]$Draws = 500,
    [int]$Tune = 1000
)

# ValidateSet ignores case; the Python side expects lower case
$Defence = $Defence.ToLowerInvariant()
$Dispersion = $Dispersion.ToLowerInvariant()
$Stream = $Stream.ToLowerInvariant()
$AsOf = $AsOf.Trim()
if ($AsOf -ieq "latest") { $AsOf = "latest" }

# 'Continue', not 'Stop': under Windows PowerShell 5.1 a redirected stderr line
# from pip/python (warnings, the progress bar) would otherwise abort the run.
# Failures are caught through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"

$saved = @{}
foreach ($k in "DECOMP_DEFENCE", "DECOMP_DISPERSION", "DECOMP_STREAM", "DECOMP_DAYS",
               "DECOMP_ASOF", "DECOMP_CHAINS", "DECOMP_DRAWS", "DECOMP_TUNE") {
    $saved[$k] = [Environment]::GetEnvironmentVariable($k, "Process")
}

Push-Location -LiteralPath $PSScriptRoot
try {
    $venvPy = ".\.venv\Scripts\python.exe"
    $stamp  = ".\.venv\requirements.sha256"

    if (-not (Test-Path -LiteralPath $venvPy)) {
        if (Test-Path -LiteralPath .venv) { Remove-Item -LiteralPath .venv -Recurse -Force }   # half-made venv

        # snowflake-snowpark-python supports Python < 3.15: prefer a known-good
        # version, then any Python 3 (py launcher first, then python on PATH)
        $made = $false
        if (Get-Command py -ErrorAction SilentlyContinue) {
            foreach ($v in "3.14", "3.13", "3.12", "3.11") {
                py "-$v" -c "import sys" 2>$null
                if ($LASTEXITCODE -eq 0) { py "-$v" -m venv .venv; $made = $true; break }
            }
            if (-not $made) { py -3 -m venv .venv; $made = $true }
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            python -m venv .venv; $made = $true
        }
        if (-not $made) { throw "No Python found. Install Python 3.12-3.14 from python.org (tick 'Add to PATH')." }
        if (-not (Test-Path -LiteralPath $venvPy)) { throw "Failed to create .venv - see the error above." }
        & $venvPy -m pip install --upgrade pip
    }

    # (re)install whenever requirements.txt has changed since the last install
    $reqHash = (Get-FileHash -LiteralPath requirements.txt -Algorithm SHA256).Hash
    $oldHash = if (Test-Path -LiteralPath $stamp) { "$(Get-Content -LiteralPath $stamp -Raw)".Trim() } else { "" }
    if ($reqHash -ne $oldHash) {
        & $venvPy -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "Package install failed - see the error above." }
        Set-Content -LiteralPath $stamp -Value $reqHash
    }

    $env:DECOMP_DEFENCE    = $Defence
    $env:DECOMP_DISPERSION = $Dispersion
    $env:DECOMP_STREAM     = $Stream
    $env:DECOMP_DAYS       = "$Days"
    $env:DECOMP_ASOF       = $AsOf
    $env:DECOMP_CHAINS     = "$Chains"
    $env:DECOMP_DRAWS      = "$Draws"
    $env:DECOMP_TUNE       = "$Tune"

    & $venvPy --version
    & $venvPy decomp\decomp_defence.py $Database
    if ($LASTEXITCODE -ne 0) { throw "The decomposition failed (exit code $LASTEXITCODE) - see the error above." }
}
finally {
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], "Process") }
    Pop-Location
}
