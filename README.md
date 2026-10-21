# eBSK

Bayesian decomposition for NBA2K25 esports matches (PyMC + NumPyro).

## Model

`decomp/decomp_defence.py` fits three linked submodels:

- **Efficiency** — Negative Binomial on field-goal points, offset `log(possessions)`:
  `β0 + player + team + HCA + [opponent defence] + [chemistry] + [stream] + [session] + [match]`
- **Pace** — Negative Binomial on total possessions, offset `log(duration)`, summing both sides' effects.
- **Free throws** — Binomial make rate per player/team; attempts per possession exported empirically.

Optional terms are switched in `FitConfig`. `defence_effect=True` adds the opponent's
`def_player + def_team` to each side's efficiency (positive = weak defence).

The script is currently configured as a **smoke test**: defence on, chemistry off,
7-day window ending at `asof`, 2 chains × 500 draws.

## Running from PowerShell

```powershell
# connection: either a named connection from ~/.snowflake/connections.toml
$env:SNOWFLAKE_CONNECTION = "my_conn"
# ...or explicit settings (no password => browser SSO)
$env:SNOWFLAKE_ACCOUNT   = "xy12345.eu-west-1"
$env:SNOWFLAKE_USER      = "you@company.com"
$env:SNOWFLAKE_ROLE      = "..."
$env:SNOWFLAKE_WAREHOUSE = "..."

.\run.ps1                       # creates .venv on first run
```

If scripts are blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Outputs go to `.\decomp_out_def\` (override with `$env:DECOMP_OUT`) and are uploaded to
`@<DB>.MODELLING.STAGE_BASKETBALL_POSSESSION/decomp_out_def`.

Native Windows JAX is CPU-only; use WSL2 for GPU. Inside a Snowflake notebook the script
uses the active session as before.

## What to check in the output

- `sigma_def_player` vs `sigma_player` — size of the missing defence effect
- whether `sigma_player` falls and `alpha_pts` rises once defence is in
- `corr(off, def)` across players — schedule bias in current ratings
