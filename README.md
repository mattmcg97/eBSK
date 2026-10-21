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
7-day window ending at the latest match in the table, 2 chains × 500 draws.

## Running from PowerShell

Connection uses the same `.env` as the `sis` repo (SSO by default) — copy it across:

```powershell
Copy-Item ..\sis\.env .env    # or: Copy-Item .env.example .env and fill in
.\run.ps1                       # creates .venv on first run
```

Window: `$env:DECOMP_ASOF = "2026-10-04"` (default `latest`) and `$env:DECOMP_DAYS = "14"` (default 7).

If scripts are blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Outputs go to `.\decomp_out_def\` (override with `$env:DECOMP_OUT`). Set `$env:DECOMP_UPLOAD = "1"` to also upload to
`@<DB>.MODELLING.STAGE_BASKETBALL_POSSESSION/decomp_out_def` (needs WRITE on the stage).

Native Windows JAX is CPU-only; use WSL2 for GPU. Inside a Snowflake notebook the script
uses the active session as before.

## What to check in the output

- `sigma_def_player` vs `sigma_player` — size of the missing defence effect
- whether `sigma_player` falls and `alpha_pts` rises once defence is in
- `corr(off, def)` across players — schedule bias in current ratings
