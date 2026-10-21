# eBSK

Bayesian decomposition for NBA2K25 esports matches (PyMC + NumPyro). Offline testing:
the run only reads from Snowflake (two `SELECT`s) and writes results locally.

## Model

`decomp/decomp_defence.py` fits three linked submodels:

- **Efficiency** — Negative Binomial on field-goal points (points − FT made), offset `log(possessions)`:
  `β0 + player + team + HCA + [opponent defence] + [chemistry] + [stream] + [session] + [match]`
- **Pace** — Negative Binomial on total possessions, offset `log(duration)`, summing both sides' effects.
- **Free throws** — Binomial make rate per player/team; attempts per possession exported empirically
  (credited to the shooter only — opponent fouling is not modelled).

It is set up as a **smoke test**: chemistry off, light sampling. `-Defence off` is therefore a
like-for-like baseline for `on`/`corr`, **not** the production model (production fits pair chemistry).

### Switches

| `run.ps1` flag | Values | What it does |
|---|---|---|
| `-Defence` | `off`, `on` (default), `corr` | `on` adds the opponent's `def_player + def_team` to each side's efficiency. `corr` also correlates each player's (attack, defence) pair and reports `rho_player_od`, plus the spreads of **quality** (`off − def`, moves the margin) and **openness** (`off + def`, moves the total). `def` > 0 = opponents score more. |
| `-Dispersion` | `original` (default), `weak` | `original` keeps `alpha ~ HalfNormal(3/5)`, which **binds**: FG points given possessions are close to Poisson, but the prior holds `alpha_pts` near 50–60, overstating score variance. `weak` lets the data set it. |
| `-Stream` | `pool` (default), `week` | The data have no `STREAM_NUMBER`. `week` is the original key: just the ISO week, which controls nothing. `pool` keys the stream term on each week's connected player pools, so pool-level scoring shifts don't load onto attack *and* defence (or chemistry). |
| `-Days`, `-AsOf` | `7`, `latest` | Window of `Days` up to `AsOf` (a date, or the latest usable match). |
| `-Chains`, `-Draws`, `-Tune` | `2`, `500`, `1000` | Use `-Chains 4 -Draws 1000` before reading any numbers. |

## Running from PowerShell

```powershell
Copy-Item ..\sis\.env .env      # same .env as the sis repo (SSO by default)
.\run.ps1                        # creates .venv and installs packages on first run
.\run.ps1 -Defence off           # baseline
.\run.ps1 -AsOf 2026-09-14 -Days 7   # a single ISO week (Mon 7th - Sun 13th Sep)
```

- `run.ps1` reinstalls packages whenever `requirements.txt` changes, and restores your shell's
  folder and `DECOMP_*` variables afterwards. Use the flags, not `$env:DECOMP_*` — the flags
  overwrite them. A leftover `$env:DECOMP_OUT` still overrides the output folder.
- Python 3.12–3.14 (Snowpark doesn't support 3.15 yet). Native Windows JAX is CPU-only; use WSL2 for GPU.
- If scripts are blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- Close `runs_comparison.csv` in Excel before a run, or that one file won't be updated.

Outputs go to `.\decomp_out_<mode>_<window end>_<days>d\`: `params_summary.json`,
`scalar_summary.csv`, `player_map.csv`, `team_map.csv`, `posterior.nc` (posterior + sample stats;
reload with `az.from_netcdf`), and, for `on`/`corr`, `player_off_def.csv`. Each run adds a row with
its settings and headline numbers to `runs_comparison.csv`.

`params_summary.json` from `on`/`corr` is **not** a drop-in for the existing simulator: it ignores
the `def_*` keys, and `alpha_pts` was fitted with defence explaining part of the variance.

Inside a Snowflake notebook the script uses the active session, and outputs go to a local
folder there (nothing is uploaded).

## Reading the output

1. **`[check]` lines** from loading: dropped rows and why, `TOTAL` vs `HOME + AWAY` possessions,
   how the table's `*_FIELD_GOALS` relate to `POINTS − FT_SCORED`, the side-mapping FT% check,
   and which stream key was used.
2. **Sampler health**: divergences, tree-depth hits, worst `r_hat` and lowest ESS over all effects.
   Don't interpret anything unless every `r_hat` ≤ 1.01, ESS ≥ 400 and there are no divergences.
3. **`sigma_def_player` vs `sigma_player`**: how much opponent identity matters relative to attack.
   That shows opponents matter, not that it is *defending* — compare with the fouls correlation.
4. **`corr(off, def)`**: the correlation of posterior means is biased (and inflated in `corr`
   mode); the per-draw figure is attenuated toward 0; in `corr` mode the estimate is `rho_player_od`.
   Pool-level shifts are removed by `-Stream pool` and the within-pool line, but a player-level
   common factor survives both, so "style" is one explanation, not a finding.
5. **`alpha_pts`** only means something with `-Dispersion weak`; with `original` it is mostly prior.
