# eBSK

Bayesian decomposition for NBA2K25 esports matches (PyMC + NumPyro). Offline testing:
the run only reads from Snowflake (two `SELECT`s) and writes results locally.

## Model

`decomp/decomp_defence.py` fits three linked submodels:

- **Efficiency** — Negative Binomial (or Normal with var = φ × mean under `-Dispersion free`) on field-goal points (points − FT made), offset `log(possessions)`:
  `β0 + player + team + HCA + [opponent defence] + [chemistry] + [stream] + [session] + [match]`
- **Pace** — the same noise choice on total possessions, offset `log(duration)`, summing both sides' effects.
- **Free throws** — Binomial make rate per player/team; attempts per possession exported empirically
  (credited to the shooter only — opponent fouling is not modelled).

It is set up as a **smoke test**: chemistry off, light sampling. `-Defence off` is therefore a
like-for-like baseline for `on`/`corr`, **not** the production model (production fits pair chemistry).

### Switches

| `run.ps1` flag | Values | What it does |
|---|---|---|
| `-Defence` | `off`, `on` (default), `corr` | `on` adds the opponent's `def_player + def_team` to each side's efficiency. `corr` also correlates each player's (attack, defence) pair and reports `rho_player_od`, plus the spreads of **quality** (`off − def`, moves the margin) and **openness** (`off + def`, moves the total). `def` > 0 = opponents score more. |
| `-Dispersion` | `original` (default), `weak`, `free` | The noise model — see **Noise** below. `original` = the production model (same posterior; team pace and player FT are re-parameterised for sampling). `free` = the data set var/mean on both legs, below 1 as well as above. |
| `-Stream` | `pool` (default), `week` | Only used when EVENT has no `STREAM_NUMBER` (your tables do have it, so the term is keyed on stream × ISO week and this flag does nothing). Without it, `week` is the original fallback — just the ISO week, which controls nothing — and `pool` keys the term on each week's connected player pools. |
| `-Days`, `-AsOf` | `7`, `latest` | Window of `Days` up to `AsOf` (a date, or the latest usable match). |
| `-Chains`, `-Draws`, `-Tune` | `2`, `500`, `1000` | Use `-Chains 4 -Draws 1000` before reading any numbers. |

## Noise

| Leg | Production (`original`) | Problem | `weak` | `free` |
|---|---|---|---|---|
| FG points per side, given possessions | NegBinomial, `alpha_pts ~ HalfNormal(3)` | The prior has 99.9% of its mass below 10, yet the fit sits at ~61: there the prior's pull balances the likelihood, so alpha is a tug-of-war, not a measurement. With the weak prior the real data go to the Poisson boundary (89% lower bound ~3,500): var/mean is at most ~1.015, but the original fit implies 1.84. | NegBinomial, `log alpha_pts ~ Normal(log 100, 1.5)` | Normal, var = `phi_pts` × mean |
| Total possessions, given duration | NegBinomial, `alpha_poss ~ HalfNormal(5)` | Same tug-of-war (~94), implying var/mean 2.1. A fixed game clock makes possessions *under*-dispersed, which no NegBinomial can represent. | unchanged (original prior) | Normal, var = `phi_poss` × mean |
| FT makes | Binomial | — | unchanged | unchanged |

Inflated noise also makes the model distrust the data, so ratings are shrunk harder: on the 7-day
run, posterior-mean attack ratings have 59% and defence 66% of the spread they have with the
noise prior relaxed. `var_mean_pts` / `var_mean_poss` are reported in every mode for comparison.
`free`-mode `params_summary.json` has `phi_*` instead of `alpha_*`, so it is not a drop-in for the simulator.

## Backtest

`.\run.ps1 -Backtest` fits rolling windows and scores the days after them, for several configs on
the same folds (default `off:original,on:original,off:free,on:free`; 7 folds of 1 test day every
2 days, 7 training days). Results go to `backtest_out_<...>\` — `summary.csv` (per config),
`comparisons.csv` (paired, match by match; |z| > 2 is clear, and `z_fold_blocks` is the more
cautious version) and `matches.csv`. Re-running with the same settings resumes where it stopped;
pass an explicit `-AsOf` date if you may resume on a later day (the default `latest` moves when new
data arrive), and a `settings.json` check refuses to mix runs with different settings.

Flags: `-Configs` (e.g. `-Configs on:free,off:free`), `-Folds`, `-TrainDays`, `-TestDays`,
`-StepDays`, `-AsOf` (end of the last test fold), `-Stream`, `-Database`, and `-Chains/-Draws/-Tune`
for each fit. `-Defence`, `-Dispersion` and `-Days` are ignored — use `-Configs` and `-TrainDays`.
Optional: `$env:BT_SIM_DRAWS` (predictive draws, default 400), `$env:BT_OUT` (output folder).
Players, teams and stream-weeks not seen in training are drawn from their population
distribution, and `summary.csv` also splits every config into new-week and same-week rows.

What it answers:
- **Defence**: `lpd_pts` (log score of FG points given possessions), total/margin CRPS, on vs off.
- **Noise**: interval coverage (`*_in50/80/95` should be ~0.50/0.80/0.95), predicted vs actual
  error sd, and the residual var/mean of FG points and possessions around each config's predictions
  against what that config assumes (a config with biased means shows extra residual variance, so read
  the data's own dispersion off the best-scoring config), plus the home/away residual correlation.

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

Outputs go to `.\decomp_out_<mode>_<window end>_<days>d[_<dispersion>][_week]\`: `params_summary.json`,
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
