#!/usr/bin/env python3
"""
Bayesian decomposition for NBA2K25 match data (fouls/FT variant).

Models fitted:
  Scoring  — NB2, offset=log(possessions), learns FG-efficiency per player/team
  Pace     — NB2, offset=log(duration),    learns pace per player+team pair
  FT       — Binomial (if FT columns present), learns FT make-rate per player/team
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

import snowflake.snowpark as snowpark

# --- run settings (from run.ps1 / env), validated BEFORE connecting so a typo
# fails before the SSO login and the data pull ------------------------------
def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()

DEFENCE_MODE = _env("DECOMP_DEFENCE", "on").lower()
DISPERSION = _env("DECOMP_DISPERSION", "original").lower()
STREAM_FALLBACK = _env("DECOMP_STREAM", "pool").lower()
ASOF = _env("DECOMP_ASOF", "latest")
ASOF = "latest" if ASOF.lower() == "latest" else ASOF
try:
    DAYS = int(_env("DECOMP_DAYS", "7"))
    CHAINS = int(_env("DECOMP_CHAINS", "2"))
    DRAWS = int(_env("DECOMP_DRAWS", "500"))
    TUNE = int(_env("DECOMP_TUNE", "1000"))
    if ASOF != "latest":
        pd.to_datetime(ASOF, utc=True)
except ValueError as e:
    sys.exit(f"Bad run setting: {e}")
for _name, _val, _ok in (("DECOMP_DEFENCE", DEFENCE_MODE, ("off", "on", "corr")),
                         ("DECOMP_DISPERSION", DISPERSION, ("original", "weak")),
                         ("DECOMP_STREAM", STREAM_FALLBACK, ("pool", "week"))):
    if _val not in _ok:
        sys.exit(f"{_name}={_val!r} -- must be one of {_ok}")

try:
    # inside a Snowflake notebook / stored procedure
    session = snowpark.context.get_active_session()
except Exception:
    # local run (e.g. PowerShell): credentials from .env, as in the sis repo
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from snowflake_connect import get_session
    session = get_session()

DB_NAME = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
           else "SIS_UAT_CG_CURATED")
print("DBNAME: " + DB_NAME)

session.use_database(DB_NAME)


@dataclass(frozen=True)
class FitConfig:
    draws: int = 2000
    tune: int = 2500
    chains: int = 4
    target_accept: float = 0.95
    random_seed: int = 42
    hca_mode: str = "global"

    # ======================================================================
    # v2 additions. All default OFF so this file reproduces production
    # exactly until a switch is turned on.
    #
    # The possession variant has TWO legs, and these terms do not
    # automatically belong to both. Each is switchable per leg.
    # ======================================================================

    # --- CHEMISTRY: player x team interaction -----------------------------
    # Claim: a gamer on a team whose unobserved attacking/defensive attributes
    # suit them scores beyond the SUM of the gamer and team effects.
    # Measured before fitting (119,486 match-sides, stream-controlled):
    #   chemistry sd ~0.0224 -- about half the team main effect (~0.041)
    #   ratio to sampling noise 1.55 (1.64 uncontrolled: ~9% was the rota)
    # "pair"   = hierarchical random effect per (player, team). Identified,
    #            direct read on sigma_chem, no generalisation to unseen pairs.
    # "factor" = low-rank latent attributes, chem = sigma * (u_p . v_t). This
    #            IS the stated mechanism and generalises, which the sim needs.
    chem_mode: str = "pair"
    chem_rank: int = 1
    # --- pair-effect distribution -----------------------------------------
    # "normal"   : chem_raw ~ Normal(0,1) scaled by sigma_chem. The original.
    # "studentt" : chem_raw ~ StudentT(nu) scaled. Heavier tails, so extreme
    #              pairings shrink far less while ordinary ones still do.
    #
    # WHY. Out-of-fold validation of the pace chemistry exposed a SHRINKAGE
    # problem, not a signal problem:
    #     in-sample, using the fitted posterior means   +0.33% RMSE
    #     out of fold, re-estimated at n/(n+5)          +4.29% RMSE
    # Out-of-fold cannot legitimately beat in-sample on the same quantity. The
    # arms differ only in the chemistry VALUES: the posterior means are heavily
    # shrunk; the OOF estimator is nearly unshrunk at these pair counts (37-260
    # matches each). So the Normal prior pulls the extremes in far too hard, and
    # shipping chem_pace_pair_mean as it stands would deliver roughly a tenth of
    # the available gain.
    #
    # The pairings look heavy-tailed rather than Normal: a few strong effects
    # (GALAXY/Toronto +5.0% pace, SAINT JR/Toronto -2.8%) against a mass of
    # near-zero ones. StudentT with low nu is that shape -- it keeps shrinking
    # noise while letting genuine outliers stand.
    chem_tail: str = "normal"
    chem_tail_nu: float = 4.0
    # PPP leg = efficiency. PACE leg = does a suited team also make them play
    # FASTER? The direct variant cannot separate these; this one can, and a
    # pace-side chemistry finding would be the more useful of the two.
    chem_on_ppp: bool = True
    chem_on_pace: bool = False

    # --- MATCH-LEVEL COMMON EFFECT ----------------------------------------
    # NOTE the meaning CHANGES in this variant. In the direct model (points per
    # second, no pace sub-model) a match effect IS the latent tempo. Here tempo
    # is already modelled, so this is shared efficiency BEYOND shared pace --
    # a much cleaner object, and the one that maps onto the common-mode
    # component of trader error (measured sd 0.073 on 59,348 fixtures).
    # Expect sigma_match to come out SMALLER here than in the direct fit; that
    # difference is the part the pace model already explains.
    # No pace-leg equivalent: both sides share one possession count by
    # construction, so there is nothing for a common pace term to absorb.
    match_effect: bool = False

    # --- STREAM: a CONTROL, not a finding ---------------------------------
    # Gamers are allocated to a stream for a week and streams have DISTINCT
    # TEAM POOLS (total-variation distance from the global team mix: median
    # 0.354, max 0.457 across 6 streams). Without this, sigma_chem absorbs
    # stream-level scoring variation. Fit it whenever chem_mode != "off".
    stream_effect: bool = True
    stream_by_week: bool = True
    stream_on_pace: bool = True      # streams may also differ in tempo

    # --- SESSION: (player, calendar day) ----------------------------------
    # Daily-resolution autocorrelation of player residuals: 0.159 at lag 1 day,
    # 0.069 at lag 2, flat to 14 days. A CLIFF, not a decay -- a gamer's
    # matches within a session move together and it is gone by tomorrow.
    # That is NOT form: recency weighting cannot capture a one-day effect and
    # would cost effective sample for nothing (tau=120d already costs 28%).
    #
    # This is the one term that is SIDE-SPECIFIC rather than common-mode, so it
    # widens the MARGIN as well as the total -- the leg the trader-error
    # decomposition put at sd 0.128 and which the simulator has nothing for.
    session_effect: bool = False
    session_on_pace: bool = False

    # --- DEFENCE: opponent player/team effect on the PPP leg ---------------
    # One gamer controls both ends, so each side's efficiency should also
    # depend on who it is playing AGAINST. Positive def_* = opponent scores
    # more, i.e. a WEAK defence.
    #   "off"  : no defence term (as the original)
    #   "on"   : independent defence effects for player and team
    #   "corr" : each player gets a CORRELATED (attack, defence) pair, so the
    #            two pool information; rho_player_od is the correlation, and
    #            QUALITY (off-def, moves the margin) / OPENNESS (off+def, moves
    #            the total) their spreads. Team defence stays independent.
    # A first 7-day "on" fit gave corr(off, def) of posterior means ~ +0.54.
    # That statistic is biased, and positive correlation is also produced by
    # pool-level scoring shifts no other term absorbs -- see stream_fallback.
    defence_mode: str = "off"

    # --- DISPERSION PRIORS ------------------------------------------------
    # "original": alpha_pts ~ HalfNormal(3), alpha_poss ~ HalfNormal(5). These
    #   BIND. FG points given possessions are close to Poisson (MLE alpha in
    #   the hundreds; a {0,2,3}-per-possession simulation gives ~150-190), but
    #   the prior holds alpha_pts near 50-60 -- the 7-day fit's 61 is the
    #   prior, not the data. Score variance is overstated (~25-50% at mu ~55)
    #   and "alpha_pts rises as structure leaves" is muted to noise.
    # "weak": log(alpha_pts) ~ Normal(log 100, 1.5) -- 95% prior range about
    #   5 to 1,900, so the data set alpha. (A HalfNormal on 1/sqrt(alpha) gave
    #   the right answer in one chain but left the other stuck from a poor start.)
    #   Points only: possessions given duration are typically UNDER-dispersed,
    #   which no NB can represent -- a free alpha_poss runs off to infinity and
    #   wrecks the sampler, while the original prior pins it and overstates
    #   possession variance. See the [check] on TOTAL_POSSESSIONS at load; the
    #   real fix there is a different likelihood, not a different prior.
    dispersion_prior: str = "original"

    # --- STREAM FALLBACK (no STREAM_NUMBER in the data) --------------------
    # The SQL never returns STREAM_NUMBER, so the original stream key is just
    # "NA|<ISO week>": a 1-2 level week effect that controls nothing. Gamers
    # play in closed pools for a week, so any pool-level scoring shift lands
    # on BOTH off and def of every player in the pool (positive corr(off,def))
    # and on chemistry.
    #   "pool" : key the stream term on the connected component of the
    #            player-match graph WITHIN each ISO week -- the closed pools,
    #            if allocation is weekly -- so the stream term absorbs them.
    #   "week" : the original behaviour.
    stream_fallback: str = "pool"


def load_match_features(
        time_col: str = "SCHEDULED_START_TIME_UTC",
        asof: str | None = None,
        lookback_days: int | None = None,
) -> pd.DataFrame:
    # The stream control in fit_model needs STREAM_NUMBER, which is not in
    # POSSESSION_NB_MATCH_STATS. Take it from EVENT when it exists there;
    # otherwise the "stream" effect silently degrades to a WEEK effect.
    try:
        _ev_cols = list(session.sql(
            "SELECT * FROM SIS_PROD_CG_CURATED.SHARED.EVENT LIMIT 0").to_pandas().columns)
    except Exception as e:
        print(f"[load_match_features] could not list EVENT columns ({type(e).__name__})")
        _ev_cols = None
    _stream_sel = ", e.STREAM_NUMBER" if _ev_cols and "STREAM_NUMBER" in _ev_cols else ""
    df = session.sql(
            f"""
            SELECT s.*, e.SCHEDULED_START_TIME_UTC{_stream_sel}
            FROM MODELLING.POSSESSION_NB_MATCH_STATS s
            LEFT JOIN SIS_PROD_CG_CURATED.SHARED.EVENT e
              ON e.MATCH_CODE = s.MATCH_CODE
            """
).to_pandas()
    if "STREAM_NUMBER" not in df.columns:
        _cand = ("unknown (EVENT columns could not be listed)" if _ev_cols is None
                 else [c for c in _ev_cols if "STREAM" in c.upper()] or "none")
        print("[check] no STREAM_NUMBER -- stream term falls back to "
              f"stream_fallback. STREAM-like EVENT columns: {_cand}")

    required = [
        "MATCH_CODE",
        "PLAYER_1_HANDLE", "PLAYER_2_HANDLE",
        "PLAYER_1_TEAM", "PLAYER_2_TEAM",
        "AWAY_POINTS", "HOME_POINTS",
        "AWAY_POSSESSIONS", "HOME_POSSESSIONS", "TOTAL_POSSESSIONS",
        "GAME_DURATION_SECONDS",
    ]
    print(df.columns)
    ft_cols = ["AWAY_FREE_THROW_SCORED", "HOME_FREE_THROW_SCORED", "AWAY_FREE_THROW_ATTEMPTED", "HOME_FREE_THROW_ATTEMPTED"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    def _drop(mask, why):
        nonlocal df
        k = int(mask.sum())
        if k:
            print(f"[check] dropped {k} rows: {why}")
            df = df[~mask].copy()

    # --- validity first, so "latest" and the empty check see usable rows ---
    _drop(df["MATCH_CODE"].duplicated(), "duplicate MATCH_CODE (EVENT join fan-out?)")
    for c in ["AWAY_POINTS", "HOME_POINTS", "AWAY_POSSESSIONS", "HOME_POSSESSIONS",
              "TOTAL_POSSESSIONS", "GAME_DURATION_SECONDS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    _drop(df["GAME_DURATION_SECONDS"].isna(), "missing duration")
    _drop(df[["AWAY_POINTS", "HOME_POINTS"]].isna().any(axis=1), "missing points")
    # log(possessions) is the PPP offset: a side with 0 possessions gives -inf
    _drop(~((df["HOME_POSSESSIONS"] > 0) & (df["AWAY_POSSESSIONS"] > 0)
            & (df["TOTAL_POSSESSIONS"] > 0) & (df["GAME_DURATION_SECONDS"] > 0)),
          "zero/missing possessions or duration")
    # An NB can only be OVER-dispersed (var/mean >= 1). Pooled var/mean within
    # 1-minute duration bins is an UPPER bound on the residual dispersion the
    # pace leg sees (player pace differences are still in it).
    _g = df.groupby(df["GAME_DURATION_SECONDS"] // 60)["TOTAL_POSSESSIONS"]
    _dof = _g.count() - 1
    _vm = float((_g.var() * _dof).sum() / max((_g.mean() * _dof).sum(), 1e-9))
    print(f"[check] TOTAL_POSSESSIONS var/mean within duration bins = {_vm:.2f}"
          + ("  -- BELOW 1: under-dispersed; the NB pace leg cannot fit this, so "
             "alpha_poss is set by its prior and possession variance is overstated"
             if _vm < 1 else ""))
    _tot = (df["TOTAL_POSSESSIONS"] != df["HOME_POSSESSIONS"] + df["AWAY_POSSESSIONS"]).mean()
    print(f"[check] TOTAL_POSSESSIONS != HOME + AWAY on {_tot:.1%} of rows "
          "(pace leg uses TOTAL, PPP leg uses the sides)")

    _has_ft = all(c in df.columns for c in ft_cols)
    if _has_ft:
        for c in ft_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
        # FG points below are POINTS - FT_SCORED, which assumes POINTS
        # INCLUDES free throws. The table has its own *_FIELD_GOALS; compare.
        if "HOME_FIELD_GOALS" in df.columns:
            _tab = pd.to_numeric(df["HOME_FIELD_GOALS"], errors="coerce")
            _calc = df["HOME_POINTS"] - df["HOME_FREE_THROW_SCORED"]
            print(f"[check] table HOME_FIELD_GOALS: equals POINTS-FT_SCORED on "
                  f"{(_tab == _calc).mean():.1%} of rows, equals POINTS on "
                  f"{(_tab == df['HOME_POINTS']).mean():.1%}; "
                  f"sum(POINTS-FT)/sum(table) = {_calc.sum() / max(_tab.sum(), 1):.2f}")
        _drop((df["HOME_FREE_THROW_SCORED"] > df["HOME_FREE_THROW_ATTEMPTED"])
              | (df["AWAY_FREE_THROW_SCORED"] > df["AWAY_FREE_THROW_ATTEMPTED"]),
              "FT scored > attempted (impossible under the Binomial)")
        df["AWAY_FIELD_GOALS"] = (df["AWAY_POINTS"] - df["AWAY_FREE_THROW_SCORED"]).clip(lower=0)
        df["HOME_FIELD_GOALS"] = (df["HOME_POINTS"] - df["HOME_FREE_THROW_SCORED"]).clip(lower=0)
    else:
        df["AWAY_FIELD_GOALS"] = df["AWAY_POINTS"]
        df["HOME_FIELD_GOALS"] = df["HOME_POINTS"]

    id_cols = ["PLAYER_1_HANDLE", "PLAYER_2_HANDLE", "PLAYER_1_TEAM", "PLAYER_2_TEAM"]
    for c in id_cols:
        df[c] = df[c].astype("string").str.strip()
        df.loc[df[c].isin(["", "None", "nan", "NaN"]), c] = pd.NA

    _drop(df[id_cols].isna().any(axis=1), "missing player/team identifiers")

    if _has_ft:
        # Side mapping: AWAY_* columns are assumed to belong to PLAYER_1. FT% is
        # the shooter's own skill, so a player's FT% as PLAYER_1 (AWAY columns)
        # and as PLAYER_2 (HOME columns) should correlate across players.
        # ~0 means no FT skill spread OR a reversed mapping (then every PPP and
        # FT effect belongs to the opponent).
        _p1 = df.groupby("PLAYER_1_HANDLE")[["AWAY_FREE_THROW_SCORED", "AWAY_FREE_THROW_ATTEMPTED"]].sum()
        _p2 = df.groupby("PLAYER_2_HANDLE")[["HOME_FREE_THROW_SCORED", "HOME_FREE_THROW_ATTEMPTED"]].sum()
        _p1.columns = _p2.columns = ["made", "att"]
        _j = _p1.join(_p2, lsuffix="_1", rsuffix="_2", how="inner")
        _j = _j[(_j["att_1"] >= 30) & (_j["att_2"] >= 30)]
        if len(_j) >= 10:
            _c = np.corrcoef(_j["made_1"] / _j["att_1"], _j["made_2"] / _j["att_2"])[0, 1]
            print(f"[check] side mapping: corr(FT% as PLAYER_1, FT% as PLAYER_2) = {_c:.2f} "
                  f"over {len(_j)} players; noise level about +/-{2 / np.sqrt(len(_j)):.2f}. "
                  "Well above that supports AWAY = PLAYER_1; within it is inconclusive")

    # --- then the time window ---------------------------------------------
    if asof is not None:
        if time_col not in df.columns:
            raise ValueError(f"--asof provided but time_col '{time_col}' not found")
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        _drop(df[time_col].isna(), "no start time (no EVENT match)")
        if df.empty:
            raise ValueError("No usable matches at all -- see the [check] lines above.")
        print(f"[load_match_features] data spans {df[time_col].min()} -> "
              f"{df[time_col].max()} ({len(df)} usable matches)")
        if asof == "latest":
            asof_ts = df[time_col].max() + pd.Timedelta(1, unit="s")
        else:
            asof_ts = pd.to_datetime(asof, utc=True)
        df = df[df[time_col] < asof_ts]
        if lookback_days is not None:
            start_ts = asof_ts - pd.Timedelta(int(lookback_days), unit="D")
            df = df[df[time_col] >= start_ts]

    if "STREAM_NUMBER" in df.columns:
        # one string form for NUMBER or VARCHAR columns ("3", not "3.0")
        _sn = df["STREAM_NUMBER"]
        _num = pd.to_numeric(_sn, errors="coerce")
        _sn = (_num.astype("Int64").astype("string") if _num.notna().sum() == _sn.notna().sum()
               else _sn.astype("string").str.strip())
        print(f"[check] STREAM_NUMBER missing on {int(_sn.isna().sum())} of {len(_sn)} rows")
        df["STREAM_NUMBER"] = _sn.fillna("NA")

    if df.empty:
        raise ValueError(f"No usable matches in the window (asof={asof}, "
                         f"lookback_days={lookback_days}). Check the date range above.")
    return df


def make_index(series: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
    s = series.astype("string")
    mask = s.notna()
    cats = pd.Index(sorted(s[mask].unique()))
    mapping = {k: i for i, k in enumerate(cats)}
    idx = s.map(mapping).to_numpy()

    if pd.isna(idx).any():
        bad = s[pd.isna(idx)].unique().tolist()[:10]
        raise ValueError(f"Unmapped keys in indexing: {bad}")

    map_df = pd.DataFrame({"key": cats, "index": np.arange(len(cats), dtype=int)})
    return idx.astype(int), map_df


def stream_keys(df: pd.DataFrame, cfg: FitConfig) -> tuple[pd.Series, str]:
    """One key per match row for the stream term, and what kind of key it is."""
    day = pd.to_datetime(df["SCHEDULED_START_TIME_UTC"], utc=True, errors="coerce")
    iso = day.dt.isocalendar()
    week = (iso.week.astype("string") + "-" + iso.year.astype("string")).fillna("NA")
    if "STREAM_NUMBER" in df.columns:
        st = df["STREAM_NUMBER"].astype("string").fillna("NA")
        return (st + "|" + week) if cfg.stream_by_week else st, "stream"
    if cfg.stream_fallback == "week":
        return (("NA|" + week) if cfg.stream_by_week
                else pd.Series("NA", index=df.index, dtype="string")), "week"
    # "pool": connected components of the player-match graph within each week
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    a = (week + "|" + df["PLAYER_1_HANDLE"].astype("string")).tolist()
    b = (week + "|" + df["PLAYER_2_HANDLE"].astype("string")).tolist()
    for u, v in zip(a, b):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    return pd.Series([find(u) for u in a], index=df.index, dtype="string"), "pool"


def fit_model(df: pd.DataFrame, cfg: FitConfig):
    import pymc as pm
    import arviz as az

    def center(x):
        return x - pm.math.mean(x)

    away_player = df["PLAYER_1_HANDLE"]
    home_player = df["PLAYER_2_HANDLE"]
    away_team = df["PLAYER_1_TEAM"]
    home_team = df["PLAYER_2_TEAM"]

    both_players = pd.concat([away_player, home_player], ignore_index=True)
    both_teams = pd.concat([away_team, home_team], ignore_index=True)

    player_idx_all, _ = make_index(both_players)
    team_idx_all, _ = make_index(both_teams)

    n = len(df)
    away_p_idx = player_idx_all[:n]
    home_p_idx = player_idx_all[n:]
    away_t_idx = team_idx_all[:n]
    home_t_idx = team_idx_all[n:]

    n_players = int(max(away_p_idx.max(), home_p_idx.max()) + 1)
    n_teams = int(max(away_t_idx.max(), home_t_idx.max()) + 1)

    # --- grouping indices for the v2 variance components ------------------
    _day = None
    for _tc in ("SCHEDULED_START_TIME_UTC", "EVENT_DATE_TIME", "MATCH_DATE"):
        if _tc in df.columns:
            _day = pd.to_datetime(df[_tc], utc=True, errors="coerce").dt.floor("D")
            break
    if _day is None:
        _day = pd.Series(pd.Timestamp("2000-01-01"), index=df.index)

    _stream_key, _stream_kind = stream_keys(df, cfg)
    stream_idx, _ = make_index(_stream_key)
    n_streams = int(stream_idx.max() + 1)
    # A one-level stream term is identically 0 after centring: its sigma would
    # be sampled from the prior alone and still exported as if fitted.
    use_stream = cfg.stream_effect and n_streams >= 2
    if cfg.stream_effect:
        print(f"  stream term: {n_streams} level(s) keyed by {_stream_kind}"
              + ("" if use_stream else " -- fewer than 2, so it is SKIPPED"))

    # session = (player, calendar day); one per SIDE, so two per match row
    _dstr = _day.dt.strftime("%Y%m%d")
    _both_sess = pd.concat([away_player.astype(str) + "|" + _dstr,
                            home_player.astype(str) + "|" + _dstr],
                           ignore_index=True)
    sess_idx_all, _ = make_index(_both_sess)
    sess_away_idx = sess_idx_all[:len(df)]
    sess_home_idx = sess_idx_all[len(df):]
    n_sessions = int(sess_idx_all.max() + 1)

    y_home = df["HOME_FIELD_GOALS"].to_numpy(dtype=int)
    y_away = df["AWAY_FIELD_GOALS"].to_numpy(dtype=int)
    poss_home = df["HOME_POSSESSIONS"].to_numpy()
    poss_away = df["AWAY_POSSESSIONS"].to_numpy()
    poss_total = df["TOTAL_POSSESSIONS"].to_numpy()
    duration = df["GAME_DURATION_SECONDS"].to_numpy()

    _has_ft = "AWAY_FREE_THROW_SCORED" in df.columns and "AWAY_FREE_THROW_ATTEMPTED" in df.columns
    if _has_ft:
        ft_scored_home = df["HOME_FREE_THROW_SCORED"].to_numpy(dtype=int)
        ft_scored_away = df["AWAY_FREE_THROW_SCORED"].to_numpy(dtype=int)
        ft_att_home = np.maximum(df["HOME_FREE_THROW_ATTEMPTED"].to_numpy(float), 1e-3)
        ft_att_away = np.maximum(df["AWAY_FREE_THROW_ATTEMPTED"].to_numpy(float), 1e-3)

    with pm.Model() as model:
        emp_ppp = (y_home.sum() + y_away.sum()) / max(poss_total.sum(), 1.0)
        beta0 = pm.Normal("beta0", mu=float(np.log(max(emp_ppp, 1e-6))), sigma=0.35)

        if cfg.defence_mode not in ("off", "on", "corr"):
            raise ValueError("defence_mode must be one of: off, on, corr")
        if cfg.dispersion_prior not in ("original", "weak"):
            raise ValueError("dispersion_prior must be one of: original, weak")

        def dispersion(name, original_sigma):
            if cfg.dispersion_prior == "weak" and name == "pts":
                log_alpha = pm.Normal(f"log_alpha_{name}", mu=float(np.log(100.0)), sigma=1.5)
                return pm.Deterministic(f"alpha_{name}", pm.math.exp(log_alpha))
            return pm.HalfNormal(f"alpha_{name}", sigma=original_sigma)

        if cfg.defence_mode != "corr":
            sigma_player = pm.HalfNormal("sigma_player", sigma=0.35)
        sigma_team = pm.HalfNormal("sigma_team", sigma=0.25)

        # NON-CENTRED for player. The centred form
        #     player_raw ~ Normal(0, sigma_player)
        # funnels when the per-group signal is weak, and adding chemistry,
        # session and stream to this leg made it weak: sigma_player came back
        # r_hat 1.80 with ESS 5.9 (six independent samples across four chains).
        # The pace-leg equivalent has the identical form and converged fine
        # (ESS 9,528), so it is the competition on THIS leg that broke it, not
        # the parameterisation alone -- but non-centring is the standard remedy
        # and costs nothing. Mathematically identical model, better geometry.
        if cfg.defence_mode == "corr":
            # Bivariate non-centred: z @ L^T with L the Cholesky factor of the
            # 2x2 (attack, defence) covariance. eta=2 mildly favours small rho.
            # sd priors match off/on mode (0.35 attack, 0.25 defence).
            chol_od, corr_od, sd_od = pm.LKJCholeskyCov(
                "player_od_chol", n=2, eta=2.0,
                sd_dist=pm.HalfNormal.dist(sigma=np.array([0.35, 0.25]), shape=2),
                compute_corr=True)
            sigma_player = pm.Deterministic("sigma_player", sd_od[0])
            pm.Deterministic("sigma_def_player", sd_od[1])
            rho_od = pm.Deterministic("rho_player_od", corr_od[0, 1])
            # spread of QUALITY (off - def) and OPENNESS (off + def)
            pm.Deterministic("sigma_quality", pm.math.sqrt(
                sd_od[0] ** 2 + sd_od[1] ** 2 - 2 * rho_od * sd_od[0] * sd_od[1]))
            pm.Deterministic("sigma_openness", pm.math.sqrt(
                sd_od[0] ** 2 + sd_od[1] ** 2 + 2 * rho_od * sd_od[0] * sd_od[1]))
            player_od_z = pm.Normal("player_od_z", 0.0, 1.0, shape=(n_players, 2))
            player_od = pm.math.dot(player_od_z, chol_od.T)
            player_raw = pm.Deterministic("player_raw", player_od[:, 0])
            def_player_raw = player_od[:, 1]
        else:
            player_z = pm.Normal("player_z", mu=0.0, sigma=1.0, shape=n_players)
            player_raw = pm.Deterministic("player_raw", sigma_player * player_z)
        # team converged cleanly (r_hat 1.002, ESS 3,465) -- left as-is.
        team_raw = pm.Normal("team_raw", mu=0.0, sigma=sigma_team, shape=n_teams)

        player = pm.Deterministic("player", center(player_raw))
        team = pm.Deterministic("team", center(team_raw))

        if cfg.hca_mode == "off":
            hca_term = 0.0
        elif cfg.hca_mode == "global":
            hca_global = pm.Normal("hca_global", mu=0.0, sigma=0.15)
            hca_term = hca_global
        elif cfg.hca_mode == "team":
            sigma_hca = pm.HalfNormal("sigma_hca", sigma=0.15)
            hca_team_raw = pm.Normal("hca_team_raw", mu=0.0, sigma=sigma_hca, shape=n_teams)
            hca_team = pm.Deterministic("hca_team", center(hca_team_raw))
            hca_term = hca_team[home_t_idx]
        else:
            raise ValueError("hca_mode must be one of: global, team, off")

        alpha_pts = dispersion("pts", 3.0)

        # ---- v2 TERMS: chemistry, match, stream, session -------------------
        # Chemistry on the PPP leg = efficiency compounding.
        if cfg.chem_mode == "off" or not cfg.chem_on_ppp:
            chem_home = 0.0
            chem_away = 0.0
        elif cfg.chem_mode == "pair":
            sigma_chem = pm.HalfNormal("sigma_chem", sigma=0.10)
            if cfg.chem_tail == "studentt":
                chem_raw = pm.StudentT("chem_raw", nu=float(cfg.chem_tail_nu),
                                       mu=0.0, sigma=1.0,
                                       shape=(n_players, n_teams))
            else:
                chem_raw = pm.Normal("chem_raw", 0.0, 1.0,
                                     shape=(n_players, n_teams))
            chem = pm.Deterministic("chem", sigma_chem * chem_raw)
            chem_home = chem[home_p_idx, home_t_idx]
            chem_away = chem[away_p_idx, away_t_idx]
        elif cfg.chem_mode == "factor":
            K = int(cfg.chem_rank)
            sigma_chem = pm.HalfNormal("sigma_chem", sigma=0.10)
            u_raw = pm.Normal("player_attr_raw", 0.0, 1.0, shape=(n_players, K))
            v_raw = pm.Normal("team_attr_raw", 0.0, 1.0, shape=(n_teams, K))
            # Centre both, or the product is confounded with the main effects.
            u = pm.Deterministic("player_attr", u_raw - pm.math.mean(u_raw, axis=0))
            v = pm.Deterministic("team_attr", v_raw - pm.math.mean(v_raw, axis=0))
            # Weakly identified elementwise (u -> cu, v -> v/c leaves the
            # product alone). Read the PRODUCT and sigma_chem, never u or v.
            chem_home = sigma_chem * (u[home_p_idx] * v[home_t_idx]).sum(axis=-1)
            chem_away = sigma_chem * (u[away_p_idx] * v[away_t_idx]).sum(axis=-1)
        else:
            raise ValueError("chem_mode must be one of: off, pair, factor")

        if cfg.match_effect:
            sigma_match = pm.HalfNormal("sigma_match", sigma=0.15)
            match_raw = pm.Normal("match_raw", 0.0, 1.0, shape=n)
            match_eff = pm.Deterministic("match_effect", sigma_match * match_raw)
        else:
            match_eff = 0.0

        if use_stream:
            # CENTRED. The uncentred version was nearly collinear with beta0 --
            # a common shift on every match in a stream-week is indistinguishable
            # from a higher global baseline, so the sampler traded the two off:
            #   beta0        r_hat 1.07, ESS 64   (sd blew out 15x, 0.0046 -> 0.0699)
            #   sigma_stream r_hat 1.10, ESS 29   (came back 0.067, the LARGEST
            #                                      component in the model)
            # plus divergences. Subtracting the mean lets the term redistribute
            # between streams but not shift the level, which is all a nuisance
            # control should ever do.
            # Prior also tightened 0.10 -> 0.05: the posterior interval reached
            # 0.149, which is implausibly large for a stream nuisance term.
            sigma_stream = pm.HalfNormal("sigma_stream", sigma=0.05)
            stream_raw = pm.Normal("stream_raw", 0.0, 1.0, shape=n_streams)
            stream_c = stream_raw - pm.math.mean(stream_raw)
            stream_eff = pm.Deterministic("stream_effect", sigma_stream * stream_c)
            stream_term = stream_eff[stream_idx]
        else:
            stream_term = 0.0

        if cfg.session_effect:
            # Also centred. sigma_session came back r_hat 1.59 with ESS 7 --
            # it was funnelling alongside the stream/intercept trade-off. With
            # tens of thousands of levels the mean is well determined, so
            # centring costs nothing and removes the level ambiguity.
            # Prior tightened 0.10 -> 0.06. The previous run hit the 1023-step
            # max tree depth on a large share of iterations with the step size
            # pinned near 1e-2 -- residual funnel. With one parameter per
            # (player, calendar day) and only a handful of observations each,
            # sigma_session is the block most able to funnel even non-centred,
            # and 0.10 is loose given the pre-fit read of ~0.054.
            sigma_session = pm.HalfNormal("sigma_session", sigma=0.06)
            sess_raw = pm.Normal("session_raw", 0.0, 1.0, shape=n_sessions)
            sess_c = sess_raw - pm.math.mean(sess_raw)
            session_eff = pm.Deterministic("session_effect", sigma_session * sess_c)
            sess_home = session_eff[sess_home_idx]
            sess_away = session_eff[sess_away_idx]
        else:
            sess_home = 0.0
            sess_away = 0.0

        if cfg.defence_mode != "off":
            if cfg.defence_mode == "on":
                sigma_def_player = pm.HalfNormal("sigma_def_player", sigma=0.25)
                def_player_z = pm.Normal("def_player_z", 0.0, 1.0, shape=n_players)
                def_player_raw = sigma_def_player * def_player_z
            def_player = pm.Deterministic("def_player", center(def_player_raw))
            sigma_def_team = pm.HalfNormal("sigma_def_team", sigma=0.15)
            def_team_z = pm.Normal("def_team_z", 0.0, 1.0, shape=n_teams)
            def_team = pm.Deterministic(
                "def_team", center(sigma_def_team * def_team_z))
            # the home side scores against the AWAY defence, and vice versa
            def_home = def_player[away_p_idx] + def_team[away_t_idx]
            def_away = def_player[home_p_idx] + def_team[home_t_idx]
        else:
            def_home = 0.0
            def_away = 0.0

        eta_home = (beta0 + player[home_p_idx] + team[home_t_idx] + hca_term + def_home
                    + chem_home + match_eff + stream_term + sess_home)
        eta_away = (beta0 + player[away_p_idx] + team[away_t_idx] + def_away
                    + chem_away + match_eff + stream_term + sess_away)

        mu_home = pm.math.exp(pm.math.log(poss_home) + eta_home)
        mu_away = pm.math.exp(pm.math.log(poss_away) + eta_away)

        pm.NegativeBinomial("HOME_POINTS", mu=mu_home, alpha=alpha_pts, observed=y_home)
        pm.NegativeBinomial("AWAY_POINTS", mu=mu_away, alpha=alpha_pts, observed=y_away)

        emp_pace = poss_total.sum() / max(duration.sum(), 1.0)
        pace_beta0 = pm.Normal("pace_beta0", mu=float(np.log(max(emp_pace, 1e-9))), sigma=0.25)

        sigma_player_pace = pm.HalfNormal("sigma_player_pace", sigma=0.20)
        sigma_team_pace = pm.HalfNormal("sigma_team_pace", sigma=0.15)

        player_pace_raw = pm.Normal("player_pace_raw", mu=0.0, sigma=sigma_player_pace, shape=n_players)
        team_pace_raw = pm.Normal("team_pace_raw", mu=0.0, sigma=sigma_team_pace, shape=n_teams)

        player_pace = pm.Deterministic("player_pace", center(player_pace_raw))
        team_pace = pm.Deterministic("team_pace", center(team_pace_raw))

        alpha_poss = dispersion("poss", 5.0)

        # ---- v2 TERMS on the PACE leg --------------------------------------
        # Chemistry here asks a DIFFERENT question from the ppp leg: does a
        # gamer on a suited team also play FASTER (more transition, more shots)
        # rather than only more efficiently? The direct variant folds rate and
        # efficiency into one number and cannot separate them; this one can, and
        # a pace-side finding would matter more to the simulator, which projects
        # remaining possessions from pace.
        #
        # NOTE the pace leg is modelled at MATCH level (one TOTAL_POSSESSIONS
        # observation), so every term here enters as the SUM over both sides,
        # matching how player_pace and team_pace are already used.
        if cfg.chem_mode != "off" and cfg.chem_on_pace:
            # Prior loosened 0.08 -> 0.20. The posterior came back at 0.0154
            # (sd 0.0008) so 0.08 was not binding on the SCALE -- but the
            # per-pair shrinkage it implies is far too strong for prediction
            # (see chem_tail in FitConfig). Loosening costs nothing when the
            # data are informative and removes the prior as a suspect.
            sigma_chem_pace = pm.HalfNormal("sigma_chem_pace", sigma=0.20)
            if cfg.chem_mode == "pair":
                if cfg.chem_tail == "studentt":
                    chem_pace_raw = pm.StudentT("chem_pace_raw",
                                                nu=float(cfg.chem_tail_nu),
                                                mu=0.0, sigma=1.0,
                                                shape=(n_players, n_teams))
                else:
                    chem_pace_raw = pm.Normal("chem_pace_raw", 0.0, 1.0,
                                              shape=(n_players, n_teams))
                chem_pace = pm.Deterministic("chem_pace",
                                             sigma_chem_pace * chem_pace_raw)
                chem_pace_term = (chem_pace[home_p_idx, home_t_idx]
                                  + chem_pace[away_p_idx, away_t_idx])
            else:
                K = int(cfg.chem_rank)
                up_raw = pm.Normal("player_attr_pace_raw", 0.0, 1.0,
                                   shape=(n_players, K))
                vp_raw = pm.Normal("team_attr_pace_raw", 0.0, 1.0,
                                   shape=(n_teams, K))
                up = pm.Deterministic("player_attr_pace",
                                      up_raw - pm.math.mean(up_raw, axis=0))
                vp = pm.Deterministic("team_attr_pace",
                                      vp_raw - pm.math.mean(vp_raw, axis=0))
                chem_pace_term = sigma_chem_pace * (
                    (up[home_p_idx] * vp[home_t_idx]).sum(axis=-1)
                    + (up[away_p_idx] * vp[away_t_idx]).sum(axis=-1))
        else:
            chem_pace_term = 0.0

        # Stream on pace: streams may run at different tempo (settings, build).
        if use_stream and cfg.stream_on_pace:
            # Centred for the same reason -- it was collinear with pace_beta0,
            # whose sd went 0.0013 -> 0.0447 (34x) in the uncentred fit.
            sigma_stream_pace = pm.HalfNormal("sigma_stream_pace", sigma=0.05)
            stream_pace_raw = pm.Normal("stream_pace_raw", 0.0, 1.0,
                                        shape=n_streams)
            stream_pace_c = stream_pace_raw - pm.math.mean(stream_pace_raw)
            stream_pace_eff = pm.Deterministic("stream_pace_effect",
                                               sigma_stream_pace * stream_pace_c)
            stream_pace_term = stream_pace_eff[stream_idx]
        else:
            stream_pace_term = 0.0

        # Session on pace: a gamer in a poor session may also play differently.
        # Summed over both sides, as with the other pace terms.
        if cfg.session_effect and cfg.session_on_pace:
            sigma_session_pace = pm.HalfNormal("sigma_session_pace", sigma=0.08)
            sess_pace_raw = pm.Normal("session_pace_raw", 0.0, 1.0,
                                      shape=n_sessions)
            sess_pace_c = sess_pace_raw - pm.math.mean(sess_pace_raw)
            session_pace_eff = pm.Deterministic("session_pace_effect",
                                                sigma_session_pace * sess_pace_c)
            session_pace_term = (session_pace_eff[sess_home_idx]
                                 + session_pace_eff[sess_away_idx])
        else:
            session_pace_term = 0.0

        # NOTE: alpha_poss currently absorbs whatever structure these terms
        # describe. Expect it to RISE once they are fitted -- that is the
        # desired direction, and the same relationship holds for alpha_pts on
        # the points leg.
        eta_pace = (
                pace_beta0
                + player_pace[home_p_idx] + player_pace[away_p_idx]
                + team_pace[home_t_idx] + team_pace[away_t_idx]
                + chem_pace_term + stream_pace_term + session_pace_term
        )

        mu_poss = pm.math.exp(pm.math.log(duration) + eta_pace)
        pm.NegativeBinomial("TOTAL_POSSESSIONS", mu=mu_poss, alpha=alpha_poss, observed=poss_total.astype(int))

        if _has_ft:
            ft_beta0 = pm.Normal("ft_beta0", mu=0.0, sigma=0.5)

            sigma_player_ft = pm.HalfNormal("sigma_player_ft", sigma=0.25)
            sigma_team_ft = pm.HalfNormal("sigma_team_ft", sigma=0.15)

            player_ft_raw = pm.Normal("player_ft_raw", mu=0.0, sigma=sigma_player_ft, shape=n_players)
            team_ft_raw = pm.Normal("team_ft_raw", mu=0.0, sigma=sigma_team_ft, shape=n_teams)

            player_ft = pm.Deterministic("player_ft", center(player_ft_raw))
            team_ft = pm.Deterministic("team_ft", center(team_ft_raw))

            logit_ft_home = ft_beta0 + player_ft[home_p_idx] + team_ft[home_t_idx]
            logit_ft_away = ft_beta0 + player_ft[away_p_idx] + team_ft[away_t_idx]

            p_ft_home = pm.math.sigmoid(logit_ft_home)
            p_ft_away = pm.math.sigmoid(logit_ft_away)

            pm.Binomial("HOME_FREE_THROW_SCORED", n=ft_att_home.astype(int), p=p_ft_home,
                        observed=ft_scored_home)
            pm.Binomial("AWAY_FREE_THROW_SCORED", n=ft_att_away.astype(int), p=p_ft_away,
                        observed=ft_scored_away)

        # Call the JAX sampler DIRECTLY. Passing chain_method through
        # pm.sample does not work: pm.sample puts unknown kwargs into **kwargs,
        # and the JAX branch of _sample_external_nuts does not forward them to
        # sample_jax_nuts -- it passes an explicit argument list. So
        # chain_method is silently dropped and the default "parallel" is used,
        # which on a single GPU runs the chains ONE AT A TIME (the "not enough
        # devices" warning). Routing via nuts_sampler_kwargs instead sends it
        # into NUTS.__init__, which rejects it. Direct is the only route.
        #
        # "vectorized" batches all chains into one computation on the single
        # device -- the whole point of the GPU here. If it runs out of memory,
        # halve cfg.chains and run twice rather than reverting to parallel.
        import pymc.sampling.jax as pmjax
        idata = pmjax.sample_jax_nuts(
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            target_accept=cfg.target_accept,
            random_seed=cfg.random_seed,
            model=model,
            nuts_sampler="numpyro",
            chain_method="vectorized",
            progressbar=True,
        )

    return idata


def export_outputs(out_dir: Path, df: pd.DataFrame, idata, cfg: FitConfig) -> None:
    import arviz as az
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- posterior.nc: drop the NUISANCE blocks before writing --------------
    # The last run produced a 3.5 GB posterior.nc and the export died on it,
    # taking the whole two-hour fit with it.
    #
    # The size is the per-level draws: session is one parameter per
    # (player, calendar day) -- tens of thousands -- and chem is n_players x
    # n_teams, all stored for every draw of every chain. None of that is ever
    # inspected individually: the simulator reads params_summary.json, and the
    # experiment comparison reads scalar_summary.csv. posterior.nc exists for
    # post-hoc diagnostics on the parameters we actually interrogate.
    #
    # So keep everything EXCEPT the per-level nuisance draws. The sigmas stay,
    # which is what the diagnostics are about.
    _DROP = {
        "session_raw", "session_effect", "session_z",
        "session_pace_raw", "session_pace_effect",
        "match_raw", "match_effect",
        "stream_raw", "stream_pace_raw",
        "chem_raw", "chem_pace_raw", "chem", "chem_pace",   # means are in the JSON
        "player_z", "player_raw", "team_raw", "def_player_z", "def_team_z", "player_od_z",
        "player_pace_raw", "team_pace_raw", "player_ft_raw", "team_ft_raw",
    }
    try:
        import xarray as xr
        _keep = [v for v in idata.posterior.data_vars if v not in _DROP]

        def _ds(group):   # DataTree node (arviz 1.x) or Dataset (arviz 0.x)
            return group.to_dataset() if hasattr(group, "to_dataset") else group

        # Keep the GROUPS: az.from_netcdf needs "posterior", and sample_stats
        # carries divergences/tree depth and is small -- keep it.
        _groups = {"posterior": _ds(idata.posterior)[_keep]}
        if "sample_stats" in getattr(idata, "children", idata):
            _groups["sample_stats"] = _ds(idata.sample_stats)
        if hasattr(xr, "DataTree"):
            xr.DataTree.from_dict(_groups).to_netcdf(out_dir / "posterior.nc")
        else:
            az.InferenceData(**_groups).to_netcdf(str(out_dir / "posterior.nc"))
        print(f"    posterior.nc: kept {len(_keep)} of "
              f"{len(idata.posterior.data_vars)} variables "
              f"(dropped per-level nuisance draws)")
    except Exception as e:
        # NEVER let this take the rest of the export down. params_summary.json,
        # the maps and scalar_summary.csv are the load-bearing outputs.
        print(f"    WARNING: posterior.nc not written ({type(e).__name__}: {e})")
        try:
            (out_dir / "posterior.nc").unlink(missing_ok=True)   # no half-written file
        except OSError:
            pass
        print("    The other exports are unaffected.")

    player_keys = pd.Index(sorted(pd.concat([df["PLAYER_1_HANDLE"], df["PLAYER_2_HANDLE"]]).astype(str).unique()))
    team_keys = pd.Index(sorted(pd.concat([df["PLAYER_1_TEAM"], df["PLAYER_2_TEAM"]]).astype(str).unique()))

    pd.DataFrame({"key": player_keys, "index": np.arange(len(player_keys), dtype=int)}).to_csv(
        out_dir / "player_map.csv", index=False)
    pd.DataFrame({"key": team_keys, "index": np.arange(len(team_keys), dtype=int)}).to_csv(
        out_dir / "team_map.csv", index=False)

    def post_mean(name: str):
        return np.asarray(idata.posterior[name].mean(dim=("chain", "draw")))

    def post_sd(name: str):
        return np.asarray(idata.posterior[name].std(dim=("chain", "draw")))

    summary: Dict[str, Any] = {
        "beta0_mean": float(post_mean("beta0")),
        "beta0_sd": float(post_sd("beta0")),
        "alpha_pts_mean": float(post_mean("alpha_pts")),
        "alpha_pts_sd": float(post_sd("alpha_pts")),
        "player_mean": post_mean("player").tolist(),
        "player_sd": post_sd("player").tolist(),
        "team_mean": post_mean("team").tolist(),
        "team_sd": post_sd("team").tolist(),
    }

    if "hca_global" in idata.posterior:
        summary["hca_global_mean"] = float(post_mean("hca_global"))
        summary["hca_global_sd"] = float(post_sd("hca_global"))

    if "hca_team" in idata.posterior:
        summary["hca_team_mean"] = post_mean("hca_team").tolist()
        summary["hca_team_sd"] = post_sd("hca_team").tolist()

    if "ft_beta0" in idata.posterior:
        summary["ft_beta0_mean"] = float(post_mean("ft_beta0"))
        summary["ft_beta0_sd"] = float(post_sd("ft_beta0"))
        summary["player_ft_mean"] = post_mean("player_ft").tolist()
        summary["player_ft_sd"] = post_sd("player_ft").tolist()
        summary["team_ft_mean"] = post_mean("team_ft").tolist()
        summary["team_ft_sd"] = post_sd("team_ft").tolist()

        # --- Empirical FT ATTEMPTS per possession (the missing VOLUME) ----------
        # beta0 is fit on FG-only points (FT scored is subtracted in
        # load_match_features), and the FT submodel above gives only the MAKE rate
        # sigmoid(ft_beta0 + player_ft + team_ft). The downstream sim needs
        # attempts to turn that into points:
        #     FT points = FT_att_per_poss * possessions * make_rate
        # Export the empirical attempts/possession, aligned to the SAME index order
        # as player_mean / team_mean, so the sim can add a per-player FT leg to the
        # decomp total. player-level preferred; team-level fallback; global last.
        if all(c in df.columns for c in ("AWAY_FREE_THROW_ATTEMPTED", "HOME_FREE_THROW_ATTEMPTED",
                                         "AWAY_POSSESSIONS", "HOME_POSSESSIONS")):
            _aw = df[["PLAYER_1_TEAM", "PLAYER_1_HANDLE", "AWAY_FREE_THROW_ATTEMPTED", "AWAY_POSSESSIONS"]].rename(
                columns={"PLAYER_1_TEAM": "team", "PLAYER_1_HANDLE": "player",
                         "AWAY_FREE_THROW_ATTEMPTED": "ft", "AWAY_POSSESSIONS": "ps"})
            _hm = df[["PLAYER_2_TEAM", "PLAYER_2_HANDLE", "HOME_FREE_THROW_ATTEMPTED", "HOME_POSSESSIONS"]].rename(
                columns={"PLAYER_2_TEAM": "team", "PLAYER_2_HANDLE": "player",
                         "HOME_FREE_THROW_ATTEMPTED": "ft", "HOME_POSSESSIONS": "ps"})
            _lg = pd.concat([_aw, _hm], ignore_index=True)
            _lg["ft"] = pd.to_numeric(_lg["ft"], errors="coerce").fillna(0.0).clip(lower=0)
            _lg["ps"] = pd.to_numeric(_lg["ps"], errors="coerce").fillna(0.0).clip(lower=0)
            _global = float(_lg["ft"].sum() / max(_lg["ps"].sum(), 1.0))
            _tg = _lg.groupby("team").agg(ft=("ft", "sum"), ps=("ps", "sum"))
            _team_rate = (_tg["ft"] / _tg["ps"].clip(lower=1)).reindex(
                [str(k) for k in team_keys]).fillna(_global)
            _pg = _lg.groupby("player").agg(ft=("ft", "sum"), ps=("ps", "sum"))
            _player_rate = (_pg["ft"] / _pg["ps"].clip(lower=1)).reindex(
                [str(k) for k in player_keys]).fillna(_global)
            summary["ft_att_per_poss_global"] = _global
            summary["team_ft_att_per_poss"] = _team_rate.tolist()
            summary["player_ft_att_per_poss"] = _player_rate.tolist()

    if "def_player" in idata.posterior:
        summary["def_player_mean"] = post_mean("def_player").tolist()
        summary["def_player_sd"] = post_sd("def_player").tolist()
        summary["def_team_mean"] = post_mean("def_team").tolist()
        summary["def_team_sd"] = post_sd("def_team").tolist()
        summary["sigma_def_player_mean"] = float(post_mean("sigma_def_player"))
        summary["sigma_def_player_sd"] = float(post_sd("sigma_def_player"))
        summary["sigma_def_team_mean"] = float(post_mean("sigma_def_team"))
        summary["sigma_def_team_sd"] = float(post_sd("sigma_def_team"))
    if "rho_player_od" in idata.posterior:
        summary["rho_player_od_mean"] = float(post_mean("rho_player_od"))
        summary["rho_player_od_sd"] = float(post_sd("rho_player_od"))
        summary["sigma_quality_mean"] = float(post_mean("sigma_quality"))
        summary["sigma_openness_mean"] = float(post_mean("sigma_openness"))
    summary["defence_mode"] = cfg.defence_mode
    summary["dispersion_prior"] = cfg.dispersion_prior

    summary["pace_beta0_mean"] = float(post_mean("pace_beta0"))
    summary["pace_beta0_sd"] = float(post_sd("pace_beta0"))
    summary["alpha_poss_mean"] = float(post_mean("alpha_poss"))
    summary["alpha_poss_sd"] = float(post_sd("alpha_poss"))
    summary["player_pace_mean"] = post_mean("player_pace").tolist()
    summary["player_pace_sd"] = post_sd("player_pace").tolist()
    summary["team_pace_mean"] = post_mean("team_pace").tolist()
    summary["team_pace_sd"] = post_sd("team_pace").tolist()

    # ---- v2 exports ------------------------------------------------------
    # HOW THE SIMULATOR SHOULD USE EACH OF THESE:
    #
    #   chem / chem_pace      MEAN effects. Add chem[p,t] to that side's eta
    #                         (or sigma_chem * dot(player_attr[p], team_attr[t])
    #                         in factor mode). Flows through the existing eta
    #                         conversion with no structural change. An unseen
    #                         pair is 0 in pair mode -- the correctly shrunk
    #                         default; factor mode generalises instead.
    #
    #   sigma_match           COMMON-MODE variance. ONE draw per simulation
    #   sigma_stream          applied to BOTH etas. Widens the TOTAL, cancels
    #                         from the MARGIN. Compare sigma_match against the
    #                         common-mode leg of trader error, measured at
    #                         sd 0.073 on 59,348 fixtures -- if they agree, two
    #                         independent routes have found the same thing.
    #
    #   sigma_session         SIDE-SPECIFIC variance. INDEPENDENT draw per side.
    #                         Widens the MARGIN as well as the total. This is
    #                         the leg the trader-error decomposition put at
    #                         sd 0.128 and which the simulator currently has
    #                         nothing for -- do NOT draw it as a shared term.
    summary["chem_mode"] = cfg.chem_mode
    summary["chem_tail"] = cfg.chem_tail
    summary["chem_tail_nu"] = float(cfg.chem_tail_nu)
    if cfg.chem_mode != "off":
        if cfg.chem_on_ppp:
            summary["sigma_chem_mean"] = float(post_mean("sigma_chem"))
            summary["sigma_chem_sd"] = float(post_sd("sigma_chem"))
            if cfg.chem_mode == "pair":
                summary["chem_pair_mean"] = post_mean("chem").tolist()
            else:
                summary["chem_rank"] = int(cfg.chem_rank)
                summary["player_attr_mean"] = post_mean("player_attr").tolist()
                summary["team_attr_mean"] = post_mean("team_attr").tolist()
        if cfg.chem_on_pace:
            summary["sigma_chem_pace_mean"] = float(post_mean("sigma_chem_pace"))
            summary["sigma_chem_pace_sd"] = float(post_sd("sigma_chem_pace"))
            if cfg.chem_mode == "pair":
                summary["chem_pace_pair_mean"] = post_mean("chem_pace").tolist()
            else:
                summary["player_attr_pace_mean"] = post_mean("player_attr_pace").tolist()
                summary["team_attr_pace_mean"] = post_mean("team_attr_pace").tolist()

    summary["match_effect_fitted"] = bool(cfg.match_effect)
    if cfg.match_effect:
        summary["sigma_match_mean"] = float(post_mean("sigma_match"))
        summary["sigma_match_sd"] = float(post_sd("sigma_match"))

    summary["stream_effect_fitted"] = "sigma_stream" in idata.posterior
    if "sigma_stream" in idata.posterior:
        summary["sigma_stream_mean"] = float(post_mean("sigma_stream"))
        summary["sigma_stream_sd"] = float(post_sd("sigma_stream"))
        if "sigma_stream_pace" in idata.posterior:
            summary["sigma_stream_pace_mean"] = float(post_mean("sigma_stream_pace"))

    summary["session_effect_fitted"] = bool(cfg.session_effect)
    if cfg.session_effect:
        summary["sigma_session_mean"] = float(post_mean("sigma_session"))
        summary["sigma_session_sd"] = float(post_sd("sigma_session"))
        if cfg.session_on_pace:
            summary["sigma_session_pace_mean"] = float(post_mean("sigma_session_pace"))

    with open(out_dir / "params_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    az.summary(idata, var_names=[v for v in ["beta0", "alpha_pts", "pace_beta0", "alpha_poss", "hca_global",
                                       "sigma_chem", "sigma_chem_pace", "sigma_match",
                                       "sigma_stream", "sigma_stream_pace",
                                       "sigma_session", "sigma_session_pace",
                                       "sigma_player", "sigma_team",
                                       "sigma_def_player", "sigma_def_team",
                                       "rho_player_od", "sigma_quality", "sigma_openness",
                                       "sigma_player_pace", "sigma_team_pace",
                                       "ft_beta0", "sigma_player_ft", "sigma_team_ft"] if
                                 v in idata.posterior]).to_csv(
        out_dir / "scalar_summary.csv"
    )


DT_FMT = "%d-%m-%Y %H:%M:%S"

##############################################################################
#Start of script
##############################################################################
print("=" * 60)
print("  decomp_fouls — run summary")
print("=" * 60)

print("  Start FitConfig      - " + datetime.now(tz=timezone.utc).strftime(DT_FMT))
# ---------------------------------------------------------------------------
# v2 SWITCHES. All default OFF -- FitConfig() alone reproduces production.
#
# Fit ONE change at a time so each is attributable. Suggested order:
#
#   1. FitConfig(session_effect=True, match_effect=True)
#        The two variance components. SESSION is side-specific (widens margin
#        AND total); MATCH is common-mode (total only). Fit together -- they
#        compete for the same variation and separating them is the point.
#        Watch alpha_pts: it should RISE as real structure leaves it.
#
#   2. FitConfig(session_effect=True, match_effect=True,
#                stream_effect=True, chem_mode="pair")
#        Chemistry with its control. Do NOT fit chem without stream: streams
#        have distinct team pools and ~9% of the measured chemistry is the rota.
#
#   3. As (2) plus chem_on_pace=True
#        Does a suited team make a gamer play FASTER as well as better? Only
#        this variant can answer it.
#
#   4. FitConfig(..., chem_mode="factor", chem_rank=1)
#        Only if "pair" confirms chemistry. Generalises to unseen gamer-team
#        pairs, which the simulator needs.
#
# Measured on 119,486 match-sides BEFORE fitting, for sanity-checking posteriors:
#   chemistry sd ~0.0224 stream-controlled (ratio 1.55 over sampling noise)
#   player sd ~0.072, team sd ~0.041  -> chemistry is about half the team effect
#   session: lag-1 corr 0.159 vs lag-2 0.069 -- a cliff, not a decay
#   NO recency weighting: there is no short-horizon form to weight toward
#
# Cost warning: session adds one parameter per (player, day) -- likely
# 20-40k parameters, more than the chemistry block. Non-centred, so it should
# sample, but smoke-test on a short window before the full history.
# ---------------------------------------------------------------------------
# RE-FIT after the diagnostics on the first chemistry run:
#   sigma_chem     r_hat 1.001, ESS 2,536  -> CONVERGED. 0.0240 +/- 0.0011,
#                  matching the pre-fit screen (0.0224). Chemistry is banked.
#   sigma_player   r_hat 1.80,  ESS 5.9    \
#   sigma_session  r_hat 1.59,  ESS 7.0     > broken, plus divergences
#   sigma_stream   r_hat 1.10,  ESS 29     /
#   beta0          r_hat 1.07,  ESS 64      (sd 0.0046 -> 0.0699)
#
# Changes in this version: stream and session effects CENTRED (they were
# collinear with the intercepts), stream prior tightened 0.10 -> 0.05, player
# block non-centred, target_accept 0.9 -> 0.95, tune 1500 -> 2500.
#
# Expect sigma_chem to be essentially unchanged -- it is a within-player,
# across-teams contrast and so is orthogonal to the level ambiguity that broke
# everything else.
# Pace chemistry VALIDATED out of fold: +4.29% RMSE on held-out matches
# (a synthetic null scores -0.06%, so this is a real effect, not sample fitting).
# 8 players appear at BOTH extremes -- JACKAL is +4.5% pace with New York and
# -2.1% with Oklahoma City -- which no main effect can produce.
#
# This re-fit addresses SHRINKAGE, not signal. Check afterwards:
#   sigma_chem_pace should RISE from 0.0154
#   the IN-SAMPLE gain in pace_chem_validate.py should close toward 4.29%
#   the OOF gain should hold -- if it FALLS, the tail is now too loose
# SMOKE TEST: chemistry off, short window, light sampling. Settings come from
# run.ps1 (or DECOMP_* env vars) and were validated at start-up.
cfg = FitConfig(
    defence_mode=DEFENCE_MODE,
    dispersion_prior=DISPERSION,
    stream_fallback=STREAM_FALLBACK,
    chem_mode="off",
    chem_on_ppp=False,
    chem_on_pace=False,
    match_effect=False,
    stream_effect=True,
    session_effect=False,
    chains=CHAINS,
    draws=DRAWS,
    tune=TUNE,
)

print("  Start data frame     - " + datetime.now(tz=timezone.utc).strftime(DT_FMT))
# asof: "latest" = the most recent usable match in the table, or a date like "2026-10-04"
df = load_match_features(asof=ASOF, lookback_days=DAYS)
_w_end = df["SCHEDULED_START_TIME_UTC"].max()
print(f"  Window: {df['SCHEDULED_START_TIME_UTC'].min()} -> {_w_end}  ({len(df)} matches)")
# one folder per mode AND window, so runs over different windows don't overwrite
OUT_DIR = Path(os.environ.get("DECOMP_OUT") or
               f"decomp_out_{DEFENCE_MODE}_{_w_end:%Y%m%d}_{DAYS}d"
               + ("" if DISPERSION == "original" else f"_{DISPERSION}")
               + ("" if STREAM_FALLBACK == "pool" else f"_{STREAM_FALLBACK}"))
print("  Start players        - " + datetime.now(tz=timezone.utc).strftime(DT_FMT))
n_players = len(pd.concat([df["PLAYER_1_HANDLE"], df["PLAYER_2_HANDLE"]]).unique())

print("  Start teams          - " + datetime.now(tz=timezone.utc).strftime(DT_FMT))
n_teams = len(pd.concat([df["PLAYER_1_TEAM"], df["PLAYER_2_TEAM"]]).unique())

print("  Start free throw     - " + datetime.now(tz=timezone.utc).strftime(DT_FMT))
has_ft = all(c in df.columns for c in ["AWAY_FREE_THROW_SCORED", "HOME_FREE_THROW_SCORED", "AWAY_FREE_THROW_ATTEMPTED", "HOME_FREE_THROW_ATTEMPTED"])

#this bit take the longest
print("  Start idata          - " + datetime.now(tz=timezone.utc).strftime(DT_FMT))
idata = fit_model(df, cfg)

print("  Start export         - " + datetime.now(tz=timezone.utc).strftime(DT_FMT))
export_outputs(OUT_DIR, df, idata, cfg)

print("  End process          - " + datetime.now(tz=timezone.utc).strftime(DT_FMT))


import arviz as az
_row: Dict[str, Any] = {}

# ---- sampler health: judge convergence on EVERYTHING, not a few scalars ----
_hyper = [v for v in ["beta0", "alpha_pts", "sigma_player", "sigma_team",
          "sigma_def_player", "sigma_def_team", "rho_player_od",
          "sigma_quality", "sigma_openness", "hca_global",
          "pace_beta0", "alpha_poss", "sigma_player_pace", "sigma_team_pace",
          "sigma_stream", "sigma_stream_pace",
          "ft_beta0", "sigma_player_ft", "sigma_team_ft"] if v in idata.posterior]
_tab = az.summary(idata, var_names=_hyper)
print(_tab)
try:
    _ss = idata.sample_stats
    _div = int(_ss["diverging"].sum())
    _deep = int((_ss["tree_depth"] >= 10).sum())   # numpyro max_tree_depth = 10
    _n = int(_ss["diverging"].size)
    print(f"  divergences: {_div} of {_n} draws;  max tree depth hit: {_deep}")
    _row.update(divergences=_div, max_depth_hits=_deep)
except Exception as e:
    print(f"  (sample_stats unavailable: {type(e).__name__})")
_free = [v for v in idata.posterior.data_vars
         if v in ("player", "team", "def_player", "def_team", "player_pace", "team_pace",
                  "player_ft", "team_ft") or v in _hyper]
_full = az.summary(idata, var_names=_free)
_worst_r = _full["r_hat"].astype(float).idxmax()
_worst_e = _full["ess_bulk"].astype(float).idxmin()
print(f"  worst r_hat {float(_full.loc[_worst_r, 'r_hat']):.3f} ({_worst_r}); "
      f"lowest ESS_bulk {float(_full.loc[_worst_e, 'ess_bulk']):.0f} ({_worst_e})")
print("  rule of thumb before reading anything: every r_hat <= 1.01, ESS >= 400, "
      "0 divergences -- else re-run with more chains/draws (-Chains 4 -Draws 1000)")
_row.update(max_rhat=float(_full["r_hat"].astype(float).max()),
            min_ess_bulk=float(_full["ess_bulk"].astype(float).min()))

if cfg.defence_mode != "off":
    _pk = sorted(pd.concat([df["PLAYER_1_HANDLE"], df["PLAYER_2_HANDLE"]]).astype(str).unique())
    _sides = pd.concat([
        pd.DataFrame({"player": df["PLAYER_1_HANDLE"].astype(str), "row": df.index}),
        pd.DataFrame({"player": df["PLAYER_2_HANDLE"].astype(str), "row": df.index})])
    _off_draws = np.asarray(idata.posterior["player"]).reshape(-1, len(_pk))
    _def_draws = np.asarray(idata.posterior["def_player"]).reshape(-1, len(_pk))
    _pl = pd.DataFrame({"player": _pk,
                        "off": _off_draws.mean(0), "def": _def_draws.mean(0)})
    # quality moves the MARGIN, openness moves the TOTAL
    _pl["quality"] = _pl["off"] - _pl["def"]
    _pl["openness"] = _pl["off"] + _pl["def"]
    _pl["matches"] = _pl["player"].map(_sides["player"].value_counts())

    # Fouling check: does "defence" just pick up fouling (fewer FG chances
    # conceded, more FTs conceded)? Fouls committed per OPPONENT possession.
    if {"AWAY_FOULS_COMMITTED", "HOME_FOULS_COMMITTED"} <= set(df.columns):
        _f = pd.concat([
            pd.DataFrame({"player": df["PLAYER_1_HANDLE"].astype(str),
                          "fouls": pd.to_numeric(df["AWAY_FOULS_COMMITTED"], errors="coerce"),
                          "opp_poss": df["HOME_POSSESSIONS"]}),
            pd.DataFrame({"player": df["PLAYER_2_HANDLE"].astype(str),
                          "fouls": pd.to_numeric(df["HOME_FOULS_COMMITTED"], errors="coerce"),
                          "opp_poss": df["AWAY_POSSESSIONS"]})]).groupby("player").sum()
        _pl["fouls_per_opp_poss"] = _pl["player"].map(_f["fouls"] / _f["opp_poss"])
        _cf = _pl[["def", "fouls_per_opp_poss"]].corr().iloc[0, 1]
        print(f"corr(def, fouls per opponent possession) = {_cf:.2f}  "
              "(strongly negative = 'good defence' is partly fouling, whose FT cost the FG-only def ignores)")
        _row["corr_def_fouls"] = float(_cf)

    # The same pools the stream term used (a player who changed pool in the
    # window gets their most frequent one).
    _keys, _pool_kind = stream_keys(df, cfg)
    _sides["pool"] = _keys.reindex(_sides["row"]).to_numpy()
    _pl["pool"] = _pl["player"].map(_sides.groupby("player")["pool"].agg(lambda x: x.mode().iat[0]))
    _multi = int((_sides.groupby("player")["pool"].nunique() > 1).sum())
    _sizes = sorted(_pl["pool"].value_counts().tolist(), reverse=True)
    try:
        _pl.to_csv(OUT_DIR / "player_off_def.csv", index=False)
    except OSError as e:
        print(f"  WARNING: player_off_def.csv not written ({e})")

    # corr(off, def). The correlation of posterior MEANS is biased (shrinkage;
    # in corr mode the def means borrow from off through rho and the figure is
    # pushed toward +/-1). The per-draw correlation is attenuated toward 0 by
    # posterior noise. Neither is rho; in corr mode read rho_player_od.
    _cm = _pl[["off", "def"]].corr().iloc[0, 1]
    _zo = (_off_draws - _off_draws.mean(1, keepdims=True)) / _off_draws.std(1, keepdims=True)
    _zd = (_def_draws - _def_draws.mean(1, keepdims=True)) / _def_draws.std(1, keepdims=True)
    _cd = (_zo * _zd).mean(1)
    _lo, _hi = np.quantile(_cd, [0.055, 0.945])
    print(f"corr(off, def): of posterior means {_cm:.2f} (biased); per draw {_cd.mean():.2f} "
          f"[89%: {_lo:.2f}, {_hi:.2f}] (attenuated toward 0)")
    if cfg.defence_mode == "corr":
        print("  corr mode: def means borrow from off through rho, so the first figure is "
              "inflated -- the estimate is rho_player_od in the table above")
    _row.update(corr_means=float(_cm), corr_per_draw=float(_cd.mean()),
                n_pools=len(_sizes), pool_kind=_pool_kind)
    print(f"  pools ({_pool_kind}, as used by the stream term): {len(_sizes)} "
          f"with sizes {_sizes[:10]}; {_multi} players appear in more than one")
    if len(_sizes) > 1:
        _w = _pl[["off", "def"]] - _pl.groupby("pool")[["off", "def"]].transform("mean")
        _cw = _w.corr().iloc[0, 1]
        print(f"  corr(off, def) of means WITHIN pools: {_cw:.2f}  "
              "(removes pool-level shifts; a player-level common factor survives it)")
        print("  Rankings below compare players ACROSS pools only as far as pools mixed in the window.")
        _row["corr_within_pool"] = float(_cw)
    else:
        print("  one pool: the pool-level artefact is NOT ruled out. Fit a single ISO week, "
              "e.g. -AsOf 2026-09-14 -Days 7 (the Monday AFTER the week), or add STREAM_NUMBER.")
    _cols = ["player", "off", "def", "quality", "openness", "matches"] + (["pool"] if len(_sizes) > 1 else [])
    print("Highest quality (off - def):"); print(_pl.nlargest(10, "quality")[_cols].to_string(index=False))
    print("Most open (off + def):");       print(_pl.nlargest(10, "openness")[_cols].to_string(index=False))
    print("Tightest (off + def):");        print(_pl.nsmallest(10, "openness")[_cols].to_string(index=False))

# ---- one comparable row per run --------------------------------------------
from dataclasses import asdict
_row = {"run_utc": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "window_start": str(df["SCHEDULED_START_TIME_UTC"].min()),
        "window_end": str(_w_end), "days": DAYS, "matches": len(df),
        **asdict(cfg), **_row}
for _v in _hyper:
    _row[_v] = _tab.loc[_v, "mean"]
    _row[_v + "_rhat"] = _tab.loc[_v, "r_hat"]
_cmp = Path("runs_comparison.csv")
try:
    # rewrite rather than append: modes have different columns
    _hist = pd.read_csv(_cmp) if _cmp.exists() else pd.DataFrame()
    pd.concat([_hist, pd.DataFrame([_row])], ignore_index=True).to_csv(_cmp, index=False)
    print(f"  Added a row to {_cmp.resolve()}")
except OSError as e:
    print(f"  WARNING: {_cmp} not updated ({e}) -- is it open in Excel?")
print(f"  Outputs written to {OUT_DIR.resolve()}")

print("=" * 60)
print(f"  Matches:        {len(df)}")
print(f"  Players:        {n_players}")
print(f"  Teams:          {n_teams}")
print(f"  FT submodel:    {'YES' if has_ft else 'NO (FT columns absent)'}")
print(f"  HCA mode:       {cfg.hca_mode}")
print(f"  Defence mode:   {cfg.defence_mode}")
print(f"  Dispersion:     {cfg.dispersion_prior}")
print(f"  Stream key:     {'STREAM_NUMBER' if 'STREAM_NUMBER' in df.columns else cfg.stream_fallback}")
print(f"  MCMC:           {cfg.chains} chains x {cfg.draws} draws ({cfg.tune} tune)")
print("=" * 60)
