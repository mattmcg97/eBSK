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
    defence_effect: bool = False


def load_match_features(
        time_col: str = "SCHEDULED_START_TIME_UTC",
        asof: str | None = None,
        lookback_days: int | None = None,
) -> pd.DataFrame:
    df = session.sql(
            """
            SELECT s.*, e.SCHEDULED_START_TIME_UTC
            FROM MODELLING.POSSESSION_NB_MATCH_STATS s
            LEFT JOIN SIS_PROD_CG_CURATED.SHARED.EVENT e
              ON e.MATCH_CODE = s.MATCH_CODE
            """
).to_pandas()

    required = [
        "MATCH_CODE",
        "PLAYER_1_HANDLE", "PLAYER_2_HANDLE",
        "PLAYER_1_TEAM", "PLAYER_2_TEAM",
        "AWAY_POINTS", "HOME_POINTS",
        "AWAY_POSSESSIONS", "HOME_POSSESSIONS",
        "GAME_DURATION_SECONDS",
    ]
    print(df.columns)
    ft_cols = ["AWAY_FREE_THROW_SCORED", "HOME_FREE_THROW_SCORED", "AWAY_FREE_THROW_ATTEMPTED", "HOME_FREE_THROW_ATTEMPTED"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if asof is not None:
        if time_col not in df.columns:
            raise ValueError(f"--asof provided but time_col '{time_col}' not found")
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[time_col])

        print(f"[load_match_features] data spans {df[time_col].min()} -> "
              f"{df[time_col].max()} ({len(df)} matches with a start time)")
        if asof == "latest":
            asof_ts = df[time_col].max() + pd.Timedelta("1s")
        else:
            asof_ts = pd.to_datetime(asof, utc=True)
        df = df[df[time_col] < asof_ts]

        if lookback_days is not None:
            start_ts = asof_ts - pd.Timedelta(int(lookback_days), unit="D")
            df = df[df[time_col] >= start_ts]

    #if "POSSESSIONS_TOTAL" not in df.columns:
    #    df["POSSESSIONS_TOTAL"] = df["AWAY_POSSESSIONS"] + df["HOME_POSSESSIONS"]

    df["GAME_DURATION_SECONDS"] = pd.to_numeric(df["GAME_DURATION_SECONDS"], errors="coerce").fillna(1200.0)
    for c in ["AWAY_POINTS", "HOME_POINTS", "AWAY_POSSESSIONS", "HOME_POSSESSIONS", "TOTAL_POSSESSIONS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)

    if df.empty:
        raise ValueError(f"No matches in the window (asof={asof}, "
                         f"lookback_days={lookback_days}). Check the date range above.")

    df = df[(df["TOTAL_POSSESSIONS"] > 0) & (df["GAME_DURATION_SECONDS"] > 0)].copy()

    _has_ft = all(c in df.columns for c in ft_cols)
    if _has_ft:
        for c in ft_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
        df["AWAY_FIELD_GOALS"] = (df["AWAY_POINTS"] - df["AWAY_FREE_THROW_SCORED"]).clip(lower=0)
        df["HOME_FIELD_GOALS"] = (df["HOME_POINTS"] - df["HOME_FREE_THROW_SCORED"]).clip(lower=0)
    else:
        df["AWAY_FIELD_GOALS"] = df["AWAY_POINTS"]
        df["HOME_FIELD_GOALS"] = df["HOME_POINTS"]

    id_cols = ["PLAYER_1_HANDLE", "PLAYER_2_HANDLE", "PLAYER_1_TEAM", "PLAYER_2_TEAM"]
    for c in id_cols:
        df[c] = df[c].astype("string").str.strip()
        df.loc[df[c].isin(["", "None", "nan", "NaN"]), c] = pd.NA

    before = len(df)
    df = df.dropna(subset=id_cols).copy()
    dropped = before - len(df)
    if dropped:
        print(f"[load_match_features] Dropped {dropped} rows with missing player/team identifiers.")
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

    _stream = (df["STREAM_NUMBER"].astype(str) if "STREAM_NUMBER" in df.columns
               else pd.Series("NA", index=df.index))
    if cfg.stream_by_week:
        _iso = _day.dt.isocalendar()
        _stream_key = (_stream + "|" + _iso.week.astype(str) + "-"
                       + _iso.year.astype(str))
    else:
        _stream_key = _stream
    stream_idx, _ = make_index(_stream_key)
    n_streams = int(stream_idx.max() + 1)

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

        alpha_pts = pm.HalfNormal("alpha_pts", sigma=3.0)

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

        if cfg.stream_effect:
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

        if cfg.defence_effect:
            sigma_def_player = pm.HalfNormal("sigma_def_player", sigma=0.25)
            sigma_def_team = pm.HalfNormal("sigma_def_team", sigma=0.15)
            def_player_z = pm.Normal("def_player_z", 0.0, 1.0, shape=n_players)
            def_team_z = pm.Normal("def_team_z", 0.0, 1.0, shape=n_teams)
            def_player = pm.Deterministic(
                "def_player", center(sigma_def_player * def_player_z))
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

        alpha_poss = pm.HalfNormal("alpha_poss", sigma=5.0)

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
        if cfg.stream_effect and cfg.stream_on_pace:
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


def export_outputs(out_dir: Path, df: pd.DataFrame, idata) -> None:
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
        "chem_raw", "chem_pace_raw",
        "player_z", "player_raw", "team_raw", "def_player_z", "def_team_z",
        "player_pace_raw", "team_pace_raw", "player_ft_raw",
    }
    try:
        _keep = [v for v in idata.posterior.data_vars if v not in _DROP]
        # posterior only, as a plain xarray Dataset: works on both the
        # InferenceData and the newer DataTree-based arviz
        _post = idata.posterior
        _post = _post.to_dataset() if hasattr(_post, "to_dataset") else _post
        _post[_keep].to_netcdf(out_dir / "posterior.nc")
        print(f"    posterior.nc: kept {len(_keep)} of "
              f"{len(idata.posterior.data_vars)} variables "
              f"(dropped per-level nuisance draws)")
    except Exception as e:
        # NEVER let this take the rest of the export down. params_summary.json,
        # the maps and scalar_summary.csv are the load-bearing outputs.
        print(f"    WARNING: posterior.nc not written ({type(e).__name__}: {e})")
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
        summary["sigma_def_team_mean"] = float(post_mean("sigma_def_team"))

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

    summary["stream_effect_fitted"] = bool(cfg.stream_effect)
    if cfg.stream_effect:
        summary["sigma_stream_mean"] = float(post_mean("sigma_stream"))
        summary["sigma_stream_sd"] = float(post_sd("sigma_stream"))
        if cfg.stream_on_pace:
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
                                       "sigma_player_pace", "sigma_team_pace"] if
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
# SMOKE TEST: defence only, chemistry off, short window, light sampling.
cfg = FitConfig(
    defence_effect=True,
    chem_mode="off",
    chem_on_ppp=False,
    chem_on_pace=False,
    match_effect=False,
    stream_effect=True,
    session_effect=False,
    chains=2,
    draws=500,
    tune=1000,
)

print("  Start data frame     - " + datetime.now(tz=timezone.utc).strftime(DT_FMT))
# asof: "latest" = the most recent match in the table, or a date like "2026-10-04"
df = load_match_features(asof=os.environ.get("DECOMP_ASOF", "latest"),
                         lookback_days=int(os.environ.get("DECOMP_DAYS", "7")))
print(f"  Window: {df['SCHEDULED_START_TIME_UTC'].min()} -> "
      f"{df['SCHEDULED_START_TIME_UTC'].max()}  ({len(df)} matches)")
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
export_outputs(Path(os.environ.get("DECOMP_OUT", "decomp_out_def")), df, idata)

print("  End process          - " + datetime.now(tz=timezone.utc).strftime(DT_FMT))


import arviz as az
print(az.summary(idata, var_names=[v for v in ["beta0", "alpha_pts", "sigma_player", "sigma_team",
      "sigma_def_player", "sigma_def_team", "hca_global"] if v in idata.posterior]))
if cfg.defence_effect:
    _pk = sorted(pd.concat([df["PLAYER_1_HANDLE"], df["PLAYER_2_HANDLE"]]).astype(str).unique())
    _cnt = pd.concat([df["PLAYER_1_HANDLE"], df["PLAYER_2_HANDLE"]]).astype(str).value_counts()
    _pl = pd.DataFrame({"player": _pk,
                        "off": idata.posterior["player"].mean(("chain", "draw")).values,
                        "def": idata.posterior["def_player"].mean(("chain", "draw")).values})
    _pl["matches"] = _pl["player"].map(_cnt)
    print("corr(off, def) across players:", round(_pl[["off", "def"]].corr().iloc[0, 1], 3))
    print("Best defenders (most negative def):"); print(_pl.nsmallest(10, "def").to_string(index=False))
    print("Worst defenders:"); print(_pl.nlargest(10, "def").to_string(index=False))

_out = Path(os.environ.get("DECOMP_OUT", "decomp_out_def")).resolve()
print(f"  Outputs written to {_out}")
if os.environ.get("DECOMP_UPLOAD", "0") == "1":
    try:
        session.file.put(str(_out / "*").replace("\\", "/"),
                         f"@{DB_NAME}.MODELLING.STAGE_BASKETBALL_POSSESSION/decomp_out_def",
                         auto_compress=False, overwrite=True)
        print("  Uploaded to stage.")
    except Exception as e:
        print(f"  WARNING: stage upload failed ({type(e).__name__}): {str(e)[:200]}")

print("=" * 60)
print(f"  Matches:        {len(df)}")
print(f"  Players:        {n_players}")
print(f"  Teams:          {n_teams}")
print(f"  FT submodel:    {'YES' if has_ft else 'NO (FT columns absent)'}")
print(f"  HCA mode:       {cfg.hca_mode}")
print(f"  MCMC:           {cfg.chains} chains x {cfg.draws} draws ({cfg.tune} tune)")
print("=" * 60)
