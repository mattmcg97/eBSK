#!/usr/bin/env python3
"""
Rolling held-out test: fit a window, score the next day(s), repeat.

Every config (defence mode x noise model) is fitted on the SAME folds, so the
results are paired match by match. Each fold trains on BT_TRAIN_DAYS up to a
UTC-midnight cut and scores the matches in the following BT_TEST_DAYS.

Scores per held-out match:
  lpd_pts   log predictive density of BOTH sides' FG points given the observed
            possessions -- the cleanest test of the efficiency leg (defence).
  lpd_poss  log predictive density of total possessions given duration.
  total / margin of the final score (FG + FT), simulated from the full model
            (possessions -> FG points -> FT attempts and makes): MAE of the
            predictive mean, CRPS, PIT, and whether the central 50/80/95%
            intervals contain the result -- calibration, i.e. the noise question.
Noise diagnostics: the residual var/mean around EACH config's predictions (FG
points given possessions; possessions given duration), with the posterior
variance of the prediction removed, against what that config assumes, plus the
home/away residual correlation. A config with biased means shows a higher
residual var/mean, so read the data's own dispersion off the best-scoring one.

Players, teams and stream-weeks not seen in training are drawn from their
population distribution (N(0, sigma) per posterior draw), not set to exactly 0,
so new-week and new-player matches carry their real extra uncertainty.

Settings (run.ps1 -Backtest sets these): BT_CONFIGS, BT_FOLDS, BT_TRAIN_DAYS,
BT_TEST_DAYS, BT_STEP_DAYS, DECOMP_ASOF, DECOMP_STREAM, DECOMP_CHAINS /
DECOMP_DRAWS / DECOMP_TUNE; optional env BT_SIM_DRAWS (default 400) and BT_OUT.
Re-running with the same settings resumes (finished (fold, config) pairs are
skipped); settings.json in the output folder guards against mixing settings.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decomp_defence as dd  # noqa: E402  (validates DECOMP_* settings on import)


def _env(name, default):
    return os.environ.get(name, default).strip()


CONFIGS = [c for c in re.split(r"[,\s]+", _env("BT_CONFIGS", "off:original,on:original,off:free,on:free").lower()) if c]
FOLDS = int(_env("BT_FOLDS", "7"))
TRAIN_DAYS = int(_env("BT_TRAIN_DAYS", "7"))
TEST_DAYS = int(_env("BT_TEST_DAYS", "1"))
STEP_DAYS = int(_env("BT_STEP_DAYS", "2"))
SIM_DRAWS = int(_env("BT_SIM_DRAWS", "400"))
if not CONFIGS or min(FOLDS, TRAIN_DAYS, TEST_DAYS, STEP_DAYS, SIM_DRAWS) < 1:
    sys.exit("Backtest settings must be positive and BT_CONFIGS non-empty")
if TEST_DAYS > STEP_DAYS:
    sys.exit(f"TestDays ({TEST_DAYS}) > StepDays ({STEP_DAYS}): test windows would overlap and "
             "the same match would be scored more than once")
for c in CONFIGS:
    d, _, n = c.partition(":")
    if d not in ("off", "on", "corr") or n not in ("original", "weak", "free"):
        sys.exit(f"BT_CONFIGS entry {c!r} -- use defence:noise, e.g. on:free")

TIME = "SCHEDULED_START_TIME_UTC"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def week_key(t: pd.Series) -> pd.Series:
    """ISO week as stream_keys() writes it."""
    iso = pd.to_datetime(t, utc=True).dt.isocalendar()
    return (iso.week.astype("string") + "-" + iso.year.astype("string")).fillna("NA")


def index_map(series: pd.Series) -> dict:
    """The exact key -> index mapping fit_model used (make_index sorts)."""
    _, m = dd.make_index(series)
    return dict(zip(m["key"].astype(str), m["index"].astype(int)))


def test_stream_keys(train, test, cfg):
    """Stream key for each test row, comparable with the training keys (None = unseen)."""
    if "STREAM_NUMBER" in test.columns or cfg.stream_fallback == "week":
        return [str(k) for k in dd.stream_keys(test, cfg)[0]]
    # pool fallback: the training pool either player was in, in the same ISO week
    tk = dd.stream_keys(train, cfg)[0].astype(str).tolist()
    tw = week_key(train[TIME]).astype(str).tolist()
    look = {}
    for w, a, b, k in zip(tw, train["PLAYER_1_HANDLE"].astype(str), train["PLAYER_2_HANDLE"].astype(str), tk):
        look[f"{w}|{a}"] = k
        look[f"{w}|{b}"] = k
    ww = week_key(test[TIME]).astype(str).tolist()
    return [look.get(f"{w}|{a}", look.get(f"{w}|{b}"))
            for w, a, b in zip(ww, test["PLAYER_1_HANDLE"].astype(str), test["PLAYER_2_HANDLE"].astype(str))]


def pick(post, name, idx):
    x = np.asarray(post[name])
    return x.reshape(-1, *x.shape[2:])[idx]


class Unseen:
    """Standard-normal draws per (block, unseen key), so an unseen player or
    stream-week gets ONE population draw per posterior draw, shared by every
    row (and both sides) it appears in."""
    def __init__(self, rng, S):
        self.rng, self.S, self.z = rng, S, {}

    def __call__(self, block, key):
        k = (block, key)
        if k not in self.z:
            self.z[k] = self.rng.standard_normal(self.S)
        return self.z[k]


def gather(arr, mapping, keys, sigma=None, unseen=None, block=None, mix=None):
    """arr (S, K) indexed by mapping[key]. Unseen keys: sigma * z from the
    population (z from `unseen`; `mix` = (rho, other_block) correlates it with
    another block, for corr-mode defence), or 0 if no sigma is given."""
    S = arr.shape[0]
    out = np.zeros((S, len(keys)))
    for j, k in enumerate(keys):
        i = mapping.get(str(k), -1) if k is not None else -1
        if i >= 0:
            out[:, j] = arr[:, i]
        elif sigma is not None:
            key = str(k)
            z = unseen(block, key)
            if mix is not None:
                rho, other = mix
                z = rho * unseen(other, key) + np.sqrt(np.clip(1 - rho ** 2, 0, None)) * z
            out[:, j] = sigma * z
    return out


def log_count(y, mu, noise, par):
    """log p(y) under NB(alpha=par) or discretised Normal(var = par * mu)."""
    if noise == "nb":
        a = par[:, None]
        return stats.nbinom.logpmf(y[None, :], a, a / (a + mu))
    sd = np.sqrt(par[:, None] * mu)
    lo, hi = (y[None, :] - 0.5 - mu) / sd, (y[None, :] + 0.5 - mu) / sd
    # log(Phi(hi) - Phi(lo)) in whichever tail keeps precision
    upper = lo > 0
    big = np.where(upper, stats.norm.logsf(lo), stats.norm.logcdf(hi))
    small = np.where(upper, stats.norm.logsf(hi), stats.norm.logcdf(lo))
    return big + np.log1p(-np.exp(np.minimum(small - big, 0.0)) + 1e-300)


def draw_count(rng, mu, noise, par):
    if noise == "nb":
        a = par[:, None]
        return rng.negative_binomial(np.broadcast_to(a, mu.shape), a / (a + mu))
    return np.maximum(np.rint(rng.normal(mu, np.sqrt(par[:, None] * mu))), 0)


def crps(samples, y):
    """Sample CRPS per column: E|X - y| - 0.5 E|X - X'| (sorted-sample formula)."""
    x = np.sort(samples, axis=0)
    s = x.shape[0]
    w = (2 * np.arange(1, s + 1) - s - 1)[:, None]
    return np.abs(x - y[None, :]).mean(0) - (w * x).sum(0) / s ** 2


def ft_rates(train):
    """Empirical FT attempts per own possession: player, team, global (as export_outputs)."""
    lg = pd.concat([
        pd.DataFrame({"p": train["PLAYER_1_HANDLE"].astype(str), "t": train["PLAYER_1_TEAM"].astype(str),
                      "ft": train["AWAY_FREE_THROW_ATTEMPTED"], "ps": train["AWAY_POSSESSIONS"]}),
        pd.DataFrame({"p": train["PLAYER_2_HANDLE"].astype(str), "t": train["PLAYER_2_TEAM"].astype(str),
                      "ft": train["HOME_FREE_THROW_ATTEMPTED"], "ps": train["HOME_POSSESSIONS"]})])
    g = float(lg["ft"].sum() / max(lg["ps"].sum(), 1))
    pr = lg.groupby("p")[["ft", "ps"]].sum()
    tr = lg.groupby("t")[["ft", "ps"]].sum()
    return (pr["ft"] / pr["ps"].clip(lower=1)).to_dict(), (tr["ft"] / tr["ps"].clip(lower=1)).to_dict(), g


# ---------------------------------------------------------------------------
# predict and score one fold for one fitted config
# ---------------------------------------------------------------------------
def score_fold(idata, train, test, cfg, rng):
    post = idata.posterior
    S_all = post.sizes["chain"] * post.sizes["draw"]
    idx = rng.choice(S_all, size=min(SIM_DRAWS, S_all), replace=False)
    S = len(idx)
    has = lambda v: v in post.data_vars

    pmap = index_map(pd.concat([train["PLAYER_1_HANDLE"], train["PLAYER_2_HANDLE"]], ignore_index=True))
    tmap = index_map(pd.concat([train["PLAYER_1_TEAM"], train["PLAYER_2_TEAM"]], ignore_index=True))
    smap = index_map(dd.stream_keys(train, cfg)[0])
    pa, ph = test["PLAYER_1_HANDLE"].astype(str).tolist(), test["PLAYER_2_HANDLE"].astype(str).tolist()
    ta, th = test["PLAYER_1_TEAM"].astype(str).tolist(), test["PLAYER_2_TEAM"].astype(str).tolist()
    sk = test_stream_keys(train, test, cfg)
    # unseen pools (pool fallback, new week): one shared draw per pool among the test matches
    tpool = dd.stream_keys(test, cfg)[0].astype(str).tolist()
    sk = [k if k is not None else f"__test|{tpool[j]}" for j, k in enumerate(sk)]
    z = lambda: np.zeros((S, len(test)))
    un = Unseen(rng, S)
    sig = lambda v: pick(post, v, idx) if has(v) else None

    # efficiency leg (unseen levels: population draws, see Unseen)
    off, team = pick(post, "player", idx), pick(post, "team", idx)
    g_off = lambda keys: gather(off, pmap, keys, sig("sigma_player"), un, "off")
    g_team = lambda keys: gather(team, tmap, keys, sig("sigma_team"), un, "team")
    eta_h = pick(post, "beta0", idx)[:, None] + g_off(ph) + g_team(th)
    eta_a = pick(post, "beta0", idx)[:, None] + g_off(pa) + g_team(ta)
    if has("hca_global"):
        eta_h = eta_h + pick(post, "hca_global", idx)[:, None]
    if has("stream_effect"):
        st = gather(pick(post, "stream_effect", idx), smap, sk, sig("sigma_stream"), un, "stream")
        eta_h, eta_a = eta_h + st, eta_a + st
    if has("def_player"):
        dp, dt = pick(post, "def_player", idx), pick(post, "def_team", idx)
        mix = (pick(post, "rho_player_od", idx), "off") if has("rho_player_od") else None
        g_def = lambda keys: gather(dp, pmap, keys, sig("sigma_def_player"), un, "def", mix)
        g_dt = lambda keys: gather(dt, tmap, keys, sig("sigma_def_team"), un, "def_team")
        eta_h = eta_h + g_def(pa) + g_dt(ta)
        eta_a = eta_a + g_def(ph) + g_dt(th)
    # pace leg
    pp, tp = pick(post, "player_pace", idx), pick(post, "team_pace", idx)
    g_pp = lambda keys: gather(pp, pmap, keys, sig("sigma_player_pace"), un, "pace")
    g_tp = lambda keys: gather(tp, tmap, keys, sig("sigma_team_pace"), un, "team_pace")
    eta_p = pick(post, "pace_beta0", idx)[:, None] + g_pp(ph) + g_pp(pa) + g_tp(th) + g_tp(ta)
    if has("stream_pace_effect"):
        eta_p = eta_p + gather(pick(post, "stream_pace_effect", idx), smap, sk,
                               sig("sigma_stream_pace"), un, "stream_pace")

    noise = "normal" if cfg.dispersion_prior == "free" else "nb"
    par_pts = pick(post, "phi_pts" if noise == "normal" else "alpha_pts", idx)
    par_poss = pick(post, "phi_poss" if noise == "normal" else "alpha_poss", idx)

    poss_h = test["HOME_POSSESSIONS"].to_numpy(float)
    poss_a = test["AWAY_POSSESSIONS"].to_numpy(float)
    dur = test["GAME_DURATION_SECONDS"].to_numpy(float)
    y_h = test["HOME_FIELD_GOALS"].to_numpy(float)
    y_a = test["AWAY_FIELD_GOALS"].to_numpy(float)
    y_p = test["TOTAL_POSSESSIONS"].to_numpy(float)
    mu_h, mu_a = poss_h * np.exp(eta_h), poss_a * np.exp(eta_a)
    mu_p = dur * np.exp(eta_p)

    out = pd.DataFrame({"MATCH_CODE": test["MATCH_CODE"].astype(str).to_numpy(),
                        "time": test[TIME].astype(str).to_numpy()})
    out["unseen_player"] = [int(str(a) not in pmap or str(b) not in pmap) for a, b in zip(pa, ph)]
    out["unseen_stream"] = [int(str(k) not in smap) for k in sk] if has("stream_effect") else 0
    lp = log_count(y_h, mu_h, noise, par_pts) + log_count(y_a, mu_a, noise, par_pts)
    out["lpd_pts"] = logsumexp(lp, axis=0) - np.log(S)
    out["lpd_poss"] = logsumexp(log_count(y_p, mu_p, noise, par_poss), axis=0) - np.log(S)

    # noise diagnostics: residual variance minus parameter-uncertainty variance
    out["fg_res_h"], out["fg_res_a"] = y_h - mu_h.mean(0), y_a - mu_a.mean(0)
    out["fg_mu_h"], out["fg_mu_a"] = mu_h.mean(0), mu_a.mean(0)
    out["fg_parvar"] = (mu_h.var(0) + mu_a.var(0)) / 2
    out["poss_res"], out["poss_mu"], out["poss_parvar"] = y_p - mu_p.mean(0), mu_p.mean(0), mu_p.var(0)
    vm = "var_mean_pts" if has("var_mean_pts") else None
    out["model_vm_pts"] = float(np.asarray(post[vm]).mean()) if vm else np.nan
    out["model_vm_poss"] = float(np.asarray(post["var_mean_poss"]).mean()) if has("var_mean_poss") else np.nan

    # full simulation: possessions -> FG -> FT -> final score
    tot = np.maximum(draw_count(rng, mu_p, noise, par_poss), 2)
    sh = np.floor(tot / 2) + ((tot % 2 == 1) & (rng.random(tot.shape) < 0.5))
    sa = tot - sh
    fg_h = draw_count(rng, sh * np.exp(eta_h), noise, par_pts)
    fg_a = draw_count(rng, sa * np.exp(eta_a), noise, par_pts)
    if has("ft_beta0"):
        pr, tr, g = ft_rates(train)
        rate = lambda ps, ts: np.array([pr.get(p, tr.get(t, g)) for p, t in zip(ps, ts)])
        pft, tft = pick(post, "player_ft", idx), pick(post, "team_ft", idx)
        b0 = pick(post, "ft_beta0", idx)[:, None]
        g_pft = lambda keys: gather(pft, pmap, keys, sig("sigma_player_ft"), un, "ft")
        g_tft = lambda keys: gather(tft, tmap, keys, sig("sigma_team_ft"), un, "team_ft")
        mk_h = expit(b0 + g_pft(ph) + g_tft(th))
        mk_a = expit(b0 + g_pft(pa) + g_tft(ta))
        att_h = rng.poisson(rate(ph, th)[None, :] * sh)
        att_a = rng.poisson(rate(pa, ta)[None, :] * sa)
        ft_h, ft_a = rng.binomial(att_h, mk_h), rng.binomial(att_a, mk_a)
    else:
        ft_h = ft_a = z()
    sim_tot, sim_mar = fg_h + ft_h + fg_a + ft_a, (fg_h + ft_h) - (fg_a + ft_a)
    obs_tot = (test["HOME_POINTS"] + test["AWAY_POINTS"]).to_numpy(float)
    obs_mar = (test["HOME_POINTS"] - test["AWAY_POINTS"]).to_numpy(float)
    for nm, sim, obs in (("total", sim_tot, obs_tot), ("margin", sim_mar, obs_mar)):
        out[f"{nm}_obs"] = obs
        out[f"{nm}_mean"] = sim.mean(0)
        out[f"{nm}_sd"] = sim.std(0)
        out[f"{nm}_crps"] = crps(sim, obs)
        pit = (sim < obs).mean(0) + 0.5 * (sim == obs).mean(0)
        out[f"{nm}_pit"] = pit
        for c in (50, 80, 95):   # inside the central c% interval, via the mid-PIT
            out[f"{nm}_in{c}"] = (np.abs(pit - 0.5) <= c / 200).astype(int)
    return out


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def summarise(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [(c, "all", g) for c, g in m.groupby("config", sort=False)]
    groups += [(c, "new_week" if nw else "same_week", g)
               for (c, nw), g in m.groupby(["config", "new_week"], sort=False)]
    for cfg_name, part, g in groups:
        fg_mu = ((g["fg_mu_h"] + g["fg_mu_a"]) / 2).mean()
        fg_raw = ((g["fg_res_h"] ** 2 + g["fg_res_a"] ** 2) / 2).mean() / fg_mu
        poss_raw = (g["poss_res"] ** 2).mean() / g["poss_mu"].mean()
        fg_vm = fg_raw - g["fg_parvar"].mean() / fg_mu
        poss_vm = poss_raw - g["poss_parvar"].mean() / g["poss_mu"].mean()
        rows.append({
            "config": cfg_name, "rows": part, "matches": len(g),
            "unseen_player_share": g["unseen_player"].mean(), "unseen_stream_share": g["unseen_stream"].mean(),
            "lpd_pts_per_match": g["lpd_pts"].mean(), "lpd_poss_per_match": g["lpd_poss"].mean(),
            "total_MAE": (g["total_mean"] - g["total_obs"]).abs().mean(),
            "total_RMSE": np.sqrt(((g["total_mean"] - g["total_obs"]) ** 2).mean()),
            "total_CRPS": g["total_crps"].mean(),
            "total_pred_sd": g["total_sd"].mean(),
            "total_actual_sd_of_error": (g["total_obs"] - g["total_mean"]).std(),
            "total_in50": g["total_in50"].mean(), "total_in80": g["total_in80"].mean(), "total_in95": g["total_in95"].mean(),
            "margin_MAE": (g["margin_mean"] - g["margin_obs"]).abs().mean(),
            "margin_RMSE": np.sqrt(((g["margin_mean"] - g["margin_obs"]) ** 2).mean()),
            "margin_CRPS": g["margin_crps"].mean(),
            "margin_pred_sd": g["margin_sd"].mean(),
            "margin_actual_sd_of_error": (g["margin_obs"] - g["margin_mean"]).std(),
            "margin_in50": g["margin_in50"].mean(), "margin_in80": g["margin_in80"].mean(), "margin_in95": g["margin_in95"].mean(),
            "resid_fg_var_mean": fg_vm, "resid_fg_var_mean_raw": fg_raw,
            "model_fg_var_mean": g["model_vm_pts"].mean(),
            "resid_poss_var_mean": poss_vm, "resid_poss_var_mean_raw": poss_raw,
            "model_poss_var_mean": g["model_vm_poss"].mean(),
            "home_away_resid_corr": np.corrcoef(g["fg_res_h"], g["fg_res_a"])[0, 1],
        })
    return pd.DataFrame(rows)


def compare(m: pd.DataFrame) -> pd.DataFrame:
    """Paired differences, match by match, for every pair of configs."""
    rows = []
    cfgs = list(dict.fromkeys(m["config"]))
    piv = {c: m[m["config"] == c].set_index(["fold", "MATCH_CODE"]) for c in cfgs}
    for i, a in enumerate(cfgs):
        for b in cfgs[i + 1:]:
            j = piv[a].index.intersection(piv[b].index)
            if len(j) < 10:
                continue
            A, B = piv[a].loc[j], piv[b].loc[j]
            for metric, sign in (("lpd_pts", 1), ("lpd_poss", 1), ("total_crps", -1), ("margin_crps", -1)):
                d = sign * (B[metric] - A[metric])          # > 0 means b is better
                se = d.std(ddof=1) * np.sqrt(len(d))
                # matches in a fold share one fit: also an SE from the per-fold sums
                fs = d.groupby(level="fold").sum()
                se_f = fs.std(ddof=1) * np.sqrt(len(fs)) if len(fs) > 2 else np.nan
                rows.append({"a": a, "b": b, "metric": metric, "matches": len(d), "folds": len(fs),
                             "b_better_by_total": d.sum(), "se": se, "z": d.sum() / se if se > 0 else np.nan,
                             "z_fold_blocks": d.sum() / se_f if se_f and se_f > 0 else np.nan,
                             "b_better_new_week_only": d[A["new_week"].astype(bool)].sum()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    db_name = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
               else "SIS_UAT_CG_CURATED")
    dd.connect(db_name)
    df = dd.load_match_features(asof=dd.ASOF, lookback_days=None)
    df = df.sort_values(TIME).reset_index(drop=True)
    end = df[TIME].max().floor("D") + pd.Timedelta(1, unit="D")
    out_dir = Path(os.environ.get("BT_OUT") or
                   f"backtest_out_{end:%Y%m%d}_{FOLDS}x{TEST_DAYS}d_step{STEP_DAYS}_train{TRAIN_DAYS}d")
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {"database": db_name, "end": str(end), "train_days": TRAIN_DAYS, "test_days": TEST_DAYS,
                "step_days": STEP_DAYS, "chains": dd.CHAINS, "draws": dd.DRAWS, "tune": dd.TUNE,
                "stream": dd.STREAM_FALLBACK, "sim_draws": SIM_DRAWS}
    spath = out_dir / "settings.json"
    if spath.exists():
        old = json.loads(spath.read_text())
        diff = {k: (old.get(k), v) for k, v in settings.items() if old.get(k) != v}
        if diff:
            sys.exit(f"{out_dir} holds a backtest with different settings {diff}. "
                     "Delete or rename that folder, or set $env:BT_OUT to a new folder.")
    else:
        spath.write_text(json.dumps(settings, indent=1))
    mpath = out_dir / "matches.csv"
    done = pd.read_csv(mpath) if mpath.exists() else pd.DataFrame(columns=["fold", "config"])
    done_keys = set(zip(done["fold"].astype(int), done["config"].astype(str)))
    print(f"Backtest -> {out_dir.resolve()}\n  configs {CONFIGS}; {FOLDS} folds of {TEST_DAYS} test day(s), "
          f"step {STEP_DAYS} d, train {TRAIN_DAYS} d; {len(done_keys)} (fold, config) already done")

    rng = np.random.default_rng(2026)
    for k in range(FOLDS):
        t_end = end - pd.Timedelta(k * STEP_DAYS, unit="D")
        t0 = t_end - pd.Timedelta(TEST_DAYS, unit="D")
        tr0 = t0 - pd.Timedelta(TRAIN_DAYS, unit="D")
        train = df[(df[TIME] >= tr0) & (df[TIME] < t0)].copy()
        test = df[(df[TIME] >= t0) & (df[TIME] < t_end)].copy()
        if len(train) < 200 or len(test) < 20:
            print(f"fold {k}: skipped (train {len(train)}, test {len(test)} matches)")
            continue
        new_week = (week_key(test[TIME]) != week_key(pd.Series([train[TIME].max()]))[0]).to_numpy()
        for c in CONFIGS:
            if (k, c) in done_keys:
                continue
            dmode, _, noise = c.partition(":")
            cfg = dd.FitConfig(defence_mode=dmode, dispersion_prior=noise, stream_fallback=dd.STREAM_FALLBACK,
                               chem_mode="off", chem_on_ppp=False, chem_on_pace=False, match_effect=False,
                               stream_effect=True, session_effect=False,
                               chains=dd.CHAINS, draws=dd.DRAWS, tune=dd.TUNE)
            print(f"\n=== fold {k} [{t0:%Y-%m-%d} .. {t_end:%Y-%m-%d}) config {c}: "
                  f"train {len(train)} / test {len(test)} matches")
            idata = dd.fit_model(train, cfg)
            div = int(np.asarray(idata.sample_stats["diverging"]).sum())
            res = score_fold(idata, train, test, cfg, rng)
            res.insert(0, "config", c)
            res.insert(0, "fold", k)
            res["new_week"] = new_week.astype(int)
            res["divergences"] = div
            res.to_csv(mpath, mode="a", header=not mpath.exists(), index=False)   # survives interruption
            print(f"    lpd_pts {res['lpd_pts'].mean():.3f}/match, total CRPS {res['total_crps'].mean():.2f}, "
                  f"total in95 {res['total_in95'].mean():.0%}, divergences {div}")

    if not mpath.exists():
        sys.exit("Nothing was fitted: no fold had enough data (200 training / 20 test matches).")
    m = pd.read_csv(mpath)
    summ, comp = summarise(m), compare(m)
    summ.to_csv(out_dir / "summary.csv", index=False)
    comp.to_csv(out_dir / "comparisons.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        print("\n" + "=" * 70 + "\nSUMMARY (per config; 'all' rows, then split by new/same ISO week)\n"
              + summ.set_index(["config", "rows"]).T.to_string())
        print("\nPAIRED COMPARISONS (b_better_by_total > 0 means b is better; |z| > 2 is clear)\n"
              + comp.to_string(index=False))
    print(f"\nWritten: {mpath}, {out_dir / 'summary.csv'}, {out_dir / 'comparisons.csv'}")


if __name__ == "__main__":
    main()
