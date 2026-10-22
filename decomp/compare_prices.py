#!/usr/bin/env python3
"""
Prod vs UAT B vs candidate, scored on real pre-match prices and results.

For every NB match from CMP_FROM to CMP_TO (UTC days) with a result:
  sources    prod  = SIS_PROD_CG_CURATED.SHARED.PRICE_CHANGES
             uat_b = SIS_UAT_CG_CURATED.SHARED.PRICE_CHANGES_UATB
             each selection's LAST ACTIVE price before the scheduled start
             (PRICE_PROBABILITY is the fair, margin-free probability),
             renormalised within its market line;
  candidate  each config in CMP_CONFIGS (e.g. off:free,on:free) refitted every
             UTC midnight on the previous CMP_TRAIN_DAYS of PROD match stats,
             then simulated pre-match: 1,200 s of regulation, plus 180 s
             overtime periods while tied (game length is fixed apart from OT),
             and priced on EXACTLY the selections each source offered.

Markets (full game): moneyline, match winner (excl. OT, 3-way), handicap and
total points (main + alternative), player 1 / player 2 totals, total points
bands, winning margin (incl. and excl. OT), player points bands. Selections are
mapped to Player 1 / Player 2 by handle ("HANDLE (TEAM)"); band names are
parsed ("150-159", "Under 120", "200+", "<handle> by 1-3", "Draw" ...) and any
that cannot be parsed are listed in unparsed_selections.csv, never guessed.

Scores per market group (one market line of one match): log loss and Brier of
the realised selection; paired source-vs-candidate and prod-vs-UAT B
differences with day-block z (days share fits, so they are the honest unit);
each source's implied MEDIAN total / margin / player total (where its lines
cross 50%) against the result; and a moneyline calibration table.

Read-only on Snowflake (SELECT only). Writes compare_out_<from>_<to>_train<n>d\\
and resumes there (settings.json guard).  Usage: python -u decomp\\compare_prices.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decomp_defence as dd  # noqa: E402
import backtest as bt  # noqa: E402


def _env(name, default):
    return os.environ.get(name, default).strip()


FROM = pd.Timestamp(_env("CMP_FROM", "2026-10-12"), tz="UTC").floor("D")
TO = pd.Timestamp(_env("CMP_TO", "") or datetime.now(timezone.utc).strftime("%Y-%m-%d"), tz="UTC").floor("D")
CONFIGS = [c for c in re.split(r"[,\s]+", _env("CMP_CONFIGS", "off:free,on:free").lower()) if c]
TRAIN_DAYS = int(_env("CMP_TRAIN_DAYS", "7"))
SIM = int(_env("CMP_SIM", "2000"))
OUTCOME_DB = "SIS_PROD_CG_CURATED"
SOURCES = {"prod": "SIS_PROD_CG_CURATED.SHARED.PRICE_CHANGES",
           "uat_b": "SIS_UAT_CG_CURATED.SHARED.PRICE_CHANGES_UATB"}
REG_SECONDS, OT_SECONDS, MAX_OT = 1200, 180, 6
for c in CONFIGS:
    d, _, n = c.partition(":")
    if d not in ("off", "on", "corr") or n not in ("original", "weak", "free"):
        sys.exit(f"CMP_CONFIGS entry {c!r} -- use defence:noise, e.g. on:free")
if TO < FROM:
    sys.exit(f"CMP_TO {TO:%Y-%m-%d} is before CMP_FROM {FROM:%Y-%m-%d}")

# market name -> (kind, which scores: "ot" = final incl. overtime, "reg" = regulation)
MARKETS = {
    "Moneyline (including overtime)": ("winner", "ot"),
    "Match Winner (excluding overtime)": ("winner", "reg"),
    "Handicap Betting (including overtime)": ("handicap", "ot"),
    "Handicap Betting (including overtime) - Alternative": ("handicap", "ot"),
    "Total Points (including overtime)": ("total", "ot"),
    "Total Points (including overtime) - Alternative": ("total", "ot"),
    "Player 1 Total Points (including overtime)": ("p1_total", "ot"),
    "Player 1 Total Points (including overtime) - Alternative": ("p1_total", "ot"),
    "Player 2 Total Points (including overtime)": ("p2_total", "ot"),
    "Player 2 Total Points (including overtime) - Alternative": ("p2_total", "ot"),
    "Total Points Bands (including overtime)": ("total_band", "ot"),
    "Winning Margin (including overtime)": ("margin_band", "ot"),
    "Winning Margin (excluding overtime)": ("margin_band", "reg"),
    "Player 1 Points Bands (including overtime)": ("p1_band", "ot"),
    "Player 2 Points Bands (including overtime)": ("p2_band", "ot"),
}
LINE_KINDS = {"handicap", "total", "p1_total", "p2_total"}   # one group per market line


def sel_key(df: pd.DataFrame) -> pd.Series:
    """One text key per selection (NaN lines/handicaps kept -- groupby/pivot drop NaN keys)."""
    f = lambda c: df[c].map(lambda v: "na" if pd.isna(v) else f"{float(v):g}")
    return (df["SOURCE"].astype(str) + "|" + df["MATCH_CODE"].astype(str) + "|" + df["MARKET_TYPE_NAME"].astype(str)
            + "|" + f("MARKET_LINE") + "|" + df["SELECTION_DESCRIPTION"].astype(str) + "|" + f("SELECTION_HANDICAP"))


def col_of(config: str) -> str:            # "off:free" -> "c_off_free" (a valid attribute name)
    return "c_" + config.replace(":", "_")


def name_of(col: str) -> str:
    return col[2:].replace("_", ":", 1) if col.startswith("c_") else col


# ---------------------------------------------------------------------------
# selection parsing
# ---------------------------------------------------------------------------
def handle_of(desc: str) -> str:
    return re.sub(r"\s*\([^()]*\)\s*$", "", str(desc)).strip().upper()


def side_of(desc, p1, p2):
    h = handle_of(desc)
    return 1 if h == str(p1).strip().upper() else 2 if h == str(p2).strip().upper() else 0


def parse_range(txt: str):
    """'150-159' -> (150, 159); '200+' / '200 or more' -> (200, inf); 'Under 120' /
    '119 or less' -> (-inf, 119); 'Over 199.5' -> (200, inf); '7' -> (7, 7). None if unknown."""
    t = txt.strip().lower().replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+points?$", "", t)
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", t)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(\d+)\s*(\+|or more|and over|and above|or over|plus)", t)
    if m:
        return int(m.group(1)), np.inf
    m = re.fullmatch(r"(\d+)\s*(or less|or fewer|and under|or under|and below)", t)
    if m:
        return -np.inf, int(m.group(1))
    m = re.fullmatch(r"(?:over|more than|above)\s*(\d+(?:\.\d+)?)", t)
    if m:
        return int(np.floor(float(m.group(1)))) + 1, np.inf
    m = re.fullmatch(r"(?:under|less than|fewer than|below)\s*(\d+(?:\.\d+)?)", t)
    if m:
        return -np.inf, int(np.ceil(float(m.group(1)))) - 1
    m = re.fullmatch(r"(\d+)", t)
    if m:
        return int(m.group(1)), int(m.group(1))
    return None


def selection_rule(kind, desc, line, hcp, p1, p2):
    """Return a rule tuple describing when this selection wins, or None if unparseable.
    Rules are evaluated on (p1 points, p2 points) arrays by `wins`."""
    d = str(desc).strip()
    low = d.lower()
    if kind == "winner":
        if low in ("draw", "tie"):
            return ("draw",)
        s = side_of(d, p1, p2)
        return ("win", s) if s else None
    if kind == "handicap":
        s = side_of(d, p1, p2)
        if not s or pd.isna(hcp) or float(hcp) == int(float(hcp)):
            return None                           # integer handicaps can push: skipped
        return ("hcp", s, float(hcp))
    if kind in ("total", "p1_total", "p2_total"):
        if pd.isna(line) or float(line) == int(float(line)):
            return None
        who = {"total": 0, "p1_total": 1, "p2_total": 2}[kind]
        if low.startswith("over"):
            return ("over", who, float(line))
        if low.startswith("under"):
            return ("under", who, float(line))
        return None
    if kind in ("total_band", "p1_band", "p2_band"):
        r = parse_range(d)
        return ("band", {"total_band": 0, "p1_band": 1, "p2_band": 2}[kind], r[0], r[1]) if r else None
    if kind == "margin_band":
        if low in ("draw", "tie"):
            return ("draw",)
        m = re.match(r"^(.*?)\s+(?:by|to win by)\s+(.+)$", d, re.I)
        if not m:
            return None
        s, r = side_of(m.group(1), p1, p2), parse_range(m.group(2))
        return ("mband", s, r[0], r[1]) if s and r else None
    return None


def wins(rule, a1, a2):
    """Boolean array: does the selection win, given Player 1 / Player 2 points a1, a2."""
    k = rule[0]
    if k == "draw":
        return a1 == a2
    if k == "win":
        return (a1 > a2) if rule[1] == 1 else (a2 > a1)
    if k == "hcp":
        own, opp = (a1, a2) if rule[1] == 1 else (a2, a1)
        return own + rule[2] > opp
    if k in ("over", "under"):
        v = a1 + a2 if rule[1] == 0 else a1 if rule[1] == 1 else a2
        return v > rule[2] if k == "over" else v < rule[2]
    if k == "band":
        v = a1 + a2 if rule[1] == 0 else a1 if rule[1] == 1 else a2
        return (v >= rule[2]) & (v <= rule[3])
    if k == "mband":
        own, opp = (a1, a2) if rule[1] == 1 else (a2, a1)
        return (own - opp >= rule[2]) & (own - opp <= rule[3])
    raise ValueError(rule)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def pull_prices(session, table):
    names = ", ".join("'" + m.replace("'", "''") + "'" for m in MARKETS)
    sql = f"""
        SELECT p.MATCH_CODE, p.MARKET_TYPE_NAME, p.MARKET_LINE, p.SELECTION_DESCRIPTION,
               p.SELECTION_HANDICAP, p.PRICE_PROBABILITY, p.PRICE_ISSUE_TIME_UTC, n.START_T
        FROM {table} p
        JOIN (SELECT MATCH_CODE, SCHEDULED_START_TIME_UTC AS START_T FROM {OUTCOME_DB}.SHARED.EVENT
              WHERE SPORT_CODE = 'NB' AND SCHEDULED_START_TIME_UTC >= '{FROM:%Y-%m-%d}'
                AND SCHEDULED_START_TIME_UTC < DATEADD(day, 1, '{TO:%Y-%m-%d}'::timestamp)) n
          ON n.MATCH_CODE = p.MATCH_CODE
         AND p.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -2, '{FROM:%Y-%m-%d}'::timestamp)
        WHERE p.PRICE_ISSUE_TIME_UTC < n.START_T
          AND p.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -2, n.START_T)
          AND UPPER(p.SELECTION_STATUS) = 'ACTIVE'
          AND p.PRICE_PROBABILITY > 0 AND p.PRICE_PROBABILITY < 1
          AND p.MARKET_TYPE_NAME IN ({names})
        QUALIFY ROW_NUMBER() OVER (PARTITION BY p.MATCH_CODE, p.MARKET_TYPE_NAME, p.MARKET_LINE,
                p.SELECTION_DESCRIPTION, p.SELECTION_HANDICAP ORDER BY p.PRICE_ISSUE_TIME_UTC DESC) = 1"""
    df = session.sql(sql).to_pandas()
    df.columns = [c.upper() for c in df.columns]
    return df


def outcomes_table(df):
    """Final scores (incl. OT) by player side, and the regulation result."""
    o = pd.DataFrame({
        "MATCH_CODE": df["MATCH_CODE"].astype(str).to_numpy(),
        "day": df[bt.TIME].dt.floor("D").to_numpy(),
        "P1": df["PLAYER_1_HANDLE"].astype(str).to_numpy(), "P2": df["PLAYER_2_HANDLE"].astype(str).to_numpy(),
        "p1_pts": df["AWAY_POINTS"].to_numpy(float),   # PLAYER_1 = away in the match stats
        "p2_pts": df["HOME_POINTS"].to_numpy(float),
        "ot": (df["GAME_DURATION_SECONDS"].to_numpy(float) > REG_SECONDS).astype(int),
    })
    # regulation scores are not stored; a match that went to OT was level after regulation
    o["r1"] = np.where(o["ot"] == 1, 0.0, o["p1_pts"])
    o["r2"] = np.where(o["ot"] == 1, 0.0, o["p2_pts"])
    return o


# ---------------------------------------------------------------------------
# candidate simulation
# ---------------------------------------------------------------------------
def simulate(idata, train, test, cfg, rng):
    """SIM simulated pre-match games per test match: Player 1 / 2 final points
    (incl. OT) and regulation points, each (SIM, n)."""
    S_all = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    reps = int(np.ceil(SIM / S_all))
    out = {k: [] for k in ("f1", "f2", "r1", "r2")}
    for _ in range(reps):
        P = bt.predictors(idata, train, test, cfg, rng, n_draws=S_all)
        h, a = bt.draw_final(P, rng, REG_SECONDS)
        r_h, r_a = h.copy(), a.copy()
        for _ot in range(MAX_OT):
            tie = h == a
            if not tie.any():
                break
            oh, oa = bt.draw_final(P, rng, OT_SECONDS)
            h, a = np.where(tie, h + oh, h), np.where(tie, a + oa, a)
        tie = h == a                                  # still level after MAX_OT: coin flip
        coin = rng.random(h.shape) < 0.5
        h, a = h + (tie & coin), a + (tie & ~coin)
        out["f1"].append(a); out["f2"].append(h); out["r1"].append(r_a); out["r2"].append(r_h)
    return {k: np.concatenate(v, axis=0)[:SIM] for k, v in out.items()}


def candidate_rows(prices, test, sims, config):
    """Candidate probability for every source selection on this day's matches
    (sims columns are in the order of `test`)."""
    codes = test["MATCH_CODE"].astype(str).tolist()
    pos = {m: j for j, m in enumerate(codes)}
    p1s = dict(zip(codes, test["PLAYER_1_HANDLE"].astype(str)))
    p2s = dict(zip(codes, test["PLAYER_2_HANDLE"].astype(str)))
    rows = []
    sub = prices[prices["MATCH_CODE"].isin(pos)]
    cache = {}
    for r in sub.itertuples(index=False):
        kind, scope = MARKETS[r.MARKET_TYPE_NAME]
        rule = selection_rule(kind, r.SELECTION_DESCRIPTION, r.MARKET_LINE, r.SELECTION_HANDICAP,
                              p1s[r.MATCH_CODE], p2s[r.MATCH_CODE])
        if rule is None:
            continue
        j = pos[r.MATCH_CODE]
        key = (j, scope, rule)
        if key not in cache:
            a1 = sims["f1" if scope == "ot" else "r1"][:, j]
            a2 = sims["f2" if scope == "ot" else "r2"][:, j]
            cache[key] = float(wins(rule, a1, a2).mean())
        rows.append((r.SOURCE, r.MATCH_CODE, r.MARKET_TYPE_NAME, r.MARKET_LINE, r.SELECTION_DESCRIPTION,
                     r.SELECTION_HANDICAP, config, cache[key]))
    return pd.DataFrame(rows, columns=["SOURCE", "MATCH_CODE", "MARKET_TYPE_NAME", "MARKET_LINE",
                                       "SELECTION_DESCRIPTION", "SELECTION_HANDICAP", "config", "p_cand"])


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score_groups(prices, outc, cand):
    """One row per (source, match, market, line group, prob-provider) with log loss / Brier."""
    o = outc.set_index("MATCH_CODE")
    pr = prices[prices["MATCH_CODE"].isin(o.index)].copy()
    pr["kind"] = pr["MARKET_TYPE_NAME"].map(lambda m: MARKETS[m][0])
    pr["scope"] = pr["MARKET_TYPE_NAME"].map(lambda m: MARKETS[m][1])
    pr["group"] = np.where(pr["kind"].isin(LINE_KINDS),
                           pr["MARKET_LINE"].map(lambda v: "na" if pd.isna(v) else f"{float(v):g}"), "")
    # alternative-line markets repeat the main lines: keep each line once (main market first)
    pr["alt"] = pr["MARKET_TYPE_NAME"].str.contains("Alternative").astype(int)
    pr = pr.sort_values("alt", kind="stable").drop_duplicates(
        ["SOURCE", "MATCH_CODE", "kind", "scope", "group", "SELECTION_DESCRIPTION", "SELECTION_HANDICAP"])
    pr["KEY"] = sel_key(pr)
    pr["GROUP"] = pr["SOURCE"] + "|" + pr["MATCH_CODE"] + "|" + pr["MARKET_TYPE_NAME"] + "|" + pr["group"]
    rules, won, unparsed = [], [], []
    for r in pr.itertuples(index=False):
        oo = o.loc[r.MATCH_CODE]
        rule = selection_rule(r.kind, r.SELECTION_DESCRIPTION, r.MARKET_LINE, r.SELECTION_HANDICAP, oo.P1, oo.P2)
        rules.append(rule)
        if rule is None:
            unparsed.append((r.SOURCE, r.MARKET_TYPE_NAME, r.SELECTION_DESCRIPTION))
            won.append(np.nan)
            continue
        a1, a2 = (oo.p1_pts, oo.p2_pts) if r.scope == "ot" else (oo.r1, oo.r2)
        won.append(float(wins(rule, np.array([a1]), np.array([a2]))[0]))
    pr["won"] = won
    bad_groups = set(pr.loc[pr["won"].isna(), "GROUP"])          # any unparsed selection: drop the market
    pr = pr[~pr["GROUP"].isin(bad_groups)]
    providers = ["PRICE_PROBABILITY"]
    if len(cand):
        c2 = cand.assign(KEY=sel_key(cand), col=cand["config"].map(col_of))
        cw = c2.pivot_table(index="KEY", columns="col", values="p_cand", aggfunc="first")
        pr = pr.merge(cw, left_on="KEY", right_index=True, how="left")
        providers += list(cw.columns)
    out = []
    gk = ["SOURCE", "MATCH_CODE", "MARKET_TYPE_NAME", "group"]
    for (src, mc, mkt, grp), g in pr.groupby(gk, sort=False, dropna=False):
        if len(g) < 2 or g["won"].sum() != 1:
            continue                                   # one-sided, push, or band gap/overlap
        y = g["won"].to_numpy()
        for prov in providers:
            p = g[prov].to_numpy(float)
            if np.isnan(p).any():
                continue
            p = np.clip(p, 1.0 / (2 * SIM), None) if prov != "PRICE_PROBABILITY" else p
            p = p / p.sum()
            out.append((src, mc, mkt, grp, MARKETS[mkt][0], "source" if prov == "PRICE_PROBABILITY" else name_of(prov),
                        -np.log(p[y == 1][0]), ((p - y) ** 2).sum(), p[y == 1][0], len(g)))
    sc = pd.DataFrame(out, columns=["SOURCE", "MATCH_CODE", "MARKET_TYPE_NAME", "group", "kind", "provider",
                                    "logloss", "brier", "p_won", "n_sel"])
    sc = sc.merge(outc[["MATCH_CODE", "day"]], on="MATCH_CODE", how="left")
    un = pd.DataFrame(unparsed, columns=["SOURCE", "MARKET_TYPE_NAME", "SELECTION_DESCRIPTION"])
    return sc, pr, un, providers


def day_z(d: pd.DataFrame, col: str):
    """Sum of paired differences and its z from day blocks (days share fits)."""
    by = d.groupby("day")[col].sum()
    se = by.std(ddof=1) * np.sqrt(len(by)) if len(by) > 2 else np.nan
    return d[col].sum(), (d[col].sum() / se if se and se > 0 else np.nan), len(by)


def comparisons(sc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key = ["SOURCE", "MATCH_CODE", "MARKET_TYPE_NAME", "group"]
    wide = sc.pivot_table(index=key + ["kind", "day"], columns="provider", values="logloss").reset_index()
    cands = [c for c in wide.columns if c not in key + ["kind", "day", "source"]]
    # 1) each source vs each candidate, on that source's own selections
    for (src, kind), g in wide.groupby(["SOURCE", "kind"]):
        for c in cands:
            gg = g[g["source"].notna() & g[c].notna()].copy()
            if len(gg) < 20:
                continue
            gg["d"] = gg["source"] - gg[c]                 # > 0: candidate better
            tot, z, nd = day_z(gg, "d")
            rows.append({"comparison": f"{src} vs {c}", "kind": kind, "groups": len(gg), "days": nd,
                         "logloss_a": gg["source"].mean(), "logloss_b": gg[c].mean(),
                         "b_better_total": tot, "z_days": z})
    # 2) prod vs uat_b, on market lines both priced identically
    s = wide[wide["source"].notna()]
    pv = s.pivot_table(index=["MATCH_CODE", "MARKET_TYPE_NAME", "group", "kind", "day"], columns="SOURCE",
                       values="source").reset_index()
    if {"prod", "uat_b"} <= set(pv.columns):
        for kind, g in pv.groupby("kind"):
            gg = g[g["prod"].notna() & g["uat_b"].notna()].copy()
            if len(gg) < 20:
                continue
            gg["d"] = gg["prod"] - gg["uat_b"]             # > 0: uat_b better
            tot, z, nd = day_z(gg, "d")
            rows.append({"comparison": "prod vs uat_b", "kind": kind, "groups": len(gg), "days": nd,
                         "logloss_a": gg["prod"].mean(), "logloss_b": gg["uat_b"].mean(),
                         "b_better_total": tot, "z_days": z})
    return pd.DataFrame(rows)


def implied_medians(pr: pd.DataFrame, outc: pd.DataFrame, providers) -> pd.DataFrame:
    """Each provider's implied median of total / P1-P2 margin / player totals, by
    interpolating P(value > threshold) across its lines to 50%, vs the result."""
    o = outc.set_index("MATCH_CODE")
    pts = []
    for r in pr.itertuples(index=False):
        kind = MARKETS[r.MARKET_TYPE_NAME][0]
        if kind not in LINE_KINDS:
            continue
        oo = o.loc[r.MATCH_CODE]
        if kind == "handicap":
            s = side_of(r.SELECTION_DESCRIPTION, oo.P1, oo.P2)
            if not s:
                continue
            # side 1 with h wins iff d = p1 - p2 > -h ; side 2 with h wins iff d < h
            thr, up = (-float(r.SELECTION_HANDICAP), True) if s == 1 else (float(r.SELECTION_HANDICAP), False)
            target, actual = "margin_p1", oo.p1_pts - oo.p2_pts
        else:
            low = str(r.SELECTION_DESCRIPTION).lower()
            if not (low.startswith("over") or low.startswith("under")):
                continue
            thr, up = float(r.MARKET_LINE), low.startswith("over")
            target = {"total": "total", "p1_total": "p1_total", "p2_total": "p2_total"}[kind]
            actual = {"total": oo.p1_pts + oo.p2_pts, "p1_total": oo.p1_pts, "p2_total": oo.p2_pts}[kind]
        for prov in providers:
            p = getattr(r, prov, np.nan)
            if pd.isna(p):
                continue
            pts.append((r.SOURCE, r.MATCH_CODE, target, prov, thr, p if up else 1 - p, actual))
    d = pd.DataFrame(pts, columns=["SOURCE", "MATCH_CODE", "target", "provider", "thr", "p_above", "actual"])
    rows = []
    for (src, mc, tgt, prov), g in d.groupby(["SOURCE", "MATCH_CODE", "target", "provider"]):
        g = g.groupby("thr", as_index=False).agg(p_above=("p_above", "mean"), actual=("actual", "first"))
        g = g.sort_values("thr")
        x, y = g["thr"].to_numpy(), g["p_above"].to_numpy()
        if len(x) == 1:
            med = x[0]
        else:
            # P(above) falls with the threshold; interpolate where it crosses 0.5
            i = np.where((y[:-1] >= 0.5) & (y[1:] <= 0.5))[0]
            if len(i):
                k = i[0]
                med = x[k] if y[k] == y[k + 1] else x[k] + (y[k] - 0.5) * (x[k + 1] - x[k]) / (y[k] - y[k + 1])
            else:
                med = x[np.argmin(np.abs(y - 0.5))]
        rows.append((src, mc, tgt, "source" if prov == "PRICE_PROBABILITY" else name_of(prov), med, g["actual"].iat[0]))
    m = pd.DataFrame(rows, columns=["SOURCE", "MATCH_CODE", "target", "provider", "median", "actual"])
    m["abs_err"] = (m["median"] - m["actual"]).abs()
    return m.merge(outc[["MATCH_CODE", "day"]], on="MATCH_CODE", how="left")


def median_table(med: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wide = med.pivot_table(index=["SOURCE", "MATCH_CODE", "target", "day"], columns="provider",
                           values="abs_err").reset_index()
    cands = [c for c in wide.columns if c not in ("SOURCE", "MATCH_CODE", "target", "day", "source")]
    for (src, tgt), g in wide.groupby(["SOURCE", "target"]):
        for c in cands:
            gg = g[g["source"].notna() & g[c].notna()].copy()
            if len(gg) < 20:
                continue
            gg["d"] = gg["source"] - gg[c]                 # > 0: candidate's median closer
            tot, z, nd = day_z(gg, "d")
            rows.append({"source": src, "target": tgt, "candidate": c, "matches": len(gg), "days": nd,
                         "MAE_source": gg["source"].mean(), "MAE_candidate": gg[c].mean(),
                         "candidate_better_total": tot, "z_days": z})
    return pd.DataFrame(rows)


def calibration(sc: pd.DataFrame) -> pd.DataFrame:
    w = sc[sc["MARKET_TYPE_NAME"] == "Moneyline (including overtime)"]
    if not len(w):
        return pd.DataFrame()
    # probability that the eventual winner was given; fold into P(favourite) bins
    w = w.assign(bin=pd.cut(np.maximum(w["p_won"], 1 - w["p_won"]), [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                            include_lowest=True),
                 fav_won=(w["p_won"] >= 0.5).astype(float), p_fav=np.maximum(w["p_won"], 1 - w["p_won"]))
    return (w.groupby(["SOURCE", "provider", "bin"], observed=True)
             .agg(matches=("p_fav", "size"), mean_p_fav=("p_fav", "mean"), fav_win_rate=("fav_won", "mean"))
             .reset_index())


# ---------------------------------------------------------------------------
def main():
    out_dir = Path(os.environ.get("CMP_OUT") or f"compare_out_{FROM:%Y%m%d}_{TO:%Y%m%d}_train{TRAIN_DAYS}d")
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {"from": str(FROM), "to": str(TO), "train_days": TRAIN_DAYS, "sim": SIM,
                "chains": dd.CHAINS, "draws": dd.DRAWS, "tune": dd.TUNE, "stream": dd.STREAM_FALLBACK}
    spath = out_dir / "settings.json"
    if spath.exists():
        old = json.loads(spath.read_text())
        diff = {k: (old.get(k), v) for k, v in settings.items() if old.get(k) != v}
        if diff:
            sys.exit(f"{out_dir} holds a comparison with different settings {diff}. "
                     "Delete or rename it, or set $env:CMP_OUT to a new folder.")
    else:
        spath.write_text(json.dumps(settings, indent=1))
    print(f"Compare -> {out_dir.resolve()}\n  {FROM:%Y-%m-%d} .. {TO:%Y-%m-%d}; candidates {CONFIGS}; "
          f"train {TRAIN_DAYS} d; {SIM} simulated games per match")

    session = dd.connect(OUTCOME_DB)
    # prices (cached)
    prices = []
    for src, table in SOURCES.items():
        cpath = out_dir / f"prices_{src}.csv"
        if cpath.exists():
            p = pd.read_csv(cpath)
        else:
            print(f"  pulling last active pre-match prices from {table} ...")
            p = pull_prices(session, table)
            p.to_csv(cpath, index=False)
        p["SOURCE"] = src
        print(f"  {src}: {p['MATCH_CODE'].nunique()} matches, {len(p)} selections")
        prices.append(p)
    prices = pd.concat(prices, ignore_index=True)
    for c in ("MARKET_LINE", "SELECTION_HANDICAP", "PRICE_PROBABILITY"):
        prices[c] = pd.to_numeric(prices[c], errors="coerce")
    prices["MATCH_CODE"] = prices["MATCH_CODE"].astype(str)

    # outcomes and training data (PROD match stats)
    df = dd.load_match_features(asof=f"{TO + pd.Timedelta(1, unit='D'):%Y-%m-%d}", lookback_days=None)
    df = df.sort_values(bt.TIME).reset_index(drop=True)
    outc = outcomes_table(df)
    outc = outc[(outc["day"] >= FROM) & (outc["day"] <= TO)].reset_index(drop=True)
    outc.to_csv(out_dir / "outcomes.csv", index=False)
    print(f"  {len(outc)} matches with results in the window")

    # rolling candidate fits, one per day and config (resumable)
    cpath = out_dir / "candidate_probs.csv"
    done = pd.read_csv(cpath, usecols=["day", "config"]).drop_duplicates() if cpath.exists() else pd.DataFrame(columns=["day", "config"])
    done_keys = set(zip(done["day"].astype(str), done["config"].astype(str)))
    rng = np.random.default_rng(2026)
    for day in pd.date_range(FROM, TO, freq="D"):
        test = df[(df[bt.TIME] >= day) & (df[bt.TIME] < day + pd.Timedelta(1, unit="D"))]
        train = df[(df[bt.TIME] >= day - pd.Timedelta(TRAIN_DAYS, unit="D")) & (df[bt.TIME] < day)]
        if len(test) == 0 or len(train) < 200:
            print(f"{day:%Y-%m-%d}: skipped (train {len(train)}, test {len(test)})")
            continue
        for c in CONFIGS:
            if (str(day.date()), c) in done_keys:
                continue
            dmode, _, noise = c.partition(":")
            cfg = dd.FitConfig(defence_mode=dmode, dispersion_prior=noise, stream_fallback=dd.STREAM_FALLBACK,
                               chem_mode="off", chem_on_ppp=False, chem_on_pace=False, match_effect=False,
                               stream_effect=True, session_effect=False,
                               chains=dd.CHAINS, draws=dd.DRAWS, tune=dd.TUNE)
            print(f"\n=== {day:%Y-%m-%d} config {c}: train {len(train)} / test {len(test)} matches")
            idata = dd.fit_model(train, cfg)
            sims = simulate(idata, train, test.reset_index(drop=True), cfg, rng)
            rows = candidate_rows(prices, test.reset_index(drop=True), sims, c)
            rows.insert(0, "day", str(day.date()))
            rows.to_csv(cpath, mode="a", header=not cpath.exists(), index=False)
            print(f"    priced {len(rows)} selections; divergences {int(np.asarray(idata.sample_stats['diverging']).sum())}")

    cand = pd.read_csv(cpath) if cpath.exists() else pd.DataFrame(
        columns=["day", "SOURCE", "MATCH_CODE", "MARKET_TYPE_NAME", "MARKET_LINE", "SELECTION_DESCRIPTION",
                 "SELECTION_HANDICAP", "config", "p_cand"])
    cand["MATCH_CODE"] = cand["MATCH_CODE"].astype(str)
    for c in ("MARKET_LINE", "SELECTION_HANDICAP"):
        cand[c] = pd.to_numeric(cand[c], errors="coerce")

    sc, pr, un, providers = score_groups(prices, outc, cand)
    sc.to_csv(out_dir / "group_scores.csv", index=False)
    if len(un):
        un.value_counts().rename("n").reset_index().to_csv(out_dir / "unparsed_selections.csv", index=False)
    summ = (sc.groupby(["SOURCE", "kind", "provider"])
              .agg(groups=("logloss", "size"), logloss=("logloss", "mean"), brier=("brier", "mean"))
              .reset_index())
    summ.to_csv(out_dir / "summary.csv", index=False)
    comp = comparisons(sc)
    comp.to_csv(out_dir / "comparisons.csv", index=False)
    med = implied_medians(pr, outc, providers)
    med.to_csv(out_dir / "medians.csv", index=False)
    mt = median_table(med)
    mt.to_csv(out_dir / "median_comparisons.csv", index=False)
    cal = calibration(sc)
    cal.to_csv(out_dir / "moneyline_calibration.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print("\n" + "=" * 70 + "\nLOG LOSS BY SOURCE, MARKET KIND AND PROVIDER ('source' = that source's own price)")
        print(summ.to_string(index=False))
        print("\nPAIRED COMPARISONS (b_better_total > 0 means the second is better; |z_days| > 2.5 is clear)")
        print(comp.to_string(index=False))
        print("\nIMPLIED MEDIANS vs RESULT (MAE in points; candidate_better_total > 0 = candidate closer)")
        print(mt.to_string(index=False))
        print("\nMONEYLINE CALIBRATION (favourite probability bins)")
        print(cal.to_string(index=False))
    if len(un):
        print(f"\nUNPARSED selections ({len(un)} rows, see unparsed_selections.csv); most common:")
        print(un.value_counts().head(25).to_string())
    print(f"\nWritten to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
