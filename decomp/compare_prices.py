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
Band markets must be contiguous (no gaps or overlaps) but may be truncated at the
ends -- books leave out bands nobody reaches -- and are then scored conditional
on the offered range (band_coverage.csv counts results that fell outside).

Scores per market group (one market line of one match): log loss and Brier of
the realised selection; paired source-vs-candidate and prod-vs-UAT B
differences with day-block z (days share fits, so they are the honest unit);
each source's implied MEDIAN total / margin / player total (where its lines
cross 50%) against the result; and a moneyline calibration table.

Read-only on Snowflake (SELECT only). Writes compare_out_<from>_<to>_train<n>d\\
and resumes there (settings.json guard); once every fit is done, a re-run only
re-scores and does not log in.  Usage: python -u decomp\\compare_prices.py
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
TODAY = pd.Timestamp(datetime.now(timezone.utc)).floor("D")
TO = (pd.Timestamp(_env("CMP_TO", ""), tz="UTC").floor("D") if _env("CMP_TO", "")
      else TODAY - pd.Timedelta(1, unit="D"))                       # last complete UTC day
CONFIGS = [c for c in re.split(r"[,\s]+", _env("CMP_CONFIGS", "off:free,on:free").lower()) if c]
TRAIN_DAYS = int(_env("CMP_TRAIN_DAYS", "7"))
SIM = int(_env("CMP_SIM", "10000"))
CUTOFF_MIN = float(_env("CMP_CUTOFF_MIN", "2"))     # prices must be this many minutes before the start
SOURCE_FLOOR = 1e-5                                   # the feed's 5-decimal resolution
STALE_TOL_S = 2.0                                     # a line is current if updated within 2 s of the market's newest
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
if TO >= TODAY:
    sys.exit(f"CMP_TO {TO:%Y-%m-%d} is today or later: use the last complete UTC day "
             f"({TODAY - pd.Timedelta(1, unit='D'):%Y-%m-%d}) so prices and results are final")

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
BAND_KINDS = {"total_band", "p1_band", "p2_band", "margin_band"}


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
    """Each selection's last ACTIVE price issued at least CUTOFF_MIN minutes before the
    match start, where start = the earlier of the scheduled and the actual start
    (period 1 actual start from PERIOD_SCORE_PRICE_CHANGES) -- some matches start early,
    and prices after that are in-play."""
    names = ", ".join("'" + m.replace("'", "''") + "'" for m in MARKETS)
    lo, hi = f"'{FROM:%Y-%m-%d}'::timestamp", f"'{TO:%Y-%m-%d}'::timestamp"
    sql = f"""
        WITH n0 AS (
            SELECT MATCH_CODE, SCHEDULED_START_TIME_UTC AS SCHED FROM {OUTCOME_DB}.SHARED.EVENT
            WHERE SPORT_CODE = 'NB' AND SCHEDULED_START_TIME_UTC >= {lo}
              AND SCHEDULED_START_TIME_UTC < DATEADD(day, 1, {hi})),
        a AS (
            SELECT s.MATCH_CODE, MIN(s.PERIOD_ACTUAL_START_TIME) AS ACTUAL
            FROM {OUTCOME_DB}.SHARED.PERIOD_SCORE_PRICE_CHANGES s JOIN n0 ON n0.MATCH_CODE = s.MATCH_CODE
            WHERE s.PERIOD_NUMBER = 1 AND s.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -1, {lo})
              AND s.PRICE_ISSUE_TIME_UTC < DATEADD(day, 2, {hi})
            GROUP BY 1),
        n AS (
            SELECT n0.MATCH_CODE, n0.SCHED, a.ACTUAL,
                   LEAST(n0.SCHED, COALESCE(a.ACTUAL, n0.SCHED)) AS START_T
            FROM n0 LEFT JOIN a ON a.MATCH_CODE = n0.MATCH_CODE)
        SELECT p.MATCH_CODE, p.MARKET_TYPE_NAME, p.MARKET_LINE, p.SELECTION_DESCRIPTION,
               p.SELECTION_HANDICAP, p.PRICE_PROBABILITY, p.PRICE_ISSUE_TIME_UTC,
               n.START_T, n.SCHED, n.ACTUAL
        FROM {table} p
        JOIN n ON n.MATCH_CODE = p.MATCH_CODE
              AND p.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -2, {lo})
              AND p.PRICE_ISSUE_TIME_UTC < DATEADD(day, 1, {hi})
        WHERE p.PRICE_ISSUE_TIME_UTC < DATEADD(second, -{int(CUTOFF_MIN * 60)}, n.START_T)
          AND p.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -2, n.START_T)
          AND UPPER(p.SELECTION_STATUS) = 'ACTIVE'
          AND p.PRICE_PROBABILITY >= 0 AND p.PRICE_PROBABILITY <= 1
          AND p.MARKET_TYPE_NAME IN ({names})
        QUALIFY ROW_NUMBER() OVER (PARTITION BY p.MATCH_CODE, p.MARKET_TYPE_NAME, p.MARKET_LINE,
                p.SELECTION_DESCRIPTION, p.SELECTION_HANDICAP ORDER BY p.PRICE_ISSUE_TIME_UTC DESC) = 1"""
    df = session.sql(sql).to_pandas()
    df.columns = [c.upper() for c in df.columns]
    return df


def drop_stale_lines(prices: pd.DataFrame) -> pd.DataFrame:
    """The feed is a change log: when the main or alternative lines move, the old lines
    keep their last Active row. Keep only lines updated within STALE_TOL_S of the newest
    update of their market (per source, match and market); print what was dropped."""
    p = prices.copy()
    kind = p["MARKET_TYPE_NAME"].map(lambda m: MARKETS[m][0])
    t = pd.to_datetime(p["PRICE_ISSUE_TIME_UTC"])
    g = [p["SOURCE"], p["MATCH_CODE"], p["MARKET_TYPE_NAME"]]
    line_t = t.groupby(g + [p["MARKET_LINE"].fillna(-1e9)]).transform("max")
    mkt_t = t.groupby(g).transform("max")
    stale = kind.isin(LINE_KINDS) & ((mkt_t - line_t).dt.total_seconds() > STALE_TOL_S)
    if stale.any():
        rep = (p[stale].groupby(["SOURCE", "MARKET_TYPE_NAME"]).size()
               .rename("stale_selections_dropped").reset_index())
        print("  dropped superseded lines (old main/alternative lines still marked Active):")
        print(rep.to_string(index=False))
    return p[~stale]


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


def half_point_median(x: np.ndarray) -> float:
    """Median on the half-point line scale used by the books: interpolate P(X > k + 0.5)
    across half-points to 50% (same rule as the sources' lines)."""
    k = np.arange(np.floor(np.quantile(x, 0.2)) - 1, np.ceil(np.quantile(x, 0.8)) + 1) + 0.5
    pa = (x[:, None] > k[None, :]).mean(0)
    i = np.where((pa[:-1] >= 0.5) & (pa[1:] <= 0.5))[0]
    if not len(i):
        return float(np.median(x))
    j = i[0]
    return float(k[j] if pa[j] == pa[j + 1] else k[j] + (pa[j] - 0.5) / (pa[j] - pa[j + 1]))


def candidate_medians(test, sims, config):
    rows = []
    for j, mc in enumerate(test["MATCH_CODE"].astype(str)):
        f1, f2 = sims["f1"][:, j], sims["f2"][:, j]
        rows.append((mc, config, half_point_median(f1 + f2), half_point_median(f1 - f2),
                     half_point_median(f1), half_point_median(f2)))
    return pd.DataFrame(rows, columns=["MATCH_CODE", "config", "total", "margin_p1", "p1_total", "p2_total"])


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
def band_intervals(rules):
    """Each band as an inclusive interval of what it settles on: total / player points
    for 'band', the Player 1 margin (p1 - p2) for 'mband' and 'draw'. Sorted."""
    iv = []
    for r in rules:
        if r[0] == "band":
            iv.append((r[2], r[3]))
        elif r[0] == "mband":
            iv.append((r[2], r[3]) if r[1] == 1 else (-r[3], -r[2]))
        elif r[0] == "draw":
            iv.append((0, 0))
    return sorted(iv)


def band_layout(rules, scope):
    """None if the bands are contiguous with no gaps or overlaps, else the reason. The
    ends may be missing (books leave out bands nobody reaches, e.g. 230+ when totals run
    60-170): such a market is scored conditional on the offered range. A final margin
    cannot be 0, so incl.-overtime margin bands may skip 0."""
    iv = band_intervals(rules)
    margin = any(r[0] == "mband" for r in rules)
    for (lo0, hi0), (lo1, hi1) in zip(iv, iv[1:]):
        if lo1 == hi0 + 1 or (margin and scope == "ot" and hi0 == -1 and lo1 == 1):
            continue
        return "bands overlap" if lo1 <= hi0 else "bands have gaps"
    return None


def band_truncated(rules) -> bool:
    """True if the offered bands leave out possible results at either end."""
    iv = band_intervals(rules)
    margin = any(r[0] == "mband" for r in rules)
    return iv[-1][1] != np.inf or iv[0][0] > (-np.inf if margin else 0)


def score_groups(prices, outc, cand):
    """One row per (source, match, market, line group, provider) with log loss / Brier,
    plus a table of dropped groups by reason and a table of unparsed selections."""
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
            unparsed.append((r.SOURCE, r.MARKET_TYPE_NAME, r.SELECTION_DESCRIPTION, "unparsed"))
            won.append(np.nan)
            continue
        a1, a2 = (oo.p1_pts, oo.p2_pts) if r.scope == "ot" else (oo.r1, oo.r2)
        won.append(float(wins(rule, np.array([a1]), np.array([a2]))[0]))
    pr["rule"], pr["won"] = rules, won
    providers = ["PRICE_PROBABILITY"]
    if len(cand):
        c2 = cand.assign(KEY=sel_key(cand), col=cand["config"].map(col_of))
        cw = c2.pivot_table(index="KEY", columns="col", values="p_cand", aggfunc="first")
        pr = pr.merge(cw, left_on="KEY", right_index=True, how="left")
        providers += list(cw.columns)
    out, dropped, cover = [], [], []
    gk = ["SOURCE", "MATCH_CODE", "MARKET_TYPE_NAME", "group"]
    for (src, mc, mkt, grp), g in pr.groupby(gk, sort=False, dropna=False):
        kind = MARKETS[mkt][0]
        reason, layout = None, ""
        raw = g["PRICE_PROBABILITY"].to_numpy(float)
        if g["won"].isna().any():
            reason = "unparsed selection"
        elif len(g) < 2:
            reason = "one-sided"
        elif kind in BAND_KINDS:
            reason = band_layout(list(g["rule"]), MARKETS[mkt][1])
            if reason:
                unparsed += [(src, mkt, d, "band layout") for d in g["SELECTION_DESCRIPTION"]]
            else:
                # the offered bands (prod and UAT B are only paired on identical layouts) and the
                # probability each provider leaves outside them
                layout = ";".join(f"{lo:g}..{hi:g}" for lo, hi in band_intervals(list(g["rule"])))
                trunc = band_truncated(list(g["rule"]))
                cover.append([src, mc, mkt, trunc, trunc and g["won"].sum() == 0, raw.sum()]
                             + [g[p].sum(min_count=len(g)) for p in providers[1:]])
                if trunc and g["won"].sum() == 0:
                    reason = "result outside offered bands"
        if reason is None and g["won"].sum() == 0:
            reason = "no winning selection"
        elif reason is None and g["won"].sum() > 1:
            reason = "several winning selections"
        if reason:
            dropped.append((src, mkt, reason))
            continue
        y = g["won"].to_numpy()
        for prov in providers:
            p = g[prov].to_numpy(float)
            if np.isnan(p).any():
                continue
            floor = SOURCE_FLOOR if prov == "PRICE_PROBABILITY" else 1.0 / (2 * SIM)
            p = np.clip(p, floor, None)
            p = p / p.sum()
            out.append((src, mc, mkt, grp, kind, MARKETS[mkt][1], layout,
                        "source" if prov == "PRICE_PROBABILITY" else name_of(prov),
                        -np.log(p[y == 1][0]), ((p - y) ** 2).sum(), p[y == 1][0], len(g), raw.sum()))
    sc = pd.DataFrame(out, columns=["SOURCE", "MATCH_CODE", "MARKET_TYPE_NAME", "group", "kind", "scope", "layout",
                                    "provider", "logloss", "brier", "p_won", "n_sel", "raw_sum"])
    sc = sc.merge(outc[["MATCH_CODE", "day"]], on="MATCH_CODE", how="left")
    un = pd.DataFrame(unparsed, columns=["SOURCE", "MARKET_TYPE_NAME", "SELECTION_DESCRIPTION", "why"])
    dr = (pd.DataFrame(dropped, columns=["SOURCE", "MARKET_TYPE_NAME", "reason"]).value_counts()
          .rename("groups").reset_index() if dropped else pd.DataFrame(columns=["SOURCE", "MARKET_TYPE_NAME", "reason", "groups"]))
    cv = pd.DataFrame(cover, columns=["SOURCE", "MATCH_CODE", "MARKET_TYPE_NAME", "truncated", "outside", "source"]
                      + [name_of(p) for p in providers[1:]])
    return sc, pr, un, dr, cv, providers


def band_coverage(cv: pd.DataFrame) -> pd.DataFrame:
    """Band markets with contiguous bands: how often the offered range was truncated, how
    often the result fell outside it (those groups are not scored), each source's mean raw
    sum, and on the truncated groups the mean probability each candidate put outside the
    offered bands (the books' fair prices are normalised over what they offer)."""
    if not len(cv):
        return pd.DataFrame()
    cands = [c for c in cv.columns if c not in ("SOURCE", "MATCH_CODE", "MARKET_TYPE_NAME", "truncated", "outside", "source")]
    t = cv.assign(**{c: (1 - cv[c]).where(cv["truncated"]) for c in cands})
    return (t.groupby(["SOURCE", "MARKET_TYPE_NAME"])
             .agg(groups=("outside", "size"), truncated=("truncated", "sum"), result_outside=("outside", "sum"),
                  source_raw_sum=("source", "mean"), **{f"p_outside_{c}": (c, "mean") for c in cands})
             .reset_index())


def is_new_week(day) -> bool:
    """A Monday test day starts a new ISO week: its stream-week levels are unseen in training."""
    return pd.Timestamp(day).dayofweek == 0


def day_z(d: pd.DataFrame, col: str):
    """Sum of paired differences and its z from day blocks (days share fits)."""
    by = d.groupby("day")[col].sum()
    se = by.std(ddof=1) * np.sqrt(len(by)) if len(by) > 2 else np.nan
    return d[col].sum(), (d[col].sum() / se if se and se > 0 else np.nan), len(by)


def comparisons(sc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key = ["SOURCE", "MATCH_CODE", "MARKET_TYPE_NAME", "group"]
    wide = sc.pivot_table(index=key + ["kind", "scope", "layout", "day"], columns="provider", values="logloss").reset_index()
    cands = [c for c in wide.columns if c not in key + ["kind", "scope", "layout", "day", "source"]]
    # 1) each source vs each candidate, on that source's own selections
    for (src, kind), g in wide.groupby(["SOURCE", "kind"]):
        for c in cands:
            gg = g[g["source"].notna() & g[c].notna()].copy()
            if len(gg) < 20:
                continue
            gg["d"] = gg["source"] - gg[c]                 # > 0: candidate better
            tot, z, nd = day_z(gg, "d")
            mon = gg["day"].map(is_new_week)
            rows.append({"comparison": f"{src} vs {c}", "kind": kind, "groups": len(gg), "days": nd,
                         "logloss_a": gg["source"].mean(), "logloss_b": gg[c].mean(),
                         "b_better_total": tot, "z_days": z,
                         "b_better_mondays": gg.loc[mon, "d"].sum(), "b_better_other_days": gg.loc[~mon, "d"].sum()})
    # 2) prod vs uat_b, on market lines both priced identically
    s = wide[wide["source"].notna()]
    # same market, same line (an alternative line is not paired with the other's main line) and,
    # for bands, the same offered bands: truncated bands are scored conditional on their range
    pv = s.pivot_table(index=["MATCH_CODE", "MARKET_TYPE_NAME", "kind", "scope", "group", "layout", "day"], columns="SOURCE",
                       values="source", aggfunc="mean").reset_index()
    if {"prod", "uat_b"} <= set(pv.columns):
        for kind, g in pv.groupby("kind"):
            gg = g[g["prod"].notna() & g["uat_b"].notna()].copy()
            if len(gg) < 20:
                continue
            gg["d"] = gg["prod"] - gg["uat_b"]             # > 0: uat_b better
            tot, z, nd = day_z(gg, "d")
            mon = gg["day"].map(is_new_week)
            rows.append({"comparison": "prod vs uat_b", "kind": kind, "groups": len(gg), "days": nd,
                         "logloss_a": gg["prod"].mean(), "logloss_b": gg["uat_b"].mean(),
                         "b_better_total": tot, "z_days": z,
                         "b_better_mondays": gg.loc[mon, "d"].sum(), "b_better_other_days": gg.loc[~mon, "d"].sum()})
    return pd.DataFrame(rows)


def source_medians(pr: pd.DataFrame, outc: pd.DataFrame) -> pd.DataFrame:
    """Each source's implied median total / Player-1 margin / player totals from its current
    lines: interpolate P(value > threshold) to 50% where two lines bracket it; with one line,
    use it only if priced within 0.45-0.55 (a main line); otherwise no median (counted)."""
    o = outc.set_index("MATCH_CODE")
    pts = []
    for r in pr.itertuples(index=False):
        kind = MARKETS[r.MARKET_TYPE_NAME][0]
        if kind not in LINE_KINDS or pd.isna(r.won):
            continue
        oo = o.loc[r.MATCH_CODE]
        if kind == "handicap":
            sd = side_of(r.SELECTION_DESCRIPTION, oo.P1, oo.P2)
            if not sd:
                continue
            # side 1 with h wins iff d = p1 - p2 > -h ; side 2 with h wins iff d < h
            thr, up = (-float(r.SELECTION_HANDICAP), True) if sd == 1 else (float(r.SELECTION_HANDICAP), False)
            target, actual = "margin_p1", oo.p1_pts - oo.p2_pts
        else:
            low = str(r.SELECTION_DESCRIPTION).lower()
            if not (low.startswith("over") or low.startswith("under")):
                continue
            thr, up, target = float(r.MARKET_LINE), low.startswith("over"), kind
            actual = {"total": oo.p1_pts + oo.p2_pts, "p1_total": oo.p1_pts, "p2_total": oo.p2_pts}[kind]
        pts.append((r.SOURCE, r.MATCH_CODE, target, thr, r.PRICE_PROBABILITY if up else 1 - r.PRICE_PROBABILITY, actual))
    d = pd.DataFrame(pts, columns=["SOURCE", "MATCH_CODE", "target", "thr", "p_above", "actual"])
    rows = []
    for (src, mc, tgt), g in d.groupby(["SOURCE", "MATCH_CODE", "target"]):
        g = g.groupby("thr", as_index=False).agg(p_above=("p_above", "mean"), actual=("actual", "first")).sort_values("thr")
        x, y = g["thr"].to_numpy(), g["p_above"].to_numpy()
        i = np.where((y[:-1] >= 0.5) & (y[1:] <= 0.5))[0] if len(x) > 1 else []
        if len(i):
            k = i[0]
            med, how = (x[k] if y[k] == y[k + 1] else x[k] + (y[k] - 0.5) * (x[k + 1] - x[k]) / (y[k] - y[k + 1])), "bracketed"
        else:
            j = int(np.argmin(np.abs(y - 0.5)))
            med, how = (x[j], "main line") if abs(y[j] - 0.5) <= 0.05 else (np.nan, "no 50% line")
        rows.append((src, mc, tgt, med, how, g["actual"].iat[0]))
    return pd.DataFrame(rows, columns=["SOURCE", "MATCH_CODE", "target", "median", "how", "actual"])


def median_table(smed: pd.DataFrame, cmed: pd.DataFrame, outc: pd.DataFrame) -> pd.DataFrame:
    """MAE of medians vs the result: each source against each candidate on the same
    matches, and prod against UAT B (line placement) on matches both priced."""
    s = smed.dropna(subset=["median"]).merge(outc[["MATCH_CODE", "day"]], on="MATCH_CODE")
    s["err"] = (s["median"] - s["actual"]).abs()
    rows = []
    if len(cmed):
        c = cmed.melt(id_vars=["MATCH_CODE", "config"], var_name="target", value_name="cmedian")
        m = s.merge(c, on=["MATCH_CODE", "target"])
        m["cerr"] = (m["cmedian"] - m["actual"]).abs()
        m["d"] = m["err"] - m["cerr"]                     # > 0: candidate closer
        for (src, tgt, cfg), g in m.groupby(["SOURCE", "target", "config"]):
            if len(g) < 20:
                continue
            tot, z, nd = day_z(g, "d")
            rows.append({"comparison": f"{src} vs {cfg}", "target": tgt, "matches": len(g), "days": nd,
                         "MAE_a": g["err"].mean(), "MAE_b": g["cerr"].mean(), "b_closer_total": tot, "z_days": z})
    w = s.pivot_table(index=["MATCH_CODE", "target", "day"], columns="SOURCE", values="err").reset_index()
    if {"prod", "uat_b"} <= set(w.columns):
        for tgt, g in w.groupby("target"):
            g = g.dropna(subset=["prod", "uat_b"]).copy()
            if len(g) < 20:
                continue
            g["d"] = g["prod"] - g["uat_b"]                # > 0: uat_b closer
            tot, z, nd = day_z(g, "d")
            rows.append({"comparison": "prod vs uat_b", "target": tgt, "matches": len(g), "days": nd,
                         "MAE_a": g["prod"].mean(), "MAE_b": g["uat_b"].mean(), "b_closer_total": tot, "z_days": z})
    return pd.DataFrame(rows)


def calibration(sc: pd.DataFrame) -> pd.DataFrame:
    w = sc[sc["MARKET_TYPE_NAME"] == "Moneyline (including overtime)"]
    if not len(w):
        return pd.DataFrame()
    # probability that the eventual winner was given; fold into P(favourite) bins
    w = w.assign(bin=pd.cut(np.maximum(w["p_won"], 1 - w["p_won"]), [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                            include_lowest=True),
                 fav_won=np.where(w["p_won"] == 0.5, 0.5, (w["p_won"] > 0.5).astype(float)),   # 50/50: half
                 p_fav=np.maximum(w["p_won"], 1 - w["p_won"]))
    return (w.groupby(["SOURCE", "provider", "bin"], observed=True)
             .agg(matches=("p_fav", "size"), mean_p_fav=("p_fav", "mean"), fav_win_rate=("fav_won", "mean"))
             .reset_index())


# ---------------------------------------------------------------------------
def main():
    out_dir = Path(os.environ.get("CMP_OUT") or f"compare_out_{FROM:%Y%m%d}_{TO:%Y%m%d}_train{TRAIN_DAYS}d")
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {"from": str(FROM), "to": str(TO), "train_days": TRAIN_DAYS, "sim": SIM, "cutoff_min": CUTOFF_MIN,
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

    session = None                         # connect only if something is not cached

    def connected():
        nonlocal session
        if session is None:
            session = dd.connect(OUTCOME_DB)
        return session

    # prices (cached)
    prices = []
    for src, table in SOURCES.items():
        cpath = out_dir / f"prices_{src}.csv"
        if cpath.exists():
            p = pd.read_csv(cpath)
        else:
            print(f"  pulling last active pre-match prices from {table} ...")
            p = pull_prices(connected(), table)
            p.to_csv(cpath, index=False)
        p["SOURCE"] = src
        print(f"  {src}: {p['MATCH_CODE'].nunique()} matches, {len(p)} selections")
        prices.append(p)
    prices = pd.concat(prices, ignore_index=True)
    for c in ("MARKET_LINE", "SELECTION_HANDICAP", "PRICE_PROBABILITY"):
        prices[c] = pd.to_numeric(prices[c], errors="coerce")
    prices["MATCH_CODE"] = prices["MATCH_CODE"].astype(str)
    for c in ("PRICE_ISSUE_TIME_UTC", "START_T", "SCHED", "ACTUAL"):
        if c in prices.columns:
            prices[c] = pd.to_datetime(prices[c], errors="coerce")
    # start-time and timing diagnostics (in-play prices must not leak in)
    mt0 = prices.drop_duplicates(["SOURCE", "MATCH_CODE"])
    if "ACTUAL" in mt0.columns:
        early = (mt0["ACTUAL"] < mt0["SCHED"])
        mins = (mt0.loc[early, "SCHED"] - mt0.loc[early, "ACTUAL"]).dt.total_seconds() / 60
        print(f"  matches that started before their scheduled time: {int(early.sum())} of {len(mt0)} "
              f"(median {mins.median() if len(mins) else 0:.1f} min early); cut-off = earlier start - {CUTOFF_MIN:g} min")
    lead = (prices["START_T"] - prices["PRICE_ISSUE_TIME_UTC"]).dt.total_seconds() / 60
    ml = prices["MARKET_TYPE_NAME"] == "Moneyline (including overtime)"
    print(f"  last prices: median {lead.median():.0f} min before start; {int((lead < 10).sum())} selections within 10 min; "
          f"moneyline probabilities >= 0.98 or <= 0.02: {int((ml & ((prices['PRICE_PROBABILITY'] >= 0.98) | (prices['PRICE_PROBABILITY'] <= 0.02))).sum())}")
    prices = drop_stale_lines(prices)

    # rolling candidate fits, one per day and config (resumable)
    cpath, opath = out_dir / "candidate_probs.csv", out_dir / "outcomes.csv"
    fpath = out_dir / "fits_done.csv"      # every (day, config) handled, incl. skipped days and empty fits
    done_keys = set()
    for p in (cpath, fpath):
        if p.exists():
            done = pd.read_csv(p, usecols=["day", "config"], dtype=str).drop_duplicates()
            done_keys |= set(zip(done["day"], done["config"]))

    def mark(day, configs, status):
        pd.DataFrame({"day": str(day.date()), "config": configs, "status": status}).to_csv(
            fpath, mode="a", header=not fpath.exists(), index=False)

    need_fits = True
    if opath.exists():                     # results are final (CMP_TO < today): reuse them
        outc = pd.read_csv(opath, dtype={"MATCH_CODE": str, "P1": str, "P2": str}, keep_default_na=False)
        outc["day"] = pd.to_datetime(outc["day"], utc=True)
        need_fits = any((f"{d:%Y-%m-%d}", c) not in done_keys for d in outc["day"].unique() for c in CONFIGS)
    if not need_fits:
        print(f"  {len(outc)} matches with results (cached); every fit is done: re-scoring only, no login needed")
    else:
        # outcomes and training data (PROD match stats)
        connected()
        df = dd.load_match_features(asof=f"{TO + pd.Timedelta(1, unit='D'):%Y-%m-%d}", lookback_days=None)
        df = df.sort_values(bt.TIME).reset_index(drop=True)
        outc = outcomes_table(df)
        outc = outc[(outc["day"] >= FROM) & (outc["day"] <= TO)].reset_index(drop=True)
        outc.to_csv(opath, index=False)
        print(f"  {len(outc)} matches with results in the window")
    rng = np.random.default_rng(2026)
    for day in (pd.date_range(FROM, TO, freq="D") if need_fits else []):
        test = df[(df[bt.TIME] >= day) & (df[bt.TIME] < day + pd.Timedelta(1, unit="D"))]
        train = df[(df[bt.TIME] >= day - pd.Timedelta(TRAIN_DAYS, unit="D")) & (df[bt.TIME] < day)]
        if len(test) == 0 or len(train) < 200:
            print(f"{day:%Y-%m-%d}: skipped (train {len(train)}, test {len(test)})")
            mark(day, [c for c in CONFIGS if (str(day.date()), c) not in done_keys], "skipped")
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
            meds = candidate_medians(test.reset_index(drop=True), sims, c)
            meds.insert(0, "day", str(day.date()))
            mpath = out_dir / "candidate_medians.csv"
            meds.to_csv(mpath, mode="a", header=not mpath.exists(), index=False)
            rows.to_csv(cpath, mode="a", header=not cpath.exists(), index=False)   # marks the pair done
            mark(day, [c], "fitted")                                                # ... even if it priced nothing
            print(f"    priced {len(rows)} selections; divergences {int(np.asarray(idata.sample_stats['diverging']).sum())}")

    cand = pd.read_csv(cpath) if cpath.exists() else pd.DataFrame(
        columns=["day", "SOURCE", "MATCH_CODE", "MARKET_TYPE_NAME", "MARKET_LINE", "SELECTION_DESCRIPTION",
                 "SELECTION_HANDICAP", "config", "p_cand"])
    cand["MATCH_CODE"] = cand["MATCH_CODE"].astype(str)
    for c in ("MARKET_LINE", "SELECTION_HANDICAP"):
        cand[c] = pd.to_numeric(cand[c], errors="coerce")

    sc, pr, un, dr, cv, providers = score_groups(prices, outc, cand)
    sc.to_csv(out_dir / "group_scores.csv", index=False)
    dr.to_csv(out_dir / "dropped_groups.csv", index=False)
    bc = band_coverage(cv)
    bc.to_csv(out_dir / "band_coverage.csv", index=False)
    if len(un):
        un.value_counts().rename("n").reset_index().to_csv(out_dir / "unparsed_selections.csv", index=False)
    raw = (sc[sc["provider"] == "source"].groupby(["SOURCE", "MARKET_TYPE_NAME"])["raw_sum"]
             .agg(median_raw_sum="median", share_off_by_1pct=lambda x: float((np.abs(x - 1) > 0.01).mean()))
             .reset_index())
    summ = (sc.groupby(["SOURCE", "kind", "provider"])
              .agg(groups=("logloss", "size"), logloss=("logloss", "mean"), brier=("brier", "mean"))
              .reset_index())
    summ.to_csv(out_dir / "summary.csv", index=False)
    (sc.groupby(["day", "SOURCE", "kind", "provider"]).agg(groups=("logloss", "size"), logloss=("logloss", "mean"))
       .reset_index().to_csv(out_dir / "by_day.csv", index=False))
    comp = comparisons(sc)
    comp.to_csv(out_dir / "comparisons.csv", index=False)
    smed = source_medians(pr, outc)
    smed.to_csv(out_dir / "source_medians.csv", index=False)
    mpath = out_dir / "candidate_medians.csv"
    cmed = pd.read_csv(mpath, dtype={"MATCH_CODE": str}).drop(columns="day") if mpath.exists() else pd.DataFrame()
    mt = median_table(smed, cmed, outc)
    mt.to_csv(out_dir / "median_comparisons.csv", index=False)
    cal = calibration(sc)
    cal.to_csv(out_dir / "moneyline_calibration.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print("\n" + "=" * 70 + "\nLOG LOSS BY SOURCE, MARKET KIND AND PROVIDER ('source' = that source's own price)")
        print(summ.to_string(index=False))
        print("\nPAIRED COMPARISONS (b_better_total > 0 means the second is better; |z_days| > 2.5 is clear;"
              "\n Mondays start a new stream-week the candidate has not seen -- see the Monday/other split)")
        print(comp.to_string(index=False))
        print("  (on two-way LINE markets the books price ~50/50 at their own line, so log loss there"
              "\n   tests the candidate's lean at the book's line; line placement is in the medians below)")
        print("\nMEDIANS vs RESULT (MAE in points; b_closer_total > 0 = the second is closer)")
        print(mt.to_string(index=False))
        print("\nsource medians: " + ", ".join(f"{k} {v}" for k, v in smed["how"].value_counts().items()))
        print("\nDROPPED MARKET GROUPS by reason")
        print(dr.to_string(index=False) if len(dr) else "  none")
        print("\nBAND MARKETS: offered bands truncated at the ends (scored conditional on the offered range;"
              "\n results outside it are not scored) and the probability each provider put outside them")
        print(bc.to_string(index=False) if len(bc) else "  none")
        print("\nRAW SUM of each source's fair probabilities per market (renormalised for scoring)")
        print(raw.to_string(index=False))
        print("\nMONEYLINE CALIBRATION (favourite probability bins)")
        print(cal.to_string(index=False))
    if len(un):
        print(f"\nUNPARSED selections ({len(un)} rows, see unparsed_selections.csv); most common:")
        print(un.value_counts().head(25).to_string())
    print(f"\nWritten to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
