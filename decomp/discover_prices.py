#!/usr/bin/env python3
"""
Read-only, second discovery pass: the basketball (NB) price tables since
DISCOVER_FROM, to design the prod / UAT B / candidate comparison.

For prod PRICE_CHANGES, UAT B PRICE_CHANGES_UATB and UAT PRICE_CHANGES it
reports, on NB matches only: the market types and how many matches each
covers, the selection names, line and handicap ranges, statuses, when the
last pre-match price is issued relative to the scheduled start, the
overround of the last pre-match prices, how PRICE_PROBABILITY relates to
1/PRICE_DECIMAL, and every last pre-match price for one sample match. Then it
checks the overlap of matches between the tables, the final scores against
the match stats, and whether game duration varies.

SELECT only. Writes discovery\\prices_<time>.txt.  Usage: python -u decomp\\discover_prices.py
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snowflake_connect import get_session  # noqa: E402

SINCE = os.environ.get("DISCOVER_FROM", "2026-10-12")
PROD, UAT = "SIS_PROD_CG_CURATED", "SIS_UAT_CG_CURATED"
TABLES = {"prod": f"{PROD}.SHARED.PRICE_CHANGES",
          "uat_b": f"{UAT}.SHARED.PRICE_CHANGES_UATB",
          "uat": f"{UAT}.SHARED.PRICE_CHANGES"}
# NB matches in the window, with scheduled start (EVENT) -- every query below joins to this
NB = f"""(SELECT e.MATCH_CODE, e.SCHEDULED_START_TIME_UTC AS START_T
          FROM {PROD}.SHARED.EVENT e
          WHERE e.SPORT_CODE = 'NB' AND e.SCHEDULED_START_TIME_UTC >= '{SINCE}')"""
PRE = "p.PRICE_ISSUE_TIME_UTC < n.START_T AND p.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -2, n.START_T)"
LAST_PRE = """QUALIFY ROW_NUMBER() OVER (PARTITION BY p.MATCH_CODE, p.MARKET_TYPE_NAME, p.MARKET_LINE,
              p.SELECTION_DESCRIPTION, p.SELECTION_HANDICAP ORDER BY p.PRICE_ISSUE_TIME_UTC DESC) = 1"""


def last_pre(t):
    """Derived table: the last pre-match price of every selection, with the match start."""
    return f"""(SELECT p.*, n.START_T FROM {t} p JOIN {NB} n ON n.MATCH_CODE = p.MATCH_CODE AND p.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -2, '{SINCE}'::timestamp)
               WHERE {PRE} {LAST_PRE})"""

OUT = Path("discovery")
OUT.mkdir(exist_ok=True)
REPORT = OUT / f"prices_{datetime.now(timezone.utc):%Y%m%d_%H%M}.txt"
_lines: list[str] = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    _lines.append(line)


def q(s, sql):
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.I):
        raise ValueError("refusing non-SELECT SQL")
    return s.sql(sql).to_pandas()


def show(label, s, sql, maxrows=60):
    try:
        df = q(s, sql)
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60):
            say(f"\n-- {label} ({len(df)} rows)\n" + df.head(maxrows).to_string(index=False))
        return df
    except Exception as e:
        say(f"\n-- {label}: skipped ({type(e).__name__}: {str(e).splitlines()[0][:200]})")
        return None


def section(t):
    say("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    s = get_session()
    say(f"Price discovery {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC; NB matches since {SINCE}")

    section("A. NB matches per day (EVENT) and how many have match stats")
    show("matches per day", s, f"""
        SELECT TO_DATE(n.START_T) DAY, COUNT(*) N_EVENT, COUNT(m.MATCH_CODE) N_WITH_STATS
        FROM {NB} n LEFT JOIN {PROD}.MODELLING.POSSESSION_NB_MATCH_STATS m ON m.MATCH_CODE = n.MATCH_CODE
        GROUP BY 1 ORDER BY 1""")

    for name, t in TABLES.items():
        section(f"B. {name}: {t}")
        show("market types (all prices, NB, window)", s, f"""
            SELECT p.MARKET_TYPE_NAME, COUNT(*) N_PRICES, COUNT(DISTINCT p.MATCH_CODE) N_MATCHES,
                   COUNT(DISTINCT p.SELECTION_DESCRIPTION) N_SEL_NAMES,
                   MIN(p.MARKET_LINE) MIN_LINE, MAX(p.MARKET_LINE) MAX_LINE,
                   MIN(p.SELECTION_HANDICAP) MIN_HCP, MAX(p.SELECTION_HANDICAP) MAX_HCP,
                   SUM(IFF({PRE}, 1, 0)) N_PREMATCH
            FROM {t} p JOIN {NB} n ON n.MATCH_CODE = p.MATCH_CODE AND p.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -2, '{SINCE}'::timestamp)
            GROUP BY 1 ORDER BY N_MATCHES DESC, N_PRICES DESC""")
        show("selection names and statuses per market type (pre-match)", s, f"""
            SELECT p.MARKET_TYPE_NAME, p.SELECTION_DESCRIPTION, p.SELECTION_STATUS, COUNT(*) N,
                   COUNT(DISTINCT p.MATCH_CODE) N_MATCHES
            FROM {t} p JOIN {NB} n ON n.MATCH_CODE = p.MATCH_CODE AND p.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -2, '{SINCE}'::timestamp)
            WHERE {PRE}
            GROUP BY 1, 2, 3 ORDER BY 1, N DESC""", maxrows=120)
        show("lines per match (pre-match): distinct lines and handicaps per market", s, f"""
            SELECT MARKET_TYPE_NAME, MEDIAN(N_LINES) MED_LINES, MAX(N_LINES) MAX_LINES,
                   MEDIAN(N_HCPS) MED_HCPS, MAX(N_HCPS) MAX_HCPS
            FROM (SELECT p.MATCH_CODE, p.MARKET_TYPE_NAME, COUNT(DISTINCT p.MARKET_LINE) N_LINES,
                         COUNT(DISTINCT p.SELECTION_HANDICAP) N_HCPS
                  FROM {t} p JOIN {NB} n ON n.MATCH_CODE = p.MATCH_CODE AND p.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -2, '{SINCE}'::timestamp) WHERE {PRE} GROUP BY 1, 2)
            GROUP BY 1 ORDER BY 1""")
        show("timing of the last pre-match price (minutes before start)", s, f"""
            SELECT l.MARKET_TYPE_NAME, COUNT(*) N,
                   APPROX_PERCENTILE(DATEDIFF(second, l.PRICE_ISSUE_TIME_UTC, l.START_T) / 60, 0.1) P10_MIN,
                   APPROX_PERCENTILE(DATEDIFF(second, l.PRICE_ISSUE_TIME_UTC, l.START_T) / 60, 0.5) P50_MIN,
                   APPROX_PERCENTILE(DATEDIFF(second, l.PRICE_ISSUE_TIME_UTC, l.START_T) / 60, 0.9) P90_MIN
            FROM {last_pre(t)} l
            GROUP BY 1 ORDER BY N DESC""")
        show("overround of last pre-match prices: sum of PRICE_PROBABILITY and of 1/PRICE_DECIMAL per market line", s, f"""
            SELECT MARKET_TYPE_NAME, COUNT(*) N_MARKET_LINES, MEDIAN(N_SEL) MED_SELECTIONS,
                   MEDIAN(SUM_PROB) MED_SUM_PROB, MEDIAN(SUM_IMPLIED) MED_SUM_1_OVER_DEC
            FROM (SELECT l.MATCH_CODE, l.MARKET_TYPE_NAME, l.MARKET_LINE, ABS(COALESCE(l.SELECTION_HANDICAP, 0)) H,
                         COUNT(*) N_SEL, SUM(l.PRICE_PROBABILITY) SUM_PROB, SUM(1 / NULLIF(l.PRICE_DECIMAL, 0)) SUM_IMPLIED
                  FROM {last_pre(t)} l
                  GROUP BY 1, 2, 3, 4)
            GROUP BY 1 ORDER BY N_MARKET_LINES DESC""")
        show("PRICE_PROBABILITY vs 1/PRICE_DECIMAL (last pre-match)", s, f"""
            SELECT l.MARKET_TYPE_NAME, COUNT(*) N, MEDIAN(l.PRICE_PROBABILITY) MED_PROB,
                   MEDIAN(l.PRICE_PROBABILITY * l.PRICE_DECIMAL) MED_PROB_X_DEC,
                   MIN(l.PRICE_PROBABILITY) MIN_PROB, MAX(l.PRICE_PROBABILITY) MAX_PROB
            FROM {last_pre(t)} l
            GROUP BY 1 ORDER BY N DESC""")
        show("one sample match: every last pre-match price", s, f"""
            WITH m AS (SELECT p.MATCH_CODE FROM {t} p JOIN {NB} n ON n.MATCH_CODE = p.MATCH_CODE AND p.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -2, '{SINCE}'::timestamp)
                       WHERE {PRE} GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1)
            SELECT p.MATCH_CODE, n.START_T, p.MARKET_TYPE_NAME, p.MARKET_LINE, p.SELECTION_DESCRIPTION,
                   p.SELECTION_HANDICAP, p.SELECTION_STATUS, p.PRICE_ISSUE_TIME_UTC, p.PRICE_DECIMAL, p.PRICE_PROBABILITY
            FROM {t} p JOIN {NB} n ON n.MATCH_CODE = p.MATCH_CODE AND p.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -2, '{SINCE}'::timestamp)
            WHERE p.MATCH_CODE IN (SELECT MATCH_CODE FROM m) AND {PRE} {LAST_PRE}
            ORDER BY p.MARKET_TYPE_NAME, p.MARKET_LINE, p.SELECTION_HANDICAP, p.SELECTION_DESCRIPTION""", maxrows=150)

    section("C. Overlap: NB matches with pre-match prices in each table")
    show("overlap", s, f"""
        WITH pr AS (SELECT DISTINCT p.MATCH_CODE FROM {TABLES['prod']} p JOIN {NB} n ON n.MATCH_CODE = p.MATCH_CODE AND p.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -2, '{SINCE}'::timestamp) WHERE {PRE}),
             ub AS (SELECT DISTINCT p.MATCH_CODE FROM {TABLES['uat_b']} p JOIN {NB} n ON n.MATCH_CODE = p.MATCH_CODE AND p.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -2, '{SINCE}'::timestamp) WHERE {PRE}),
             ua AS (SELECT DISTINCT p.MATCH_CODE FROM {TABLES['uat']} p JOIN {NB} n ON n.MATCH_CODE = p.MATCH_CODE AND p.PRICE_ISSUE_TIME_UTC >= DATEADD(day, -2, '{SINCE}'::timestamp) WHERE {PRE}),
             st AS (SELECT m.MATCH_CODE FROM {PROD}.MODELLING.POSSESSION_NB_MATCH_STATS m JOIN {NB} n ON n.MATCH_CODE = m.MATCH_CODE)
        SELECT (SELECT COUNT(*) FROM st) N_WITH_STATS, (SELECT COUNT(*) FROM pr) N_PROD, (SELECT COUNT(*) FROM ub) N_UATB,
               (SELECT COUNT(*) FROM ua) N_UAT,
               (SELECT COUNT(*) FROM pr JOIN ub USING (MATCH_CODE)) N_PROD_AND_UATB,
               (SELECT COUNT(*) FROM pr JOIN ub USING (MATCH_CODE) JOIN st USING (MATCH_CODE)) N_PROD_UATB_STATS""")

    section("D. Final scores: price feed vs match stats (PLAYER_1 = AWAY in the stats)")
    show("score agreement", s, f"""
        WITH f AS (SELECT MATCH_CODE, MAX(PLAYER_1_SCORE_FINAL) P1F, MAX(PLAYER_2_SCORE_FINAL) P2F
                   FROM {PROD}.TRADING.PRICE_ENDPERIOD WHERE SCHEDULED_START_TIME_UTC >= '{SINCE}' GROUP BY 1)
        SELECT COUNT(*) N, SUM(IFF(f.P1F = m.AWAY_POINTS AND f.P2F = m.HOME_POINTS, 1, 0)) P1_IS_AWAY,
               SUM(IFF(f.P1F = m.HOME_POINTS AND f.P2F = m.AWAY_POINTS, 1, 0)) P1_IS_HOME
        FROM f JOIN {PROD}.MODELLING.POSSESSION_NB_MATCH_STATS m ON m.MATCH_CODE = f.MATCH_CODE""")

    section("E. Game duration: fixed or variable?")
    show("duration distribution (seconds)", s, f"""
        SELECT COUNT(*) N, MIN(m.GAME_DURATION_SECONDS) MIN_S,
               APPROX_PERCENTILE(m.GAME_DURATION_SECONDS, 0.05) P05, APPROX_PERCENTILE(m.GAME_DURATION_SECONDS, 0.5) P50,
               APPROX_PERCENTILE(m.GAME_DURATION_SECONDS, 0.95) P95, MAX(m.GAME_DURATION_SECONDS) MAX_S,
               COUNT(DISTINCT m.GAME_DURATION_SECONDS) N_DISTINCT
        FROM {PROD}.MODELLING.POSSESSION_NB_MATCH_STATS m JOIN {NB} n ON n.MATCH_CODE = m.MATCH_CODE""")
    show("most common durations", s, f"""
        SELECT m.GAME_DURATION_SECONDS, COUNT(*) N
        FROM {PROD}.MODELLING.POSSESSION_NB_MATCH_STATS m JOIN {NB} n ON n.MATCH_CODE = m.MATCH_CODE
        GROUP BY 1 ORDER BY 2 DESC LIMIT 15""")
    say(f"\nReport written to {REPORT.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say(traceback.format_exc())
        raise
    finally:
        REPORT.write_text("\n".join(_lines), encoding="utf-8")
