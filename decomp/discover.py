#!/usr/bin/env python3
"""
Read-only discovery for the prod / UAT B / candidate comparison.

Finds, without changing anything (SHOW, LIST and SELECT only):
  1. the databases this login can see;
  2. in every SIS* database: all tables, with the ones that look like model
     output, prices or markets highlighted (GAMEPLAI_STREAM, *_CANDIDATE, *_B,
     PRICE*, MARKET*, PROB*, PRED*, MODEL*, DECOMP*, POSSESSION*);
  3. for those: columns, size, freshness, and rows since DISCOVER_FROM;
  4. which SPORT_CODE the NBA2K matches carry (via the match-stats table);
  5. for model/price streams: the markets and a sample for one recent match;
  6. the decomp output stages (where the prod model's parameters are kept);
  7. how far the match stats run in each database.

Everything is written to discovery\\ (text report + full table list CSV) so it
can be pushed for analysis. Usage:  python -u decomp\\discover.py
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
DB_FILTER = re.compile(os.environ.get("DISCOVER_DB_REGEX", r"^SIS"), re.I)
INTEREST = re.compile(r"GAMEPLAI|STREAM|CANDIDATE|(^|_)B($|_)|UAT|PRICE|MARKET|SELECTION|PROB|PRED|"
                      r"MODEL|DECOMP|POSSESSION|SIMUL|FORECAST|ODDS|LINE", re.I)
TIME_COLS = ["PUBLISH_TIME", "PRICE_ISSUE_TIME_UTC", "CHANGE_TIME", "FILE_TIME", "EVENT_TIME",
             "SCHEDULED_START_TIME_UTC", "UPDATED_TIME_UTC", "CREATED_TIME_UTC", "UPDATED_AT", "CREATED_AT"]
MAX_DETAILED = int(os.environ.get("DISCOVER_MAX_TABLES", "60"))

OUT = Path("discovery")
OUT.mkdir(exist_ok=True)
REPORT = OUT / f"discovery_{datetime.now(timezone.utc):%Y%m%d_%H%M}.txt"
_lines: list[str] = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    _lines.append(line)


def q(session, sql):
    """Read-only query -> DataFrame (refuses anything that is not SHOW/LIST/SELECT/DESC)."""
    if not re.match(r"^\s*(SHOW|LIST|SELECT|DESC|DESCRIBE|WITH)\b", sql, re.I):
        raise ValueError(f"refusing non-read-only SQL: {sql[:60]}")
    return session.sql(sql).to_pandas()


def section(title):
    say("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def safe(label, fn):
    try:
        return fn()
    except Exception as e:  # keep going: one denied database must not stop the rest
        say(f"  [{label}] skipped: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        return None


def main():
    s = get_session()
    say(f"Discovery run {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC; window since {SINCE}")
    who = safe("whoami", lambda: q(s, "SELECT CURRENT_USER() U, CURRENT_ROLE() R, CURRENT_AVAILABLE_ROLES() AR"))
    if who is not None:
        say(who.to_string(index=False))

    # 1. databases ------------------------------------------------------------
    section("1. Databases visible to this login")
    dbs = safe("show databases", lambda: q(s, "SHOW DATABASES"))
    names = []
    if dbs is not None:
        col = next(c for c in dbs.columns if c.strip('"').lower() == "name")
        names = [n for n in dbs[col].astype(str)]
        say(", ".join(names))
    targets = [n for n in names if DB_FILTER.search(n)]
    say(f"Inspecting: {targets}")

    # 2. tables ---------------------------------------------------------------
    section("2. Tables in each database (interesting ones flagged *)")
    all_tables = []
    for db in targets:
        t = safe(db, lambda: q(s, f"""SELECT table_schema, table_name, table_type, row_count, bytes, last_altered
                                       FROM {db}.INFORMATION_SCHEMA.TABLES
                                       WHERE table_schema <> 'INFORMATION_SCHEMA'
                                       ORDER BY table_schema, table_name"""))
        if t is None:
            continue
        t.columns = [c.upper() for c in t.columns]
        t.insert(0, "DATABASE", db)
        all_tables.append(t)
        say(f"\n{db}: {len(t)} tables in schemas {sorted(t['TABLE_SCHEMA'].unique())}")
        for _, r in t.iterrows():
            flag = "*" if INTEREST.search(f"{r.TABLE_SCHEMA}.{r.TABLE_NAME}") else " "
            if flag == "*":
                say(f"  {flag} {r.TABLE_SCHEMA}.{r.TABLE_NAME}  ({r.TABLE_TYPE}, rows {r.ROW_COUNT}, altered {r.LAST_ALTERED})")
    if not all_tables:
        say("No tables could be listed.")
        return
    tables = pd.concat(all_tables, ignore_index=True)
    tables.to_csv(OUT / "all_tables.csv", index=False)
    say(f"\nFull list: {OUT / 'all_tables.csv'} ({len(tables)} tables)")

    # 3. NBA2K sport code ------------------------------------------------------
    section("3. Sport code of the NBA2K matches (match stats joined to EVENT)")
    sport = None
    for db in targets:
        if not ((tables["DATABASE"] == db) & (tables["TABLE_NAME"] == "POSSESSION_NB_MATCH_STATS")).any():
            continue
        r = safe(f"{db} sport", lambda: q(s, f"""
            SELECT e.SPORT_CODE, COUNT(*) N, MIN(e.SCHEDULED_START_TIME_UTC) FIRST_MATCH,
                   MAX(e.SCHEDULED_START_TIME_UTC) LAST_MATCH,
                   SUM(IFF(e.SCHEDULED_START_TIME_UTC >= '{SINCE}', 1, 0)) N_SINCE
            FROM {db}.MODELLING.POSSESSION_NB_MATCH_STATS m
            LEFT JOIN SIS_PROD_CG_CURATED.SHARED.EVENT e ON e.MATCH_CODE = m.MATCH_CODE
            GROUP BY 1 ORDER BY 2 DESC"""))
        if r is not None:
            say(f"{db}.MODELLING.POSSESSION_NB_MATCH_STATS:\n{r.to_string(index=False)}")
            codes = [c for c in r.iloc[:, 0].dropna().astype(str)]
            sport = sport or (codes[0] if codes else None)
    say(f"Sport code used below: {sport}")

    # 4. interesting tables in detail -------------------------------------------
    section(f"4. Columns, freshness and rows since {SINCE} (up to {MAX_DETAILED} tables)")
    pick = tables[tables.apply(lambda r: bool(INTEREST.search(f"{r.TABLE_SCHEMA}.{r.TABLE_NAME}")), axis=1)]
    pick = pick[~pick["TABLE_SCHEMA"].str.upper().isin(["INFORMATION_SCHEMA"])]
    if len(pick) > MAX_DETAILED:
        say(f"  {len(pick)} candidates; detailing the {MAX_DETAILED} most recently altered "
            f"(raise $env:DISCOVER_MAX_TABLES for more)")
        pick = pick.sort_values("LAST_ALTERED", ascending=False).head(MAX_DETAILED)
    streams = []
    for _, r in pick.iterrows():
        fq = f"{r.DATABASE}.{r.TABLE_SCHEMA}.{r.TABLE_NAME}"
        cols = safe(fq, lambda: q(s, f"""SELECT column_name, data_type FROM {r.DATABASE}.INFORMATION_SCHEMA.COLUMNS
                                          WHERE table_schema = '{r.TABLE_SCHEMA}' AND table_name = '{r.TABLE_NAME}'
                                          ORDER BY ordinal_position"""))
        if cols is None:
            continue
        cols.columns = [c.upper() for c in cols.columns]
        cn = cols["COLUMN_NAME"].str.upper().tolist()
        say(f"\n- {fq}  (rows {r.ROW_COUNT})")
        say("    columns: " + ", ".join(f"{a}:{b}" for a, b in zip(cols["COLUMN_NAME"], cols["DATA_TYPE"])))
        tcol = next((c for c in TIME_COLS if c in cn), None)
        where_sport = f" AND SPORT_CODE = '{sport}'" if sport and "SPORT_CODE" in cn else ""
        if tcol:
            span = safe(fq + " span", lambda: q(s, f"""SELECT MIN({tcol}) FIRST_T, MAX({tcol}) LAST_T,
                       SUM(IFF({tcol} >= '{SINCE}', 1, 0)) ROWS_SINCE
                       FROM {fq} WHERE {tcol} >= DATEADD(day, -60, '{SINCE}'::timestamp){where_sport}"""))
            if span is not None:
                say(f"    {tcol} (last 60 days before {SINCE} onward{', ' + sport if where_sport else ''}): "
                    + span.to_string(index=False, header=False))
        if "SPORT_CODE" in cn and tcol:
            sc = safe(fq + " sports", lambda: q(s, f"""SELECT SPORT_CODE, COUNT(*) N FROM {fq}
                       WHERE {tcol} >= '{SINCE}' GROUP BY 1 ORDER BY 2 DESC LIMIT 10"""))
            if sc is not None:
                say("    sport codes since window: " + ", ".join(f"{a}={b}" for a, b in sc.itertuples(index=False)))
        if {"MATCH_CODE", "PROBABILITY"} <= set(cn) or {"MATCH_CODE", "PRICE"} <= set(cn) or "MARKET_TYPE" in cn:
            streams.append((fq, cn, tcol, where_sport))

    # 5. model / price streams: markets and one sample match ----------------------
    section("5. Markets in model/price streams, and a sample of one recent match")
    for fq, cn, tcol, where_sport in streams:
        if not tcol:
            continue
        mk = [c for c in ("MARKET_TYPE", "MARKET_ID", "MARKET_DESCRIPTION") if c in cn]
        if mk:
            m = safe(fq + " markets", lambda: q(s, f"""SELECT {', '.join(mk)}, COUNT(*) N FROM {fq}
                       WHERE {tcol} >= '{SINCE}'{where_sport} GROUP BY {', '.join(mk)} ORDER BY N DESC LIMIT 25"""))
            if m is not None:
                say(f"\n- {fq} markets since {SINCE}:\n{m.to_string(index=False)}")
        if "MATCH_CODE" in cn:
            mc = safe(fq + " match", lambda: q(s, f"""SELECT MATCH_CODE FROM {fq}
                       WHERE {tcol} >= '{SINCE}'{where_sport} ORDER BY {tcol} DESC LIMIT 1"""))
            if mc is not None and len(mc):
                code = str(mc.iloc[0, 0]).replace("'", "''")
                smp = safe(fq + " sample", lambda: q(s, f"""SELECT * FROM {fq} WHERE MATCH_CODE = '{code}'
                           ORDER BY {tcol} LIMIT 12"""))
                if smp is not None:
                    with pd.option_context("display.width", 250, "display.max_columns", 40,
                                           "display.max_colwidth", 40):
                        say(f"  sample rows for MATCH_CODE {code}:\n{smp.to_string(index=False)}")

    # 6. decomp output stages -------------------------------------------------------
    section("6. Stages in MODELLING schemas (prod decomp parameters live in one)")
    for db in targets:
        st = safe(f"{db} stages", lambda: q(s, f"SHOW STAGES IN SCHEMA {db}.MODELLING"))
        if st is None or not len(st):
            continue
        ncol = next(c for c in st.columns if c.strip('"').lower() == "name")
        for name in st[ncol].astype(str):
            ls = safe(f"{db} list {name}", lambda: q(s, f"LIST @{db}.MODELLING.{name}"))
            if ls is not None:
                ls.columns = [c.strip('"').lower() for c in ls.columns]
                ls = ls.sort_values("last_modified", ascending=False) if "last_modified" in ls.columns else ls
                say(f"\n- @{db}.MODELLING.{name}: {len(ls)} files; most recent:\n"
                    + ls.head(15).to_string(index=False))

    # 7. match-stats coverage ---------------------------------------------------------
    section(f"7. Match stats available since {SINCE}, by database")
    for db in targets:
        if ((tables["DATABASE"] == db) & (tables["TABLE_NAME"] == "POSSESSION_NB_MATCH_STATS")).any():
            c = safe(db + " coverage", lambda: q(s, f"""
                SELECT COUNT(*) N_MATCHES, MIN(e.SCHEDULED_START_TIME_UTC) FIRST_MATCH,
                       MAX(e.SCHEDULED_START_TIME_UTC) LAST_MATCH
                FROM {db}.MODELLING.POSSESSION_NB_MATCH_STATS m
                JOIN SIS_PROD_CG_CURATED.SHARED.EVENT e ON e.MATCH_CODE = m.MATCH_CODE
                WHERE e.SCHEDULED_START_TIME_UTC >= '{SINCE}'"""))
            if c is not None:
                say(f"{db}: " + c.to_string(index=False, header=False))

    say(f"\nReport written to {REPORT.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say(traceback.format_exc())
        raise
    finally:
        REPORT.write_text("\n".join(_lines), encoding="utf-8")
