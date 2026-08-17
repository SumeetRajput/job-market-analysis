#!/usr/bin/env python3
"""
================================================================================
 SQL ANALYTICS LAYER
================================================================================

The project brief names SQL in the required stack. This module loads the
cleaned dataset into a relational database and re-expresses the core
aggregations as real SQL -- CTEs, window functions, CASE expressions and
multi-level grouping -- rather than pandas groupbys.

WHY SQLITE BY DEFAULT
SQLite ships with Python and needs no server, so `python sql_analytics.py`
works immediately with zero setup. The same SQL is standard enough to run on
PostgreSQL: pass --postgres to use the DATABASE_URL connection instead, which
is what docker-compose provisions.

WHY THIS IS NOT DUPLICATED WORK
The pandas pipeline and the SQL layer answer the same questions two ways, and
`--verify` cross-checks them against each other. Agreement between two
independent implementations is real evidence the numbers are right; a single
implementation is just an assertion.

Usage:
    python sql_analytics.py --build          # load parquet into jobs.db
    python sql_analytics.py --queries        # run all analytical queries
    python sql_analytics.py --verify         # cross-check SQL vs pandas
    python sql_analytics.py --query top_countries
================================================================================
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import textwrap
from pathlib import Path

import pandas as pd

DB_PATH = Path(os.getenv("SQLITE_DB", "jobs.db"))
PARQUET = Path(os.getenv("DATA_DIR", "results")) / "jobs_clean.parquet"

# =============================================================================
# SCHEMA
# Indexes are chosen from the actual query patterns below: every analytical
# query filters on ts_reliable or pay_analysable and groups by country,
# category or date. Without these the country query does a full scan of
# 245k rows on every call.
# =============================================================================
SCHEMA = """
DROP TABLE IF EXISTS jobs;

CREATE TABLE jobs (
    job_id            INTEGER PRIMARY KEY,
    title             TEXT    NOT NULL,
    link              TEXT    UNIQUE,
    published_at      TEXT    NOT NULL,
    post_date         TEXT    NOT NULL,
    post_week         TEXT,
    post_month        TEXT,
    post_dow          TEXT,
    post_hour         INTEGER,
    is_weekend        INTEGER,
    -- Data-quality flags. Carried into the database rather than applied as
    -- deletions, so SQL consumers make the same explicit choices the pandas
    -- layer does.
    ts_reliable       INTEGER NOT NULL,
    pay_disclosed     INTEGER NOT NULL,
    pay_analysable    INTEGER NOT NULL,
    country           TEXT    NOT NULL,
    country_grouped   TEXT,
    is_hourly         INTEGER NOT NULL,
    pay_type          TEXT    NOT NULL,
    hourly_low        REAL,
    hourly_high       REAL,
    hourly_mid        REAL,
    budget            REAL,
    category          TEXT    NOT NULL,
    category_tier     TEXT    NOT NULL
);

CREATE INDEX idx_jobs_date      ON jobs(post_date);
CREATE INDEX idx_jobs_country   ON jobs(country);
CREATE INDEX idx_jobs_category  ON jobs(category);
CREATE INDEX idx_jobs_ts        ON jobs(ts_reliable);
CREATE INDEX idx_jobs_pay       ON jobs(pay_analysable, is_hourly);
"""

# =============================================================================
# ANALYTICAL QUERIES
# Each maps to a task from the brief and demonstrates a distinct SQL technique.
# =============================================================================
QUERIES: dict[str, tuple[str, str]] = {

    # -- Task 4 -------------------------------------------------------------
    "top_countries": (
        "Task 4: median hourly rate by client country, with global comparison",
        """
        WITH ranked AS (
            -- Window function assigns each row its position within its
            -- country, which is how a median is computed without a
            -- percentile function (SQLite has none).
            SELECT country,
                   hourly_mid,
                   ROW_NUMBER() OVER (PARTITION BY country ORDER BY hourly_mid) AS rn,
                   COUNT(*)     OVER (PARTITION BY country)                     AS n
            FROM jobs
            WHERE is_hourly = 1 AND pay_analysable = 1 AND country <> 'Unknown'
        ),
        medians AS (
            -- The middle one or two rows per country average to the median.
            SELECT country,
                   n AS postings,
                   AVG(hourly_mid) AS median_rate
            FROM ranked
            WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
            GROUP BY country, n
        ),
        global_median AS (
            SELECT AVG(hourly_mid) AS gm FROM (
                SELECT hourly_mid,
                       ROW_NUMBER() OVER (ORDER BY hourly_mid) AS rn,
                       COUNT(*)     OVER ()                    AS n
                FROM jobs
                WHERE is_hourly = 1 AND pay_analysable = 1
            ) WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
        )
        SELECT m.country,
               m.postings,
               ROUND(m.median_rate, 2)                              AS median_rate,
               ROUND(g.gm, 2)                                       AS global_median,
               ROUND((m.median_rate / g.gm - 1) * 100, 1)           AS vs_global_pct,
               RANK() OVER (ORDER BY m.median_rate DESC)            AS rate_rank
        FROM medians m CROSS JOIN global_median g
        WHERE m.postings >= 200
        ORDER BY m.median_rate DESC;
        """),

    # -- Task 2 -------------------------------------------------------------
    "category_summary": (
        "Task 2: category volume, pay and classification confidence",
        """
        SELECT category,
               COUNT(*)                                                     AS postings,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)           AS pct_of_market,
               -- CASE inside AVG turns a categorical flag into a percentage.
               ROUND(100.0 * AVG(CASE WHEN category_tier = 'specific'
                                      THEN 1.0 ELSE 0.0 END), 1)            AS pct_high_confidence,
               SUM(CASE WHEN is_hourly = 1 THEN 1 ELSE 0 END)               AS hourly_jobs,
               SUM(CASE WHEN is_hourly = 0 THEN 1 ELSE 0 END)               AS fixed_jobs,
               ROUND(AVG(CASE WHEN is_hourly = 1 AND pay_analysable = 1
                              THEN hourly_mid END), 2)                      AS avg_hourly_rate,
               ROUND(100.0 * AVG(pay_disclosed), 1)                         AS pct_pay_disclosed
        FROM jobs
        GROUP BY category
        ORDER BY postings DESC;
        """),

    # -- Task 6 -------------------------------------------------------------
    "weekly_trend": (
        "Task 6: weekly market dynamics with week-over-week change",
        """
        WITH weekly AS (
            SELECT post_week,
                   COUNT(*)                                    AS postings,
                   COUNT(DISTINCT post_date)                   AS days,
                   ROUND(100.0 * AVG(is_hourly), 1)            AS pct_hourly,
                   ROUND(100.0 * AVG(pay_disclosed), 1)        AS pct_disclosed,
                   COUNT(DISTINCT country)                     AS countries
            FROM jobs
            WHERE ts_reliable = 1        -- excludes backfill, outage, partial day
            GROUP BY post_week
        )
        SELECT post_week,
               postings,
               days,
               ROUND(1.0 * postings / days, 0)                              AS per_day,
               pct_hourly,
               pct_disclosed,
               countries,
               -- LAG reaches back one row to compute week-over-week change.
               LAG(postings) OVER (ORDER BY post_week)                      AS prev_postings,
               ROUND(100.0 * (postings - LAG(postings) OVER (ORDER BY post_week))
                     / LAG(postings) OVER (ORDER BY post_week), 1)          AS wow_change_pct
        FROM weekly
        ORDER BY post_week;
        """),

    # -- Task 3 -------------------------------------------------------------
    "seasonality": (
        "Task 3: day-of-week seasonality, the dominant forecasting signal",
        """
        WITH daily AS (
            SELECT post_date, post_dow, is_weekend, COUNT(*) AS n
            FROM jobs WHERE ts_reliable = 1
            GROUP BY post_date, post_dow, is_weekend
        )
        SELECT post_dow,
               COUNT(*)                                          AS days_observed,
               ROUND(AVG(n), 0)                                  AS avg_postings,
               MIN(n)                                            AS min_postings,
               MAX(n)                                            AS max_postings,
               ROUND(100.0 * AVG(n) / (SELECT AVG(n) FROM daily WHERE is_weekend = 0)
                     - 100, 1)                                   AS vs_weekday_pct
        FROM daily
        GROUP BY post_dow, is_weekend
        ORDER BY avg_postings DESC;
        """),

    # -- Task 1 -------------------------------------------------------------
    "pay_by_category_country": (
        "Task 1/4: cross-tab of pay by category and top country",
        """
        WITH top_countries AS (
            SELECT country FROM jobs
            WHERE country <> 'Unknown'
            GROUP BY country ORDER BY COUNT(*) DESC LIMIT 6
        )
        SELECT j.category,
               j.country,
               COUNT(*)                                          AS postings,
               ROUND(AVG(CASE WHEN is_hourly = 1 AND pay_analysable = 1
                              THEN hourly_mid END), 2)           AS avg_hourly,
               ROUND(AVG(CASE WHEN is_hourly = 0 AND pay_analysable = 1
                              THEN budget END), 0)               AS avg_budget,
               -- Rank categories within each country, not globally.
               RANK() OVER (PARTITION BY j.country
                            ORDER BY COUNT(*) DESC)              AS rank_in_country
        FROM jobs j
        INNER JOIN top_countries t ON j.country = t.country
        WHERE j.ts_reliable = 1
        GROUP BY j.category, j.country
        HAVING COUNT(*) >= 50
        ORDER BY j.country, postings DESC;
        """),

    # -- Task 7 -------------------------------------------------------------
    "market_concentration": (
        "Task 7: geographic concentration of remote demand",
        """
        WITH country_vol AS (
            SELECT country, COUNT(*) AS postings
            FROM jobs WHERE ts_reliable = 1 AND country <> 'Unknown'
            GROUP BY country
        ),
        with_share AS (
            SELECT country, postings,
                   100.0 * postings / SUM(postings) OVER ()      AS pct_share,
                   -- Running total gives cumulative concentration directly.
                   100.0 * SUM(postings) OVER (ORDER BY postings DESC)
                         / SUM(postings) OVER ()                 AS cumulative_pct,
                   ROW_NUMBER() OVER (ORDER BY postings DESC)    AS rank
            FROM country_vol
        )
        SELECT rank, country, postings,
               ROUND(pct_share, 2)     AS pct_share,
               ROUND(cumulative_pct, 2) AS cumulative_pct
        FROM with_share
        WHERE rank <= 20
        ORDER BY rank;
        """),

    # -- Data quality --------------------------------------------------------
    "data_quality": (
        "Data quality: exclusion flags and their impact",
        """
        SELECT 'Total rows'                AS metric, COUNT(*)                          AS rows,
               100.0                                                                    AS pct
        FROM jobs
        UNION ALL
        SELECT 'Time-series reliable',     SUM(ts_reliable),
               ROUND(100.0 * AVG(ts_reliable), 2) FROM jobs
        UNION ALL
        SELECT 'Pay disclosed',            SUM(pay_disclosed),
               ROUND(100.0 * AVG(pay_disclosed), 2) FROM jobs
        UNION ALL
        SELECT 'Pay analysable',           SUM(pay_analysable),
               ROUND(100.0 * AVG(pay_analysable), 2) FROM jobs
        UNION ALL
        SELECT 'High-confidence category', SUM(CASE WHEN category_tier = 'specific' THEN 1 ELSE 0 END),
               ROUND(100.0 * AVG(CASE WHEN category_tier = 'specific' THEN 1.0 ELSE 0.0 END), 2)
        FROM jobs;
        """),

    # -- Task 5 support ------------------------------------------------------
    "best_paying_roles": (
        "Task 5: highest-paying categories for a job seeker",
        """
        WITH hourly_stats AS (
            SELECT category, hourly_mid,
                   ROW_NUMBER() OVER (PARTITION BY category ORDER BY hourly_mid) AS rn,
                   COUNT(*)     OVER (PARTITION BY category)                     AS n
            FROM jobs
            WHERE is_hourly = 1 AND pay_analysable = 1
        )
        SELECT category,
               n                              AS hourly_postings,
               ROUND(AVG(hourly_mid), 2)      AS median_hourly_rate
        FROM hourly_stats
        WHERE rn IN ((n + 1) / 2, (n + 2) / 2) AND n >= 500
        GROUP BY category, n
        ORDER BY median_hourly_rate DESC;
        """),
}


# =============================================================================
# ENGINE
# =============================================================================
def get_connection(postgres: bool = False):
    if postgres:
        try:
            import psycopg2
        except ImportError:
            print("psycopg2 not installed. Run: pip install psycopg2-binary", file=sys.stderr)
            sys.exit(1)
        url = os.getenv("DATABASE_URL", "postgresql://jobs:jobs@localhost:5432/jobmarket")
        return psycopg2.connect(url)
    return sqlite3.connect(DB_PATH)


def build_database(postgres: bool = False) -> int:
    """Load the cleaned parquet into the database."""
    if not PARQUET.exists():
        print(f"ERROR: {PARQUET} not found.\n"
              f"Run first: python job_market_analysis.py --data jobs.csv", file=sys.stderr)
        return 2

    print(f"Loading {PARQUET}...")
    df = pd.read_parquet(PARQUET)

    cols = ["title", "link", "published_at", "post_date", "post_week", "post_month",
            "post_dow", "post_hour", "is_weekend", "ts_reliable", "pay_disclosed",
            "pay_analysable", "country", "country_grouped", "is_hourly", "pay_type",
            "hourly_low", "hourly_high", "hourly_mid", "budget", "category",
            "category_tier"]
    out = df[cols].copy()
    for c in ["published_at", "post_date"]:
        out[c] = out[c].astype(str)
    for c in ["is_weekend", "ts_reliable", "pay_disclosed", "pay_analysable", "is_hourly"]:
        out[c] = out[c].astype(int)
    out.insert(0, "job_id", range(1, len(out) + 1))

    conn = get_connection(postgres)
    cur = conn.cursor()
    for stmt in SCHEMA.split(";"):
        if stmt.strip():
            cur.execute(stmt)
    conn.commit()

    print(f"Inserting {len(out):,} rows...")
    if postgres:
        from io import StringIO
        buf = StringIO()
        out.to_csv(buf, index=False, header=False)
        buf.seek(0)
        cur.copy_from(buf, "jobs", sep=",", null="")
    else:
        out.to_sql("jobs", conn, if_exists="append", index=False)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM jobs")
    n = cur.fetchone()[0]
    conn.close()
    print(f"Database built: {n:,} rows in {DB_PATH if not postgres else 'PostgreSQL'}")
    return 0


def run_query(name: str, postgres: bool = False, show: bool = True) -> pd.DataFrame:
    desc, sql = QUERIES[name]
    conn = get_connection(postgres)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    if show:
        print("\n" + "=" * 78)
        print(f"{name.upper()}  |  {desc}")
        print("=" * 78)
        print(df.head(25).to_string(index=False))
        if len(df) > 25:
            print(f"... {len(df) - 25} more rows")
    return df


def verify_against_pandas() -> int:
    """Cross-check SQL results against the pandas pipeline.

    Two independent implementations agreeing is evidence; one implementation
    is only an assertion. Any mismatch here means one of the two layers has a
    bug, and it is worth knowing which.
    """
    print("\n" + "=" * 78)
    print("VERIFICATION: SQL vs pandas")
    print("=" * 78)
    df = pd.read_parquet(PARQUET)
    checks, failed = [], 0

    # 1 | row count
    conn = get_connection()
    n_sql = pd.read_sql_query("SELECT COUNT(*) AS n FROM jobs", conn)["n"].iloc[0]
    checks.append(("Total rows", len(df), n_sql, len(df) == n_sql))

    # 2 | ts_reliable count
    n_sql = pd.read_sql_query(
        "SELECT SUM(ts_reliable) AS n FROM jobs", conn)["n"].iloc[0]
    checks.append(("ts_reliable rows", int(df.ts_reliable.sum()), int(n_sql),
                   int(df.ts_reliable.sum()) == int(n_sql)))

    # 3 | category count
    n_sql = pd.read_sql_query(
        "SELECT COUNT(DISTINCT category) AS n FROM jobs", conn)["n"].iloc[0]
    checks.append(("Distinct categories", df.category.nunique(), n_sql,
                   df.category.nunique() == n_sql))
    conn.close()

    # 4 | US median hourly rate (median via window function vs pandas)
    sq = run_query("top_countries", show=False)
    us_sql = float(sq[sq.country == "United States"]["median_rate"].iloc[0])
    us_pd = float(df[(df.is_hourly) & (df.pay_analysable) &
                     (df.country == "United States")]["hourly_mid"].median())
    checks.append(("US median hourly", us_pd, us_sql, abs(us_pd - us_sql) < 0.51))

    # 5 | top category by volume
    cs = run_query("category_summary", show=False)
    top_sql = cs.iloc[0]["category"]
    top_pd = df.category.value_counts().index[0]
    checks.append(("Top category", top_pd, top_sql, top_pd == top_sql))

    for name, expected, got, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  {status}  {name:<28} pandas={expected!s:<22} sql={got}")

    print(f"\n  {len(checks) - failed} passed, {failed} failed")
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SQL analytics layer")
    ap.add_argument("--build", action="store_true", help="load parquet into the database")
    ap.add_argument("--queries", action="store_true", help="run all analytical queries")
    ap.add_argument("--query", type=str, choices=list(QUERIES), help="run one query")
    ap.add_argument("--verify", action="store_true", help="cross-check SQL vs pandas")
    ap.add_argument("--postgres", action="store_true", help="use PostgreSQL not SQLite")
    ap.add_argument("--export", action="store_true", help="write query results to CSV")
    args = ap.parse_args(argv)

    if not any([args.build, args.queries, args.query, args.verify]):
        ap.print_help()
        print("\nAvailable queries:")
        for k, (d, _) in QUERIES.items():
            print(f"  {k:<26} {d}")
        return 0

    if args.build:
        rc = build_database(args.postgres)
        if rc:
            return rc

    if not DB_PATH.exists() and not args.postgres:
        print(f"Database not found. Run: python sql_analytics.py --build", file=sys.stderr)
        return 2

    if args.query:
        df = run_query(args.query, args.postgres)
        if args.export:
            out = Path("results/sql") / f"{args.query}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"\nExported: {out}")

    if args.queries:
        outdir = Path("results/sql")
        outdir.mkdir(parents=True, exist_ok=True)
        for name in QUERIES:
            df = run_query(name, args.postgres)
            if args.export:
                df.to_csv(outdir / f"{name}.csv", index=False)
        if args.export:
            print(f"\nAll query results exported to {outdir}/")

    if args.verify:
        return verify_against_pandas()
    return 0


if __name__ == "__main__":
    sys.exit(main())
