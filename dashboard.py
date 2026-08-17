#!/usr/bin/env python3
"""
================================================================================
 JOB MARKET DASHBOARD  (Task 6 deliverable)
================================================================================

Interactive dashboard over the cleaned Upwork dataset.

The brief asks for a dashboard that "updates monthly". The clean window is ~39
days spanning two PARTIAL calendar months, so month-over-month growth is not
computable from this extract. The grain selector below therefore offers
daily / weekly / monthly: the monthly view is wired and functional, and will
be meaningful the moment two complete months exist. Until then it carries an
explicit incompleteness warning rather than presenting misleading growth.

Run:  streamlit run dashboard.py
================================================================================
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from job_market_analysis import CFG, JobRecommender, clean_data

st.set_page_config(page_title="Job Market Dashboard", page_icon="*", layout="wide")

DATA_DIR = Path(os.getenv("DATA_DIR", "results"))
RAW_CSV = Path(os.getenv("RAW_CSV", "jobs.csv"))


# ------------------------------------------------------------------ loading
@st.cache_data(show_spinner="Loading cleaned dataset...")
def load_data() -> pd.DataFrame:
    parquet = DATA_DIR / "jobs_clean.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if RAW_CSV.exists():
        df, _ = clean_data(pd.read_csv(RAW_CSV))
        return df
    st.error(f"No data found. Expected {parquet} or {RAW_CSV}.\n\n"
             f"Run: `python job_market_analysis.py --data <csv>`")
    st.stop()


@st.cache_resource(show_spinner="Building recommendation index...")
def load_recommender(_df: pd.DataFrame) -> JobRecommender:
    return JobRecommender().fit(_df)


@st.cache_data
def load_table(name: str) -> pd.DataFrame | None:
    p = DATA_DIR / "tables" / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


df = load_data()
ts = df[df["ts_reliable"]]

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Job Market Dashboard")
st.sidebar.caption("Upwork postings | Feb-Mar 2024")

page = st.sidebar.radio("View", [
    "Market Overview", "Pay Analysis", "Categories", "Geography",
    "Job Search", "Data Quality"])

st.sidebar.divider()
st.sidebar.metric("Postings", f"{len(df):,}")
st.sidebar.metric("Time-series reliable", f"{len(ts):,}")
st.sidebar.caption(
    f"{(~df['ts_reliable']).sum():,} rows excluded from time-series views "
    f"(scraper backfill, collection outage, truncated final day)")


# ============================================================ MARKET OVERVIEW
if page == "Market Overview":
    st.title("Market Overview")

    grain = st.radio("Aggregation grain", ["daily", "weekly", "monthly"],
                     horizontal=True, index=1)
    if grain == "monthly":
        st.warning(
            "**Monthly view is incomplete.** The clean window spans ~39 days across "
            "two PARTIAL months (February starts 13 Feb when collection began; March "
            "is truncated on the 24th). Month-over-month change computed here would "
            "measure collection coverage, not the market. The view is wired and "
            "becomes valid once two complete months are available.")

    key = {"daily": "post_date", "weekly": "post_week", "monthly": "post_month"}[grain]
    g = ts.groupby(key)
    agg = pd.DataFrame({
        "postings": g.size(),
        "pct_hourly": g["is_hourly"].mean() * 100,
        "pct_pay_disclosed": g["pay_disclosed"].mean() * 100,
        "median_hourly": g.apply(
            lambda x: x.loc[x["is_hourly"] & x["pay_analysable"], "hourly_mid"].median()),
    })
    agg.index = agg.index.astype(str)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Postings/day (avg)", f"{len(ts) / ts['post_date'].nunique():,.0f}")
    c2.metric("Hourly share", f"{ts['is_hourly'].mean() * 100:.1f}%")
    c3.metric("Pay disclosed", f"{ts['pay_disclosed'].mean() * 100:.1f}%")
    c4.metric("Clean days", ts["post_date"].nunique())

    st.subheader("Posting volume")
    st.line_chart(agg["postings"], height=280)

    st.subheader("Weekly seasonality")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = ts.groupby("post_dow").size().reindex(order)
    days = ts.groupby("post_dow")["post_date"].nunique().reindex(order)
    st.bar_chart((dow / days).rename("avg postings/day"), height=260)

    wk = (dow / days)[:5].mean()
    we = (dow / days)[5:].mean()
    st.info(
        f"**Weekend collapse:** weekdays average {wk:,.0f} postings vs "
        f"{we:,.0f} at weekends, a {(1 - we / wk) * 100:.0f}% drop. This weekly cycle "
        f"is the strongest pattern in the data and dominates any short-horizon forecast.")

    st.subheader("Market composition stability")
    st.line_chart(agg[["pct_hourly", "pct_pay_disclosed"]], height=250)
    st.caption(
        f"Volume swings ~32% on the weekly cycle while STRUCTURE barely moves "
        f"(hourly share std dev: {ts.groupby('post_date')['is_hourly'].mean().std() * 100:.2f} "
        f"percentage points). Volume change is not the same as market change.")


# =============================================================== PAY ANALYSIS
elif page == "Pay Analysis":
    st.title("Pay Analysis")
    st.warning(
        "**Hourly rates and fixed budgets are never averaged together.** They are "
        "incompatible units: converting a $500 project budget to an hourly "
        "equivalent requires project duration, which this dataset does not contain. "
        "The two tracks are shown separately throughout.")

    h = df[df["is_hourly"] & df["pay_analysable"]]
    b = df[~df["is_hourly"] & df["pay_analysable"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median hourly", f"${h['hourly_mid'].median():.2f}/hr")
    c2.metric("Median budget", f"${b['budget'].median():,.0f}")
    c3.metric("Hourly postings", f"{len(h):,}")
    c4.metric("Fixed postings", f"{len(b):,}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Hourly rate distribution")
        hist = np.histogram(h["hourly_mid"].clip(0, 150), bins=50)
        st.bar_chart(pd.Series(hist[0], index=np.round(hist[1][:-1], 1)), height=280)
    with c2:
        st.subheader("Fixed budget distribution (log10 $)")
        hist = np.histogram(np.log10(b["budget"].clip(lower=1)), bins=50)
        st.bar_chart(pd.Series(hist[0], index=np.round(hist[1][:-1], 2)), height=280)

    st.subheader("Pay by category")
    g = df.groupby("category")
    pay = pd.DataFrame({
        "postings": g.size(),
        "median_hourly": g.apply(lambda x: x.loc[x["is_hourly"] & x["pay_analysable"], "hourly_mid"].median()),
        "median_budget": g.apply(lambda x: x.loc[~x["is_hourly"] & x["pay_analysable"], "budget"].median()),
    }).sort_values("median_hourly", ascending=False)
    st.dataframe(pay, use_container_width=True)

    st.subheader("Keyword pay premiums")
    track = st.radio("Track", ["hourly", "fixed"], horizontal=True)
    kw = load_table(f"task1_keywords_{track}")
    if kw is not None:
        sig = kw[kw["significant_fdr"] & (kw["effect"] != "negligible")]
        st.caption(
            f"{len(sig):,} of {len(kw):,} tested keywords are statistically significant "
            f"after Benjamini-Hochberg FDR correction AND show a non-negligible effect "
            f"size (Cliff's delta). With n>100k, p-values alone are not enough -- "
            f"trivial differences reach significance, so effect size is the filter "
            f"that matters.")
        c1, c2 = st.columns(2)
        cols = ["keyword", "n_postings", "median_pay", "vs_overall_pct", "effect"]
        c1.write("**Premium keywords**")
        c1.dataframe(sig.head(20)[cols], use_container_width=True, hide_index=True)
        c2.write("**Discount keywords**")
        c2.dataframe(sig.tail(20)[cols].iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("Run `python job_market_analysis.py --data <csv>` to generate keyword tables.")


# ================================================================= CATEGORIES
elif page == "Categories":
    st.title("Categories")
    st.caption(
        "The dataset has NO category column. Categories are DERIVED from title text "
        "using a two-tier keyword taxonomy. Tier 1 uses specific phrases; tier 2 "
        "catches bare generic tokens that tier 1 cannot match. Measured precision "
        "against independent anchor terms: 89.6% overall.")

    tiers = df["category_tier"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Tier 1 (specific)", f"{tiers.get('specific', 0):,}",
              f"{tiers.get('specific', 0) / len(df) * 100:.1f}%")
    c2.metric("Tier 2 (fallback)", f"{tiers.get('fallback', 0):,}",
              f"{tiers.get('fallback', 0) / len(df) * 100:.1f}%")
    c3.metric("Unmatched", f"{tiers.get('unmatched', 0):,}",
              f"{tiers.get('unmatched', 0) / len(df) * 100:.1f}%")

    st.subheader("Distribution")
    st.bar_chart(df["category"].value_counts(), height=380)

    st.subheader("Classification confidence by category")
    conf = pd.crosstab(df["category"], df["category_tier"])
    conf["pct_specific"] = (conf.get("specific", 0) / conf.sum(axis=1) * 100).round(1)
    st.dataframe(conf.sort_values("pct_specific", ascending=False), use_container_width=True)

    mom = load_table("task2_category_momentum")
    if mom is not None:
        st.subheader("Category momentum")
        st.dataframe(mom, use_container_width=True, hide_index=True)
        n_sig = int(mom["significant_fdr"].sum())
        st.warning(
            f"Only {n_sig} of {len(mom)} categories show a statistically significant "
            f"trend over the ~39-day window. The rest are flat-with-noise. A 5-week "
            f"slope cannot distinguish a genuinely emerging category from a seasonal "
            f"blip -- R-squared is shown for every category so weak fits stay visible.")


# ================================================================== GEOGRAPHY
elif page == "Geography":
    st.title("Geography")
    st.warning(
        "**`country` is the CLIENT's location, not the freelancer's.** This shows what "
        "clients in each country PAY, not what workers there EARN. Rates are also not "
        "adjusted for purchasing power parity or cost of living.")

    h = df[df["is_hourly"] & df["pay_analysable"] & (df["country"] != "Unknown")]
    g = h.groupby("country")["hourly_mid"]
    stats = pd.DataFrame({"postings": g.size(), "median_rate": g.median().round(2),
                          "p25": g.quantile(.25).round(2), "p75": g.quantile(.75).round(2)})
    stats = stats[stats["postings"] >= CFG.MIN_COUNTRY_POSTINGS]

    vol = ts["country"].value_counts().drop("Unknown", errors="ignore")
    share = vol / vol.sum() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Countries (min volume)", len(stats))
    c2.metric("Top-1 share", f"{share.iloc[0]:.1f}%", share.index[0])
    c3.metric("Top-5 concentration", f"{share.head(5).sum():.1f}%")

    st.subheader("Where the demand is")
    st.bar_chart(share.head(15).rename("% of postings"), height=320)

    st.subheader("Median hourly rate offered, by client country")
    n = st.slider("Countries to show", 10, len(stats), min(25, len(stats)))
    st.bar_chart(stats.sort_values("median_rate", ascending=False)["median_rate"].head(n), height=380)
    st.dataframe(stats.sort_values("median_rate", ascending=False), use_container_width=True)

    st.subheader("Posting activity by hour (UTC)")
    hourly = ts.groupby("post_hour").size()
    st.bar_chart(hourly.rename("postings"), height=260)
    st.caption(
        f"Peak-to-trough ratio of only {hourly.max() / hourly.min():.1f}x across 24 hours. "
        f"The remote market never fully sleeps -- structural evidence of genuinely "
        f"distributed global demand rather than a single-timezone marketplace.")


# ================================================================ JOB SEARCH
elif page == "Job Search":
    st.title("Job Search")
    rec = load_recommender(df)
    ev = rec.evaluate(k=10, n_queries=150)

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision@10", f"{ev['precision@10']:.3f}")
    c2.metric("Random baseline", f"{ev['random_baseline']:.3f}")
    c3.metric("Lift", f"{ev['lift_vs_random']:.1f}x")
    st.caption(
        "Measured performance, not an assumption that cosine similarity works. "
        "Relevance is proxied by the derived category, which carries its own error "
        "rate -- so this measures topical coherence, not whether a human would want "
        "the job. The engine sees only job TITLES: no description, skills or client "
        "rating, so it cannot judge job quality.")

    query = st.text_input("Skills or role", "python data analyst power bi")
    c1, c2, c3 = st.columns(3)
    cat = c1.selectbox("Category filter", ["Any"] + sorted(df["category"].unique()))
    min_rate = c2.number_input("Min hourly rate ($)", 0.0, 200.0, 0.0, 5.0)
    k = c3.slider("Results", 5, 50, 15)

    if query:
        out = rec.recommend(query, k=k,
                            category=None if cat == "Any" else cat,
                            min_rate=min_rate if min_rate > 0 else None)
        if out.empty:
            st.info("No matches. Try broader terms or relax the filters.")
        else:
            show = out[["score", "title_clean", "category", "pay_type",
                        "hourly_mid", "budget", "country", "post_date", "link"]].copy()
            show.columns = ["score", "title", "category", "type", "$/hr",
                            "budget", "country", "posted", "link"]
            st.dataframe(show, use_container_width=True, hide_index=True,
                         column_config={"link": st.column_config.LinkColumn("link", display_text="open")})


# =============================================================== DATA QUALITY
elif page == "Data Quality":
    st.title("Data Quality")
    st.caption(
        "Cleaning design principle: FLAG, DON'T DELETE. Rows carry quality flags "
        "rather than being removed, so each analysis chooses its own exclusions and "
        "every decision stays auditable. Only null-title rows were dropped.")

    st.subheader("The backfill artifact")
    allday = df.groupby("post_date").size()
    clean = ts.groupby("post_date").size()
    c1, c2 = st.columns(2)
    c1.write("**RAW** — apparent explosive growth")
    c1.line_chart(allday, height=260)
    c2.write("**CLEANED** — stable market, weekly cycle")
    c2.line_chart(clean, height=260)
    st.error(
        f"The raw file holds 283 rows across the 48 days before "
        f"{CFG.COLLECTION_START} (~6/day), then ~5,960/day after. Those early rows "
        f"are scraper backfill, not a quiet market. Plotting raw volume 'discovers' "
        f"~100,000% growth in February that is entirely an artifact of when "
        f"collection was switched on. This is the single most consequential "
        f"finding in the dataset -- it invalidates every time-series result "
        f"computed on raw data.")

    st.subheader("Exclusion flags")
    flags = pd.DataFrame({
        "flag": ["Scraper backfill (pre-collection)", "Collection outage (2024-02-15)",
                 "Truncated final day (2024-03-24)", "-> Usable for time series",
                 "No pay disclosed", "Implausible hourly rate", "Extreme budget",
                 "-> Usable for pay analysis"],
        "rows": [int((~df["in_collection_window"]).sum()), int(df["is_outage_day"].sum()),
                 int(df["is_partial_day"].sum()), int(df["ts_reliable"].sum()),
                 int((~df["pay_disclosed"]).sum()), int(df["hourly_implausible"].sum()),
                 int(df["budget_extreme"].sum()), int(df["pay_analysable"].sum())],
    })
    flags["pct"] = (flags["rows"] / len(df) * 100).round(2)
    st.dataframe(flags, use_container_width=True, hide_index=True)

    st.info(
        "**Missing pay is not imputed.** 38,514 postings (15.7%) disclose no rate. "
        "Filling these with a mean would invent a sixth of the pay data and pull "
        "every statistic toward the centre. They are treated as the informative "
        "category 'rate undisclosed'.")

    audit = DATA_DIR / "cleaning_audit.csv"
    if audit.exists():
        st.subheader("Cleaning audit trail")
        st.dataframe(pd.read_csv(audit), use_container_width=True, hide_index=True)

    val = load_table("taxonomy_validation")
    if val is not None:
        st.subheader("Taxonomy precision (measured, not assumed)")
        st.dataframe(val.groupby("tier")["agrees"].agg(["mean", "size"]).round(3),
                     use_container_width=True)
        st.caption(
            "Each sampled title was re-checked against INDEPENDENT anchor terms, not "
            "the rules that assigned it. Conservative estimate: a mismatch may mean a "
            "wrong label OR merely unusual phrasing.")
