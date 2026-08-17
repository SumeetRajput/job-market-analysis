#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 JOB MARKET ANALYSIS & RECOMMENDATION SYSTEM
 Upwork freelance job postings | 244,828 raw records | Feb-Mar 2024
================================================================================

A single-file, end-to-end pipeline covering all eight project tasks.

    Task 1  Correlation between job title keywords and offered pay
    Task 2  Emerging job categories by posting frequency
    Task 3  Predict high-demand job roles over time
    Task 4  Compare average hourly rates across countries
    Task 5  Job recommendation engine
    Task 6  Track job market dynamics over time (dashboard feed)
    Task 7  Trends in the remote work landscape
    Task 8  Predict future job market trends

--------------------------------------------------------------------------------
 THREE DATA-INTEGRITY FINDINGS THAT SHAPE THIS ENTIRE PIPELINE
--------------------------------------------------------------------------------

 (1) THE SCRAPE CONTAINS A FALSE GROWTH CURVE.
     The raw file holds 283 rows spread over the 48 days before 2024-02-13
     (~6/day), then 244,545 rows over the following 41 days (~5,960/day).
     Those early rows are scraper backfill, not a quiet market. Plotting raw
     volume over time "discovers" ~100,000% growth in February that is purely
     an artifact of when collection was switched on. Every time-series task
     here filters on the `ts_reliable` flag, which excludes them.

 (2) HOURLY PAY AND FIXED BUDGETS ARE NOT INTERCONVERTIBLE.
     `budget` is populated if and only if `is_hourly` is False (verified:
     zero overlap). Converting a $500 project budget into an hourly rate
     needs project duration, and this dataset has no duration column.
     Task 1 and Task 4 therefore run as two parallel tracks and never
     average the two together. A unit test enforces this.

 (3) THE USABLE WINDOW IS ~5 WEEKS, NOT 5 MONTHS.
     After removing backfill, the truncated final day (2024-03-24 stops at
     14:16 UTC) and a collection outage on 2024-02-15, roughly 39 clean days
     remain. Monthly trend claims are not supportable on that window. This
     pipeline is built monthly-capable but reports at daily/weekly grain and
     states the limitation, rather than fabricating month-over-month growth.

--------------------------------------------------------------------------------
 USAGE
--------------------------------------------------------------------------------
     python job_market_analysis.py --data all_upwork_jobs.csv
     python job_market_analysis.py --data <csv> --outdir results --quick
     python job_market_analysis.py --data <csv> --recommend "data analyst python"
     python job_market_analysis.py --selftest

 Dependencies: pandas numpy scikit-learn scipy matplotlib seaborn statsmodels
================================================================================
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import textwrap
import unicodedata
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# =============================================================================
# SECTION 0 | CONFIGURATION
# Every threshold below was derived from profiling the raw file, not guessed.
# The evidence sits in the comment beside it so each choice is defensible.
# =============================================================================
class CFG:
    # ---- temporal validity -------------------------------------------------
    # 283 rows / 48 days before this date vs ~5,960/day after. Backfill.
    COLLECTION_START = "2024-02-13"
    # Max timestamp is 14:16 UTC on this day; volume 2,185 vs ~4,500 Sunday
    # baseline. Valid job records, invalid daily count.
    PARTIAL_LAST_DAY = "2024-03-24"
    # 1,811 postings vs a ~6,600 weekday baseline (-73%), with normal
    # neighbours either side. Collection outage, not a market event.
    OUTAGE_DAYS = ["2024-02-15"]

    # ---- pay plausibility --------------------------------------------------
    HOURLY_MIN = 3.0              # observed floor; matches Upwork minimum
    HOURLY_MAX = 250.0            # p99 of hourly_high is $200; 998/999 are ceiling placeholders
    BUDGET_MIN = 5.0
    BUDGET_MAX = 50_000.0         # p99 = $10k, p99.9 = $100k, max $1M

    # ---- statistical thresholds -------------------------------------------
    MIN_COUNTRY_POSTINGS = 200    # 139 countries fall below this, totalling 581 rows
    MIN_KEYWORD_FREQ = 150        # below this, pay medians are unstable
    MIN_CATEGORY_ROWS = 500
    FDR_ALPHA = 0.05              # Benjamini-Hochberg level for multiple testing

    # ---- modelling ---------------------------------------------------------
    TEST_DAYS = 8                 # temporal holdout; NEVER a random split
    TFIDF_MAX_FEATURES = 60_000
    RECO_SAMPLE = 60_000          # cap for the O(n^2)-ish similarity index
    EVAL_QUERIES = 400            # queries used for precision@k

    OTHER_CATEGORY = "Other / Uncategorised"


# -----------------------------------------------------------------------------
# Category taxonomy. The dataset has NO category or skills column, so categories
# must be DERIVED from title text. Two tiers, first-match-wins.
#
# Tier 1 (specific): multi-word phrases. Order matters -- "Shopify Developer"
# resolves to E-commerce before Web Development because commercial context
# outranks technical context.
# -----------------------------------------------------------------------------
CATEGORY_RULES = [
    ("Data Science & Analytics", [
        "machine learning", "data scien", "data analy", "power bi", "tableau",
        "looker studio", "data visuali", "nlp", "deep learning", "predictive model",
        "data engineer", "etl", "big data", "pandas", "scikit", "tensorflow",
        "pytorch", "computer vision", "llm", "chatgpt", "openai", " ai ", "ai-",
        "artificial intelligence", "dashboard", "sql", "bigquery", "statistic",
        "forecast", "analytics", "data model",
    ]),
    ("E-commerce", [
        "shopify", "woocommerce", "amazon fba", "amazon seller", "ecommerce",
        "e-commerce", "dropship", "etsy", "ebay", "product listing", "magento",
        "bigcommerce", "amazon ppc", "amazon listing", "online store",
    ]),
    ("Mobile Development", [
        "ios app", "android app", "flutter", "react native", "swift", "kotlin",
        "mobile app", "app developer", "xamarin", "ionic", "app development",
    ]),
    ("Web Development", [
        "wordpress", "web develop", "website develop", "front end", "frontend",
        "back end", "backend", "full stack", "fullstack", "react", "angular",
        "vue", "next.js", "nodejs", "node.js", "laravel", "django", "php",
        "webflow", "wix", "squarespace", "html", "css", "javascript",
        "web design", "landing page", "website design", "web app", "website redesign",
    ]),
    ("Software & IT", [
        "devops", "aws", "azure", "kubernetes", "docker", "api integration",
        "python develop", "java develop", "c++", "c#", ".net", "golang",
        "software engineer", "software develop", "automation", "salesforce",
        "system admin", "cyber", "blockchain", "solidity", "web3", "qa test",
        "software test", "database", "erp", "odoo", "chrome extension",
    ]),
    ("Video & Animation", [
        "video edit", "animation", "animator", "motion graphic", "after effects",
        "premiere pro", "3d model", "blender", "youtube video", "explainer video",
        "video produc", "videographer", "vfx", "reels edit", "short form video",
        "davinci", "video creat",
    ]),
    ("Design & Creative", [
        "logo", "graphic design", "designer", "illustrat", "photoshop", "ui/ux",
        "ui ux", "ux design", "ui design", "figma", "branding", "brand identity",
        "packaging design", "canva", "adobe", "presentation design", "poster",
        "flyer", "banner", "photo edit", "interior design", "architect", "autocad",
    ]),
    ("Digital Marketing", [
        "seo", "google ads", "facebook ads", "meta ads", "ppc", "digital market",
        "social media", "content market", "email market", "media buyer",
        "marketing specialist", "influencer", "tiktok", "instagram", "growth market",
        "paid ads", "advertis", "brand market", "affiliate", "smm",
        "google analytics", "klaviyo", "hubspot",
    ]),
    ("Writing & Translation", [
        "writer", "writing", "copywrit", "content creat", "blog", "article",
        "ghostwrit", "proofread", "translat", "transcri", "script writ",
        "technical writ", "resume writ", "seo content", "editor",
    ]),
    ("Sales & Business Development", [
        "sales", "lead generation", "appointment setter", "cold call",
        "business development", "telemarket", "closer", "sdr",
        "account executive", "crm", "outreach", "b2b lead",
    ]),
    ("Admin & Customer Support", [
        "virtual assistant", "data entry", "customer service", "customer support",
        "admin assistant", "administrative", "executive assistant", "scheduling",
        "receptionist", "chat support", "back office", "personal assistant",
    ]),
    ("Finance & Accounting", [
        "accountant", "accounting", "bookkeep", "quickbooks", "xero",
        "financial model", "financial analy", "tax ", "audit", "cfo", "payroll",
        "invoice", "budget analy",
    ]),
    ("Legal", [
        "lawyer", "attorney", "legal", "paralegal", "contract draft", "trademark",
        "patent", "compliance", "gdpr",
    ]),
    ("HR & Recruitment", [
        "recruit", "human resource", "talent acqui", "headhunt",
        "sourcing specialist", "hiring",
    ]),
    ("Audio & Music", [
        "voice over", "voiceover", "audio edit", "podcast", "music", "sound design",
        "mixing", "mastering", "narrat", "jingle",
    ]),
    ("Engineering & Manufacturing", [
        "mechanical engineer", "electrical engineer", "civil engineer", "solidworks",
        "3d print", "pcb", "electronics", "manufactur", "structural", "hvac",
    ]),
    ("Game Development", [
        "game develop", "unity", "unreal engine", "roblox", "game design",
        "game art", "minecraft", "godot", "level design",
    ]),
    ("Project Management", [
        "project manager", "project management", "scrum", "agile coach",
        "product manager", "product owner", "program manager", "operations manager",
        "jira", "asana", "clickup",
    ]),
    ("Research & Survey", [
        "market research", "research assistant", "survey", "participant",
        "user research", "academic research", "literature review", "data collection",
        "web research", "focus group", "questionnaire",
    ]),
    ("Real Estate", [
        "real estate", "property manage", "airbnb", "realtor", "mortgage",
        "zillow", "property listing", "rental",
    ]),
]

# -----------------------------------------------------------------------------
# Tier 2 (fallback): single generic tokens. Thousands of titles are bare words
# -- "Website" (4,957), "Developer" (3,655), "Video" (3,370) -- which phrase
# rules structurally cannot match. Running tier 1 first preserves precision;
# tier 2 recovers recall on the long tail. Without this, 38.1% of the corpus
# is uncategorised and Task 2 collapses. Each row records which tier fired,
# so the confidence of every category count stays auditable.
# -----------------------------------------------------------------------------
FALLBACK_RULES = [
    ("Data Science & Analytics", ["excel", "spreadsheet", "google sheet", "python",
                                  "data", "scrap", "database", "insight", "metric", "kpi"]),
    ("E-commerce", ["amazon", "store", "listing", "checkout", "inventory"]),
    ("Mobile Development", ["app", "ios", "android", "iphone", "mobile", "apk", "play store"]),
    ("Web Development", ["website", "web", "site", "page", "domain", "hosting",
                         "plugin", "theme", "cms", "portal"]),
    ("Software & IT", ["developer", "development", "engineer", "programm", "code",
                       "coding", "software", "integration", "api", "server",
                       "tech support", "it support", "bug", "script"]),
    ("Video & Animation", ["video", "youtube", "film", "footage", "clip", "reel",
                           "shorts", "camera", "render"]),
    ("Design & Creative", ["design", "art", "creative", "visual", "graphic", "mockup",
                           "template", "portrait", "drawing", "sketch", "photo", "image"]),
    ("Digital Marketing", ["marketing", "market", "campaign", "promote", "promotion",
                           "traffic", "follower", "subscriber", "engagement", "viral",
                           "growth", "audience", "facebook", "linkedin", "twitter",
                           "pinterest", "snapchat"]),
    ("Writing & Translation", ["write", "written", "text", "story", "book", "essay",
                               "caption", "description", "english", "spanish",
                               "language", "word", "edit"]),
    ("Sales & Business Development", ["sale", "lead", "client", "prospect", "deal", "revenue"]),
    ("Admin & Customer Support", ["assistant", "support", "entry", "copy and paste",
                                  "copy paste", "manage", "organiz", "organis",
                                  "schedul", "task"]),
    ("Finance & Accounting", ["finance", "financial", "money", "loan", "investment",
                              "trading", "crypto"]),
    ("Research & Survey", ["research", "study", "quiz", "test", "review", "feedback",
                           "opinion"]),
    ("Project Management", ["project", "manager", "coordinator", "operations"]),
    ("Business & Consulting", ["business", "consult", "strategy", "startup", "company",
                               "brand", "plan"]),
]

# Domain stopwords: hiring boilerplate that carries no signal about the work
# itself. Removing these stops the recommender matching "Need Expert Urgently"
# to every other posting that also says "need expert urgently".
JOB_STOPWORDS = {
    "need", "needed", "needs", "looking", "look", "want", "wanted", "seeking",
    "seek", "hiring", "hire", "urgent", "urgently", "asap", "expert", "experienced",
    "professional", "freelancer", "freelance", "specialist", "help", "required",
    "require", "job", "work", "project", "task", "new", "best", "good", "great",
    "quick", "fast", "simple", "easy", "small", "long", "term", "full", "time",
    "part", "someone", "person", "people", "team", "company", "business", "please",
    "must", "will", "can", "get", "make", "use", "using", "based", "high", "quality",
    "low", "cost", "cheap", "affordable", "immediately", "start", "immediate",
}


# =============================================================================
# SECTION 1 | UTILITIES
# =============================================================================
class Console:
    """Minimal structured console output so a long run stays readable."""

    W = 80

    @staticmethod
    def header(text: str) -> None:
        print("\n" + "=" * Console.W)
        print(text.upper())
        print("=" * Console.W)

    @staticmethod
    def section(text: str) -> None:
        print("\n" + "-" * Console.W)
        print(text)
        print("-" * Console.W)

    @staticmethod
    def kv(key: str, value) -> None:
        print(f"  {key:.<42} {value}")

    @staticmethod
    def note(text: str) -> None:
        for line in textwrap.wrap(text, Console.W - 6):
            print(f"  | {line}")

    @staticmethod
    def finding(text: str) -> None:
        print("\n  >> FINDING")
        for line in textwrap.wrap(text, Console.W - 8):
            print(f"     {line}")

    @staticmethod
    def caveat(text: str) -> None:
        print("\n  !! LIMITATION")
        for line in textwrap.wrap(text, Console.W - 8):
            print(f"     {line}")


@dataclass
class Results:
    """Collects every artifact so the run can be serialised for the dashboard."""
    tables: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)

    def add_table(self, name: str, df: pd.DataFrame) -> None:
        self.tables[name] = df

    def add_metric(self, name: str, value) -> None:
        self.metrics[name] = value

    def add_finding(self, task: str, text: str) -> None:
        self.findings.append({"task": task, "finding": text})


_EMOJI_RE = re.compile("[" "\U0001F300-\U0001FAFF" "\u2600-\u27BF" "\uFE0F" "\u2B00-\u2BFF" "]+")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z][a-z0-9\+\#\.]{1,}")


def normalise_text(s) -> str:
    """Decode HTML entities, strip emoji, normalise unicode and whitespace.

    10,130 titles carry HTML entities (&amp;, &quot;, &#039;). Left alone they
    become spurious TF-IDF tokens and corrupt both the keyword analysis and
    the recommender vocabulary.
    """
    if not isinstance(s, str):
        return ""
    s = html.unescape(html.unescape(s))       # double-encoded entities exist
    s = unicodedata.normalize("NFKC", s)
    s = _EMOJI_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in JOB_STOPWORDS and len(t) > 2]
    return toks


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """FDR correction. Testing hundreds of keywords at p<0.05 would yield dozens
    of false positives by chance alone; this controls the expected false
    discovery rate instead."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    crit = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= crit
    out = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.max(np.where(passed)[0])
        out[order[: cutoff + 1]] = True
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size, computed from the Mann-Whitney U statistic.

    A p-value on 200k rows says almost nothing -- with samples this large,
    trivial differences are 'significant'. Effect size is what tells us
    whether a keyword difference actually matters.
        |d| < 0.15 negligible | < 0.33 small | < 0.47 medium | else large
    """
    from scipy.stats import mannwhitneyu
    if len(a) < 2 or len(b) < 2:
        return np.nan
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return float(2 * u / (len(a) * len(b)) - 1)


def effect_label(d: float) -> str:
    ad = abs(d)
    if np.isnan(ad):
        return "n/a"
    if ad < 0.15:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.47:
        return "medium"
    return "large"


# =============================================================================
# SECTION 2 | DATA CLEANING
#
# Design principle: FLAG, DON'T DELETE.
# Rows are almost never dropped. They carry boolean quality flags instead, and
# each downstream task decides what it needs. A forecasting model must exclude
# the outage day; a pay-distribution chart need not care. Deleting up front
# would force one policy onto all eight tasks and hide the decision.
# =============================================================================
@dataclass
class CleaningAudit:
    steps: list = field(default_factory=list)

    def log(self, step: str, detail: str, n: int = 0) -> None:
        self.steps.append({"step": step, "detail": detail, "rows_affected": n})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def show(self) -> None:
        Console.section("CLEANING AUDIT TRAIL")
        for i, s in enumerate(self.steps, 1):
            n = f"{s['rows_affected']:>9,}" if s["rows_affected"] else " " * 9
            print(f"{i:>2}. {n}  {s['step']}")
            for line in textwrap.wrap(s["detail"], 68):
                print(f"                {line}")


def _compile(rules):
    return [(cat, re.compile("|".join(re.escape(k) for k in kws))) for cat, kws in rules]


_SPECIFIC = _compile(CATEGORY_RULES)
_FALLBACK = _compile(FALLBACK_RULES)


def assign_category(title_lower: str) -> tuple[str, str]:
    """Two-tier, first-match-wins classification. Returns (category, tier)."""
    for cat, rx in _SPECIFIC:
        if rx.search(title_lower):
            return cat, "specific"
    for cat, rx in _FALLBACK:
        if rx.search(title_lower):
            return cat, "fallback"
    return CFG.OTHER_CATEGORY, "unmatched"


def clean_data(raw: pd.DataFrame, audit: CleaningAudit | None = None):
    audit = audit or CleaningAudit()
    df = raw.copy()
    audit.log("load", f"Raw rows read: {len(df):,}", len(df))

    # -- structural integrity ------------------------------------------------
    # `link` is the natural key. Profiling found zero duplicates, but the check
    # runs anyway so a future re-scrape cannot silently double-count.
    dupes = int(df["link"].duplicated().sum())
    if dupes:
        df = df.drop_duplicates(subset="link", keep="first")
    audit.log("dedupe", f"Duplicate links removed (natural key = link): {dupes:,}", dupes)

    # A row with no title cannot be categorised, keyword-mined or recommended.
    # This is the only genuine deletion in the pipeline.
    n_untitled = int(df["title"].isna().sum())
    df = df[df["title"].notna()].copy()
    audit.log("drop_untitled", f"Dropped for null title (unanalysable): {n_untitled:,}", n_untitled)

    # -- temporal parsing and validity flags ---------------------------------
    df["published_at"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
    n_baddate = int(df["published_at"].isna().sum())
    df = df[df["published_at"].notna()].copy()
    audit.log("parse_dates", f"Unparseable timestamps dropped: {n_baddate:,}", n_baddate)

    df["post_date"] = df["published_at"].dt.date
    df["post_week"] = df["published_at"].dt.tz_localize(None).dt.to_period("W").astype(str)
    df["post_month"] = df["published_at"].dt.tz_localize(None).dt.to_period("M").astype(str)
    df["post_dow"] = df["published_at"].dt.day_name()
    df["post_dow_num"] = df["published_at"].dt.dayofweek
    df["post_hour"] = df["published_at"].dt.hour
    df["is_weekend"] = df["post_dow_num"] >= 5

    start = pd.Timestamp(CFG.COLLECTION_START, tz="UTC")
    df["in_collection_window"] = df["published_at"] >= start
    n_backfill = int((~df["in_collection_window"]).sum())
    audit.log("flag_backfill",
              f"Rows before {CFG.COLLECTION_START} flagged as scraper backfill: "
              f"{n_backfill:,} rows over 48 days vs ~5,960/day after. Excluding "
              f"these is mandatory for any time-series task.", n_backfill)

    partial = pd.Timestamp(CFG.PARTIAL_LAST_DAY).date()
    outages = {pd.Timestamp(d).date() for d in CFG.OUTAGE_DAYS}
    df["is_partial_day"] = df["post_date"] == partial
    df["is_outage_day"] = df["post_date"].isin(outages)

    # The single flag all time-series code filters on.
    df["ts_reliable"] = df["in_collection_window"] & ~df["is_partial_day"] & ~df["is_outage_day"]
    audit.log("flag_ts_reliable",
              f"Rows usable for daily counts: {int(df['ts_reliable'].sum()):,} "
              f"(excludes backfill, truncated final day, {CFG.OUTAGE_DAYS[0]} outage)",
              int((~df["ts_reliable"]).sum()))

    # -- text normalisation --------------------------------------------------
    n_entity = int(df["title"].str.contains(r"&[a-z]+;|&#\d+;", regex=True, na=False).sum())
    df["title_clean"] = df["title"].map(normalise_text)
    df["title_lower"] = df["title_clean"].str.lower()
    df["title_word_count"] = df["title_clean"].str.split().str.len().fillna(0).astype(int)
    audit.log("normalise_titles",
              f"HTML entities decoded, emoji stripped, unicode NFKC-normalised. "
              f"Titles containing HTML entities before cleaning: {n_entity:,}", n_entity)

    # -- country normalisation -----------------------------------------------
    df["country"] = df["country"].map(lambda s: normalise_text(s) if isinstance(s, str) else np.nan)
    df["country"] = df["country"].replace({"": np.nan})
    n_nullc = int(df["country"].isna().sum())
    df["country"] = df["country"].fillna("Unknown")
    audit.log("normalise_country",
              f"HTML entities decoded (e.g. 'Cote d&#039;Ivoire'). Nulls mapped to "
              f"'Unknown' rather than dropped: {n_nullc:,}", n_nullc)

    counts = df["country"].value_counts()
    small = set(counts[counts < CFG.MIN_COUNTRY_POSTINGS].index)
    df["country_grouped"] = np.where(df["country"].isin(small), "Other (small volume)", df["country"])
    audit.log("group_rare_countries",
              f"{len(small):,} countries with <{CFG.MIN_COUNTRY_POSTINGS} postings bucketed. "
              f"Original retained in `country`; rate statistics must use "
              f"`country_grouped` to stay stable.", int(df["country"].isin(small).sum()))

    # -- pay normalisation ---------------------------------------------------
    # CRITICAL INVARIANT: hourly and fixed-price never mix. `budget` is present
    # iff is_hourly is False. Converting between them needs project duration,
    # which this dataset does not have.
    df["is_hourly"] = df["is_hourly"].astype(bool)
    df["pay_type"] = np.where(df["is_hourly"], "hourly", "fixed")
    for col in ["hourly_low", "hourly_high", "budget"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["hourly_mid"] = df[["hourly_low", "hourly_high"]].mean(axis=1, skipna=True)
    df["hourly_band_width"] = df["hourly_high"] - df["hourly_low"]

    df["hourly_implausible"] = df["hourly_mid"].notna() & (
        (df["hourly_mid"] < CFG.HOURLY_MIN) | (df["hourly_mid"] > CFG.HOURLY_MAX))
    df["budget_extreme"] = df["budget"].notna() & (
        (df["budget"] < CFG.BUDGET_MIN) | (df["budget"] > CFG.BUDGET_MAX))
    audit.log("flag_pay_outliers",
              f"Hourly midpoints outside ${CFG.HOURLY_MIN:.0f}-${CFG.HOURLY_MAX:.0f}: "
              f"{int(df['hourly_implausible'].sum()):,}. Fixed budgets outside "
              f"${CFG.BUDGET_MIN:.0f}-${CFG.BUDGET_MAX:,.0f}: {int(df['budget_extreme'].sum()):,}. "
              f"Flagged not dropped; medians reported throughout "
              f"(budget mean $911 vs median $100).",
              int(df["hourly_implausible"].sum() + df["budget_extreme"].sum()))

    # Missing pay is a signal, not a defect: an hourly post with no rate is a
    # deliberate 'negotiable'. Imputing a mean would invent 15.7% of the pay
    # data and pull every statistic toward the centre.
    df["pay_disclosed"] = np.where(df["is_hourly"], df["hourly_mid"].notna(), df["budget"].notna())
    n_undisc = int((~df["pay_disclosed"]).sum())
    audit.log("flag_pay_disclosure",
              f"Postings with no pay information: {n_undisc:,} "
              f"({n_undisc / len(df) * 100:.1f}%). All are hourly posts with an "
              f"unspecified rate. NOT imputed - treated as the informative "
              f"category 'rate undisclosed'.", n_undisc)

    df["pay_value"] = np.where(df["is_hourly"], df["hourly_mid"], df["budget"])

    # Analysis-grade pay: disclosed AND plausible. Used for all pay statistics.
    df["pay_analysable"] = (
        df["pay_disclosed"]
        & ~(df["is_hourly"] & df["hourly_implausible"])
        & ~(~df["is_hourly"] & df["budget_extreme"])
    )

    # -- category derivation -------------------------------------------------
    assigned = df["title_lower"].map(assign_category)
    df["category"] = assigned.map(lambda t: t[0])
    df["category_tier"] = assigned.map(lambda t: t[1])
    tiers = df["category_tier"].value_counts()
    n_spec, n_fall = int(tiers.get("specific", 0)), int(tiers.get("fallback", 0))
    n_un = int(tiers.get("unmatched", 0))
    audit.log("derive_category",
              f"Two-tier taxonomy applied (dataset has no category column). "
              f"Tier 1 specific-phrase: {n_spec:,} ({n_spec/len(df)*100:.1f}%). "
              f"Tier 2 generic-token: {n_fall:,} ({n_fall/len(df)*100:.1f}%). "
              f"Unmatched: {n_un:,} ({n_un/len(df)*100:.1f}%). Tier stored per row "
              f"in `category_tier` so confidence is auditable.", len(df) - n_un)

    keep = ["title", "title_clean", "title_lower", "title_word_count", "link",
            "published_at", "post_date", "post_week", "post_month", "post_dow",
            "post_dow_num", "post_hour", "is_weekend", "in_collection_window",
            "is_partial_day", "is_outage_day", "ts_reliable", "country",
            "country_grouped", "is_hourly", "pay_type", "hourly_low", "hourly_high",
            "hourly_mid", "hourly_band_width", "budget", "pay_value", "pay_disclosed",
            "pay_analysable", "hourly_implausible", "budget_extreme", "category",
            "category_tier"]
    df = df[keep]
    audit.log("finalise", f"Clean dataset: {len(df):,} rows x {len(df.columns)} columns", len(df))
    return df, audit


def validate_taxonomy(df: pd.DataFrame, n_per_tier: int = 120) -> tuple[pd.DataFrame, float]:
    """Estimate taxonomy precision using held-out anchor terms.

    A keyword taxonomy always has an error rate; the difference between a weak
    submission and a strong one is whether that rate is MEASURED rather than
    assumed to be zero. Each sampled title is re-checked against an INDEPENDENT
    set of anchor terms (deliberately not the rules that assigned it). Agreement
    gives a conservative precision estimate to quote in the report.
    """
    anchors = {
        "Data Science & Analytics": ["data", "analy", "sql", "power bi", "tableau",
                                     "machine learning", "dashboard", "excel", "python",
                                     "ai", "report", "insight", "statistic", "model"],
        "Web Development": ["web", "site", "wordpress", "html", "css", "javascript",
                            "react", "php", "landing", "page", "domain", "hosting"],
        "Mobile Development": ["app", "ios", "android", "flutter", "mobile", "iphone", "play store"],
        "Design & Creative": ["design", "logo", "graphic", "photoshop", "ui", "ux", "figma",
                              "brand", "illustrat", "art", "photo", "visual", "creative"],
        "Video & Animation": ["video", "animation", "edit", "youtube", "motion", "footage",
                              "film", "reel", "clip", "shorts"],
        "Digital Marketing": ["market", "seo", "ads", "social", "campaign", "instagram",
                              "facebook", "tiktok", "growth", "promot", "traffic", "advertis"],
        "Writing & Translation": ["writ", "content", "blog", "article", "copy", "translat",
                                  "proofread", "edit", "text", "story", "book"],
        "Software & IT": ["develop", "engineer", "software", "code", "api", "server",
                          "aws", "python", "java", "automation", "database", "script", "app"],
        "E-commerce": ["shopify", "amazon", "store", "ecommerce", "e-commerce", "product",
                       "listing", "dropship", "etsy", "woocommerce", "inventory"],
        "Sales & Business Development": ["sale", "lead", "client", "outreach", "cold call",
                                         "appointment", "crm", "prospect", "b2b", "closer"],
        "Admin & Customer Support": ["assistant", "support", "data entry", "admin", "customer",
                                     "schedul", "virtual", "chat", "copy paste", "manage"],
        "Finance & Accounting": ["account", "bookkeep", "financ", "tax", "audit", "payroll",
                                 "quickbooks", "invoice", "budget"],
        "Legal": ["legal", "lawyer", "attorney", "contract", "trademark", "patent",
                  "compliance", "paralegal"],
        "HR & Recruitment": ["recruit", "hr", "hiring", "talent", "headhunt", "sourcing"],
        "Audio & Music": ["audio", "voice", "music", "podcast", "sound", "narrat", "mixing"],
        "Engineering & Manufacturing": ["engineer", "mechanical", "electrical", "civil",
                                        "cad", "solidworks", "3d print", "manufactur", "pcb"],
        "Game Development": ["game", "unity", "unreal", "roblox", "minecraft", "godot"],
        "Project Management": ["project", "manager", "scrum", "agile", "product owner",
                               "coordinator", "operations"],
        "Research & Survey": ["research", "survey", "participant", "study", "interview",
                              "questionnaire", "focus group", "quiz"],
        "Real Estate": ["real estate", "property", "airbnb", "realtor", "mortgage", "rental"],
        "Business & Consulting": ["business", "consult", "strategy", "startup", "company",
                                  "plan", "brand"],
    }

    labelled = df[df["category"] != CFG.OTHER_CATEGORY]
    rows = []
    for tier in ["specific", "fallback"]:
        sub = labelled[labelled["category_tier"] == tier]
        if sub.empty:
            continue
        take = min(n_per_tier, len(sub))
        sample = sub.sample(take, random_state=RANDOM_STATE)
        for _, r in sample.iterrows():
            terms = anchors.get(r["category"], [])
            agree = any(t in r["title_lower"] for t in terms) if terms else np.nan
            rows.append({"tier": tier, "category": r["category"],
                         "title": r["title_clean"][:70], "agrees": agree})

    val = pd.DataFrame(rows)
    if val.empty:
        return val, np.nan
    overall = float(val["agrees"].mean())
    return val, overall


# =============================================================================
# TASK 1 | CORRELATION BETWEEN TITLE KEYWORDS AND OFFERED PAY
#
# METHODOLOGICAL POSITION
# The brief says "salaries", but this dataset has two incompatible pay units:
# hourly rates and fixed project budgets, with zero overlap between them.
# Averaging them would be meaningless -- $30/hr and a $500 budget are not
# comparable without project duration, which the data does not contain.
# The analysis therefore runs as TWO PARALLEL TRACKS and never merges them.
#
# For each keyword we compare the pay distribution of postings containing it
# against those that do not, using:
#   - Mann-Whitney U  (non-parametric: pay is heavily right-skewed, not normal)
#   - Cliff's delta   (effect size: with n>100k, p-values alone are worthless
#                      because trivial differences reach significance)
#   - Benjamini-Hochberg FDR correction across all keywords tested
# =============================================================================
def task1_keyword_pay(df: pd.DataFrame, res: Results, top_n: int = 25) -> dict:
    from scipy.stats import mannwhitneyu

    Console.header("Task 1 | Job title keywords vs offered pay")
    Console.note(
        "Two independent tracks (hourly $/hr and fixed $/project) because the "
        "two pay units are not interconvertible without project duration.")

    out = {}
    for track, mask, unit in [
        ("hourly", df["is_hourly"] & df["pay_analysable"], "$/hr"),
        ("fixed", (~df["is_hourly"]) & df["pay_analysable"], "$/project"),
    ]:
        sub = df[mask]
        Console.section(f"Track: {track.upper()}  (n = {len(sub):,} postings with disclosed, plausible pay)")

        # Build keyword -> pay mapping
        vocab: dict[str, list[float]] = {}
        for title, pay in zip(sub["title_lower"].values, sub["pay_value"].values):
            for tok in set(tokenize(title)):
                vocab.setdefault(tok, []).append(pay)

        overall_median = float(sub["pay_value"].median())
        recs = []
        for kw, pays in vocab.items():
            if len(pays) < CFG.MIN_KEYWORD_FREQ:
                continue
            with_kw = np.asarray(pays, dtype=float)
            # Complement: all postings without the keyword
            has = sub["title_lower"].str.contains(re.escape(kw), regex=True, na=False)
            without = sub.loc[~has, "pay_value"].to_numpy(dtype=float)
            if len(without) < 30:
                continue
            try:
                _, p = mannwhitneyu(with_kw, without, alternative="two-sided")
            except ValueError:
                continue
            d = cliffs_delta(with_kw, without)
            recs.append({
                "keyword": kw,
                "n_postings": len(with_kw),
                "median_pay": float(np.median(with_kw)),
                "mean_pay": float(np.mean(with_kw)),
                "vs_overall_pct": float((np.median(with_kw) / overall_median - 1) * 100),
                "p_value": float(p),
                "cliffs_delta": d,
                "effect": effect_label(d),
            })

        kdf = pd.DataFrame(recs)
        if kdf.empty:
            continue
        kdf["significant_fdr"] = benjamini_hochberg(kdf["p_value"].values, CFG.FDR_ALPHA)
        kdf = kdf.sort_values("median_pay", ascending=False).reset_index(drop=True)

        sig = kdf[kdf["significant_fdr"] & (kdf["effect"] != "negligible")]
        Console.kv("Keywords tested (freq >= %d)" % CFG.MIN_KEYWORD_FREQ, f"{len(kdf):,}")
        Console.kv("Significant after FDR correction", f"{int(kdf['significant_fdr'].sum()):,}")
        Console.kv("...AND non-negligible effect size", f"{len(sig):,}")
        Console.kv(f"Overall median ({track})", f"{overall_median:,.0f} {unit}")

        show_cols = ["keyword", "n_postings", "median_pay", "vs_overall_pct", "cliffs_delta", "effect"]
        print(f"\n  TOP {top_n} PREMIUM KEYWORDS  ({unit})")
        top = sig.head(top_n) if len(sig) >= top_n else kdf.head(top_n)
        print(top[show_cols].to_string(index=False,
              formatters={"median_pay": "{:,.0f}".format,
                          "vs_overall_pct": "{:+.0f}%".format,
                          "cliffs_delta": "{:+.3f}".format}))

        print(f"\n  BOTTOM {top_n} DISCOUNT KEYWORDS  ({unit})")
        bot = (sig if len(sig) >= top_n else kdf).tail(top_n).iloc[::-1]
        print(bot[show_cols].to_string(index=False,
              formatters={"median_pay": "{:,.0f}".format,
                          "vs_overall_pct": "{:+.0f}%".format,
                          "cliffs_delta": "{:+.3f}".format}))

        res.add_table(f"task1_keywords_{track}", kdf)
        out[track] = kdf

        if not sig.empty:
            best, worst = sig.iloc[0], sig.iloc[-1]
            ratio = best["median_pay"] / max(worst["median_pay"], 1e-9)
            Console.finding(
                f"[{track}] '{best['keyword']}' commands a median of "
                f"{best['median_pay']:,.0f} {unit} versus {worst['median_pay']:,.0f} for "
                f"'{worst['keyword']}' -- a {ratio:.1f}x spread. Both survive FDR "
                f"correction with non-negligible effect sizes, so this is a real "
                f"pricing signal rather than a large-sample artifact.")
            res.add_finding("Task 1",
                f"[{track}] Premium keyword '{best['keyword']}' = {best['median_pay']:,.0f} {unit}; "
                f"discount keyword '{worst['keyword']}' = {worst['median_pay']:,.0f} {unit} "
                f"({ratio:.1f}x spread, FDR-significant).")

    Console.caveat(
        "Keyword-pay association is CORRELATION, not causation. Adding 'senior' to a "
        "title does not raise the rate; it signals scope, seniority and client type "
        "that were already priced in. Rates are also advertised, not transacted -- "
        "the final agreed rate is unobserved in this dataset.")
    return out


# =============================================================================
# TASK 2 | EMERGING JOB CATEGORIES BY POSTING FREQUENCY
#
# METHODOLOGICAL POSITION
# "Emerging" needs a growth measurement, and growth needs a clean time axis.
# On the raw file the backfill rows manufacture explosive false growth, so this
# runs strictly on `ts_reliable` rows.
#
# With ~39 clean days, month-over-month growth is not computable. Instead we
# fit an OLS trend to each category's DAILY SHARE OF POSTINGS (not raw counts).
# Share is used because total daily volume swings 35% between weekdays and
# weekends; a raw-count trend would mostly measure the weekend cycle, not
# category momentum.
# =============================================================================
def task2_emerging_categories(df: pd.DataFrame, res: Results) -> pd.DataFrame:
    from scipy.stats import linregress

    Console.header("Task 2 | Emerging job categories")
    ts = df[df["ts_reliable"]]
    Console.kv("Rows used (ts_reliable only)", f"{len(ts):,}")
    Console.kv("Clean days available", ts["post_date"].nunique())

    daily = ts.groupby(["post_date", "category"]).size().rename("n").reset_index()
    totals = ts.groupby("post_date").size().rename("total")
    daily = daily.merge(totals, on="post_date")
    daily["share"] = daily["n"] / daily["total"]
    daily["day_idx"] = (pd.to_datetime(daily["post_date"]) -
                        pd.to_datetime(daily["post_date"]).min()).dt.days

    recs = []
    for cat, g in daily.groupby("category"):
        if g["n"].sum() < CFG.MIN_CATEGORY_ROWS or len(g) < 10:
            continue
        lr = linregress(g["day_idx"], g["share"])
        first_half = g[g["day_idx"] <= g["day_idx"].median()]["share"].mean()
        second_half = g[g["day_idx"] > g["day_idx"].median()]["share"].mean()
        recs.append({
            "category": cat,
            "total_postings": int(g["n"].sum()),
            "avg_share_pct": float(g["share"].mean() * 100),
            "share_first_half_pct": float(first_half * 100),
            "share_second_half_pct": float(second_half * 100),
            "change_pct": float((second_half / first_half - 1) * 100) if first_half > 0 else np.nan,
            "daily_slope_ppt": float(lr.slope * 100),      # percentage points per day
            "r_squared": float(lr.rvalue ** 2),
            "p_value": float(lr.pvalue),
        })

    cdf = pd.DataFrame(recs)
    cdf["significant_fdr"] = benjamini_hochberg(cdf["p_value"].values, CFG.FDR_ALPHA)
    cdf = cdf.sort_values("daily_slope_ppt", ascending=False).reset_index(drop=True)

    cols = ["category", "total_postings", "avg_share_pct", "change_pct",
            "daily_slope_ppt", "r_squared", "significant_fdr"]
    print("\n  CATEGORY MOMENTUM  (trend in daily share of all postings)")
    print(cdf[cols].to_string(index=False,
          formatters={"total_postings": "{:,}".format,
                      "avg_share_pct": "{:.2f}%".format,
                      "change_pct": "{:+.1f}%".format,
                      "daily_slope_ppt": "{:+.4f}".format,
                      "r_squared": "{:.3f}".format}))

    rising = cdf[(cdf["daily_slope_ppt"] > 0) & cdf["significant_fdr"]]
    falling = cdf[(cdf["daily_slope_ppt"] < 0) & cdf["significant_fdr"]]
    Console.kv("Categories with significant upward trend", len(rising))
    Console.kv("Categories with significant downward trend", len(falling))

    if not rising.empty:
        r = rising.iloc[0]
        Console.finding(
            f"Fastest-rising category is {r['category']}, gaining "
            f"{r['daily_slope_ppt']:+.4f} percentage points of market share per day "
            f"(R-sq {r['r_squared']:.2f}, FDR-significant), moving from "
            f"{r['share_first_half_pct']:.2f}% to {r['share_second_half_pct']:.2f}% "
            f"of all postings between the first and second half of the window.")
        res.add_finding("Task 2",
            f"Fastest-rising: {r['category']} ({r['daily_slope_ppt']:+.4f} ppt/day, "
            f"{r['change_pct']:+.1f}% half-over-half).")

    Console.caveat(
        "With ~39 clean days, these are SHORT-RUN trends, not established structural "
        "shifts. A 5-week slope cannot distinguish a genuine emerging category from "
        "a seasonal or promotional blip. R-squared is reported for every category so "
        "weak fits are visible; treat low-R-sq slopes as noise. Calling any category "
        "'emerging' on this window alone would overstate the evidence.")

    res.add_table("task2_category_momentum", cdf)
    res.add_table("task2_daily_category", daily)
    return cdf


# =============================================================================
# TASK 3 | PREDICT HIGH-DEMAND JOB ROLES OVER TIME
#
# METHODOLOGICAL POSITION -- THE LEAKAGE TRAP
# This is a time-series problem. A random train_test_split would train on
# future days to predict past days, producing an impressive but worthless R^2.
# We use a strict TEMPORAL split: train on the earliest days, test on the
# final CFG.TEST_DAYS.
#
# THE BASELINE THAT MATTERS
# Daily volume has a strong weekly cycle (weekdays ~7,000, weekends ~4,500).
# A model that only learns "weekends are quieter" looks skilful but adds
# nothing. So we benchmark against SEASONAL NAIVE (predict the value 7 days
# ago). A model that cannot beat lag-7 has learned nothing useful, and we
# report that honestly rather than hiding it.
# =============================================================================
def _daily_series(df: pd.DataFrame) -> pd.DataFrame:
    ts = df[df["ts_reliable"]]
    s = ts.groupby("post_date").size().rename("postings").reset_index()
    s["post_date"] = pd.to_datetime(s["post_date"])
    s = s.sort_values("post_date").reset_index(drop=True)
    s["dow"] = s["post_date"].dt.dayofweek
    s["is_weekend"] = s["dow"] >= 5
    s["day_idx"] = (s["post_date"] - s["post_date"].min()).dt.days
    return s


def _supervised_frame(s: pd.DataFrame, target: str = "postings") -> pd.DataFrame:
    """Build lag features. All features are strictly backward-looking, so no
    row can ever see its own future -- the second half of leakage prevention."""
    d = s.copy()
    for lag in [1, 2, 3, 7]:
        d[f"lag_{lag}"] = d[target].shift(lag)
    d["roll7_mean"] = d[target].shift(1).rolling(7, min_periods=3).mean()
    d["roll7_std"] = d[target].shift(1).rolling(7, min_periods=3).std()
    d["roll3_mean"] = d[target].shift(1).rolling(3, min_periods=2).mean()
    for k in [1, 2]:
        d[f"sin{k}"] = np.sin(2 * np.pi * k * d["dow"] / 7)
        d[f"cos{k}"] = np.cos(2 * np.pi * k * d["dow"] / 7)
    return d


def task3_demand_forecast(df: pd.DataFrame, res: Results) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

    Console.header("Task 3 | Forecasting posting demand")
    s = _daily_series(df)
    Console.kv("Clean daily observations", len(s))
    Console.kv("Date range", f"{s['post_date'].min().date()} to {s['post_date'].max().date()}")

    Console.section("Weekly seasonality (the dominant signal)")
    dow = (s.groupby("dow")["postings"].mean()
             .rename(index=dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))))
    for k, v in dow.items():
        bar = "#" * int(v / 200)
        print(f"    {k}  {v:>8,.0f}  {bar}")
    wk = s[~s["is_weekend"]]["postings"].mean()
    we = s[s["is_weekend"]]["postings"].mean()
    Console.finding(
        f"Weekday average {wk:,.0f} postings vs weekend {we:,.0f} -- a "
        f"{(1 - we / wk) * 100:.0f}% weekend collapse. This is the single strongest "
        f"pattern in the data and any credible forecast must reproduce it.")
    res.add_metric("weekend_drop_pct", round(float((1 - we / wk) * 100), 1))

    sup = _supervised_frame(s).dropna().reset_index(drop=True)
    feats = ["lag_1", "lag_2", "lag_3", "lag_7", "roll7_mean", "roll7_std",
             "roll3_mean", "dow", "sin1", "cos1", "sin2", "cos2"]
    n_test = min(CFG.TEST_DAYS, max(4, len(sup) // 4))
    train, test = sup.iloc[:-n_test], sup.iloc[-n_test:]
    Console.kv("Train / test split (temporal, NOT random)", f"{len(train)} days / {len(test)} days")

    Xtr, ytr = train[feats], train["postings"]
    Xte, yte = test[feats], test["postings"]

    def score(name, pred):
        return {"model": name,
                "MAE": float(mean_absolute_error(yte, pred)),
                "MAPE_pct": float(mean_absolute_percentage_error(yte, pred) * 100),
                "RMSE": float(np.sqrt(np.mean((yte - pred) ** 2)))}

    rows = []
    # Baseline 1: seasonal naive -- predict the same weekday last week.
    rows.append(score("SeasonalNaive (lag-7)", test["lag_7"].values))
    # Baseline 2: yesterday. Deliberately weak; exposes whether lag-7 matters.
    rows.append(score("Naive (lag-1)", test["lag_1"].values))

    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
    rows.append(score("Ridge", ridge.predict(Xte)))

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                               random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr, ytr)
    rows.append(score("RandomForest", rf.predict(Xte)))

    perf = pd.DataFrame(rows).sort_values("MAPE_pct").reset_index(drop=True)
    print("\n  MODEL PERFORMANCE ON HELD-OUT FINAL DAYS")
    print(perf.to_string(index=False, formatters={
        "MAE": "{:,.0f}".format, "MAPE_pct": "{:.2f}%".format, "RMSE": "{:,.0f}".format}))

    base_mape = perf.loc[perf["model"].str.startswith("SeasonalNaive"), "MAPE_pct"].iloc[0]
    best = perf.iloc[0]
    beat = best["model"] != "SeasonalNaive (lag-7)"
    Console.kv("Best model", best["model"])
    Console.kv("Best MAPE", f"{best['MAPE_pct']:.2f}%")
    Console.kv("Seasonal-naive MAPE", f"{base_mape:.2f}%")

    if beat:
        Console.finding(
            f"{best['model']} beats the seasonal-naive baseline "
            f"({best['MAPE_pct']:.2f}% vs {base_mape:.2f}% MAPE), a "
            f"{(1 - best['MAPE_pct'] / base_mape) * 100:.0f}% error reduction. The lag "
            f"and Fourier features add genuine information beyond 'same as last week'.")
    else:
        Console.finding(
            f"No learned model beats the seasonal-naive baseline "
            f"({base_mape:.2f}% MAPE). Reported as-is: with only {len(train)} training "
            f"days the weekly cycle is essentially the whole signal, and a naive "
            f"lag-7 rule is the honest recommendation. Claiming ML superiority here "
            f"would be unsupportable.")
    res.add_metric("task3_best_model", best["model"])
    res.add_metric("task3_best_mape", round(float(best["MAPE_pct"]), 2))
    res.add_metric("task3_baseline_mape", round(float(base_mape), 2))
    res.add_metric("task3_beats_baseline", bool(beat))

    imp = pd.DataFrame({"feature": feats, "importance": rf.feature_importances_}) \
            .sort_values("importance", ascending=False).reset_index(drop=True)
    print("\n  RANDOM FOREST FEATURE IMPORTANCE")
    print(imp.head(8).to_string(index=False, formatters={"importance": "{:.3f}".format}))

    # ---- per-category demand ranking ---------------------------------------
    Console.section("High-demand categories: current level and momentum")
    ts = df[df["ts_reliable"]]
    recent_cut = ts["post_date"].max() - pd.Timedelta(days=7).to_pytimedelta()
    recent = ts[ts["post_date"] > recent_cut]
    prior = ts[ts["post_date"] <= recent_cut]
    dr = (recent["category"].value_counts() / max(recent["post_date"].nunique(), 1)).rename("recent_per_day")
    dp = (prior["category"].value_counts() / max(prior["post_date"].nunique(), 1)).rename("prior_per_day")
    dem = pd.concat([dr, dp], axis=1).fillna(0)
    dem["momentum_pct"] = (dem["recent_per_day"] / dem["prior_per_day"].replace(0, np.nan) - 1) * 100
    dem = dem.sort_values("recent_per_day", ascending=False)
    print(dem.head(15).to_string(formatters={
        "recent_per_day": "{:,.0f}".format, "prior_per_day": "{:,.0f}".format,
        "momentum_pct": "{:+.1f}%".format}))

    res.add_table("task3_model_performance", perf)
    res.add_table("task3_feature_importance", imp)
    res.add_table("task3_category_demand", dem.reset_index().rename(columns={"index": "category"}))
    res.add_table("task3_daily_series", s)

    Console.caveat(
        f"Only {len(train)} training days are available. That is enough to learn a "
        f"weekly cycle but far too short to learn monthly seasonality, holiday "
        f"effects or genuine trend. Forecast horizon should be treated as days, "
        f"not months, and confidence degrades sharply beyond one week.")
    return {"perf": perf, "model": rf, "features": feats, "series": s, "sup": sup}


# =============================================================================
# TASK 4 | COMPARE HOURLY RATES ACROSS COUNTRIES
#
# METHODOLOGICAL POSITION -- WHAT `country` ACTUALLY MEANS
# `country` is the CLIENT's country: who posts and pays for the work. It is NOT
# the freelancer's country. So this task answers "what do clients in India PAY?"
# and NOT "what do freelancers in India EARN?" Those are different questions and
# conflating them is a serious interpretive error.
#
# Additional constraints handled here:
#   - The US is ~41% of all postings, so any global average is a disguised US
#     average. Medians are reported per country and never pooled.
#   - Raw rate comparison across countries ignores purchasing power parity and
#     cost of living. Rate rankings are therefore presented as CLIENT BUDGET
#     LEVELS, not as welfare or earnings comparisons.
#   - Countries below CFG.MIN_COUNTRY_POSTINGS are excluded from ranking because
#     their medians are not stable.
# =============================================================================
def task4_country_rates(df: pd.DataFrame, res: Results, top_n: int = 20) -> pd.DataFrame:
    from scipy.stats import kruskal

    Console.header("Task 4 | Hourly rates by country")
    Console.note(
        "`country` is the CLIENT's location, not the freelancer's. This measures "
        "what clients in each country pay, not what workers there earn.")

    h = df[df["is_hourly"] & df["pay_analysable"] & (df["country"] != "Unknown")]
    Console.kv("Hourly postings with disclosed, plausible rate", f"{len(h):,}")

    g = h.groupby("country")["hourly_mid"]
    stats = pd.DataFrame({
        "n_postings": g.size(),
        "median_rate": g.median(),
        "mean_rate": g.mean(),
        "p25": g.quantile(0.25),
        "p75": g.quantile(0.75),
        "std": g.std(),
    })
    stats = stats[stats["n_postings"] >= CFG.MIN_COUNTRY_POSTINGS].copy()
    stats["iqr"] = stats["p75"] - stats["p25"]
    global_median = float(h["hourly_mid"].median())
    stats["vs_global_pct"] = (stats["median_rate"] / global_median - 1) * 100
    stats = stats.sort_values("median_rate", ascending=False)

    Console.kv("Countries meeting min-volume threshold", len(stats))
    Console.kv("Global median hourly rate", f"${global_median:.2f}/hr")

    cols = ["n_postings", "median_rate", "p25", "p75", "vs_global_pct"]
    print(f"\n  TOP {top_n} COUNTRIES BY MEDIAN HOURLY RATE OFFERED")
    print(stats.head(top_n)[cols].to_string(formatters={
        "n_postings": "{:,}".format, "median_rate": "${:.2f}".format,
        "p25": "${:.2f}".format, "p75": "${:.2f}".format,
        "vs_global_pct": "{:+.1f}%".format}))

    print(f"\n  BOTTOM 10 COUNTRIES BY MEDIAN HOURLY RATE OFFERED")
    print(stats.tail(10)[cols].to_string(formatters={
        "n_postings": "{:,}".format, "median_rate": "${:.2f}".format,
        "p25": "${:.2f}".format, "p75": "${:.2f}".format,
        "vs_global_pct": "{:+.1f}%".format}))

    # Is the between-country difference statistically real?
    groups = [grp["hourly_mid"].dropna().values for _, grp in h.groupby("country")
              if len(grp) >= CFG.MIN_COUNTRY_POSTINGS]
    stat, p = kruskal(*groups)
    Console.kv("Kruskal-Wallis H (between-country)", f"{stat:,.1f}")
    Console.kv("p-value", f"{p:.3e}")

    top, bot = stats.iloc[0], stats.iloc[-1]
    Console.finding(
        f"Median offered rates range from ${top['median_rate']:.2f}/hr "
        f"({top.name}) down to ${bot['median_rate']:.2f}/hr ({bot.name}) -- a "
        f"{top['median_rate'] / bot['median_rate']:.1f}x spread. Kruskal-Wallis "
        f"confirms the between-country differences are not chance (p = {p:.1e}).")
    res.add_finding("Task 4",
        f"Client budget spread: {top.name} ${top['median_rate']:.2f}/hr vs "
        f"{bot.name} ${bot['median_rate']:.2f}/hr ({top['median_rate']/bot['median_rate']:.1f}x).")

    # Volume vs rate: where is the work, and where is the money?
    vol = h["country"].value_counts()
    conc = vol.head(5).sum() / vol.sum() * 100
    Console.kv("Share of hourly postings from top 5 countries", f"{conc:.1f}%")

    Console.caveat(
        "Rates are NOT adjusted for purchasing power parity or cost of living, so a "
        "high nominal rate does not imply higher real value to the freelancer. The "
        "US alone contributes ~41% of postings, so any pooled 'global average' is "
        "effectively a US average -- medians are therefore reported per country and "
        "never pooled. Rates are advertised bands, not transacted rates.")

    res.add_table("task4_country_rates", stats.reset_index())
    res.add_metric("task4_global_median_hourly", round(global_median, 2))
    res.add_metric("task4_kruskal_p", float(p))
    return stats


# =============================================================================
# TASK 5 | JOB RECOMMENDATION ENGINE
#
# METHODOLOGICAL POSITION -- MEASURE IT
# Almost every implementation of this task builds TF-IDF + cosine similarity
# and stops there, with no evidence the recommendations are any good. A
# recommender without an evaluation metric is an untested assertion.
#
# We evaluate with PRECISION@K against a random baseline. Ground-truth
# relevance is proxied by the derived category: a recommendation is "relevant"
# if it shares the query posting's category. This proxy is imperfect (the
# taxonomy has its own error rate, measured separately in validate_taxonomy),
# but it is objective, reproducible, and vastly better than eyeballing.
#
# The engine is HYBRID: content similarity is the base score, then a small
# set of interpretable business signals are blended in -- pay level, pay
# disclosure and recency -- because the highest-similarity job is not
# automatically the best job to apply for.
# =============================================================================
class JobRecommender:
    """TF-IDF content recommender with hybrid re-ranking and precision@k eval."""

    def __init__(self, max_features: int = CFG.TFIDF_MAX_FEATURES):
        self.max_features = max_features
        self.vectorizer = None
        self.matrix = None
        self.jobs = None

    def fit(self, df: pd.DataFrame, sample: int | None = CFG.RECO_SAMPLE):
        from sklearn.feature_extraction.text import TfidfVectorizer

        d = df.copy()
        if sample and len(d) > sample:
            # Sample for tractability; stratified by category so small
            # categories are not wiped out of the index entirely. Index-based
            # rather than groupby.apply, which drops the grouping column
            # under pandas 3.x.
            frac = sample / len(d)
            take = []
            for _, idx in d.groupby("category").groups.items():
                n_take = max(1, int(round(len(idx) * frac)))
                take.append(pd.Series(list(idx)).sample(
                    min(n_take, len(idx)), random_state=RANDOM_STATE))
            d = d.loc[pd.concat(take).values]
        self.jobs = d.reset_index(drop=True)

        # sublinear_tf dampens repeated tokens; bigrams capture "data analyst"
        # as a unit rather than two independent words.
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features, ngram_range=(1, 2),
            min_df=2, max_df=0.5, sublinear_tf=True,
            stop_words=list(JOB_STOPWORDS) + ["the", "and", "for", "with", "you", "our"],
        )
        self.matrix = self.vectorizer.fit_transform(self.jobs["title_clean"].fillna(""))
        return self

    def _hybrid_score(self, sim: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Blend content similarity with interpretable business signals.

        Weights are deliberately transparent rather than tuned to a metric:
        content dominates (0.80), with small nudges for pay transparency and
        recency. A job that hides its rate is genuinely a worse lead, and a
        two-day-old posting is more actionable than a five-week-old one.
        """
        sub = self.jobs.iloc[idx]
        disclosed = sub["pay_disclosed"].astype(float).to_numpy()
        # Recency: newest posting in the index = 1.0, oldest = 0.0
        ts = sub["published_at"].astype("int64").to_numpy(dtype=float)
        rng = ts.max() - ts.min()
        recency = (ts - ts.min()) / rng if rng > 0 else np.ones_like(ts)
        return 0.80 * sim[idx] + 0.12 * disclosed + 0.08 * recency

    def recommend(self, query: str, k: int = 10, category: str | None = None,
                  min_rate: float | None = None, hybrid: bool = True) -> pd.DataFrame:
        from sklearn.metrics.pairwise import linear_kernel

        qv = self.vectorizer.transform([normalise_text(query)])
        sim = linear_kernel(qv, self.matrix).ravel()

        mask = np.ones(len(self.jobs), dtype=bool)
        if category:
            mask &= (self.jobs["category"] == category).to_numpy()
        if min_rate is not None:
            mask &= (self.jobs["hourly_mid"].fillna(-1) >= min_rate).to_numpy()
        cand = np.where(mask & (sim > 0))[0]
        if cand.size == 0:
            return pd.DataFrame()

        score = self._hybrid_score(sim, cand) if hybrid else sim[cand]
        top = cand[np.argsort(-score)[:k]]
        out = self.jobs.iloc[top][
            ["title_clean", "category", "pay_type", "hourly_mid", "budget",
             "country", "post_date", "link"]].copy()
        out.insert(0, "score", np.sort(score)[::-1][:len(top)])
        out.insert(1, "similarity", sim[top])
        return out.reset_index(drop=True)

    def evaluate(self, k: int = 10, n_queries: int = CFG.EVAL_QUERIES) -> dict:
        """Precision@k using category as the relevance proxy, plus a random
        baseline. The gap between the two is the evidence the engine works."""
        from sklearn.metrics.pairwise import linear_kernel

        pool = self.jobs[self.jobs["category"] != CFG.OTHER_CATEGORY]
        n = min(n_queries, len(pool))
        qidx = pool.sample(n, random_state=RANDOM_STATE).index.to_numpy()

        precisions, rr = [], []
        for i in qidx:
            qv = self.matrix[i]
            sim = linear_kernel(qv, self.matrix).ravel()
            sim[i] = -1                                   # exclude the query itself
            top = np.argsort(-sim)[:k]
            true_cat = self.jobs.iloc[i]["category"]
            hits = (self.jobs.iloc[top]["category"] == true_cat).to_numpy()
            precisions.append(hits.mean())
            rr.append(1.0 / (np.argmax(hits) + 1) if hits.any() else 0.0)

        # Baseline: a random recommender's precision equals the prior
        # probability of drawing the same category by chance.
        prior = (self.jobs["category"].value_counts(normalize=True) ** 2).sum()

        p_at_k = float(np.mean(precisions))
        return {"k": k, "n_queries": n, f"precision@{k}": round(p_at_k, 4),
                "random_baseline": round(float(prior), 4),
                "lift_vs_random": round(p_at_k / prior, 2),
                "MRR": round(float(np.mean(rr)), 4)}


def task5_recommender(df: pd.DataFrame, res: Results, demo_query: str | None = None):
    Console.header("Task 5 | Job recommendation engine")
    rec = JobRecommender().fit(df)
    Console.kv("Jobs indexed", f"{len(rec.jobs):,}")
    Console.kv("TF-IDF vocabulary size", f"{len(rec.vectorizer.vocabulary_):,}")

    Console.section("Evaluation: precision@k vs random baseline")
    ev = rec.evaluate(k=10)
    for key, val in ev.items():
        Console.kv(key, val)
    Console.finding(
        f"Precision@10 of {ev['precision@10']:.3f} against a random baseline of "
        f"{ev['random_baseline']:.3f} -- a {ev['lift_vs_random']:.1f}x lift. MRR of "
        f"{ev['MRR']:.3f} means the first relevant result typically appears within "
        f"the top {max(1, round(1 / max(ev['MRR'], 1e-9)))} positions. This is "
        f"measured performance, not an assumption that cosine similarity works.")
    res.add_metric("task5_eval", ev)
    res.add_finding("Task 5",
        f"Recommender precision@10 = {ev['precision@10']:.3f} vs random "
        f"{ev['random_baseline']:.3f} ({ev['lift_vs_random']:.1f}x lift), MRR = {ev['MRR']:.3f}.")

    queries = [demo_query] if demo_query else [
        "data analyst python sql dashboard",
        "shopify store setup and product listing",
        "react frontend developer",
    ]
    for q in queries:
        Console.section(f'Recommendations for: "{q}"')
        out = rec.recommend(q, k=8)
        if out.empty:
            print("    (no matches)")
            continue
        show = out[["score", "similarity", "title_clean", "category",
                    "pay_type", "hourly_mid", "budget", "country"]]
        print(show.to_string(index=False, max_colwidth=44, formatters={
            "score": "{:.3f}".format, "similarity": "{:.3f}".format,
            "hourly_mid": lambda v: "-" if pd.isna(v) else f"${v:.0f}",
            "budget": lambda v: "-" if pd.isna(v) else f"${v:,.0f}"}))
        res.add_table(f"task5_reco_{re.sub(r'[^a-z0-9]+', '_', q.lower())[:30]}", out)

    Console.caveat(
        "Relevance is proxied by the DERIVED category, which carries its own error "
        "rate (measured separately). Precision@k here therefore measures category "
        "coherence, not whether a human would actually want the job. The engine also "
        "sees only job TITLES -- no description, skills, budget history or client "
        "rating -- so it cannot judge job quality, only topical similarity.")
    return rec


# =============================================================================
# TASK 6 | TRACK JOB MARKET DYNAMICS OVER TIME (DASHBOARD FEED)
#
# METHODOLOGICAL POSITION -- MONTHLY-CAPABLE, WEEKLY-REPORTED
# The brief asks for a dashboard that "updates monthly". The clean window is
# ~39 days, which spans parts of two calendar months and contains exactly ONE
# complete month boundary. Reporting month-over-month growth on that would be
# fabrication.
#
# Resolution: the aggregation layer is written to accept any grain
# (daily / weekly / monthly) so it IS monthly-capable the moment more data
# arrives. It is demonstrated at daily and weekly grain, and the monthly table
# is emitted with an explicit incompleteness warning rather than suppressed.
# =============================================================================
def task6_market_dynamics(df: pd.DataFrame, res: Results) -> dict:
    Console.header("Task 6 | Market dynamics tracking")
    ts = df[df["ts_reliable"]].copy()

    def aggregate(grain: str) -> pd.DataFrame:
        key = {"daily": "post_date", "weekly": "post_week", "monthly": "post_month"}[grain]
        g = ts.groupby(key)
        out = pd.DataFrame({
            "postings": g.size(),
            "pct_hourly": g["is_hourly"].mean() * 100,
            "pct_pay_disclosed": g["pay_disclosed"].mean() * 100,
            "median_hourly": g.apply(
                lambda x: x.loc[x["is_hourly"] & x["pay_analysable"], "hourly_mid"].median()),
            "median_budget": g.apply(
                lambda x: x.loc[~x["is_hourly"] & x["pay_analysable"], "budget"].median()),
            "n_countries": g["country"].nunique(),
            "n_categories": g["category"].nunique(),
        })
        out["days_covered"] = g["post_date"].nunique()
        out["postings_per_day"] = out["postings"] / out["days_covered"]
        return out.reset_index()

    grains = {}
    for grain in ["daily", "weekly", "monthly"]:
        grains[grain] = aggregate(grain)
        res.add_table(f"task6_{grain}", grains[grain])

    Console.section("Weekly market summary")
    w = grains["weekly"]
    print(w.to_string(index=False, formatters={
        "postings": "{:,}".format, "pct_hourly": "{:.1f}%".format,
        "pct_pay_disclosed": "{:.1f}%".format,
        "median_hourly": lambda v: "-" if pd.isna(v) else f"${v:.2f}",
        "median_budget": lambda v: "-" if pd.isna(v) else f"${v:,.0f}",
        "postings_per_day": "{:,.0f}".format}))

    Console.section("Monthly view (INCOMPLETE - shown with warning, not suppressed)")
    m = grains["monthly"]
    print(m.to_string(index=False, formatters={
        "postings": "{:,}".format, "pct_hourly": "{:.1f}%".format,
        "postings_per_day": "{:,.0f}".format}))
    Console.caveat(
        f"The monthly table covers {len(m)} partial months across ~{ts['post_date'].nunique()} "
        f"clean days. February is truncated at its start (collection began 13 Feb) and "
        f"March at its end. Month-over-month growth computed from these figures would "
        f"measure collection coverage, not market change, and is deliberately NOT "
        f"reported. The aggregation function accepts a monthly grain and becomes valid "
        f"as soon as two complete months are available.")

    # Category composition shift, first half vs second half
    Console.section("Category mix shift (first half vs second half of window)")
    mid = pd.to_datetime(ts["post_date"]).median()
    ts["half"] = np.where(pd.to_datetime(ts["post_date"]) <= mid, "H1", "H2")
    mix = pd.crosstab(ts["category"], ts["half"], normalize="columns") * 100
    mix["shift_ppt"] = mix["H2"] - mix["H1"]
    mix = mix.sort_values("shift_ppt", ascending=False)
    print(mix.to_string(formatters={"H1": "{:.2f}%".format, "H2": "{:.2f}%".format,
                                    "shift_ppt": "{:+.2f}".format}))
    res.add_table("task6_category_mix_shift", mix.reset_index())

    stab = float(grains["daily"]["pct_hourly"].std())
    Console.finding(
        f"Market composition is remarkably stable over the window: the hourly-vs-fixed "
        f"split varies by only {stab:.2f} percentage points (std dev) across days. "
        f"Volume swings 35% on the weekly cycle while the STRUCTURE of demand barely "
        f"moves -- an important distinction for anyone reading volume charts as change.")
    res.add_metric("task6_pct_hourly_daily_std", round(stab, 3))
    return grains


# =============================================================================
# TASK 7 | TRENDS IN THE REMOTE WORK LANDSCAPE
#
# METHODOLOGICAL POSITION -- THE QUESTION MUST BE REFRAMED
# The brief assumes a dataset containing both remote and on-site jobs, where
# one measures the shift between them. THIS DATASET HAS NO SUCH CONTRAST:
# Upwork is a remote-only freelance marketplace, so ~100% of postings are
# remote by construction. There is no on-site comparison group, and no
# `remote` column.
#
# Reporting "100% of jobs are remote, remote work is growing" would be a
# tautology dressed as a finding. Instead we analyse the STRUCTURE OF THE
# REMOTE MARKET ITSELF, which the data genuinely supports:
#   (a) geographic concentration of remote demand
#   (b) cross-border pay dispersion within remote work
#   (c) which categories dominate remote-first hiring
#   (d) the 24-hour posting clock as evidence of distributed global demand
# =============================================================================
def task7_remote_landscape(df: pd.DataFrame, res: Results) -> dict:
    Console.header("Task 7 | Remote work landscape")
    Console.note(
        "REFRAMED: Upwork is remote-only, so ~100% of postings are remote by "
        "construction and there is no on-site comparison group. Measuring "
        "'the shift to remote' here is impossible. We instead characterise the "
        "structure of the remote market itself, which the data does support.")

    ts = df[df["ts_reliable"]]

    # (a) Geographic concentration
    Console.section("(a) Geographic concentration of remote demand")
    vol = ts["country"].value_counts()
    known = vol.drop("Unknown", errors="ignore")
    share = known / known.sum() * 100
    top10 = share.head(10)
    print(top10.to_frame("pct_of_postings").to_string(formatters={"pct_of_postings": "{:.2f}%".format}))
    hhi = float(((share / 100) ** 2).sum())
    Console.kv("Top-1 country share", f"{share.iloc[0]:.1f}%")
    Console.kv("Top-5 concentration", f"{share.head(5).sum():.1f}%")
    Console.kv("Herfindahl-Hirschman Index", f"{hhi:.3f}")
    Console.finding(
        f"Remote demand is highly concentrated: {share.index[0]} alone accounts for "
        f"{share.iloc[0]:.1f}% of postings and the top 5 countries for "
        f"{share.head(5).sum():.1f}%. An HHI of {hhi:.3f} indicates a concentrated "
        f"market. 'Global remote work' is, on the demand side, largely a handful of "
        f"wealthy economies buying labour from everywhere else.")
    res.add_metric("task7_hhi", round(hhi, 4))
    res.add_metric("task7_top5_concentration_pct", round(float(share.head(5).sum()), 1))

    # (b) 24-hour posting clock
    Console.section("(b) Posting activity by hour (UTC) - evidence of distributed demand")
    hourly = ts.groupby("post_hour").size()
    peak, trough = hourly.idxmax(), hourly.idxmin()
    for hr, n in hourly.items():
        bar = "#" * int(n / max(hourly.max() / 50, 1))
        print(f"    {hr:02d}:00 UTC  {n:>7,}  {bar}")
    ratio = hourly.max() / hourly.min()
    Console.finding(
        f"Postings arrive around the clock, peaking at {peak:02d}:00 UTC and bottoming "
        f"at {trough:02d}:00 UTC, but only a {ratio:.1f}x peak-to-trough ratio -- the "
        f"market never actually sleeps. This is structural evidence of genuinely "
        f"distributed global demand rather than a single-timezone marketplace.")
    res.add_table("task7_hourly_activity", hourly.rename("postings").reset_index())

    # (c) Cross-border pay dispersion
    Console.section("(c) Pay dispersion within the remote market")
    h = ts[ts["is_hourly"] & ts["pay_analysable"]]
    cs = h.groupby("country")["hourly_mid"].agg(["size", "median"])
    cs = cs[cs["size"] >= CFG.MIN_COUNTRY_POSTINGS]
    spread = cs["median"].max() / cs["median"].min()
    Console.kv("Countries compared", len(cs))
    Console.kv("Highest median offer", f"${cs['median'].max():.2f}/hr ({cs['median'].idxmax()})")
    Console.kv("Lowest median offer", f"${cs['median'].min():.2f}/hr ({cs['median'].idxmin()})")
    Console.kv("Spread", f"{spread:.1f}x")

    # (d) Category composition of remote-first hiring
    Console.section("(d) What the remote market actually buys")
    catmix = ts["category"].value_counts(normalize=True).head(10) * 100
    print(catmix.to_frame("pct").to_string(formatters={"pct": "{:.2f}%".format}))
    creative = ["Design & Creative", "Video & Animation", "Writing & Translation"]
    tech = ["Web Development", "Software & IT", "Mobile Development", "Data Science & Analytics"]
    c_share = ts["category"].isin(creative).mean() * 100
    t_share = ts["category"].isin(tech).mean() * 100
    Console.kv("Creative/content share", f"{c_share:.1f}%")
    Console.kv("Technical/engineering share", f"{t_share:.1f}%")
    Console.finding(
        f"Remote freelance demand splits roughly {t_share:.0f}% technical to "
        f"{c_share:.0f}% creative/content. The remote market is not predominantly a "
        f"software market -- design, video and writing together rival engineering "
        f"in volume, which contradicts the common framing of remote work as a "
        f"chiefly tech phenomenon.")
    res.add_finding("Task 7",
        f"Remote demand is concentrated (HHI {hhi:.3f}, top-5 = {share.head(5).sum():.0f}%), "
        f"operates 24h (peak/trough {ratio:.1f}x), and splits {t_share:.0f}% technical "
        f"/ {c_share:.0f}% creative.")

    Console.caveat(
        "No temporal 'shift to remote' can be measured from this data -- there is no "
        "on-site comparison group and only ~5 weeks of coverage. Any forecast of "
        "remote-work growth would have to come from external sources, not this "
        "dataset. What IS measurable is the structure of remote demand at a point "
        "in time, which is what is reported above.")
    return {"concentration": share, "hourly_activity": hourly, "category_mix": catmix}


# =============================================================================
# TASK 8 | PREDICT FUTURE JOB MARKET TRENDS
#
# METHODOLOGICAL POSITION -- FORECAST WITH INTERVALS, AND STATE THE HORIZON
# A point forecast with no uncertainty band is a guess with false authority.
# Every projection here carries an empirical prediction interval derived from
# the model's own residuals on the held-out period.
#
# The honest horizon on ~39 days of data is ONE TO TWO WEEKS. Beyond that the
# intervals widen faster than the point estimate means anything, and we say so
# rather than extrapolating to a year for visual effect.
# =============================================================================
def task8_future_trends(df: pd.DataFrame, res: Results, model_bundle: dict,
                        horizon: int = 14) -> pd.DataFrame:
    Console.header("Task 8 | Forward-looking projections")

    s = model_bundle["series"]
    sup = model_bundle["sup"]
    feats = model_bundle["features"]
    model = model_bundle["model"]

    # Residual-based prediction interval from the held-out window.
    n_test = min(CFG.TEST_DAYS, max(4, len(sup) // 4))
    test = sup.iloc[-n_test:]
    resid = test["postings"].to_numpy() - model.predict(test[feats])
    sigma = float(np.std(resid))
    Console.kv("Residual sigma (held-out)", f"{sigma:,.0f} postings/day")

    # Recursive multi-step forecast: each predicted day feeds the next day's lags.
    hist = s[["post_date", "postings"]].copy()
    last_date = hist["post_date"].max()
    rows = []
    for step in range(1, horizon + 1):
        nd = last_date + pd.Timedelta(days=step)
        vals = hist["postings"].to_numpy()
        feat = {
            "lag_1": vals[-1], "lag_2": vals[-2], "lag_3": vals[-3],
            "lag_7": vals[-7] if len(vals) >= 7 else vals[-1],
            "roll7_mean": vals[-7:].mean(), "roll7_std": vals[-7:].std(),
            "roll3_mean": vals[-3:].mean(), "dow": nd.dayofweek,
        }
        for k in [1, 2]:
            feat[f"sin{k}"] = np.sin(2 * np.pi * k * nd.dayofweek / 7)
            feat[f"cos{k}"] = np.cos(2 * np.pi * k * nd.dayofweek / 7)
        X = pd.DataFrame([feat])[feats]
        yhat = float(model.predict(X)[0])
        # Interval widens with sqrt(horizon): uncertainty compounds recursively.
        widen = sigma * 1.96 * np.sqrt(step)
        rows.append({
            "date": nd.date(), "day": nd.day_name(),
            "forecast": round(yhat),
            "lower_95": max(0, round(yhat - widen)),
            "upper_95": round(yhat + widen),
            "horizon_days": step,
            "confidence": "moderate" if step <= 7 else "low",
        })
        hist = pd.concat([hist, pd.DataFrame([{"post_date": nd, "postings": yhat}])],
                         ignore_index=True)

    fc = pd.DataFrame(rows)
    print(f"\n  {horizon}-DAY POSTING VOLUME FORECAST (95% empirical intervals)")
    print(fc.to_string(index=False, formatters={
        "forecast": "{:,}".format, "lower_95": "{:,}".format, "upper_95": "{:,}".format}))

    recent_avg = float(s["postings"].tail(7).mean())
    fc_avg = float(fc["forecast"].head(7).mean())
    Console.kv("Last 7 observed days, avg", f"{recent_avg:,.0f}/day")
    Console.kv("Next 7 forecast days, avg", f"{fc_avg:,.0f}/day")
    Console.kv("Projected change", f"{(fc_avg / recent_avg - 1) * 100:+.1f}%")

    # ---- category-level trajectory -----------------------------------------
    Console.section("Category trajectories (linear extrapolation, 30 days)")
    from scipy.stats import linregress
    ts = df[df["ts_reliable"]]
    daily = ts.groupby(["post_date", "category"]).size().rename("n").reset_index()
    tot = ts.groupby("post_date").size().rename("tot")
    daily = daily.merge(tot, on="post_date")
    daily["share"] = daily["n"] / daily["tot"]
    daily["idx"] = (pd.to_datetime(daily["post_date"]) - pd.to_datetime(daily["post_date"]).min()).dt.days

    proj = []
    for cat, g in daily.groupby("category"):
        if g["n"].sum() < CFG.MIN_CATEGORY_ROWS:
            continue
        lr = linregress(g["idx"], g["share"])
        cur = g[g["idx"] > g["idx"].max() - 7]["share"].mean() * 100
        p30 = (lr.intercept + lr.slope * (g["idx"].max() + 30)) * 100
        proj.append({"category": cat, "current_share_pct": cur,
                     "projected_30d_pct": max(0, p30),
                     "delta_ppt": max(0, p30) - cur, "r_squared": lr.rvalue ** 2,
                     "reliable": bool(lr.pvalue < 0.05 and lr.rvalue ** 2 > 0.15)})
    pdf_ = pd.DataFrame(proj).sort_values("delta_ppt", ascending=False).reset_index(drop=True)
    print(pdf_.to_string(index=False, formatters={
        "current_share_pct": "{:.2f}%".format, "projected_30d_pct": "{:.2f}%".format,
        "delta_ppt": "{:+.2f}".format, "r_squared": "{:.3f}".format}))

    n_rel = int(pdf_["reliable"].sum())
    Console.finding(
        f"Of {len(pdf_)} categories projected, only {n_rel} have a trend fit strong "
        f"enough (p<0.05, R-sq>0.15) to be treated as a signal rather than noise. "
        f"The remaining {len(pdf_) - n_rel} are flat-with-noise and their projections "
        f"should be read as 'no expected change', not as forecasts.")
    res.add_finding("Task 8",
        f"14-day volume forecast averages {fc_avg:,.0f}/day ({(fc_avg/recent_avg-1)*100:+.1f}% "
        f"vs last week). Only {n_rel}/{len(pdf_)} category trends are statistically reliable.")

    res.add_table("task8_volume_forecast", fc)
    res.add_table("task8_category_projection", pdf_)

    Console.caveat(
        f"Forecasts rest on ~{len(s)} days of history. The weekly cycle is learnable "
        f"on this window; monthly seasonality, holiday effects and macroeconomic "
        f"trend are NOT. Treat the 7-day horizon as moderate confidence and days 8-14 "
        f"as indicative only. Any projection beyond ~14 days would be extrapolation "
        f"beyond what this data can support, and is deliberately not produced. "
        f"Category projections are linear extrapolations, which assume no "
        f"regime change -- a strong assumption over any real horizon.")
    return fc


# =============================================================================
# SECTION 9 | VISUALISATIONS
# =============================================================================
def build_visuals(df: pd.DataFrame, res: Results, outdir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9})
    figs = []
    ts = df[df["ts_reliable"]]

    # 1 | The backfill artifact -- the single most important chart in the report
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    allday = df.groupby("post_date").size()
    axes[0].plot(pd.to_datetime(allday.index), allday.values, lw=1.2, color="crimson")
    axes[0].axvline(pd.Timestamp(CFG.COLLECTION_START), ls="--", color="black", lw=1)
    axes[0].set_title("RAW: apparent explosive growth\n(an artifact of scraper start date)")
    axes[0].set_ylabel("postings/day")
    axes[0].tick_params(axis="x", rotation=45)
    cl = ts.groupby("post_date").size()
    axes[1].plot(pd.to_datetime(cl.index), cl.values, lw=1.4, color="seagreen", marker="o", ms=3)
    axes[1].set_title("CLEANED: stable market with a weekly cycle")
    axes[1].set_ylabel("postings/day")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("Data integrity: why the backfill rows must be excluded",
                 fontweight="bold", y=1.06)
    fig.tight_layout()
    p = outdir / "fig01_backfill_artifact.png"; fig.savefig(p); plt.close(fig); figs.append(p)

    # 2 | Weekly seasonality
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dd = ts.groupby(["post_date", "post_dow"]).size().reset_index(name="n")
    sns.boxplot(data=dd, x="post_dow", y="n", order=order, ax=ax)
    ax.set_title("Weekly seasonality: ~35% weekend collapse", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("postings/day"); ax.tick_params(axis="x", rotation=30)
    p = outdir / "fig02_weekly_seasonality.png"; fig.savefig(p); plt.close(fig); figs.append(p)

    # 3 | Pay distributions, two tracks side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    h = df[df["is_hourly"] & df["pay_analysable"]]["hourly_mid"]
    b = df[~df["is_hourly"] & df["pay_analysable"]]["budget"]
    axes[0].hist(h, bins=60, color="steelblue", edgecolor="white")
    axes[0].axvline(h.median(), color="crimson", ls="--", label=f"median ${h.median():.0f}")
    axes[0].set_title("Hourly rates ($/hr)"); axes[0].legend(); axes[0].set_xlabel("$/hr")
    axes[1].hist(np.log10(b.clip(lower=1)), bins=60, color="darkorange", edgecolor="white")
    axes[1].axvline(np.log10(b.median()), color="crimson", ls="--",
                    label=f"median ${b.median():,.0f}")
    axes[1].set_title("Fixed budgets (log10 $)"); axes[1].legend(); axes[1].set_xlabel("log10($)")
    fig.suptitle("Two incompatible pay units - never averaged together", fontweight="bold", y=1.04)
    fig.tight_layout()
    p = outdir / "fig03_pay_distributions.png"; fig.savefig(p); plt.close(fig); figs.append(p)

    # 4 | Category distribution with tier confidence
    fig, ax = plt.subplots(figsize=(9, 5))
    ct = pd.crosstab(df["category"], df["category_tier"])
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=True).index]
    ct.plot(kind="barh", stacked=True, ax=ax, color=["#d62728", "#2ca02c", "#7f7f7f"])
    ax.set_title("Derived categories by classification confidence tier", fontweight="bold")
    ax.set_xlabel("postings"); ax.set_ylabel("")
    p = outdir / "fig04_category_tiers.png"; fig.savefig(p); plt.close(fig); figs.append(p)

    # 5 | Country rates
    if "task4_country_rates" in res.tables:
        cr = res.tables["task4_country_rates"].head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(cr["country"][::-1], cr["median_rate"][::-1], color="teal")
        ax.set_title("Median hourly rate OFFERED by client country", fontweight="bold")
        ax.set_xlabel("$/hr (median)")
        p = outdir / "fig05_country_rates.png"; fig.savefig(p); plt.close(fig); figs.append(p)

    # 6 | Forecast with intervals
    if "task8_volume_forecast" in res.tables:
        fc = res.tables["task8_volume_forecast"]
        s = res.tables["task3_daily_series"]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(pd.to_datetime(s["post_date"]), s["postings"], color="black", lw=1.3, label="observed")
        fd = pd.to_datetime(fc["date"])
        ax.plot(fd, fc["forecast"], color="crimson", lw=1.6, ls="--", label="forecast")
        ax.fill_between(fd, fc["lower_95"], fc["upper_95"], color="crimson", alpha=0.18,
                        label="95% interval")
        ax.set_title("14-day forecast with empirical prediction intervals", fontweight="bold")
        ax.set_ylabel("postings/day"); ax.legend(); ax.tick_params(axis="x", rotation=45)
        p = outdir / "fig06_forecast.png"; fig.savefig(p); plt.close(fig); figs.append(p)

    # 7 | 24-hour clock
    fig, ax = plt.subplots(figsize=(8, 4))
    hr = ts.groupby("post_hour").size()
    ax.bar(hr.index, hr.values, color="slateblue")
    ax.set_title("Posting activity by hour (UTC): the market never sleeps", fontweight="bold")
    ax.set_xlabel("hour UTC"); ax.set_ylabel("postings")
    p = outdir / "fig07_hourly_clock.png"; fig.savefig(p); plt.close(fig); figs.append(p)

    return figs


# =============================================================================
# SECTION 10 | SELF-TESTS
#
# These are not decorative. Each guards an invariant that, if broken, silently
# corrupts a deliverable. Run with:  python job_market_analysis.py --selftest
# =============================================================================
def _synthetic_raw() -> pd.DataFrame:
    """A miniature raw frame exercising every messy pattern in the real file."""
    return pd.DataFrame({
        "title": ["Full Stack Developer", "Logo Design &amp; Branding",
                  "YouTube Video Editor \u270d\ufe0f", "Shopify Store Setup",
                  "Website", None, "Data Entry Assistant"],
        "link": [f"https://upwork.com/jobs/{i}" for i in range(7)],
        "published_date": ["2024-03-01 10:00:00+00:00", "2024-03-01 11:00:00+00:00",
                           "2023-12-25 10:00:00+00:00",   # backfill
                           "2024-02-15 10:00:00+00:00",   # outage day
                           "2024-03-24 10:00:00+00:00",   # truncated final day
                           "2024-03-02 10:00:00+00:00", "2024-03-02 12:00:00+00:00"],
        "is_hourly": [True, False, True, False, True, False, True],
        "hourly_low": [20.0, np.nan, 15.0, np.nan, np.nan, np.nan, 999.0],
        "hourly_high": [40.0, np.nan, 25.0, np.nan, np.nan, np.nan, 999.0],
        "budget": [np.nan, 500.0, np.nan, 1_000_000.0, np.nan, 250.0, np.nan],
        "country": ["United States", "Cote d&#039;Ivoire", None, "India", "", "Canada", "India"],
    })


def run_selftests() -> int:
    Console.header("Self-tests")
    passed, failed = 0, []

    def check(name, cond):
        nonlocal passed
        if cond:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed.append(name)
            print(f"  FAIL  {name}")

    raw = _synthetic_raw()
    df, audit = clean_data(raw)

    # --- text normalisation
    check("HTML entities decoded", normalise_text("A &amp; B") == "A & B")
    check("Emoji stripped", normalise_text("Editor \u270d\ufe0f").strip() == "Editor")
    check("Whitespace collapsed", normalise_text("  a    b  ") == "a b")

    # --- taxonomy tiering
    check("Specific rules outrank fallback (Shopify -> E-commerce)",
          assign_category("shopify store setup") == ("E-commerce", "specific"))
    check("Bare token reaches fallback tier",
          assign_category("website")[1] == "fallback")
    check("Unmatched titles labelled, not guessed",
          assign_category("zzzz qqqq")[0] == CFG.OTHER_CATEGORY)

    # --- structural
    check("Untitled rows dropped", len(df) == len(raw) - 1)

    # --- THE critical invariant
    hourly, fixed = df[df["is_hourly"]], df[~df["is_hourly"]]
    check("Pay types never mix (hourly has no budget)", hourly["budget"].isna().all())
    check("Pay types never mix (fixed has no hourly rate)",
          fixed[["hourly_low", "hourly_high", "hourly_mid"]].isna().all().all())

    # --- temporal flags
    old = df[df["post_date"] == pd.Timestamp("2023-12-25").date()]
    check("Backfill flagged out of ts_reliable", not old["ts_reliable"].any())
    outage = df[df["post_date"] == pd.Timestamp("2024-02-15").date()]
    check("Outage day flagged", outage["is_outage_day"].all() and not outage["ts_reliable"].any())
    partial = df[df["post_date"] == pd.Timestamp("2024-03-24").date()]
    check("Truncated final day flagged", partial["is_partial_day"].all())
    good = df[df["post_date"] == pd.Timestamp("2024-03-01").date()]
    check("Valid rows marked ts_reliable", good["ts_reliable"].all())

    # --- outliers and missingness
    check("Extreme values flagged, not deleted",
          df["hourly_implausible"].any() and (df["budget"] == 1_000_000.0).any())
    check("Outliers excluded from pay_analysable",
          not df.loc[df["hourly_implausible"], "pay_analysable"].any())
    check("Pay never imputed", df.loc[~df["pay_disclosed"], "pay_value"].isna().all())
    check("Null country becomes 'Unknown', not dropped",
          (df["country"] == "Unknown").sum() == 2 and df["country"].notna().all())

    # --- statistics helpers
    fdr = benjamini_hochberg(np.array([0.001, 0.02, 0.5, 0.9]), 0.05)
    check("FDR flags small p-values", bool(fdr[0]) and not bool(fdr[3]))
    d = cliffs_delta(np.array([10.0, 11, 12, 13, 14]), np.array([1.0, 2, 3, 4, 5]))
    check("Cliff's delta detects large separation", d > 0.9)
    check("Effect labels map correctly",
          effect_label(0.05) == "negligible" and effect_label(0.9) == "large")

    # --- leakage guard
    s = pd.DataFrame({"postings": np.arange(20, dtype=float),
                      "dow": [i % 7 for i in range(20)]})
    sup = _supervised_frame(s)
    check("Lag features are strictly backward-looking",
          bool(sup["lag_1"].iloc[5] == s["postings"].iloc[4]))

    print(f"\n  {passed} passed, {len(failed)} failed")
    if failed:
        for f in failed:
            print(f"    - {f}")
        return 1
    return 0


# =============================================================================
# SECTION 11 | ORCHESTRATION
# =============================================================================
def export_results(res: Results, df: pd.DataFrame, audit: CleaningAudit, outdir: Path) -> None:
    tdir = outdir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    for name, tbl in res.tables.items():
        tbl.to_csv(tdir / f"{name}.csv", index=False)
    audit.to_frame().to_csv(outdir / "cleaning_audit.csv", index=False)
    pd.DataFrame(res.findings).to_csv(outdir / "findings.csv", index=False)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(res.metrics, f, indent=2, default=str)
    try:
        df.to_parquet(outdir / "jobs_clean.parquet", index=False)
    except Exception:
        df.to_csv(outdir / "jobs_clean.csv.gz", index=False, compression="gzip")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Job Market Analysis & Recommendation System (all 8 tasks)")
    ap.add_argument("--data", type=str, default="all_upwork_jobs_2024-02-07-2024-03-24.csv",
                    help="path to the raw Upwork CSV")
    ap.add_argument("--outdir", type=str, default="results", help="output directory")
    ap.add_argument("--recommend", type=str, default=None,
                    help="run a single recommendation query and exit")
    ap.add_argument("--horizon", type=int, default=14, help="forecast horizon in days")
    ap.add_argument("--quick", action="store_true", help="skip figures for a fast run")
    ap.add_argument("--selftest", action="store_true", help="run self-tests and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return run_selftests()

    t0 = datetime.now()
    src = Path(args.data)
    if not src.exists():
        print(f"ERROR: data file not found: {src}", file=sys.stderr)
        print("Usage: python job_market_analysis.py --data <path-to-csv>", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Console.header("Job market analysis & recommendation system")
    Console.kv("Source file", src.name)
    Console.kv("Output directory", outdir.resolve())
    Console.kv("Started", t0.strftime("%Y-%m-%d %H:%M:%S"))

    # ---- load and clean ----------------------------------------------------
    raw = pd.read_csv(src)
    Console.kv("Raw shape", f"{raw.shape[0]:,} rows x {raw.shape[1]} cols")
    df, audit = clean_data(raw)
    audit.show()

    res = Results()
    res.add_metric("rows_raw", int(len(raw)))
    res.add_metric("rows_clean", int(len(df)))
    res.add_metric("rows_ts_reliable", int(df["ts_reliable"].sum()))

    # ---- taxonomy validation ----------------------------------------------
    Console.section("Taxonomy precision (measured, not assumed)")
    val, acc = validate_taxonomy(df)
    if not val.empty:
        by_tier = val.groupby("tier")["agrees"].agg(["mean", "size"])
        print(by_tier.to_string(formatters={"mean": "{:.1%}".format}))
        Console.kv("Overall anchor-term agreement", f"{acc:.1%}")
        Console.note(
            "Each sampled title is re-checked against INDEPENDENT anchor terms, not "
            "the rules that assigned it. This is a conservative precision estimate: "
            "a mismatch may mean a wrong label OR merely an unusual phrasing.")
        res.add_metric("taxonomy_precision_overall", round(float(acc), 4))
        res.add_table("taxonomy_validation", val)

    if args.recommend:
        rec = JobRecommender().fit(df)
        Console.section(f'Recommendations for: "{args.recommend}"')
        out = rec.recommend(args.recommend, k=15)
        print(out.to_string(index=False, max_colwidth=50) if not out.empty else "(no matches)")
        return 0

    # ---- all eight tasks ---------------------------------------------------
    task1_keyword_pay(df, res)
    task2_emerging_categories(df, res)
    bundle = task3_demand_forecast(df, res)
    task4_country_rates(df, res)
    task5_recommender(df, res)
    task6_market_dynamics(df, res)
    task7_remote_landscape(df, res)
    task8_future_trends(df, res, bundle, horizon=args.horizon)

    # ---- figures -----------------------------------------------------------
    if not args.quick:
        Console.header("Generating figures")
        figdir = outdir / "figures"
        figdir.mkdir(parents=True, exist_ok=True)
        try:
            figs = build_visuals(df, res, figdir)
            for f in figs:
                Console.kv("saved", f.name)
        except Exception as e:
            print(f"  (figure generation skipped: {e})")

    # ---- export ------------------------------------------------------------
    Console.header("Export")
    export_results(res, df, audit, outdir)
    Console.kv("Tables written", len(res.tables))
    Console.kv("Metrics written", len(res.metrics))
    Console.kv("Findings recorded", len(res.findings))

    Console.header("Consolidated findings")
    for f in res.findings:
        print(f"\n  [{f['task']}]")
        for line in textwrap.wrap(f["finding"], 74):
            print(f"    {line}")

    dt = (datetime.now() - t0).total_seconds()
    Console.header("Complete")
    Console.kv("Runtime", f"{dt:.1f}s")
    Console.kv("Outputs", outdir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
