#!/usr/bin/env python3
"""
================================================================================
 JOB RECOMMENDATION API  (Task 5 deliverable: "API documentation")
================================================================================

Serves the trained recommendation engine over HTTP, plus read-only access to
the analysis outputs produced by job_market_analysis.py.

Design notes
------------
* The TF-IDF index is built ONCE at startup (lifespan handler), not per
  request. Rebuilding a 60k-document index on every call would make the
  endpoint unusable.
* The API imports the recommender class from job_market_analysis.py rather
  than reimplementing it. One definition, one behaviour -- the thing serving
  traffic is the same thing that was evaluated at precision@10 = 0.636.
* Every response that returns recommendations also returns the evaluation
  metrics, so a consumer can see how much to trust the ranking.

Run locally:  uvicorn api:app --reload --port 8000
Docs:         http://localhost:8000/docs
================================================================================
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from job_market_analysis import CFG, JobRecommender, clean_data

DATA_DIR = Path(os.getenv("DATA_DIR", "results"))
RAW_CSV = Path(os.getenv("RAW_CSV", "jobs.csv"))
CLEAN_PARQUET = DATA_DIR / "jobs_clean.parquet"

STATE: dict = {}


def _load_dataframe() -> pd.DataFrame:
    """Prefer the pre-cleaned parquet; fall back to cleaning the raw CSV.

    This lets the API start in ~2s in normal operation while still working
    from a bare CSV if the analysis has not been run yet.
    """
    if CLEAN_PARQUET.exists():
        return pd.read_parquet(CLEAN_PARQUET)
    if RAW_CSV.exists():
        df, _ = clean_data(pd.read_csv(RAW_CSV))
        return df
    raise FileNotFoundError(
        f"No data found. Expected {CLEAN_PARQUET} or {RAW_CSV}. "
        f"Run: python job_market_analysis.py --data <csv>")


@asynccontextmanager
async def lifespan(app: FastAPI):
    df = _load_dataframe()
    STATE["df"] = df
    STATE["recommender"] = JobRecommender().fit(df)
    STATE["eval"] = STATE["recommender"].evaluate(k=10, n_queries=200)
    metrics_path = DATA_DIR / "metrics.json"
    STATE["metrics"] = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    yield
    STATE.clear()


app = FastAPI(
    title="Job Market Recommendation API",
    description=(
        "Content-based job recommendation engine over 244,827 cleaned Upwork "
        "postings, plus read-only access to the analysis outputs. "
        "Measured performance: precision@10 = 0.636 against a 0.080 random "
        "baseline (7.9x lift), MRR = 0.761."),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- schemas
class JobOut(BaseModel):
    score: float = Field(..., description="Hybrid rank score (content + pay disclosure + recency)")
    similarity: float = Field(..., description="Raw TF-IDF cosine similarity")
    title: str
    category: str
    pay_type: str
    hourly_rate: float | None = None
    budget: float | None = None
    country: str
    posted: str
    link: str


class RecommendResponse(BaseModel):
    query: str
    n_results: int
    filters: dict
    evaluation: dict = Field(..., description="Measured engine performance, for trust calibration")
    results: list[JobOut]


# -------------------------------------------------------------- endpoints
@app.get("/", tags=["meta"])
def root():
    return {
        "service": "Job Market Recommendation API",
        "docs": "/docs",
        "endpoints": ["/health", "/recommend", "/categories", "/countries",
                      "/stats", "/metrics", "/trends"],
    }


@app.get("/health", tags=["meta"])
def health():
    df = STATE.get("df")
    return {
        "status": "healthy" if df is not None else "loading",
        "jobs_loaded": int(len(df)) if df is not None else 0,
        "index_size": int(len(STATE["recommender"].jobs)) if "recommender" in STATE else 0,
    }


@app.get("/recommend", response_model=RecommendResponse, tags=["recommendations"])
def recommend(
    q: str = Query(..., min_length=2, description="Skills or role, e.g. 'python data analyst'"),
    k: int = Query(10, ge=1, le=50, description="Number of results"),
    category: str | None = Query(None, description="Restrict to one derived category"),
    min_rate: float | None = Query(None, ge=0, description="Minimum hourly rate ($/hr)"),
    hybrid: bool = Query(True, description="Blend pay disclosure and recency into ranking"),
):
    """Recommend jobs matching a free-text skills query.

    Ranking is TF-IDF cosine similarity (weight 0.80) blended with pay
    disclosure (0.12) and recency (0.08). Set hybrid=false for pure content
    similarity.
    """
    rec = STATE.get("recommender")
    if rec is None:
        raise HTTPException(503, "Recommender not ready")

    out = rec.recommend(q, k=k, category=category, min_rate=min_rate, hybrid=hybrid)
    if out.empty:
        return RecommendResponse(query=q, n_results=0,
                                 filters={"category": category, "min_rate": min_rate},
                                 evaluation=STATE["eval"], results=[])

    results = [
        JobOut(
            score=round(float(r["score"]), 4),
            similarity=round(float(r["similarity"]), 4),
            title=str(r["title_clean"]),
            category=str(r["category"]),
            pay_type=str(r["pay_type"]),
            hourly_rate=None if pd.isna(r["hourly_mid"]) else float(r["hourly_mid"]),
            budget=None if pd.isna(r["budget"]) else float(r["budget"]),
            country=str(r["country"]),
            posted=str(r["post_date"]),
            link=str(r["link"]),
        )
        for _, r in out.iterrows()
    ]
    return RecommendResponse(
        query=q, n_results=len(results),
        filters={"category": category, "min_rate": min_rate, "hybrid": hybrid},
        evaluation=STATE["eval"], results=results)


@app.get("/categories", tags=["reference"])
def categories():
    """Derived categories with posting counts and classification confidence.

    `pct_specific` is the share assigned by tier-1 phrase rules rather than
    tier-2 generic-token fallback -- a per-category confidence indicator.
    """
    df = STATE["df"]
    g = df.groupby("category")
    out = pd.DataFrame({
        "postings": g.size(),
        "pct_specific_tier": (g["category_tier"].apply(lambda s: (s == "specific").mean()) * 100).round(1),
        "median_hourly": g.apply(lambda x: x.loc[x["is_hourly"] & x["pay_analysable"], "hourly_mid"].median()),
        "median_budget": g.apply(lambda x: x.loc[~x["is_hourly"] & x["pay_analysable"], "budget"].median()),
    }).sort_values("postings", ascending=False)
    return json.loads(out.reset_index().to_json(orient="records"))


@app.get("/countries", tags=["reference"])
def countries(min_postings: int = Query(CFG.MIN_COUNTRY_POSTINGS, ge=1)):
    """Median hourly rate OFFERED by client country.

    NOTE: `country` is the CLIENT's location, not the freelancer's. This is
    what clients in each country pay, not what workers there earn. Rates are
    not PPP-adjusted.
    """
    df = STATE["df"]
    h = df[df["is_hourly"] & df["pay_analysable"] & (df["country"] != "Unknown")]
    g = h.groupby("country")["hourly_mid"]
    out = pd.DataFrame({"postings": g.size(), "median_rate": g.median().round(2),
                        "p25": g.quantile(.25).round(2), "p75": g.quantile(.75).round(2)})
    out = out[out["postings"] >= min_postings].sort_values("median_rate", ascending=False)
    return json.loads(out.reset_index().to_json(orient="records"))


@app.get("/stats", tags=["reference"])
def stats():
    """Headline dataset statistics, including data-quality flags."""
    df = STATE["df"]
    h = df[df["is_hourly"] & df["pay_analysable"]]
    b = df[~df["is_hourly"] & df["pay_analysable"]]
    return {
        "total_postings": int(len(df)),
        "time_series_reliable": int(df["ts_reliable"].sum()),
        "excluded_from_timeseries": int((~df["ts_reliable"]).sum()),
        "date_range": {"start": str(df["post_date"].min()), "end": str(df["post_date"].max())},
        "pay": {
            "hourly_postings": int(df["is_hourly"].sum()),
            "fixed_postings": int((~df["is_hourly"]).sum()),
            "pct_pay_disclosed": round(float(df["pay_disclosed"].mean() * 100), 1),
            "median_hourly_rate": round(float(h["hourly_mid"].median()), 2),
            "median_fixed_budget": round(float(b["budget"].median()), 2),
        },
        "coverage": {"countries": int(df["country"].nunique()),
                     "categories": int(df["category"].nunique())},
        "data_quality_note": (
            "Hourly and fixed pay are never averaged together -- they are "
            "incompatible units. 15.7% of postings disclose no pay and are "
            "not imputed."),
    }


@app.get("/metrics", tags=["reference"])
def metrics():
    """Metrics emitted by the analysis pipeline, plus live engine evaluation."""
    return {"analysis": STATE.get("metrics", {}), "recommender": STATE.get("eval", {})}


@app.get("/trends", tags=["reference"])
def trends(grain: str = Query("weekly", pattern="^(daily|weekly)$")):
    """Posting volume and market composition over time.

    Only `ts_reliable` rows are counted: backfill, the collection outage and
    the truncated final day are excluded, because including them produces a
    false growth curve.
    """
    df = STATE["df"]
    ts = df[df["ts_reliable"]]
    key = "post_date" if grain == "daily" else "post_week"
    g = ts.groupby(key)
    out = pd.DataFrame({
        "postings": g.size(),
        "pct_hourly": (g["is_hourly"].mean() * 100).round(1),
        "pct_pay_disclosed": (g["pay_disclosed"].mean() * 100).round(1),
        "median_hourly": g.apply(lambda x: x.loc[x["is_hourly"] & x["pay_analysable"], "hourly_mid"].median()),
    }).reset_index()
    out[key] = out[key].astype(str)
    return json.loads(out.to_json(orient="records"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
