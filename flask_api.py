#!/usr/bin/env python3
"""
================================================================================
 FLASK API  (brief-compliant implementation)
================================================================================

The project brief specifies "Flask or Django for API development". This module
provides that Flask implementation.

WHY BOTH FLASK AND FASTAPI EXIST IN THIS PROJECT
`api.py` implements the same service in FastAPI, which was the initial choice
for three practical reasons: automatic OpenAPI/Swagger documentation (the brief
asks for "API documentation" as a Task 5 deliverable), Pydantic request and
response validation, and ASGI async support.

Rather than substitute a preferred tool for a specified one, both are provided:
this file satisfies the brief literally, `api.py` demonstrates the production
alternative, and both import the SAME `JobRecommender` from
job_market_analysis.py. There is one model, evaluated once at
precision@10 = 0.636 -- the framework is a transport layer, not a
reimplementation.

Run:  python flask_api.py
      FLASK_PORT=5000 python flask_api.py
Docs: http://localhost:5000/  (hand-written API index, since Flask has no
      automatic schema generation -- itself an illustration of the trade-off)
================================================================================
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

from job_market_analysis import CFG, JobRecommender, clean_data

DATA_DIR = Path(os.getenv("DATA_DIR", "results"))
RAW_CSV = Path(os.getenv("RAW_CSV", "jobs.csv"))
CLEAN_PARQUET = DATA_DIR / "jobs_clean.parquet"

app = Flask(__name__)
app.json.sort_keys = False

STATE: dict = {}


def _load() -> pd.DataFrame:
    if CLEAN_PARQUET.exists():
        return pd.read_parquet(CLEAN_PARQUET)
    if RAW_CSV.exists():
        df, _ = clean_data(pd.read_csv(RAW_CSV))
        return df
    raise FileNotFoundError(
        f"No data found. Expected {CLEAN_PARQUET} or {RAW_CSV}. "
        f"Run: python job_market_analysis.py --data jobs.csv")


def init_state() -> None:
    """Build the index once at startup.

    Flask has no lifespan hook equivalent to FastAPI's, so this is called
    explicitly before serving. Building a 60k-document TF-IDF index per
    request would make the endpoint unusable.
    """
    if "recommender" in STATE:
        return
    print("Loading data and building recommendation index...")
    df = _load()
    STATE["df"] = df
    STATE["recommender"] = JobRecommender().fit(df)
    STATE["eval"] = STATE["recommender"].evaluate(k=10, n_queries=200)
    print(f"Ready: {len(df):,} postings, "
          f"precision@10 = {STATE['eval']['precision@10']:.3f}")


# ------------------------------------------------------------------ routes
@app.get("/")
def index():
    """Hand-written API index.

    FastAPI generates this automatically from type hints; Flask does not.
    Keeping the contrast visible is deliberate -- it is the concrete reason
    the FastAPI variant also exists.
    """
    return jsonify({
        "service": "Job Market Recommendation API (Flask)",
        "version": "1.0.0",
        "dataset": "244,827 cleaned Upwork postings, Feb-Mar 2024",
        "endpoints": {
            "GET /health": "service status",
            "GET /recommend?q=<skills>&k=<n>&category=<cat>&min_rate=<n>":
                "job recommendations",
            "GET /categories": "derived categories with confidence tiers",
            "GET /countries?min_postings=<n>": "median hourly rate by client country",
            "GET /stats": "dataset statistics",
            "GET /trends?grain=<daily|weekly>": "market dynamics over time",
        },
        "note": ("A FastAPI implementation of the same service is available in "
                 "api.py with auto-generated Swagger docs at /docs."),
    })


@app.get("/health")
def health():
    df = STATE.get("df")
    return jsonify({
        "status": "healthy" if df is not None else "loading",
        "jobs_loaded": int(len(df)) if df is not None else 0,
        "index_size": int(len(STATE["recommender"].jobs)) if "recommender" in STATE else 0,
    })


@app.get("/recommend")
def recommend():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify({"error": "Query parameter 'q' is required (min 2 characters)"}), 400

    try:
        k = min(max(int(request.args.get("k", 10)), 1), 50)
    except ValueError:
        return jsonify({"error": "'k' must be an integer"}), 400

    category = request.args.get("category")
    min_rate = request.args.get("min_rate", type=float)

    rec = STATE.get("recommender")
    if rec is None:
        return jsonify({"error": "Recommender not ready"}), 503

    out = rec.recommend(q, k=k, category=category, min_rate=min_rate)
    results = [] if out.empty else [
        {
            "score": round(float(r["score"]), 4),
            "similarity": round(float(r["similarity"]), 4),
            "title": str(r["title_clean"]),
            "category": str(r["category"]),
            "pay_type": str(r["pay_type"]),
            "hourly_rate": None if pd.isna(r["hourly_mid"]) else float(r["hourly_mid"]),
            "budget": None if pd.isna(r["budget"]) else float(r["budget"]),
            "country": str(r["country"]),
            "posted": str(r["post_date"]),
            "link": str(r["link"]),
        }
        for _, r in out.iterrows()
    ]
    # The evaluation travels with every response so a consumer can calibrate
    # how much to trust the ranking.
    return jsonify({
        "query": q,
        "n_results": len(results),
        "filters": {"category": category, "min_rate": min_rate},
        "evaluation": STATE["eval"],
        "results": results,
    })


@app.get("/categories")
def categories():
    df = STATE["df"]
    g = df.groupby("category")
    out = pd.DataFrame({
        "postings": g.size(),
        "pct_specific_tier": (g["category_tier"].apply(
            lambda s: (s == "specific").mean()) * 100).round(1),
        "median_hourly": g.apply(
            lambda x: x.loc[x["is_hourly"] & x["pay_analysable"], "hourly_mid"].median()),
    }).sort_values("postings", ascending=False).reset_index()
    return app.response_class(out.to_json(orient="records"), mimetype="application/json")


@app.get("/countries")
def countries():
    min_postings = request.args.get("min_postings", CFG.MIN_COUNTRY_POSTINGS, type=int)
    df = STATE["df"]
    h = df[df["is_hourly"] & df["pay_analysable"] & (df["country"] != "Unknown")]
    g = h.groupby("country")["hourly_mid"]
    out = pd.DataFrame({"postings": g.size(), "median_rate": g.median().round(2),
                        "p25": g.quantile(.25).round(2), "p75": g.quantile(.75).round(2)})
    out = out[out["postings"] >= min_postings].sort_values(
        "median_rate", ascending=False).reset_index()
    return app.response_class(out.to_json(orient="records"), mimetype="application/json")


@app.get("/stats")
def stats():
    df = STATE["df"]
    h = df[df["is_hourly"] & df["pay_analysable"]]
    b = df[~df["is_hourly"] & df["pay_analysable"]]
    return jsonify({
        "total_postings": int(len(df)),
        "time_series_reliable": int(df["ts_reliable"].sum()),
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
        "data_quality_note": ("Hourly and fixed pay are never averaged -- incompatible "
                              "units. 15.7% of postings disclose no pay and are not imputed."),
    })


@app.get("/trends")
def trends():
    grain = request.args.get("grain", "weekly")
    if grain not in {"daily", "weekly"}:
        return jsonify({"error": "grain must be 'daily' or 'weekly'"}), 400
    ts = STATE["df"]
    ts = ts[ts["ts_reliable"]]
    key = "post_date" if grain == "daily" else "post_week"
    g = ts.groupby(key)
    out = pd.DataFrame({
        "postings": g.size(),
        "pct_hourly": (g["is_hourly"].mean() * 100).round(1),
        "pct_pay_disclosed": (g["pay_disclosed"].mean() * 100).round(1),
    }).reset_index()
    out[key] = out[key].astype(str)
    return app.response_class(out.to_json(orient="records"), mimetype="application/json")


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found. See GET / for the API index."}), 404


if __name__ == "__main__":
    init_state()
    app.run(host="0.0.0.0", port=int(os.getenv("FLASK_PORT", 5000)), debug=False)
