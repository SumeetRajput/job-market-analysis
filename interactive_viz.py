#!/usr/bin/env python3
"""
================================================================================
 INTERACTIVE VISUALISATIONS  (Plotly)
================================================================================

Task 4 of the brief asks specifically for "an interactive map or chart showing
hourly rates by country". Static matplotlib figures do not satisfy that
requirement, so this module produces self-contained interactive HTML.

Outputs open in any browser with no server, no Python and no internet
(Plotly.js is inlined), which makes them straightforward to submit alongside a
report or embed in a portfolio.

Usage:
    python interactive_viz.py                 # build all
    python interactive_viz.py --open          # build and print paths
================================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Plotly not installed. Run: pip install plotly", file=sys.stderr)
    sys.exit(1)

DATA_DIR = Path(os.getenv("DATA_DIR", "results"))
PARQUET = DATA_DIR / "jobs_clean.parquet"
OUTDIR = DATA_DIR / "interactive"

# Consistent palette across every chart.
THEME = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=12),
    title_font_size=18,
    margin=dict(l=60, r=40, t=80, b=60),
)

# Plotly's choropleth needs ISO-3 codes. Only countries meeting the volume
# threshold need mapping, so this covers the analysed set rather than all 212.
ISO3 = {
    "United States": "USA", "United Kingdom": "GBR", "India": "IND",
    "Australia": "AUS", "Canada": "CAN", "Pakistan": "PAK", "Germany": "DEU",
    "Netherlands": "NLD", "United Arab Emirates": "ARE", "France": "FRA",
    "Singapore": "SGP", "Spain": "ESP", "Italy": "ITA", "Switzerland": "CHE",
    "Sweden": "SWE", "Ireland": "IRL", "New Zealand": "NZL", "Israel": "ISR",
    "Philippines": "PHL", "Bangladesh": "BGD", "Nigeria": "NGA", "Poland": "POL",
    "Brazil": "BRA", "Ukraine": "UKR", "Denmark": "DNK", "Belgium": "BEL",
    "Norway": "NOR", "Austria": "AUT", "Turkey": "TUR", "Saudi Arabia": "SAU",
    "Kenya": "KEN", "Vietnam": "VNM", "China": "CHN", "Japan": "JPN",
    "South Korea": "KOR", "Egypt": "EGY", "Thailand": "THA", "Colombia": "COL",
    "Mexico": "MEX", "Indonesia": "IDN", "Malaysia": "MYS", "Portugal": "PRT",
    "Greece": "GRC", "Romania": "ROU", "Czechia": "CZE", "Hungary": "HUN",
    "Finland": "FIN", "South Africa": "ZAF", "Argentina": "ARG", "Chile": "CHL",
    "Hong Kong": "HKG", "Qatar": "QAT", "Kuwait": "KWT", "Morocco": "MAR",
    "Serbia": "SRB", "Croatia": "HRV", "Bulgaria": "BGR", "Estonia": "EST",
    "Lithuania": "LTU", "Latvia": "LVA", "Slovenia": "SVN", "Slovakia": "SVK",
    "Cyprus": "CYP", "Malta": "MLT", "Luxembourg": "LUX", "Iceland": "ISL",
}


def load() -> pd.DataFrame:
    if not PARQUET.exists():
        print(f"ERROR: {PARQUET} not found.\n"
              f"Run first: python job_market_analysis.py --data jobs.csv", file=sys.stderr)
        sys.exit(2)
    return pd.read_parquet(PARQUET)


# Set by --cdn. Inline mode embeds Plotly.js in every file (~4.7 MB each,
# fully offline). CDN mode loads it from a script tag (~10 KB each, needs
# internet). Inline is better for submission on a USB stick; CDN is essential
# for GitHub, where 8 inline files would add 38 MB to the repository.
PLOTLYJS_MODE = "inline"


def save(fig, name: str) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{name}.html"
    fig.write_html(path, include_plotlyjs=PLOTLYJS_MODE, full_html=True)
    return path


# =============================================================================
# TASK 4 | Interactive world map of hourly rates
# =============================================================================
def build_rate_map(df: pd.DataFrame, min_postings: int = 200) -> Path:
    h = df[df.is_hourly & df.pay_analysable & (df.country != "Unknown")]
    g = h.groupby("country")["hourly_mid"]
    stats = pd.DataFrame({
        "postings": g.size(),
        "median_rate": g.median().round(2),
        "p25": g.quantile(.25).round(2),
        "p75": g.quantile(.75).round(2),
    }).reset_index()
    stats = stats[stats.postings >= min_postings]
    stats["iso3"] = stats["country"].map(ISO3)

    unmapped = stats[stats.iso3.isna()]["country"].tolist()
    stats = stats.dropna(subset=["iso3"])

    fig = px.choropleth(
        stats, locations="iso3", color="median_rate", hover_name="country",
        hover_data={"iso3": False, "median_rate": ":$.2f", "postings": ":,",
                    "p25": ":$.2f", "p75": ":$.2f"},
        color_continuous_scale="Viridis",
        labels={"median_rate": "Median $/hr"},
    )
    fig.update_layout(
        title=dict(text=(
            "<b>Median Hourly Rate Offered, by Client Country</b><br>"
            "<sup>Hover for posting volume and interquartile range. "
            "`country` is the CLIENT's location, not the freelancer's — this is what "
            "clients PAY, not what workers EARN. Rates are not PPP-adjusted.</sup>"),
            x=0.5, xanchor="center"),
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
        height=620, **THEME)
    if unmapped:
        print(f"  note: {len(unmapped)} countries lack an ISO-3 mapping and are "
              f"absent from the map (present in the bar chart): {unmapped[:5]}")
    return save(fig, "task4_rate_map")


def build_rate_bars(df: pd.DataFrame, min_postings: int = 200) -> Path:
    """Companion to the map: a sortable bar chart that also shows volume.

    A choropleth encodes rate by colour but hides how much evidence sits behind
    each country. Pairing it with volume prevents reading a 254-posting median
    as equal in weight to a 48,000-posting one.
    """
    h = df[df.is_hourly & df.pay_analysable & (df.country != "Unknown")]
    g = h.groupby("country")["hourly_mid"]
    stats = pd.DataFrame({"postings": g.size(), "median_rate": g.median().round(2),
                          "p25": g.quantile(.25).round(2),
                          "p75": g.quantile(.75).round(2)}).reset_index()
    stats = stats[stats.postings >= min_postings].sort_values("median_rate", ascending=True)

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
        subplot_titles=("Median hourly rate ($/hr)", "Posting volume (log scale)"))

    fig.add_trace(go.Bar(
        x=stats.median_rate, y=stats.country, orientation="h",
        marker=dict(color=stats.median_rate, colorscale="Viridis", showscale=False),
        customdata=np.stack([stats.p25, stats.p75, stats.postings], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Median: $%{x:.2f}/hr<br>"
                       "IQR: $%{customdata[0]:.2f} - $%{customdata[1]:.2f}<br>"
                       "Postings: %{customdata[2]:,}<extra></extra>"),
        name="median rate"), row=1, col=1)

    fig.add_trace(go.Bar(
        x=stats.postings, y=stats.country, orientation="h",
        marker=dict(color="#888"),
        hovertemplate="<b>%{y}</b><br>Postings: %{x:,}<extra></extra>",
        name="postings"), row=1, col=2)

    fig.update_xaxes(type="log", row=1, col=2)
    fig.update_layout(
        title=dict(text=(
            "<b>Hourly Rates vs Evidence Base, by Client Country</b><br>"
            "<sup>Rate alone is misleading without volume: a country with 254 postings "
            "is far less certain than one with 48,000.</sup>"), x=0.5, xanchor="center"),
        height=max(600, 22 * len(stats)), showlegend=False, **THEME)
    return save(fig, "task4_rate_bars")


# =============================================================================
# TASK 1 | Keyword pay premiums
# =============================================================================
def build_keyword_chart(track: str = "hourly") -> Path | None:
    path = DATA_DIR / "tables" / f"task1_keywords_{track}.csv"
    if not path.exists():
        return None
    kw = pd.read_csv(path)
    sig = kw[kw.significant_fdr & (kw.effect != "negligible")].copy()
    top = pd.concat([sig.nlargest(20, "median_pay"), sig.nsmallest(20, "median_pay")])
    top = top.sort_values("median_pay")
    unit = "$/hr" if track == "hourly" else "$/project"

    fig = go.Figure(go.Bar(
        x=top.median_pay, y=top.keyword, orientation="h",
        marker=dict(color=top.cliffs_delta, colorscale="RdYlGn", cmid=0,
                    colorbar=dict(title="Cliff's<br>delta")),
        customdata=np.stack([top.n_postings, top.vs_overall_pct, top.effect], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Median: %{x:,.0f} " + unit +
                       "<br>Postings: %{customdata[0]:,}"
                       "<br>vs overall: %{customdata[1]:+.0f}%"
                       "<br>Effect size: %{customdata[2]}<extra></extra>")))
    fig.update_layout(
        title=dict(text=(
            f"<b>Keyword Pay Premiums and Discounts ({track})</b><br>"
            f"<sup>Only keywords surviving Benjamini-Hochberg FDR correction AND showing a "
            f"non-negligible Cliff's delta. Correlation, not causation.</sup>"),
            x=0.5, xanchor="center"),
        xaxis_title=f"Median pay ({unit})", yaxis_title="", height=900, **THEME)
    if track == "fixed":
        fig.update_xaxes(type="log")
    return save(fig, f"task1_keywords_{track}")


# =============================================================================
# TASK 3/6 | Time series with seasonality
# =============================================================================
def build_timeseries(df: pd.DataFrame) -> Path:
    all_days = df.groupby("post_date").size().reset_index(name="postings")
    clean = df[df.ts_reliable].groupby("post_date").size().reset_index(name="postings")
    all_days["post_date"] = pd.to_datetime(all_days["post_date"])
    clean["post_date"] = pd.to_datetime(clean["post_date"])

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.16,
        subplot_titles=(
            "RAW — apparent explosive growth (an artifact of the scraper start date)",
            "CLEANED — stable market with a weekly cycle"))

    fig.add_trace(go.Scatter(
        x=all_days.post_date, y=all_days.postings, mode="lines",
        line=dict(color="crimson", width=1.5), name="raw",
        hovertemplate="%{x|%d %b %Y}<br>%{y:,} postings<extra></extra>"), row=1, col=1)
    fig.add_vline(x=pd.Timestamp("2024-02-13"), line_dash="dash",
                  line_color="black", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=clean.post_date, y=clean.postings, mode="lines+markers",
        line=dict(color="seagreen", width=2), marker=dict(size=5), name="cleaned",
        hovertemplate="%{x|%a %d %b}<br>%{y:,} postings<extra></extra>"), row=2, col=1)

    fig.update_layout(
        title=dict(text=(
            "<b>Data Integrity: Why 283 Rows Had to Be Excluded</b><br>"
            "<sup>283 rows across the 48 days before 13 Feb (~6/day) vs ~5,960/day after. "
            "Those early rows are scraper backfill, not a quiet market.</sup>"),
            x=0.5, xanchor="center"),
        height=760, showlegend=False, **THEME)
    return save(fig, "task3_timeseries")


def build_seasonality(df: pd.DataFrame) -> Path:
    ts = df[df.ts_reliable]
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = ts.groupby(["post_date", "post_dow"]).size().reset_index(name="n")

    fig = go.Figure()
    for d in order:
        sub = daily[daily.post_dow == d]
        fig.add_trace(go.Box(y=sub.n, name=d[:3],
                             marker_color="#d62728" if d in ("Saturday", "Sunday") else "#1f77b4",
                             boxmean=True))
    wk = daily[~daily.post_dow.isin(["Saturday", "Sunday"])].n.mean()
    we = daily[daily.post_dow.isin(["Saturday", "Sunday"])].n.mean()
    fig.update_layout(
        title=dict(text=(
            f"<b>Weekly Seasonality: {(1 - we / wk) * 100:.0f}% Weekend Collapse</b><br>"
            f"<sup>Weekdays average {wk:,.0f} postings/day vs {we:,.0f} at weekends. "
            f"This is the dominant signal in any short-horizon forecast.</sup>"),
            x=0.5, xanchor="center"),
        yaxis_title="Postings per day", height=520, showlegend=False, **THEME)
    return save(fig, "task3_seasonality")


# =============================================================================
# TASK 2 | Category treemap
# =============================================================================
def build_category_treemap(df: pd.DataFrame) -> Path:
    g = df.groupby(["category", "category_tier"]).size().reset_index(name="postings")
    fig = px.treemap(
        g, path=[px.Constant("All postings"), "category", "category_tier"],
        values="postings", color="postings", color_continuous_scale="Blues")
    fig.update_traces(hovertemplate="<b>%{label}</b><br>%{value:,} postings<extra></extra>")
    fig.update_layout(
        title=dict(text=(
            "<b>Derived Category Distribution and Classification Confidence</b><br>"
            "<sup>The dataset has NO category column — these are derived from title text. "
            "'specific' = tier-1 phrase match, 'fallback' = tier-2 generic token. "
            "Measured precision: 89.6%.</sup>"), x=0.5, xanchor="center"),
        height=680, **THEME)
    return save(fig, "task2_category_treemap")


# =============================================================================
# TASK 8 | Forecast
# =============================================================================
def build_forecast(df: pd.DataFrame) -> Path | None:
    fp = DATA_DIR / "tables" / "task8_volume_forecast.csv"
    sp = DATA_DIR / "tables" / "task3_daily_series.csv"
    if not (fp.exists() and sp.exists()):
        return None
    fc = pd.read_csv(fp)
    s = pd.read_csv(sp)
    fc["date"] = pd.to_datetime(fc["date"])
    s["post_date"] = pd.to_datetime(s["post_date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc.date, y=fc.upper_95, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=fc.date, y=fc.lower_95, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(220,20,60,0.18)",
        name="95% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=s.post_date, y=s.postings, mode="lines+markers",
        line=dict(color="black", width=2), marker=dict(size=4), name="observed",
        hovertemplate="%{x|%a %d %b}<br>%{y:,} postings<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=fc.date, y=fc.forecast, mode="lines+markers",
        line=dict(color="crimson", width=2, dash="dash"), marker=dict(size=5),
        name="forecast",
        customdata=np.stack([fc.confidence, fc.horizon_days], axis=-1),
        hovertemplate=("%{x|%a %d %b}<br>%{y:,} postings"
                       "<br>Confidence: %{customdata[0]}"
                       "<br>Horizon: day %{customdata[1]}<extra></extra>")))
    fig.update_layout(
        title=dict(text=(
            "<b>14-Day Forecast with Empirical Prediction Intervals</b><br>"
            "<sup>Intervals widen with the square root of horizon because uncertainty "
            "compounds in a recursive forecast. Days 1-7 moderate confidence; 8-14 "
            "indicative only.</sup>"), x=0.5, xanchor="center"),
        yaxis_title="Postings per day", height=560, hovermode="x unified", **THEME)
    return save(fig, "task8_forecast")


# =============================================================================
# INDEX
# =============================================================================
def build_index(entries: list[dict]) -> Path:
    """Build the landing page.

    Entries carry an explicit `seq` so the list follows the order of the
    project brief exactly, rather than sorting alphabetically within a task.
    Tasks whose deliverable is not a chart (5, 6, 7) are still listed, marked
    as external, so the index reads as a complete 1-to-8 map of the project
    rather than a list with unexplained gaps.
    """
    entries = sorted(entries, key=lambda e: e["seq"])
    items = []
    for e in entries:
        if e.get("external"):
            items.append(f'''      <li>
        <div class="row ext">
          <span class="tag ext-tag">Task {e['task']}</span>
          <span class="name">{e['title']}</span>
          <span class="desc">{e['desc']}</span>
        </div>
      </li>''')
        else:
            items.append(f'''      <li>
        <a class="row" href="{e['path'].name}" target="_blank">
          <span class="tag">Task {e['task']}</span>
          <span class="name">{e['title']}</span>
          <span class="desc">{e['desc']}</span>
        </a>
      </li>''')

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Job Market Analysis &mdash; Interactive Visualisations</title>
<style>
  body {{ font-family: "Segoe UI", Helvetica, Arial, sans-serif; max-width: 860px;
         margin: 56px auto; padding: 0 24px; color: #1c2733; line-height: 1.55; }}
  h1 {{ margin: 0 0 4px; font-size: 30px; color: #1a3a5c; }}
  .sub {{ color: #667; margin-bottom: 28px; font-size: 14.5px; }}
  ul {{ list-style: none; padding: 0; margin: 0; }}
  li {{ margin: 8px 0; }}
  .row {{ display: grid; grid-template-columns: 78px 1fr; gap: 3px 15px;
          padding: 13px 18px; background: #f6f8fa; border-radius: 8px;
          text-decoration: none; color: inherit; border: 1px solid #e2e8ee; }}
  a.row {{ transition: background .12s, border-color .12s; }}
  a.row:hover {{ background: #eaf1f8; border-color: #b9cde0; }}
  .ext {{ background: #fafbfc; border-style: dashed; opacity: .85; }}
  .tag {{ grid-row: 1 / span 2; align-self: center; font-size: 11.5px;
          font-weight: 700; color: #fff; background: #1a3a5c; text-align: center;
          padding: 5px 0; border-radius: 5px; letter-spacing: .3px; }}
  .ext-tag {{ background: #8b9aa8; }}
  .name {{ font-weight: 600; font-size: 15.5px; color: #1a3a5c; }}
  .ext .name {{ color: #5a6470; }}
  .desc {{ font-size: 13px; color: #5a6470; }}
  .legend {{ margin: 18px 0 0; font-size: 12.5px; color: #7a848f; }}
  .note {{ margin-top: 26px; padding: 15px 18px; background: #fff8e6;
           border-left: 4px solid #f0ad4e; font-size: 13.5px; border-radius: 0 6px 6px 0; }}
</style></head><body>
  <h1>Job Market Analysis</h1>
  <p class="sub">Interactive visualisations &middot; 244,827 Upwork postings &middot; February&ndash;March 2024</p>
  <ul>
{chr(10).join(items)}
  </ul>
  <p class="legend">Solid rows open an interactive chart. Dashed rows are project
     deliverables whose output is not a chart &mdash; they are listed so this index
     maps the full set of eight tasks.</p>
  <div class="note">
    <strong>Reading these charts:</strong> country refers to the <em>client's</em>
    location, not the freelancer's. Hourly rates and fixed budgets are never
    averaged together &mdash; they are incompatible units. All time-series views
    exclude scraper backfill, the 15 February collection outage and the truncated
    final day.
  </div>
</body></html>"""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / "index.html"
    p.write_text(html, encoding="utf-8")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build interactive Plotly visualisations")
    ap.add_argument("--open", action="store_true", help="print output paths when done")
    ap.add_argument("--cdn", action="store_true",
                    help="load Plotly.js from CDN (~10 KB/file instead of ~4.7 MB; "
                         "required for GitHub, needs internet to view)")
    args = ap.parse_args(argv)

    global PLOTLYJS_MODE
    PLOTLYJS_MODE = "cdn" if args.cdn else "inline"

    df = load()
    print("Building interactive visualisations...")

    # seq controls display order explicitly. Chart entries carry a builder;
    # entries for Tasks 5-7 have no chart because their deliverable is the
    # API, the dashboard and the written report respectively.
    SPEC = [
        dict(seq=1, task="1", title="Keyword Pay Premiums \u2014 Hourly",
             desc="Which title keywords command higher or lower hourly rates",
             fn=lambda: build_keyword_chart("hourly")),
        dict(seq=2, task="1", title="Keyword Pay Premiums \u2014 Fixed",
             desc="The same analysis on fixed-price project budgets",
             fn=lambda: build_keyword_chart("fixed")),
        dict(seq=3, task="2", title="Category Treemap",
             desc="Derived categories sized by volume, split by classification confidence",
             fn=lambda: build_category_treemap(df)),
        dict(seq=4, task="3", title="Posting Volume Over Time",
             desc="Raw vs cleaned daily volume, showing the backfill artifact",
             fn=lambda: build_timeseries(df)),
        dict(seq=5, task="3", title="Weekly Seasonality",
             desc="Distribution of daily postings by weekday",
             fn=lambda: build_seasonality(df)),
        dict(seq=6, task="4", title="Hourly Rate World Map",
             desc="Interactive choropleth of median rate by client country",
             fn=lambda: build_rate_map(df)),
        dict(seq=7, task="4", title="Rates vs Evidence Base",
             desc="Median rate paired with posting volume per country",
             fn=lambda: build_rate_bars(df)),
        dict(seq=8, task="5", title="Recommendation Engine",
             desc="Delivered as a REST API (api.py, flask_api.py) and the dashboard search page",
             external=True),
        dict(seq=9, task="6", title="Market Dynamics Dashboard",
             desc="Delivered as the six-page Streamlit app (dashboard.py)",
             external=True),
        dict(seq=10, task="7", title="Remote Work Landscape",
             desc="Delivered as a written analysis in section 9 of the PDF report",
             external=True),
        dict(seq=11, task="8", title="14-Day Volume Forecast",
             desc="Projection with 95% empirical prediction intervals",
             fn=lambda: build_forecast(df)),
    ]

    entries = []
    for spec in sorted(SPEC, key=lambda s: s["seq"]):
        label = f"Task {spec['task']}: {spec['title']}"
        if spec.get("external"):
            entries.append(spec)
            print(f"  --   {label:<44} (deliverable is not a chart)")
            continue
        try:
            p = spec["fn"]()
            if p:
                entries.append({**spec, "path": p})
                print(f"  OK   {label:<44} -> {p.name}")
            else:
                print(f"  SKIP {label:<44} (source table missing)")
        except Exception as e:
            print(f"  FAIL {label:<44} {e}")

    idx = build_index(entries)
    n_charts = sum(1 for e in entries if not e.get("external"))
    print(f"\n  Index: {idx}")
    print(f"  {n_charts} interactive charts in {OUTDIR}/")
    if args.open:
        for e in entries:
            if not e.get("external"):
                print(f"    {e['path'].resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
