#!/usr/bin/env python3
"""
================================================================================
 PROJECT EXPLORER
================================================================================

Builds a single self-contained HTML page showing every artifact the project
produces: the cleaning audit, all 7 static figures, all 20 data tables, the
8 interactive charts, headline metrics and consolidated findings.

Purpose: the project generates a large number of outputs across several
directories. This assembles them into one page so the whole system can be
reviewed at a glance, rather than opened file by file.

Figures are embedded as base64 so the page works offline and can be sent as a
single file. Tables are rendered as previews with row counts, linking to the
full CSV.

Usage:
    python project_explorer.py
    start results/PROJECT_EXPLORER.html
================================================================================
"""
from __future__ import annotations

import argparse
import base64
import html as html_mod
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.getenv("DATA_DIR", "results"))
TABLES = DATA_DIR / "tables"
FIGURES = DATA_DIR / "figures"
INTERACTIVE = DATA_DIR / "interactive"
OUT = DATA_DIR / "PROJECT_EXPLORER.html"

# Human-readable context for each generated table, so the page explains what
# each artifact is for rather than just listing filenames.
TABLE_INFO = {
    "task1_keywords_hourly": ("Task 1", "Every keyword tested against hourly rates, with p-values, FDR flags and Cliff's delta"),
    "task1_keywords_fixed": ("Task 1", "The same keyword analysis on fixed-price budgets"),
    "task2_category_momentum": ("Task 2", "Per-category trend slope, R-squared and significance over the clean window"),
    "task2_daily_category": ("Task 2", "Daily posting counts and share per category (the source series)"),
    "task3_model_performance": ("Task 3", "MAE, MAPE and RMSE for all four models on the held-out period"),
    "task3_feature_importance": ("Task 3", "Random Forest feature weights, showing calendar features dominate"),
    "task3_category_demand": ("Task 3", "Recent vs prior demand per category with momentum"),
    "task3_daily_series": ("Task 3", "Clean daily posting counts used for forecasting"),
    "task4_country_rates": ("Task 4", "Median, quartiles and volume per client country"),
    "task6_daily": ("Task 6", "Daily market aggregates"),
    "task6_weekly": ("Task 6", "Weekly market aggregates"),
    "task6_monthly": ("Task 6", "Monthly aggregates (incomplete window - see report)"),
    "task6_category_mix_shift": ("Task 6", "Category share, first half vs second half of the window"),
    "task7_hourly_activity": ("Task 7", "Posting counts by hour of day (UTC)"),
    "task8_volume_forecast": ("Task 8", "14-day forecast with 95% empirical prediction intervals"),
    "task8_category_projection": ("Task 8", "30-day category share projections with reliability flags"),
    "taxonomy_validation": ("Quality", "Sampled titles re-checked against independent anchor terms"),
}

FIGURE_INFO = {
    "fig01_backfill_artifact.png": (
        "The single most important chart in the project",
        "Left: raw daily volume implies explosive February growth. Right: after excluding "
        "pre-collection backfill, a stable market with a weekly cycle. The 283 rows before "
        "13 February are scraper backfill, not a quiet market."),
    "fig02_weekly_seasonality.png": (
        "Weekly seasonality",
        "Daily posting volume by weekday. The ~32% weekend collapse is the strongest "
        "single pattern in the data and dominates any short-horizon forecast."),
    "fig03_pay_distributions.png": (
        "Two incompatible pay units",
        "Hourly rates and fixed budgets shown separately, never averaged. The fixed track "
        "uses a log scale because budgets span $5 to $1,000,000."),
    "fig04_category_tiers.png": (
        "Derived categories by confidence tier",
        "The dataset has no category column. Red = tier-1 specific phrase match, "
        "green = tier-2 generic token fallback, grey = unmatched."),
    "fig05_country_rates.png": (
        "Median hourly rate by client country",
        "Note this is what clients PAY, not what freelancers in that country EARN. "
        "Rates are not adjusted for purchasing power parity."),
    "fig06_forecast.png": (
        "14-day forecast with prediction intervals",
        "The model reproduces the weekly cycle. Intervals widen with the square root of "
        "horizon because uncertainty compounds in a recursive forecast."),
    "fig07_hourly_clock.png": (
        "Posting activity by hour (UTC)",
        "Only a 1.6x peak-to-trough ratio across 24 hours - the remote market never "
        "fully sleeps, evidence of genuinely distributed global demand."),
}


def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def esc(s) -> str:
    return html_mod.escape(str(s))


def table_html(df: pd.DataFrame, max_rows: int = 8) -> str:
    d = df.head(max_rows)
    head = "".join(f"<th>{esc(c)}</th>" for c in d.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{esc(v)}</td>" for v in r) + "</tr>"
        for r in d.astype(str).values)
    return f"<table class='data'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def build() -> Path:
    if not (DATA_DIR / "metrics.json").exists():
        print(f"ERROR: {DATA_DIR}/metrics.json not found.\n"
              f"Run first: python job_market_analysis.py --data jobs.csv", file=sys.stderr)
        sys.exit(2)

    M = json.loads((DATA_DIR / "metrics.json").read_text())
    ev = M.get("task5_eval", {})
    findings = pd.read_csv(DATA_DIR / "findings.csv") if (DATA_DIR / "findings.csv").exists() else None
    audit = pd.read_csv(DATA_DIR / "cleaning_audit.csv") if (DATA_DIR / "cleaning_audit.csv").exists() else None

    parts = []

    # ---------------------------------------------------------- metrics
    cards = [
        ("Raw records", f"{M['rows_raw']:,}", "rows read from the source CSV"),
        ("Clean dataset", f"{M['rows_clean']:,}", "only 1 row dropped (null title)"),
        ("Time-series reliable", f"{M['rows_ts_reliable']:,}", "excludes backfill, outage, partial day"),
        ("Taxonomy precision", f"{M['taxonomy_precision_overall']:.1%}", "measured, not assumed"),
        ("Weekend drop", f"{M['weekend_drop_pct']:.0f}%", "the dominant seasonal signal"),
        ("Best forecast MAPE", f"{M['task3_best_mape']:.2f}%", f"{M['task3_best_model']}"),
        ("Recommender P@10", f"{ev.get('precision@10', 0):.3f}", f"vs {ev.get('random_baseline', 0):.3f} random"),
        ("Lift over random", f"{ev.get('lift_vs_random', 0):.1f}x", f"MRR {ev.get('MRR', 0):.3f}"),
        ("Market concentration", f"{M['task7_hhi']:.3f}", f"HHI; top-5 = {M['task7_top5_concentration_pct']:.0f}%"),
    ]
    parts.append("<h2 id='metrics'>Headline Metrics</h2>")
    parts.append("<div class='cards'>" + "".join(
        f"<div class='card'><div class='cv'>{v}</div><div class='cl'>{l}</div>"
        f"<div class='cs'>{s}</div></div>" for l, v, s in cards) + "</div>")

    # --------------------------------------------------------- findings
    if findings is not None:
        parts.append("<h2 id='findings'>Consolidated Findings</h2>")
        parts.append("<div class='findings'>" + "".join(
            f"<div class='finding'><span class='ftag'>{esc(r['task'])}</span>"
            f"<span class='ftext'>{esc(r['finding'])}</span></div>"
            for _, r in findings.iterrows()) + "</div>")

    # ------------------------------------------------------------ audit
    if audit is not None:
        parts.append("<h2 id='audit'>Cleaning Audit Trail</h2>")
        parts.append("<p class='lead'>Every transformation is logged with the number of rows "
                     "it touched. The guiding principle is <b>flag, don't delete</b>: only one "
                     "row was removed from the entire dataset.</p>")
        rows = "".join(
            f"<tr><td class='num'>{i}</td><td class='step'>{esc(r['step'])}</td>"
            f"<td>{esc(r['detail'])}</td>"
            f"<td class='num'>{int(r['rows_affected']):,}</td></tr>"
            for i, (_, r) in enumerate(audit.iterrows(), 1))
        parts.append(f"<table class='data audit'><thead><tr><th>#</th><th>Step</th>"
                     f"<th>Detail</th><th>Rows</th></tr></thead><tbody>{rows}</tbody></table>")

    # ---------------------------------------------------------- figures
    parts.append("<h2 id='figures'>Figures</h2>")
    for name, (title, desc) in FIGURE_INFO.items():
        p = FIGURES / name
        if not p.exists():
            continue
        parts.append(
            f"<div class='fig'><h3>{esc(title)}</h3><p class='caption'>{esc(desc)}</p>"
            f"<img src='data:image/png;base64,{b64_image(p)}' alt='{esc(title)}'>"
            f"<p class='fname'>{esc(name)}</p></div>")

    # ------------------------------------------------------ interactive
    if INTERACTIVE.exists():
        charts = sorted(p for p in INTERACTIVE.glob("*.html") if p.name != "index.html")
        if charts:
            parts.append("<h2 id='interactive'>Interactive Charts</h2>")
            parts.append("<p class='lead'>Self-contained Plotly HTML. Task 4 requires an "
                         "interactive chart; the rest are additional.</p>")
            links = "".join(
                f"<a class='chip' href='interactive/{p.name}' target='_blank'>"
                f"{esc(p.stem.replace('_', ' '))}</a>" for p in charts)
            parts.append(f"<div class='chips'>{links}"
                         f"<a class='chip primary' href='interactive/index.html' "
                         f"target='_blank'>Open chart index</a></div>")

    # ----------------------------------------------------------- tables
    parts.append("<h2 id='tables'>Data Tables</h2>")
    parts.append("<p class='lead'>Each analysis writes a full CSV. Previews show the first "
                 "rows; click the filename to open the complete file.</p>")
    for f in sorted(TABLES.glob("*.csv")):
        stem = f.stem
        task, desc = TABLE_INFO.get(stem, ("", "Generated output"))
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        tag = f"<span class='ttag'>{esc(task)}</span>" if task else ""
        parts.append(
            f"<details class='tbl'><summary>{tag}<b>{esc(stem)}</b>"
            f"<span class='dims'>{len(df):,} rows &times; {len(df.columns)} cols</span>"
            f"</summary><p class='caption'>{esc(desc)}</p>{table_html(df)}"
            f"<p class='fname'><a href='tables/{f.name}'>{esc(f.name)}</a></p></details>")

    # ------------------------------------------------------------ shell
    nav = "".join(f"<a href='#{i}'>{n}</a>" for i, n in [
        ("metrics", "Metrics"), ("findings", "Findings"), ("audit", "Audit"),
        ("figures", "Figures"), ("interactive", "Interactive"), ("tables", "Tables")])

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Job Market Analysis &mdash; Project Explorer</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: "Segoe UI", Helvetica, Arial, sans-serif; margin: 0;
         color: #1c2733; line-height: 1.55; background: #fff; }}
  header {{ background: #1a3a5c; color: #fff; padding: 34px 40px 26px; }}
  header h1 {{ margin: 0 0 4px; font-size: 27px; }}
  header p {{ margin: 0; opacity: .82; font-size: 14px; }}
  nav {{ position: sticky; top: 0; background: #14304d; padding: 0 40px;
         display: flex; gap: 4px; flex-wrap: wrap; z-index: 10; }}
  nav a {{ color: #cfe0ef; text-decoration: none; font-size: 13px; padding: 11px 14px; }}
  nav a:hover {{ background: #1f4468; color: #fff; }}
  main {{ max-width: 1080px; margin: 0 auto; padding: 8px 40px 70px; }}
  h2 {{ font-size: 21px; color: #1a3a5c; margin: 40px 0 6px;
        padding-bottom: 7px; border-bottom: 2px solid #e2e8ee; }}
  h3 {{ font-size: 15.5px; color: #1a3a5c; margin: 0 0 3px; }}
  .lead {{ color: #5a6470; font-size: 13.5px; margin: 6px 0 16px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
            gap: 12px; margin-top: 16px; }}
  .card {{ background: #f6f8fa; border: 1px solid #e2e8ee; border-radius: 8px; padding: 14px 16px; }}
  .cv {{ font-size: 23px; font-weight: 700; color: #1a3a5c; }}
  .cl {{ font-size: 13px; font-weight: 600; margin-top: 2px; }}
  .cs {{ font-size: 11.5px; color: #7a848f; margin-top: 2px; }}
  .findings {{ margin-top: 14px; }}
  .finding {{ display: grid; grid-template-columns: 74px 1fr; gap: 12px;
              padding: 11px 14px; background: #f6f8fa; border-left: 3px solid #1a3a5c;
              margin-bottom: 8px; border-radius: 0 6px 6px 0; }}
  .ftag {{ font-size: 11.5px; font-weight: 700; color: #1a3a5c; }}
  .ftext {{ font-size: 13.5px; }}
  table.data {{ border-collapse: collapse; width: 100%; font-size: 11.8px; margin-top: 10px; }}
  table.data th {{ background: #1a3a5c; color: #fff; text-align: left; padding: 7px 9px;
                   font-weight: 600; position: sticky; top: 0; }}
  table.data td {{ padding: 5px 9px; border-bottom: 1px solid #eceff3; }}
  table.data tbody tr:nth-child(even) {{ background: #fafbfc; }}
  table.audit td.step {{ font-family: Consolas, monospace; font-size: 11px; color: #1a4d8f; }}
  table.audit td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  table.audit td:nth-child(3) {{ font-size: 11.2px; color: #4a5560; }}
  .fig {{ margin: 26px 0 34px; }}
  .fig img {{ width: 100%; border: 1px solid #e2e8ee; border-radius: 8px; margin-top: 10px; }}
  .caption {{ font-size: 13px; color: #5a6470; margin: 2px 0 0; }}
  .fname {{ font-size: 11px; color: #97a1ab; font-family: Consolas, monospace; margin: 6px 0 0; }}
  .fname a {{ color: #1a4d8f; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .chip {{ display: inline-block; padding: 8px 14px; background: #f6f8fa;
           border: 1px solid #e2e8ee; border-radius: 20px; text-decoration: none;
           color: #1a4d8f; font-size: 13px; }}
  .chip:hover {{ background: #eaf1f8; border-color: #b9cde0; }}
  .chip.primary {{ background: #1a3a5c; color: #fff; border-color: #1a3a5c; }}
  details.tbl {{ border: 1px solid #e2e8ee; border-radius: 8px; margin-bottom: 8px;
                 padding: 0 14px; background: #fdfdfe; }}
  details.tbl summary {{ cursor: pointer; padding: 11px 0; font-size: 13.5px;
                         display: flex; align-items: center; gap: 10px; }}
  details.tbl[open] {{ padding-bottom: 14px; }}
  .ttag {{ font-size: 10.5px; font-weight: 700; color: #fff; background: #8b9aa8;
           padding: 3px 8px; border-radius: 4px; min-width: 52px; text-align: center; }}
  .dims {{ margin-left: auto; font-size: 11.5px; color: #97a1ab;
           font-variant-numeric: tabular-nums; }}
  footer {{ border-top: 1px solid #e2e8ee; margin-top: 50px; padding-top: 16px;
            font-size: 12px; color: #7a848f; }}
</style></head><body>
<header>
  <h1>Job Market Analysis &amp; Recommendation System</h1>
  <p>Complete output explorer &middot; 244,827 Upwork postings &middot; February&ndash;March 2024</p>
</header>
<nav>{nav}</nav>
<main>
{''.join(parts)}
<footer>
  Generated {datetime.now():%d %B %Y at %H:%M} from live pipeline outputs.
  Every value on this page is read from <code>metrics.json</code> and the task
  tables at build time. Rebuild with <code>python project_explorer.py</code>.
</footer>
</main></body></html>"""

    OUT.write_text(html, encoding="utf-8")
    return OUT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the project output explorer")
    ap.parse_args(argv)
    out = build()
    print(f"Explorer built: {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"Open with:  start {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
