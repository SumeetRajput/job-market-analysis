# Job Market Analysis & Recommendation System

End-to-end analytics pipeline, recommendation engine, dual REST APIs, interactive
dashboard, SQL analytics layer and containerised deployment over 244,827 Upwork
job postings (Feb–Mar 2024). Covers all eight project tasks.

```
job-market-project/
├── job_market_analysis.py     # all 8 tasks, single file, ~50s run
├── api.py                     # FastAPI service (auto Swagger docs)
├── flask_api.py               # Flask service (brief-specified stack)
├── dashboard.py               # Streamlit dashboard, 6 pages
├── sql_analytics.py           # SQL layer: CTEs + window functions
├── interactive_viz.py         # Plotly interactive charts + choropleth
├── generate_report.py         # builds the consolidated PDF report
├── Dockerfile                 # multi-stage, non-root
├── docker-compose.yml         # 4 services, one command
├── requirements.txt
├── jobs.csv                   # your raw data
└── results/                   # generated outputs
```

---

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\activate                 # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Run in this order** — everything downstream reads the analysis outputs:

```powershell
python job_market_analysis.py --selftest        # 21 invariant tests, <1s
python job_market_analysis.py --data jobs.csv   # full analysis, ~50s
python sql_analytics.py --build                 # load into SQLite
python sql_analytics.py --verify                # cross-check SQL vs pandas
python interactive_viz.py                       # 8 interactive charts
python generate_report.py                       # 17-page PDF report
```

Then the services, each in its own terminal:

```powershell
uvicorn api:app --reload --port 8000    # http://localhost:8000/docs
python flask_api.py                     # http://localhost:5000
streamlit run dashboard.py              # http://localhost:8501
```

## Docker

```bash
docker compose up --build
```

Runs the analysis to completion, then starts all three services. The
`service_completed_successfully` condition means the services wait for the
analysis and load the cleaned parquet, rather than each re-cleaning 245k rows.

| Service | URL |
|---|---|
| FastAPI docs | http://localhost:8000/docs |
| Flask API | http://localhost:5000 |
| Dashboard | http://localhost:8501 |

---

## Three data-integrity findings that shape the whole pipeline

**1. The raw scrape contains a false growth curve.** 283 rows over the 48 days
before 2024-02-13 (~6/day), then 244,545 rows over the next 41 days (~5,960/day).
Those early rows are scraper backfill, not a quiet market. Plotting raw volume
"discovers" ~100,000% growth in February that is purely an artifact of when
collection started. Every time-series task filters on `ts_reliable`.

**2. Hourly rates and fixed budgets are not interconvertible.** `budget` is
populated if and only if `is_hourly` is False (zero overlap). Converting a $500
project budget to an hourly rate needs project duration, which the dataset lacks.
Tasks 1 and 4 run as two parallel tracks. A self-test enforces this.

**3. The usable window is ~5 weeks, not 5 months.** After removing backfill, the
truncated final day and the 2024-02-15 outage, 39 clean days remain. The pipeline
is monthly-capable but reports at daily/weekly grain rather than fabricating
month-over-month growth.

---

## Results

| Task | Result |
|---|---|
| 1 | Hourly: `attorney` $105/hr vs `philippines` $4/hr (26x). Fixed: `ticket` $1,000 vs `quiz` $5 (200x). 990 keywords tested, FDR-corrected, effect sizes reported |
| 2 | **Zero** categories significantly rising. Two significantly falling: Writing & Translation (R²=0.44), Admin & Support (R²=0.27) |
| 3 | Seasonal naive (lag-7) wins at **1.60% MAPE**; RandomForest 3.81%, Ridge 5.67% |
| 4 | 46 countries qualify. Vietnam $27.50/hr → South Korea $10.00/hr (2.8x). Kruskal-Wallis p < 1e-300 |
| 5 | **precision@10 = 0.636 vs 0.080 random (7.9x lift)**, MRR 0.761 |
| 6 | Volume swings 32% weekly; hourly/fixed split varies only 1.93 ppt |
| 7 | Reframed. HHI 0.192, top-5 = 66%, 24h activity (1.6x peak/trough), 29% technical vs 28% creative |
| 8 | 14-day forecast, widening empirical intervals. Only 2/22 category trends reliable |

### Three results that look like failures and aren't

**Task 3: the naive baseline won.** RandomForest lost to "same as last week."
Feature importance shows why — `sin1`, `dow`, `lag_7` carry 93% of the weight, so
the model spent its capacity relearning the calendar. Reported, not hidden.

**Task 2: nothing is emerging.** Zero significant risers across 39 days.
Manufacturing "emerging categories" from noise is the error the R² column exposes.

**Task 7: unanswerable as posed.** Upwork is remote-only, so "the shift to remote"
has no comparison group. The structure of remote demand is analysed instead.

---

## Methodological positions

- **Flag, don't delete.** Rows carry quality flags; each task chooses its own
  exclusions. Only null-title rows dropped (1 row).
- **No pay imputation.** 38,514 postings (15.7%) disclose no rate. Imputing would
  invent a sixth of the pay data.
- **Temporal split, never random.** A random split on time-series data trains on
  the future to predict the past. Train = first 24 days, test = final 8.
- **Baseline before model.** Every forecast is benchmarked against seasonal naive.
- **Effect size, not just p-values.** With n>100k, trivial differences reach
  significance. Cliff's delta + Benjamini-Hochberg FDR throughout.
- **Taxonomy precision measured:** 89.6% against independent anchor terms
  (92.5% tier-1, 86.7% tier-2) — not assumed.

## Technology choices

- **Both Flask and FastAPI provided.** The brief specifies Flask; FastAPI was
  preferred for auto-generated OpenAPI docs (a Task 5 deliverable) and validation.
  Both import the same `JobRecommender` — one model, evaluated once.
- **TensorFlow deliberately not used.** With 24 training days and a naive baseline
  at 1.60% MAPE, a neural network would be slower, less interpretable and less
  accurate. Model choice should be proportionate to data volume.
- **SQL runs alongside pandas.** `--verify` cross-checks the SQL layer against the
  pandas pipeline (5/5 agree). Two independent implementations agreeing is
  evidence; one is an assertion.

---

## CLI reference

```bash
# Analysis
python job_market_analysis.py --data jobs.csv [--quick] [--horizon 21]
python job_market_analysis.py --recommend "power bi dashboard"
python job_market_analysis.py --selftest

# SQL
python sql_analytics.py --build
python sql_analytics.py --queries --export
python sql_analytics.py --query top_countries
python sql_analytics.py --verify

# Visuals and report
python interactive_viz.py
python generate_report.py --output Report.pdf
```

## API reference

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service and index status |
| `GET /recommend?q=...&k=10` | Recommendations, with evaluation metrics attached |
| `GET /categories` | Categories with counts and confidence tiers |
| `GET /countries` | Median hourly rate by client country |
| `GET /stats` | Dataset headline statistics |
| `GET /trends?grain=weekly` | Volume and composition over time |

```bash
curl "http://localhost:8000/recommend?q=python%20data%20analyst&k=5"
```

## Outputs

```
results/
├── tables/          20 CSVs, one per analysis
├── figures/         7 static PNGs
├── interactive/     8 Plotly HTML charts + index.html
├── sql/             SQL query exports
├── cleaning_audit.csv
├── findings.csv
├── metrics.json
└── jobs_clean.parquet
Job_Market_Analysis_Report.pdf      17-page consolidated report
```

## Key derived columns

| Column | Meaning |
|---|---|
| `ts_reliable` | Safe for time-series counts |
| `pay_analysable` | Pay disclosed AND plausible |
| `pay_disclosed` | Whether a rate/budget was stated |
| `category` / `category_tier` | Derived category + assigning rule tier |
| `country_grouped` | Rare countries bucketed for stable statistics |

## Troubleshooting

**`data file not found`** — terminal isn't in the project folder, or the filename
differs. Run `dir` / `ls`. Filenames with spaces or `(1)` need double quotes.

**`Could not import module "api"`** — `api.py` isn't in the folder, or you're not
running from the project directory.

**`ModuleNotFoundError`** — venv not active, or VS Code is using a different
interpreter. Check for `(.venv)` in the prompt.

**PowerShell blocks `activate`** — run once:
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**Pasting multiple lines fails** — PowerShell mangles multi-line pastes. Paste one
command at a time.
