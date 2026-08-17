# =============================================================================
# Multi-stage build. The builder compiles wheels; the runtime image copies only
# the installed packages, so build toolchains never ship to production.
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


FROM python:3.11-slim AS runtime

# Non-root user: a container that runs as root is a container that can do
# damage if the process is ever compromised.
RUN useradd -m -u 1000 appuser

WORKDIR /app
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

COPY --chown=appuser:appuser job_market_analysis.py api.py flask_api.py dashboard.py \
     sql_analytics.py interactive_viz.py generate_report.py ./
RUN mkdir -p /app/results /app/data && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000 5000 8501

# Overridden per service in docker-compose.yml.
CMD ["python", "job_market_analysis.py", "--help"]
