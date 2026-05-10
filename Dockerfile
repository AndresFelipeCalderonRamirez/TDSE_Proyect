# =============================================================================
# WaterTwinML — Single Dockerfile
# Multi-stage build: installs all deps, runs Streamlit dashboard as default
# Also supports evaluation, ML training, and simulator via CMD override
# =============================================================================
# Usage:
#   docker build -t watertwin .
#   docker run -p 8501:8501 --env-file .env watertwin
#   docker run --env-file .env watertwin python evaluation/measure_metrics.py --all
#   docker run -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" \
#     watertwin python ml/train_models.py --generate-data --train-ensemble --tenant both
#   docker run -v "$(pwd)/data:/app/data" watertwin \
#     python simulator/local_simulator.py --tenant both --duration 300
# =============================================================================

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

# Install system build deps (for numpy, scipy, xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install all project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Health check for dashboard
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

# Default: launch the Streamlit dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501"]
