# ──────────────────────────────────────────────────────────────
# Dockerfile — Sensitive Data Detection & Compliance Assistant
# ──────────────────────────────────────────────────────────────
# Build:  docker build -t sdd-assistant .
# Run:    docker run -p 8501:8501 --env-file .env sdd-assistant
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps for OCR (tesseract) and PDF→image (poppler)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Streamlit config: disable telemetry, set server port
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "app.py", "--server.headless=true"]
