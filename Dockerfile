# OCR Bridge Service — Docker container
# Includes Tesseract OCR (with Latvian language pack) and Poppler.
#
# Build:   docker build -t ocr-bridge .
# Run:     docker run -p 8000:8000 ocr-bridge
# Railway: Deploy repo root — Railway detects Dockerfile automatically.

FROM python:3.11-slim

# System deps: Tesseract + Latvian lang pack + Poppler (for pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-lav \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying source (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full repo (extractors, bridge modules, config)
COPY . .

# Bridge listens on 0.0.0.0:8000
ENV OCR_BRIDGE_HOST=0.0.0.0
ENV OCR_BRIDGE_PORT=8000

EXPOSE 8000

# Health check — Railway and most hosts use this
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"

CMD ["python", "tools/ocr_bridge_service.py"]
