FROM python:3.14-slim

# LightGBM needs the OpenMP runtime library, which slim images don't include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (this layer is cached until requirements change)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Then copy the application code and model
COPY configs/ configs/
COPY src/ src/
COPY models/ models/

# Don't run as root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]