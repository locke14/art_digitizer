FROM python:3.11-slim

# System deps needed by OpenCV (opencv-python)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY web/requirements.txt /app/web/requirements.txt
RUN pip install --no-cache-dir -r /app/web/requirements.txt \
    && pip install --no-cache-dir opencv-python pillow numpy

# Copy source
COPY . /app

# Cloud Run provides PORT env; default to 8080
ENV PORT=8080

# Use /tmp for ephemeral per-run data by default in containerized envs
ENV DATA_DIR=/tmp/artdig_data

CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port ${PORT}"]

