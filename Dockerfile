# ─── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS build

# System deps: ffmpeg (clip encoding) + OpenCV native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from build stage
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy project source
COPY src/       ./src/
COPY frontend/  ./frontend/
COPY models/    ./models/

# Runtime data directories (mounted via volume in docker-compose)
RUN mkdir -p data/clips/blurred data/clips/secure data/members

EXPOSE 8000

CMD ["uvicorn", "src.app_api:app", "--host", "0.0.0.0", "--port", "8000"]
