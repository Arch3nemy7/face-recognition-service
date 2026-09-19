# Multi-stage Dockerfile for Face Recognition Service
# This Dockerfile creates an optimized production image with minimal size

# Stage 1: Builder
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages.
# insightface builds a Cython extension from sdist, so the compilers stay.
# The GL/X11 dev libs (libgl1, libglx-mesa0, libsm6, libxext6,
# libxrender-dev) are dropped: they were only ever needed by
# opencv-python's GUI/imshow bindings, and requirements.txt now installs
# opencv-python-headless, which has no such bindings to build against.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    libglib2.0-0 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt constraints.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -c constraints.txt

# antelopev2.zip is an unauthenticated third-party download (GitHub release
# asset, no package signing), so its integrity is pinned by hash rather
# than trusted on TLS alone.
ARG ANTELOPEV2_SHA256=8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5

# Download antelopev2 model pack during build so it's baked into the image.
# The zip may have a nested folder (antelopev2/antelopev2/*.onnx), so we
# detect and flatten it to ensure .onnx files sit directly under models/antelopev2/.
# `curl -fL` so an HTTP error (e.g. a dead release link) fails the build
# instead of baking an HTML error page in as "antelopev2.zip".
RUN mkdir -p /app/.insightface/models && \
    curl -fL "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip" \
         -o /tmp/antelopev2.zip && \
    echo "${ANTELOPEV2_SHA256}  /tmp/antelopev2.zip" | sha256sum -c - && \
    unzip -o /tmp/antelopev2.zip -d /tmp/antelopev2_extract && \
    if [ -d "/tmp/antelopev2_extract/antelopev2/antelopev2" ]; then \
        mv /tmp/antelopev2_extract/antelopev2/antelopev2 /app/.insightface/models/antelopev2; \
    elif [ -d "/tmp/antelopev2_extract/antelopev2" ]; then \
        mv /tmp/antelopev2_extract/antelopev2 /app/.insightface/models/antelopev2; \
    else \
        mkdir -p /app/.insightface/models/antelopev2 && \
        mv /tmp/antelopev2_extract/*.onnx /app/.insightface/models/antelopev2/; \
    fi && \
    rm -rf /tmp/antelopev2.zip /tmp/antelopev2_extract


# Stage 2: Runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime system dependencies. opencv-python-headless has no GUI/X11
# bindings, so the GL/X11 runtime libs (libgl1, libglx-mesa0, libsm6,
# libxext6, libxrender1) that non-headless opencv needed are dropped.
# libglib2.0-0 stays: several shared libs opencv/onnxruntime load pull GLib
# in as a real runtime link dependency, not just a GUI toolkit.
# libgomp1 stays: onnxruntime links libgomp for OpenMP-based threading.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy pre-downloaded antelopev2 model from builder
COPY --from=builder /app/.insightface /app/.insightface

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # InsightFace model download location
    INSIGHTFACE_HOME=/app/.insightface

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser face_recognition_service /app/face_recognition_service

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

# Run the application. `python -m ...` (rather than invoking uvicorn
# directly) runs main.py's own `if __name__ == "__main__":` block, which
# passes host/port/log_level from Settings -- so LOG_LEVEL actually reaches
# uvicorn instead of it always defaulting to "info" regardless of the
# environment.
CMD ["python", "-m", "face_recognition_service.main"]
