# Multi-stage Dockerfile for Face Recognition Service
# This Dockerfile creates an optimized production image with minimal size

# Stage 1: Builder
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download antelopev2 model pack during build so it's baked into the image.
# The zip may have a nested folder (antelopev2/antelopev2/*.onnx), so we
# detect and flatten it to ensure .onnx files sit directly under models/antelopev2/.
RUN mkdir -p /app/.insightface/models && \
    curl -L "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip" \
         -o /tmp/antelopev2.zip && \
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

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
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

# Run the application
CMD ["uvicorn", "face_recognition_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
