FROM python:3.11-slim

LABEL maintainer="DA5402 Medical Imaging MLOps"
LABEL description="Inference server for pneumonia and brain tumor classification"

WORKDIR /app

# System deps for Pillow and psutil
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (large layer, benefits from Docker cache)
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir \
        torch==2.11.0+cpu \
        torchvision==0.26.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining inference dependencies
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy application source
COPY src/ ./src/

# Bake trained model checkpoints into the image
# These are the production weights — no retraining needed at runtime
COPY models/pneumonia/pneumonia_resnet50.pt  ./models/pneumonia/pneumonia_resnet50.pt
COPY models/brain_tumor/brain_resnet50.pt   ./models/brain_tumor/brain_resnet50.pt

# Copy frontend assets
COPY frontend/ ./frontend/

# Runtime temp directory for uploaded images
RUN mkdir -p temp

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

# Liveness probe — curl /healthz every 30s, 20s startup grace period
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
