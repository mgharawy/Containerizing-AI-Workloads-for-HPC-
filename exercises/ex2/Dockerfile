# Stage 1: Base CUDA runtime image on Ubuntu 22.04
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install Python and core OS utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel utilities
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch built for CUDA 12.4
RUN python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install common ML dependencies
RUN python3 -m pip install --no-cache-dir \
    transformers \
    accelerate \
    datasets \
    scikit-learn \
    tqdm

# Set workspace and embed the verification script
WORKDIR /app
COPY verify_ml.py /app/verify_ml.py

# Default execution entrypoint
CMD ["python3", "/app/verify_ml.py"]
