# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11 and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    curl \
    ca-certificates \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

RUN mkdir -p /app/models && chmod 777 /app/models

# Environment variables for HuggingFace and CUDA
ENV HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models \
    HF_DATASETS_CACHE=/app/models \
    TRITON_CACHE_DIR=/app/models/.triton \
    TORCH_HOME=/app/models/.cache/torch \
    XDG_CACHE_HOME=/app/models/.cache \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    TORCH_COMPILE=0 \
    TORCHINDUCTOR_DISABLE=1 \
    UNSLOTH_COMPILE_DISABLE=1 \
    UNSLOTH_DISABLE_FAST_GENERATION=1

COPY requirements.txt .

# Install PyTorch with CUDA support FIRST (before requirements.txt)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .


CMD ["python3", "arc_main.py"]
