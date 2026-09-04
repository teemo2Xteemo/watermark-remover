# CUDA runtime (NVIDIA GPU). Pin is major.minor (CUDA 12.4, Ubuntu 22.04).
# Weights are NOT downloaded at build time — mount /models at run time.
# Requires nvidia-container-toolkit on the host.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models \
    PATH="/opt/venv/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY scripts /app/scripts

# lama extra is CPU onnxruntime; CUDA image uses onnxruntime-gpu instead.
RUN pip install --upgrade pip \
    && pip install --no-cache-dir ".[ui,video]" "onnxruntime-gpu>=1.29" \
    && groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --home-dir /home/app \
        --shell /usr/sbin/nologin app \
    && mkdir -p /models /input /output \
    && chown -R app:app /app /models /input /output /home/app

USER app

VOLUME ["/models", "/input", "/output"]

EXPOSE 7860

ENTRYPOINT ["watermark-remover"]
CMD ["ui"]
