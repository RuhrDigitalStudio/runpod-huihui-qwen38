FROM ollama/ollama@sha256:b23b01ffcfbcf197571b0c8b0b0e73e437be02fc2d2786badea8a5734e047188

LABEL org.opencontainers.image.source="https://github.com/RuhrDigitalStudio/runpod-huihui-qwen38"
LABEL org.opencontainers.image.description="RunPod Serverless Ollama worker for Huihui Qwen3.8"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

COPY handler.py main.py /app/worker/

ENV OLLAMA_HOST=127.0.0.1:11434 \
    HOME=/runpod-volume/ollama-home \
    OLLAMA_MODELS=/runpod-volume/ollama \
    OLLAMA_CONTEXT_LENGTH=65536 \
    OLLAMA_FLASH_ATTENTION=1 \
    OLLAMA_KV_CACHE_TYPE=q8_0 \
    OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_NUM_PARALLEL=1 \
    MODEL_NAME=huihui_ai/Qwen3.8-abliterated \
    SERVED_MODEL_NAME=huihui-qwen38-27b-abliterated-q4 \
    STARTUP_TIMEOUT=1800

WORKDIR /app/worker
ENTRYPOINT ["python3", "/app/worker/main.py"]
