FROM ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:5f245844a3244287bf69d4964670a021acf6f336467ae1feaa0024b4665c018d

LABEL org.opencontainers.image.source="https://github.com/RuhrDigitalStudio/runpod-huihui-qwen38"
LABEL org.opencontainers.image.description="RunPod Serverless llama.cpp worker for Huihui Qwen3.8 GGUF"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

COPY handler.py main.py /app/worker/

ENV HF_HOME=/runpod-volume/huggingface-cache \
    LLAMA_CACHE=/runpod-volume/huggingface-cache \
    LD_LIBRARY_PATH=/app \
    MODEL_NAME=huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF:Q4_K \
    SERVED_MODEL_NAME=huihui-qwen38-27b-abliterated-q4 \
    CONTEXT_SIZE=65536 \
    MIN_CONTEXT_SIZE=32768

WORKDIR /app/worker
ENTRYPOINT ["python3", "/app/worker/main.py"]
