"""Start llama-server, wait for model readiness, then accept RunPod jobs."""

import logging
import os
import signal
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HOST = "127.0.0.1"
PORT = os.getenv("LLAMA_PORT", "8000")

llama_process = None


def build_args() -> list[str]:
    model = os.environ["MODEL_NAME"]
    alias = os.getenv("SERVED_MODEL_NAME", "huihui-qwen38-27b-abliterated-q4")
    return [
        "/app/llama-server",
        "--host", HOST,
        "--port", PORT,
        "--hf-repo", model,
        "--alias", alias,
        "--ctx-size", os.getenv("CONTEXT_SIZE", "65536"),
        "--parallel", "1",
        "--n-gpu-layers", "999",
        "--flash-attn", "on",
        "--cache-type-k", os.getenv("CACHE_TYPE_K", "q8_0"),
        "--cache-type-v", os.getenv("CACHE_TYPE_V", "q8_0"),
        "--batch-size", os.getenv("BATCH_SIZE", "1024"),
        "--ubatch-size", os.getenv("UBATCH_SIZE", "256"),
        "--fit", "on",
        "--fit-ctx", os.getenv("MIN_CONTEXT_SIZE", "32768"),
        "--jinja",
        "--reasoning", "auto",
        "--no-mmproj",
    ]


def forward_signal(signum, _frame) -> None:
    if llama_process is not None and llama_process.poll() is None:
        llama_process.send_signal(signum)
    sys.exit(128 + signum)


def main() -> None:
    global llama_process
    if not os.getenv("MODEL_NAME"):
        raise RuntimeError("MODEL_NAME is required")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, forward_signal)

    args = build_args()
    logging.info("Starting llama-server for %s", os.environ["MODEL_NAME"])
    llama_process = subprocess.Popen(args)

    import handler as proxy_handler
    import runpod

    proxy_handler.llama_process = llama_process
    logging.info("Starting RunPod handler while llama-server initializes")
    runpod.serverless.start(
        {
            "handler": proxy_handler.handler,
            "concurrency_modifier": lambda _current: 1,
            "return_aggregate_stream": True,
        }
    )


if __name__ == "__main__":
    main()
