"""Start Ollama and accept RunPod jobs while the model initializes."""

import logging
import os
import signal
import subprocess
import sys
import threading
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ollama_process = None
model_ready = threading.Event()
model_error = threading.Event()


def initialize_model() -> None:
    model = os.environ["MODEL_NAME"]
    alias = os.environ["SERVED_MODEL_NAME"]
    deadline = time.monotonic() + float(os.getenv("STARTUP_TIMEOUT", "1800"))

    while time.monotonic() < deadline:
        if ollama_process is not None and ollama_process.poll() is not None:
            model_error.set()
            return
        result = subprocess.run(
            ["ollama", "list"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            break
        time.sleep(2)
    else:
        logging.error("Ollama did not become ready before the startup timeout")
        model_error.set()
        return

    logging.info("Ensuring persistent Ollama model %s", model)
    pull = subprocess.run(
        ["ollama", "pull", model], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    if pull.returncode != 0:
        logging.error("Ollama model pull failed with exit code %s", pull.returncode)
        model_error.set()
        return

    copy = subprocess.run(
        ["ollama", "cp", model, alias],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    if copy.returncode != 0:
        logging.error("Ollama model alias creation failed with exit code %s", copy.returncode)
        model_error.set()
        return

    logging.info("Ollama model is ready as %s", alias)
    model_ready.set()


def forward_signal(signum, _frame) -> None:
    if ollama_process is not None and ollama_process.poll() is None:
        ollama_process.send_signal(signum)
    sys.exit(128 + signum)


def main() -> None:
    global ollama_process
    if not os.getenv("MODEL_NAME"):
        raise RuntimeError("MODEL_NAME is required")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, forward_signal)

    logging.info("Starting Ollama for %s", os.environ["MODEL_NAME"])
    ollama_process = subprocess.Popen(["ollama", "serve"])
    threading.Thread(target=initialize_model, name="model-initializer", daemon=True).start()

    import handler as proxy_handler
    import runpod

    proxy_handler.ollama_process = ollama_process
    proxy_handler.model_ready = model_ready
    proxy_handler.model_error = model_error
    logging.info("Starting RunPod handler while Ollama initializes")
    runpod.serverless.start(
        {
            "handler": proxy_handler.handler,
            "concurrency_modifier": lambda _current: 1,
            "return_aggregate_stream": True,
        }
    )


if __name__ == "__main__":
    main()
