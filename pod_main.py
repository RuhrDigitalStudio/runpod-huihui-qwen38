"""Run an authenticated Ollama API for an on-demand RunPod Pod."""

import asyncio
from contextlib import suppress
import hmac
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Optional
from urllib.request import Request, urlopen

import aiohttp
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
PROXY_PORT = int(os.getenv("POD_PROXY_PORT", "8000"))
STARTUP_TIMEOUT = float(os.getenv("STARTUP_TIMEOUT", "7200"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "3600"))
STREAM_HEARTBEAT_INTERVAL = float(os.getenv("STREAM_HEARTBEAT_INTERVAL", "15"))

ollama_process: Optional[subprocess.Popen] = None
model_ready = threading.Event()
model_error = threading.Event()


def message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(parts)


def normalize_fable_responses_request(body: bytes) -> bytes:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if payload.get("model") != os.getenv("SECONDARY_MODEL_NAME"):
        return body
    messages = payload.get("input")
    if not isinstance(messages, list):
        return body

    instructions = []
    existing = message_text(payload.get("instructions"))
    if existing:
        instructions.append(existing)
    remaining = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") in {"system", "developer"}:
            text = message_text(message.get("content"))
            if text:
                instructions.append(text)
        else:
            remaining.append(message)
    if len(remaining) == len(messages):
        return body
    payload["input"] = remaining
    payload["instructions"] = "\n\n".join(instructions)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def is_streaming_responses_request(request: web.Request, body: bytes) -> bool:
    if request.method != "POST" or request.path != "/v1/responses":
        return False
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return payload.get("stream") is True


def command_succeeds(args: list[str]) -> bool:
    return subprocess.run(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def context_length_for_vram(vram_gib: float) -> int:
    if vram_gib >= 48:
        return 262144
    if vram_gib >= 40:
        return 131072
    if vram_gib >= 32:
        return 65536
    return 32768


def configure_context_length() -> int:
    configured = os.getenv("OLLAMA_CONTEXT_LENGTH", "AUTO").strip().upper()
    if configured != "AUTO":
        context_length = int(configured)
    else:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        total_mib = sum(int(line.strip()) for line in result.stdout.splitlines() if line.strip())
        context_length = context_length_for_vram(total_mib / 1024)
        logging.info(
            "Selected %s-token context for %.1f GiB aggregate VRAM",
            context_length,
            total_mib / 1024,
        )
    os.environ["OLLAMA_CONTEXT_LENGTH"] = str(context_length)
    return context_length


def warm_model(alias: str) -> bool:
    payload = json.dumps(
        {
            "model": alias,
            "prompt": " ",
            "stream": False,
            "keep_alive": -1,
            "options": {"num_predict": 1},
        }
    ).encode("utf-8")
    request = Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    logging.info("Warming %s before exposing the API as ready", alias)
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            if response.status != 200:
                logging.error("Ollama warmup returned HTTP %s", response.status)
                return False
            response.read()
    except (OSError, TimeoutError) as exc:
        logging.error("Ollama warmup failed: %s", exc)
        return False
    logging.info("Ollama warmup completed for %s", alias)
    return True


def ensure_secondary_model() -> bool:
    url = os.getenv("SECONDARY_MODEL_URL")
    alias = os.getenv("SECONDARY_MODEL_NAME")
    if not url or not alias:
        return True
    base_alias = os.getenv("SECONDARY_BASE_MODEL_NAME", f"{alias}-base")
    context_length = int(os.environ["OLLAMA_CONTEXT_LENGTH"])

    if not command_succeeds(["ollama", "show", base_alias]) and command_succeeds(
        ["ollama", "show", alias]
    ):
        if not command_succeeds(["ollama", "cp", alias, base_alias]):
            logging.error("Could not migrate the cached secondary model")
            return False

    if not command_succeeds(["ollama", "show", base_alias]):
        model_dir = Path("/runpod-volume/imports")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "fable-fusion-711-q4-k-m.gguf"
        partial_path = model_path.with_suffix(".gguf.part")
        expected_size = int(os.environ["SECONDARY_MODEL_SIZE"])

        logging.info("Downloading the non-MTP Fable Fusion 711 Q4_K_M model")
        if command_succeeds(["aria2c", "--version"]):
            download_args = [
                "aria2c",
                "--continue=true",
                "--max-connection-per-server=16",
                "--split=16",
                "--min-split-size=16M",
                "--file-allocation=none",
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--max-tries=12",
                "--retry-wait=5",
                "--summary-interval=30",
                "--dir", str(partial_path.parent),
                "--out", partial_path.name,
                url,
            ]
        else:
            download_args = [
                "curl",
                "--fail",
                "--location",
                "--retry", "8",
                "--retry-all-errors",
                "--continue-at", "-",
                "--output", str(partial_path),
                url,
            ]
        download = subprocess.run(download_args)
        if (
            download.returncode != 0
            or not partial_path.exists()
            or partial_path.stat().st_size != expected_size
        ):
            logging.error("Secondary model download failed or has an unexpected size")
            return False
        partial_path.replace(model_path)

        base_modelfile = model_dir / "fable-fusion-711-base.Modelfile"
        base_modelfile.write_text(f"FROM {model_path}\n", encoding="ascii")
        create = subprocess.run(
            ["ollama", "create", base_alias, "--file", str(base_modelfile)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        if create.returncode != 0:
            logging.error("Secondary Ollama base model creation failed")
            return False
        model_path.unlink()
        base_modelfile.unlink()

    runtime_modelfile = Path("/tmp/fable-fusion-711-runtime.Modelfile")
    runtime_modelfile.write_text(
        f"FROM {base_alias}\nPARAMETER num_ctx {context_length}\n", encoding="ascii"
    )
    create = subprocess.run(
        ["ollama", "create", alias, "--file", str(runtime_modelfile)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    runtime_modelfile.unlink(missing_ok=True)
    if create.returncode != 0:
        logging.error("Secondary Ollama runtime model creation failed")
        return False

    logging.info("Secondary model is ready as %s with %s context", alias, context_length)
    return True


def initialize_model() -> None:
    model = os.environ["MODEL_NAME"]
    alias = os.environ["SERVED_MODEL_NAME"]
    deadline = time.monotonic() + STARTUP_TIMEOUT

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

    if not ensure_secondary_model():
        model_error.set()
        return

    if not warm_model(alias):
        model_error.set()
        return

    logging.info("Ollama model is ready as %s", alias)
    model_ready.set()


@web.middleware
async def authenticate(request: web.Request, handler):
    if request.path == "/healthz":
        return await handler(request)
    expected = f"Bearer {os.environ['POD_API_SECRET']}"
    if not hmac.compare_digest(request.headers.get("Authorization", ""), expected):
        raise web.HTTPUnauthorized()
    return await handler(request)


async def health(_request: web.Request) -> web.Response:
    if model_error.is_set():
        return web.json_response({"status": "error"}, status=500)
    status = "ready" if model_ready.is_set() else "initializing"
    return web.json_response({"status": status})


async def proxy_streaming_responses(
    request: web.Request, body: bytes, headers: dict[str, str]
) -> web.StreamResponse:
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        upstream_task = asyncio.ensure_future(
            session.request(
                request.method,
                f"{OLLAMA_BASE_URL}{request.rel_url}",
                data=body,
                headers=headers,
            )
        )
        upstream = None
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)
        await response.write(b": connected\n\n")

        try:
            while upstream is None:
                try:
                    upstream = await asyncio.wait_for(
                        asyncio.shield(upstream_task),
                        timeout=STREAM_HEARTBEAT_INTERVAL,
                    )
                except asyncio.TimeoutError:
                    await response.write(b": keep-alive\n\n")

            if upstream.status >= 400:
                message = (await upstream.read()).decode("utf-8", errors="replace")
                payload = json.dumps(
                    {
                        "error": {
                            "message": message or f"Upstream returned HTTP {upstream.status}",
                            "type": "upstream_error",
                            "code": str(upstream.status),
                        }
                    },
                    separators=(",", ":"),
                )
                await response.write(f"event: error\ndata: {payload}\n\n".encode("utf-8"))
            else:
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)

            await response.write_eof()
            return response
        finally:
            if upstream is not None:
                upstream.release()
            if not upstream_task.done():
                upstream_task.cancel()
                with suppress(asyncio.CancelledError):
                    await upstream_task


async def proxy(request: web.Request) -> web.StreamResponse:
    if not model_ready.is_set():
        return web.json_response({"error": "model is initializing"}, status=503)

    body = await request.read()
    if request.method == "POST" and request.path == "/v1/responses":
        body = normalize_fable_responses_request(body)
    headers = {"Content-Type": request.headers.get("Content-Type", "application/json")}
    if is_streaming_responses_request(request, body):
        return await proxy_streaming_responses(request, body, headers)

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request(
            request.method,
            f"{OLLAMA_BASE_URL}{request.rel_url}",
            data=body or None,
            headers=headers,
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status,
                headers={"Content-Type": upstream.headers.get("Content-Type", "application/json")},
            )
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response


def stop_process() -> None:
    if ollama_process is not None and ollama_process.poll() is None:
        ollama_process.terminate()


def main() -> None:
    global ollama_process
    for name in ("MODEL_NAME", "SERVED_MODEL_NAME", "POD_API_SECRET"):
        if not os.getenv(name):
            raise RuntimeError(f"{name} is required")

    configure_context_length()
    ollama_process = subprocess.Popen(["ollama", "serve"])
    threading.Thread(target=initialize_model, name="model-initializer", daemon=True).start()

    app = web.Application(middlewares=[authenticate])
    app.router.add_get("/healthz", health)
    app.router.add_route("*", "/{path:.*}", proxy)

    try:
        web.run_app(app, host="0.0.0.0", port=PROXY_PORT, access_log=None)
    finally:
        stop_process()


if __name__ == "__main__":
    main()
