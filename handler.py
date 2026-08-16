"""Proxy RunPod Serverless jobs to the local Ollama OpenAI server."""

import asyncio
import logging
import os
import threading
from typing import Any, AsyncGenerator, Optional, Tuple

import aiohttp

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "3600"))
STARTUP_TIMEOUT = float(os.getenv("STARTUP_TIMEOUT", "1800"))

ollama_process = None
model_ready = threading.Event()
model_error = threading.Event()


def _normalize_job_input(job_input: dict) -> Tuple[str, str, Optional[dict]]:
    if job_input.get("openai_input"):
        return (
            job_input.get("openai_route") or "/v1/chat/completions",
            "POST",
            job_input["openai_input"],
        )
    if job_input.get("openai_route"):
        return job_input["openai_route"], "GET", None
    if job_input.get("route"):
        body = job_input.get("body")
        method = (job_input.get("method") or ("POST" if body else "GET")).upper()
        return job_input["route"], method, body
    raise ValueError("Job input must contain openai_input/openai_route or route/body")


def _error(message: str) -> dict:
    return {"error": {"message": message, "type": "worker_error", "code": None}}


async def _wait_until_ready() -> bool:
    deadline = asyncio.get_running_loop().time() + STARTUP_TIMEOUT
    while asyncio.get_running_loop().time() < deadline:
        if model_ready.is_set():
            return True
        if model_error.is_set():
            return False
        if ollama_process is not None and ollama_process.poll() is not None:
            return False
        await asyncio.sleep(2)
    return False


async def handler(job: dict) -> AsyncGenerator[Any, None]:
    try:
        route, method, body = _normalize_job_input(job.get("input") or {})
    except ValueError as exc:
        yield _error(str(exc))
        return

    if not await _wait_until_ready():
        yield _error("Ollama model did not become ready")
        return

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method,
                f"{OLLAMA_BASE_URL}{route}",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status >= 400:
                    detail = await response.text()
                    logging.error("Ollama HTTP %s: %s", response.status, detail)
                    yield _error(f"Ollama returned HTTP {response.status}: {detail}")
                    return

                if isinstance(body, dict) and body.get("stream") is True:
                    async for chunk in response.content.iter_any():
                        yield chunk.decode("utf-8", errors="replace")
                else:
                    yield await response.json(content_type=None)
    except aiohttp.ClientError as exc:
        logging.exception("Request to Ollama failed")
        yield _error(f"Request to Ollama failed: {exc}")
