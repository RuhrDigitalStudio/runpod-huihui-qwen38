"""Proxy RunPod Serverless jobs to the local llama.cpp OpenAI server."""

import logging
import os
from typing import Any, AsyncGenerator, Optional, Tuple

import aiohttp

LLAMA_PORT = os.getenv("LLAMA_PORT", "8000")
LLAMA_BASE_URL = f"http://127.0.0.1:{LLAMA_PORT}"
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "3600"))

llama_process = None


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


async def handler(job: dict) -> AsyncGenerator[Any, None]:
    try:
        route, method, body = _normalize_job_input(job.get("input") or {})
    except ValueError as exc:
        yield _error(str(exc))
        return

    if llama_process is not None and llama_process.poll() is not None:
        yield _error("llama-server is not running")
        return

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method,
                f"{LLAMA_BASE_URL}{route}",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status >= 400:
                    detail = await response.text()
                    logging.error("llama-server HTTP %s: %s", response.status, detail)
                    yield _error(f"llama-server returned HTTP {response.status}: {detail}")
                    return

                if isinstance(body, dict) and body.get("stream") is True:
                    async for chunk in response.content.iter_any():
                        yield chunk.decode("utf-8", errors="replace")
                else:
                    yield await response.json(content_type=None)
    except aiohttp.ClientError as exc:
        logging.exception("Request to llama-server failed")
        yield _error(f"Request to llama-server failed: {exc}")
