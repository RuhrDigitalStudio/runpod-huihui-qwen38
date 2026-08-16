# RunPod Huihui Qwen3.8 Worker

RunPod worker around the official CUDA Ollama runtime. It provides
OpenAI-compatible chat and Responses API routes for the Q4_K_M build of
`huihui_ai/Qwen3.8-abliterated` in Serverless and Pod modes.

The default setup uses one GPU, 262,144 tokens of context, Q4 KV cache, and a
persistent Ollama cache at `/runpod-volume/ollama`.

`pod_main.py` additionally exposes an authenticated API on port 8000 for a
manually created Pod. The Pod profile uses the native 262,144-token context
with a Q4 KV cache and is intended to be created only for an active Codex
session, then stopped while retaining its Pod volume.
