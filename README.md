# RunPod Huihui Qwen3.8 Worker

RunPod Serverless wrapper around the official CUDA Ollama runtime. It provides
OpenAI-compatible chat and Responses API routes for the Q4_K_M build of
`huihui_ai/Qwen3.8-abliterated`.

The default setup uses one GPU, 65,536 tokens of context, Q8 KV cache, and a
persistent Ollama cache at `/runpod-volume/ollama`.
