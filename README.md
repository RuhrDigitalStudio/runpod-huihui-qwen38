# RunPod Huihui Qwen3.8 Worker

RunPod Serverless wrapper around the official CUDA `llama.cpp` server. It
provides OpenAI-compatible chat and Responses API routes for the Q4 GGUF build
of `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`.

The default setup uses one GPU, 65,536 tokens of context, Q8 KV cache, and a
persistent Hugging Face cache at `/runpod-volume/huggingface-cache`.
