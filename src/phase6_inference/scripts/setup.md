# Phase 6 — Inference & Agent Stack: Setup

Code-only deployment instructions. **None of the steps below run by default in
the project's `uv sync`** because vLLM (built from source for sm_120 Blackwell)
and Node.js MCP servers don't belong in the training-pipeline dependency tree.
Treat Phase 6 as a separate environment that lives on the inference host.

## Prereqs (RTX 5060 Ti box)

- NVIDIA driver ≥ R570 (Blackwell)
- CUDA 12.8 toolkit (runtime + nvcc; needed to build vLLM)
- Python 3.12 (matches `requires-python` in `pyproject.toml`)
- Node.js ≥ 20 (for `npx`-launched MCP servers)
- `uvx` (for `mcp-server-fetch`)

## 1. Build vLLM nightly with sm_120 support

```bash
# Fresh venv, separate from the training one
python3.12 -m venv ~/venv-vllm
source ~/venv-vllm/bin/activate
pip install -U pip wheel

# vLLM source build for Blackwell (sm_120)
git clone https://github.com/vllm-project/vllm
cd vllm
TORCH_CUDA_ARCH_LIST="12.0" pip install -e . --no-build-isolation
```

Build is heavy (~30 min on a fast machine). The `TORCH_CUDA_ARCH_LIST=12.0`
ensures kernels are compiled for sm_120; without it, vLLM falls back to JIT
compilation on first use and may fail on Blackwell.

## 2. Install Qwen-Agent + MCP CLI runners

```bash
pip install qwen-agent
# MCP server runners
npm install -g @modelcontextprotocol/server-filesystem
pip install uv  # provides uvx
uvx mcp-server-fetch --help   # warm cache
```

## 3. Pull Qwen3.5-9B + Qwen2.5-Coder-14B GGUF weights

The architecture doc recommends Q6_K for the primary and Q4_K_M for the coder
hot-swap target. vLLM serves HF safetensors directly (preferred); use GGUF
variants only if running through llama.cpp instead.

```bash
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ~/models/Qwen3.5-9B
huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct \
  --local-dir ~/models/Qwen2.5-Coder-14B-Instruct
```

## 4. Smoke test

```bash
# Terminal A — start the server
bash src/phase6_inference/server/vllm_serve.sh

# Terminal B — once `curl localhost:8000/health` returns 200:
python -m src.phase6_inference.agent.qwen_agent_client \
  --query "List the files in /workspace and summarize what's there."
```

## 5. Hot-swap test

```bash
# Swap to the coder model
python -m src.phase6_inference.scripts.hot_swap --model Qwen/Qwen2.5-Coder-14B-Instruct
# ... use it ...
# Swap back
python -m src.phase6_inference.scripts.hot_swap --model Qwen/Qwen3.5-9B
```

Expected swap latency: 3-10s on NVMe.

## What's NOT in scope here

- Production process management (use systemd / pm2 / supervisord on the actual
  inference host).
- Multi-GPU / tensor parallelism — single 5060 Ti per the architecture doc.
- Concurrent dual-model serving — explicitly traded against hot-swap in the doc
  (line 131).
- Authentication on the vLLM endpoint — assumed bound to localhost only.
