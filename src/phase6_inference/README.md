# Phase 6 — Inference & Agent Stack

Local-GPU runtime for the RTPI project. Serves Qwen3.5-9B as the primary
model with Qwen-Agent + MCP for tool calling, and supports a hot-swap to
Qwen2.5-Coder-14B-Instruct for compiled-language code generation.

## Layout

```
src/phase6_inference/
├── server/
│   └── vllm_serve.sh                — Launch vLLM with the documented flags
├── agent/
│   ├── qwen_agent_client.py         — Qwen-Agent wired to local vLLM + MCP
│   └── mcp_servers.json             — MCP server definitions (filesystem, fetch)
└── scripts/
    ├── hot_swap.py                  — Stop/start vLLM with a different model
    └── setup.md                     — One-time install on the inference host
```

## Quick start

See `scripts/setup.md` for one-time install. After that:

```bash
# Terminal A — primary model
bash src/phase6_inference/server/vllm_serve.sh

# Terminal B — agent client
python -m src.phase6_inference.agent.qwen_agent_client \
  --query "What files are in /workspace?"
```

Hot-swap to the coder specialist:

```bash
python -m src.phase6_inference.scripts.hot_swap \
  --model Qwen/Qwen2.5-Coder-14B-Instruct
```

## Why this isn't in `pyproject.toml`

vLLM (sm_120 source build), Qwen-Agent, and Node-based MCP servers belong on
the inference host, not in the training pipeline's dependency graph. Phase 6
is intentionally an isolated runtime — the training environment should not
need vLLM to install, and the inference host should not need bitsandbytes /
trl / datasets to serve traffic.

## See also

- `docs/plan/Inference-Agent-Architecture.md` — full design rationale, model
  selection, throughput estimates.
- `docs/plan/ROADMAP.md` — Phase 6 status & success criteria.
