"""Phase 6 — Inference & Agent Stack.

Local single-GPU runtime: vLLM nightly serves Qwen3.5-9B (Q6_K) on RTX 5060 Ti
as the primary model. Qwen-Agent provides the MCP tool-calling bridge.
Hot-swap to Qwen2.5-Coder-14B (Q4_K_M) is supported for compiled-language
code generation. See docs/plan/Inference-Agent-Architecture.md for the
full design rationale.
"""
