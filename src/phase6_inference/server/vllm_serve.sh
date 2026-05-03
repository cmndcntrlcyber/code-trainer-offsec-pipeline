#!/usr/bin/env bash
# Launch vLLM serving Qwen3.5-9B (or a hot-swap target) on the local RTX 5060 Ti.
#
# Prereq: vLLM nightly built from source with sm_120 (Blackwell) compute capability.
# See src/phase6_inference/scripts/setup.md for the build procedure.
#
# Usage:
#   bash src/phase6_inference/server/vllm_serve.sh                    # primary
#   MODEL=Qwen/Qwen2.5-Coder-14B-Instruct \
#     bash src/phase6_inference/server/vllm_serve.sh                  # hot-swap
#
# Stop with Ctrl-C; vLLM exits gracefully.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

# --language-model-only skips the vision tower (~1 GB VRAM saving). Drop it
# when website-debugging mode is needed and rely on per-request model swap.
#
# When pairing with Qwen-Agent, leave --enable-auto-tool-choice OFF and let
# Qwen-Agent parse tool calls itself (per the architecture doc, line 117).
# Set ENABLE_AUTO_TOOLS=1 to flip it on for direct-API use.
EXTRA_FLAGS=()
if [[ "${ENABLE_AUTO_TOOLS:-0}" == "1" ]]; then
  EXTRA_FLAGS+=("--enable-auto-tool-choice" "--tool-call-parser" "qwen3_coder")
fi

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --reasoning-parser qwen3 \
  --language-model-only \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  "${EXTRA_FLAGS[@]}"
