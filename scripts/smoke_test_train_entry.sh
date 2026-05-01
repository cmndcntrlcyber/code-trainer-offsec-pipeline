#!/usr/bin/env bash
# Local smoke test for Phase 3 train_entry.py.
# Loads the v2-multimodal Hub revision, slices to --limit 50, runs 1 epoch with
# tiny batches on the local GPU (device_profile=5060ti → INT4) or CPU.
#
# Usage:
#   bash scripts/smoke_test_train_entry.sh            # default 5060ti profile
#   PROFILE=cpu bash scripts/smoke_test_train_entry.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a && source .env && set +a
fi

PROFILE="${PROFILE:-5060ti}"
LIMIT="${LIMIT:-50}"
OUT_DIR="${OUT_DIR:-/tmp/phase3-smoke}"
DATASET_DIR="${DATASET_DIR:-/tmp/phase3-smoke-dataset}"

# Tiny hyperparams for a ~2-3 minute smoke run.
export PHASE3_PARAMS_JSON=$(python3 - <<'PY'
import json
print(json.dumps({
    "vision_encoder": "microsoft/swin-base-patch4-window7-224",
    "decoder": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "batch_size": 1,
    "gradient_accumulation": 1,
    "learning_rate": 2e-4,
    "num_epochs": 1,
    "max_seq_length": 2048,
    "dataset_id": "cmndcntrlcyber/code-trainer-offsec-dataset",
    "dataset_revision": "v2-multimodal",
}))
PY
)
export WANDB_PROJECT="rtpi-phase3-smoke"
export WANDB_MODE="${WANDB_MODE:-offline}"   # no online W&B noise for a smoke test

# train_entry.py defaults HF_HOME to /workspace/.hf-cache for the A100 container.
# Override locally so the cache lands somewhere writable.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

echo "[smoke] profile=${PROFILE}  limit=${LIMIT}  out=${OUT_DIR}  HF_HOME=${HF_HOME}"
rm -rf "${OUT_DIR}" "${DATASET_DIR}"

uv run python -m src.phase3_vision_model.hf_skills.train_entry \
  --dataset-id cmndcntrlcyber/code-trainer-offsec-dataset \
  --dataset-revision v2-multimodal \
  --output-dir "${OUT_DIR}" \
  --local-dataset-dir "${DATASET_DIR}" \
  --limit "${LIMIT}" \
  --device-profile "${PROFILE}" \
  --skip-push

echo "[smoke] OK  — artifacts at ${OUT_DIR}"
