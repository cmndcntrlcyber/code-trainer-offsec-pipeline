"""
phase4_qwen_finetuning/scripts/launch_validation_sweep.py

Launch Phase 4A: 3 parallel validation sweep jobs on HF Skills A100-large.

Usage:
    python -m src.phase4_qwen_finetuning.scripts.launch_validation_sweep \
        --config src/config/v6_config.yaml

    # Dry run (print config, don't launch):
    python -m src.phase4_qwen_finetuning.scripts.launch_validation_sweep \
        --config src/config/v6_config.yaml --dry-run
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.settings import load_config
from src.phase4_qwen_finetuning.configs.sweep_configs import SWEEP_CONFIGS
from src.phase4_qwen_finetuning.hf_skills.job_client import HFSkillsClient
from src.phase4_qwen_finetuning.hf_skills.job_monitor import JobMonitor
from src.phase4_qwen_finetuning.hf_skills.sweep_orchestrator import SweepOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 4A: Launch validation sweep")
    parser.add_argument("--config", default="src/config/v6_config.yaml")
    parser.add_argument("--results-dir", default="data/sweep_results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print job specs without launching")
    args = parser.parse_args()

    config = load_config(args.config)
    qf_cfg = config.get("qwen_finetuning", {})
    pp_cfg = config.get("preprocessing", {})

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set")
        sys.exit(1)

    dataset_id = pp_cfg.get("dataset_name", "")
    model_id = qf_cfg.get("model", "Qwen/Qwen2.5-Coder-14B-Instruct")
    hardware = qf_cfg.get("hardware", "a100-large")

    if args.dry_run:
        print("\n=== DRY RUN — Sweep Configs ===")
        for cfg in SWEEP_CONFIGS:
            print(f"\n{cfg.name}:")
            print(f"  LoRA r={cfg.lora_r} alpha={cfg.lora_alpha}")
            print(f"  lr={cfg.learning_rate}  bs={cfg.batch_size}  accum={cfg.gradient_accumulation}")
            print(f"  effective_batch={cfg.effective_batch}")
        print(f"\nModel:    {model_id}")
        print(f"Dataset:  {dataset_id}")
        print(f"Hardware: {hardware}")
        return

    client = HFSkillsClient(token=hf_token)
    monitor = JobMonitor(client)
    orch = SweepOrchestrator(
        client=client,
        dataset_id=dataset_id,
        model_id=model_id,
        hardware=hardware,
        results_dir=Path(args.results_dir),
    )

    results = orch.run_sweep()

    # Save job IDs for monitoring
    job_ids = {r["config_name"]: r.get("job_id", "") for r in results}
    monitor.save_jobs(job_ids)

    best = results[0] if results else None
    if best:
        logger.info(f"\nBest config: {best['config_name']} (eval_loss={best.get('eval_loss')})")
        logger.info(f"Proceed to launch_full_training.py with --best-config {best['config_name']}")


if __name__ == "__main__":
    main()
