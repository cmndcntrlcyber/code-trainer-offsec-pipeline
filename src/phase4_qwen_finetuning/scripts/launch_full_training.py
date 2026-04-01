"""
phase4_qwen_finetuning/scripts/launch_full_training.py

Launch Phase 4B: Full 3-epoch training with top-2 sweep configs on A100-large.

Usage:
    # Auto-select top-2 from sweep results:
    python -m src.phase4_qwen_finetuning.scripts.launch_full_training \
        --config src/config/v6_config.yaml \
        --sweep-results data/sweep_results/sweep_summary.json

    # Specify configs manually:
    python -m src.phase4_qwen_finetuning.scripts.launch_full_training \
        --config src/config/v6_config.yaml \
        --configs conservative standard
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.settings import load_config
from src.phase4_qwen_finetuning.configs.sweep_configs import SWEEP_CONFIG_MAP
from src.phase4_qwen_finetuning.hf_skills.job_client import HFSkillsClient, JobSpec
from src.phase4_qwen_finetuning.hf_skills.job_monitor import JobMonitor
from src.phase4_qwen_finetuning.hf_skills.sweep_orchestrator import SweepOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 4B: Full training")
    parser.add_argument("--config", default="src/config/v6_config.yaml")
    parser.add_argument("--sweep-results", default="data/sweep_results/sweep_summary.json",
                        help="Path to sweep_summary.json from Phase 4A")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Config names to train (overrides sweep results)")
    parser.add_argument("--results-dir", default="data/full_training_results")
    args = parser.parse_args()

    config = load_config(args.config)
    qf_cfg = config.get("qwen_finetuning", {})
    pp_cfg = config.get("preprocessing", {})

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN not set")
        sys.exit(1)

    # Determine which configs to train
    if args.configs:
        selected_names = args.configs
    else:
        sweep_path = Path(args.sweep_results)
        if not sweep_path.exists():
            logger.error(f"Sweep results not found: {sweep_path}")
            logger.error("Run launch_validation_sweep.py first")
            sys.exit(1)
        summary = json.loads(sweep_path.read_text())
        ranked = summary.get("sweep_results", [])
        top_n = qf_cfg.get("full_training", {}).get("top_n_configs", 2)
        selected_names = [r["config_name"] for r in ranked[:top_n]]

    logger.info(f"Full training configs: {selected_names}")
    selected_configs = [SWEEP_CONFIG_MAP[n] for n in selected_names if n in SWEEP_CONFIG_MAP]

    num_epochs = qf_cfg.get("full_training", {}).get("num_epochs", 3)
    dataset_id = pp_cfg.get("dataset_name", "")
    model_id = qf_cfg.get("model", "Qwen/Qwen2.5-Coder-14B-Instruct")
    hardware = qf_cfg.get("hardware", "a100-large")
    output_base = qf_cfg.get("output_base", "combatcougar/qwen14b-code-trainer-v6")

    client = HFSkillsClient(token=hf_token)
    monitor = JobMonitor(client, state_file=Path("data/phase4b_jobs.json"))
    orch = SweepOrchestrator(
        client=client,
        dataset_id=dataset_id,
        model_id=model_id,
        hardware=hardware,
        results_dir=Path(args.results_dir),
    )

    # Override configs for full training (num_epochs=3)
    results = orch.run_sweep(configs=selected_configs)
    monitor.save_jobs({r["config_name"]: r.get("job_id", "") for r in results})

    best = results[0] if results else None
    if best:
        logger.info(f"\nBest full-training config: {best['config_name']}")
        logger.info(f"Model should be pushed to: {output_base}-{best['config_name']}")
        logger.info("Next step: run convert_to_gguf.py (Phase 5)")


if __name__ == "__main__":
    main()
