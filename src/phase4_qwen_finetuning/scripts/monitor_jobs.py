"""
phase4_qwen_finetuning/scripts/monitor_jobs.py

Monitor Phase 4 job status. Can poll continuously or show a one-time snapshot.

Usage:
    # Continuous monitoring (polls every 60s):
    python -m src.phase4_qwen_finetuning.scripts.monitor_jobs --watch

    # One-time status snapshot:
    python -m src.phase4_qwen_finetuning.scripts.monitor_jobs
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.phase4_qwen_finetuning.hf_skills.job_client import HFSkillsClient
from src.phase4_qwen_finetuning.hf_skills.job_monitor import JobMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Monitor Phase 4 HF Skills jobs")
    parser.add_argument("--jobs-file", default="data/phase4_jobs.json")
    parser.add_argument("--watch", action="store_true", help="Continuous polling")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN not set")
        sys.exit(1)

    client = HFSkillsClient(token=hf_token)
    monitor = JobMonitor(client, state_file=Path(args.jobs_file))

    if args.watch:
        monitor.monitor(poll_interval=args.interval)
    else:
        jobs = monitor.load_jobs()
        if not jobs:
            print("No jobs found in state file")
            return
        for name, job_id in jobs.items():
            status = client.get_status(job_id)
            print(f"{name}: {status.status} ({job_id})")


if __name__ == "__main__":
    main()
