"""
phase4_qwen_finetuning/hf_skills/job_monitor.py

Live job monitoring — prints status table for all active Phase 4 jobs.
"""
import json
import logging
import sys
import time
from pathlib import Path

from .job_client import HFSkillsClient

logger = logging.getLogger(__name__)


class JobMonitor:
    """Monitor multiple HF Skills jobs and display live status."""

    def __init__(self, client: HFSkillsClient, state_file: Path = Path("data/phase4_jobs.json")):
        self.client = client
        self.state_file = state_file

    def load_jobs(self) -> dict[str, str]:
        """Load {name: job_id} map from state file."""
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {}

    def save_jobs(self, jobs: dict[str, str]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(jobs, indent=2))

    def monitor(self, poll_interval: int = 60) -> None:
        """Continuously print job status until all complete or fail."""
        jobs = self.load_jobs()
        if not jobs:
            logger.warning("No jobs in state file")
            return

        active = dict(jobs)
        while active:
            print(f"\n{'─'*60}")
            print(f"{'Job Name':<30} {'Status':<15} {'Job ID'}")
            print(f"{'─'*60}")

            completed = []
            for name, job_id in active.items():
                try:
                    status = self.client.get_status(job_id)
                    print(f"{name:<30} {status.status:<15} {job_id}")
                    if status.status in ("completed", "success", "failed", "error", "cancelled"):
                        completed.append(name)
                except Exception as e:
                    print(f"{name:<30} {'error':<15} {e}")

            for name in completed:
                del active[name]

            if active:
                print(f"\nPolling again in {poll_interval}s... (Ctrl+C to stop)")
                time.sleep(poll_interval)

        print("\nAll jobs finished.")
