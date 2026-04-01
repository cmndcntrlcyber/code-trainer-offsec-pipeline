"""
phase4_qwen_finetuning/hf_skills/job_client.py

HuggingFace Skills API client for launching and monitoring cloud training jobs.
Uses the huggingface_hub library to submit jobs to A100-large hardware.

Note: HF Skills (formerly AutoTrain) API may vary — this client targets the
      current endpoint as of early 2026. Check HF docs if endpoint changes.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

HF_SKILLS_BASE_URL = "https://huggingface.co/api/autotrain"


@dataclass
class JobSpec:
    """Specification for a HF Skills training job."""
    name: str
    model_id: str
    dataset_id: str
    hardware: str                       # e.g. "a100-large"
    script_path: str                    # path to training script in repo
    hyperparams: dict[str, Any] = field(default_factory=dict)
    num_epochs: int = 1
    env_vars: dict[str, str] = field(default_factory=dict)


@dataclass
class JobStatus:
    job_id: str
    name: str
    status: str                         # queued, running, completed, failed
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    logs_url: str = ""
    eval_loss: float | None = None


class HFSkillsClient:
    """
    Client for HuggingFace Skills (AutoTrain) cloud training API.

    Usage:
        client = HFSkillsClient(token="hf_...")
        job_id = client.launch_job(spec)
        status = client.wait_for_completion(job_id)
    """

    def __init__(self, token: str):
        self.token = token
        self.api = HfApi(token=token)
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def launch_job(self, spec: JobSpec) -> str:
        """
        Launch a training job on HF Skills.

        Returns:
            job_id string
        """
        payload = {
            "model": spec.model_id,
            "task": "text-generation",
            "backend": spec.hardware,
            "training_type": "lora",
            "data": {
                "path": spec.dataset_id,
                "train_split": "train",
                "valid_split": "validation",
                "col_mapping": {
                    "text": "messages",
                },
            },
            "params": {
                "epochs": spec.num_epochs,
                **spec.hyperparams,
            },
            "username": self.api.whoami()["name"],
            "name": spec.name,
        }

        logger.info(f"Launching job: {spec.name} on {spec.hardware}")
        resp = requests.post(
            f"{HF_SKILLS_BASE_URL}/create",
            headers=self.headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("id") or data.get("job_id")
        logger.info(f"Job launched: {job_id}")
        return str(job_id)

    def get_status(self, job_id: str) -> JobStatus:
        """Poll status of a running job."""
        resp = requests.get(
            f"{HF_SKILLS_BASE_URL}/status/{job_id}",
            headers=self.headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        return JobStatus(
            job_id=job_id,
            name=data.get("name", ""),
            status=data.get("status", "unknown"),
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at", ""),
            logs_url=data.get("logs_url", ""),
        )

    def get_logs(self, job_id: str) -> str:
        """Fetch training logs for a job."""
        resp = requests.get(
            f"{HF_SKILLS_BASE_URL}/logs/{job_id}",
            headers=self.headers,
            timeout=30,
        )
        if resp.ok:
            return resp.text
        return f"[logs unavailable: {resp.status_code}]"

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 60,
        timeout: int = 14400,  # 4 hours
    ) -> JobStatus:
        """
        Block until a job completes (or fails), polling every poll_interval seconds.

        Returns:
            Final JobStatus
        """
        start = time.time()
        while True:
            status = self.get_status(job_id)
            logger.info(f"Job {job_id} status: {status.status}")

            if status.status in ("completed", "success"):
                logger.info(f"Job {job_id} completed successfully")
                return status

            if status.status in ("failed", "error", "cancelled"):
                logger.error(f"Job {job_id} failed: {status.status}")
                logger.error(self.get_logs(job_id))
                return status

            elapsed = time.time() - start
            if elapsed > timeout:
                logger.error(f"Job {job_id} timed out after {elapsed:.0f}s")
                return status

            time.sleep(poll_interval)

    def cancel_job(self, job_id: str) -> None:
        """Cancel a running job."""
        resp = requests.post(
            f"{HF_SKILLS_BASE_URL}/cancel/{job_id}",
            headers=self.headers,
            timeout=30,
        )
        if resp.ok:
            logger.info(f"Job {job_id} cancelled")
        else:
            logger.warning(f"Cancel request failed: {resp.status_code}")
