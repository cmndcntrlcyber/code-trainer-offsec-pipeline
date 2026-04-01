"""
phase2_preprocessing/scripts/upload_to_hub.py

Upload the locally-built HuggingFace dataset to HF Hub.

Usage:
    python -m src.phase2_preprocessing.scripts.upload_to_hub \
        --config src/config/v6_config.yaml \
        --dataset-dir data/hf_dataset

    # Upload as public repo:
    python -m src.phase2_preprocessing.scripts.upload_to_hub \
        --config src/config/v6_config.yaml --public
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datasets import load_from_disk
from huggingface_hub import HfApi

from src.config.settings import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Upload Phase 2 dataset to HF Hub")
    parser.add_argument("--config", default="src/config/v6_config.yaml")
    parser.add_argument("--dataset-dir", default="data/hf_dataset",
                        help="Path to the locally-saved HF dataset")
    parser.add_argument("--public", action="store_true",
                        help="Upload as public repo (default: private)")
    args = parser.parse_args()

    config = load_config(args.config)
    pp_cfg = config.get("preprocessing", {})
    dataset_name = pp_cfg.get("dataset_name")
    private = not args.public and pp_cfg.get("private", True)

    if not dataset_name:
        logger.error("preprocessing.dataset_name not set in config")
        sys.exit(1)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        logger.error("Run build_dataset.py first.")
        sys.exit(1)

    logger.info(f"Loading dataset from {dataset_dir}")
    dataset = load_from_disk(str(dataset_dir))

    logger.info(f"Uploading to HF Hub: {dataset_name} (private={private})")
    dataset.push_to_hub(
        dataset_name,
        private=private,
        commit_message="Phase 2: Upload code-trainer-v6 screenshot dataset",
    )

    logger.info(f"Dataset uploaded: https://huggingface.co/datasets/{dataset_name}")

    # Upload statistics.json as a dataset card artifact
    stats_path = dataset_dir / "statistics.json"
    if stats_path.exists():
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(stats_path),
            path_in_repo="statistics.json",
            repo_id=dataset_name,
            repo_type="dataset",
            commit_message="Add dataset statistics",
        )
        logger.info("statistics.json uploaded")


if __name__ == "__main__":
    main()
