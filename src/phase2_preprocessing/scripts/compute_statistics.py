"""
phase2_preprocessing/scripts/compute_statistics.py

Compute and print statistics for an already-built HF dataset.
Useful for QA before uploading, and for generating publication tables.

Usage:
    python -m src.phase2_preprocessing.scripts.compute_statistics \
        --dataset-dir data/hf_dataset
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compute Phase 2 dataset statistics")
    parser.add_argument("--dataset-dir", default="data/hf_dataset")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    dset_dir = Path(args.dataset_dir)
    if not dset_dir.exists():
        logger.error(f"Dataset not found: {dset_dir}")
        sys.exit(1)

    # Load pre-computed stats if available
    stats_path = dset_dir / "statistics.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if args.json:
            print(json.dumps(stats, indent=2))
            return
    else:
        stats = None

    dataset = load_from_disk(str(dset_dir))

    print("\n" + "=" * 60)
    print("PHASE 2 DATASET STATISTICS")
    print("=" * 60)
    for split_name, split in dataset.items():
        print(f"\n{split_name.upper()}: {len(split)} samples")
        if "language" in split.features:
            from collections import Counter
            langs = Counter(split["language"])
            for lang, count in langs.most_common():
                pct = 100 * count / max(len(split), 1)
                print(f"  {lang:<15} {count:>6}  ({pct:.1f}%)")

    if stats:
        total = stats.get("overall", {}).get("total_samples", "?")
        print(f"\nTotal samples: {total}")

    print("=" * 60)


if __name__ == "__main__":
    main()
