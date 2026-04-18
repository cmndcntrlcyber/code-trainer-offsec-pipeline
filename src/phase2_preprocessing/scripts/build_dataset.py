"""
phase2_preprocessing/scripts/build_dataset.py

Full Phase 2 preprocessing pipeline:
  1. Scan Phase 1 capture directories
  2. Load source code + metadata
  3. Apply quality filtering & deduplication
  4. Format into Qwen chat messages
  5. Encode screenshots to base64 WebP
  6. Split into train/validation/test
  7. Build HuggingFace Dataset and save to disk

Usage:
    python -m src.phase2_preprocessing.scripts.build_dataset \
        --config src/config/v6_config.yaml \
        --captures-dir data/sample-data/captures \
        --output-dir data/hf_dataset

    # Skip image encoding (text-only dataset, faster):
    python -m src.phase2_preprocessing.scripts.build_dataset \
        --config src/config/v6_config.yaml --no-images
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datasets import Dataset, DatasetDict, Features, Sequence, Value

from src.config.settings import load_config
from src.phase2_preprocessing.converters.hf_dataset_converter import (
    convert_captures_to_records,
    split_records,
)
from src.phase2_preprocessing.converters.image_encoder import encode_capture_dir
from src.phase2_preprocessing.validation.quality_filter import filter_records
from src.phase2_preprocessing.validation.statistics import (
    compute_statistics,
    log_statistics,
    save_statistics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def encode_images_in_records(
    records: list[dict],
    show_progress: bool = True,
    max_dim: int = 1024,
) -> list[dict]:
    """Encode screenshots for all records in-place. Drops records whose image fails."""
    out = []
    for i, rec in enumerate(records):
        cap_dir = Path(rec["cap_dir"])
        try:
            rec["image"] = encode_capture_dir(cap_dir, max_dim=max_dim)
            out.append(rec)
        except Exception as e:
            logger.warning(f"Skipping {cap_dir.name}: image encode failed: {e}")
        if show_progress and (i + 1) % 500 == 0:
            logger.info(f"  Encoded {i + 1}/{len(records)} images")
    return out


def records_to_hf_dataset(records: list[dict], include_images: bool) -> Dataset:
    """Convert a list of record dicts to a HuggingFace Dataset."""
    # Build columns
    data: dict[str, list] = {
        "file_hash": [],
        "language": [],
        "line_count": [],
        "theme": [],
        "source_code": [],
        "prompt_idx": [],
        "messages": [],
    }
    if include_images:
        data["image"] = []

    for rec in records:
        data["file_hash"].append(rec["file_hash"])
        data["language"].append(rec["language"])
        data["line_count"].append(rec["line_count"])
        data["theme"].append(rec["theme"])
        data["source_code"].append(rec["source_code"])
        data["prompt_idx"].append(rec["prompt_idx"])
        data["messages"].append(rec["messages"])
        if include_images:
            data["image"].append(rec.get("image", ""))

    return Dataset.from_dict(data)


def main():
    parser = argparse.ArgumentParser(description="Build Phase 2 HuggingFace dataset")
    parser.add_argument("--config", default="src/config/v6_config.yaml")
    parser.add_argument("--captures-dir", default=None,
                        help="Override captures directory from config")
    parser.add_argument("--output-dir", default="data/hf_dataset",
                        help="Directory to save the HF dataset")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip image encoding (faster, text-only dataset)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of captures (for testing)")
    parser.add_argument("--image-max-dim", type=int, default=None,
                        help="Max pixel dimension for WebP images "
                             "(default: preprocessing.image_max_dim in config, or 1024)")
    args = parser.parse_args()

    config = load_config(args.config)
    pp_cfg = config.get("preprocessing", {})

    captures_dir = Path(args.captures_dir or config["data_collection"]["captures_dir"])
    output_dir = Path(args.output_dir)
    max_code_length = pp_cfg.get("max_code_length", 8192)
    train_ratio = pp_cfg.get("train_split", 0.8)
    val_ratio = pp_cfg.get("val_split", 0.1)
    image_max_dim = args.image_max_dim or pp_cfg.get("image_max_dim", 1024)

    logger.info(f"Phase 2: Building dataset from {captures_dir}")
    logger.info(f"Output: {output_dir}  Images: {not args.no_images}  "
                f"Image max_dim: {image_max_dim}")

    # Step 1: Load records
    records = convert_captures_to_records(captures_dir, max_code_length=max_code_length)

    if args.limit:
        records = records[: args.limit]
        logger.info(f"Limited to {len(records)} records")

    # Step 2: Quality filter
    records, filter_stats = filter_records(records)

    # Step 3: Encode images
    if not args.no_images:
        logger.info(f"Encoding screenshots to base64 WebP (max_dim={image_max_dim})...")
        records = encode_images_in_records(records, max_dim=image_max_dim)

    # Step 4: Split
    splits = split_records(records, train_ratio=train_ratio, val_ratio=val_ratio)
    logger.info(
        f"Split: train={len(splits['train'])} val={len(splits['validation'])} "
        f"test={len(splits['test'])}"
    )

    # Step 5: Build HF DatasetDict
    dataset_dict = DatasetDict({
        split: records_to_hf_dataset(recs, include_images=not args.no_images)
        for split, recs in splits.items()
    })

    # Step 6: Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    logger.info(f"Dataset saved to {output_dir}")

    # Step 7: Statistics
    stats = compute_statistics(splits)
    log_statistics(stats)
    save_statistics(stats, output_dir / "statistics.json")

    logger.info("Phase 2 build complete.")


if __name__ == "__main__":
    main()
