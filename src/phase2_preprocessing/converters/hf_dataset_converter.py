"""
phase2_preprocessing/converters/hf_dataset_converter.py

Converts Phase 1 capture directories into a HuggingFace Dataset object
with Qwen chat format and base64 WebP images.

Output schema per sample:
    {
        "file_hash":     str   — unique ID (16-char sha256 prefix)
        "language":      str   — programming language
        "line_count":    int
        "theme":         str   — Monaco theme used for capture
        "image":         str   — base64 WebP of representative frame
        "messages":      list  — [system, user, assistant] Qwen chat turns
        "source_code":   str   — raw source (truncated to max_code_length)
        "prompt_idx":    int   — which of 7 user prompt variations was used
    }
"""
import json
import logging
import random
from pathlib import Path
from typing import Any, Iterator

from .chat_formatter import PROMPT_VARIATIONS, format_sample
from .image_encoder import encode_capture_dir

logger = logging.getLogger(__name__)


def _iter_capture_dirs(captures_dir: Path) -> Iterator[Path]:
    """Yield all valid capture hash directories under captures_dir."""
    for prefix_dir in sorted(captures_dir.iterdir()):
        if not prefix_dir.is_dir():
            continue
        for cap_dir in sorted(prefix_dir.iterdir()):
            if cap_dir.is_dir() and (cap_dir / "metadata.json").exists():
                yield cap_dir


def _load_capture(cap_dir: Path, max_code_length: int) -> dict[str, Any] | None:
    """
    Load and validate a single capture directory into a raw dict.
    Returns None if the capture is incomplete or unreadable.
    """
    metadata_path = cap_dir / "metadata.json"
    source_path = cap_dir / "source.txt"

    try:
        metadata = json.loads(metadata_path.read_text())
        source_code = source_path.read_text(errors="replace")
    except Exception as e:
        logger.debug(f"Skipping {cap_dir}: {e}")
        return None

    if not source_code.strip():
        return None

    # Truncate to max_code_length characters
    if len(source_code) > max_code_length:
        source_code = source_code[:max_code_length]

    return {
        "cap_dir": cap_dir,
        "file_hash": cap_dir.name,
        "language": metadata.get("language", "unknown"),
        "line_count": metadata.get("line_count", 0),
        "theme": metadata.get("theme", ""),
        "source_code": source_code,
        "metadata": metadata,
    }


def convert_captures_to_records(
    captures_dir: Path,
    max_code_length: int = 8192,
    seed: int = 42,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """
    Convert all Phase 1 capture directories to dataset records.

    Each record gets a deterministic prompt_idx based on its position
    in the sorted list, cycling through all 7 prompt variations.

    Args:
        captures_dir: Root captures directory (data/sample-data/captures)
        max_code_length: Max characters for source_code field
        seed: Random seed for reproducible shuffle before split
        show_progress: Log progress every 1000 samples

    Returns:
        List of dataset record dicts (without images yet — images encoded
        lazily during HF dataset build to avoid OOM on large datasets).
    """
    records = []
    skipped = 0
    num_prompts = len(PROMPT_VARIATIONS)

    cap_dirs = list(_iter_capture_dirs(captures_dir))
    logger.info(f"Found {len(cap_dirs)} capture directories")

    for i, cap_dir in enumerate(cap_dirs):
        raw = _load_capture(cap_dir, max_code_length)
        if raw is None:
            skipped += 1
            continue

        prompt_idx = i % num_prompts
        formatted = format_sample(raw["source_code"], prompt_idx=prompt_idx)

        records.append({
            "file_hash": raw["file_hash"],
            "language": raw["language"],
            "line_count": raw["line_count"],
            "theme": raw["theme"],
            "cap_dir": str(raw["cap_dir"]),  # path; image encoded at build time
            "source_code": raw["source_code"],
            "prompt_idx": prompt_idx,
            "messages": formatted["messages"],
        })

        if show_progress and (i + 1) % 1000 == 0:
            logger.info(f"  Loaded {i + 1}/{len(cap_dirs)} captures ({skipped} skipped)")

    logger.info(f"Loaded {len(records)} records ({skipped} skipped)")

    # Shuffle deterministically before splitting
    rng = random.Random(seed)
    rng.shuffle(records)
    return records


def split_records(
    records: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[dict]]:
    """Split records into train/val/test sets."""
    n = len(records)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": records[:train_end],
        "validation": records[train_end:val_end],
        "test": records[val_end:],
    }
