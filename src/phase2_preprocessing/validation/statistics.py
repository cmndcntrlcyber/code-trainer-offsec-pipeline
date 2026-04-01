"""
phase2_preprocessing/validation/statistics.py

Compute and report dataset statistics for Phase 2 QA and the ReadyTensor
publication (tables, charts, language distribution).
"""
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_statistics(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """
    Compute statistics across train/validation/test splits.

    Returns a nested dict suitable for JSON serialisation and logging.
    """
    stats: dict[str, Any] = {"splits": {}, "overall": {}}

    all_records = []
    for split_name, records in splits.items():
        langs = Counter(r.get("language", "unknown") for r in records)
        themes = Counter(r.get("theme", "unknown") for r in records)
        line_counts = [r.get("line_count", 0) for r in records]
        code_lens = [len(r.get("source_code", "")) for r in records]
        prompts = Counter(r.get("prompt_idx", 0) for r in records)

        stats["splits"][split_name] = {
            "count": len(records),
            "languages": dict(langs.most_common()),
            "themes": dict(themes.most_common()),
            "line_count": {
                "min": min(line_counts, default=0),
                "max": max(line_counts, default=0),
                "mean": round(sum(line_counts) / max(len(line_counts), 1), 1),
            },
            "code_chars": {
                "min": min(code_lens, default=0),
                "max": max(code_lens, default=0),
                "mean": round(sum(code_lens) / max(len(code_lens), 1), 1),
            },
            "prompt_distribution": dict(sorted(prompts.items())),
        }
        all_records.extend(records)

    # Overall
    total = len(all_records)
    all_langs = Counter(r.get("language", "unknown") for r in all_records)
    stats["overall"] = {
        "total_samples": total,
        "languages": dict(all_langs.most_common()),
        "split_sizes": {k: len(v) for k, v in splits.items()},
    }

    return stats


def log_statistics(stats: dict[str, Any]) -> None:
    """Log dataset statistics to the console."""
    overall = stats.get("overall", {})
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total samples: {overall.get('total_samples', 0)}")

    for split, info in stats.get("splits", {}).items():
        logger.info(f"\n{split.upper()} split: {info['count']} samples")
        logger.info(f"  Languages: {info['languages']}")
        logger.info(f"  Line count: {info['line_count']}")
        logger.info(f"  Code chars: {info['code_chars']}")
        logger.info(f"  Prompt distribution: {info['prompt_distribution']}")


def save_statistics(stats: dict[str, Any], output_path: Path) -> None:
    """Save statistics JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2))
    logger.info(f"Statistics saved to {output_path}")
