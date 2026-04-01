"""
phase2_preprocessing/validation/quality_filter.py

Filters low-quality samples before dataset upload.
Quality checks: code length, line count, language diversity, deduplication.
"""
import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Minimum characters of source code to be considered a useful training sample
MIN_CODE_CHARS = 50
# Minimum lines (from metadata)
MIN_LINE_COUNT = 5
# Languages to accept; empty set = accept all
ACCEPTED_LANGUAGES: set[str] = set()


def compute_code_hash(source_code: str) -> str:
    """SHA256 of normalized source code for deduplication."""
    normalized = " ".join(source_code.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def filter_records(
    records: list[dict[str, Any]],
    min_code_chars: int = MIN_CODE_CHARS,
    min_line_count: int = MIN_LINE_COUNT,
    deduplicate: bool = True,
    accepted_languages: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Filter records for quality and optionally deduplicate.

    Args:
        records: List of raw dataset records from hf_dataset_converter
        min_code_chars: Drop samples shorter than this
        min_line_count: Drop samples with fewer lines than this
        deduplicate: Remove exact duplicate source code (by normalized hash)
        accepted_languages: If non-empty, drop records not in this set

    Returns:
        (filtered_records, stats_dict) where stats_dict counts each drop reason
    """
    stats = {
        "total_in": len(records),
        "too_short": 0,
        "too_few_lines": 0,
        "wrong_language": 0,
        "duplicate": 0,
        "passed": 0,
    }

    accepted = accepted_languages or ACCEPTED_LANGUAGES
    seen_hashes: set[str] = set()
    filtered = []

    for rec in records:
        code = rec.get("source_code", "")
        lang = rec.get("language", "")
        lines = rec.get("line_count", 0)

        if len(code) < min_code_chars:
            stats["too_short"] += 1
            continue

        if lines < min_line_count:
            stats["too_few_lines"] += 1
            continue

        if accepted and lang not in accepted:
            stats["wrong_language"] += 1
            continue

        if deduplicate:
            h = compute_code_hash(code)
            if h in seen_hashes:
                stats["duplicate"] += 1
                continue
            seen_hashes.add(h)

        filtered.append(rec)
        stats["passed"] += 1

    logger.info(
        f"Quality filter: {stats['total_in']} in → {stats['passed']} passed "
        f"(short={stats['too_short']}, few_lines={stats['too_few_lines']}, "
        f"lang={stats['wrong_language']}, dup={stats['duplicate']})"
    )
    return filtered, stats
