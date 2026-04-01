"""
phase3_vision_model/evaluation/metrics.py

Code generation metrics for Capstone baseline + post-fine-tuning evaluation.

Metrics:
    - exact_match:      Exact string equality after normalization
    - token_accuracy:   Token-level accuracy (ignoring padding)
    - bleu_4:           BLEU-4 score (sacrebleu)
    - edit_distance:    Normalized Levenshtein distance
    - syntax_valid:     AST parse success rate (Python/JS only)
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def normalize_code(code: str) -> str:
    """Normalize code for comparison: strip trailing whitespace, normalize newlines."""
    lines = [line.rstrip() for line in code.strip().splitlines()]
    return "\n".join(lines)


def exact_match(predictions: list[str], references: list[str]) -> float:
    """Fraction of predictions that exactly match their reference."""
    if not predictions:
        return 0.0
    matches = sum(
        normalize_code(p) == normalize_code(r)
        for p, r in zip(predictions, references)
    )
    return matches / len(predictions)


def bleu_4(predictions: list[str], references: list[str]) -> float:
    """BLEU-4 score using sacrebleu (character-level tokenization for code)."""
    try:
        from sacrebleu.metrics import BLEU
        metric = BLEU(tokenize="char")
        result = metric.corpus_score(predictions, [references])
        return result.score / 100.0  # normalize to [0, 1]
    except ImportError:
        logger.warning("sacrebleu not installed; BLEU-4 skipped")
        return 0.0


def normalized_edit_distance(pred: str, ref: str) -> float:
    """Normalized Levenshtein distance (0 = identical, 1 = completely different)."""
    p, r = normalize_code(pred), normalize_code(ref)
    if not p and not r:
        return 0.0
    if not p or not r:
        return 1.0

    # Dynamic programming
    m, n = len(p), len(r)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if p[i - 1] == r[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n] / max(m, n)


def mean_edit_similarity(predictions: list[str], references: list[str]) -> float:
    """Mean edit similarity (1 - normalized edit distance)."""
    if not predictions:
        return 0.0
    scores = [
        1.0 - normalized_edit_distance(p, r)
        for p, r in zip(predictions, references)
    ]
    return sum(scores) / len(scores)


def syntax_valid_rate(predictions: list[str], language: str = "python") -> float:
    """Fraction of predictions that parse without syntax errors."""
    if language != "python":
        return None  # Only Python AST checking supported

    import ast
    valid = 0
    for pred in predictions:
        try:
            ast.parse(pred)
            valid += 1
        except SyntaxError:
            pass
    return valid / max(len(predictions), 1)


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    language: str = "python",
) -> dict[str, Any]:
    """Compute all available metrics for a prediction/reference pair list."""
    metrics = {
        "exact_match": exact_match(predictions, references),
        "bleu_4": bleu_4(predictions, references),
        "mean_edit_similarity": mean_edit_similarity(predictions, references),
        "num_samples": len(predictions),
    }
    syntax = syntax_valid_rate(predictions, language)
    if syntax is not None:
        metrics["syntax_valid_rate"] = syntax
    return metrics
