"""
src/scripts/render_publication_charts.py

Generate the static charts embedded in the Ready Tensor publication. Outputs
land in `docs/ReadyTensor Submission/assets/` as PNGs.

Charts:
1. phase4a_sweep.png — Phase 4A LoRA-config sweep eval_loss bar chart
   (source: docs/sweep/phase4a-summary.json).
2. phase3_metrics.png — Phase 3 baseline-vs-fine-tuned metric panel
   (source: docs/eval/phase3-summary.json).
3. gsm8k_forgetting.png — GSM8K catastrophic-forgetting check
   (source: phase4-benchmark-gsm8k{,-base}.json on the adapter repo, or local
   copies if the user dropped them under docs/eval/).

Usage:
    # matplotlib is intentionally not in pyproject.toml — it's a publication-
    # only dependency. Pull it in transiently with `uv run --with matplotlib`:
    uv run --with matplotlib --with numpy \
        python -m src.scripts.render_publication_charts
    uv run --with matplotlib --with numpy \
        python -m src.scripts.render_publication_charts --skip gsm8k
"""
import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "ReadyTensor Submission" / "assets"
SWEEP_JSON = ROOT / "docs" / "sweep" / "phase4a-summary.json"
PHASE3_JSON = ROOT / "docs" / "eval" / "phase3-summary.json"
GSM8K_FT_JSON = ROOT / "docs" / "eval" / "phase4-benchmark-gsm8k.json"
GSM8K_BASE_JSON = ROOT / "docs" / "eval" / "phase4-benchmark-gsm8k-base.json"


def render_phase4a_sweep():
    import matplotlib.pyplot as plt

    data = json.loads(SWEEP_JSON.read_text())
    rows = sorted(data["rows"], key=lambda r: r["result"]["eval_loss"])
    names = [r["name"] for r in rows]
    losses = [r["result"]["eval_loss"] for r in rows]
    colors = ["#2a9d8f" if i == 0 else "#264653" for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, losses, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("eval_loss (full validation, 3,265 rows)")
    ax.set_title("Phase 4A LoRA sweep — Qwen2.5-Coder-14B-Instruct")
    ax.set_ylim(min(losses) - 0.01, max(losses) + 0.005)
    fig.tight_layout()
    out = ASSETS / "phase4a_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out.relative_to(ROOT))


def render_phase3_metrics():
    import matplotlib.pyplot as plt
    import numpy as np

    if not PHASE3_JSON.exists():
        logger.warning("Skipping phase3 chart — %s missing", PHASE3_JSON)
        return
    data = json.loads(PHASE3_JSON.read_text())
    metrics = ["mean_edit_similarity", "syntax_valid_rate"]
    base = [data["baseline"][m] for m in metrics]
    ft = [data["finetuned"][m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, base, w, label="Base (Qwen2.5-Coder-1.5B)",
           color="#264653", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, ft, w, label="Fine-tuned (this work)",
           color="#e76f51", edgecolor="black", linewidth=0.5)
    for i, (b, f) in enumerate(zip(base, ft)):
        ax.text(i - w / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, max(ft) * 1.2 + 0.05)
    ax.set_ylabel("score")
    ax.set_title("Phase 3 vision model — base vs fine-tuned (test split, n=200)")
    ax.legend()
    fig.tight_layout()
    out = ASSETS / "phase3_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out.relative_to(ROOT))


def _gsm8k_score(payload: dict) -> float | None:
    """Pick the flexible-extract exact_match (regex over chat output). The
    strict-match score is 0 for both base and adapter because the chat-trained
    model emits prose ("The answer is 42") rather than GSM8K's #### 42 raw
    format — so it would compare two zeros, hiding all signal."""
    results = payload.get("results", {})
    # Prefer flexible-extract; fall back to any exact_match-like key.
    flex = results.get("exact_match,flexible-extract")
    if isinstance(flex, (int, float)):
        return float(flex)
    for k, v in results.items():
        if k.startswith("exact_match") and "stderr" not in k \
                and isinstance(v, (int, float)):
            return float(v)
    return None


def render_gsm8k_forgetting():
    import matplotlib.pyplot as plt

    if not (GSM8K_FT_JSON.exists() and GSM8K_BASE_JSON.exists()):
        logger.warning(
            "Skipping GSM8K chart — drop phase4-benchmark-gsm8k{,-base}.json "
            "into docs/eval/ first (download from the adapter repo).")
        return
    base = json.loads(GSM8K_BASE_JSON.read_text())
    ft = json.loads(GSM8K_FT_JSON.read_text())
    base_score = _gsm8k_score(base)
    ft_score = _gsm8k_score(ft)
    if base_score is None or ft_score is None:
        logger.warning("Skipping GSM8K chart — could not extract exact_match")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Base", "+ adapter"], [base_score, ft_score],
        color=["#264653", "#e76f51"], edgecolor="black", linewidth=0.5,
    )
    for bar, val in zip(bars, [base_score, ft_score]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("GSM8K exact_match (0-shot)")
    ax.set_ylim(0, max(base_score, ft_score) * 1.25 + 0.05)
    delta = (ft_score - base_score) / max(base_score, 1e-6) * 100
    ax.set_title(
        "Catastrophic-forgetting check — GSM8K 0-shot "
        f"({delta:+.1f} % relative)")
    fig.tight_layout()
    out = ASSETS / "gsm8k_forgetting.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Render Ready Tensor publication charts.")
    parser.add_argument("--skip", action="append", default=[],
                        choices=["phase4a", "phase3", "gsm8k"],
                        help="Skip a specific chart (repeatable).")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    if "phase4a" not in args.skip:
        render_phase4a_sweep()
    if "phase3" not in args.skip:
        render_phase3_metrics()
    if "gsm8k" not in args.skip:
        render_gsm8k_forgetting()


if __name__ == "__main__":
    main()
