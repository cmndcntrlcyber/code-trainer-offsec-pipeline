"""
phase4_qwen_finetuning/scripts/generate_report.py

Generate a Phase 4 training report from sweep and full-training results.
Outputs a Markdown summary for the ReadyTensor publication and GitHub README.

Usage:
    python -m src.phase4_qwen_finetuning.scripts.generate_report \
        --sweep-results data/sweep_results/sweep_summary.json \
        --full-results data/full_training_results \
        --output data/phase4_report.md
"""
import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 4 training report")
    parser.add_argument("--sweep-results", default="data/sweep_results/sweep_summary.json")
    parser.add_argument("--full-results", default="data/full_training_results")
    parser.add_argument("--output", default="data/phase4_report.md")
    args = parser.parse_args()

    lines = ["# Phase 4: Qwen2.5-Coder-14B Fine-tuning Report\n"]

    # Sweep results
    sweep_path = Path(args.sweep_results)
    if sweep_path.exists():
        summary = json.loads(sweep_path.read_text())
        results = summary.get("sweep_results", [])
        lines.append("## Validation Sweep (Phase 4A)\n")
        lines.append("| Rank | Config | LoRA r | lr | Batch | Eval Loss |")
        lines.append("|---|---|---|---|---|---|")
        for i, r in enumerate(results):
            cfg = r.get("config", {})
            lines.append(
                f"| {i+1} | {r['config_name']} | {cfg.get('lora_r')} | "
                f"{cfg.get('learning_rate')} | {cfg.get('effective_batch')} | "
                f"{r.get('eval_loss', 'N/A')} |"
            )
        best = summary.get("best", {})
        if best:
            lines.append(f"\n**Best config:** `{best.get('config_name')}` "
                        f"(eval_loss={best.get('eval_loss', 'N/A')})\n")

    # Full training results
    full_dir = Path(args.full_results)
    if full_dir.exists():
        lines.append("## Full Training (Phase 4B)\n")
        for result_file in sorted(full_dir.glob("*.json")):
            data = json.loads(result_file.read_text())
            lines.append(f"### {data.get('config_name', result_file.stem)}")
            lines.append(f"- Status: {data.get('status')}")
            lines.append(f"- Eval loss: {data.get('eval_loss', 'N/A')}")
            if data.get("logs_url"):
                lines.append(f"- Logs: {data['logs_url']}")
            lines.append("")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
