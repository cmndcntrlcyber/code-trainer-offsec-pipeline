"""
phase3_vision_model/evaluation/evaluator.py

Evaluation pipeline for baseline (pre-fine-tune) and post-fine-tune assessment.
Satisfies Capstone Requirements #2 and #4.

Usage — baseline evaluation (before training):
    python -m src.phase3_vision_model.scripts.evaluate --baseline

Usage — post-fine-tuning:
    python -m src.phase3_vision_model.scripts.evaluate \
        --checkpoint models/vision_model/best
"""
import json
import logging
from pathlib import Path
from typing import Any

import torch

from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class VisionModelEvaluator:
    """
    Generates code from screenshots and computes evaluation metrics.
    Can evaluate the base model (no LoRA) or a fine-tuned checkpoint.
    """

    def __init__(
        self,
        model,
        tokenizer,
        feature_extractor,
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @torch.no_grad()
    def generate_predictions(
        self,
        dataloader,
        num_samples: int | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Generate code predictions for a dataloader.

        Returns:
            (predictions, references, languages)
        """
        self.model.eval()
        predictions, references, languages = [], [], []

        for i, batch in enumerate(dataloader):
            if num_samples and len(predictions) >= num_samples:
                break

            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Encode image → visual tokens
            vision_features = self.model.vision_encoder(pixel_values)
            visual_tokens = self.model.projector(vision_features)

            # Get text embeddings for prompt only (no labels in input for generation)
            text_embeds = self.model.decoder.model.embed_tokens(input_ids)
            combined = torch.cat([visual_tokens, text_embeds], dim=1)

            B, N = visual_tokens.shape[:2]
            visual_mask = torch.ones(B, N, device=attention_mask.device, dtype=attention_mask.dtype)
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

            output_ids = self.model.decoder.generate(
                inputs_embeds=combined,
                attention_mask=combined_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            )

            for b in range(output_ids.shape[0]):
                pred_ids = output_ids[b][combined.shape[1]:]  # strip input prefix
                pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                predictions.append(pred_text)

            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {len(predictions)} predictions...")

        return predictions, references, languages

    def evaluate_from_dataloader(
        self,
        dataloader,
        language: str = "python",
        num_samples: int = 200,
        run_name: str = "eval",
    ) -> dict[str, Any]:
        """
        Run full evaluation: generate + compute metrics.

        Args:
            dataloader: Val/test DataLoader
            language: Primary language for syntax checking
            num_samples: Number of samples to evaluate (subset for speed)
            run_name: Label for W&B logging ('baseline' or 'finetuned')

        Returns:
            metrics dict
        """
        logger.info(f"Evaluating {run_name} on {num_samples} samples...")

        # Collect references from dataloader
        refs, langs = [], []
        for batch in dataloader:
            for idx in range(batch["input_ids"].shape[0]):
                label_ids = batch["labels"][idx]
                valid = label_ids[label_ids != -100]
                refs.append(self.tokenizer.decode(valid, skip_special_tokens=True))
                if len(refs) >= num_samples:
                    break
            if len(refs) >= num_samples:
                break

        preds, _, _ = self.generate_predictions(dataloader, num_samples=num_samples)
        preds = preds[: len(refs)]

        metrics = compute_all_metrics(preds, refs, language=language)
        metrics["run_name"] = run_name

        logger.info(f"Results [{run_name}]:")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")

        try:
            import wandb
            if wandb.run:
                wandb.log({f"{run_name}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        except ImportError:
            pass

        return metrics

    def save_results(self, metrics: dict, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"Evaluation results saved to {output_path}")
