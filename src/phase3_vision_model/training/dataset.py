"""
phase3_vision_model/training/dataset.py

PyTorch Dataset for screenshot-code pairs.
Loads from the HF DatasetDict built in Phase 2.
"""
import base64
import io
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoFeatureExtractor

logger = logging.getLogger(__name__)


class ScreenshotCodeDataset(Dataset):
    """
    Loads screenshot-code pairs from a Phase 2 HuggingFace dataset.

    Each sample returns:
        pixel_values:   [3, 224, 224] float tensor
        input_ids:      [seq_len] long tensor  (prompt + code)
        attention_mask: [seq_len] long tensor
        labels:         [seq_len] long tensor  (-100 for prompt positions)
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        tokenizer: AutoTokenizer = None,
        feature_extractor: AutoFeatureExtractor = None,
        max_seq_length: int = 2048,
        image_key: str = "image",
    ):
        self.dataset = load_from_disk(str(dataset_dir))[split]
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_seq_length = max_seq_length
        self.image_key = image_key
        self.has_images = image_key in self.dataset.features

        logger.info(
            f"Dataset [{split}]: {len(self.dataset)} samples, "
            f"images={'yes' if self.has_images else 'no (load from disk)'}"
        )

    def __len__(self):
        return len(self.dataset)

    def _load_image(self, sample: dict[str, Any]) -> Image.Image:
        """Load image from base64 (dataset) or from cap_dir path on disk."""
        if self.has_images and sample.get(self.image_key):
            img_bytes = base64.b64decode(sample[self.image_key])
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Fallback: load first PNG from cap_dir
        cap_dir = Path(sample.get("cap_dir", ""))
        pngs = sorted(cap_dir.glob("*.png"))
        if pngs:
            return Image.open(pngs[0]).convert("RGB")

        # Last resort: blank image
        logger.warning(f"No image for sample {sample.get('file_hash')} — using blank")
        return Image.new("RGB", (224, 224), color=(30, 30, 30))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        messages = sample["messages"]  # [system, user, assistant]

        # Build prompt (system + user turns) and target (assistant turn)
        prompt_text = (
            f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        target_text = messages[2]["content"] + "<|im_end|>"

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        target_ids = self.tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        # Truncate to fit max_seq_length (keep end of target if too long)
        max_target = self.max_seq_length - len(prompt_ids)
        if max_target <= 0:
            prompt_ids = prompt_ids[: self.max_seq_length // 2]
            max_target = self.max_seq_length - len(prompt_ids)
        target_ids = target_ids[:max_target]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        # Image
        image = self._load_image(sample)
        pixel_values = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"][0]

        return {
            "pixel_values": pixel_values.to(torch.bfloat16),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
