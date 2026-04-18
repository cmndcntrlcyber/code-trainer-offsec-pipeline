"""
phase3_vision_model/architecture/vision_encoder.py

Swin-B vision encoder wrapper (frozen during training).
Extracts patch embeddings from VS Code screenshot images.

Model: microsoft/swin-base-patch4-window7-224
Params: ~88M  |  BF16 frozen: ~1.5 GB VRAM
Output: [batch, 49, 1024] patch features (7×7 spatial grid, 1024-dim)
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, SwinModel

logger = logging.getLogger(__name__)

SWIN_MODEL_ID = "microsoft/swin-base-patch4-window7-224"


class VisionEncoder(nn.Module):
    """Frozen Swin-B encoder that maps screenshot images to patch features."""

    def __init__(self, model_id: str = SWIN_MODEL_ID, device: str = "cuda"):
        super().__init__()
        logger.info(f"Loading vision encoder: {model_id}")
        self.encoder = SwinModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
        )
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_id)
        self.hidden_size = self.encoder.config.hidden_size  # 1024 for Swin-B

        # Freeze all parameters — encoder is a fixed feature extractor
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.to(device=device)
        self.encoder.eval()
        logger.info(
            f"Vision encoder loaded: hidden_size={self.hidden_size}, "
            f"frozen, {sum(p.numel() for p in self.encoder.parameters())/1e6:.1f}M params"
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch, 3, 224, 224] BF16 image tensors

        Returns:
            patch_features: [batch, num_patches, hidden_size] BF16
        """
        with torch.no_grad():
            outputs = self.encoder(pixel_values=pixel_values)
        # last_hidden_state: [batch, seq_len, hidden_size]
        return outputs.last_hidden_state

    def preprocess_images(self, images: list) -> torch.Tensor:
        """
        Preprocess a list of PIL images for the encoder.

        Args:
            images: List of PIL Image objects

        Returns:
            pixel_values tensor [batch, 3, 224, 224]
        """
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(torch.bfloat16)
