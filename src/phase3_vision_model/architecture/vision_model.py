"""
phase3_vision_model/architecture/vision_model.py

Full CodeVisionModel: VisionEncoder → MultimodalProjector → CodeDecoder.

Architecture:
    1. VisionEncoder (Swin-B, frozen) → patch features [B, N_patches, 1024]
    2. MultimodalProjector (MLP) → projected features [B, N_patches, 2048]
    3. Prepend projected visual tokens to text embeddings
    4. CodeDecoder (Qwen-1.5B + LoRA) → output logits

Total VRAM budget on RTX 5060 Ti 16GB:
    Swin-B (frozen BF16):      ~1.5 GB
    Projector:                 ~0.3 GB
    Qwen-1.5B (INT4 + LoRA):  ~1.2 GB
    Activations + optimizer:  ~10.0 GB (gradient checkpointing active)
    ─────────────────────────────────
    Total peak:               ~13.0 GB  (3 GB headroom)
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn

from .code_decoder import DECODER_MODEL_ID, load_decoder_with_lora
from .multimodal_projector import MultimodalProjector
from .vision_encoder import SWIN_MODEL_ID, VisionEncoder

logger = logging.getLogger(__name__)


class CodeVisionModel(nn.Module):
    """Multimodal code generation model: screenshot → source code."""

    def __init__(
        self,
        vision_model_id: str = SWIN_MODEL_ID,
        decoder_model_id: str = DECODER_MODEL_ID,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        device: str = "cuda",
        device_profile: str = "a100",
    ):
        super().__init__()

        # 1. Vision encoder (frozen)
        self.vision_encoder = VisionEncoder(model_id=vision_model_id, device=device)
        vision_hidden = self.vision_encoder.hidden_size

        # 2. Decoder + tokenizer
        self.decoder, self.tokenizer = load_decoder_with_lora(
            model_id=decoder_model_id,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device_map=device,
            device_profile=device_profile,
        )
        text_hidden = self.decoder.config.hidden_size  # 2048 for Qwen-1.5B

        # 3. Projector (trainable)
        self.projector = MultimodalProjector(
            vision_hidden_size=vision_hidden,
            text_hidden_size=text_hidden,
        ).to(device=device, dtype=torch.bfloat16)

        logger.info(
            f"CodeVisionModel ready — "
            f"projector: {vision_hidden}→{text_hidden}, "
            f"decoder: {decoder_model_id}"
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass: encode image, prepend visual tokens, run decoder.

        Args:
            pixel_values:    [B, 3, 224, 224]  — preprocessed screenshots
            input_ids:       [B, seq_len]        — tokenised text (prompt + code)
            attention_mask:  [B, seq_len]
            labels:          [B, seq_len] with -100 for visual prefix positions

        Returns:
            CausalLMOutput with .loss and .logits
        """
        # Encode image → visual tokens
        vision_features = self.vision_encoder(pixel_values)        # [B, N, 1024]
        visual_tokens = self.projector(vision_features)            # [B, N, 2048]

        # Get text embeddings from decoder
        text_embeds = self.decoder.model.model.embed_tokens(input_ids)   # [B, T, 2048]

        # Concatenate: [visual | text]
        combined = torch.cat([visual_tokens, text_embeds], dim=1)  # [B, N+T, 2048]

        # Extend attention mask for visual prefix
        B, N = visual_tokens.shape[:2]
        visual_mask = torch.ones(B, N, device=attention_mask.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

        # Extend labels: ignore visual prefix tokens
        if labels is not None:
            visual_labels = torch.full((B, N), -100, device=labels.device, dtype=labels.dtype)
            combined_labels = torch.cat([visual_labels, labels], dim=1)
        else:
            combined_labels = None

        return self.decoder(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

    def save_pretrained(self, output_dir: str | Path):
        """Save LoRA adapters + projector weights."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.decoder.save_pretrained(str(output_dir / "decoder_lora"))
        torch.save(
            self.projector.state_dict(),
            output_dir / "projector.pt",
        )
        logger.info(f"Model saved to {output_dir}")

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path, **kwargs) -> "CodeVisionModel":
        """Load from a save_pretrained checkpoint."""
        import os
        from peft import PeftModel

        checkpoint_dir = Path(checkpoint_dir)
        model = cls(**kwargs)

        # Load projector
        projector_path = checkpoint_dir / "projector.pt"
        if projector_path.exists():
            model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
            logger.info("Projector weights loaded")

        # Load LoRA adapters
        lora_path = checkpoint_dir / "decoder_lora"
        if lora_path.exists():
            model.decoder = PeftModel.from_pretrained(model.decoder.base_model.model, str(lora_path))
            logger.info("LoRA adapters loaded")

        return model
