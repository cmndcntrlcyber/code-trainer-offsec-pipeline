"""
phase3_vision_model/architecture/multimodal_projector.py

2-layer MLP projector: maps Swin-B patch features → Qwen decoder embedding space.

Input:  [batch, num_patches, vision_hidden_size]  (1024-dim from Swin-B)
Output: [batch, num_patches, text_hidden_size]    (2048-dim for Qwen-1.5B)

Params: ~4M  |  BF16: ~0.3 GB VRAM
"""
import torch
import torch.nn as nn


class MultimodalProjector(nn.Module):
    """
    2-layer MLP that projects vision patch features into text embedding space.
    Uses GELU activation and LayerNorm for training stability.
    """

    def __init__(self, vision_hidden_size: int = 1024, text_hidden_size: int = 2048):
        super().__init__()
        intermediate_size = (vision_hidden_size + text_hidden_size) // 2

        self.proj = nn.Sequential(
            nn.Linear(vision_hidden_size, intermediate_size, bias=True),
            nn.GELU(),
            nn.LayerNorm(intermediate_size),
            nn.Linear(intermediate_size, text_hidden_size, bias=True),
            nn.LayerNorm(text_hidden_size),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [batch, num_patches, vision_hidden_size]

        Returns:
            projected: [batch, num_patches, text_hidden_size]
        """
        return self.proj(vision_features)
