"""
phase3_vision_model/configs/a100_config.py

Hyperparameter configuration for the Phase 3 A100 cloud training job.

Single config (no sweep for v1) — if results are insufficient we add a sweep
module mirroring phase4_qwen_finetuning/configs/sweep_configs.py.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class A100VisionConfig:
    """Vision-model training hyperparameters for A100-large (40GB)."""
    name: str = "a100-standard"

    # Model
    vision_encoder: str = "microsoft/swin-base-patch4-window7-224"
    decoder: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    device_profile: str = "a100"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Optimization
    batch_size: int = 8
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_seq_length: int = 2048
    use_8bit_optimizer: bool = False
    gradient_checkpointing: bool = True

    # Dataset
    dataset_id: str = "cmndcntrlcyber/code-trainer-offsec-dataset"
    dataset_revision: str = "v2-multimodal"

    def to_hyperparams(self) -> dict[str, Any]:
        """Flat dict passed to HF Skills JobSpec.hyperparams."""
        return {
            "vision_encoder": self.vision_encoder,
            "decoder": self.decoder,
            "device_profile": self.device_profile,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "max_seq_length": self.max_seq_length,
            "use_8bit_optimizer": self.use_8bit_optimizer,
            "gradient_checkpointing": self.gradient_checkpointing,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
        }


DEFAULT_A100_CONFIG = A100VisionConfig()
