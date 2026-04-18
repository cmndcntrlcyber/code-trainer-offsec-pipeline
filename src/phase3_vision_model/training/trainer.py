"""
phase3_vision_model/training/trainer.py

RTX 5060 Ti optimized trainer for CodeVisionModel.

Config targets:
    Batch size:          2
    Gradient accum:      8  (effective batch: 16)
    Compute dtype:       BF16 (Blackwell tensor cores)
    Optimizer:           8-bit AdamW (bitsandbytes)
    Gradient checkpointing: enabled
    Sequence length:     2048 tokens
    Epochs:              10
"""
import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from ..architecture.vision_model import CodeVisionModel
from .callbacks import BestModelCallback, GPUMonitorCallback
from .collator import ScreenshotCodeCollator
from .dataset import ScreenshotCodeDataset

logger = logging.getLogger(__name__)

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


class VisionModelTrainer:
    """
    Trains CodeVisionModel on RTX 5060 Ti 16GB with BF16 + gradient checkpointing.
    Logs all metrics to W&B.
    """

    def __init__(
        self,
        model: CodeVisionModel,
        dataset_dir: str | Path,
        output_dir: str | Path,
        config: dict,
    ):
        self.model = model
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vm_cfg = config.get("vision_model", {})
        self.device_profile = vm_cfg.get("device_profile", "a100")
        self.batch_size = vm_cfg.get("batch_size", 8 if self.device_profile == "a100" else 2)
        self.grad_accum = vm_cfg.get("gradient_accumulation", 4 if self.device_profile == "a100" else 8)
        self.lr = float(vm_cfg.get("learning_rate", 2e-4))
        self.num_epochs = vm_cfg.get("num_epochs", 3 if self.device_profile == "a100" else 10)
        self.max_seq_length = vm_cfg.get("max_seq_length", 2048)
        self.use_8bit_optimizer = vm_cfg.get(
            "use_8bit_optimizer", self.device_profile == "5060ti"
        )
        captures_dir = config.get("data_collection", {}).get("captures_dir")
        self.captures_dir = Path(captures_dir) if captures_dir else None

        self.gpu_monitor = GPUMonitorCallback()
        self.best_model = BestModelCallback()

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        collator = ScreenshotCodeCollator(
            pad_token_id=self.model.tokenizer.pad_token_id or 0
        )
        train_ds = ScreenshotCodeDataset(
            self.dataset_dir, split="train",
            tokenizer=self.model.tokenizer,
            feature_extractor=self.model.vision_encoder.feature_extractor,
            max_seq_length=self.max_seq_length,
            captures_dir=self.captures_dir,
        )
        val_ds = ScreenshotCodeDataset(
            self.dataset_dir, split="validation",
            tokenizer=self.model.tokenizer,
            feature_extractor=self.model.vision_encoder.feature_extractor,
            max_seq_length=self.max_seq_length,
            captures_dir=self.captures_dir,
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            collate_fn=collator, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            collate_fn=collator, num_workers=2, pin_memory=True,
        )
        return train_loader, val_loader

    def _build_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_8bit_optimizer and HAS_BNB:
            logger.info("Using 8-bit AdamW optimizer (bitsandbytes)")
            return bnb.optim.AdamW8bit(trainable_params, lr=self.lr, weight_decay=0.01)
        logger.info("Using standard AdamW optimizer")
        return AdamW(trainable_params, lr=self.lr, weight_decay=0.01)

    def train(self, start_epoch: int = 0):
        import wandb

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)

        # Enable gradient checkpointing on decoder
        if hasattr(self.model.decoder, "enable_input_require_grads"):
            self.model.decoder.enable_input_require_grads()
        if hasattr(self.model.decoder, "gradient_checkpointing_enable"):
            self.model.decoder.gradient_checkpointing_enable()

        train_loader, val_loader = self._build_dataloaders()
        optimizer = self._build_optimizer()

        total_steps = len(train_loader) * self.num_epochs // self.grad_accum
        warmup_steps = total_steps // 10
        completed_steps = len(train_loader) * start_epoch // self.grad_accum
        # Required for last_epoch > 0: optimizer must have initial_lr set
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", self.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps, last_epoch=completed_steps - 1
        )

        wandb_project = (
            self.config.get("vision_model", {})
            .get("cloud", {})
            .get("wandb_project", "rtpi-phase3-vision")
        )
        wandb.init(
            project=wandb_project,
            resume="allow",
            config={
                "model": "Qwen2.5-Coder-1.5B + Swin-B",
                "device_profile": self.device_profile,
                "batch_size": self.batch_size,
                "grad_accum": self.grad_accum,
                "effective_batch": self.batch_size * self.grad_accum,
                "lr": self.lr,
                "epochs": self.num_epochs,
                "max_seq_length": self.max_seq_length,
                "start_epoch": start_epoch,
            }
        )

        scaler = torch.amp.GradScaler("cuda", enabled=False)  # BF16 doesn't need scaler
        global_step = completed_steps
        best_val_loss = float("inf")

        for epoch in range(start_epoch, self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.grad_accum

                loss.backward()
                epoch_loss += loss.item() * self.grad_accum

                if (step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 50 == 0:
                        avg_loss = epoch_loss / (step + 1)
                        lr_now = scheduler.get_last_lr()[0]
                        vram = torch.cuda.memory_allocated() / 1e9
                        logger.info(
                            f"Epoch {epoch+1} step {global_step}: "
                            f"loss={avg_loss:.4f} lr={lr_now:.2e} vram={vram:.1f}GB"
                        )
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr": lr_now,
                            "train/epoch": epoch + step / len(train_loader),
                            "gpu/vram_gb": vram,
                        }, step=global_step)

            # Validation
            val_loss = self._evaluate(val_loader, global_step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_pretrained(self.output_dir / "best")
                logger.info(f"Best model saved (val_loss={val_loss:.4f})")
                wandb.log({"val/best_loss": val_loss}, step=global_step)

            # Save epoch checkpoint
            self.model.save_pretrained(self.output_dir / f"epoch-{epoch+1}")

        wandb.finish()
        logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")

    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader, step: int) -> float:
        import wandb
        self.model.eval()
        total_loss = 0.0

        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(**batch)
            total_loss += outputs.loss.item()

        avg_loss = total_loss / max(len(val_loader), 1)
        vram_peak = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"Val loss: {avg_loss:.4f}  VRAM peak: {vram_peak:.1f} GB")
        wandb.log({"val/loss": avg_loss, "gpu/vram_peak_gb": vram_peak}, step=step)

        self.model.train()
        return avg_loss
