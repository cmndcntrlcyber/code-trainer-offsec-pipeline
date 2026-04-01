"""
phase3_vision_model/training/callbacks.py

Training callbacks for:
- GPU VRAM monitoring (logged to W&B)
- Early stopping on validation loss plateau
- Checkpoint saving best model
"""
import logging

import torch

logger = logging.getLogger(__name__)


class GPUMonitorCallback:
    """Log GPU VRAM usage to W&B at each eval step."""

    def __init__(self, device: int = 0):
        self.device = device

    def on_evaluate(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9

        metrics = {
            "gpu/vram_allocated_gb": round(allocated, 2),
            "gpu/vram_reserved_gb": round(reserved, 2),
            "gpu/vram_peak_gb": round(max_allocated, 2),
        }
        logger.info(f"VRAM: {allocated:.1f}/{reserved:.1f} GB (peak {max_allocated:.1f} GB)")

        try:
            import wandb
            if wandb.run:
                wandb.log(metrics, step=state.global_step)
        except ImportError:
            pass


class BestModelCallback:
    """Track best validation loss and save best checkpoint path."""

    def __init__(self):
        self.best_eval_loss = float("inf")
        self.best_checkpoint = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss", float("inf"))
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_checkpoint = f"checkpoint-{state.global_step}"
            logger.info(f"New best eval_loss: {eval_loss:.4f} at step {state.global_step}")
