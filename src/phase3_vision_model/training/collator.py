"""
phase3_vision_model/training/collator.py

Data collator that pads variable-length sequences in a batch.
Handles pixel_values (fixed size), input_ids, attention_mask, and labels.
"""
import torch
from torch.nn.utils.rnn import pad_sequence


class ScreenshotCodeCollator:
    """
    Pads input_ids, attention_mask, and labels to the longest sequence in the batch.
    pixel_values are already uniform size [3, 224, 224] so they're just stacked.
    """

    def __init__(self, pad_token_id: int = 0, label_pad_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([s["pixel_values"] for s in samples])

        input_ids = pad_sequence(
            [s["input_ids"] for s in samples],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [s["attention_mask"] for s in samples],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [s["labels"] for s in samples],
            batch_first=True,
            padding_value=self.label_pad_id,
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
