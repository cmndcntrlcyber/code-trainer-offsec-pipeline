"""
phase2_preprocessing/converters/chat_formatter.py

Formats screenshot-code pairs into Qwen chat template messages.
Cycles through 7 diverse user prompt variations for training diversity.
"""
import itertools
from typing import Any


# 7 diverse prompt variations for data diversity
PROMPT_VARIATIONS = [
    "What code is displayed in this VS Code screenshot?",
    "Please transcribe the source code shown in this editor screenshot.",
    "Extract and reproduce the code visible in this screenshot.",
    "What is the exact code shown in this VS Code window?",
    "Reproduce the code from this screenshot.",
    "What source code is visible in this editor?",
    "Transcribe the code from this VS Code screenshot.",
]

SYSTEM_PROMPT = (
    "You are a code transcription assistant. When given a screenshot of code in a "
    "VS Code editor, you reproduce the exact source code shown, preserving all "
    "formatting, indentation, and syntax."
)

_prompt_cycle = itertools.cycle(PROMPT_VARIATIONS)


def format_sample(source_code: str, prompt_idx: int | None = None) -> dict[str, Any]:
    """
    Format a screenshot-code pair as a Qwen chat message dict.

    Args:
        source_code: Raw source code text from source.txt
        prompt_idx: Fixed prompt index (0-6). If None, cycles through all 7.

    Returns:
        dict with 'messages' key containing system/user/assistant turns.
        The user turn contains an image placeholder; actual image bytes are
        attached separately by the dataset builder.
    """
    if prompt_idx is not None:
        prompt = PROMPT_VARIATIONS[prompt_idx % len(PROMPT_VARIATIONS)]
    else:
        prompt = next(_prompt_cycle)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": source_code},
        ]
    }


def format_batch(samples: list[dict]) -> list[dict]:
    """
    Format a list of raw samples (each with 'source_code' and optional 'prompt_idx').
    Returns list of formatted dicts with 'messages'.
    """
    return [
        format_sample(s["source_code"], s.get("prompt_idx"))
        for s in samples
    ]
