"""
phase3_vision_model/architecture/code_decoder.py

Qwen2.5-Coder-1.5B-Instruct decoder with INT4 quantization + LoRA (r=16).

Config:
    Base model: Qwen/Qwen2.5-Coder-1.5B-Instruct
    Quantization: 4-bit NF4 (bitsandbytes)
    LoRA: r=16, alpha=32, target=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
    VRAM: ~1.0 GB (INT4 base) + ~0.2 GB (LoRA adapters)
"""
import logging

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen2ForCausalLM

logger = logging.getLogger(__name__)

DECODER_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_decoder_with_lora(
    model_id: str = DECODER_MODEL_ID,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    device_map: str = "cuda",
    device_profile: str = "a100",
) -> tuple:
    """
    Load Qwen2.5-Coder-1.5B-Instruct with LoRA.

    device_profile:
        "a100"    — full BF16 weights (A100 40GB has VRAM; faster than INT4 dequant).
        "5060ti"  — 4-bit NF4 quantization via bitsandbytes (16GB VRAM budget).

    Returns:
        (model, tokenizer)
    """
    use_4bit = device_profile == "5060ti"
    logger.info(
        f"Loading decoder: {model_id} "
        f"(profile={device_profile}, "
        f"{'INT4+LoRA' if use_4bit else 'BF16+LoRA'} r={lora_r})"
    )

    from_pretrained_kwargs = dict(
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if use_4bit:
        from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen2ForCausalLM.from_pretrained(model_id, **from_pretrained_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer
