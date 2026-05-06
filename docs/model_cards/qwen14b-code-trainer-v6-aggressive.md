---
base_model: Qwen/Qwen2.5-Coder-14B-Instruct
library_name: peft
license: apache-2.0
tags:
- code-generation
- lora
- peft
- qwen2.5-coder
- code-trainer-v6
datasets:
- cmndcntrlcyber/code-trainer-offsec-dataset
pipeline_tag: text-generation
---

# qwen14b-code-trainer-v6-aggressive

LoRA adapter for **Qwen/Qwen2.5-Coder-14B-Instruct**, fine-tuned to generate
source code from chat-formatted instructions in the
[`code-trainer-offsec-dataset`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-offsec-dataset).
This is the winner of the Phase 4A validation sweep across three LoRA
configurations (eval_loss = **0.4724** on the full 3,265-row validation
split — see [eval results](#evaluation)).

It is the canonical Phase 5 conversion target for the Code-Trainer V6 / RTPI
project ([GitHub](https://github.com/cmndcntrlcyber/code-trainer-offsec-pipeline)),
chosen over the 3-epoch Phase 4B variant after a head-to-head full-validation
comparison.

## Intended use

* **Direct use:** load the adapter on top of `Qwen/Qwen2.5-Coder-14B-Instruct`
  and use it for instruction-following code generation in the eight languages
  of the dataset (Python, JavaScript, TypeScript, Java, Go, Rust, C++, C#).
* **Downstream:** merge into the base model and quantize to GGUF for local
  serving — see
  [`qwen14b-code-trainer-v6-gguf`](https://huggingface.co/cmndcntrlcyber/qwen14b-code-trainer-v6-gguf).
* **Out of scope:** this adapter was *not* trained for safety alignment, RLHF,
  or any non-code task; treat any non-code response as undefined behaviour.

## Training data

* **Dataset:** [`cmndcntrlcyber/code-trainer-offsec-dataset`](https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-offsec-dataset)
  (revision `main`, text-only chat format).
* **Splits seen:** 26,126 train rows, 3,265 validation rows.
* **Source:** scraped GitHub repositories across 8 languages, filtered by a
  composite quality score (stars, activity, docs, code quality, community)
  and code-shape heuristics (20–500 lines, 200 B – 50 KB per file).
* **Format:** OpenAI-style `messages` (system / user / assistant) where the
  user turn carries an instruction and the assistant turn is the source code
  to produce.

## Training procedure

| Knob | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-14B-Instruct` |
| Adapter | LoRA (PEFT), `r = 64`, `alpha = 128`, `dropout = 0.05` |
| Optimizer | fused AdamW (BF16 master weights) |
| Learning rate | 3e-4 (cosine decay, warmup ratio 0.03) |
| Batch size × grad accum | 4 × 4 (effective batch = 16) |
| Epochs | 1 (full data — see Phase 4A vs 4B note below) |
| Sequence length | 2,048 |
| Precision | bfloat16 + gradient checkpointing |
| Hardware | HF Skills `a100-large` (1× A100 80 GB) |
| Frameworks | `transformers`, `peft`, `trl` (SFTTrainer) |

The Phase 4A sweep also covered a `conservative` (r=16, lr=1.5e-4) and
`standard` (r=32, lr=2e-4) configuration; `aggressive` won by ≥ 0.007 eval_loss
on identical data. See [`docs/sweep/phase4a-summary.md`](https://github.com/cmndcntrlcyber/code-trainer-offsec-pipeline/blob/main/docs/sweep/phase4a-summary.md)
for the full sweep table.

A second-pass **Phase 4B** experiment ran the same `aggressive` config for 3
epochs over an 8 K slice of the data (24 K total samples seen). On the full
validation split it scored 0.5126 — measurably worse than this adapter's 0.4724
across 1 epoch on the full 26 K. Conclusion: more unique examples > more passes
for this dataset. See
[`docs/sweep/phase4b-summary.md`](https://github.com/cmndcntrlcyber/code-trainer-offsec-pipeline/blob/main/docs/sweep/phase4b-summary.md).

## Evaluation

### Task-specific (full validation split, 3,265 rows)

| Metric | Value |
|---|---|
| eval_loss | **0.4724** |
| eval_runtime | 677.88 s |

Source: HF Job [`69f7659298a8d679adfb8b8e`](https://huggingface.co/jobs/cmndcntrlcyber/69f7659298a8d679adfb8b8e),
re-evaluated post-hoc against the full val split via job
[`69f89e5b9d85bec4d76f217e`](https://huggingface.co/jobs/cmndcntrlcyber/69f89e5b9d85bec4d76f217e)
(also `phase4-eval-full.json` in this repo).

### Sweep comparison (Phase 4A — full val)

| Config | r / α | LR | eval_loss |
|---|---|---|---|
| `conservative` | 16 / 32 | 1.5e-4 | 0.4819 |
| `standard`     | 32 / 64 | 2e-4   | 0.4798 |
| **`aggressive` (this)** | **64 / 128** | **3e-4** | **0.4724** |

### General benchmark — catastrophic-forgetting check (GSM8K, 0-shot)

> Numbers populated by `src.phase4_qwen_finetuning.scripts.launch_benchmark`.
> Both rows use the same lm-evaluation-harness pipeline (`lm-eval==0.4.4`).

| Run | exact-match (strict) |
|---|---|
| Base `Qwen/Qwen2.5-Coder-14B-Instruct` | _see `phase4-benchmark-gsm8k-base.json`_ |
| **+ this adapter** | _see `phase4-benchmark-gsm8k.json`_ |

GSM8K (math word problems) is orthogonal to the screenshot-to-code training
task, so any large drop here indicates catastrophic forgetting on general
reasoning.

## Limitations

* **Multilingual eval is approximate.** The Phase 3 `syntax_valid_rate` metric
  uses a Python parser, so the absolute numbers across the dataset's 8
  languages are not directly comparable; the *delta* against the base model is
  the meaningful signal.
* **No safety tuning.** This adapter is for code synthesis; it inherits the
  base model's safety properties and adds no extra alignment.
* **One epoch.** A single pass through the data leaves further headroom; gains
  are likely possible with longer training or larger sequence length, at the
  cost of compute.
* **Adapter, not full weights.** You need the base model to use this — total
  download is ~28 GB (base) + ~0.7 GB (this adapter).

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
adapter_id = "cmndcntrlcyber/qwen14b-code-trainer-v6-aggressive"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(
    base_id, dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

messages = [
    {"role": "system", "content": "You are a senior software engineer."},
    {"role": "user", "content": "Write a Rust function that parses an ISO-8601 timestamp."},
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True,
).to(model.device)
out = model.generate(inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

## Reproducibility

* **Code:** [github.com/cmndcntrlcyber/code-trainer-offsec-pipeline](https://github.com/cmndcntrlcyber/code-trainer-offsec-pipeline)
* **Launch command (HF Jobs):**
  ```bash
  python -m src.phase4_qwen_finetuning.scripts.launch_full_training \
      --config src/config/v6_config.yaml --best-config aggressive --wait
  ```
* **W&B project:** `rtpi-phase4-qwen14b` (W&B link in the GitHub README).
* **Cost:** ~$22 on `a100-large` (~7 h, eventually marked ERROR by HF Skills'
  lazy timeout enforcement, but the adapter was already pushed to Hub before
  the kill — see Phase 4A summary).
