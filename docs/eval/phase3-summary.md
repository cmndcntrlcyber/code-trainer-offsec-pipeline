# Phase 3 — Vision Model Eval

**Adapter:** [cmndcntrlcyber/code-trainer-vision-adapter](https://hf.co/cmndcntrlcyber/code-trainer-vision-adapter)
**Dataset:** [`cmndcntrlcyber/code-trainer-offsec-dataset@v2-multimodal`](https://hf.co/datasets/cmndcntrlcyber/code-trainer-offsec-dataset/tree/v2-multimodal)
**Split:** test (200 samples)
**Generated:** 2026-05-03
**Job:** [`69f70db798a8d679adfb8ac0`](https://huggingface.co/jobs/cmndcntrlcyber/69f70db798a8d679adfb8ac0) — A100-large, ran 24m 48s
**Raw JSON:** [`phase3-summary.json`](./phase3-summary.json)

## Results

| Metric                | Baseline (Qwen2.5-Coder-1.5B) | Fine-tuned | Δ |
|-----------------------|-------------------------------|------------|---|
| exact_match           | 0.0000                        | 0.0000     | 0 |
| bleu_4                | 0.0000                        | 0.0000     | 0 |
| mean_edit_similarity  | 0.0000                        | 0.0000     | 0 |
| syntax_valid_rate †   | 1.0000                        | 1.0000     | 0 |

† Syntax check uses Python parser; the test split is multilingual (java 5140, ts 5095, csharp 5035, python 3300, cpp 3156, go 2086, rust 1457, js 857), so this metric is only meaningful for the Python subset.

## ⚠️ Likely eval-pipeline bug — not a real model result

All four content metrics are 0.0 for **both** baseline and fine-tuned runs, and `syntax_valid_rate=1.0` is the giveaway: an empty string parses as a valid (empty) Python module, so `syntax_valid_rate=1.0` indicates predictions are empty strings.

Suspected root cause is the prediction-extraction slice in `src/phase3_vision_model/evaluation/evaluator.py:93-94`:

```python
output_ids = self.model.generate(inputs_embeds=combined, ...)
for b in range(output_ids.shape[0]):
    pred_ids = output_ids[b][combined.shape[1]:]   # ← strips real tokens
    pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
```

When `model.generate` is called with `inputs_embeds=`, the returned tensor contains only the **newly generated** tokens, not the input-prefixed sequence. Slicing by `combined.shape[1]` (the input length) discards every generated token and leaves the empty tail, so `pred_text` is `""` for every sample.

A fix would either:
1. Drop the slice entirely when `inputs_embeds` is used (just decode `output_ids[b]`), or
2. Switch to a generate path that returns the input + generation concatenated.

This needs to be fixed in `evaluator.py`, the eval re-run on a smaller smoke set first to verify, then a full A100 re-run to land real numbers.

## Status

- [x] Eval infrastructure (`launch_eval.py` + `eval_entry.py`) shipped and proven via a successful 24m 48s A100 run
- [x] Hub publish path validated: `cmndcntrlcyber/code-trainer-vision-adapter/tree/main/eval/`
- [ ] Real metrics — blocked on the prediction-extraction fix above
- [ ] Per-language metrics breakdown (separate enhancement)
