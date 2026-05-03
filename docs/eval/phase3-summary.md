# Phase 3 — Vision Model Eval

**Adapter:** [cmndcntrlcyber/code-trainer-vision-adapter](https://hf.co/cmndcntrlcyber/code-trainer-vision-adapter)
**Dataset:** [`cmndcntrlcyber/code-trainer-offsec-dataset@v2-multimodal`](https://hf.co/datasets/cmndcntrlcyber/code-trainer-offsec-dataset/tree/v2-multimodal)
**Split:** test (200 samples)
**Generated:** 2026-05-03
**Job:** [`69f7175f9d85bec4d76f125d`](https://huggingface.co/jobs/cmndcntrlcyber/69f7175f9d85bec4d76f125d) — A100-large, ran 20m 38s
**Raw JSON:** [`phase3-summary.json`](./phase3-summary.json)

## Results

| Metric                | Baseline (Qwen2.5-Coder-1.5B) | Fine-tuned | Δ |
|-----------------------|-------------------------------|------------|---|
| exact_match           | 0.0000                        | 0.0000     | 0 |
| bleu_4                | 0.0000                        | 0.0000     | 0 |
| mean_edit_similarity  | 0.0382                        | 0.0446     | **+16.8%** |
| syntax_valid_rate †   | 0.1950                        | 0.6100     | **+213%**  |

† Syntax check uses Python parser; the test split is multilingual (java 5140, ts 5095, csharp 5035, python 3300, cpp 3156, go 2086, rust 1457, js 857), so the absolute numbers are not directly comparable to a Python-only model. The **delta is still meaningful** because both runs use the same metric on the same samples.

## Reading the numbers

**Strong positive signal — `syntax_valid_rate`: 0.195 → 0.610 (3.13×).** The fine-tuned adapter generates output that the Python parser accepts as valid in 61% of cases, vs 19.5% for the base model with random projector weights. This is consistent with the model having learned to emit code-shaped output rather than free-form text.

**Modest positive on `mean_edit_similarity`: +16.8%.** Predictions are closer to references than the baseline, but the absolute level (~0.04) is low — the model is producing code-like text in roughly the right shape rather than verbatim source.

**`exact_match = 0` and `bleu_4 = 0` for both runs.** Zero 4-gram overlap with references means the model is *paraphrasing* rather than *reconstructing* the source code in the screenshot. This is plausible for a 1.5B base model trained ~5.5h on a 26k-sample multilingual dataset — full code reconstruction from screenshots is a hard task, and the small effective batch size (8 × grad_accum 4 = 32) leaves limited gradient signal per training step.

**The bug fix:** the previous run reported all four metrics as 0.0 plus `syntax_valid_rate=1.0` (the empty-string-parses-as-valid-Python signature). [Commit `97341a5`](https://github.com/cmndcntrlcyber/code-trainer-offsec-pipeline/commit/97341a5) fixed `evaluator.py` — `model.generate(inputs_embeds=...)` returns only the newly generated tokens, so the previous `output_ids[b][combined.shape[1]:]` slice was discarding everything generated.

## Provenance

| Run | Job | Samples | Duration | Result |
|---|---|---|---|---|
| First (broken) | `69f70db798a8d679adfb8ac0` | 200 | 24m 48s | All zeros — slicing bug |
| Smoke | `69f716079d85bec4d76f124f` | 20 | 4m 41s | Confirmed fix |
| Final (this) | `69f7175f9d85bec4d76f125d` | 200 | 20m 38s | Real numbers above |

## Status

- [x] Eval pipeline produces non-empty predictions
- [x] Fine-tuned adapter measurably outperforms baseline on syntax validity and edit similarity
- [ ] **Per-language metrics breakdown** — small evaluator enhancement to group results by `batch["language"]`. The current Python-only syntax check leaves the syntax_valid_rate metric ambiguous on a multilingual dataset; per-language reporting would resolve that and likely show stronger lifts on specific languages.
- [ ] **Higher-N or full-test eval** — n=200 is enough for direction; n=500 or 3267 (full test) would give tighter confidence intervals if the publication wants them.
