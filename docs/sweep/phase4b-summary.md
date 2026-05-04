# Phase 4B — Full Training Summary

**Adapter:** [cmndcntrlcyber/qwen14b-code-trainer-v6-aggressive-full3](https://hf.co/cmndcntrlcyber/qwen14b-code-trainer-v6-aggressive-full3) (1.05 GB)
**Job:** [`69f8188e9d85bec4d76f1c5e`](https://huggingface.co/jobs/cmndcntrlcyber/69f8188e9d85bec4d76f1c5e) — A100-large, 4h 53m, COMPLETED
**Generated:** 2026-05-04
**Raw JSON:** [`phase4b-result.json`](./phase4b-result.json)

## Configuration

| Knob | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-14B-Instruct` |
| LoRA r / α | 64 / 128 |
| Learning rate | 3e-4 |
| Batch size × accum | 4 × 4 (eff_batch = 16) |
| Epochs | 3 |
| Train rows | 8,000 (sliced from 26,126) |
| Val rows | 500 (sliced from 3,265) |
| Precision | BF16 + gradient checkpointing |

## Result

```json
{
  "num_epochs": 3,
  "eval_loss": 0.5101475119590759,
  "eval_runtime": 102.88
}
```

## Phase 4A vs Phase 4B (caveat: val sets differ)

| Metric | Phase 4A `aggressive` (1 ep × full 26k) | Phase 4B `aggressive-full3` (3 ep × 8k) |
|---|---|---|
| eval_loss | **0.4724** | 0.5102 |
| val rows | 3,265 | 500 |
| Total samples seen | 26,126 | 24,000 |
| Job runtime | 6h 39m (timeout) | 4h 53m |

**Direct comparison is invalid** — Phase 4B used `--val-limit 500` to shrink eval time, so the 0.5102 score is on a different (smaller, possibly easier or harder) val subset than Phase 4A's 0.4724. The fact that 4B is *worse* on its own slice is suggestive but not conclusive.

Two plausible explanations:

1. **Val-split artifact.** A 500-row random head slice may simply contain a harder language mix (more Java/C++ vs more Python), inflating the loss.
2. **Overfitting on the train slice.** 3 epochs over 8k rows = 24k samples seen with repetition; Phase 4A saw 26k unique samples once. Repetition may have memorized rather than generalized.

To resolve: run a 5-min eval-only A100 job (~$0.30) of the `aggressive-full3` adapter against the full 3,265-row val split. That gives a comparable number.

## Cost log

| Phase | Run | Hours | Cost |
|---|---|---|---|
| 4A | conservative | 7.0 (timeout) | ~$22.40 |
| 4A | standard | 6.7 (timeout) | ~$21.30 |
| 4A | aggressive | 7.0 (timeout) | ~$22.40 |
| 4B | aggressive-full3 | 4.9 | ~$15.60 |
| **Phase 4 total** | | | **~$81.70** |

(All Phase 4A runs marked ERROR by HF Skills' lazy timeout but pushed adapters successfully before the kill — see `phase4a-summary.md`.)

## Status

- [x] Phase 4B trained checkpoint published (1.05 GB LoRA adapter)
- [ ] Comparable eval_loss against full val split (eval-only A100 job)
- [ ] Phase 5 GGUF conversion (merge LoRA → base, quantize to Q4_K_M)
