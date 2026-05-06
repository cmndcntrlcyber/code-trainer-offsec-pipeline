# Code-Trainer V6 → RTPI: Deploying a Fine-Tuned 14B Coder Model into an Agentic Red-Team Platform

**Author:** cmndcntrlcyber
**Module 1 model artifact:** [`cmndcntrlcyber/qwen14b-code-trainer-v6-gguf`](https://huggingface.co/cmndcntrlcyber/qwen14b-code-trainer-v6-gguf) (Q4_K_M, 8.4 GB)
**LoRA source:** [`cmndcntrlcyber/qwen14b-code-trainer-v6-aggressive`](https://huggingface.co/cmndcntrlcyber/qwen14b-code-trainer-v6-aggressive)
**Deployment target (this module's GitHub repo):** [`cmndcntrlcyber/rtpi`](https://github.com/cmndcntrlcyber/rtpi) — Red Team Portable Infrastructure
**Companion (Module 1):** [`docs/ReadyTensor Submission/publication.md`](./publication.md) · GitHub [`code-trainer-offsec-pipeline`](https://github.com/cmndcntrlcyber/code-trainer-offsec-pipeline)

---

## TL;DR

Module 1 produced a Qwen2.5-Coder-14B-Instruct LoRA fine-tuned on screenshot-to-
code, merged and quantized to Q4_K_M GGUF (8.4 GB). This module deploys that
model as a first-class **inference provider inside RTPI** — a production-shape
agentic red-team platform with a multi-provider inference registry, vLLM and
Ollama clients, a built-in OWASP-LLM assessment surface, Docker Compose prod
configuration, and integrated performance benchmarking UI. The deployment plan
covers self-hosted single-GPU primary, optional cloud burst, monitoring against
RTPI's existing telemetry, and an opinionated security posture inherited from
RTPI's red-team threat model.

---

## 1. Use Case Definition

### Problem statement

Red-team operators need an **on-prem, code-fluent LLM** they can call from
inside an agentic workflow without leaking sensitive engagement data to a
third-party provider. Off-the-shelf coder LLMs are good at synthesis but lack
domain priors for offsec tooling, screenshot-driven code reading, and
compiled-language code generation under tight latency budgets. The Module 1
fine-tune addresses the domain fit; Module 2 makes it usable.

### Target users

* **Primary:** authorized red-team operators using RTPI for engagement
  scoping, payload synthesis, log analysis, and report generation.
* **Secondary:** the platform's own agents (Reporter, Ops Manager, OWASP-LLM
  assessor) calling the model via RTPI's `inference-provider-registry`.

### Inputs and outputs

| Surface | Input shape | Output shape |
|---|---|---|
| Reporter agent | finding metadata + raw evidence (text) | structured incident report (Markdown) |
| Ops Manager chat | natural-language operator query | tool-calling messages → final answer |
| OWASP-LLM assessment | prompt + target context | structured vulnerability + mitigation table |
| Code-from-screenshot (research) | base64 image (carries through Phase 3 stack) | source code (text) |

### Concrete I/O example (Ops Manager)

```text
USER (operator):
  We have shell on a Windows host (kerberoastable accounts found).
  Generate a Rubeus invocation that requests TGS for service "MSSQLSvc/db01"
  using the captured TGT, output to `roast.kirbi`.

MODEL:
  Rubeus.exe asktgs /service:MSSQLSvc/db01 /ticket:<base64-tgt> /nowrap /outfile:roast.kirbi
  …
  Notes: The /nowrap flag prevents line breaks in the output …
```

### Success criteria

| Dimension | Target | Measurement |
|---|---|---|
| Latency (p95) | ≤ 2.0 s for 256-token completions on Q4_K_M | RTPI `PerformanceBenchmarks` panel |
| Throughput | ≥ 25 tokens/s sustained on a single A100 80 GB or RTX 5060 Ti | same |
| Cold-start | ≤ 30 s (model already on disk) | docker compose timing |
| GSM8K (forgetting check) | ≥ 60 % flexible-extract — confirmed in Module 1 | Module 1 §4.3 |
| eval_loss vs base | ≥ 1.5 % improvement over base — confirmed in Module 1 (0.4724) | Module 1 §4.2 |
| Availability | ≥ 99 % during engagement windows (single-host) | RTPI `/health` + alerting |

### Traffic expectations

Per-engagement, not 24×7 SaaS:

| Phase | Concurrent operators | Tokens/min (peak) | Sessions/day |
|---|---|---|---|
| Recon / scoping | 1–2 | ~1,500 | 5–10 |
| Active testing | 2–4 | ~5,000 | 20–40 |
| Reporting | 1 | ~10,000 (long generations) | 1–3 |

This shape favours **bursty single-host inference** over horizontal scaling.

---

## 2. Model Selection and Configuration

### Model

| Field | Value |
|---|---|
| Name | Qwen2.5-Coder-14B-Instruct + Code-Trainer V6 LoRA (`aggressive` config) |
| Source — base | [`Qwen/Qwen2.5-Coder-14B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) |
| Source — adapter | [`cmndcntrlcyber/qwen14b-code-trainer-v6-aggressive`](https://huggingface.co/cmndcntrlcyber/qwen14b-code-trainer-v6-aggressive) |
| Source — merged + quantized | [`cmndcntrlcyber/qwen14b-code-trainer-v6-gguf`](https://huggingface.co/cmndcntrlcyber/qwen14b-code-trainer-v6-gguf) |
| Parameters | 14.7 B (base) + ~280 M LoRA (r=64, α=128) |
| Quantization | Q4_K_M GGUF — 8.4 GB on disk; ~10 GB VRAM at runtime |
| Context length | 4,096 tokens (matches Module 1 training; configurable up to 32K via base) |
| Max output tokens | 1,024 default; 4,096 cap for reporting flows |
| Tokenizer | Qwen2.5 BPE (151,665 vocab) |

### Why this model + this quantization

* **Why a 14 B coder over GPT-4-class APIs.** On-prem requirement (red-team
  data sensitivity) plus zero-egress threat model means cloud APIs are out.
  14 B is the largest model we can hot-swap on a single A100 80 GB or run on
  a 16 GB consumer GPU at Q4_K_M.
* **Why Qwen2.5-Coder over CodeLlama / DeepSeek-Coder.** Module 1 sweep
  explicitly chose this base; eval_loss on a multilingual offsec dataset was
  the deciding factor.
* **Why Q4_K_M over Q5_K_M / Q8_0 / FP16.** Q4_K_M is the architecture-doc
  recommendation for the Phase 6 hot-swap target. Trade-offs:

  | Quant | Size | VRAM | Quality (vs FP16) | Decision |
  |---|---|---|---|---|
  | F16    | 28 GB | ~32 GB | baseline | Out — exceeds consumer VRAM |
  | Q8_0   | 15 GB | ~17 GB | ~−1 % | Future variant for high-quality reporting |
  | **Q4_K_M** | **8.4 GB** | **~10 GB** | **~−2–4 %** | **Default — fits 16 GB GPU + headroom** |
  | Q5_K_M | 10 GB | ~12 GB | ~−1.5 % | Optional middle-ground |

* **Why this LoRA over Phase 4B's 3-epoch slice.** Module 1 §4.2 — 0.4724
  beats 0.5126 on full validation; +12 % relative on GSM8K shows no
  catastrophic forgetting.

### Configuration knobs at serve time

```yaml
# Ollama / llama.cpp server flags (production)
n_ctx: 4096
n_predict: 1024
temperature: 0.2          # deterministic for code; raise to 0.7 for chat
top_p: 0.9
top_k: 40
repeat_penalty: 1.05
seed: -1
n_gpu_layers: 999         # offload all layers to GPU
n_threads: 8              # CPU fallback layers, if any
batch_size: 1             # red-team workloads are bursty, not batched
```

### Adversarial configuration (RTPI's OWASP-LLM panel)

A second, sandboxed deployment runs the same model with stricter guardrails
for adversarial-prompt testing — see §6.

---

## 3. Deployment Strategy

### Platform

**Primary path:** RTPI's `inference-provider-registry` registers the
fine-tuned model under two backends:

* **Ollama** for the Q4_K_M GGUF (default, lightest). RTPI's
  `server/services/ollama-ai-client.ts` and `ollama-manager.ts` already wrap
  the API. Modelfile shipped with the GGUF Hub repo.
* **vLLM** for the merged FP16 / BF16 weights (high-quality lane). RTPI's
  `server/services/inference/vllm-client.ts` already implements the client.

Both expose a uniform OpenAI-compatible `/v1/chat/completions` interface so
RTPI's agent code is agnostic.

### Hardware

| Tier | GPU | Use |
|---|---|---|
| **Edge (operator laptop / red-team kit)** | RTX 4090 / 5090 (24 GB) or 5060 Ti (16 GB) | Q4_K_M via Ollama; ≥ 30 tokens/s; <1 s p95 for short completions |
| **Engagement HQ** | A100 80 GB (1×) | Either Q4_K_M with massive headroom, or FP16 via vLLM for reporting; supports hot-swap to a different specialist model |
| **Cloud burst (optional)** | HF Skills `a100-large` ($3.20/h) | Same image, ephemeral — used only when local is overloaded or unavailable |

### Endpoint type

* **Internal HTTP** (`localhost:8080` for llama-server / `localhost:11434`
  for Ollama) — never exposed externally.
* **Authenticated WebSocket** (RTPI) — operators interact through the RTPI
  frontend; the LLM is reached only through RTPI's authenticated API.

### Geographic region

* On-prem: lives wherever the engagement infrastructure lives — typically
  the same network segment as the operator workstations to keep latency
  predictable and traffic egress-free.
* Cloud burst (optional): match the HF Skills region to the operator
  region to keep p99 latency tolerable; defer to engagement-specific
  data-residency requirements.

### Scaling approach

| Knob | Choice | Rationale |
|---|---|---|
| Vertical (single-host VRAM) | Default — 1× A100 or 1× consumer GPU | Traffic is bursty + isolated per engagement |
| Horizontal | None for primary path | Adds attack surface and engagement-data leakage risk |
| Hot-swap | RTPI's `phase6_inference/scripts/hot_swap.py` | Swap from Qwen-3.5-9B (general / chat) to this 14 B coder when compiled-language tasks come up |
| Cloud burst | Submit-and-poll via HF Jobs | Zero standing cost; pay only when local is saturated |

### Container topology (RTPI prod)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Operator browser → RTPI frontend (port 5000)                        │
└──────────────┬──────────────────────────────────────────────────────┘
               │ authenticated WebSocket / REST
┌──────────────▼──────────────────────────────────────────────────────┐
│ rtpi-server (port 3001) — Express + TypeScript                      │
│   ├─ inference-provider-registry                                    │
│   │    ├─ ollama-ai-client → http://ollama:11434                    │
│   │    └─ vllm-client     → http://vllm:8000                        │
│   ├─ owasp-llm-parser  (input validation / prompt-injection scan)   │
│   ├─ auth (admin + RBAC)                                            │
│   └─ telemetry (PerformanceBenchmarks UI feed)                      │
├─────────────────────────────────────────────────────────────────────┤
│ ollama (port 11434) — pulls qwen14b-code-trainer-v6-gguf:Q4_K_M     │
│ vllm   (port 8000)  — serves merged FP16 (optional lane)            │
│ postgres (5434)  redis (6381)                                       │
└─────────────────────────────────────────────────────────────────────┘
```

All defined in RTPI's `docker-compose.prod.yml`.

### Deployment script (one-line ops)

```bash
git clone https://github.com/cmndcntrlcyber/rtpi.git && cd rtpi
cp .env.example .env                                # set HF_TOKEN, admin pwd
docker compose -f docker-compose.prod.yml up -d
docker exec rtpi-ollama ollama pull \
  hf.co/cmndcntrlcyber/qwen14b-code-trainer-v6-gguf:Q4_K_M
curl -X POST http://localhost:3001/api/v1/inference/providers \
  -H "Authorization: Bearer $RTPI_TOKEN" \
  -d @configs/provider-qwen14b-coder.json
```

---

## 4. Cost Analysis

> _Section in progress — final numbers land after the live `PerformanceBenchmarks`
> run and the Modal/HF-endpoint comparison. Placeholder framework below._

### Self-hosted reference TCO (Engagement HQ tier, monthly)

| Line item | Cost (USD) | Notes |
|---|---|---|
| GPU amortization (1× A100 80 GB) | _TBD: hardware/36 months_ | Capex amortized over typical 3-year refresh |
| Power (24×7 idle + 4 h/day under load) | _TBD: kWh × $0.12_ | A100 SXM4 ~250 W idle, ~400 W under load |
| Cooling overhead | _TBD: 40 % of power_ | datacenter PUE ~1.4 |
| Storage (model weights + RTPI logs) | _TBD: ≤ $5_ | 100 GB on local NVMe |
| Network egress | $0 | On-prem |
| Monitoring (Prometheus + Grafana, self-hosted) | $0 | Containerized alongside RTPI |
| **Self-hosted subtotal (per month)** | _TBD_ | |

### Cloud-burst reference (HF Skills `a100-large`)

| Line item | Cost (USD) | Notes |
|---|---|---|
| HF Skills hourly rate | $3.20 | a100-large per public pricing |
| Typical burst hours per engagement | _TBD_ | Estimated from past Phase 4/5 jobs |
| Per-engagement burst cost | _TBD_ | |

### Cost per request

| Workload | Typical input | Typical output | Tokens/s sustained | Wall time | $ self-hosted | $ HF Skills |
|---|---|---|---|---|---|---|
| Quick chat | 256 | 256 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Reporter (long) | 1,500 | 2,000 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Code synthesis | 800 | 1,000 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Optimization strategies (planned)

| Strategy | Status | Impact |
|---|---|---|
| **Quantization to Q4_K_M** | ✅ shipped (Module 1 Phase 5) | ~3.3× smaller model, ~30 % faster inference vs FP16 on consumer GPU |
| **KV-cache reuse** | Default in Ollama / vLLM | Free for multi-turn chats |
| **Batching** | Disabled (workload is bursty / single-operator) | N/A |
| **Speculative decoding** | Future — pair Q4_K_M with smaller draft model | Estimated 1.5–2× speedup on coder workloads |
| **Hot-swap to smaller primary** | ✅ implemented in `phase6_inference` | Use Qwen-3.5-9B for chat; swap to 14 B coder only when needed |
| **Spot / cloud burst** | Optional via HF Skills | Zero standing cost when idle |
| **Local cache of merged FP16 weights** | RTPI Ollama volume | One-time download, reused across container restarts |

---

## 5. Monitoring and Observability Plan

### Metrics

| Tier | Metric | Source | SLO |
|---|---|---|---|
| **Latency** | p50 / p95 / p99 first-token-time | RTPI `PerformanceBenchmarks` + telemetry middleware | p95 ≤ 2.0 s |
| | p50 / p95 total completion time | same | p95 ≤ 8 s for 1K-token outputs |
| **Throughput** | tokens / second (decode) | llama.cpp / vLLM `/metrics` | ≥ 25 t/s sustained |
| | requests / minute | RTPI HTTP middleware | _no upper bound — alert on saturation_ |
| **Quality** | eval_loss drift on a held-out probe set | scheduled cron in RTPI | < 5 % drift over 30 days |
| | OWASP-LLM panel pass rate | RTPI `owasp-llm-parser` | ≥ 95 % rule pass |
| **Resource** | GPU utilization | `nvidia-smi` exporter → Prometheus | < 90 % sustained → alert on saturation |
| | GPU memory | same | < 90 % of card capacity |
| | Host CPU + RAM | `node-exporter` | < 80 % |
| **Errors** | HTTP 5xx rate | RTPI middleware | < 0.5 % over 5 min |
| | Token-budget overruns | RTPI middleware | < 1 % over 1 h |
| | Provider-fallback events (Ollama → vLLM → cloud) | inference-provider-registry | log + count, alert on > 5 / h |
| **Cost** | Tokens consumed / engagement | RTPI usage table | per-engagement budget alerts |
| | Cloud-burst hours | HF Jobs API | budget alerts |

### Tools

| Layer | Choice | Why |
|---|---|---|
| Metric collection | **Prometheus** (containerized alongside RTPI) | Self-hosted, no egress |
| GPU exporter | **nvidia_gpu_exporter** | Standard, low overhead |
| Process exporter | **node-exporter** | Standard |
| Application metrics | **RTPI's existing telemetry middleware** + custom histograms via `prom-client` | Already in the codebase |
| Dashboards | **Grafana** (containerized) | Standard; RTPI's `PerformanceBenchmarks` UI consumes a subset for the operator-facing view |
| Logs | **Loki** (or Postgres for structured rows) | Self-hosted; correlate with engagement IDs |
| Traces | **OpenTelemetry → Jaeger** | Optional — useful for diagnosing provider-fallback chains |
| Alerting | **Alertmanager → Slack / email / SMTP-relay** | Self-hosted; operator-on-call runbook in `docs/runbooks/` |

### Alert strategy

| Severity | Condition | Action |
|---|---|---|
| **Page** (1) | Inference unavailable > 2 min during active engagement | Auto-page on-call operator; auto-failover to cloud burst if approved |
| **Page** (2) | p95 first-token-time > 5 s for 5 min | Page; check GPU saturation, fallback eligibility |
| **Page** (3) | Quality probe drift > 10 % | Page; trigger model-card re-eval |
| **Warn** | GPU memory > 90 % for 10 min | Slack to operator; suggest hot-swap or batch reduction |
| **Warn** | OWASP-LLM panel pass rate < 90 % | Slack; flag for prompt-template review |
| **Info** | Cloud-burst hours > 80 % of monthly budget | Slack |

### Quality probes

A 50-prompt smoke set lives in `rtpi/tests/inference/quality-probes.json`,
re-run nightly via cron. Outputs are scored via lm-evaluation-harness
(reusing the Module 1 GSM8K wiring) plus a structural-format check (was the
output JSON valid? did the report contain all required sections?).

---

## 6. Security Considerations

RTPI inherits a red-team threat model — the LLM is being deployed *into* a
hostile-tooling context, not exposed to public traffic. The security
considerations skew accordingly.

### API authentication

| Layer | Mechanism |
|---|---|
| Frontend → RTPI API | Session cookie + CSRF token; admin / operator roles |
| Operator → LLM endpoint | Goes only through authenticated RTPI API; the Ollama / vLLM ports are container-internal, never bound to host externally |
| RTPI → upstream Ollama / vLLM | mTLS optional (engagement HQ); plain HTTP within Docker network for edge tier |
| Inter-engagement isolation | Per-engagement Postgres schema + RBAC; LLM context never crosses engagements |

Default admin credentials (`admin / Admin123!@`) are explicitly flagged in
RTPI's README as **must rotate on first login**.

### Rate limiting

* **Per-user request budget** (default: 60 req/min, configurable per role) —
  enforced in RTPI's Express middleware.
* **Per-engagement token budget** (default: 1 M tokens / 24 h) — enforced in
  the inference-provider-registry.
* **Provider-tier rate limits** — Ollama and vLLM each have a configurable
  concurrent-request cap; excess requests queue with backpressure.

### Input validation

| Vector | Defence |
|---|---|
| Prompt injection | RTPI's **`owasp-llm-parser`** + the OWASP-LLM Assessment panel — checks against the OWASP Top 10 LLM list (LLM01 prompt injection, LLM02 insecure output, …) before forwarding to the model |
| Tool-call injection | MCP servers run sandboxed; allow-listed tool registry per engagement |
| Long-input DoS | Hard 4 K-token cap on input; fail-closed |
| Token-smuggling | Strip control tokens (`<\|im_start\|>` etc.) from operator input before chat-template assembly |
| Encoded-payload exfil | Output scanner checks for base64 / hex blobs > 1 KB; flagged for review before display |

### PII handling

* **No PII in training data.** The Module 1 dataset is GitHub-sourced public
  code; the dataset card documents this explicitly.
* **Engagement data PII.** Operators may paste identifiers, emails, IPs into
  prompts — RTPI logs these per-engagement and never forwards prompts or
  completions to any third-party provider when the local model is in use.
  The cloud-burst path (HF Jobs) is **opt-in per engagement** and explicitly
  warns the operator before forwarding.
* **Right-to-erasure.** Per-engagement schemas can be dropped wholesale; no
  cross-engagement model-state retention beyond the model weights themselves.

### Access control

* **Role-based (RBAC):** `admin`, `operator`, `auditor` — granular
  permissions on engagement creation, tool execution, model-config edits,
  log access.
* **Audit log:** every model invocation writes to RTPI's audit table with
  user ID, engagement ID, prompt hash, completion hash, latency, token
  count.
* **Network:** model containers bind to `127.0.0.1` only; external access
  goes through the authenticated RTPI API. The Docker network is isolated
  per-host.
* **Secrets:** HF tokens, admin passwords, etc. via `.env` (gitignored) or
  Docker secrets; never committed.

### Adversarial use considerations

This is a security-domain deployment of a security-domain model, so the
"misuse" threat is operator-centric rather than external-attacker-centric.
RTPI requires **explicit engagement scoping** before tool execution — the
LLM cannot trigger external network actions outside scoped target ranges.
The OWASP-LLM Assessment panel is itself a reflexive check: a red-team
operator can run the model against itself to demonstrate resistance to the
top adversarial-prompt categories, satisfying both research and audit
requirements.

---

## 7. Reproducibility

### To redeploy the full stack

```bash
# 1. Get RTPI
git clone https://github.com/cmndcntrlcyber/rtpi.git && cd rtpi
cp .env.example .env
# Edit .env: set HF_TOKEN, ADMIN_PASSWORD, RTPI_BASE_URL

# 2. Bring up infrastructure
docker compose -f docker-compose.prod.yml up -d postgres redis
npm install
npm run db:push
npm run db:create-admin

# 3. Pull the fine-tuned model into Ollama
docker compose -f docker-compose.prod.yml up -d ollama
docker exec rtpi-ollama ollama pull \
  hf.co/cmndcntrlcyber/qwen14b-code-trainer-v6-gguf:Q4_K_M

# 4. Register as inference provider via RTPI API
curl -X POST http://localhost:3001/api/v1/inference/providers \
  -H "Authorization: Bearer $RTPI_TOKEN" \
  -H "Content-Type: application/json" \
  -d @configs/provider-qwen14b-coder.json

# 5. (Optional) bring up vLLM lane for FP16 quality
docker compose -f docker-compose.prod.yml up -d vllm

# 6. Smoke test
npm run test -- --testPathPattern=inference
```

### Repos

| Layer | Repo |
|---|---|
| Module 1 — fine-tuning | <https://github.com/cmndcntrlcyber/code-trainer-offsec-pipeline> |
| Module 2 — deployment (this) | <https://github.com/cmndcntrlcyber/rtpi> |

### Hub artifacts

| Artifact | URL |
|---|---|
| LoRA adapter | <https://huggingface.co/cmndcntrlcyber/qwen14b-code-trainer-v6-aggressive> |
| GGUF Q4_K_M | <https://huggingface.co/cmndcntrlcyber/qwen14b-code-trainer-v6-gguf> |
| Vision adapter (Module 1 multimodal stage) | <https://huggingface.co/cmndcntrlcyber/code-trainer-vision-adapter> |
| Dataset | <https://huggingface.co/datasets/cmndcntrlcyber/code-trainer-offsec-dataset> |

---

## 8. What's still pending

This scaffold ships ahead of the live integration so the Module 2 narrative
is reviewable. The following items will be filled in before submission:

- [ ] **§4 Cost Analysis** — replace `_TBD_` rows with live numbers from a
      single A100 measurement + a Modal/HF-endpoint comparison quote.
- [ ] **§5 Monitoring** — capture Grafana dashboard screenshot + add a
      sample alert rule from `rtpi/configs/alerting/`.
- [ ] **§3 Deployment Strategy** — link to the actual provider-registration
      PR in rtpi (`configs/provider-qwen14b-coder.json` + migration row).
- [ ] **Architecture diagram** (bonus) — convert the ASCII container
      topology to a proper diagram and embed as `assets/rtpi-deployment.png`.
- [ ] **Performance numbers** — RTPI `PerformanceBenchmarks` panel
      screenshot + JSON export, embedded in §1 Success Criteria + §5 SLOs.

Each item is a 30–60 minute task; the writeup framework above does not
change once they land.

---

## Acknowledgements

Built on Module 1's training pipeline plus open-source primitives:
[Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct),
[llama.cpp](https://github.com/ggerganov/llama.cpp),
[vLLM](https://github.com/vllm-project/vllm),
[Ollama](https://github.com/ollama/ollama),
[Qwen-Agent](https://github.com/QwenLM/Qwen-Agent), and the
[RTPI](https://github.com/cmndcntrlcyber/rtpi) platform's existing
inference-provider-registry, OWASP-LLM panel, and Docker Compose stack.
