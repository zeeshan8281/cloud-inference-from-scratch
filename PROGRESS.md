# Cloud Inference Engine Lab — Build Progress

**Date:** 2026-08-24
**Repo:** `/Users/zeeshan/Downloads/JSBF;SOA/cloud-inference-from-scratch` (new, isolated repository; baseline committed)
**Source PRD:** `/Users/zeeshan/cloud-inference-engine-prd.md`
**Local runner:** `python3.11` via `.venv/` (system `python3` is 3.9.6 — too old; Modal image uses 3.12)

---

## 1. Milestone status overview

| Milestone | Scope | Status |
|---|---|---|
| M0 | Repo skeleton, pins, config, LICENSE, pyproject | **Complete** |
| M1 | Custom Qwen2 model + explicit safetensors loader + naive decoding | **Code complete** (parity unverified — needs GPU) |
| M2 | Contiguous KV cache mode | **Code complete** (unverified on GPU) |
| M3 | Continuous batching scheduler + metrics | **Code complete**, local tests green |
| M4 | Paged KV allocation + torch reference paged attention | **Code complete** (unverified on GPU) |
| M5 | Triton direct-block decode attention kernel | **Code complete** (unverified on GPU) |
| M6 | API subset, SSE streaming, health/metrics, Modal deployment wiring | **Code complete** (integration unverified) |
| M7 | Benchmarks + workloads + docs + README + launch package | **Partially complete** — code/docs written; measurements and launch proof pending |

Overall: local implementation and documentation are complete; correctness on real hardware is **unverified** (no billable commands have been run).

---

## 2. Verified external pins (researched, per PRD §21.3)

| Item | Value | Source |
|---|---|---|
| Model revision SHA | `060db6499f32faf8b98477b0a26969ef7d8b9987` | HF API for Qwen/Qwen2.5-0.5B |
| Model dtype on disk | BF16 safetensors (494M params) → loader casts to FP16 | HF card metadata |
| Config facts | hidden=896, layers=24, heads=14, kv_heads=2, head_dim=64, intermediate=4864, vocab=151936, rope_theta=1e6, rms_eps=1e-6, tied embeddings, eos=151643 | raw config.json at pinned rev |
| Modal lifecycle hook | `@modal.enter()` / `@modal.exit()` on `@app.cls` (old `__enter__` deprecated) | modal.com/docs/guide/lifecycle |
| Autoscaler params | `max_containers`, `min_containers`, `buffer_containers`, `scaledown_window` — all current names | modal.com/docs/guide/scale |

Pins recorded in `engine_config.json`: torch 2.7.1, triton 3.3.1, transformers 4.51.3, tokenizers 0.21.1, safetensors 0.5.3, huggingface_hub 0.30.2, fastapi 0.115.12, starlette 0.46.2, pydantic 2.11.5, httpx 0.28.1, python 3.12.

> ⚠️ These versions were chosen from known-good pairings, not installed/tested. First container build may require bumping one pin (e.g., if a wheel conflicts). That is expected tuning, not drift.

---

## 3. What exists on disk right now

```
cloud-inference-from-scratch/
├── engine_config.json          ✅ all pins + defaults (FR1/FR5 values)
├── pyproject.toml              ✅ empty deps (G2), dev extra = ruff only
├── LICENSE                     ✅ MIT
├── .gitignore                  ✅ artifacts/, .venv/, __pycache__, .modal/
├── artifacts/.gitkeep          ✅
├── modal_app.py                ✅ smoke / benchmark / remote_gpu_tests /
│                                  api_lifecycle_tests / ApiServer(asgi) / main help
├── README.md                   ✅ PRD-ordered quickstart, stages, pending results,
│                                  API examples, limits, teardown, licenses
├── docs/                       ✅ architecture + five executable stage chapters
├── src/cloud_engine/
│   ├── __init__.py             ✅ version
│   ├── config.py               ✅ EngineConfig, build_config, active-limit clamp,
│   │                              finds engine_config.json upward + /root fallback
│   ├── metrics.py              ✅ bounded rolling windows, percentiles, FR11 schema
│   │                              snapshot; reset_runtime() for benchmark runs
│   ├── cache.py                ✅ BlockAllocator (pure python, invariant-checked),
│   │                              ContiguousKVCache, PagedKVCache (pools + block
│   │                              tables + gather accounting), torch optional at import
│   ├── attention.py            ✅ RMSNorm, RoPE build/apply, causal_attention math,
│   │                              AttentionBackend (naive/cached/gathered),
│   │                              TritonDecodeAttentionBackend w/ fail-closed + opt-in fallback
│   ├── model.py                ✅ custom Qwen2 (embedding/RMSNorm/RoPE/GQA/SwiGLU/tied head),
│   │                              StepContext, greedy_sample, dims validation;
│   │                              module names mirror HF for strict load_state_dict
│   ├── weights.py              ✅ explicit key map + shape verify + dup/missing/unexpected
│   │                              errors, tied-lm_head handling, download + tokenizer helpers
│   ├── scheduler.py            ✅ RequestState machine, GenerationConfig guard,
│   │                              RejectedError(QUEUE_FULL|KV_CAPACITY|CONTEXT_OVERFLOW),
│   │                              iteration loop (reap→decode→admit/prefill), FIFO,
│   │                              token budget, queue timeout, slow-consumer watchdog,
│   │                              lossless pending backlog, single _finalize funnel
│   ├── engine.py               ✅ InferenceEngine facade, NaiveRunner/CachedRunner
│   │                              (to_thread off event loop), RequestHandle(stream/wait/cancel)
│   └── api.py                  ✅ pure validate_payload (strict unknown-field/input
│                                  rejection), hmac bearer auth, Responses object builder,
│                                  SSE event sequencer (created→deltas→done→completed),
│                                  create_app() with /v1/responses /healthz /metrics,
│                                  64 KiB body cap, error mapping incl. Retry-After
├── benchmarks/
│   ├── __init__.py             ✅
│   ├── workloads.json          ✅ decode/mixed/fragmentation profiles + word bank + seed
│   └── run.py                  ✅ deterministic prompt builder targeting token counts,
│                                  PRD §12.2 protocol (warmup×2, reset, 3 runs, median +
│                                  all runs), metadata (versions/GPU/hashes), no prompts in output
└── tests/
    ├── test_allocator.py       ✅ 9 cases (exhaust/reuse/double-free/churn invariants)
    ├── test_scheduler.py       ✅ 15 cases (FIFO, limits, budget deferral, overflow,
    │                              queue-full, cancel idempotency, timeout, model failure,
    │                              capacity rejection counting, slow-consumer cancel,
    │                              ordered delivery, max-output cap)
    ├── test_api.py             ✅ validation matrix, auth compare, response schema,
    │                              SSE ordering/sequence-number/delta-concat checks
    ├── test_metrics.py         ✅ percentile, capacity bound, time pruning, reset
    └── remote_gpu_tests.py     ✅ §13.3 suite: logits parity, 5-mode token parity (10 prompts),
                                   paged==contiguous @ [1,15,16,17,127,128,129,2048],
                                   triton==torch-paged @ batch {1,2,8,16}, 16-concurrent leak check,
                                   fault-path zero-block checks, streamed==blocking equality
```

## 4. Local test status (stdlib-only suite, no Torch)

Command: `.venv/bin/python -m unittest discover -s tests -p "test_*.py"`

Latest full run: **40 tests, 40 pass, 0 failures**. `ruff check .`, Python byte-compilation, and `git diff --check` also pass.

The apparent scheduler timeouts were correct backpressure: tests had stopped consuming bounded token queues. Tests now consume streams like real clients. The allocator duplicate-free path now validates the whole operation before mutation.

### Bugs found & already fixed during this session
1. `cache.py` stray trailing-comma `assert` → SyntaxError killing module import.
2. RoPE positions ignored cached prefix length during decode (`kv_start` offset added in `model.forward`).
3. Triton kernel used K-pool block stride for V pool loads → now passes `stride_vn` explicitly.
4. Pending-token backlog could deadlock (nothing re-drained) → `_drain_pending` called in decode loop + publish path.
5. STOP could overtake undelivered tokens at finalization → STOP appended to backlog end, ordering preserved.
6. Cancellation didn't increment its metric → unified counting inside `_finalize` funnel (+ `Request.rejected` flag distinguishes rejected-vs-failed).
7. FakeRunner defaulted to instant-EOS making several scheduler scenarios vacuous → default now generates non-EOS tokens up to the cap.
8. Capacity rejection detection moved from class-name string match to `is_capacity_error` attribute (keeps scheduler decoupled from cache imports, per §8 dependency direction).
9. Cached runners and the scheduler both incremented `tokens_fed`, breaking the second decode step → runner is now the sole owner.
10. `modal_app.py` image construction lacked its closing parenthesis and `app` declaration → import syntax and `modal.App` restored.
11. Triton validation required exact table length despite eager worst-case reservation → now accepts reserved tables with at least the live sequence's blocks.

---

## 5. Design decisions & documented deviations

| Decision | Rationale | Where to document |
|---|---|---|
| Eager worst-case block reservation at admission (`reserve(prompt+max_output)` allocates all blocks up front) | Guarantees no mid-generation OOM without eviction/preemption (both non-goals in v1); makes fragmentation measurable and honest | docs/04 + architecture.md "shortcuts" section |
| Batching is iteration-level; forwards remain per-sequence (B=1 per forward) | Correctness-first, keeps model path readable; aggregate-throughput claim vs contiguous still holds because contiguous serves 1 request at a time | docs/03 |
| `naive`/`contiguous` clamp `max_active_sequences` to 1 | Makes the batched≥1.25× gate meaningful (PRD G3 sequencing) | README stage table + docs/03 |
| STOP/backlog ordering + stall watchdog | Satisfies FR9 (capacity 32, 10 s) without token loss | docs/03 |
| Local tests via stdlib `unittest` (not pytest) | PRD §13.1 "may not require Torch"; zero-dependency promise | README dev section |
| `benchmarks/__init__.py` added (tests kept non-package) | Remote import via `add_local_python_source`; PRD layout otherwise intact | repo map note |
| `create_app()` imports FastAPI lazily; validation/auth/SSE logic pure | Local testability without web stack | docs/architecture |
| Deployed server mode fixed to `triton` server-side | PRD §14 clients cannot choose mode | README deployment |

---

## 6. What's left (ordered)

### A. Remote verification (requires user go-ahead — BILLABLE)
Nothing cloud-related has been executed; no Modal objects exist yet. Sequence per PRD §21.6:
1. User confirms Modal auth + cost permission (`modal setup` done client-side).
2. `modal volume create cloud-inference-model-cache` (or let smoke auto-create).
3. `modal run modal_app.py::smoke` — builds image (first build downloads ~2–3 GB CUDA wheels into Modal's registry cache), downloads ~1 GB weights into the Volume, runs contiguous-mode generation, compares vs HF oracle, prints TTFT/throughput/peak-mem.
4. Fix whatever the first real run shakes out (likely candidates: a dependency pin conflict, transformers generate() kwarg differences for greedy, triton stride/num_warps tuning, fp16 argmax flip near-ties → mitigate by computing lm_head in fp32, already implemented).
5. `modal run modal_app.py::remote_gpu_tests` — full §13.3 matrix green.
6. `modal run modal_app.py::api_lifecycle_tests` — CPU-container suite green (same unittest discovery inside image where fastapi/httpx exist).
7. `modal deploy modal_app.py` + secret create → manual SSE/non-SSE curl verification against deployed URL; confirm scale-to-zero + max_containers=1.

### B. Benchmarks & gates (~1 GPU hour)
8. Run all 5 modes × decode profile + batched/paged × mixed/fragmentation per §12.2; save JSONs via `--output`.
9. Evaluate publication gates (§12.4): contiguous>naive; batched≥1.25×contiguous@8; paged −40% reserved-unused vs batched on fragmentation; triton ≥ paged median with zero gather temp. Any miss ⇒ honest "Known limitations" entry, no workload tuning.
10. Fill measurement tables in chapters/README from artifacts; record `source_revision` env at deploy for reproducibility metadata.

### C. Launch-package leftovers (PRD §20, post-gates)
11. Terminal recording naive-vs-batched; verify ten-minute quickstart on a clean account; acceptance checklist walk (§17 items 1–20).

---

## 7. Cost / cloud state right now

- **$0 spent, nothing created**: no Volume, no Secret, no app, no image built, no GPU seconds used.
- First billable action will be the smoke run (image build + weight download ≈ few minutes of L4 + storage).
- Cost controls already encoded: `max_containers=1`, `min_containers=0`, `buffer_containers=0`, `scaledown_window=60`, timeout 1200 s, `@modal.concurrent(max_inputs=32)`, no schedules, no auto-benchmarks.

## 8. Handy commands

```bash
# local (already set up)
cd "cloud-inference-from-scratch"
.venv/bin/python -m unittest discover -s tests -p "test_*.py"   # stdlib-only suite

# remote (needs modal setup + cost OK)
modal run modal_app.py::main --command help
modal run modal_app.py::smoke
modal run modal_app.py::benchmark --mode naive   # ...contiguous|batched|paged|triton
modal run modal_app.py::remote_gpu_tests
modal run modal_app.py::api_lifecycle_tests
modal secret create cloud-inference-api ENGINE_API_KEY=<value>
modal deploy modal_app.py

# teardown
modal app list && modal app stop <app-name>
modal volume get cloud-inference-model-cache     # deleting removes cached weights
```
