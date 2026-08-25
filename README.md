# Cloud Inference Engine Lab

Build the engine behind an LLM API without owning a GPU.

> **Current release — Ragged L4 Engine:** Qwen2.5-3B, real packed
> multi-request forwards, decode-first scheduling, chunked prefill,
> demand-paged KV with prefix reuse and recompute preemption, and mixed ragged
> Triton attention.
> Verified against serial Triton and vLLM on the same NVIDIA L4.

This repository is an educational, from-scratch inference server for a pinned
`Qwen/Qwen2.5-3B` base model. It implements a flat multi-request transformer
forward, decode-first token-budget scheduling, chunked prefill, transactional
demand-paged KV allocation, bounded prefix reuse, recompute preemption, mixed ragged Triton attention,
and authenticated Responses JSON/SSE delivery. Heavy execution runs on one
serverless NVIDIA L4 through Modal; the local package intentionally has no ML
runtime dependencies.

> This is tested systems code, not a production service or a replacement for
> vLLM. The five-stage 0.5B baseline and its failed optimization gates remain
> published; the sixth stage fixes their architectural causes and is measured
> against pinned vLLM without hiding the remaining performance gap.

## Release status

Verified on 2026-08-26:

- 53/53 dependency-light local tests passed.
- 53/53 tests plus real FastAPI route/auth integration passed in Modal's CPU image.
- 34/34 legacy L4 regression checks passed in 208.3 seconds.
- 20/20 Qwen2.5-3B Ragged L4 checks passed in 66.5 seconds: exact HF tokens,
  four request IDs in one transformer invocation, 4,000-token chunked prefill,
  decode priority, real pressure recomputation, prefix-hit parity/work reduction,
  and zero leaked request blocks.
- Ragged Triton matched the Torch oracle at batches 1/2/4/8/16 and contexts
  through 4,002 tokens; worst observed absolute difference was 0.00195. The
  serial Triton kernel also matched at head dimension 128 and context 4,096.
- Nine fixed-protocol benchmark artifacts were recorded and committed.
- A three-way warm HTTP sweep completed 225/225 raw requests without error on
  the same pinned 3B revision and workload hash for serial Triton, Ragged, and
  vLLM 0.10.0.
- The deployed Ragged API passed public readiness, unauthenticated rejection,
  authenticated blocking generation, ordered SSE with token usage, and
  authenticated metrics checks against the external production URL.
- NVIDIA AIPerf 0.12.0 completed 60/60 measured Responses-API requests across
  three runs with zero errors, plus two excluded warmups per run.
- NVIDIA Nsight Systems 2026.1.3 captured and validated the engine's NVTX phase
  ranges on L4; a companion PyTorch trace contains 13,667 CUDA kernel records.
- A 120-second, concurrency-eight L4 reliability soak completed 1,936 requests:
  1,760 generated, 176 deliberately cancelled, zero failed, zero leaked request
  blocks, 1,752 prefix hits, packed batches of eight, and a clean engine restart.

## What “from scratch” means here

The engine uses PyTorch tensor primitives and the official tokenizer and
safetensor weights. It does **not** call vLLM, SGLang, Hugging Face `generate()`,
or Hugging Face `past_key_values` in its serving path. The custom implementation
contains embedding lookup, RMSNorm, RoPE, grouped-query attention, SwiGLU,
residual connections, the tied output projection, cache writes, scheduling,
streaming, and the Triton kernel. Hugging Face model execution exists only as a
correctness oracle in smoke and remote GPU tests.

## Architecture in 90 seconds

```text
client
  |
  v
FastAPI auth + strict validation
  |
  v
InferenceEngine.submit
  |
  v
decode-first Scheduler ---- bounded per-request queues ----> JSON or ordered SSE
  |
  v
custom Qwen2 forward
  |
  +-- naive:      recompute the whole sequence
  +-- contiguous: per-request contiguous KV tensors
  +-- batched:    iteration-level multi-request scheduling
  +-- paged:      shared block pool + torch logical gather
  +-- triton:     shared block pool + direct-block decode kernel
  +-- ragged:     BatchPlan -> PackedBatch -> one mixed Triton forward
```

`InferenceEngine` is the composition root. The scheduler alone owns request
state, FIFO admission, token/sequence limits, backpressure, timeout, and terminal
cleanup. Every terminal path releases cache ownership, closes the stream in
order, and resolves the request future. Paged modes keep one authoritative
physical KV pool—there is no shadow cache.

See [the architecture walkthrough](docs/architecture.md) and the stage chapters:
[naive](docs/01-naive.md), [contiguous KV](docs/02-kv-cache.md),
[continuous batching](docs/03-continuous-batching.md),
[paged KV](docs/04-paged-kv.md), [Triton attention](docs/05-triton-attention.md),
and [packed Ragged inference](docs/06-ragged-engine.md).
The editable [Excalidraw architecture](system-architecture.excalidraw.json)
can be copied into Excalidraw as clipboard JSON.

## The six stages

| Mode | Change from the prior stage | Decode cache read path | Active requests |
|---|---|---|---:|
| `naive` | Full-sequence recomputation | None retained between steps | 1 |
| `contiguous` | Prefill once, append one token | Per-request contiguous tensors | 1 |
| `batched` | FIFO continuous scheduler | Per-request contiguous tensors | Up to 16 |
| `paged` | Shared 16-token block allocator | Temporary torch gather | Up to 16 |
| `triton` | Direct physical-block attention | Triton kernel; no decode gather | Up to 16 |
| `ragged` | Flat multi-request forward + demand paging | Mixed prefill/decode Triton kernel | Up to 16 |

The naïve path normally recomputes the full sequence. FP16 GEMM shape changes can
flip an argmax when the two best logits are nearly tied, so ambiguous top-two
decisions are replayed through a temporary contiguous cache and immediately
discarded. No KV state survives between naïve decode steps. This small,
documented parity guard preserves the instructional baseline while keeping all
five greedy outputs deterministic on the tested L4.

## Ragged L4 Engine

The older scheduler rotates across requests but still awaits one complete
`runner.step(request)` model forward per request. Paged admission also reserves
`prompt + max_output_tokens` capacity up front. The shipped `ragged` mode
replaces that execution seam rather than relabeling it.

```text
waiting + running requests
          |
          v
decode-first token-budget scheduler
          |
          v
BatchPlan(request, query length, context length, sample flag)
          |
          v
transactional demand-paged KV allocation
          |
          v
PackedBatch(input IDs, positions, query offsets, block tables, slot mapping)
          |
          v
one flat Qwen2.5-3B model forward per iteration
          |
          v
ragged paged attention: Torch reference -> L4 Triton kernel
          |
          v
ordered tokens, metrics, and terminal cleanup
```

The verified target is one pinned `Qwen/Qwen2.5-3B` revision in FP16,
one NVIDIA L4, a 4,096-token project context limit, up to 16 active sequences,
and deterministic greedy generation. The 0.5B model remains the fast regression
oracle.

The implementation adds four concrete runtime contracts:

- `BatchPlan`: one iteration's per-request token work under a shared token budget;
- `PackedBatch`: one flat token axis plus ragged positions, offsets, sequence
  lengths, block tables, slot mappings, and selected logits positions;
- transactional KV growth: allocate only blocks crossed by scheduled tokens,
  with atomic failure and recompute preemption under pressure; and
- one integrated paged-attention path for mixed decode and chunked-prefill work,
  using device-resident metadata and no reconstructed logical K/V tensors.

Evidence for the acceptance gates:

1. A traced model invocation contains multiple request IDs; the transformer is
   never executed once per scheduled request.
2. Mixed packed execution matches the sequential oracle token-for-token.
3. A 4,096-token prompt advances over real prefill chunks without blocking a
   continuously runnable decode for more than one scheduler iteration.
4. Repeated prompts reuse block-aligned KV prefixes while recomputing the final
   prompt token, preserving exact first-token and generated-token parity.
5. An intentionally undersized KV pool causes non-zero preemptions and
   recomputation without OOM, deadlock, token drift, double-free, or leaked blocks.
6. Torch and Triton ragged attention match over batch sizes `1/2/4/8/16`, block
   boundaries, mixed query lengths, and contexts through 4,096 tokens.
7. Warm HTTP arrival-rate sweeps report TTFT, ITL, E2E p50/p95/p99, throughput,
   SLO goodput, errors, queue depth, preemptions, and raw per-request results.
8. The same protocol compares the old serial Triton baseline, the new ragged
   engine, and pinned vLLM 0.10.0 on the same L4 and model revision.

Gates 1–6 passed in the 20-check L4 suite. Gate 7 passed in the authenticated
225-request HTTP sweep retained per request in the three-way artifact. Gate 8 uses
the same workload hash, source identity, 3B revision, L4, active-sequence cap,
token budget, greedy decoding, and EOS behavior across all implementations.

KV offload, CUDA graphs, quantization, speculative decoding,
multi-GPU execution, broader APIs, and additional model families remain
deliberately deferred beyond this packed/demand-paged milestone.

## Quickstart: correctness on a cloud GPU

For a recording-ready walkthrough of the deployed engine:

```bash
modal run demo.py
```

To change scheduling order, capacity-pressure victim selection, or safe batching
knobs and measure them against the production default, copy the
[starter experiment](experiments/starter.py) and run:

```bash
modal run experiment.py --experiment experiments/starter.py
```

This refuses to benchmark unless token parity, packed execution, and KV cleanup
pass first, then writes raw and compact baseline/experiment results to
`artifacts/`. The committed [compact starter result](artifacts/experiment-short-prefill-first-summary.json)
records 70/70 requests without error across baseline and experiment sweeps.

For standardized endpoint load data and GPU timelines:

```bash
modal run nvidia_aiperf.py
modal run nvidia_profile.py
modal run reliability.py --duration-seconds 120
```

The first command writes NVIDIA AIPerf's native multi-run records. The second
writes an Nsight `.nsys-rep`, a PyTorch CUDA trace, checksums, source identity,
and profiler capability flags. All three refuse to publish an artifact when
their workload or report validation fails. The reliability runner additionally
injects deterministic cancellations and rebuilds the engine after timed load.

The script reads the existing API key from the Modal Secret, then demonstrates
readiness, rejected unauthenticated access, blocking generation, live SSE text,
ordered events, token usage, and engine metrics without printing the key.

Prerequisites:

- Python 3.10 or newer
- a Modal account and authenticated CLI
- Git
- permission to incur serverless GPU/storage charges

```bash
git clone https://github.com/zeeshan8281/cloud-inference-from-scratch.git
cd cloud-inference-from-scratch

# Install the lightweight CLI outside the project environment.
uv tool install modal            # or: python3 -m pip install --user modal
modal setup

# The Volume is also created automatically if absent.
modal volume create cloud-inference-model-cache

# Builds the image, downloads the pinned 3B model remotely, and checks packing/HF parity.
modal run modal_app.py::ragged_smoke
```

The first run downloads the pinned 3B model snapshot into the Modal Volume.
Weights, Torch, CUDA work, and Triton compilation stay in Modal; they are not
downloaded to your Mac. Later runs reuse the image and Volume caches. Ephemeral
containers scale to zero after 60 seconds.

## Reproducible benchmark results

### NVIDIA AIPerf Responses-API profile

The committed [native AIPerf artifact](artifacts/aiperf-responses.zip) uses
AIPerf 0.12.0 against the deployed streaming `/v1/responses` endpoint: three
profile runs, 20 requests per run, concurrency four, deterministic synthetic
128-token inputs, 32 requested output tokens, and two excluded warmups per run.
All 60 measured requests completed with exact server-reported token counts and
zero errors.

| Aggregate across three runs | Mean | Range |
|---|---:|---:|
| TTFT p50 | 705.9 ms | 702.6–711.9 ms |
| TTFT p99 | 1,109.6 ms | 1,085.1–1,125.2 ms |
| ITL p50 | 72.18 ms | 72.14–72.24 ms |
| ITL p99 | 76.43 ms | 72.75–78.30 ms |
| Request latency p99 | 3,345.8 ms | 3,322.9–3,364.4 ms |
| Output throughput | 41.08 tok/s | 40.83–41.22 tok/s |
| Request throughput | 1.284 req/s | 1.276–1.288 req/s |

The archive contains all per-request JSONL records, input metadata, per-run and
aggregate JSON/CSV summaries, logs, AIPerf version, model revision, and source
tree identity. The API accepts AIPerf's standard text-message input shape; it
still rejects multimodal and non-user message blocks.

### NVIDIA GPU profiling

The committed [profiling artifact](artifacts/nsight-ragged-l4.zip) contains:

- an NVIDIA Nsight Systems 2026.1.3 `.nsys-rep` with validated NVTX ranges for
  batch planning, batch execution, mixed/decode forwards, KV growth/commit, and
  greedy sampling; and
- a PyTorch/CUPTI Chrome trace with 13,667 CUDA kernel records from the same
  four-request, 32-output-token packed L4 workload.

Modal's gVisor container exposed NVTX ranges but no CUDA kernel records to
Nsight itself, and the artifact says so (`nsight_cuda_kernel_records: false`).
The companion trace supplies the kernel timeline
(`pytorch_cuda_kernel_records: true`). On an NVIDIA host such as DGX Spark,
the same opt-in ranges are available with `ENGINE_NVTX=1` for native Nsight
capture; DGX Spark execution remains unverified until run on that hardware.

### Reliability soak

The committed [L4 soak artifact](artifacts/reliability-soak-l4.json) records a
120.2-second run at concurrency eight. It issued 1,936 requests: 1,760 completed
generation, 176 were deliberately cancelled, and none failed. The run produced
14,080 output tokens at 16.11 issued requests/second, exercised packed batches
of eight and 1,752 prefix hits, drained to zero request-owned KV blocks, asserted
allocator invariants, then constructed a fresh engine and completed another
eight-token request without a leak.

### Ragged 3B online comparison

The current release uses warm authenticated streaming HTTP, fixed interval
arrivals at 0.5/1/2/4 requests per second, ten seconds per rate, 16–256-token
prompts, and 32 requested output tokens. Serial Triton, Ragged, and vLLM use the
same pinned Qwen2.5-3B revision, NVIDIA L4, FP16, 4,096-token context, 16 active
sequences, 2,048 scheduled tokens, temperature zero, and EOS behavior. The raw
[JSON artifact](artifacts/ragged-vllm-online.json) includes every request and the
prompt-covering workload hash; it does not include prompt text.

Prefix caching is disabled in both vLLM and Ragged for this historical comparison
because prompts repeat across arrival-rate sweeps; otherwise later rates would
measure cache hits instead of the same inference work. The deployed Ragged engine
now enables its separately verified bounded prefix cache.

The custom engines use the configured 4 GiB KV pool. vLLM uses its normal 0.85
GPU-memory-utilization policy (12.76 GiB KV reported in this environment). The
workload does not pressure either pool, but this is a throughput/latency
comparison, not a memory-normalized capacity comparison.

vLLM is the performance reference, not a dependency of the custom serving path.

| Arrival | Engine | TTFT p99 | ITL p99 | E2E p99 | Output tok/s | SLO goodput | Errors | Queue max |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 req/s | serial Triton | 83.6 ms | 65.7 ms | 1,900.9 ms | 13.6 | 0.618 req/s | 0 | 0 |
|  | Ragged | 63.4 ms | 54.3 ms | 1,928.9 ms | 13.6 | 0.495 req/s | 0 | 1 |
|  | vLLM 0.10.0 | 100.3 ms | 30.2 ms | 920.9 ms | 13.7 | 0.621 req/s | 0 | n/a |
| 1 req/s | serial Triton | 255.4 ms | 313.5 ms | 7,109.2 ms | 16.6 | 0.208 req/s | 0 | 1 |
|  | Ragged | 108.0 ms | 60.7 ms | 1,779.9 ms | 22.3 | 0.933 req/s | 0 | 1 |
|  | vLLM 0.10.0 | 73.1 ms | 30.0 ms | 918.7 ms | 24.1 | 1.010 req/s | 0 | n/a |
| 2 req/s | serial Triton | 531.8 ms | 754.8 ms | 14,025.6 ms | 16.7 | 0.538 req/s | 0 | 1 |
|  | Ragged | 127.7 ms | 63.6 ms | 1,859.2 ms | 32.4 | 2.086 req/s | 0 | 1 |
|  | vLLM 0.10.0 | 72.2 ms | 34.0 ms | 940.4 ms | 32.6 | 2.095 req/s | 0 | n/a |
| 4 req/s | serial Triton | 13,231.2 ms | 1,163.6 ms | 27,083.3 ms | 16.7 | 0.624 req/s | 0 | 3 |
|  | Ragged | 592.6 ms | 87.2 ms | 2,450.0 ms | 49.3 | 3.432 req/s | 0 | 2 |
|  | vLLM 0.10.0 | 72.1 ms | 36.0 ms | 979.3 ms | 53.5 | 3.825 req/s | 0 | n/a |

At 4 req/s, Ragged delivered **2.96×** the serial engine's output throughput
and **92.0%** of vLLM's while keeping TTFT and ITL p99 inside the stated SLO.
Its p99 TTFT was still 8.2× vLLM and p99 E2E was 2.5× vLLM, so the result proves
real packed scheduling—not production parity. All 225 measured requests
completed without error. The 4 GiB online pool did not preempt; non-zero
preemption and recomputation are proven separately by the 5 MiB pressure test.
vLLM queue depth is `n/a` because its HTTP API did not expose an equivalent
per-sweep queue sample; the artifact records `null`, not a fabricated zero.

### Historical five-stage 0.5B matrix

Environment: one NVIDIA L4, FP16, Python 3.12, PyTorch 2.7.1, Triton 3.3.1,
Transformers 4.51.3, model revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`. Each artifact uses two unmeasured
warmups followed by three recorded runs; the table reports medians and the JSON
retains all runs. Workloads are deterministically generated from a public word
bank. Artifacts store counts and a prompt-covering SHA-256 hash, never prompt text.

The measured source revision is
[`1480bb49`](https://github.com/zeeshan8281/cloud-inference-from-scratch/commit/1480bb49e5e2be2460726b1fa145f70241657459).
Later commits changed API request injection, terminal metrics reporting, tests,
and docs—not the measured model, kernel, generation path, or workload.

### Fixed decode profile

Eight concurrent requests, approximately 32 prompt tokens each, requesting 128
output tokens each:

| Mode | Output tok/s | TTFT p50 | E2E p50 | Peak allocated GPU | Peak KV unused | Raw runs |
|---|---:|---:|---:|---:|---:|---|
| `naive` | 16.4 | 12,973.7 ms | 19,030.3 ms | 1.46 GiB | n/a¹ | [JSON](artifacts/naive-decode.json) |
| `contiguous` | 27.5 | 9,276.0 ms | 11,516.0 ms | 1.46 GiB | 1.83 MiB | [JSON](artifacts/contiguous-decode.json) |
| `batched` | 21.9 | 230.0 ms | 16,416.2 ms | 1.48 GiB | 9.00 MiB | [JSON](artifacts/batched-decode.json) |
| `paged` | 19.5 | 249.9 ms | 18,432.4 ms | 5.46 GiB | 9.28 MiB | [JSON](artifacts/paged-decode.json) |
| `triton` | 21.4 | 262.6 ms | 16,893.7 ms | 5.46 GiB | 9.28 MiB | [JSON](artifacts/triton-decode.json) |

¹ The artifact can briefly observe the naïve parity guard's temporary cache; it
is not persistent KV capacity and is excluded from the stage comparison.

### Mixed and fragmentation profiles

| Mode | Profile | Output tok/s | TTFT p50 | E2E p50 | Peak KV unused | Raw runs |
|---|---|---:|---:|---:|---:|---|
| `batched` | mixed | 24.9 | 365.9 ms | 410.7 ms | 6.71 MiB | [JSON](artifacts/batched-mixed.json) |
| `paged` | mixed | 19.0 | 457.2 ms | 511.2 ms | 6.36 MiB | [JSON](artifacts/paged-mixed.json) |
| `batched` | fragmentation | 21.7 | 418.7 ms | 59,719.6 ms | 22.93 MiB | [JSON](artifacts/batched-fragmentation.json) |
| `paged` | fragmentation | 19.0 | 502.1 ms | 69,081.9 ms | 23.95 MiB | [JSON](artifacts/paged-fragmentation.json) |

### Optimization gates

| Gate | Required | Measured | Result |
|---|---:|---:|---|
| Contiguous cache vs naïve decode throughput | `>1.00×` | `1.68×` | **Pass** |
| Batched vs contiguous decode throughput | `>=1.25×` | `0.80×` | **Fail** |
| Paged reduction in unused KV on fragmentation | `>=40%` | `-4.45%` (worse) | **Fail** |
| Triton vs torch-paged decode throughput | `>=1.00×` | `1.10×` | **Pass** |

The failures explain the next engineering work. “Batched” schedules many requests
but still launches one model forward per sequence, improving TTFT while losing
aggregate throughput to launch overhead. “Paged” eagerly reserves the full
prompt-plus-maximum-output block table, so block rounding makes unused capacity
slightly worse than exact contiguous allocation. Dynamic block growth and true
tensor batching are required before claiming those benefits.

The paged reference gathered 567.7 MiB of temporary K/V over the decode run. The
Triton run gathered only 2.63 MiB for prefill; decode reads physical blocks
directly. Triton therefore removed the full-cache decode gather while improving
median throughput by 9.7%.

Run or regenerate the matrix:

```bash
modal run modal_app.py::benchmark --mode naive --profile decode --output artifacts/naive-decode.json
modal run modal_app.py::benchmark --mode contiguous --profile decode --output artifacts/contiguous-decode.json
modal run modal_app.py::benchmark --mode batched --profile decode --output artifacts/batched-decode.json
modal run modal_app.py::benchmark --mode paged --profile decode --output artifacts/paged-decode.json
modal run modal_app.py::benchmark --mode triton --profile decode --output artifacts/triton-decode.json

modal run modal_app.py::benchmark --mode batched --profile mixed --output artifacts/batched-mixed.json
modal run modal_app.py::benchmark --mode paged --profile mixed --output artifacts/paged-mixed.json
modal run modal_app.py::benchmark --mode batched --profile fragmentation --output artifacts/batched-fragmentation.json
modal run modal_app.py::benchmark --mode paged --profile fragmentation --output artifacts/paged-fragmentation.json
```

## Correctness and failure-path verification

```bash
# No Torch, FastAPI, or GPU needed locally.
uv sync --extra dev
uv run python -m unittest discover -s tests -p 'test_*.py' -v
uv run ruff check .

# Same suite plus real FastAPI construction/auth in the Modal CPU image.
modal run modal_app.py::api_lifecycle_tests

# Billable L4: model/oracle/cache/kernel/concurrency/failure/stream checks.
modal run modal_app.py::remote_gpu_tests
modal run modal_app.py::remote_ragged_gpu_tests

# Billable three-way warm HTTP comparison; writes raw JSON locally.
modal run modal_app.py::online_compare --rates 0.5,1,2,4 --duration-seconds 10
```

The GPU suite covers:

- custom-model logits against Hugging Face on three prompts (`rtol=2e-2`,
  `atol=2e-2`; observed maximum absolute difference 0.0078);
- exact greedy token equality across all five modes for 10 prompts up to 32 tokens;
- exact cached-engine token equality against Hugging Face generation for 10 prompts;
- paged versus contiguous outputs at lengths 1, 15, 16, 17, 127, 128, 129, and 2,048;
- Triton versus torch-paged outputs at batch sizes 1, 2, 8, and 16;
- 16 concurrent completions followed by zero owned blocks;
- paged and Triton cancellation/timeout paths followed by zero owned blocks; and
- streamed token IDs/text/order exactly equal to blocking generation.

The allocator and scheduler tests cover exhaustion, ownership, atomic invalid
free, FIFO admission, queue/token/context limits, backpressure, slow consumers,
idempotent cancellation, injected decode failure, timeout, and terminal cleanup.

The Ragged suite additionally covers one real multi-request forward, exact 3B
HF token parity, transactional growth, 256-token prefill chunks, decode-first
ordering, recompute preemption under a 5 MiB pool, mixed-query Triton/Torch
parity through 4,096 tokens, and post-run allocator invariants.

## Deploy the API

Generate a strong key, store it in Modal, and deploy the server-fixed `ragged`
mode. Do not commit or paste the key into source files.

```bash
ENGINE_API_KEY="$(openssl rand -hex 32)"
modal secret create cloud-inference-api ENGINE_API_KEY="$ENGINE_API_KEY" --force
modal deploy modal_app.py

# Copy the Web Function URL printed by Modal.
export ENGINE_URL='https://YOUR-WORKSPACE--cloud-inference-lab-apiserver-serve.modal.run'
```

Modal documents its [Secret CLI](https://modal.com/docs/cli/latest/secret),
[deployed Web Functions](https://modal.com/docs/guide/webhooks), and
[deployment lifecycle](https://modal.com/docs/guide/managing-deployments).

### Blocking response

```bash
curl -sS "$ENGINE_URL/v1/responses" \
  -H "Authorization: Bearer $ENGINE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Qwen/Qwen2.5-3B",
    "input":"Explain a KV cache in one sentence.",
    "max_output_tokens":64,
    "temperature":0,
    "stream":false
  }' | jq
```

### Streaming response

```bash
curl -sS -N "$ENGINE_URL/v1/responses" \
  -H "Authorization: Bearer $ENGINE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Qwen/Qwen2.5-3B",
    "input":"Explain paged attention in one sentence.",
    "max_output_tokens":64,
    "temperature":0,
    "stream":true
  }'
```

SSE ordering is:

```text
response.created
response.output_text.delta  (zero or more, sequence numbers strictly increase)
response.output_text.done
response.completed
data: [DONE]
```

Concatenated deltas equal both `response.output_text.done.text` and the final
response text. The completed event contains input, output, and total token usage.

### Endpoint and request contract

| Endpoint | Auth | Behavior |
|---|---|---|
| `GET /healthz` | Public | readiness, pinned model, fixed mode |
| `GET /metrics` | Bearer | bounded counters, latency, scheduler, KV, GPU gauges |
| `POST /v1/responses` | Bearer | blocking JSON or SSE generation |

Accepted request fields are `model`, `input`, optional `instructions`,
`max_output_tokens` (1–256), `temperature` (must be `0`), `stream` (boolean),
and `stream_options: {"include_usage": true}`. `input` may be a non-empty string
or AIPerf/OpenAI-style user text messages containing `input_text` blocks.
Unknown fields, multimodal/non-user blocks, the wrong model, nonzero temperature,
invalid JSON, bodies over 64 KiB, and context overflow are rejected. Clients
cannot select engine mode, model revision, cache size, hardware, or fallback
behavior.

Expected operational errors include `401 authentication_failed`,
`400 invalid_request`/`context_length_exceeded`, `429 queue_full`, and
`503 capacity_exhausted` with `Retry-After: 1`.

## Security model

- The app refuses startup when `ENGINE_API_KEY` is missing.
- Bearer tokens are compared with `hmac.compare_digest`.
- Generation and metrics require auth; only readiness is public.
- FastAPI interactive docs are disabled.
- Request bodies are capped at 64 KiB before JSON parsing.
- Prompts, output text, bearer values, and keys are absent from benchmark artifacts.
- Runtime mode, revision, hardware, and reference fallback are server-controlled.
- Triton validation fails closed on unsupported dtype, shape, block size, batch,
  or context unless reference fallback is explicitly enabled in non-deployed code.

This is still a single shared API key, not tenant isolation. For an internet-facing
service, add Modal proxy auth or an identity-aware gateway, per-user quotas,
rate limits, audit policy, abuse monitoring, and key rotation.

## Metrics

Authenticated `/metrics` returns bounded in-memory data:

- request completion/failure/cancellation/rejection/timeout counters;
- TTFT, inter-token latency, and E2E p50/p95;
- input/output token counters and 60-second throughput;
- scheduler iteration and batch-size gauges;
- cache kind, request/prefix blocks, prefix hits/misses, utilization,
  reserved/occupied/fragmentation/gather bytes; and
- CUDA allocated, reserved, and peak-allocated bytes.

Latency and scheduler samples use fixed-capacity, time-pruned windows; they do not
grow for the life of the container. Metrics reset when a scale-to-zero container
is replaced and are not a durable observability backend.

## Cost and local disk footprint

As checked on 2026-08-24, Modal lists L4 compute at `$0.000222/s` (about
`$0.80/hour`), physical CPU at `$0.0000131/core/s`, memory at
`$0.00000222/GiB/s`, and Volume storage at `$0.09/GiB/month`; prices and free
credits can change, so check [Modal pricing](https://modal.com/pricing) before
running. The published nine-workload matrix used roughly 20–22 successful L4
minutes, or about `$0.27–$0.29` in GPU time alone. Image builds, CPU, memory,
retries, tests, and storage are additional.

Cost controls in `engine_config.json`:

- exactly one L4 container maximum;
- zero minimum and zero buffer containers;
- 60-second scale-down window;
- 20-minute per-call timeout;
- no schedule and no benchmark-on-deploy; and
- roughly 6 GB of persistent 3B model data, plus the retained 0.5B oracle.

On the development Mac, the measured working tree was 28 MB including a 27 MB
virtual environment; the Modal CLI occupied 32 MB. The shared uv cache was 1.5
GB but belonged to all uv projects. Model weights remain remote. A clean clone
without a virtual environment is much smaller.

## Stop compute and delete cloud state

```bash
# Stop the deployed app; it otherwise already scales to zero when idle.
modal app stop cloud-inference-lab

# Inspect before deleting. Deletion removes cached weights and compiled artifacts.
modal volume get cloud-inference-model-cache
modal volume delete cloud-inference-model-cache

# Rotate or remove the bearer secret separately.
modal secret delete cloud-inference-api
```

The next run recreates the Volume and downloads the public model again.

## Repository map

```text
modal_app.py                  Modal image, jobs, deployment, cost controls
demo.py                       one-command deployed API walkthrough
experiment.py                 correctness-gated scheduler experiment runner
nvidia_aiperf.py              NVIDIA AIPerf Responses-API load runner
nvidia_profile.py             Nsight NVTX + CUDA-kernel trace runner
reliability.py                concurrent cancellation/restart L4 soak
engine_config.json           model/dependency/scheduler/cache/cloud pins
src/cloud_engine/model.py    custom Qwen2 forward path
src/cloud_engine/weights.py  explicit safetensor mapping and shape validation
src/cloud_engine/cache.py    contiguous cache, paged pool, block allocator
src/cloud_engine/attention.py torch backends and Triton dispatch
src/cloud_engine/kernel.py   serial decode and mixed ragged Triton attention
src/cloud_engine/scheduler.py lifecycle, BatchPlan, priority, preemption
src/cloud_engine/engine.py   runners, PackedBatch, engine facade and handles
src/cloud_engine/api.py      validation, auth, Responses JSON, SSE
src/cloud_engine/metrics.py  bounded measurements
benchmarks/                  deterministic offline and online HTTP workloads
artifacts/                   raw benchmarks, AIPerf records, GPU traces
tests/                       local/CPU tests and L4 correctness matrix
docs/                        architecture and one chapter per stage
```

## Current-release limitations

The deployed release supports one pinned 3B base-model revision, FP16, a single
L4, greedy text-only generation, a 4,096-token project context limit, and a
narrow Responses-style API. It has no sampling, instruction/chat template,
quantization, cache swapping/offload, CUDA graphs, fused non-
attention kernels, speculative decoding, tensor parallelism, multi-GPU support,
persistent metrics, tenant isolation, or user system.

It is a real packed inference engine with bounded scope, not a production stack.
The vLLM comparison quantifies the remaining kernel/runtime optimization gap.

## Troubleshooting

- **Cold request is slow:** a scale-to-zero call must start the L4 container and
  load weights. Retry only after allowing a few minutes for the first request.
- **401:** verify the exact `ENGINE_API_KEY` in the named Modal Secret and bearer
  header; redeploy after rotating a secret if an existing container stays warm.
- **422 asking for a `request` query parameter:** update to the release containing
  the FastAPI request-injection fix and redeploy.
- **`capacity_exhausted`:** reduce concurrency or context size. Ragged demand
  paging can preempt and recompute active work, but cannot admit a request whose
  own context cannot fit the physical pool.
- **Container preempted:** Modal can transparently retry ephemeral runs. Benchmark
  artifacts are written only after one complete two-warmup/three-run attempt.
- **Triton refuses a shape:** this is fail-closed behavior. Use the pinned model,
  FP16, block size 16, head dimension 64 or 128, and contexts no longer than 4,096.

## References and license

The design is informed by the
[PagedAttention paper](https://arxiv.org/abs/2309.06180),
[Triton](https://triton-lang.org/), the
[Qwen2.5-3B model](https://huggingface.co/Qwen/Qwen2.5-3B), and
[Modal GPU/Web Function documentation](https://modal.com/docs/guide/gpu).

Repository code is [MIT licensed](LICENSE). Runtime components and downloaded
weights retain their own licenses:

| Component | License |
|---|---|
| Qwen/Qwen2.5-0.5B and 3B weights/tokenizers | Apache-2.0 |
| PyTorch | BSD-3-Clause |
| Triton | MIT |
| Transformers, Tokenizers, safetensors, huggingface_hub | Apache-2.0 |
| FastAPI, Starlette, Pydantic, HTTPX | Their published MIT/BSD licenses |
| Modal Python client | Apache-2.0 |

Contributions should keep the comparison honest: run the local suite, attach a
correctness case for behavior changes, attach raw JSON for performance claims,
and never weaken or selectively omit a workload to make a gate pass.
