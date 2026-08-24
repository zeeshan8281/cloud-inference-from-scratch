# Cloud Inference Engine Lab

Build the engine behind an LLM API without owning a GPU.

This repository is an educational, from-scratch inference server for the pinned
`Qwen/Qwen2.5-0.5B` base model. It implements the model forward pass, greedy
decoding, KV caching, continuous request scheduling, paged KV allocation,
Responses-style JSON/SSE delivery, and a direct-block Triton decode-attention
kernel. Heavy execution runs on one serverless NVIDIA L4 through Modal; the
local package intentionally has no ML runtime dependencies.

> This is tested educational systems code, not a production service or a
> replacement for vLLM. Two intended optimization gates failed, and the results
> below show them rather than hiding them.

## Release status

Verified on 2026-08-24:

- 45/45 dependency-light local tests passed.
- 45/45 tests plus real FastAPI route/auth integration passed in Modal's CPU image.
- 34/34 L4 correctness checks passed in 197.3 seconds on the documented release commit.
- All five modes produced exactly the same greedy tokens for 10 prompts, and the
  cached baseline matched Hugging Face `generate()` for all 10.
- Triton matched the torch-paged reference through context length 2,048 and
  batches 1, 2, 8, and 16; worst observed absolute difference was below 0.003.
- Nine fixed-protocol benchmark artifacts were recorded and committed.
- A deployed Triton API passed public health, unauthenticated rejection,
  authenticated blocking generation, SSE sequencing/content, token accounting,
  and authenticated metrics checks.

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
FIFO Scheduler ---- bounded per-request queues ----> JSON or ordered SSE
  |
  v
custom Qwen2 forward
  |
  +-- naive:      recompute the whole sequence
  +-- contiguous: per-request contiguous KV tensors
  +-- batched:    iteration-level multi-request scheduling
  +-- paged:      shared block pool + torch logical gather
  +-- triton:     shared block pool + direct-block decode kernel
```

`InferenceEngine` is the composition root. The scheduler alone owns request
state, FIFO admission, token/sequence limits, backpressure, timeout, and terminal
cleanup. Every terminal path releases cache ownership, closes the stream in
order, and resolves the request future. Paged modes keep one authoritative
physical KV pool—there is no shadow cache.

See [the architecture walkthrough](docs/architecture.md) and the stage chapters:
[naive](docs/01-naive.md), [contiguous KV](docs/02-kv-cache.md),
[continuous batching](docs/03-continuous-batching.md),
[paged KV](docs/04-paged-kv.md), and [Triton attention](docs/05-triton-attention.md).

## The five stages

| Mode | Change from the prior stage | Decode cache read path | Active requests |
|---|---|---|---:|
| `naive` | Full-sequence recomputation | None retained between steps | 1 |
| `contiguous` | Prefill once, append one token | Per-request contiguous tensors | 1 |
| `batched` | FIFO continuous scheduler | Per-request contiguous tensors | Up to 16 |
| `paged` | Shared 16-token block allocator | Temporary torch gather | Up to 16 |
| `triton` | Direct physical-block attention | Triton kernel; no decode gather | Up to 16 |

The naïve path normally recomputes the full sequence. FP16 GEMM shape changes can
flip an argmax when the two best logits are nearly tied, so ambiguous top-two
decisions are replayed through a temporary contiguous cache and immediately
discarded. No KV state survives between naïve decode steps. This small,
documented parity guard preserves the instructional baseline while keeping all
five greedy outputs deterministic on the tested L4.

## Quickstart: correctness on a cloud GPU

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

# Builds the image, downloads the pinned model remotely, and checks HF parity.
modal run modal_app.py::smoke
```

The first run downloads the pinned ~953 MB model snapshot into the Modal Volume.
Weights, Torch, CUDA work, and Triton compilation stay in Modal; they are not
downloaded to your Mac. Later runs reuse the image and Volume caches. Ephemeral
containers scale to zero after 60 seconds.

## Reproducible benchmark results

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

## Deploy the API

Generate a strong key, store it in Modal, and deploy the server-fixed `triton`
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
    "model":"Qwen/Qwen2.5-0.5B",
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
    "model":"Qwen/Qwen2.5-0.5B",
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

Accepted request fields are `model`, `input` (non-empty string),
`max_output_tokens` (1–256), `temperature` (must be `0`), and `stream` (boolean).
Unknown fields, array/multimodal inputs, the wrong model, nonzero temperature,
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
- cache kind, blocks, utilization, reserved/occupied/fragmentation/gather bytes; and
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
- roughly 953 MB of persistent model data.

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
engine_config.json           model/dependency/scheduler/cache/cloud pins
src/cloud_engine/model.py    custom Qwen2 forward path
src/cloud_engine/weights.py  explicit safetensor mapping and shape validation
src/cloud_engine/cache.py    contiguous cache, paged pool, block allocator
src/cloud_engine/attention.py torch backends and Triton dispatch
src/cloud_engine/kernel.py   direct-block Triton decode attention
src/cloud_engine/scheduler.py lifecycle, admission, backpressure, cleanup
src/cloud_engine/engine.py   engine facade, runners, request handles
src/cloud_engine/api.py      validation, auth, Responses JSON, SSE
src/cloud_engine/metrics.py  bounded measurements
benchmarks/                  deterministic workloads and three-run protocol
artifacts/                   all nine raw benchmark results
tests/                       local/CPU tests and L4 correctness matrix
docs/                        architecture and one chapter per stage
```

## Known limitations and next work

This release supports one pinned base model revision, FP16, a single L4, greedy
text-only generation, a 2,048-token context, and a narrow Responses-style API.
It has no sampling, instruction/chat template, quantization, prefix cache,
chunked prefill, dynamic KV growth, eviction, preemption, cache swapping, CUDA
graphs, tensor parallelism, multi-GPU support, persistent metrics, or user system.

The highest-value next changes are evidence-driven:

1. Batch active sequences into actual tensor forwards; the current B=1 loop is
   why iteration-level batching missed its throughput gate.
2. Allocate paged blocks on demand with an admission-safe capacity policy; eager
   worst-case reservation missed the memory gate.
3. Add a paged prefill kernel; Triton decode is direct, but prefill still gathers.
4. Add kernel autotuning/CUDA graphs only after the first three are measured.

## Troubleshooting

- **Cold request is slow:** a scale-to-zero call must start the L4 container and
  load weights. Retry only after allowing a few minutes for the first request.
- **401:** verify the exact `ENGINE_API_KEY` in the named Modal Secret and bearer
  header; redeploy after rotating a secret if an existing container stays warm.
- **422 asking for a `request` query parameter:** update to the release containing
  the FastAPI request-injection fix and redeploy.
- **`capacity_exhausted`:** reduce concurrent context/output reservations; v1
  reserves worst-case blocks at admission and does not evict.
- **Container preempted:** Modal can transparently retry ephemeral runs. Benchmark
  artifacts are written only after one complete two-warmup/three-run attempt.
- **Triton refuses a shape:** this is fail-closed behavior. Use the pinned model,
  FP16, block size 16, and contexts no longer than 2,048.

## References and license

The design is informed by the
[PagedAttention paper](https://arxiv.org/abs/2309.06180),
[Triton](https://triton-lang.org/), the
[Qwen2.5-0.5B model](https://huggingface.co/Qwen/Qwen2.5-0.5B), and
[Modal GPU/Web Function documentation](https://modal.com/docs/guide/gpu).

Repository code is [MIT licensed](LICENSE). Runtime components and downloaded
weights retain their own licenses:

| Component | License |
|---|---|
| Qwen/Qwen2.5-0.5B weights and tokenizer | Apache-2.0 |
| PyTorch | BSD-3-Clause |
| Triton | MIT |
| Transformers, Tokenizers, safetensors, huggingface_hub | Apache-2.0 |
| FastAPI, Starlette, Pydantic, HTTPX | Their published MIT/BSD licenses |
| Modal Python client | Apache-2.0 |

Contributions should keep the comparison honest: run the local suite, attach a
correctness case for behavior changes, attach raw JSON for performance claims,
and never weaken or selectively omit a workload to make a gate pass.
