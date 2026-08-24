# Build the engine behind an LLM API without owning a GPU

Cloud Inference Engine Lab is an educational, from-scratch Qwen2.5-0.5B inference server that builds KV caching, continuous batching, paged KV allocation, streaming, and Triton decode attention on one cloud NVIDIA L4.

## Architecture in 90 seconds

A small authenticated FastAPI surface validates a Responses-style request and submits it to one long-lived scheduler. The scheduler owns a FIFO waiting queue, admits work within token and sequence limits, advances every active request one token per iteration, and publishes tokens to bounded per-request queues. `InferenceEngine` owns the tokenizer, custom Qwen2 model, scheduler, and the selected cache/attention backend.

```text
client -> FastAPI -> InferenceEngine -> Scheduler -> custom Qwen2
                                            |             |
                                      token stream   attention backend
                                                          |
                                         none / contiguous / paged pool
                                                          |
                                              torch reference / Triton
```

The five modes preserve the same weights and greedy workload while changing one serving mechanism at a time. Heavy packages, weights, CUDA execution, and Triton compilation exist only in the Modal image; the local package has no runtime dependencies.

The engine does **not** call vLLM, SGLang, Hugging Face `generate()`, or Hugging Face `past_key_values`. A `generate()` call exists only in the remote smoke-test oracle used to compare token IDs.

## Cloud-only quickstart

```bash
git clone <repository-url>
cd cloud-inference-from-scratch
python3 -m pip install --user modal
modal setup
modal volume create cloud-inference-model-cache
modal run modal_app.py::smoke
```

The first smoke run starts one L4 container, downloads the pinned Qwen snapshot into the persistent Volume, loads the custom contiguous-cache engine, generates a deterministic answer, compares its token IDs with the Hugging Face oracle, and prints TTFT, throughput, peak GPU memory, and a Modal dashboard link. Later runs reuse cached weights. The container scales to zero after 60 seconds.

> Modal GPU time and persistent Volume storage are billable. The app is capped at one container with zero minimum containers; no benchmark runs automatically. The first model and image download can take several minutes and roughly 1 GB of persistent model storage.

## Stages

| Mode | Change from previous stage | Cache read path | Active requests |
|---|---|---|---:|
| `naive` | Recompute the full sequence for each token | No cache | 1 |
| `contiguous` | Prefill once; append to a per-request tensor | Direct contiguous tensor | 1 |
| `batched` | Add FIFO iteration-level scheduling | Direct contiguous tensor | Up to 16 |
| `paged` | Replace per-request tensors with shared 16-token blocks | Temporary torch gather | Up to 16 |
| `triton` | Decode directly through physical block tables | Triton direct-block kernel | Up to 16 |

Run a stage with:

```bash
modal run modal_app.py::benchmark --mode naive --profile decode --output artifacts/naive-decode.json
modal run modal_app.py::benchmark --mode contiguous --profile decode --output artifacts/contiguous-decode.json
modal run modal_app.py::benchmark --mode batched --profile decode --output artifacts/batched-decode.json
modal run modal_app.py::benchmark --mode paged --profile fragmentation --output artifacts/paged-fragmentation.json
modal run modal_app.py::benchmark --mode triton --profile decode --output artifacts/triton-decode.json
```

## Reproducible results

GPU measurements have not been run yet. This table stays deliberately blank until the fixed protocol—two warmups, three recorded runs, median, and all raw runs—passes correctness and publication gates.

| Mode | Profile | Output tok/s median | TTFT p50 | Peak GPU memory | KV fragmentation | Raw result |
|---|---|---:|---:|---:|---:|---|
| `naive` | decode | pending | pending | pending | n/a | pending |
| `contiguous` | decode | pending | pending | pending | pending | pending |
| `batched` | decode | pending | pending | pending | pending | pending |
| `paged` | fragmentation | pending | pending | pending | pending | pending |
| `triton` | decode | pending | pending | pending | pending | 0 full-cache decode gather required |

Each result records package versions, the full model revision, GPU name, workload hash, mode, and source revision. Results contain no prompts or secrets.

## API

Create a secret and deploy the server-side-fixed `triton` mode:

```bash
modal secret create cloud-inference-api ENGINE_API_KEY=<user-generated-value>
modal deploy modal_app.py
```

Non-streamed request:

```bash
curl -sS "$ENGINE_URL/v1/responses" \
  -H "Authorization: Bearer $ENGINE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B","input":"Explain a KV cache in one sentence.","max_output_tokens":64,"temperature":0,"stream":false}'
```

Streamed request:

```bash
curl -N "$ENGINE_URL/v1/responses" \
  -H "Authorization: Bearer $ENGINE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B","input":"Explain paged attention in one sentence.","max_output_tokens":64,"temperature":0,"stream":true}'
```

`GET /healthz` is public. `GET /metrics` and `POST /v1/responses` require the bearer secret. This is a strict subset: unknown fields and unsupported input forms are rejected.

## Repository map

```text
modal_app.py                 Modal image, GPU jobs, deployment, cost limits
engine_config.json          model, dependency, scheduler, cache, cloud pins
src/cloud_engine/model.py   custom Qwen2 forward path
src/cloud_engine/weights.py explicit safetensors mapping and shape checks
src/cloud_engine/cache.py   contiguous cache, paged pool, block allocator
src/cloud_engine/attention.py + kernel.py  torch and Triton attention paths
src/cloud_engine/scheduler.py request lifecycle, admission, backpressure
src/cloud_engine/engine.py  public engine and request handles
src/cloud_engine/api.py     validation, auth, Responses JSON and SSE
src/cloud_engine/metrics.py bounded runtime measurements
benchmarks/                 fixed workloads and three-run benchmark protocol
tests/                      local stdlib tests and remote L4 parity tests
docs/                       architecture and one chapter per engine stage
```

## Limitations and non-goals

This is educational code, **not production-ready and not a replacement for vLLM**. V1 supports one pinned Qwen2.5-0.5B revision, FP16 on one L4, greedy text generation, a 2,048-token context, and a narrow Responses API. There is no sampling, quantization, prefix cache, chunked prefill, preemption, cache swapping, multi-GPU execution, user system, database, or public unauthenticated endpoint. Scheduler iterations cover multiple requests, but model forwards currently remain per-sequence. Paged admission reserves worst-case blocks up front; paged prefill still uses the torch gather path. Remote correctness and performance remain unverified until the billable test sequence is run.

## Stop compute and remove cloud resources

```bash
modal app list
modal app stop cloud-inference-lab
modal volume get cloud-inference-model-cache
modal volume delete cloud-inference-model-cache
```

Stopping the app ends deployed compute. Deleting the Volume removes the cached public model weights and Triton artifacts; the next smoke run must download them again. Rotate `ENGINE_API_KEY` if it is ever exposed.

## Contributing and license

Run the zero-ML local suite before sending a change:

```bash
python3 -m pip install -e '.[dev]'
python3 -m unittest discover -s tests -p 'test_*.py'
ruff check .
```

Please attach a correctness case or raw benchmark JSON to measured optimization changes. Do not weaken workloads or omit unfavorable runs.

Repository code is [MIT licensed](LICENSE). Runtime components retain their own licenses:

| Component | License |
|---|---|
| Qwen/Qwen2.5-0.5B weights and tokenizer | Apache-2.0 |
| PyTorch | BSD-3-Clause |
| Triton | MIT |
| Transformers / Tokenizers / safetensors / huggingface_hub | Apache-2.0 |
| FastAPI / Starlette / Pydantic / HTTPX | MIT / BSD-3-Clause as published by each project |
| Modal Python client | Apache-2.0 |

