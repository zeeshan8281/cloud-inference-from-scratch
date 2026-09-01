# Cloud LLM Inference Engine — Video Walkthrough Script

Target length: 5–6 minutes

## Before recording

- Warm the deployed endpoint once so a cold GPU start does not interrupt the take.
- Keep the API key hidden. The demo reads it from the Modal Secret.
- Open the repository, architecture diagram, README benchmark table, and terminal beforehand.
- Start the terminal inside the repository:

```bash
cd '/Users/zeeshan/Downloads/JSBF;SOA/cloud-inference-from-scratch'
```

## 0:00–0:20 — Hook

On screen: repository homepage, then the architecture diagram.

Say:

> This is a GPU inference engine I built from scratch to understand what actually happens beneath an LLM serving API.
>
I built the inference runtime itself instead of wrapping vLLM or calling Transformers’ generation API. It continuously batches incoming requests, manages the KV
  > cache in pages, reuses shared prompt prefixes, chunks long prefills, and recomputes evicted sequences under memory pressure—with custom Triton attention kernels
  > and CUDA Graphs accelerating the decode path.
## 0:20–0:45 — The problem

On screen: README architecture section.

Say:

> A basic inference server runs one prompt at a time and stores each request's KV cache as one large contiguous allocation.
>
> That wastes GPU capacity and causes head-of-line blocking. A production-style engine needs to combine unrelated requests into shared GPU forwards while safely managing fragmented KV memory.

## 0:45–1:35 — System architecture

On screen: open `diagrams/inference-runtime.excalidraw.json` and follow the numbered boxes.

Say:

> A request first enters the FastAPI layer, where bearer authentication, tenant authorization, body limits, and the Responses API contract are enforced.
>
> Admission control reserves both a concurrency slot and a rolling token budget. A single replica uses an in-process gate. Multi-replica deployments use an atomic Redis Lua transaction, so quotas remain consistent across containers.
>
> Accepted requests enter a decode-first scheduler. It supports 64 queued requests, 16 active sequences, a 2,048-token iteration budget, and 256-token chunked prefill.
>
> The scheduler produces a BatchPlan. The engine converts that into one PackedBatch containing a flat token axis, positions, sequence offsets, block tables, slot mappings, and selected-logit positions.
>
> The important part is that multiple request IDs execute inside one transformer invocation. This is real continuous batching, not a Python loop calling the model once per request.

## 1:35–2:15 — Paged KV memory and attention

On screen: zoom into the green KV-memory section, then open `src/cloud_engine/cache.py`.

Say:

> KV memory comes from a four-gigabyte physical pool divided into 16-token blocks.
>
> Requests grow transactionally, so partial allocation failures cannot corrupt ownership. Under pressure, the scheduler preempts a sequence, releases its blocks, and later recomputes it.
>
> Repeated prompts can reuse block-aligned prefixes through a bounded 256-block LRU cache.
>
> Attention consumes the block tables directly. There is a Torch reference implementation and a custom Triton ragged paged-attention kernel covering mixed decode and chunked-prefill work.

## 2:15–2:40 — CUDA Graph decode

On screen: open the CUDA Graph section of `src/cloud_engine/engine.py`.

Say:

> Decode workloads repeatedly execute a small set of batch shapes, so the engine pre-captures CUDA Graphs for batches 1, 2, 4, 8, and 16.
>
> Other live batch sizes are padded to the next captured bucket. This avoids continuously creating graphs while reducing Python and kernel-launch overhead during decode.

## 2:40–2:50 — Start the live demo

On screen: terminal inside the repository.

Run:

```bash
modal run demo.py
```

Say:

> Now I'll run one script against the deployed NVIDIA L4 engine. The API key comes from the Modal Secret and is never printed.

## 2:50–4:10 — Narrate the live demo

### Liveness and readiness

Say:

> First, the script confirms that the container is alive and that the model, KV pool, and CUDA Graphs have completed initialization.

### Authentication and validation

Say:

> An unauthenticated generation request is rejected with 401.
>
> It also sends a non-zero temperature, which is rejected because this release deliberately exposes greedy generation only.

### Model discovery

Say:

> The authenticated models endpoint returns the exact pinned model served by this deployment: Qwen2.5-3B.

### Blocking generation

Say:

> This is a normal blocking Responses-style request. The result includes generated text and exact input, output, and total token usage.

### SSE streaming

Say:

> The same prompt now streams through server-sent events.
>
> The client verifies monotonically increasing sequence numbers and the complete event order: response created, text deltas, text done, response completed, and the final done marker.

### Concurrent generation

Say:

> The script now launches four requests concurrently.
>
> They remain separate API requests, but the scheduler can pack their token work into shared transformer invocations.

### Metrics and batching proof

Say:

> Finally, the script reads the engine's own metrics.
>
> The key value is max-forward-request-count. A value greater than one proves that a single model forward processed multiple requests.
>
> We can also inspect queue depth, batch size, output throughput, preemptions, KV utilization, and prefix-cache hits.

## 4:10–4:50 — Correctness evidence

On screen: open `artifacts/ragged-l4-correctness.json`, then the README evidence section.

Say:

> The live demo proves that the deployed system works, but it does not prove numerical correctness by itself.
>
> The committed L4 suite contains 20 checks covering exact Hugging Face token parity, packed multi-request execution, chunked prefill, decode priority, recompute preemption, prefix reuse, attention parity, cancellation, and allocator cleanup.
>
> The identical suite also passed on an NVIDIA A100.
>
> Every published artifact contains the pinned model revision and source identity, so benchmark results remain tied to the implementation that produced them.

## 4:50–5:20 — Performance position

On screen: README's Ragged-versus-vLLM table.

Say:

> I also compared the engine against pinned vLLM 0.10 on the same L4, model revision, prompts, arrival rates, and output lengths.
>
> At four requests per second, the Ragged engine reached 51.1 output tokens per second with 138.7 milliseconds p99 time-to-first-token. That was 95.7 percent of vLLM's throughput for this measured workload.
>
> This is not a claim that the engine replaces vLLM. It shows that the architecture is real enough to benchmark against a mature inference runtime without hiding the remaining gap.

## 5:20–5:45 — Current limitations

On screen: README limitations section.

Say:

> The current release intentionally has a narrow envelope: one L4 per replica, one pinned model, greedy text generation, a 4,096-token context limit, and no tensor parallelism or speculative decoding.
>
> The focus is depth rather than feature count: scheduling, memory management, packed execution, custom kernels, correctness, and measurable behavior.

## 5:45–6:05 — Closing CTA

On screen: architecture diagram, repository URL, and GitHub star button.

Say:

> If you're building inference infrastructure, learning CUDA and Triton, or just want to see how continuous batching and paged KV caching work below an API, clone the repository and run the experiments yourself.
>
> Read the scheduler, challenge the benchmark methodology, and open an issue with the next optimization you would try.
>
> If this helped you understand inference systems, star the repository, share the walkthrough with another systems engineer, and follow me because I'll be publishing the next performance iteration with the raw numbers.

Final on-screen text:

> Clone it. Benchmark it. Break it. Improve it.
>
> GitHub: zeeshan8281/cloud-inference-from-scratch
