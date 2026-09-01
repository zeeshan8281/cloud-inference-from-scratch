---
task_id: e
role: Principal Inference Systems Architect and Skeptical Scope Reviewer
status: complete
sources_found: 20
as_of: 2026-08-24
---

## Sources

[1] Orca: A Distributed Serving System for Transformer-Based Generative Models | https://www.usenix.org/system/files/osdi22-yu.pdf | Source-Type: academic | Accessibility: public | As Of: 2022-07 | Authority: 10/10
[2] vLLM V1 scheduler implementation | https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py | Source-Type: official | Accessibility: public | As Of: 2026-08-24 | Authority: 10/10
[3] vLLM V1 GPU model runner implementation | https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py | Source-Type: official | Accessibility: public | As Of: 2026-08-24 | Authority: 10/10
[4] vLLM V1 packed input-batch implementation | https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/input_batch.py | Source-Type: official | Accessibility: public | As Of: 2026-08-24 | Authority: 10/10
[5] Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve | https://www.usenix.org/system/files/osdi24-agrawal.pdf | Source-Type: academic | Accessibility: public | As Of: 2024-07 | Authority: 10/10
[6] Efficient Memory Management for Large Language Model Serving with PagedAttention | https://arxiv.org/abs/2309.06180 | Source-Type: academic | Accessibility: public | As Of: 2023-09 | Authority: 10/10
[7] vLLM V1 KVCacheManager implementation | https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py | Source-Type: official | Accessibility: public | As Of: 2026-08-24 | Authority: 10/10
[8] FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving | https://openreview.net/pdf/7f47176a6913ff4d00c2d8cb9f8b9d53cd095373.pdf | Source-Type: academic | Accessibility: public | As Of: 2025-04 | Authority: 10/10
[9] Fused Attention — Triton documentation | https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html | Source-Type: official | Accessibility: public | As Of: 2026-08 | Authority: 9/10
[10] vLLM Benchmark CLI | https://docs.vllm.ai/en/latest/benchmarking/cli/ | Source-Type: official | Accessibility: public | As Of: 2026-08-24 | Authority: 9/10
[11] SGLang Benchmark and Profiling Guide | https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md | Source-Type: official | Accessibility: public | As Of: 2026-08 | Authority: 9/10
[12] MLPerf Inference Benchmarks | https://docs.mlcommons.org/inference/ | Source-Type: official | Accessibility: public | As Of: 2026-08 | Authority: 10/10
[13] cloud-inference-from-scratch scheduler implementation | https://github.com/zeeshan8281/cloud-inference-from-scratch/blob/main/src/cloud_engine/scheduler.py | Source-Type: official | Accessibility: public | As Of: 2026-08-24 | Authority: 8/10
[14] cloud-inference-from-scratch cache, model, attention, and kernel implementation | https://github.com/zeeshan8281/cloud-inference-from-scratch | Source-Type: official | Accessibility: public | As Of: 2026-08-24 | Authority: 8/10
[15] Qwen2.5-1.5B official config and files | https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main | Source-Type: official | Accessibility: public | As Of: 2026-08-25 | Authority: 10/10
[16] Qwen2.5-3B official config | https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/config.json | Source-Type: official | Accessibility: public | As Of: 2026-08-25 | Authority: 10/10
[17] Qwen2.5-3B official files | https://huggingface.co/Qwen/Qwen2.5-3B/tree/main | Source-Type: official | Accessibility: public | As Of: 2026-08-25 | Authority: 10/10
[18] Qwen2.5-7B official config and files | https://huggingface.co/Qwen/Qwen2.5-7B/tree/main | Source-Type: official | Accessibility: public | As Of: 2026-08-25 | Authority: 10/10
[19] NVIDIA L4 product specifications | https://www.nvidia.com/en-us/data-center/l4/ | Source-Type: official | Accessibility: public | As Of: 2026-08-25 | Authority: 10/10
[20] Modal pricing | https://modal.com/pricing | Source-Type: official | Accessibility: public | As Of: 2026-08-25 | Authority: 10/10

## Findings

- The one defensible thesis is: **build a deterministic single-GPU ragged-token engine for one pinned Qwen2 model on one NVIDIA L4, where every scheduler iteration becomes one packed model forward, prompt work is chunked under a token budget, and KV blocks are acquired only as tokens are computed**. [1][2][3][4][5][6]
- This is one coherent engine thesis rather than three unrelated features because the scheduler's per-request token counts determine both the packed tensor layout and the exact KV slots that attention must read and write. [2][3][4][7][8]
- The current repo is not that engine yet: `Scheduler` awaits `runner.step(request)` per request, `CachedRunner` sends a one-request tensor through the model, `StepContext` carries one request ID and one scalar KV start, and paged admission reserves prompt plus maximum output capacity before execution. [13][14]
- The target execution contract is a single flat token axis for embedding, projections, output projection and MLP, plus ragged attention metadata containing query offsets, positions, sequence lengths, block tables and slot mappings; a Python loop over full per-request model forwards is forbidden. [1][3][4][8]
- Decode-first chunked prefill is required for real online behavior: each iteration schedules runnable decode tokens first and uses the remaining profiled token budget for at most one prefill chunk, preventing a long prompt from monopolizing an iteration. [2][5]
- Dynamic paging must be transactional and scheduler-coupled: allocate only the blocks crossed by the current scheduled tokens, and if progress cannot be reserved, preempt a request and recompute its KV later from prompt plus already accepted output tokens. [6][7]
- The original technical kernel should be one L4-specialized batched paged-attention path that consumes device-resident metadata directly for ragged decode and chunked prefill; CUDA graphs, a kernel matrix for many architectures, and broad model support are explicitly outside the thesis. [8][9]
- Correctness must be established at packed-layout, cache-ownership, kernel-numerics, request-lifecycle and final-token levels; final text parity alone can miss cross-request KV contamination, invalid block reuse and scheduler starvation. [6][7][8][9]
- Performance must be measured through the HTTP server across offered-load sweeps with TTFT, ITL, E2E latency, throughput and SLO goodput, while kernel microbenchmarks and closed-loop engine tests remain separate diagnostic layers. [10][11][12]
- The project is complete only if it produces a real multi-request GPU forward, survives deliberate KV pressure through preemption/recomputation, improves over the current serial `triton` mode, and reports the same-protocol gap to pinned vLLM without claiming to replace it. [2][3][6][7][10][13][14]

## Deep Read Notes

### Sources [1], [2], [3], and [4]: iteration scheduling becomes packed execution
Key data: Orca changes membership at iteration boundaries; vLLM schedules token counts per request and materializes a flat token buffer, prefix-sum query offsets, absolute positions, block tables, slot mappings and logits indices.
Key insight: the architectural seam is not `Scheduler -> request`; it is `Scheduler -> BatchPlan -> PackedBatch -> one model forward`.
Useful for: the central thesis, scheduler state, tensor contracts and single-forward acceptance gate.

### Sources [5], [6], and [7]: bounded work and transactional memory progress
Key data: Sarathi-Serve places decodes ahead of bounded prefill chunks; PagedAttention and vLLM allocate KV incrementally and use preemption/recomputation when live growth cannot obtain slots.
Key insight: token budget and block allocation must commit together or roll back together, otherwise a schedule can consume budget without guaranteed KV progress.
Useful for: scheduler ordering, cache API, preemption lifecycle and no-progress invariants.

### Sources [8] and [9]: the kernel is a runtime contract, not a decorative Triton file
Key data: FlashInfer separates plan from run, uses a caller-owned workspace and device metadata, and handles ragged/paged sequences as one workload; Triton's reference validates fused attention numerically and tunes launch shape.
Key insight: the credible custom contribution is one measured L4 path with reusable metadata buffers and deterministic ragged attention, not a large catalog of unintegrated kernels.
Useful for: kernel boundary, workspace ownership, correctness envelope and microbenchmark matrix.

### Sources [10], [11], and [12]: service claims require load, latency and correctness together
Key data: current tools vary request rate, concurrency and burstiness, recommend steady-state online runs, and evaluate throughput subject to latency and accuracy constraints.
Key insight: a saturated batch tokens-per-second number cannot prove a serving engine; the accepted artifact is a throughput-latency frontier and SLO goodput under a declared arrival process.
Useful for: benchmark harness changes, vLLM comparison and release gates.

### Sources [13] and [14]: exact local seam and reusable foundation
Key data: the repo already has strict weight loading, a generic flat-token Qwen forward, a physical KV pool, a batched Triton decode function, a terminal cleanup funnel and reproducible cloud tests, but these pieces are joined through a single-request context and eager reservations.
Key insight: retain the API, model weights, queue/backpressure behavior and terminal funnel; replace only the scheduler-runner-model-attention-cache execution seam.
Useful for: keeping the implementation narrow and avoiding a rewrite.

## Selected Project Thesis

**Build `Ragged L4 Engine`: a deterministic greedy Qwen2.5 inference runtime that serves mixed prompt/decode traffic on one NVIDIA L4 using one packed token-space forward per iteration, decode-first chunked prefill, demand-paged KV, recompute preemption, and one custom batched paged-attention implementation.**

The final proof model should be one pinned **Qwen2.5-3B** revision in FP16 with a 4,096-token project context limit, up to 16 active sequences and one Modal L4 container. Keep Qwen2.5-0.5B only as the fast development and regression oracle. The external API remains greedy text generation; sampling breadth, tool calling and OpenAI surface expansion are not part of the engine thesis.

The claim is deliberately limited: **this repository implements the core control/data-plane mechanics of a real single-GPU inference engine and measures them honestly against its old engine and pinned vLLM**. It does not claim production completeness or general vLLM superiority.

## Exact Data Structures and Module Changes

### `src/cloud_engine/scheduler.py`

- Replace `tokens_fed` with `num_computed_tokens`; add `num_preemptions` and `PREEMPTED` state while retaining prompt plus accepted output IDs as the recomputation source of truth.
- Add `ScheduledItem(request, query_start, query_len, context_len, should_sample)` and `BatchPlan(items, total_tokens)` dataclasses; offsets are deterministic prefix sums in scheduler order.
- Replace the per-request `_step` loop with one `_plan_iteration()` and one `runner.step_batch(plan)` call.
- Policy: schedule one token for every runnable decode first; continue at most one partial prefill; then admit FIFO work into the remaining token budget.
- Preserve `_finalize` as the only terminal cleanup funnel and preserve existing stream backpressure semantics.

### `src/cloud_engine/engine.py`

- Replace `CachedRunner.step(request)` with `PackedRunner.step_batch(plan)`.
- Add `PackedBatch(input_ids, positions, query_start_loc, seq_lens, block_tables, slot_mapping, logits_indices)`; all tensors are built once per iteration, with metadata buffers reused at fixed maximum capacities.
- `step_batch` asks the cache to transactionally reserve the exact target positions, packs prompt slices or prior accepted decode tokens, calls the model once, and returns sampled tokens keyed by request ID.
- On preemption, release the victim's KV, set its computed cursor to zero, preserve its accepted output IDs and original arrival age, then rebuild known tokens in later chunks without resampling them.

### `src/cloud_engine/cache.py`

- Extend `BlockAllocator` into the single authoritative pool with atomic `allocate(n) -> list[int] | None`; retain preallocated K/V tensors and never allocate them after startup.
- Replace `reserve(request_id, token_capacity)` with `begin(request_id)`, `ensure_slots(request_id, target_len)`, `commit_written(request_id, target_len)` and `release(request_id)`.
- Add `RequestCacheState(block_ids, committed_len, reserved_len)`; `ensure_slots` either reserves every required new block or leaves allocator and table state unchanged.
- Defer prefix hashes, refcounts, sharing, COW and host residence; every block has exactly one request owner in this thesis.

### `src/cloud_engine/model.py`

- Change `forward(input_ids, StepContext)` to `forward(PackedBatch)` or equivalent tensor arguments.
- Keep embeddings, RMSNorm, QKV projections, output projection and MLP on the flat `[total_scheduled_tokens, hidden_size]` axis so they execute once per layer for the whole iteration.
- Apply RoPE from the packed absolute `positions`; select logits only at `logits_indices`, so intermediate prefill chunks cannot accidentally sample.

### `src/cloud_engine/attention.py` and `src/cloud_engine/kernel.py`

- Replace scalar `StepContext` with a `RaggedAttentionMetadata` view over `query_start_loc`, `seq_lens`, `block_tables` and `slot_mapping`.
- First implement a Torch segmented reference that is correct for mixed decode and chunked-prefill batches while the rest of the model remains one packed forward.
- Integrate the existing `decode_attention_batched` path with device-resident reusable tables instead of creating Python-list-derived tensors per layer.
- Then implement one FP16 L4 paged-attention path for the final Qwen2.5-3B head shape: direct block-table reads, online softmax with FP32 state, GQA mapping, causal ragged prefill and decode, zero logical K/V gather.

### `src/cloud_engine/config.py`, `src/cloud_engine/metrics.py`, `benchmarks/run.py`, and tests

- Replace stage-like serving modes with one `ragged` mode while retaining `contiguous` and current `triton` only as baselines.
- Profile and pin `max_batched_tokens`; do not assume the current 2,048-token budget is appropriate for mixed iterations.
- Record scheduled tokens, request count, model-forward calls, preemptions, recomputed tokens, block allocation failures, live/committed/reserved/free KV bytes and per-request ITL samples.
- Extend the benchmark runner with arrival timestamps, request-rate/burst controls, steady-state duration, raw per-request JSON and a same-protocol pinned-vLLM target.

## Implementation Sequence and Stop/Go Gates

### Phase 0 — Freeze truth and instrumentation

Pin the new model revision and vLLM/container versions; save current `triton` artifacts; add model-forward and request-ID tracing without changing behavior.

**GO:** the existing correctness suite remains green and artifacts identify every full-model call. **STOP:** do not change kernels until the current B=1 execution has been proven in traces.

### Phase 1 — Functional packed execution

Implement `BatchPlan`, `PackedBatch`, flat model execution and Torch segmented attention; keep the existing eager cache temporarily.

**GO:** with 16 concurrent requests, at least one model invocation contains at least four request IDs; no iteration invokes the full model once per scheduled request; greedy outputs are token-identical to sequential execution. **STOP:** if linear/MLP layers still execute in a per-request loop, this phase is not complete regardless of scheduler metrics.

### Phase 2 — Chunked prefill and demand-paged KV

Add `num_computed_tokens`, decode-first planning, transactional `ensure_slots`, incremental block tables and no-sample prefill chunks.

**GO:** a 4,096-token prompt under a 256-token budget advances through at least 16 real forwards, allocates blocks as it grows, emits no token before prompt completion, and does not cause a runnable decode to miss more than one iteration. **STOP:** any eager prompt-plus-max-output reservation, partial allocation mutation or zero-progress retry fails the phase.

### Phase 3 — Pressure recovery by recomputation

Add `PREEMPTED`, victim selection, atomic rollback and chunked KV reconstruction from known tokens.

**GO:** an intentionally undersized pool produces non-zero preemptions, no OOM/deadlock, exact output parity, eventual completion/rejection for every request and a fully reclaimed pool. **STOP:** do not add swapping or prefix caching to conceal allocator/preemption bugs.

### Phase 4 — Original integrated L4 attention path

Make metadata device-resident and reusable, integrate batched paged decode, then add causal ragged paged prefill for the one final model/head shape.

**GO:** kernel outputs match the Torch reference over boundary and adversarial cases, decode and prefill perform zero full-cache gather bytes, and median plus p99 step latency improves over the Torch segmented reference for useful batch/length regions. **STOP:** if the custom kernel loses end-to-end, retain it as a documented experiment and ship the packed Torch reference; do not add autotuning breadth or CUDA graphs to rescue an unprofiled kernel.

### Phase 5 — Online proof and release decision

Run warm steady-state HTTP load sweeps against current `triton`, new `ragged`, and pinned vLLM on the same L4, model, prompts, output policy and request process.

**GO:** new `ragged` improves steady-state output throughput by at least 20% over current `triton`, keeps p99 ITL within 1.5x of the decode-only p99 when long prefills arrive, passes pressure/leak tests, and publishes its measured vLLM gap. **STOP:** if the 20% current-engine gain fails, profile and fix the dominant packed-path bottleneck before adding any new feature; if correctness or progress fails, do not publish performance claims.

## Correctness Invariants

1. `query_start_loc[0] == 0`, `query_start_loc[-1] == input_ids.numel() == positions.numel() == sum(item.query_len)`, and every adjacent offset difference equals that item's query length.
2. `BatchPlan.total_tokens <= max_batched_tokens`; every scheduled token has exactly one request, absolute position and KV slot, and every `logits_index` belongs to an item with `should_sample=True`.
3. A request may sample only when all known prompt/output tokens through its current frontier have been computed; replay after preemption never appends a duplicate accepted token.
4. Every physical block is either free, transactionally reserved for one in-flight batch, or committed to exactly one request; the state counts always sum to total blocks.
5. Allocation failure is atomic across the free list, request block table, reserved length and metrics.
6. A request's committed block table covers exactly `ceil(committed_len / block_size)` logical blocks, contains no duplicate writable block and maps each committed position to one physical slot.
7. Block reuse occurs only after the synchronous `step_batch` completion boundary or an explicit GPU completion fence; cancellation, failure and preemption release ownership exactly once.
8. Every scheduler iteration either executes tokens, terminally changes a request, preempts a block-owning victim or sleeps; allocation retry cannot spin without freeing capacity.
9. Mixed packed execution matches the sequential contiguous oracle token-for-token, and optimized attention matches the Torch ragged reference within the declared FP16 tolerance before sampling parity is evaluated.
10. After all requests terminate, `reserved_blocks == committed_blocks == 0`, the free count equals total blocks, no pending batch owns slots and all stream futures resolve exactly once.

## Benchmark Matrix

| Layer | Matrix | Baselines | Required outputs |
|---|---|---|---|
| Allocator/property | block boundaries `15/16/17`, `31/32/33`, through 4,096; randomized allocate/grow/preempt/release | pure-Python reference | 100,000 operations, invariant check after each, final full reclamation |
| Kernel correctness | batch `1/2/4/8/16`; query length `1/16/64/256`; KV length `16/17/128/512/2048/4096`; mixed ragged rows | Torch segmented FP32/FP16 reference | max/mean error, token parity, zero gather bytes |
| Kernel speed | same shape grid, warmup plus at least 100 timed iterations | Torch segmented attention, old Triton decode | p50/p95/p99 microseconds, effective bandwidth, workspace bytes |
| Closed-loop engine | concurrency `1/4/8/16`; profiles `32->128`, `512->128`, mixed `16..2048 -> 16..256`, forced KV pressure | current `contiguous`, current `triton`, new `ragged` | input/output tok/s, requests/s, forward calls, requests/forward, preemptions, recompute tokens, KV accounting |
| Online HTTP | Poisson request-rate sweep from underload through saturation, plus burstiness and probe requests; at least `5x` max-concurrency prompts and a warm steady-state window | new `ragged`, pinned vLLM, current deployed protocol | TTFT/ITL/E2E p50/p95/p99, throughput, SLO goodput, errors/rejections, queue depth |
| Interference | continuous short decodes while 2,048/4,096-token prompts arrive | decode-only control, chunk sizes `128/256/512` | p99 ITL inflation, prefill TTFT, starvation/skipped-iteration count |
| Soak/failure | 30-minute mixed arrival stream with cancellations, slow consumers, injected model exception and undersized KV | new `ragged` only | zero leak/deadlock, bounded queue, all terminal futures resolved, stable GPU/KV bytes |

All artifacts must record source commit, model revision, runtime/container versions, GPU name, workload hash, arrival seed, warmup, duration and every measured run; one median without raw runs is insufficient.

## Explicitly Rejected Work

- **Prefix caching/radix trees:** useful later, but it cannot compensate for fake batching or eager ownership and introduces sharing, refcounts, COW, security namespaces and cache-aware fairness.
- **CPU/NVMe KV offload:** rejected until measured L4 transfer-versus-recompute crossover proves a benefit.
- **CUDA graphs:** rejected until packed eager execution is correct and profiling shows host launch overhead is the limiting term.
- **Quantization, speculative decoding and sampling features:** each creates a separate numerical/runtime thesis and obscures whether batching and paging work.
- **Multi-GPU, tensor parallelism and distributed scheduling:** rejected because one L4 is the declared hardware scope.
- **Many models, hardware-generic kernels and broad head dimensions:** rejected; one pinned 3B proof model plus the 0.5B regression oracle is enough.
- **More API endpoints, dashboards, SDKs, OpenRouter wiring or deployment polish:** low-value feature-list work that does not change GPU execution or establish engine reality.
- **Trying to beat vLLM everywhere:** rejected; the project must report the gap and may claim a niche advantage only if the same-protocol throughput-latency frontier demonstrates one.

## Model-Size Counter-Review

**Recommendation: use Qwen2.5-3B as the final proof model; keep 0.5B for fast tests; do not make 1.5B or 7B release blockers.**

The transparent FP16 estimates below use the official architecture fields and the standard decoder-only KV formula:

`KV bytes/token = 2 (K,V) × layers × KV heads × head_dim × 2 FP16 bytes`, where `head_dim = hidden_size / attention_heads`.

The 16-request KV column assumes each live sequence reaches the full project limit of 4,096 total tokens. Weight size uses the published safetensor repository size, so these figures deliberately exclude PyTorch allocator overhead, activations, logits, metadata, Triton workspace and temporary correctness-oracle memory.

| Candidate | Official shape | Published weights | FP16 KV/token | KV at `16 × 4096` | Weights + KV | Decision |
|---|---|---:|---:|---:|---:|---|
| Qwen2.5-1.5B | 28 layers, 12 Q heads, 2 KV heads, head dim 128, tied embeddings [15] | 3.09 GB = 2.88 GiB [15] | 28 KiB | 1.75 GiB | ~4.63 GiB | Too light for the final proof; good development oracle but leaves most of the L4 idle and makes host/metadata overhead unusually dominant. |
| Qwen2.5-3B | 36 layers, 16 Q heads, 2 KV heads, head dim 128, tied embeddings [16] | 6.18 GB = 5.76 GiB [17] | 36 KiB | 2.25 GiB | ~8.01 GiB | **Best balance:** meaningful compute, same head-dim-128 kernel challenge, ample room for Torch reference runs/workspaces, and no new untied-LM-head architecture path. |
| Qwen2.5-7B | 28 layers, 28 Q heads, 4 KV heads, head dim 128, untied embeddings [18] | 15.2 GB = 14.16 GiB [18] | 56 KiB | 3.50 GiB | ~17.66 GiB | Too risky for the first real-engine milestone: only ~6.34 GiB remains on a 24 GB L4 before runtime overhead, and the repo currently rejects untied embeddings. |

NVIDIA specifies 24 GB HBM and 300 GB/s memory bandwidth for L4. [19] The 3B model therefore leaves about 16 GiB beyond worst-case weights-plus-KV for activations, packed buffers, the Torch reference and allocator headroom; 7B leaves only about 6.3 GiB and would turn ordinary debugging/reference validation into an OOM-management exercise before the new scheduler is proven.

The 3B choice does **not** fake memory pressure. Production engines explicitly cap the KV pool below total free HBM; configure a real 1.5–2.0 GiB GPU KV pool during the pressure suite so a legal `16 × 4096` workload exceeds cache capacity and exercises real transactional allocation and recompute preemption. Run the normal throughput suite with a larger profiled pool so artificial scarcity does not distort steady-state speed.

Modal's current L4 price is `$0.000222/s`, or `$0.7992/GPU-hour`, independent of model size; model size affects cost through runtime, not the per-second rate. [20] Base L4 spend is therefore about `$7.99` for 10 GPU-hours, `$19.98` for 25 hours and `$39.96` for 50 hours, before CPU/memory charges. Modal lists `$30/month` Starter compute credit, equivalent to about 37.5 L4-hours if spent only on the GPU line item. [20] Do not predict a precise 1.5B/3B/7B cost multiplier before measuring throughput, but 7B necessarily makes every parity, shape-grid and online sweep materially slower while providing no new scheduler invariant.

**Go gate for the model migration:** Qwen2.5-3B must load within a measured HBM budget, match Hugging Face logits/tokens on the existing oracle suite, and complete one 16-request 4,096-context pressure run before custom-kernel tuning begins. If that fails for a root cause unrelated to the new scheduler, fall back to 1.5B for implementation—not 7B—and keep 3B as a later graduation gate.

## Gaps

- No source establishes a universal token budget, preemption victim or 20%/1.5x acceptance threshold for Qwen2.5-3B on an L4; those are explicit project gates to validate by profiling, not borrowed performance predictions.
- The notes do not contain a primary-source performance profile for this exact model/head shape on L4, so kernel launch configuration and attainable bandwidth remain unknown until measured.
- Moving the final proof from 0.5B to 3B requires pinning and independently validating the weight/config compatibility and numerical oracle before the engine benchmark is meaningful.
- The current benchmark harness is closed-loop and batch-oriented; implementing a reproducible online arrival generator and same-protocol vLLM adapter is required before service-level claims.
- The plan deliberately uses exclusive block ownership; prefix reuse value and conversational traces remain unmeasured and are not part of this thesis.

## Counter-Claim

A packed, demand-paged engine can satisfy every architectural invariant and still lose to the current implementation on Qwen2.5-0.5B because metadata construction, segmented attention and launch overhead dominate tiny-model arithmetic; it can also remain far behind vLLM because mature libraries have broader kernel fusion and runtime optimization. [8][10][11][14] That outcome would not make the implementation a simulation, but it would invalidate any performance-superiority story. The release must therefore separate three claims: **real multi-request execution**, **correct pressure handling**, and **measured speed**; only the first two are mandatory for the technical thesis, while the third determines whether the custom kernel remains a serving default or a documented experiment.
