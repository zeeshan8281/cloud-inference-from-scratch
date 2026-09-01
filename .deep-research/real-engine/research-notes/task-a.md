---
task_id: a
role: LLM Scheduler and Batching Systems Researcher
status: complete
sources_found: 9
as_of: 2026-08-24
---

## Sources

[1] vLLM V1 scheduler implementation | https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py | Source-Type: official | Accessibility: public | Date: continuously updated; accessed 2026-08-24 | As Of: 2026-08-24 | Authority: 10/10
[2] vLLM V1 GPU model runner implementation | https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py | Source-Type: official | Accessibility: public | Date: continuously updated; accessed 2026-08-24 | As Of: 2026-08-24 | Authority: 10/10
[3] vLLM V1 packed input-batch implementation | https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/input_batch.py | Source-Type: official | Accessibility: public | Date: continuously updated; accessed 2026-08-24 | As Of: 2026-08-24 | Authority: 10/10
[4] vLLM optimization and tuning guide | https://docs.vllm.ai/en/stable/configuration/optimization/ | Source-Type: official | Accessibility: public | Date: continuously updated; accessed 2026-08-24 | As Of: 2026-08-24 | Authority: 9/10
[5] Orca: A Distributed Serving System for Transformer-Based Generative Models | https://www.usenix.org/system/files/osdi22-yu.pdf | Source-Type: academic | Accessibility: public | Date: 2022-07 | As Of: 2022-07 | Authority: 10/10
[6] Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve | https://www.usenix.org/system/files/osdi24-agrawal.pdf | Source-Type: academic | Accessibility: public | Date: 2024-07 | As Of: 2024-07 | Authority: 10/10
[7] cloud-inference-from-scratch scheduler implementation | https://github.com/zeeshan8281/cloud-inference-from-scratch/blob/main/src/cloud_engine/scheduler.py | Source-Type: official | Accessibility: public | Date: accessed 2026-08-24 | As Of: 2026-08-24 | Authority: 8/10
[8] cloud-inference-from-scratch runner implementation | https://github.com/zeeshan8281/cloud-inference-from-scratch/blob/main/src/cloud_engine/engine.py | Source-Type: official | Accessibility: public | Date: accessed 2026-08-24 | As Of: 2026-08-24 | Authority: 8/10
[9] cloud-inference-from-scratch KV-cache implementation | https://github.com/zeeshan8281/cloud-inference-from-scratch/blob/main/src/cloud_engine/cache.py | Source-Type: official | Accessibility: public | Date: accessed 2026-08-24 | As Of: 2026-08-24 | Authority: 8/10

## Findings

- The current repo does iteration-level request selection but awaits `runner.step(request)` separately for every active request, and `CachedRunner.step` creates a one-request 1-D token tensor and calls the model once, so its “batched” mode never executes a multi-request model forward. [7][8]
- Its paged cache also reserves `ceil((prompt_tokens + max_output_tokens) / block_size)` blocks at admission, so it cannot grow KV on demand or recover from mid-run pressure through preemption. [8][9]
- Orca's defining contract is that the scheduler may change request membership after every model iteration, while selected requests share batched parameterized operations and only irregular attention work is segmented per request. [5]
- Current vLLM represents one iteration as `num_scheduled_tokens: dict[request_id, count]`, packs the ragged queries into one 1-D token buffer, derives `query_start_loc` by prefix sum, carries per-token positions plus block tables/slot mappings, and submits that combined input in one model execution. [1][2][3]
- vLLM V1 avoids a hard prefill/decode state split by tracking each request's `num_computed_tokens` and scheduling the gap to its known token count, which naturally makes a prompt larger than the remaining token budget a partial prefill chunk. [1]
- Decode-first/token-budget scheduling means existing running requests consume one decode token each before leftover budget is assigned to partial or new prefills; vLLM documents this policy, while Sarathi-Serve formalizes it as all decodes, at most one ongoing prefill chunk, then new admissions. [4][6]
- Sarathi-Serve reports that unrestricted hybrid prefills can raise time-between-token latency by up to 28.3x, whereas a fixed token budget bounds per-iteration work; it recommends selecting that budget from hardware profiling against a target TBT SLO rather than treating the budget as a universal constant. [6]
- The minimum real implementation for this repo therefore needs one `BatchPlan` per iteration containing request IDs, token counts, context starts, sample flags and newly allocated blocks, followed by one packed `step_batch(BatchPlan)` model call using `input_ids`, `positions`, `query_start_loc`, `seq_lens`, `block_tables`, `slot_mapping`, and `logits_indices`. [1][2][3][5]
- Under KV pressure, vLLM repeatedly frees a lowest-priority running victim until allocation succeeds, moves the victim back to waiting, and uses recomputation rather than swap by default; a minimal implementation here can do the same only if it records preemption count and can reconstruct KV from prompt plus accepted output tokens. [1][4]
- A system is demonstrably real only when traces prove that one GPU forward contains multiple request IDs and fewer forwards produce the same tokens, while correctness checks enforce packed-offset, token-budget, KV-ownership, preemption-recovery, output-parity, and no-starvation invariants under mixed prompt lengths. [1][2][3][5][6]

## Deep Read Notes

### Source [1]: vLLM V1 scheduler implementation
Key data: one schedule returns request-to-token counts, asserts their sum is within `max_num_scheduled_tokens`, schedules running requests first, and allocates only the KV slots needed for the scheduled tokens.
Key insight: `num_computed_tokens` is the unifying cursor; chunked prefill is not a special fake mode but repeated real forward progress over prompt slices.
Useful for: the scheduler state model, token-budget accounting, allocation rollback, recompute preemption, and executable invariants.

### Sources [2][3]: vLLM GPU runner and packed input batch
Key data: per-request scheduled counts are prefix-summed into `query_start_loc`; packed input IDs, absolute positions, block tables, slot mappings and end-of-query logits indices describe variable-length requests without padding them to a common query length.
Key insight: the proof of batching is below the scheduler—the model receives one combined token axis and one attention metadata object, not a Python loop of `B=1` forwards.
Useful for: the minimum `BatchPlan`/`PackedBatch` structures and the model/attention API boundary.

### Source [5]: Orca
Key data: Orca schedules exactly one model iteration at a time and permits late joins and early exits after each iteration; its selective batching concatenates work for parameterized layers while handling variable-shaped attention state separately.
Key insight: iteration-level scheduling alone is insufficient; multi-request tensor execution is the technical mechanism that converts flexible scheduling into throughput.
Useful for: rejecting a scheduler-only claim of continuous batching and defining the single-forward requirement.

### Source [6]: Sarathi-Serve
Key data: every iteration first packs ongoing decodes, then an optional partial prefill, then new requests within a token budget; the paper reports up to 28.3x TBT inflation for naive full-prefill hybrid batches and observes chunk-size/tile effects such as 257 tokens taking 32% longer than 256 in one case.
Key insight: chunked prefill is a latency control mechanism whose chunk size must be benchmarked on the actual model and GPU, not merely split for API-level fairness.
Useful for: decode-first scheduling, chunk profiling, TBT SLO gates, and adversarial long-prefill tests.

### Required implementation for this repo (evidence-derived synthesis)

1. Extend `Request` with `num_computed_tokens`, `num_preemptions`, and an explicit `RUNNING/PREEMPTED` lifecycle; keep prompt and accepted output IDs as the recomputation source of truth.
2. Replace `runner.step(request)` with `runner.step_batch(plan)`, where each `ScheduledItem` holds `request`, `num_query_tokens`, `context_start`, `is_prefill`, `should_sample`, and allocated block IDs.
3. Build one `PackedBatch` with contiguous 1-D `input_ids`, `positions`, `query_start_loc=[0,cumsum(query_tokens)]`, `seq_lens`, dense/padded block tables only as attention metadata, per-token KV slot mappings, and `logits_indices` only for decodes or final prefill chunks.
4. Change the model so embedding, QKV, output projection and MLP consume the combined `[total_scheduled_tokens, hidden]` tensor once per layer; attention must use query segment offsets and each request's logical context/block table without falling back to a Python loop of full model forwards.
5. Schedule in this order: one token for every runnable decode, continue at most one partial prefill, then admit FIFO requests with `chunk=min(prompt_remaining, budget_remaining)`; decrement one shared token budget for every packed token.
6. Allocate KV blocks only when a scheduled chunk crosses a block boundary; if allocation fails, free a lowest-priority/youngest running victim, reset its computed cursor for recomputation, move it to waiting, restore any current-step token budget, and retry.

### Benchmark and invariant gates that prove it is real

- **Single-forward gate:** instrument `model.forward`; with 16 concurrent requests, at least one invocation must contain at least 4 distinct request IDs, and no scheduler iteration may invoke the full model once per scheduled request.
- **Amortization gate:** for 16 requests generating 64 tokens each, `model_forward_calls / output_tokens` must be at most 0.20 while outputs remain token-identical to single-request execution.
- **Packed-layout gate:** every iteration must assert `query_start_loc[-1] == input_ids.numel() == sum(num_scheduled_tokens)`, adjacent offset differences equal each request's query count, positions are correct per request, and total scheduled tokens never exceed the configured budget.
- **Chunked-prefill gate:** a 2,048-token prompt under a 256-token budget must advance over at least eight real model calls, allocate KV incrementally, emit no token before its final prompt chunk, and coexist with decodes that receive a slot in every iteration where they are runnable.
- **Cross-request correctness gate:** mixed prompt lengths and mixed prefill/decode batches must match a trusted sequential run token-for-token and show no KV/block-table contamination across request IDs.
- **Preemption gate:** with KV capacity intentionally below aggregate demand, the run must produce `preemptions_total > 0`, avoid OOM, release every victim block exactly once, resume the victim by recomputation, complete all requests with output parity, and finish with all blocks free.
- **Performance gate:** on the same L4, model, prompts and concurrency, packed continuous batching must improve steady-state output throughput by at least 20% over the current serial “batched” mode; otherwise the implementation is functionally real but not yet a useful optimization.
- **Latency-under-arrival gate:** when long prompts arrive during active decoding, p99 ITL must stay within 1.5x the decode-only p99 baseline at the chosen token budget, and no continuously runnable decode may be skipped for more than one scheduler iteration.
- **Capacity gate:** under a fixed p99 TTFT and p99 ITL SLO, the new engine must sustain a higher admitted request rate than the current implementation for a traced mixed-length workload; report the arrival process, prompt/output distribution, warm-up, sample count and confidence intervals.

## Gaps

- vLLM main is a moving target as of 2026-08-24, so implementation details should be pinned to a commit before code is copied or benchmark results are claimed reproducible.
- Orca's selective attention design predates modern paged/varlen attention kernels; it establishes the batching principle but is not a drop-in kernel design for this repo.
- Sarathi-Serve's reported gains use larger models and A100/A40-class deployments, so its numeric improvements cannot be transferred to Qwen2.5-0.5B on an L4.
- No primary source establishes that the proposed 20% throughput and 1.5x ITL gates are universal; they are explicit project acceptance thresholds and should be revised only from measured L4 profiles.
- Counter-claim: a packed implementation can pass every functional “real batching” invariant yet be slower on a 0.5B model because packing, metadata construction and launch overhead dominate, so technical reality and benchmark superiority must be reported as separate outcomes.
