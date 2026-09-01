---
task_id: b
role: KV Memory Management Researcher
status: complete
sources_found: 10
as_of: 2026-08-24
---

## Sources

[1] Efficient Memory Management for Large Language Model Serving with PagedAttention | https://arxiv.org/abs/2309.06180 | Source-Type: academic | Accessibility: public | Date: 2023-09-12 (SOSP 2023) | As Of: 2026-08-24 | Authority: 10/10
[2] Automatic Prefix Caching — vLLM design documentation | https://docs.vllm.ai/en/stable/design/prefix_caching/ | Source-Type: official | Accessibility: public | Date: live stable documentation, accessed 2026-08-24 | As Of: 2026-08-24 | Authority: 9/10
[3] vLLM V1 KVCacheManager source | https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py | Source-Type: official | Accessibility: public | Date: current main, accessed 2026-08-24 | As Of: 2026-08-24 | Authority: 10/10
[4] vLLM V1 scheduler source | https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py | Source-Type: official | Accessibility: public | Date: current main, accessed 2026-08-24 | As Of: 2026-08-24 | Authority: 10/10
[5] SGLang: Efficient Execution of Structured Language Model Programs | https://arxiv.org/abs/2312.07104 | Source-Type: academic | Accessibility: public | Date: 2023-12-12; revised 2024-06-06; NeurIPS 2024 | As Of: 2026-08-24 | Authority: 10/10
[6] SGLang RadixCache source | https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py | Source-Type: official | Accessibility: public | Date: current main, accessed 2026-08-24 | As Of: 2026-08-24 | Authority: 10/10
[7] SGLang HiCache: Fast Hierarchical KV Caching with Your Favorite Storage Backends | https://www.lmsys.org/blog/2025-09-10-sglang-hicache/ | Source-Type: official | Accessibility: public | Date: 2025-09-10 | As Of: 2026-08-24 | Authority: 8/10
[8] TensorRT-LLM KV Cache System | https://nvidia.github.io/TensorRT-LLM/features/kvcache.html | Source-Type: official | Accessibility: public | Date: live documentation, accessed 2026-08-24 | As Of: 2026-08-24 | Authority: 9/10
[9] TensorRT-LLM KV Cache Reuse | https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html | Source-Type: official | Accessibility: public | Date: live documentation, accessed 2026-08-24 | As Of: 2026-08-24 | Authority: 9/10
[10] vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention | https://arxiv.org/abs/2405.04437 | Source-Type: academic | Accessibility: public | Date: 2024-05-07; ASPLOS 2025 | As Of: 2026-08-24 | Authority: 9/10

## Findings

- A serious paged cache preallocates one physical GPU pool but assigns blocks to each request only as its logical sequence grows, bounding per-request internal waste to the final partially filled block instead of reserving `prompt + max_output_tokens` at admission. [1][2][3]
- Incremental allocation makes output-length uncertainty a scheduler problem: allocation must be attempted before every scheduled step, with headroom/reservations for in-flight work and an explicit preemption path when the next slots cannot be obtained. [1][3][4]
- Memory pressure has two distinct victims: reusable prefix blocks with zero live references can be evicted without interrupting a request, while blocks owned by a running request require request-level preemption followed by recomputation or swap-in. [1][2][5][6]
- Prefix reuse needs an identity stronger than token IDs alone: a chained block hash must include the prior-prefix hash and execution namespace such as model revision, adapter or prompt-embedding identity, multimodal inputs, and tenant/cache salt to prevent false reuse and cross-tenant leakage. [2][6][9]
- Shared blocks require reference counts or lock references; a block referenced by a running request is non-evictable, and an attempted write to a shared partial block must trigger copy-on-write or be forbidden by publishing only immutable full blocks. [1][2][5][6]
- vLLM uses a hash-to-block map plus an intrusive LRU/free queue, whereas SGLang uses a radix tree with leaf-LRU and longest-shared-prefix-aware scheduling; the former is the smaller full-block implementation, while the latter captures multi-level conversational and agent-workflow reuse. [2][5][6]
- TensorRT-LLM adds prioritized LRU, per-token-range retention, partial-block reuse controls, and separate pools for attention layouts, demonstrating that eviction value and block compatibility eventually become policy dimensions rather than a single free list. [8]
- Host offload is useful only when transfer plus queueing is cheaper than recomputation: PagedAttention explicitly treats swap and recompute as alternatives, TensorRT-LLM warns that host offload can lose on older interconnects, and HiCache overlaps layer-wise transfer and exposes prefetch deadlines because I/O latency is workload- and hardware-dependent. [1][7][9]
- Fragmentation accounting must separate physical-pool slack, last-block internal slack, reusable cached capacity, live protected capacity, transfer-reserved capacity, and duplicated cache entries; one aggregate `reserved - occupied` number cannot diagnose paging behavior. [1][2][3][6]
- The correctness surface is concurrency-heavy: allocation must be all-or-nothing, refcount transitions must exactly track request tables, cache publication must occur only after KV writes complete, GPU-visible blocks cannot be recycled before their consuming work completes, and every terminal/preempted path must release or deliberately retain each reference exactly once. [2][3][4][6]

## Deep Read Notes

### Source [1]: PagedAttention paper
Key data: fixed-size logical blocks map to non-contiguous physical blocks allocated as needed; waste is limited to one final block, and shared blocks use refcounts plus block-granularity copy-on-write.
Key insight: paging is co-designed with preemptive scheduling; when live output growth exhausts HBM, recovery is a measured choice between CPU swapping and prompt-phase recomputation.
Useful for: dynamic growth semantics, COW, fragmentation definitions, request-level preemption, swap-versus-recompute benchmark.

### Source [2]: vLLM V1 Automatic Prefix Caching design
Key data: the pool owns pre-created `KVCacheBlock` records; each record carries immutable ID, hash, `ref_cnt`, and intrusive free-queue links, with hash map and per-request block tables as the other core indexes.
Key insight: cached blocks with `ref_cnt == 0` deliberately remain on the free/LRU queue, so cached capacity and free-to-repurpose capacity overlap rather than forming disjoint pools.
Useful for: smallest credible data model, O(1) touch/evict/free operations, full-block hash-chain prefix reuse.

### Sources [3] and [4]: current vLLM KVCacheManager and scheduler
Key data: `allocate_slots` includes lookahead, external KV, full-sequence-fit, watermark, and in-flight reserved-block accounting; the scheduler retries allocation after selecting a preemption victim and defers frees until GPU work is safe.
Key insight: safe admission is not a one-time capacity check—requests, external transfers, speculative slots, and asynchronous GPU completion all reserve future progress, so the allocator and scheduler need one transactional interface.
Useful for: admission API, no-progress/deadlock prevention, block lifetime fences, preemption metrics.

### Source [5]: SGLang / RadixAttention paper
Key data: radix nodes retain KV after requests finish, leaf-LRU evicts only nodes with zero reference count, and longest-shared-prefix-first scheduling achieved 96% of the offline-optimal cache-hit rate on the paper's workloads while tree-management overhead was under 0.3%.
Key insight: prefix cache policy and request scheduling are coupled; preserving useful prefixes can raise throughput, but greedy cache-aware ordering can starve requests and therefore needs a fairness bound.
Useful for: radix index design, protected/evictable accounting, cache-affinity scheduling with aging.

### Source [6]: current SGLang RadixCache implementation
Key data: each node tracks `lock_ref`, last-access time, hit count, host refcount/value, priority, and device value; eviction walks an evictable-leaf heap and frees exact page-aligned segments.
Key insight: cached KV is not leaked memory—finished-request data becomes evictable cache, while live paths move byte/token counts between protected and evictable accounting as lock references change.
Useful for: invariants, prefix-node split behavior, leaf eviction, host-transfer protection.

### Sources [7], [8], and [9]: HiCache and TensorRT-LLM tiering
Key data: HiCache records per-prefix residence across GPU/CPU/storage, supports deadline-sensitive prefetch and write-through/selective/write-back policies; TensorRT-LLM supports prioritized LRU, CPU secondary cache, partial reuse/COW controls, and cache identity extras.
Key insight: offload is a cache hierarchy, not a synchronous `tensor.cpu()` fallback—the control plane must represent in-flight transfer, residence, admission reservation, cancellation, and stale-entry invalidation.
Useful for: later CPU tier, policy knobs, security namespace, offload benchmark gates.

## Concrete Code Structures

Minimum serious change, reusing `src/cloud_engine/cache.py` and `src/cloud_engine/scheduler.py` instead of adding a subsystem hierarchy:

```python
@dataclass
class Block:
    block_id: int
    state: Literal["free", "private", "shared", "offloading", "host"]
    ref_count: int = 0
    valid_tokens: int = 0
    prefix_hash: bytes | None = None
    last_access_ns: int = 0
    gpu_fence: Any | None = None

@dataclass
class RequestCacheState:
    block_ids: list[int]
    seq_len: int
    computed_tokens: int
    preemptions: int = 0
```

- Extend `BlockAllocator` into one authoritative `BlockPool` with `free_lru`, `blocks`, and transactional `allocate(n) -> list[int] | None`; do not allocate Torch tensors after startup.
- Replace `PagedKVCache.reserve(request_id, token_capacity)` with `begin(request_id)`, `ensure_slots(request_id, target_tokens)`, and `release(request_id)`; `append` calls `ensure_slots` before crossing a block boundary.
- Add `PrefixIndex[(namespace, parent_hash, block_token_ids)] -> block_id`; initially publish only full, completed blocks, which gets safe prefix sharing without partial-block COW complexity.
- Add `touch_prefix`, `publish_full_blocks`, `evict_reusable(n)`, `preempt(request_id)`, and `resume(request_id)` on `PagedKVCache`; the scheduler, not the attention kernel, chooses victims.
- Extend `RequestState` with `PREEMPTED`; on allocation failure: evict zero-ref cached blocks, retry, then preempt the newest/lowest-priority running request and recompute it later while preserving FCFS age.
- Add a free-block watermark and a next-step progress reserve (at least one impending decode block per active request near a boundary); expose full-sequence admission as an optional anti-thrash mode, not as eager ownership.
- Add CPU offload only after a measured `swap_ms < recompute_ms` crossover on the target L4 host; its minimum state machine is `GPU -> OFFLOADING -> HOST -> LOADING -> GPU`, with cancellation and GPU-event fencing.

## Correctness Invariants

1. Every physical block is in exactly one allocator state, and `free + private + shared + transfer_reserved + host_mapped == total` for each pool/tier.
2. `ref_count` equals the number of live request-table references; `ref_count > 0` blocks are never eviction victims, and `ref_count == 0` blocks are either free/reusable-cache/offloaded, never privately owned.
3. A request block table contains no duplicate writable block, covers `ceil(seq_len / block_size)` logical blocks, and its final `valid_tokens` equals `seq_len % block_size` (or `block_size` when aligned).
4. Allocation failure is atomic: request table, refcounts, free queue, prefix index, and metrics are byte-for-byte unchanged.
5. A shared block is immutable; any append into a block with `ref_count > 1` copies it first, unless the implementation deliberately reuses full blocks only and allocates a fresh tail.
6. A prefix entry becomes visible only after all layers' KV writes have completed; namespace/hash mismatch must always become a miss.
7. Blocks referenced by submitted GPU work are recycled only after its completion fence, including cancellation, timeout, preemption, and async streaming cleanup.
8. Eviction removes every lookup key pointing at a block before that block is reused, and tier transfer never leaves two mutable authoritative copies.
9. Every scheduler iteration either executes tokens, completes a transfer, evicts/preempts at least one block-owning victim, or sleeps/rejects; allocation retry cannot spin without freeing capacity.
10. After all requests finish and the prefix cache is explicitly cleared, `used_blocks == 0`, all transfer queues are empty, and allocator invariants pass.

## Benchmark Gates

- **Dynamic-growth memory gate:** mixed 1–2,048-token sequences must show `reserved_blocks == ceil(current_live_tokens / block_size) + final-block slack`, and at least 40% less unused KV capacity than the current eager-paged artifact at identical concurrency.
- **Correctness gate:** token-for-token parity with the contiguous reference for block-boundary lengths `{15,16,17,31,32,33,2047,2048}` across single and batched requests.
- **Churn gate:** at least 100,000 randomized allocate/grow/share/COW/free/preempt operations with invariants checked after every operation, zero double ownership, zero negative refs, and final full reclamation.
- **Pressure/progress gate:** under a pool intentionally smaller than aggregate requested outputs, every admitted request completes or receives an explicit rejection within a fixed deadline; zero deadlocks, zero zero-progress allocation loops, and starvation bounded by an aging test.
- **Prefix-reuse gate:** repeated system-prompt and multi-turn traces must produce exact output parity, non-zero physical block sharing, and at least 30% lower prefill tokens plus improved p50 TTFT versus prefix caching disabled.
- **COW gate:** two requests sharing a prefix then diverging in the same partial block must retain identical shared-prefix KV, own distinct writable tails, and free all refs in both completion orders.
- **Eviction gate:** fill the pool with reusable prefixes, admit unrelated work, verify only zero-ref LRU/priority victims are evicted, then confirm misses recompute correctly without output drift.
- **Fragmentation telemetry gate:** reconcile bytes exactly: `pool = live_occupied + last_block_slack + reusable_cache + free + transfer_reserved`; report each term and reject negative or unaccounted bytes.
- **Swap/recompute gate:** measure GPU↔pinned-CPU round trip and prompt recomputation across `{256,512,1024,2048}` tokens on the deployed L4; do not enable offload unless it improves p95 resume latency or overload throughput by at least 10% without increasing p99 TTFT by more than 10%.
- **Performance gate:** dynamic allocation and full-block prefix indexing add no more than 5% decode-throughput regression on a zero-reuse workload; otherwise the metadata path is not production-worthy.

## Gaps

- No primary source establishes a universally optimal watermark, preemption victim, block size, or swap threshold; these are model, hardware, request-distribution, and SLO dependent.
- TensorRT-LLM documents several current constraints, including static division among heterogeneous cache pools and leaf-only eviction in its radix structure, so its feature set should not be treated as a finished reference design.
- HiCache's largest published wins use large multi-GPU models and fast storage/interconnects; they do not predict whether CPU offload helps this repository's 0.5B model on a Modal L4.
- Current vLLM and SGLang `main` contain 2026-era features beyond their original papers; implementation details are current as of the access date but may change rapidly.
- The repo has no real prefix-sharing workload trace, so proposed hit-rate and TTFT gates require adding deterministic repeated-prefix and multi-turn inputs before claims can be made.

## Counter-Claim

Dynamic block growth, prefix trees, and host offload are not automatically “more technical” or faster: vAttention demonstrates that contiguous virtual addressing can avoid paged-kernel complexity, SGLang notes cache-aware ordering can starve work, and both PagedAttention and TensorRT-LLM make swap value hardware-dependent; for this small single-model engine, the defensible first milestone is incremental GPU block ownership plus preemption/recomputation and full-block hash reuse, with radix scheduling, partial-block COW, and CPU/L3 tiering added only after workload-specific gates prove value. [10][5][1][9]
