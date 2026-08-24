# 04 — Paged KV allocation

## Bottleneck

One worst-case contiguous allocation per request reserves a long unused tail for short generations. Variable-length requests therefore waste KV memory and can limit concurrency.

## Exact change

`src/cloud_engine/cache.py::PagedKVCache` creates shared key and value pools shaped `[layers, blocks, 16, kv_heads, head_dim]`. `BlockAllocator` gives each request a logical-order block table. `append` maps token positions to physical block/slot indices; `view` gathers a temporary logical tensor for the PyTorch reference attention path.

```text
contiguous
A -> [A A A A A . . . . . . .]
B -> [B B . . . . . . . . . .]

paged physical pool
block 0 [B B . .] <- B table [0]
block 1 [A A A A] <- A table [1, 3]
block 2 [free]
block 3 [A . . .]
```

The pools are the only authoritative KV storage. The temporary gathered tensors live only for attention and their bytes are reported.

## Correctness invariant

Free plus allocated blocks always equals the pool size; a block has at most one owner; invalid and duplicate frees are atomic failures; terminal requests own zero blocks. Boundary tests cover positions around block edges.

## Measure it

```bash
modal run modal_app.py::benchmark --mode batched --profile fragmentation --output artifacts/batched-fragmentation.json
modal run modal_app.py::benchmark --mode paged --profile fragmentation --output artifacts/paged-fragmentation.json
```

The gate failed. Peak reserved-but-unused KV was 23.95 MiB for paged versus 22.93 MiB for batched: paged was 4.45% worse, not 40% better. Eager worst-case reservation removes the key memory benefit of on-demand paging, and 16-token block rounding adds waste. See [`batched-fragmentation.json`](../artifacts/batched-fragmentation.json) and [`paged-fragmentation.json`](../artifacts/paged-fragmentation.json).

## Remaining production shortcut

Admission eagerly reserves enough blocks for prompt plus maximum output. This guarantees no mid-generation OOM without eviction, but leaves block-granularity and unused-reservation waste. PyTorch attention gathers full logical K/V tensors, so this stage is **paged KV allocation**, not optimized paged attention.
