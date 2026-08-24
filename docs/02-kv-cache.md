# 02 — Contiguous KV cache

## Bottleneck

Naive decoding recomputes keys and values for every prior token on every step. Their layer outputs do not change, so repeated projection and attention input construction wastes compute.

## Exact change

`src/cloud_engine/cache.py::ContiguousKVCache.reserve` allocates one `[max_sequence, kv_heads, head_dim]` key tensor and value tensor per layer and request. `append` writes new positions; `view` exposes only occupied positions. `src/cloud_engine/engine.py::CachedRunner.step` performs one prompt prefill, then feeds one previous output token per decode step. `StepContext.kv_start` keeps RoPE and cache positions aligned.

```text
request A, layer L
K: [prompt tokens | generated tokens |........unused capacity........]
V: [prompt tokens | generated tokens |........unused capacity........]
                    ^ occupied         ^ internal fragmentation
```

## Correctness invariant

The generated IDs must equal naive and the Hugging Face oracle. Each request has one reservation, cache writes use logical positions, and every terminal path calls `release`.

## Measure it

```bash
modal run modal_app.py::benchmark --mode naive --profile decode --output artifacts/naive-decode.json
modal run modal_app.py::benchmark --mode contiguous --profile decode --output artifacts/contiguous-decode.json
```

The fixed three-run protocol measured 27.5 output tok/s for contiguous versus 16.4 for naïve, passing the gate at 1.68×. Median decode TTFT fell from 12,973.7 ms to 9,276.0 ms. See [`contiguous-decode.json`](../artifacts/contiguous-decode.json).

## Remaining production shortcut

Worst-case per-request preallocation makes admission simple but wastes the unused tail. The mode is deliberately single-active-request so the next chapter can isolate scheduling throughput.
