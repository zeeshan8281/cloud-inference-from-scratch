# Sentinel pilot findings: direct-engine closed-batch microbenchmark

Not an HTTP or production-serving benchmark. `c*` in a cell name is offered concurrency, not guaranteed simultaneous admission. All GPU memory figures are operational device footprint samples under each mode's own memory policy, not a general efficiency comparison except where the resource-normalized mode's matched KV capacity applies.

## resource_normalized

**STOPPED.** No performance claim may be generated from a stopped pilot.

Pair 01: stop kind `token_mismatch`.
4 of the pair's sentinel requests mismatched, confined to concurrency [8, 32] cells; first differing output position ranged 0-2 tokens in. A separate self-consistency check (see below) shows each engine reproduces its own output exactly across repeated runs, so this is not per-run randomness: the two engines compute deterministically different results from each other under concurrent batched execution at fp16, not a harness defect.

| Cell | Concurrency | Phase | Request | First diff position | Sequence length |
|---|---:|---|---:|---:|---:|
| `in512-out128-c8` | 8 | unique | 1 | 0 | 128 |
| `in512-out128-c8` | 8 | unique | 3 | 1 | 128 |
| `in1024-out256-c32` | 32 | unique | 7 | 2 | 256 |
| `in1024-out256-c32` | 32 | unique | 10 | 2 | 256 |

## complete_system

**STOPPED.** No performance claim may be generated from a stopped pilot.

Pair 01: stop kind `token_mismatch`.
6 of the pair's sentinel requests mismatched, confined to concurrency [8, 32] cells; first differing output position ranged 0-2 tokens in. A separate self-consistency check (see below) shows each engine reproduces its own output exactly across repeated runs, so this is not per-run randomness: the two engines compute deterministically different results from each other under concurrent batched execution at fp16, not a harness defect.

| Cell | Concurrency | Phase | Request | First diff position | Sequence length |
|---|---:|---|---:|---:|---:|
| `in512-out128-c8` | 8 | cold | 2 | 1 | 128 |
| `in512-out128-c8` | 8 | warm | 1 | 1 | 128 |
| `in1024-out256-c32` | 32 | cold | 18 | 1 | 256 |
| `in1024-out256-c32` | 32 | cold | 24 | 2 | 256 |
| `in1024-out256-c32` | 32 | warm | 5 | 2 | 256 |
| `in1024-out256-c32` | 32 | warm | 28 | 0 | 256 |

## Self-consistency diagnostic

Not part of the 10-pair protocol: each engine run twice, independently, in fresh subprocesses, on the identical materialized concurrency-8/32 workload, with the other engine entirely absent from the comparison.

- Custom engine self-consistent across repeated runs: **True** (0 mismatches).
- vLLM self-consistent across repeated runs: **True** (0 mismatches).

Both self-consistent means the token mismatches found above are not per-run randomness: the two engines deterministically compute different results from each other under concurrent batched execution at fp16, reproducibly, not due to noise.

