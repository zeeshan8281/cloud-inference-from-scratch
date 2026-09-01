# Controlled experiment findings

All values are medians across the clean-process restarts in `results.csv`.

- Custom/vLLM median throughput ratio across the nine cells: 0.590×.
- Raw request errors: 255; all 255 were scheduler timeouts in slow ablations.
- Successful output-hash mismatches across servers, variants, and restarts: 0.
- Largest observed restart throughput range: 29.4%. Additional restarts were added for: no_cuda_graph.
- Maximum variation: `no_cuda_graph/in1024-out256-c1` (14.094-18.835 output tok/s). Two added restarts changed its variant's median effect from 0.768× to 0.769×.

## Complete-system comparison

| Workload | Custom output tok/s | vLLM output tok/s | Custom/vLLM |
|---|---:|---:|---:|
| `in1024-out256-c1` | 20.714 | 35.091 | 0.590× |
| `in1024-out256-c32` | 250.097 | 512.916 | 0.488× |
| `in1024-out256-c8` | 123.859 | 260.044 | 0.476× |
| `in128-out128-c1` | 24.754 | 33.387 | 0.741× |
| `in128-out128-c32` | 344.812 | 532.164 | 0.648× |
| `in128-out128-c8` | 178.986 | 268.881 | 0.666× |
| `in512-out128-c1` | 22.518 | 35.052 | 0.642× |
| `in512-out128-c32` | 292.509 | 517.048 | 0.566× |
| `in512-out128-c8` | 142.812 | 260.846 | 0.547× |

## Ablations

| Variant | Comparison | Median throughput ratio | Failure-free cells | Failed requests |
|---|---|---:|---:|---:|
| `no_continuous_batching` | `complete` | 0.998× | 5/9 | 207 |
| `no_cuda_graph` | `complete` | 0.769× | 9/9 | 0 |
| `no_prefix_reuse` | `complete` | 0.945× | 9/9 | 0 |
| `no_triton` | `no_cuda_graph` | 0.273× | 8/9 | 48 |
