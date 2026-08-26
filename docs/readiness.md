# 9/10 readiness gates

Scores in this project are earned by runnable evidence, not feature names. A
category reaches 9/10 only when every gate below passes on its declared support
envelope. A tenth point is reserved for sustained independent adoption and
multi-release operating history; a repository cannot manufacture that evidence.

## Production inference engine

- Correctness: oracle parity, deterministic replay, failure-path cleanup.
- Runtime: packed batching, chunked prefill, demand-paged KV, preemption.
- Reliability: overload, cancellation, timeout, restart, and soak coverage.
- Performance: standardized load tests, raw records, regression thresholds.
- Operations: authentication, bounded metrics, health, safe configuration.
- Breadth: at least two model families, quantized and FP16 paths, two NVIDIA
  hardware targets, and an upgrade-compatibility matrix.

Current evidence satisfies correctness, the core runtime, baseline operations,
standardized load measurement, and a 1,936-request cancellation/restart soak on
one pinned Qwen model and one L4. At 4 req/s, pre-captured CUDA Graph buckets
delivered 51.1 output tok/s (95.7% of pinned vLLM), 138.7 ms p99 TTFT, 80.0 ms
p99 ITL, zero errors, and 3.65 SLO-good req/s. Bounded prefix reuse has L4 parity and
work-reduction evidence, the same 20-check suite passes on A100, and exact packed
oracle parity passes on both Qwen2 and Llama-family checkpoints.
CUDA graph decode has exact-token and 3.24x paired L4 evidence. A candidate
PyTorch 2.13/CUDA 13, Triton 3.7, and Transformers 5.15 stack passed packed
exact-oracle and CUDA-graph execution on L4; the release stack remains pinned
until promotion. The opt-in MLP LLM.int8 capacity path passed its paired
Qwen2.5-3B/L4 gate over 2,040 hashed teacher-forced tokens and 48 generated
sequences: 1.0087x FP16 perplexity, 23.5% less steady GPU memory, packed batches
of 16, and 2.31x latency. Its eager-only execution and generated-token drift are
reported as capacity-mode trade-offs, not hidden. Secret-backed tenant policies enforce
per-tenant concurrency, rolling token budgets, and admin-only metrics; prompt-free
structured audit logs survive container teardown under Modal's plan retention
and can be exported through its OpenTelemetry integration. Multi-replica API
admission uses one atomic Redis Lua transaction, fails closed on backend loss,
and passed races across eight independent clients; deployments above one replica
are rejected unless that shared gate is configured. Every gate above now passes
for the declared replicated-API/single-GPU-worker, greedy text-generation envelope, earning
9/10; the tenth point still requires sustained independent adoption and
multi-release operating history.

## General inference devtool

- One-command setup, demo, benchmark, profile, and experiment flows.
- Strict public contract with a standard ecosystem benchmark client.
- Extension points for multiple policies and bounded runtime parameters.
- Machine-readable, source-identified, correctness-gated artifacts.
- Contributor documentation and stable examples across supported models.
- Commit-pinned CI across every supported Python minor line, including evidence
  hash and acceptance-threshold checks.

The zero-dependency `cie` CLI, browser operator console, one-command demo,
NVIDIA AIPerf runner, and GPU profiler are working. Experiments can
change scheduling order, preemption victim selection, active-sequence limits,
batch token budgets, and prefill chunk size; they emit raw and compact artifacts
only after correctness gates pass. Pinned Qwen2 and Llama-family examples both
have machine-readable exact-oracle GPU evidence. CI runs the contract and
artifact gates on Python 3.10/3.12/3.13 plus a real Redis service gate. Every gate above now passes for
the declared greedy, text-only, single-GPU envelope, earning 9/10; the final
point requires independent adoption and multi-release history.

## NVIDIA-native experimentation workbench

- Triton CUDA kernel plus Torch oracle and numerical envelope.
- NVTX phase annotations and a real GPU timeline artifact.
- NVIDIA AIPerf native multi-run endpoint artifact.
- Same-hardware vLLM comparison with raw request records.
- Reproducible execution on a second NVIDIA target; DGX Spark is preferred but
  must not be claimed without GB10 evidence.

All five gates pass: the current complete 20-check suite passes on L4 (Ada), an
earlier source-identified run passed the same suite on A100-SXM4-40GB (Ampere),
and the current online artifact records same-model/same-L4 raw requests against
vLLM 0.10.0. Ragged reaches 95.7% of vLLM output throughput at 4 req/s while
meeting the declared TTFT/ITL SLOs. The A100 result is committed as a
source-identified machine-readable artifact. This earns 9/10 for the declared
NVIDIA cloud-GPU workbench envelope. DGX Spark remains a documented, unverified
ARM64/GB10 target; claiming DGX support still requires correctness, AIPerf, and
profiling evidence from that device.
