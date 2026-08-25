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
one pinned Qwen model and one L4. Bounded prefix reuse has L4 parity and
work-reduction evidence, the same 20-check suite passes on A100, and exact packed
oracle parity passes on both Qwen2 and Llama-family checkpoints.
CUDA graph decode has exact-token and 3.24x paired L4 evidence. Quantization,
durable telemetry, tenant controls, and upgrade coverage remain open. This is
therefore not yet a 9/10 production engine.

## General inference devtool

- One-command setup, demo, benchmark, profile, and experiment flows.
- Strict public contract with a standard ecosystem benchmark client.
- Extension points for multiple policies and bounded runtime parameters.
- Machine-readable, source-identified, correctness-gated artifacts.
- Contributor documentation and stable examples across supported models.
- Commit-pinned CI across every supported Python minor line, including evidence
  hash and acceptance-threshold checks.

The demo, NVIDIA AIPerf runner, and GPU profiler are working. Experiments can
change scheduling order, preemption victim selection, active-sequence limits,
batch token budgets, and prefill chunk size; they emit raw and compact artifacts
only after correctness gates pass. Pinned Qwen2 and Llama-family examples both
have machine-readable exact-oracle GPU evidence. Every gate above now passes for
the declared greedy, text-only, single-GPU envelope, earning 9/10; the final
point requires independent adoption and multi-release history.

## NVIDIA-native experimentation workbench

- Triton CUDA kernel plus Torch oracle and numerical envelope.
- NVTX phase annotations and a real GPU timeline artifact.
- NVIDIA AIPerf native multi-run endpoint artifact.
- Same-hardware vLLM comparison with raw request records.
- Reproducible execution on a second NVIDIA target; DGX Spark is preferred but
  must not be claimed without GB10 evidence.

All five gates pass: the complete 20-check suite passed on both L4 (Ada) and
A100-SXM4-40GB (Ampere), with the A100 result committed as a source-identified
machine-readable artifact. This earns 9/10 for the declared NVIDIA cloud-GPU
workbench envelope. DGX Spark remains a documented, unverified ARM64/GB10 target;
claiming DGX support still requires correctness, AIPerf, and profiling evidence
from that device.
