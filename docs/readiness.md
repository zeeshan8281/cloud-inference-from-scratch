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
and standardized load measurement on one pinned Qwen model and one L4. Prefix
reuse, quantization, CUDA graphs, model/hardware breadth, durable telemetry,
tenant controls, soak/chaos evidence, and upgrade coverage remain open. This is
therefore not yet a 9/10 production engine.

## General inference devtool

- One-command setup, demo, benchmark, profile, and experiment flows.
- Strict public contract with a standard ecosystem benchmark client.
- Extension points for multiple policies and bounded runtime parameters.
- Machine-readable, source-identified, correctness-gated artifacts.
- Contributor documentation and stable examples across supported models.

The demo, scheduler template, NVIDIA AIPerf runner, and GPU profiler are working.
The extension surface is still scheduler-priority-only and model support is
narrow, so this category is not yet 9/10.

## NVIDIA-native experimentation workbench

- Triton CUDA kernel plus Torch oracle and numerical envelope.
- NVTX phase annotations and a real GPU timeline artifact.
- NVIDIA AIPerf native multi-run endpoint artifact.
- Same-hardware vLLM comparison with raw request records.
- Reproducible execution on a second NVIDIA target, ideally DGX Spark.

The first four gates pass on L4. DGX Spark is a documented target, not a verified
claim: its ARM64/GB10 execution must produce the same correctness, AIPerf, and
profiling artifacts before this category reaches 9/10.
