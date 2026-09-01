---
task_id: d
role: Inference Benchmark Methodology Researcher
status: complete
sources_found: 5
---

## Sources

[1] vLLM Benchmark CLI | https://docs.vllm.ai/en/latest/benchmarking/cli/ | Source-Type: official | Accessibility: public | As Of: 2026-08-24 | Authority: 9/10
[2] SGLang Benchmark and Profiling Guide | https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md | Source-Type: official | Accessibility: public | As Of: 2026-08 | Authority: 9/10
[3] NVIDIA GenAI-Perf | https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2500/user-guide/docs/perf_analyzer/genai-perf/README.html | Source-Type: official | Accessibility: public | As Of: 2024-06 | Authority: 9/10
[4] MLPerf Inference Benchmarks | https://docs.mlcommons.org/inference/ | Source-Type: official | Accessibility: public | As Of: 2026-08 | Authority: 10/10
[5] Cloud Inference Engine Lab | https://github.com/zeeshan8281/cloud-inference-from-scratch | Source-Type: official | Accessibility: public | As Of: 2026-08-24 | Authority: 8/10

## Findings

- A technically credible serving benchmark must drive the HTTP server under controlled arrival rates and concurrency rather than benchmark only an in-process runner or one static batch. [1][2][3]
- vLLM's current benchmark tooling separates request rate, burstiness, and maximum concurrency, allowing maximum-throughput, Poisson-arrival, burst, capacity-planning, and SLA-style workloads. [1]
- SGLang recommends its online serving benchmark by default and warns that a single-batch server test never reaches steady state and therefore produces biased metrics. [2]
- The core online metrics shared by vLLM, SGLang, NVIDIA GenAI-Perf, and MLPerf are request/output throughput, time to first token, time or latency per output token, inter-token latency, and percentile request latency. [1][2][3][4]
- A serious benchmark must sweep offered load and report the throughput-latency frontier because a single saturated tokens-per-second number cannot show queueing collapse or whether latency objectives remain satisfied. [1][4]
- vLLM supports probe requests that bypass the main concurrency cap, providing a direct method to measure interference imposed on small requests by heavy traffic. [1]
- NVIDIA GenAI-Perf records benchmark outputs in machine-readable CSV and JSON and supports synthetic or dataset-based lengths plus controlled request rate or concurrency, enabling repeatable cross-engine comparison. [3]
- MLPerf couples server throughput with explicit TTFT and TPOT constraints and an accuracy target, demonstrating that output speed without latency and correctness constraints is not sufficient evidence. [4]
- The current project has reproducible warmups, repeated runs, raw JSON, TTFT and throughput, but it lacks an arrival-process sweep, steady-state duration, p95/p99 under load, goodput under latency objectives, and same-protocol comparison against vLLM. [1][2][4][5]
- The minimum defensible gate is not simply beating vLLM overall; it is matching token correctness while showing a reproducible Pareto improvement or bounded overhead in one declared niche on the same model, GPU, prompts, request process, concurrency, and runtime versions. [1][3][4][5]

## Deep Read Notes

### Source [1]: vLLM Benchmark CLI
Key data: The tool controls request rate, Gamma-distributed burstiness, concurrency, ramp-up, probe traffic, datasets, warmups, and detailed per-request result export.
Key insight: Serving performance is a surface over offered load and workload composition, not one tokens-per-second scalar.
Useful for: defining arrival-rate sweeps, interference tests, machine-readable artifacts, and a same-protocol vLLM baseline.

### Source [2]: SGLang Benchmark and Profiling Guide
Key data: SGLang distinguishes online HTTP serving, single-batch server, offline scheduler, and kernel-only benchmarks and recommends at least five times as many prompts as maximum concurrency for steady-state online tests.
Key insight: Each benchmark layer answers a different question; kernel speed cannot be presented as scheduler or service throughput.
Useful for: requiring separate kernel microbenchmarks, engine throughput tests, and end-to-end online load tests.

### Source [4]: MLPerf Inference Benchmarks
Key data: MLPerf defines server scenarios with a standard load generator, accuracy requirements, and for Llama 2 70B a 2,000 ms TTFT and 200 ms TPOT latency constraint.
Key insight: A result is credible only when throughput, latency compliance, and output quality are evaluated together under a declared scenario.
Useful for: defining goodput gates and correctness-preserving comparisons instead of unconstrained peak throughput.

## Gaps

- MLPerf's large-model scenarios are too expensive and broad for this single-L4 project, so its methodology is useful but full compliance is not a practical target.
- Current benchmark tools and CLI details evolve quickly; commands must be pinned to exact versions in every artifact.
- Counter-claim: a custom benchmark harness may reveal engine internals better than standard tools, but without at least one compatible external load generator and identical vLLM comparison, reviewers can reasonably dismiss favorable results as harness-specific.

## END
