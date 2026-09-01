---
task_id: a
role: LLM Inference Systems Foundations Specialist
status: complete
sources_found: 4
---

## Sources

[1] Efficient Memory Management for Large Language Model Serving with PagedAttention | https://arxiv.org/abs/2309.06180 | Source-Type: academic | Accessibility: public | As Of: 2023-09 | Authority: 10/10
[2] Orca: A Distributed Serving System for Transformer-Based Generative Models | https://www.usenix.org/system/files/osdi22-yu.pdf | Source-Type: academic | Accessibility: public | As Of: 2022-07 | Authority: 10/10
[3] Fused Attention — Triton documentation | https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html | Source-Type: official | Accessibility: public | As Of: 2026-08 | Authority: 9/10
[4] cloud-inference-from-scratch — project documentation and measured artifacts | https://github.com/zeeshan8281/cloud-inference-from-scratch | Source-Type: official | Accessibility: public | As Of: 2026-08-24 | Authority: 9/10

## Findings

- Autoregressive generation depends on all prior keys and values, so retaining the KV cache avoids recomputing unchanged prior-token projections and turns decode into incremental one-token work; the project makes this visible by comparing naïve full recomputation with a contiguous cache. [1][4]
- In the project's fixed decode benchmark, contiguous KV caching raised output throughput from 16.4 to 27.5 tokens/s, a measured 1.68× improvement, while preserving token equality against its oracle. [4]
- Continuous batching is iteration-level scheduling: after each generated-token iteration, finished requests can leave and newly arrived requests can join, reducing head-of-line waiting and allowing the active batch to change over time. [2][4]
- The project's batching stage is especially educational because it improved median time-to-first-token from 9,276.0 ms to 230.0 ms but reduced throughput from 27.5 to 21.9 tokens/s, showing that a scheduler alone does not create efficient tensor batching. [4]
- PagedAttention separates logical sequence order from physical KV placement with fixed-size blocks and per-request block tables, allowing caches to grow without reserving one maximum-length contiguous region per request. [1][4]
- vLLM reported that contiguous-serving baselines stored actual token state in only 20.4%–38.2% of reserved KV memory and that PagedAttention-based serving achieved 2–4× higher throughput at similar latency in its evaluated systems and workloads. [1]
- The project deliberately exposes the distinction between paged allocation and optimized paged attention: its PyTorch paged stage still gathers physical blocks into temporary contiguous tensors, whereas its Triton decode kernel follows block-table addresses directly. [4]
- Triton teaches the kernel-level mechanism behind efficient attention: tile K/V loads, keep query tiles and FP32 online-softmax state on chip, and accumulate the context without materializing the full attention matrix or reconstructing a contiguous cache. [3][4]
- In the project benchmark, direct-block Triton decode eliminated the 567.7 MiB full-cache decode gather seen in the PyTorch paged reference and improved median throughput by 9.7%, while prefill still used a 2.63 MiB gather. [4]
- Counter-claim: these techniques are not automatic wins—PagedAttention adds block-table and branching overhead, and the project's eager block reservation made paged fragmentation 4.45% worse; the benefit depends on dynamic allocation, real batching, workload lengths, and whether the system is memory-bound. [1][4]

## Deep Read Notes

### Source [1]: Efficient Memory Management for Large Language Model Serving with PagedAttention
Key data: A single OPT-13B request could require up to 1.6 GB of KV cache; prior systems used only 20.4%–38.2% of reserved KV memory; vLLM reported 2–4× throughput at similar latency.
Key insight: PagedAttention is a co-design of block-addressed attention, dynamic KV allocation, and scheduling—not merely storing tensors in blocks.
Useful for: explaining why KV memory controls concurrency and why the project's eager-reservation result is a useful negative demonstration.

### Source [2]: Orca
Key data: ORCA schedules one model iteration at a time and reported up to 36.9× throughput improvement over FasterTransformer at the same latency for its GPT-3 175B evaluation.
Key insight: continuous batching solves early-finished and late-joining request problems, but heterogeneous sequence lengths make efficient batching of attention operations non-trivial.
Useful for: explaining the project's scheduler and why per-sequence B=1 forwards can improve TTFT without improving aggregate throughput.

### Source [3]: Fused Attention — Triton documentation
Key data: The official kernel iterates over BLOCK_N K/V tiles, keeps an FP32 accumulator plus running maximum and normalization sum, and uses autotuned block sizes and warp/stage configurations.
Key insight: Triton exposes GPU tiling, pointer/address calculation, on-chip reuse, and numerically stable online softmax in a compact Python-embedded kernel language.
Useful for: explaining why the project's direct-block kernel removes cache-gather traffic and builds practical GPU-kernel skills beyond framework APIs.

## Gaps

- No independent third-party reproduction of this project's benchmark artifacts was found; project-specific performance numbers are self-reported but the repository exposes the commands and raw artifacts needed to reproduce them.
- The project intentionally omits production techniques including dynamic KV growth, preemption, chunked prefill, CUDA graphs, kernel autotuning, quantization, and distributed execution, so it teaches the mechanism rather than parity with vLLM-class serving.
- Alternative interpretation: the failed batching and paging gates could be read as incomplete optimization, but pedagogically they are valuable because they isolate the missing conditions—true tensor batching and on-demand block growth—that the original systems papers rely on.
