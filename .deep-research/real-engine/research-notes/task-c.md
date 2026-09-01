---
task_id: c
role: GPU Inference Kernel and Runtime Researcher
status: complete
sources_found: 5
---

## Sources

[1] FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving | https://openreview.net/pdf/7f47176a6913ff4d00c2d8cb9f8b9d53cd095373.pdf | Source-Type: academic | Accessibility: public | As Of: 2025-04 | Authority: 10/10
[2] FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | https://arxiv.org/abs/2407.08608 | Source-Type: academic | Accessibility: public | As Of: 2024-07 | Authority: 10/10
[3] CUDA Graphs - vLLM Design Documentation | https://docs.vllm.ai/en/latest/design/cuda_graphs/ | Source-Type: official | Accessibility: public | As Of: 2026-08 | Authority: 9/10
[4] CUDA Programming Guide: CUDA Graphs | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html | Source-Type: official | Accessibility: public | As Of: 2026-08 | Authority: 10/10
[5] Fused Attention - Triton Documentation | https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html | Source-Type: official | Accessibility: public | As Of: 2026-08 | Authority: 9/10

## Findings

- Real continuous batching requires one ragged GPU execution over multiple requests, because FlashInfer treats changing per-request query/KV lengths as kernel inputs and schedules work tiles across the whole batch rather than issuing sequential request-level forwards. [1]
- The minimum serious decode path is a device-resident interface `Q[B,Hq,D] + block_table[B,max_pages] + seq_lens[B] -> O[B,Hq,D]` invoked once per layer and iteration, with metadata prepared once in a reusable workspace rather than Python lists and new GPU tensors constructed on every launch. [1][4]
- Paged prefill must consume page tables directly and implement tiled online-softmax over ragged query and KV spans without reconstructing contiguous K/V, since FlashInfer uses block-sparse storage as the common representation for both prefill and batched decoding. [1]
- Variable-length batches need a planning stage that splits long KV ranges into chunks, load-balances chunks across CTAs, and performs deterministic attention-output contraction; FlashInfer explicitly avoids atomic aggregation to preserve deterministic outputs. [1]
- Kernel specialization and autotuning should key on at least query length, KV-length bucket, head dimension, page size, dtype, causal mode, and GPU capability, following Triton's use of pruned `BLOCK_M`, `BLOCK_N`, warp, and pipeline-stage configurations rather than one fixed launch configuration. [1][5]
- CUDA Graph integration needs fixed-capacity input/output/metadata buffers plus captured batch-shape buckets and eager fallback, because vLLM dispatches graphs by a compact batch descriptor and separates full uniform-decode capture from piecewise prefill/mixed execution. [3]
- Graph-safe KV management must preserve pointer addresses across replay, update lengths/page-table contents in place, and launch/upload on a consistent stream; CUDA documents fixed graph allocation addresses, expensive remapping conditions, and lifetime errors that Compute Sanitizer can detect. [4]
- Numerical validation must compare every optimized kernel against an FP32 or trusted Torch reference over ragged batches, GQA head mappings, page-boundary lengths, long contexts, and adversarial logits, with Triton's official attention test using FP16 forward `atol=1e-2, rtol=0` as a baseline rather than relying only on final text parity. [5]
- Performance acceptance should require one fused batched launch per layer, zero full-cache gather bytes, lower median and p99 decode-step latency than the current Triton path at batch sizes 1/2/4/8/16, and end-to-end throughput that improves as active batch size grows; FlashInfer's published evaluation measures both kernel utilization and end-to-end TTFT/ITL because kernel-only wins can disappear at system level. [1]
- FlashAttention-3's TMA, warp-specialized, and FP8 results are Hopper-specific, so an L4 implementation should first prove coalesced paged loads, online softmax, balanced CTA work, FP32 accumulation, and launch-overhead reduction before attempting hardware features that the deployed GPU does not expose. [2][4]

## Deep Read Notes

### Source [1]: FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving
Key data: The MLSys 2025 paper reports 29-69% ITL reduction, 28-30% long-context latency reduction, and 13-17% parallel-generation speedup; its runtime separates `plan(sequence_lengths)` from graph-captured `run(...)` and stores schedules/partials in a caller-owned workspace.
Key insight: A production attention kernel is a compiler/runtime pair: block-sparse KV format, specialized kernel, and deterministic dynamic work scheduler are all required to handle ragged batches efficiently.
Useful for: Implement `step_batch`, a persistent GPU metadata/workspace arena, fused paged decode, paged prefill, and deterministic split-KV contraction.

### Source [2]: FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
Key data: On H100, FA3 reports 1.5-2.0x FP16 forward speedup over FA2, up to 740 TFLOP/s at 75% utilization, nearly 1.2 PFLOP/s in FP8, and 2.6x lower FP8 numerical error than a baseline FP8 implementation.
Key insight: The large gains come from architecture-specific overlap of Tensor Cores and TMA, warp specialization, interleaved matmul/softmax, and numerically careful low precision—not from expressing attention in a GPU DSL alone.
Useful for: Set the ambition and profiling method, while explicitly excluding Hopper-only techniques from the first L4 milestone.

### Source [3]: CUDA Graphs - vLLM Design Documentation
Key data: vLLM distinguishes `NONE`, `PIECEWISE`, `FULL`, `FULL_DECODE_ONLY`, and `FULL_AND_PIECEWISE`, dispatching with `BatchDescriptor(num_tokens, num_reqs, uniform, has_lora)` and falling back when a graph key is unavailable.
Key insight: Graph capture is a batch-runtime feature, not a decorator around `model.forward`; decode, prefill, and mixed batches have different capture compatibility and memory/startup costs.
Useful for: Implement shape buckets such as B=1/2/4/8/16, padded static metadata buffers, capture warm-up, replay dispatch, and eager fallback.

### Source [4]: CUDA Programming Guide: CUDA Graphs
Key data: CUDA graphs amortize per-kernel host launch setup; graph allocations have fixed virtual addresses, may reuse physical memory when lifetimes do not overlap, and can incur costly remapping when launch streams change or live allocations overlap.
Key insight: Dynamic request state must be represented as in-place data updates behind stable pointers, while allocation/free and Python-created temporaries stay outside the replayed decode graph.
Useful for: Design a graph-safe KV/page-table arena, stable output/logit buffers, consistent stream ownership, graph upload, memory accounting, and Compute Sanitizer checks.

### Source [5]: Fused Attention - Triton Documentation
Key data: The official kernel keeps Q in SRAM, iterates K/V tiles with FP32 max/normalizer/accumulator state, autotunes block sizes/warps/stages, prunes invalid configurations, verifies FP16 output at `atol=1e-2`, and benchmarks with `triton.testing.do_bench`.
Key insight: Correctness envelopes, layout constraints, autotuning keys, and reproducible microbenchmarks are part of the kernel implementation, not optional polish.
Useful for: Build reference parity tests, an autotune cache per L4 shape family, and isolated kernel benchmarks before end-to-end gates.

## Gaps

- No public primary source found a tuned paged-prefill Triton implementation specifically for NVIDIA L4/Ada with Qwen2.5-0.5B's exact GQA shape, so achievable thresholds must be established by local Nsight/benchmark measurements rather than borrowed H100 numbers.
- The strongest counter-claim is that writing every kernel from scratch may make the repository look technical while producing a slower and less reliable engine; the defensible scope is one original, measured L4 specialization with FlashInfer/vLLM used only as external baselines, not a broad reimplementation of their feature matrices.

## END
