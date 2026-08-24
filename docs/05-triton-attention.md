# 05 — Triton direct-block decode attention

## Bottleneck

Paged allocation reduces reservation waste, but the PyTorch reference path gathers every physical block into contiguous logical K/V tensors before each attention call. Decode pays extra allocation and memory traffic.

## Exact change

`src/cloud_engine/attention.py::TritonDecodeAttentionBackend.attend` appends the new K/V and calls `src/cloud_engine/kernel.py::decode_attention_direct` for single-token decode. The kernel receives the query, physical layer pools, request block table, sequence length, and scale; it resolves each logical token to its physical block and slot without `torch.cat` or a full-cache gather.

```text
query token
    |
    v
block table [7, 2, 11] ----+
                            v
K/V physical pools -> Triton loads addressed blocks -> online softmax -> context
                            |
                            +-- no reconstructed logical K/V tensor
```

Prefill stays on the torch reference path. Unsupported dtype, head shape, block size, batch range, or context length fails before launch. Reference fallback occurs only when explicitly enabled.

## Correctness invariant

For batch sizes `1, 2, 8, 16` and boundary lengths through 2,048, Triton output must match the torch paged reference within `rtol=2e-2, atol=2e-2`. Decode must report zero full-cache gather temporary bytes.

## Measure it

```bash
modal run modal_app.py::remote_gpu_tests
modal run modal_app.py::benchmark --mode paged --profile decode --output artifacts/paged-decode.json
modal run modal_app.py::benchmark --mode triton --profile decode --output artifacts/triton-decode.json
```

The gate passed: Triton measured 21.4 output tok/s versus torch-paged at 19.5, a 1.10× ratio. The paged reference gathered 567.7 MiB over the decode workload; Triton gathered 2.63 MiB for prefill only and performed no full-cache decode gather. See [`paged-decode.json`](../artifacts/paged-decode.json) and [`triton-decode.json`](../artifacts/triton-decode.json).

## Remaining production shortcut

The first kernel targets the pinned Qwen shape, FP16, block size 16, a single L4, and decode batches up to 16. Prefill is not paged-kernel optimized, and the implementation has no autotuning, CUDA graphs, or multi-query fusion.
