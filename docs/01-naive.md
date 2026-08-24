# 01 — Naive autoregressive decoding

## Bottleneck

The baseline must make autoregressive cost visible: each output token needs the prompt and every earlier output token, so recomputing all transformer layers over the full sequence repeats work.

## Exact change

`src/cloud_engine/model.py::Qwen2CausalLM.forward` implements token embedding, RMSNorm, RoPE, grouped-query causal attention, SwiGLU layers, residuals, final norm, and the tied FP32 language-model projection. `src/cloud_engine/engine.py::NaiveRunner.step` rebuilds the full token list and stores no KV state. `src/cloud_engine/weights.py::load_state_dict` explicitly maps and shape-checks the pinned safetensors.

The Hugging Face model is permitted only in `modal_app.py::_reference_generate` and remote tests as an oracle.

## Correctness invariant

The custom model must match reference logits within `rtol=2e-2, atol=2e-2`, and greedy token IDs must match all later modes. Weight loading fails instead of silently accepting an architecture change.

## Measure it

```bash
modal run modal_app.py::smoke
modal run modal_app.py::benchmark --mode naive --profile decode --output artifacts/naive-decode.json
```

Before/after measurements are pending the first approved L4 run. The expected comparison is `naive` versus `contiguous`; no performance claim is published yet.

## Remaining production shortcut

This readable single-sequence path has no fused projections, fused MLP, CUDA graphs, quantization, or tensor parallelism. Those are intentionally out of scope until model parity is proven.

