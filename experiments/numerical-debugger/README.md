# Numerical divergence debugger

This extends the existing sentinel diagnostic rather than replacing the inference engine. It replays the original failing c8 request in natural order, target-first order, solo, and a fresh-process repeat, then runs an explicitly separate shared-prefix re-prefill probe. Each engine runs in a fresh child process on one Modal L4 allocation.

```bash
modal run debugger_app.py::numerical_debugger --output experiments/numerical-debugger/my-run
PYTHONPATH=src:. uv run python -m experiments.numerical_debugger \
  experiments/numerical-debugger/my-run/raw.json --output experiments/numerical-debugger/my-run
```

The GPU entrypoint refuses to overwrite retained raw evidence. Reports include JSON, Markdown, and standalone HTML. Capture provenance includes the source-tree fingerprint (including experiment code), pinned model revision, actual package versions, GPU UUID/driver and timestamps. The raw file retains all submitted prompt token IDs, target positions and engine options. Custom-engine captures also retain actual forward plans and sampled request IDs.

Comparisons stop at the first disagreement, inclusive: later free-running histories differ and are not comparable. Missing/non-finite/invalid top-k values, unverified captures and unequal output windows do not become successful comparisons. Log-probability differences use only the observed top-k intersection, never fabricated missing-token values. Exact ties may validly select any top-ranked token.

The shared-prefix probe appends the custom engine's first observed output token to the same target prompt in both engines, retaining peer input order. It is a fresh prefill with a changed target length, not teacher-forcing inside the original decode cache. It must not be used to claim the original scheduling/cache state was preserved.

## Scope

This is a three-token, one-target diagnostic, not a performance benchmark, quality evaluation, causal kernel proof, corruption exclusion, or Protocol V3 implementation. It preserves the original prompt content/order but uses the existing diagnostic runner's settings and fresh engines; the original pilot's warmup and full 128-token execution are not reproduced by this command. Original stopped pilot and sealed V2 results remain unchanged. No tolerances are tuned or correctness gate relaxed.

The legacy classifier and old analysis prose are not used to generate the new report. In particular, agreeing with one reference is not proof of correctness, and identical/different output sequences alone cannot prove/exclude KV contamination.

Local tests use synthetic fixtures only for report logic. Real validation must come from the Modal captures, not those tests.
