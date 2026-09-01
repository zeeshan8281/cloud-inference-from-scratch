# Excluded ablations

- The no-Triton run uses eager execution because the Torch reference backend cannot replay the Triton-oriented CUDA graph. Its isolated kernel effect must be computed against `no_cuda_graph`, not against `complete`.
- Paged KV off: excluded because the packed runner has no contiguous-KV backend; using the legacy batched mode would also change the runner, scheduler, and kernel.
- Eviction off: excluded because recompute preemption is not independently configurable and this matrix does not intentionally force KV pressure.
- KV block sizes 32 and 64: excluded because the v1 kernel supports only 16.
- Custom scheduler off: excluded because the scheduler owns the request lifecycle; `no_continuous_batching` only sets `max_active_sequences` to 1 and is not a scheduler-off run.
