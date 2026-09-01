# Invalid vLLM latency restarts

The three `vllm-complete-*-pre-step-timestamps` files have valid output tokens,
throughput, request counts, and memory samples, but invalid TTFT/ITL/E2E timing:
the harness timestamped immediately before `engine.step()` instead of after it
returned. They are excluded from every summary and plot.

Only the vLLM arm was rerun after this correction. The custom engine already
timestamps emitted token events after execution, and its server/runtime source,
model, workload, warm-up, and environment did not change. Raw metadata retains
the distinct harness tree identities so this correction is auditable.
