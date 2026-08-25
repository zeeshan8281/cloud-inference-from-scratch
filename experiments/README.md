# Experiment template

Copy `starter.py`, edit its `priority(candidate)` function, then run:

```bash
modal run experiment.py --experiment experiments/starter.py
```

The runner uses the pinned Qwen2.5-3B model on one Modal L4. It first checks
baseline/experiment token parity, packed execution, failures, and KV cleanup.
Only then does it run the same deterministic online workload against both
policies and write a source-identified JSON artifact under `artifacts/`.

`candidate` exposes only `request_id`, `phase`, `remaining_tokens`,
`prompt_tokens`, `generated_tokens`, and `arrival_ns`. The scheduler continues
to enforce the shared token budget, chunking, cache capacity, and lifecycle.
