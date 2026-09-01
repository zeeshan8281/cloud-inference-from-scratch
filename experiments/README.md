# Experiment template

## Controlled vLLM paper experiment

Run the correctness gate, three clean-process matrix restarts, and supported
one-switch ablations with:

```bash
./experiments/reproduce.sh
```

The command writes exact input token IDs to `workloads.jsonl`, every raw request
record under `raw/`, median restart summaries to `summaries/results.csv`, and
the measured software/hardware identity to `environment.md`. It stops before
timing if the two runtimes differ on deterministic token IDs, workload hashes,
or the fixed environment fields.

The v1 ragged kernel supports only 16-token KV blocks. Paged-KV and eviction
ablations are recorded as excluded because neither can be disabled without also
changing another runtime component; legacy modes are not mislabeled as isolated
ablations.

## Scheduler policy template

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
