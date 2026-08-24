# 03 — Continuous batching

## Bottleneck

Contiguous mode leaves the GPU serving one request at a time. A request that is decoding only contributes one new token of work, while new prompts wait for the whole request to finish.

## Exact change

`src/cloud_engine/scheduler.py::Scheduler._run` keeps one FIFO queue and one active list. Each iteration reaps timeouts/stalls, advances active sequences within `max_batched_tokens`, then admits and prefills waiting prompts within the remaining sequence and token capacity. New requests can join between decode iterations. `Request`, `RequestState`, and `_finalize` form the authoritative lifecycle.

```text
iteration N:   decode A | decode B | prefill C | publish
iteration N+1: decode A | decode B | decode C  | prefill D | publish
iteration N+2: reap B   | decode A | decode C  | decode D  | publish
```

`pending_events` plus a bounded `output_queue` provides lossless ordered delivery. A slow consumer stops only its request and is cancelled after the configured watchdog interval.

## Correctness invariant

Admission is FIFO; generated tokens never exceed the request cap; terminal transitions are final; cancellation is idempotent; every terminal path resolves the future and releases the runner. Local scheduler tests cover these paths.

## Measure it

```bash
modal run modal_app.py::benchmark --mode contiguous --profile decode --output artifacts/contiguous-decode.json
modal run modal_app.py::benchmark --mode batched --profile decode --output artifacts/batched-decode.json
```

Measurements are pending. The publication gate is at least `1.25x` aggregate median output throughput at concurrency eight.

## Remaining production shortcut

Scheduling is multi-request, but forwards remain per-sequence rather than one tensor batch. Prefill is not chunked, so an over-budget prompt waits for a fresh iteration. Preemption and fairness policies beyond FIFO are out of scope.

