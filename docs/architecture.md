# Architecture

## Boundaries

`api -> engine -> model -> attention/cache`; `benchmarks -> engine`; `modal_app -> api + benchmarks`. FastAPI is imported lazily, and Modal is confined to `modal_app.py`, so local allocator, scheduler, validation, and metrics tests need no ML or web packages.

```text
POST /v1/responses
        |
        v
 validate + auth
        |
        v
InferenceEngine.submit
        |
        v
   FIFO waiting --queue timeout--> timed_out
        |
        v
      prefill --------------------> failed
        |
        v
     decoding --EOS/token cap----> completed
        |  |
        |  +--disconnect/stall---> cancelled
        +------------------------> failed

Every terminal state -> release cache -> append stream STOP -> resolve future
```

`InferenceEngine` is the composition root. It selects `NaiveRunner` or `CachedRunner`, one cache backend, and one attention backend from the server-side `EngineConfig`. `Scheduler._finalize` is the only terminal cleanup funnel. `RequestHandle` exposes streaming, waiting, and cancellation without leaking cache or scheduler internals into HTTP handlers.

## One scheduling iteration

```text
reap waiting timeouts + stalled consumers
                  |
                  v
decode one token for active requests within token budget
                  |
                  v
drain pending stream events; stalled request does no new work
                  |
                  v
admit FIFO requests while sequence + prompt-token budgets permit
                  |
                  v
prefill admitted requests -> sample/publish first token -> repeat
```

Tokens first enter a per-request `pending_events` backlog and then a bounded `asyncio.Queue`. This preserves ordering and avoids token loss. A full queue stalls only that request; ten seconds without progress cancels it and releases its cache.

## Ownership and failure behavior

- The scheduler owns request state; terminal transitions are irreversible.
- The engine owns the selected cache; paged modes have no shadow KV cache.
- `BlockAllocator` validates an entire free operation before mutating ownership.
- Weight loading uses an explicit tensor map and rejects missing, duplicate, unexpected, or wrongly shaped tensors.
- Triton decode fails closed for unsupported shapes unless reference fallback is explicitly enabled.
- API mode, model revision, cache budget, and hardware are server-side values.

## Deliberate shortcuts

Paged admission reserves `prompt + max_output_tokens` blocks up front. This prevents mid-generation OOM without adding eviction or preemption, at the cost of reserved-but-unused memory. Scheduler iterations span requests, but each forward is currently B=1. `naive` and `contiguous` therefore clamp active sequences to one, while later modes admit up to the configured limit. Paged prefill and `paged` decode gather a temporary logical tensor; only `triton` decode reads physical blocks directly.

These ceilings are measured rather than hidden. Add dynamic block growth, true tensor batching, or paged prefill only after the current correctness matrix and benchmark gates pass.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
modal run modal_app.py::remote_gpu_tests       # billable L4
modal run modal_app.py::api_lifecycle_tests
```

Verified 2026-08-24: 45/45 local tests, 45/45 Modal CPU tests plus real FastAPI route/auth checks, and 34/34 L4 checks passed. The deployed Triton API also passed live health, unauthorized rejection, blocking JSON, ordered SSE, metrics, and token-accounting checks.
