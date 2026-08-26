# Architecture

## Boundaries

`client -> api -> tenant gate -> engine -> scheduler -> model -> attention/cache`;
`benchmarks -> api`; `modal_app -> api + benchmarks`. FastAPI, Redis, and Modal stay at the edges, so local
allocator, scheduler, validation, and metrics tests do not need a GPU runtime.

```text
POST /v1/responses
        |
        v
 request ID + strict validation + bearer auth
        |
        v
 process-local gate (1 replica) or atomic Redis Lua gate (2–4 replicas)
        |
        v
InferenceEngine.submit
        |
        v
 FIFO waiting --queue timeout--> timed_out
        |
        v
 decode-first BatchPlan under one token budget
        |
        v
 demand allocation -> PackedBatch -> one flat model forward
        |
        v
 ragged Triton attention -> sampled request tails -> ordered SSE

Every terminal state -> release all owned blocks -> STOP -> resolve future
```

`InferenceEngine` is the composition root. The deployed `ragged` mode selects
`RaggedRunner`, one physical `PagedKVCache`, and
`RaggedTritonAttentionBackend`. The five earlier runners remain available as
measured baselines. `Scheduler._finalize` is the only terminal cleanup funnel.

## One ragged scheduling iteration

```text
reap timeouts + drain bounded output queues
                    |
                    v
schedule one decode token per runnable decoder first
                    |
                    v
spend remaining shared token budget on <=256-token prefill chunks
                    |
                    v
atomically grow every crossed KV block or preempt one resident sequence
                    |
                    v
flatten token IDs + positions + offsets + block tables + slot mapping
                    |
                    v
one Qwen2.5 forward for the complete BatchPlan
                    |
                    v
commit cache lengths -> sample only sequence-tail logits -> publish
```

`BatchPlan` is the scheduler contract. `PackedBatch` is the GPU contract. The
model sees one flat token axis; device metadata maps each query token to its
sequence and physical KV blocks. Decode and chunked prefill can coexist in the
same transformer invocation.

Pure decode batches are padded to the next power-of-two bucket. Shapes
1/2/4/8/16 are captured into CUDA Graphs before the engine becomes ready and
share one private memory pool; live traffic only copies device metadata and
replays a graph. The real logits are sliced back from padded rows before greedy
sampling. The L4 correctness suite validates exact tokens after this path, and
the online release gate rejects p99 ITL above 100 ms.

## KV ownership and pressure recovery

- Admission creates an empty block table; it does not reserve worst-case output.
- A repeat request may copy the longest block-aligned prompt prefix from a
  bounded LRU entry in the same physical pool. The final prompt token is always
  recomputed so its sampling logits stay exact.
- `ensure_capacity_batch` validates the whole growth operation before assigning
  blocks, so allocation failure cannot partially mutate a batch.
- K/V writes use a flat slot mapping; cache lengths commit only after a
  successful model call.
- Under pressure, the scheduler releases one resident sequence, resets its
  computed prefix, and later recomputes prompt plus already-emitted tokens.
- Generated tokens are never emitted twice and remain the deterministic source
  of truth during recomputation.
- Cancellation, timeout, backpressure failure, model failure, and success all
  converge on the same idempotent release path.
- Prefix entries are evicted before active requests are preempted and are
  accounted separately from request-owned blocks.

## Attention paths

The Torch ragged backend is the numerical oracle. The serving backend launches
one Triton program per flat query token and query head. It follows the
device-resident block table, reads the physical K/V pool directly, and performs
online FP32 softmax over contexts up to 4,096 tokens. Supported pinned head
dimensions are 64 and 128. No logical K/V tensor is reconstructed in the ragged
serving path.

The earlier `triton` mode remains deliberately serial at the model-forward
boundary. Its decode kernel now accepts the 3B model's 128-wide heads so the
online benchmark can compare old and new scheduling on the same model revision.

## Replicas, HTTP, and observability

`/livez` reports process liveness; `/readyz` and `/healthz` report model
readiness. Tenant-authenticated `/v1/models` and `/v1/responses` coexist with
admin-only JSON and Prometheus metrics. Every response carries a bounded request
ID and defensive response headers. Streaming uses bounded per-request queues with
strict event ordering. The online benchmark drives the real authenticated SSE
route over warm HTTP and records raw request TTFT, ITL, E2E, token counts,
errors, queue samples, throughput, SLO goodput, preemption, and recomputation.

One replica uses the in-process tenant gate. Multi-replica deployment requires
`ADMISSION_REDIS_URL`; a Lua transaction atomically cleans expired leases,
checks concurrent requests and rolling reserved tokens, and creates the lease.
Release and rollback are visible across clients. Redis failure returns 503, and
`ENGINE_MAX_CONTAINERS>1` without Redis fails startup. This coordinates API
admission across workers; model execution itself remains single-GPU per worker.

## Verification

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
REDIS_TEST_URL=redis://127.0.0.1:6379/15 uv run python -m unittest tests.test_redis_admission
modal run modal_app.py::api_lifecycle_tests
modal run modal_app.py::remote_ragged_gpu_tests
modal run modal_app.py::online_compare --rates 0.5,1,2,4 --duration-seconds 10
```

See [the Ragged engine chapter](06-ragged-engine.md) and the committed raw
[three-way online artifact](../artifacts/ragged-vllm-online.json).
