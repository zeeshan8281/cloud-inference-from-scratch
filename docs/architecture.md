# Architecture

## Boundaries

`api -> engine -> scheduler -> model -> attention/cache`; `benchmarks -> api`;
`modal_app -> api + benchmarks`. FastAPI and Modal stay at the edges, so local
allocator, scheduler, validation, and metrics tests do not need a GPU runtime.

```text
POST /v1/responses
        |
        v
 strict validation + bearer auth
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

## HTTP and observability

The API exposes public readiness plus authenticated blocking/streaming
Responses requests and metrics. Streaming uses bounded per-request queues with
strict event ordering. The online benchmark drives the real authenticated SSE
route over warm HTTP and records raw request TTFT, ITL, E2E, token counts,
errors, queue samples, throughput, SLO goodput, preemption, and recomputation.

## Verification

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
modal run modal_app.py::api_lifecycle_tests
modal run modal_app.py::remote_ragged_gpu_tests
modal run modal_app.py::online_compare --rates 0.5,1,2,4 --duration-seconds 10
```

See [the Ragged engine chapter](06-ragged-engine.md) and the committed raw
[three-way online artifact](../artifacts/ragged-vllm-online.json).
