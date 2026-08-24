# Stage 6: Packed Ragged Inference

The sixth stage replaces iteration-level request rotation with an actual flat,
multi-request transformer invocation. It keeps the earlier engines intact as
baselines and deploys the pinned `Qwen/Qwen2.5-3B` revision on one NVIDIA L4.

## Scheduler contract

Each iteration creates a `BatchPlan` under `max_batched_tokens=2048`:

1. one token from every runnable decode request, in FIFO order;
2. remaining capacity assigned to prefill in chunks no larger than 256 tokens;
3. no request appears twice in one plan; and
4. sampling occurs only when that request's scheduled query reaches its current
   sequence tail.

This makes decode latency independent of a concurrent 4,000-token prefill: the
prefill advances, but cannot occupy a later iteration before ready decode work.

## Packed GPU contract

`RaggedRunner` converts the plan to one `PackedBatch` containing:

- flat input IDs and absolute positions;
- query start offsets and per-sequence context/final lengths;
- device-resident padded physical block tables;
- one physical cache slot per flat token;
- one sequence row per query token; and
- logits indices only for requests that must sample.

The model is invoked once with that object. Every transformer layer consumes
the same ragged metadata, writes K/V through the slot mapping, and launches the
mixed prefill/decode Triton attention kernel.

## Transactional demand paging

Requests own empty page tables at admission. Before a packed forward, the cache
computes every additional 16-token block needed by the plan. It either assigns
all of them or none. Cache lengths advance only after the model call succeeds.

If the pool cannot grow, the scheduler releases one resident victim and later
recomputes its prompt plus already-generated prefix. Emitted token IDs remain
authoritative, so recovery does not duplicate or change client-visible output.

## Numerical and systems proof

The L4 suite checks exact greedy output against Hugging Face, a real forward
trace containing four request IDs, mixed-query Torch/Triton parity at batch
sizes 1/2/4/8/16, block boundaries and contexts through 4,096, decode priority
during a 4,000-token chunked prefill, and forced recomputation in a 5 MiB KV
pool. Every run ends with allocator invariants and zero leaked blocks.

The online proof uses the authenticated streaming HTTP path for both custom
engines and vLLM. All three use the same pinned model/revision, L4, deterministic
workload hash, arrival schedule, active-sequence limit, batch-token ceiling,
greedy decoding, and EOS behavior. The JSON artifact retains every request;
aggregate tables in the README are only a view of that evidence.

vLLM prefix caching is explicitly disabled so repeated prompts across rates do
not turn later sweeps into a different workload.

The custom engines use a fixed 4 GiB KV pool; vLLM uses its 0.85 GPU-memory
utilization policy. The online workload does not exhaust either pool, so the
artifact is not presented as a memory-normalized capacity comparison.

## Deliberate limits

This is a focused single-GPU engine, not a vLLM replacement. It has no CUDA
graphs, fused MLP/RMSNorm/RoPE kernels, prefix caching, speculative decoding,
quantization, swap/offload, distributed execution, or instruction-template
layer. The benchmark is expected to expose those missing optimizations rather
than conceal them.
