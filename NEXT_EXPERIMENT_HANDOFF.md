# Handoff: paired sentinel pilot

## Objective

Starting from commit `a0c189992a59226a4af7e7ed6e41cf90ce5d723d`, implement and run the next controlled experiment: a fixed 10-pair, direct-engine closed-batch pilot comparing the custom engine with vLLM 0.10.0 on one NVIDIA L4.

Do not rerun the full nine-cell matrix yet. Do not describe this as an HTTP or production-serving benchmark.

## Established baseline

- The existing run used real vLLM `LLMEngine`.
- Its primary result was an unweighted median of nine cell-wise throughput ratios: `0.590×` vLLM, range `0.476×–0.741×`.
- vLLM was faster in every complete-system cell.
- Tested deterministic outputs matched exactly.
- Both complete systems had zero request failures.
- All 255 timeouts were confined to custom ablations.
- Existing memory values are operational footprints under unequal KV-memory policies, not an efficiency comparison.
- Existing prompts repeat four token IDs and strongly favor prefix reuse.
- The existing source digest is attribution, not execution attestation.

Preserve the old artifacts unchanged.

## Phase 0: fix evidence code without using a GPU

Before running the pilot:

1. Make custom/vLLM plot series order explicit in `experiments/report.py`; never derive display order from a set.
2. Add a test that regenerates the plot under at least two `PYTHONHASHSEED` values and requires byte-identical output.
3. Add focused tests for:
   - vLLM timestamps being recorded after `engine.step()` returns;
   - correct accounting when one vLLM step exposes multiple tokens;
   - exclusion of `raw/invalid/` from summaries;
   - raw-to-summary arithmetic using a small golden fixture.
4. Rename new reports and headings to **direct-engine closed-batch microbenchmark**. Label c32 as offered concurrency and memory as operational device footprint.
5. Keep timeout counts beside every ablation/effect value and report per-cell effects.

Commit these changes before spending GPU time. Run the pilot only from a clean checkout of that commit.

## Fixed pilot protocol

### Sentinel cells

Each engine child must run these three cells:

| Input tokens | Output tokens | Offered concurrency |
|---:|---:|---:|
| 128 | 128 | 1 |
| 512 | 128 | 8 |
| 1,024 | 256 | 32 |

Use Qwen2.5-3B revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`, FP16, greedy decoding, `temperature=0`, `ignore_eos=true`, and the existing 4,096-token model limit.

### Pairing and order

- Run exactly 10 paired rounds per comparison mode.
- One pair is one custom child and one vLLM child, run sequentially inside the same parent Modal GPU allocation.
- Each child runs all three sentinel cells.
- Odd pairs: custom then vLLM. Even pairs: vLLM then custom.
- Each engine execution must be a fresh isolated child process.
- Do not replace or silently omit a failed pair. Persist it and stop under the rules below.
- Do not add trials after inspecting results. Any larger sample size belongs to a new preregistered run.

The current `_controlled_custom.remote()` / `_controlled_vllm.remote()` split does not satisfy pairing because it can use different workers. Add one parent GPU function, using one image capable of running both engines, that launches the two child processes sequentially.

### GPU identity and state

Record GPU state before the first child, between children, and after the second child in every pair:

- GPU UUID, name and PCI bus ID;
- driver and CUDA versions;
- total, used and free device memory;
- power draw/limit, temperature, P-state and SM/memory clocks;
- available clock-throttle reasons;
- UTC timestamp and monotonic timestamp.

The UUID must remain identical throughout the parent allocation. Persist the Modal app/call identifiers when available.

### Prompts

- Replace the four-token cyclic prompts with deterministic, high-entropy token-ID sequences.
- Derive each sequence from a recorded seed containing comparison mode, pair, cell, request index and phase.
- Exclude special/control tokens and ensure every measured request within a mode is unique unless it is deliberately replayed for the warm-cache measurement.
- Materialize the exact token IDs before either engine runs. Both engines must consume the same IDs and workload hash.
- Warmup prompts must be disjoint from measured cold-cache prompts.
- Persist seeds, token IDs, hashes and request order.

Do not use wall-clock randomness or tokenizer text generation during timing.

## Comparison modes

Run and report these as separate experiments. Never pool them into one headline ratio.

### A. Resource-normalized core

- Unique prompts.
- Prefix caching disabled on both engines.
- CUDA graphs disabled on both engines.
- Match scheduler limits, block size and maximum resident KV-token capacity.
- Apply one common total-device-memory headroom rule.
- Persist requested and resolved KV bytes, blocks and token capacity.
- Report PyTorch allocated/reserved memory and sampled device-used memory separately.

If vLLM 0.10.0 cannot be given an equivalent explicit KV capacity, stop and document the blocker; do not claim resource-normalized memory efficiency.

### B. Complete-system policy

Use each engine's documented complete graph/cache policy under the same GPU, model and no-OOM constraint.

For every child, report two measurements separately:

1. **Cold prefix cache:** after shape-matched compilation/graph warmup using disjoint prompts, measure prompts that have never been submitted to that process.
2. **Warm prefix cache:** prime the exact measured prompts outside the timed interval, then replay them with new request IDs.

The compilation/graph warmup is not the warm-cache measurement. Record eligible prefix tokens and cache hit, miss, copied/shared and evicted blocks for both engines. Persist each cell's realized CUDA-graph capture, replay or eager-fallback path. If equivalent cache accounting or graph-path attribution cannot be established, mark that metric unresolved rather than inferring it.

## Correctness and stop rules

Before performance collection, require exact generated-token-ID parity for every sentinel cell and cache mode.

Stop the pilot immediately and retain all evidence on any:

- token mismatch;
- crash, OOM or timeout;
- GPU UUID change;
- unexpected multi-token delivery that the timing code cannot account for;
- unresolved graph execution path;
- workload/hash mismatch;
- dirty or unidentified source tree.

No performance claim may be generated from a stopped pilot.

## Timing and primary analysis

- Use shape-matched warmup to convergence under a predeclared rule. A valid minimal rule is three warmups followed by additional warmups until the last three throughputs span no more than 3%, capped at ten; persist all warmups.
- Keep cell order fixed and identical inside both children, or predeclare one deterministic per-pair order and give both children that same order.
- Preserve request-level TTFT, per-token timestamps, E2E latency, output tokens, failures and raw wall time.
- Throughput is the primary cross-engine metric. Treat latency as descriptive until both clients expose equivalent delivery boundaries.

For each comparison mode and sentinel cell:

1. Compute `log(custom throughput / vLLM throughput)` within each of the 10 pairs.
2. Report all 10 raw paired ratios.
3. Report the geometric-mean ratio, the arithmetic mean and median ratio, and a two-sided 95% paired t-interval on the log ratios (`t=2.262`, 9 degrees of freedom), transformed back to ratio scale.
4. Report odd/even engine-order results separately as an order-sensitivity check.
5. Report every failure; do not calculate a winner from a censored subset.

Do not select a preferred cell, mode or estimator after seeing results.

## Provenance

Create a machine-readable manifest before execution containing:

- literal Git commit and tree IDs;
- dirty flag, which must be false;
- every benchmark source path and SHA-256;
- full command and arguments;
- resolved dependency lock and versions;
- Modal image identifier/digest;
- model revision and hashes of local model/tokenizer files;
- protocol version and all random seeds.

Hash every raw and generated artifact after execution. Source and result manifests must be sufficient to distinguish attribution from execution attestation.

## Artifact layout

Keep the pilot separate from the original matrix:

```text
experiments/sentinel-pilot/
├── README.md
├── protocol.json
├── source-manifest.json
├── workloads.jsonl
├── raw/
│   ├── resource-normalized/
│   └── complete-policy/
├── summaries/
│   ├── correctness.json
│   ├── paired-results.csv
│   ├── findings.md
│   └── exclusions.md
├── plots/
├── artifact-manifest.json
└── reproduce.sh
```

Raw files must include every warmup, request, pair/order marker, engine configuration, GPU-state sample, cache/graph counter, stdout/stderr path and failure.

## Acceptance criteria

The handoff is complete only when:

- Phase 0 tests and the existing suite pass.
- A clean committed source revision was used.
- Each completed comparison mode contains exactly 10 pairs for all three cells.
- Pair order alternates exactly and the GPU UUID is invariant within every pair.
- Input token IDs and output token IDs match between engines.
- Cold-cache and warm-cache complete-policy results are separate.
- Resource-normalized results use cache-off/graphs-off and matched KV capacity, or are explicitly blocked without a memory-efficiency claim.
- Raw records regenerate summaries and plots byte-for-byte with one command.
- Invalid or stopped runs are retained and clearly excluded.
- The final report states the benchmark boundary and does not claim production serving performance.

## Out of scope

- The full nine-cell rerun.
- HTTP/OpenAI-compatible serving benchmarks.
- Additional models or GPUs.
- Optimizing the custom engine after seeing pilot results.

Those are follow-ups only after this sentinel pilot establishes acceptable variance, pairing, instrumentation and provenance.
