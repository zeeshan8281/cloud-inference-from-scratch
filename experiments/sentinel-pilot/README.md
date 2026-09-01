# Sentinel pilot: direct-engine closed-batch microbenchmark

This is a **direct-engine closed-batch microbenchmark**, not an HTTP or
production-serving benchmark. It compares the custom engine and vLLM 0.10.0
by calling each engine's Python interface directly (no HTTP server, no load
generator) over a fixed, small set of closed-loop request batches on one
NVIDIA L4.

It is separate from and does not replace `experiments/` (the original
nine-cell matrix). Those artifacts are preserved unchanged; see
`../summaries/findings.md` for that run's own results.

## Scope

- Three fixed sentinel cells: `in128-out128-c1`, `in512-out128-c8`,
  `in1024-out256-c32`. **`c*` is offered concurrency** — the number of
  requests submitted at once, not a guarantee that every engine admits all of
  them into its running batch simultaneously (both engines cap concurrent
  admission below 32 by their own scheduler policy in the complete-system
  mode; see each run's `engine_config`).
- Two comparison modes, reported and never pooled together:
  - **Resource-normalized core**: prefix caching and CUDA graphs off on both
    engines, matched scheduler limits, block size, and KV-token capacity.
  - **Complete-system policy**: each engine's own documented default
    graph/cache policy, with cold- and warm-prefix-cache measured separately.
- 10 paired rounds per mode; pair order alternates which engine runs first.
- All GPU memory figures are **operational device footprint** samples
  (`nvidia-smi` used memory, plus PyTorch allocated/reserved bytes) under each
  mode's own memory policy — not a resource-normalized efficiency comparison
  except where mode A's matched KV capacity applies.

## Out of scope

The full nine-cell rerun, HTTP/OpenAI-compatible serving benchmarks,
additional models or GPUs, and any custom-engine optimization made in
response to this pilot's results. See `../../NEXT_EXPERIMENT_HANDOFF.md`.

## Reproducing

```bash
./reproduce.sh
```

Regenerates `summaries/`, `plots/`, and `artifact-manifest.json` from
`raw/` byte-for-byte. It does not re-run the GPU pilot; that is a separate,
billable step (`modal run modal_app.py::sentinel_pilot`).

## Layout

```text
experiments/sentinel-pilot/
├── README.md
├── protocol.json           # frozen protocol constants for this run
├── source-manifest.json    # git commit/tree, dirty flag, source SHA-256s
├── workloads.jsonl         # every materialized workload, one line per (mode, pair)
├── raw/
│   ├── resource-normalized/   # one file per pair: both children + GPU states
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
