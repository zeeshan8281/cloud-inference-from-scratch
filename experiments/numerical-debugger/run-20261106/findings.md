# Numerical divergence study

Unit-test fixtures are not study results; raw capture provenance is recorded below.

Raw SHA-256: `75ca9f11a67addf1f6eefeed8adff5648265dbf8c95c8c66254d31d88605dcf1`

Runs: 10; crashes: 0.

| Comparison | Status | First differing position | Left / right margin | Intersection max Δ log p |
|---|---|---:|---:|---:|
| Cross-engine / original | diverged | 0 | 0.0147972 / 0.0000000 | 0.0094150 |
| Cross-engine / target_first | exact_in_observed_window | — | 0.0777321 / 0.0781250 | 0.0106478 |
| Cross-engine / solo | exact_in_observed_window | — | 0.0760174 / 0.0937500 | 0.0115849 |
| Cross-engine / repeat | diverged | 0 | 0.0147972 / 0.0000000 | 0.0094150 |
| custom / order sensitivity | exact_in_observed_window | — | 0.0777321 / 0.0777321 | 0.0000000 |
| custom / repeatability | exact_in_observed_window | — | 0.0777321 / 0.0777321 | 0.0000000 |
| vllm / order sensitivity | diverged | 0 | 0.0000000 / 0.0156250 | 0.0089192 |
| vllm / repeatability | exact_in_observed_window | — | 0.0781250 / 0.0781250 | 0.0000000 |
| Cross-engine / shared-prefix re-prefill | exact_in_observed_window | — | 0.2210102 / 0.2187500 | 0.0105954 |

For exact windows, margin/drift describes the final compared step; otherwise the first disagreement.

## Environment
```json
{
  "started_utc": "2026-11-06T08:39:40.265871+00:00",
  "source": "174715839aa256a2010b21a796da716cae1a46f4+tree-c57804e1356d",
  "model": {
    "id": "Qwen/Qwen2.5-3B",
    "revision": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
    "dtype": "float16",
    "max_model_len": 4096,
    "max_output_tokens": 256,
    "eos_token_id": 151643
  },
  "gpu": {
    "uuid": "GPU-d795209d-2683-a09f-a1dc-5602fa1658bd",
    "name": "NVIDIA L4",
    "pci.bus_id": "[Unknown Error]",
    "driver_version": "610.57.04",
    "memory.total": "23034 MiB",
    "memory.used": "6 MiB",
    "memory.free": "22587 MiB",
    "power.draw": "17.82 W",
    "power.limit": "72.00 W",
    "temperature.gpu": "64",
    "pstate": "P8",
    "clocks.sm": "210 MHz",
    "clocks.mem": "405 MHz",
    "clocks_throttle_reasons.active": "0x0000000000000001",
    "cuda_version": "12.6",
    "utc_timestamp": "2026-11-06T08:39:40.319642+00:00",
    "monotonic_timestamp": 11.49112258
  },
  "packages": {
    "torch": "2.7.1",
    "triton": "3.3.1",
    "vllm": "0.10.0",
    "transformers": "4.53.2",
    "tokenizers": "0.21.2"
  },
  "window_tokens": 3,
  "scope": "original c8 prompt/order; diagnostic engine settings, fresh subprocesses",
  "finished_utc": "2026-11-06T08:45:25.142871+00:00",
  "gpu_identity_stable": true,
  "gpu_uuids": [
    "GPU-d795209d-2683-a09f-a1dc-5602fa1658bd"
  ]
}
```

## Boundaries
- One diagnosed sentinel request, not a new sealed holdout or quality certification.
- Top-k intersection metrics are not full-vocabulary error bounds.
- Observed batch sensitivity does not identify a causal kernel operation.
- Instrumentation and fresh-process diagnostic configuration differ from the full pilot.
- Shared-prefix probes re-prefill a modified target input; they do not preserve decode cache state.

unchanged; stopped pilot and V2 holdout remain stopped
