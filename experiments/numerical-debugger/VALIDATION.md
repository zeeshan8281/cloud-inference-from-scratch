# Real GPU validation — 2026-11-06

The debugger captured 10 fresh child-process runs on one physical NVIDIA L4, with zero crashes. Qwen2.5-3B revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`, FP16, Torch 2.7.1, Triton 3.3.1, vLLM 0.10.0, Transformers 4.53.2. Modal run: https://modal.com/apps/zeeshan8281/main/ap-VRQboc5s1wzIPBf7ZmedjE

## Measured findings

- Original c8 order: custom `[1401, 429, 3015]`; vLLM `[323, 264, 1401]`. This reproduces the original recorded target's first three tokens exactly.
- First difference: output position 0. Custom top-two log-probability margin `0.014797210693359819`; vLLM margin `0.0`. Cross-engine intersection max absolute log-probability difference `0.0094150` (rounded).
- Target-first and solo: exact token agreement over the three-token window. In target-first order, vLLM's first margin is `0.015625`.
- Custom original versus target-first: the three observed top-k lists are unchanged. vLLM changes its first token with order; first-step intersection max difference is `0.0089192` (rounded).
- Fresh-process repeats: both engines reproduce their own observed outputs; both engines' captured top-k intersection differences are zero at each repeated position.
- Shared-prefix re-prefill: appending custom's first token to the target prompt in both engines produces matching next-three-token outputs. This is a separate fresh-prefill diagnostic with changed target length, not original-cache teacher forcing.

## Checks performed

181 local tests ran: 179 passed with two existing skips (Redis-dependent coverage); lint and diff whitespace checks passed. Synthetic fixtures test report rejection of invalid/missing/non-finite captures, stopping at first divergence, unequal lengths, ties, escaped HTML, and deterministic report generation. They are not GPU evidence.

The actual GPU capture checked every emitted custom token against its observed argmax. The new reporter additionally checks selected tokens against finite, sorted, unique top-k data (allowing exact ties). It never imputes missing probabilities and never compares later free-running positions after the first difference.

The raw target IDs, natural batch order, and first-three-token outputs were checked against the existing committed diagnostic. GPU UUIDs were checked across all 10 cases. The recorded source fingerprint was reconstructed exactly using the retained executed source snapshots: `174715839aa256a2010b21a796da716cae1a46f4+tree-c57804e1356d`.

Raw capture SHA-256: `75ca9f11a67addf1f6eefeed8adff5648265dbf8c95c8c66254d31d88605dcf1`.

## Source and timestamp note

The executed versions of `debugger_app.py` and `modal_app.py` can be reconstructed by applying `executed-source.patch` to the delivered source; `source-manifest.json` records every fingerprint input. Full snapshots are also retained in the downloadable evidence bundle. After capture began, two host/integration-only corrections were made: copying the study-level environment instead of aliasing the first case's dictionary; and lazy-loading the shared subprocess helper so non-diagnostic Modal images do not need the sentinel module at import time. GPU capture and comparison logic are unchanged by those corrections. The complete final source passes local checks; the non-diagnostic deployed API was not redeployed or tested.

The unmodified raw capture retains the original host metadata: `case_environments.custom_original.finished_utc` was overwritten with the study end by that alias. Its true earlier timestamp remains in `first-case-environment.json`. Study start/end and the GPU/token/log-probability measurements are unaffected. No raw measurements were silently rewritten.

## Limits

One previously diagnosed target, three-token windows, one GPU, two repeats per engine in original order. This is not an independent quality holdout, full 128-token pilot rerun, full-vocabulary error bound, proof of a kernel defect, or proof of absent cross-request corruption. Diagnostic settings and warmup differ from the full pilot. No performance gain is claimed. Original pilot and sealed V2 correctness gates remain stopped; V3 was not run.

Earlier launch attempts were stopped after packaging/startup problems; they are not counted as successful GPU captures. Startup probes only establish device/package availability. The report and video use only the completed 10-case capture bundle.
