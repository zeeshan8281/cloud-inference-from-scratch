# Divergence root-cause analysis

Bounded investigation only. Does not rerun the 10-pair protocol and does not
relax the correctness gate. Raw data: `divergence-diagnostic.json`. Target
request: `in512-out128-c8`, `request_index` 1, from
`raw/resource-normalized/pair-01.json` (the pair that stopped the pilot).

## Reproduction

The exact original mismatch reproduces byte-for-byte:

| | step 0 | step 1 | step 2 |
|---|---:|---:|---:|
| Custom engine (any configuration tested) | 1401 | 429 | 3015 |
| vLLM, original submission order (target at index 1 of 8) | 323 | 264 | 1401 |
| vLLM, target-first order (index 0 of 8), and every other config | 1401 | 429 | 3015 |
| HF Transformers reference, fp16 and fp32 | 1401 | 429 | 3015 |
| HF Transformers reference, bf16 | 323 | 264 | 1401 |

`raw/resource-normalized/pair-01.json`'s actual recorded output for this
request: custom `[1401, 429, 3015, ...]`, vLLM `[323, 264, 1401, ...]` --
matches exactly.

## What varies and what doesn't

Tested: concurrency 1/2/8/32 (target always first), the exact original
8-request batch in its natural submission order (target at its real
position, index 1), Triton vs Torch reference attention, CUDA graphs on/off,
prefix caching on/off, and HF at fp16/bf16/fp32 -- 18 runs, 0 crashes, every
custom-engine logit capture self-verified (argmax matches the emitted token).

- **The custom engine never diverges**, under any of the 9 configurations
  it was run in. Its per-step top-20 log-probabilities are numerically
  identical (matching to 6 decimal places) whether the target request is
  first or in its original position 1-of-8. It matches the HF fp16 and fp32
  reference exactly in every case.
- **vLLM diverges in exactly one condition**: the original submission order.
  Every other vLLM configuration (concurrency 1/2/8/32 with target first,
  CUDA graphs on, prefix caching on) matches the custom engine and the HF
  reference.

## Where it happens

Step 0 is where the two vLLM conditions actually separate:

| | vLLM, original order | vLLM, target-first order |
|---|---:|---:|
| Top-1 vs top-2 log-prob margin | **0.000000** (exact tie) | 0.015625 |
| Max abs log-prob difference vs. the other order | 0.0089 | 0.0089 |
| Cosine similarity vs. the other order's top-20 | 0.999998 | 0.999998 |
| Top-20 token overlap vs. the other order | 100% | 100% |

The custom engine's margin at this same step is 0.014797, **identical to six
decimal places** whether the target is first or at its original position.

This is the whole finding: batch order/composition perturbs vLLM's computed
log-probabilities for this request by about 0.009 in log-prob space --
extremely small, and the top-20 token identities and ranking barely move
(cosine similarity 0.999998, 100% top-20 overlap). But at step 0 the top two
candidate tokens for this specific request happen to be separated by almost
exactly that same tiny margin under the original order (0.0 -- an exact tie),
so that perturbation is enough to flip which token wins strict argmax.
Custom's margin at this position (0.0148) isn't close enough to zero for the
same-scale perturbation to matter, and the diagnostic found no evidence
custom has any comparable order sensitivity at all.

Steps 1 and 2's much larger differences (cosine similarity 0.60-0.63,
max abs diff ~10.5) are the expected downstream consequence of step 0's
divergence, not independent divergences: once vLLM emits token 323 instead
of 1401 at step 0, its step-1 forward pass is conditioned on a different
previous token entirely, so everything after is expected to differ. Layer
bisection was not performed -- per the preregistered rule, only warranted if
the output-logit and backend matrix couldn't isolate the cause, and here
they did, cleanly, at step 0.

## Classification

Per the preregistered decision rule:

- Custom matches the HF reference in every tested condition (fp16 and fp32) → not a custom-engine correctness bug.
- vLLM matches the HF reference and the custom engine in every condition **except** the original submission order → **vLLM backend/configuration effect**, specifically order/batch-composition-dependent, isolated to an exact top-1/top-2 tie at the first generated token.
- Not cross-request contamination: the target's *content* is identical across every run; only its *position/co-batched neighbors* changed, and only vLLM's answer moved.
- Not per-run randomness: both engines were independently confirmed self-consistent across repeated runs of the identical workload in the earlier self-consistency check.

**Verdict: `vllm_backend_or_configuration_effect`, mechanism = order/batch-composition-dependent numerical divergence, localized to a genuine near-tie at the first generated token.**

This is consistent with the custom engine and vLLM using different attention
kernels with different floating-point accumulation order across a batch;
whether *this specific* request lands on a tie is a property of exactly
which other requests it's batched with and in what order, not of concurrency
as a scalar count. That explains why the original resource-normalized and
complete-system pilot runs (natural request order) each hit a handful of
mismatches at c8/c32 and never at c1: c1 has no batch composition to vary.

## Exact reproduction command

```bash
modal run modal_app.py::sentinel_divergence_diagnostic
```

Reads `experiments/sentinel-pilot/raw/resource-normalized/pair-01.json`'s
target request deterministically (same call the pilot itself made:
`materialize_cell_workload("resource_normalized", 1, Cell(512, 128, 8),
"unique", tokenizer)`, `request_index` 1), runs the 18-configuration sweep,
and writes `summaries/divergence-diagnostic.json`. Regenerate this file from
that JSON with:

```bash
python3 -m experiments.sentinel_diagnostics  # see classify_divergence, step_margin, compare_top_k
```

## Proposed next correctness protocol

**Superseded by `CORRECTNESS_PROTOCOL_V2.md`** (repo root). The sketch below
proposed comparing "the two engines' output-token log-probability ... for
the remainder of the sequence" after a first-token disagreement -- but once
two engines emit different tokens at a position, every later position is
conditioned on a different generated history for each engine, so that
comparison is contaminated by construction. `CORRECTNESS_PROTOCOL_V2.md`
replaces it: compare only while both engines share a generated prefix, and
if a later-position comparison is ever needed, teacher-force the same fixed
prefix into both engines as an explicitly separate diagnostic. The findings
above (reproduction, margins, classification) are unaffected and stand as
reported; only this proposal section had the flaw.

The current gate (bit-exact token-ID match, any concurrency) is not
achievable between these two engines whenever a batch happens to contain a
near-tied logit for some request -- which is a property of the workload and
batch composition, not a defect either engine's authors would consider a
bug. Recommended revision, to be preregistered and tested on **fresh holdout
prompts** (not the prompts used for this diagnosis) before any performance
measurement resumes:

1. Keep the bit-exact gate at concurrency 1 (already always holds -- if it
   ever fails there, that *would* be a real bug).
2. At concurrency > 1, replace bit-exact matching with a bounded, named
   tolerance evaluated per request: exact match, OR (top-1/top-2 margin at
   the first divergent position was below a preregistered epsilon in at
   least one engine) AND (the two engines' output-token log-probability
   under the *other* engine's own distribution stays within a preregistered
   bound for the remainder of the sequence). This directly encodes "this was
   a genuine near-tie, not a wrong answer" rather than picking an arbitrary
   token-count tolerance.
3. Report, per pair, how many requests needed the tolerance clause and at
   what margin -- never silently absorb this into a pass/fail bit.
4. Any epsilon chosen from this diagnosis's prompts must be validated against
   new, disjoint prompts before it governs a real performance run; a
   tolerance tuned and validated on the same data is not independent
   evidence that it generalizes.

No tolerance has been chosen or applied here. The stopped pilot's evidence
(`raw/resource-normalized/pair-01.json`, `raw/complete-policy/pair-01.json`)
remains the valid, retained result of the existing protocol.
