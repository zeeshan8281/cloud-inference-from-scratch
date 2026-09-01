# Correctness Protocol V2

Status: **calibration and sealed holdout both executed; requirement-10 gate
does not pass (2 hard failures on holdout)**. Performance measurement
remains paused -- see "Sealed holdout results" below. The original stopped
pilot
(`experiments/sentinel-pilot/raw/resource-normalized/pair-01.json`,
`experiments/sentinel-pilot/raw/complete-policy/pair-01.json`) is preserved
unchanged and remains the valid result of the protocol that produced it.
`experiments/sentinel-pilot/summaries/divergence-analysis.md`'s bounded
root-cause diagnostic is complete and is not reopened by this document.

This supersedes the "Proposed next correctness protocol" sketch at the end
of `divergence-analysis.md`, which had a real flaw: it proposed comparing
"the two engines' output-token log-probability ... for the remainder of the
sequence" after a first-token disagreement. Once two engines emit different
tokens at a position, every later position is conditioned on a different
generated history for each engine -- their subsequent distributions are not
comparable without forcing them onto the same prefix first. That comparison
was contaminated by construction and must not be used.

## Why a V2 is needed

The bounded diagnostic (`divergence-analysis.md`) established that this
codebase's custom engine and vLLM 0.10.0 can compute a deterministically
different argmax at concurrency > 1 for a request whose top-1/top-2 logit
margin is a near-exact tie, driven by batch composition/order rather than
any config knob, precision setting, or per-run randomness. The mechanism is
plausible (different attention kernels, different floating-point
accumulation order across a batch) but not proven at the source level --
this diagnostic isolated *where* the disagreement originates (the first
divergent token) and *what kind* of disagreement it is (a genuine near-tie,
not a wrong answer with a wide margin), not the exact kernel operation
responsible. The existing sentinel pilot's gate (bit-exact token-ID equality
at any concurrency) cannot distinguish "genuine near-tie, either answer
defensible" from "one engine is actually wrong," and stops on both
identically. V2 exists to make that distinction explicit and reportable
instead of collapsing it into one pass/fail bit.

## Requirements

### 1. Concurrency 1 stays bit-exact

No change from the existing protocol. If two engines disagree at
concurrency 1 (no batch composition to vary), that is a correctness bug in
one of them, full stop -- there is no near-tie-across-batches explanation
available at concurrency 1, and no tolerance applies there.

### 2. At concurrency > 1, compare only while both engines share the same generated prefix

Do not compare token N of engine A against token N of engine B once they
have already diverged at some earlier position. The comparison at any
position beyond the first disagreement is meaningless unless both engines
were forced onto an identical prefix up to that point (see requirement 4).

### 3. At the first token disagreement, capture full diagnostic detail

For every request where the two engines' greedy tokens first differ,
retain, before doing anything else:

- both engines' top-k logits/log-probabilities at that position;
- the top-1/top-2 margin for each engine independently;
- max absolute logit (or log-probability) difference between the two
  engines' distributions at that position;
- mean absolute difference over the same support;
- cosine similarity over the same support;
- top-k token-ID overlap between the two engines;
- whether each engine's *selected* token appears anywhere in the *other*
  engine's top-k;
- a classification of the disagreement as **near-tie-qualified** (both
  engines' own top-1/top-2 margins are within a preregistered epsilon, and
  each engine's selected token appears in the other's top-k) or **hard
  failure** (anything else).

All of this must be retained per-request, not summarized away -- the point
of this protocol is that a single pass/fail bit is exactly what made the
original stop rule unable to tell these two situations apart.

### 4. Do not compare later free-running tokens after a disagreement

Once engine A and engine B disagree at position N, their position-N+1
distributions are conditioned on different tokens and are not comparable.
If a later-position numerical comparison is needed (e.g. to check whether a
near-tie "recovers" or compounds), it must be run as a **separate,
explicitly labeled diagnostic**: teacher-force the same fixed prefix
(one engine's actual output, or a third fixed reference sequence) into
*both* engines and compare their next-token distributions under that shared,
controlled context. This is not part of the main first-divergence
correctness check and must never be conflated with it in a report.

### 5. Hard failures are defined independently of token equality

The following are hard failures regardless of whether they happen to
coincide with a near-tie, and regardless of margin:

- non-finite logits (NaN/Inf) from either engine;
- an engine's own selected token missing from its own top-k (an internal
  consistency failure, not a cross-engine one);
- large batch-vs-solo logit drift for a request run alone versus batched
  (i.e. failing the kind of self-consistency check already used in
  `experiments/sentinel-pilot/summaries/self-consistency-check.json`, but
  now at the logit level rather than only the token level);
- low top-k overlap between the two engines at the disagreement position
  (a near-tie should still land in a similar region of the distribution;
  low overlap means the distributions themselves differ, not just the
  argmax);
- cross-request identity or KV-cache corruption (one request's output
  containing content attributable to a different request in the same
  batch);
- crash, OOM, or timeout (unchanged from the existing sentinel-pilot stop
  rules);
- any disagreement where **neither** engine's own top-1/top-2 margin is
  near a tie (i.e. both engines were confident, and confidently disagreed --
  that is not explained by "near-tie, either answer defensible" and must be
  treated as a hard failure until shown otherwise).

### 6. No epsilon or bound is chosen from the diagnosed request

The 0.009 max-abs-logit-difference and 0.0/0.015625 margins observed in
`divergence-analysis.md` describe **one** request. They must not be reused,
directly or by "eyeballing a similar number," as the tolerance for this
protocol. Any epsilon or bound this protocol uses must come from the
calibration set defined in requirement 7, and must be committed (requirement
9) before it is ever applied to held-out data.

### 7. Two deterministic, disjoint prompt sets

- **Calibration set**: used only to observe the *distribution* of margins,
  overlaps, and drift, and to propose an epsilon/bound from that
  distribution. Never used to report a final pass/fail result.
- **Sealed holdout set**: disjoint from the calibration set (no shared
  seeds, no shared content), evaluated exactly once, only after the
  protocol and thresholds from requirement 9 are committed. If the holdout
  set is ever re-evaluated after seeing its own results, a new, differently
  seeded holdout set must be drawn -- reusing it defeats the point of
  sealing it.

Both sets must be built the same way the sentinel pilot already builds
prompts (deterministic, seed-derived, materialized before either engine
runs, persisted alongside seeds/hashes/order) -- no wall-clock randomness,
no reuse of `experiments/sentinel_diagnostics.py`'s existing target/filler
prompts for anything beyond the diagnosis already done with them.

### 8. Required reporting

For every evaluation (calibration and, separately, holdout), report:

- number and percentage of requests with exact token-ID equality;
- number and percentage requiring the near-tie clause, broken out from (8a);
- every accepted margin, for every near-tie-qualified disagreement (not
  just a mean or a count);
- every hard failure, in full (per requirement 5's categories);
- all of the above broken out by concurrency level and by sequence length.

A single aggregate "N% passed" number is not sufficient reporting under this
protocol; the point is that near-tie-qualified and hard-failure requests are
never collapsed into each other or into one summary statistic.

### 9. Commit before executing the sealed holdout set

The protocol document (this file, or its finalized successor), the
calibration-derived epsilon/bounds, and the holdout prompt set's hashes must
all be committed to source control *before* the holdout set is run. This
mirrors the sentinel pilot's own existing rule (a dirty or unidentified
source tree is a stop condition) extended to the tolerance itself: a
threshold chosen after seeing the data it's tested against is not a
preregistered threshold.

### 10. Resume the 10-pair performance protocol only if all of the following hold

- Concurrency-1 bit-exact equality passes on the sealed holdout set (zero
  exceptions -- requirement 1 has no tolerance).
- The sealed holdout set has zero hard failures (requirement 5).
- Every tolerated disagreement in the sealed holdout set satisfies the
  preregistered near-tie rule from requirement 6/9 -- not a post-hoc
  justification.
- Every exception is reported individually per requirement 8, not hidden
  inside one pass/fail value.

If any of these fail, the correctness gate is not satisfied, and this
protocol's own recommendation is the same as the existing sentinel pilot's:
stop, retain the evidence, and do not generate a performance claim.

## Calibration results and committed epsilon

Executed via `modal run modal_app.py::protocol_v2_calibration`
(`experiments/protocol_v2.py`, commit `af13518` and later). 107 requests
(concurrency 1 x3 batches, 8 x5, 32 x2; `V2_STEPS=8`), 0 crashes.

| | |
|---|---:|
| Exact match | 97 / 107 (90.65%) |
| Disagreements (pending epsilon) | 10 / 107 (9.35%) |
| Hard failures among disagreements | 0 |
| Candidate margins observed | 0.000275 – 0.012600 |

All 10 disagreements were "clean" candidates: no own-top-k inconsistency, no
non-finite logits, no low top-k overlap, no missing cross-presence in the
other engine's top-k, and none at concurrency 1 (confirming requirement 1
held throughout calibration -- zero concurrency-1 disagreements at all).

**Committed epsilon (requirement 6/7): `0.012599945068359375`** -- the
maximum margin among the 10 clean candidates, per the rule already specified
above (never chosen by inspection after the fact). Full data:
`experiments/sentinel-pilot/summaries/protocol-v2-calibration.json`.

## Sealed holdout results: gate does not pass

Executed via `modal run modal_app.py::protocol_v2_holdout --epsilon
0.012599945068359375`, on the disjoint holdout set (same 107-request
structure, different seed namespace, never inspected before this run), using
exactly the epsilon committed above. 0 crashes.

| | |
|---|---:|
| Exact match | 95 / 107 (88.79%) |
| Near-tie qualified | 10 / 107 (9.35%) |
| Hard failures | 2 / 107 (1.87%) |

**Requirement 10 gate: DOES NOT PASS** (hard failures must be zero).

The 2 hard failures, both `confident_disagreement_no_near_tie` (one engine's
own margin exceeded the committed epsilon):

1. `in512-out128-c8`, position 1: custom margin 0.0136 (marginally above
   epsilon), vLLM margin 0.0 (exact tie), top-20 overlap 0.95.
   **Correction:** this document originally reported "max abs log-prob diff
   10.0" here as evidence the two distributions differ "substantially...
   not just by a marginal perturbation." That number was a measurement
   artifact of `compare_top_k`'s now-fixed union-with-synthetic-floor
   comparison, which imputed a missing top-20 token as `min(logprob) - 10`
   -- with 0.95 overlap (only 1/20 tokens not shared), that single imputed
   token alone produced the ~10 figure; it was not a real property of
   either engine's output distribution. Recomputed over the true
   intersection only, this case's actual diff is small, consistent with the
   originally diagnosed request. See
   `experiments/sentinel-pilot/summaries/protocol-v2-audit.md` for the
   corrected re-examination (diagnostic only; does not change the verdict
   below).
2. `in512-out128-c32`, position 1: custom margin 0.0000267 (itself an
   essentially exact tie), vLLM margin 0.015625 -- exceeds the committed
   epsilon by about 24%. Max abs diff 0.0094, overlap 1.0: same small-scale
   signature as the originally diagnosed request, just past the specific
   numeric cutoff a 10-sample calibration set happened to produce. This
   case's metric was not affected by the `compare_top_k` bug above.

This is the expected, correct behavior of a properly sealed evaluation: an
epsilon derived from only 10 calibration candidates does not fully
generalize, and the holdout set's job is exactly to catch that rather than
let a small sample's ceiling pass silently. Full data:
`experiments/sentinel-pilot/summaries/protocol-v2-holdout.json`.

**Per requirement 10, the 10-pair performance protocol does not resume.**
The originally reported concern that failure 1's "large max-abs-diff" was
qualitatively different from every other disagreement was itself based on
the measurement artifact corrected above, not a real distributional
difference. The corrected next step (executed as a bounded, diagnostic-only
audit -- see `experiments/sentinel-pilot/summaries/protocol-v2-audit.md`)
is a larger, statistically-justified calibration sample against a fresh
sealed holdout namespace (Protocol V3), not re-deriving epsilon from this
holdout set, which would be tuning a threshold on the data used to test it.

## What this document does not do

It does not modify
`experiments/sentinel-pilot/summaries/divergence-analysis.md`'s findings,
which stand as reported. It does not resume the 10-pair performance
protocol -- that is a separate, explicit decision gated on requirement 10,
made only after the sealed holdout set (not calibration) is evaluated.
