# Correctness Protocol V2 (proposed; not executed)

Status: **specification only**. Nothing in this document has been run.
Performance measurement remains paused. The original stopped pilot
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

## What this document does not do

It does not choose an epsilon. It does not draw the calibration or holdout
prompt sets. It does not run any engine. It does not modify
`experiments/sentinel-pilot/summaries/divergence-analysis.md`'s findings,
which stand as reported. Implementing requirements 3-10 (calibration
harness, holdout sealing, reporting) is future work, tracked separately from
both the stopped pilot and the completed bounded diagnostic.
