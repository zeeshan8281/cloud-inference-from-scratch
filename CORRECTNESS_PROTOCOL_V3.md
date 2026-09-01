# Correctness Protocol V3

Status: **design only, not executed.** Written after the bounded audit
(`experiments/sentinel-pilot/summaries/protocol-v2-audit.md`) of Protocol
V2's two sealed-holdout hard failures. Performance measurement remains
paused; the 10-pair performance protocol does not resume until a V3 holdout
run (not yet executed) shows zero hard failures under the design below (see
`CORRECTNESS_PROTOCOL_V2.md` requirement 10, carried forward unchanged).

## Why V2 is not enough, and what carries forward unchanged

Requirements 1-5, 7 (structure), 8, and 9 of `CORRECTNESS_PROTOCOL_V2.md`
were sound and are unchanged here: concurrency-1 stays bit-exact,
comparisons only happen on a shared prefix, full diagnostic detail is
captured at first disagreement, disjoint calibration/holdout namespaces are
built the same deterministic way, and thresholds must be committed before
the sealed holdout runs. The audit did not find a flaw in *how* V2 applied
its epsilon -- it applied a committed number correctly, exactly once, exactly
as preregistered.

Two things the audit exposed as gaps:

1. **Requirement 6's epsilon was the max of only 10 calibration candidates.**
   A maximum of a small sample is expected to be exceeded by the next few
   draws -- that is what happened (both holdout hard failures had margins
   only slightly past the committed 0.012599945068359375). This is not a bug
   in requirement 6 as stated, it is under-sampling: V2 committed to "the
   max is the bound" without a large enough sample for that max to be
   stable.
2. **Requirement 5's batch-vs-solo-drift and cross-request-identity checks
   were specified but never implemented or run.** They are now implemented
   (`experiments.protocol_v2.batch_vs_solo_drift`,
   `experiments.sentinel_diagnostics.check_cross_request_identity`) and
   exercised only by the bounded audit, not by the calibration/holdout
   runner itself (`_protocol_v2_run` in `modal_app.py`). V3 must wire them
   into that runner so every calibration and holdout request is actually
   checked against the full requirement-5 list, not the subset V2 checked.

The audit additionally found that a request's hard-failure/near-tie
classification can flip between batch positions for the *same* content
(`protocol-v2-audit.md` section 4). V3 does not attempt to solve that by
re-running every holdout request at multiple positions -- that would
multiply the run's cost and is not what was asked for here. Instead, V3
treats it as the reason a **single held-out sample's hard-failure count must
be zero at a statistically meaningful sample size**, not as something a
smarter per-request check can route around. Section 3 below sizes the
sample with that in mind.

## 1. Epsilon: a statistically justified quantile, not a small sample's max

Requirement 6 is revised: epsilon is the **99th percentile** of the observed
margin distribution among clean near-tie candidates in the calibration set
(same disqualification list as `propose_epsilon` already uses), not the
maximum. A percentile estimated from a large enough sample is a stable
property of the underlying distribution; a sample maximum is not -- it grows
with sample size by construction, which is exactly the failure mode V2 hit.

This requires enough calibration candidates for a 99th-percentile estimate
to be meaningful. V2's calibration produced 10 candidates from 107 requests
(a ~9.3% disagreement rate). To get on the order of 200-300 candidates at
that rate requires roughly 2,200-3,300 calibration requests -- call it
**2,500 requests** (compare to V2's 107). `V2_BATCHES_PER_CELL` must scale
accordingly; the exact per-cell split is an implementation choice at build
time, not part of this spec, but must be committed (requirement 4 below)
before the holdout set is run.

Report both the 99th percentile and the maximum observed in the calibration
set, so the gap between them (V2's failure mode) is visible in every report,
not hidden.

## 2. Requirement 5 is fully wired into the runner, not just the audit

`_protocol_v2_run`'s per-request classification must now pass
`batch_vs_solo_drift` and `identity_check` to `classify_request` for every
calibration and holdout request, not leave them as `None` (the current
default, kept only for backward compatibility with already-collected V2
data). Concretely: for every batch of concurrency > 1, also run every
request in that batch alone (as its own separate cheap c1 run) to obtain the
solo baseline, and run `check_cross_request_identity` against every other
member of the same batch. This roughly doubles the per-batch request count
(one solo run added per batched request) -- accounted for in the sample-size
estimate above being a target range, not a hard number.

## 3. New sealed holdout namespace

The holdout set must be disjoint from:
- the original Protocol V2 calibration set,
- the original Protocol V2 sealed holdout set (both its 95 exact matches and
  its 10 near-tie-qualified requests -- not just the 2 hard failures),
- the two hard-failure requests inspected in this audit
  (`in512-out128-c8` batch_index=1 index_in_batch=6;
  `in512-out128-c32` batch_index=1 index_in_batch=22, and every batch
  variant derived from them: alone, original, reordered).

Achieved the same way V2's calibration/holdout split was: a distinct seed
namespace string (e.g. `"protocol-v3-calibration"` / `"protocol-v3-holdout"`
in place of V2's `"protocol-v2-{split}"`), never overlapping any
previously-used namespace string. This is a new prompt set, not a
re-evaluation of V2's holdout under a wider epsilon.

## 4. Commit before executing

Same as V2 requirement 9, unchanged: this document, the calibration-derived
epsilon (99th percentile, per section 1), and the new holdout namespace's
seed strings must be committed to source control before the V3 holdout set
is run. A dirty or unidentified source tree remains a stop condition,
inherited from the original sentinel pilot.

## 5. Resume condition (carries V2 requirement 10 forward)

The 10-pair performance protocol resumes only if, on the V3 sealed holdout:

- concurrency-1 disagreements: zero (unchanged, zero tolerance);
- hard failures under the full requirement-5 list (now including
  batch-vs-solo drift and identity/KV checks, per section 2): zero;
- every tolerated disagreement is within the committed 99th-percentile
  epsilon, applied without modification after being observed;
- every exception reported individually, not collapsed into one statistic
  (unchanged from V2 requirement 8).

If any V3 holdout run fails this gate, the same rule as V2 applies: stop,
retain the evidence, do not generate a performance claim, and treat it as
new information about the true near-tie margin distribution rather than a
reason to widen epsilon post hoc.
