# Protocol V2 sealed-holdout audit: bounded re-examination of the 2 hard failures

**This is a diagnostic re-examination only.** It does not reopen or change the
sealed `protocol_v2_holdout` verdict recorded in `CORRECTNESS_PROTOCOL_V2.md`:
Protocol V2's requirement-10 gate still **does not pass** (2 hard failures, as
originally preregistered and evaluated exactly once against a committed
epsilon). Performance measurement remains paused. The two requests audited
here are no longer validation data -- they are used only to understand *why*
they failed, per the audit instructions.

Raw data (both engines' full top-20 lists at every position, for every
variant): `experiments/sentinel-pilot/summaries/protocol-v2-audit.json`.
Executed via `modal run modal_app.py::protocol_v2_audit`.

## 1. The reported `max_abs_diff≈10` was a measurement artifact, now fixed

`compare_top_k` (`experiments/sentinel_diagnostics.py`) previously imputed a
missing top-20 token's log-probability as `min(present values) - 10` when
building a union-based comparison vector. Case 1
(`in512-out128-c8`, batch_index=1, index_in_batch=6) has `top_k_overlap=0.95`
-- only 1 of 20 tokens isn't shared between the two engines' top-20 -- and
that single imputed token alone produced the reported "max abs diff 10.0".
It was never a property of either engine's actual output distribution.

Recomputed over the true intersection only (19/20 shared tokens, no invented
values): `intersection_max_abs_diff = 0.0101`, `intersection_mean_abs_diff =
0.0056`, `intersection_cosine_similarity = 0.99999906`. This is the same
small-scale signature as every other near-tie observed in this entire
investigation, including case 2 (which the bug did not affect:
`intersection_max_abs_diff = 0.0094`, unchanged from what was originally
reported). **Case 1 was never "qualitatively different" from case 2 or from
the originally diagnosed single-request divergence. Both cases look like
textbook near-ties under every metric except the specific epsilon cutoff.**

## 2. Cross-choice diagnostics: both disagreements are rank-2-vs-rank-1 near-ties

| | case 1 (`c8`, pos 1) | case 2 (`c32`, pos 1, original batch) |
|---|---:|---:|
| custom chose | token 432 | token 1899 |
| vLLM chose | token 279 | token 2525 |
| custom's own margin (its choice vs. vLLM's choice) | 0.01359 | 0.0000267 |
| vLLM's own margin (its choice vs. custom's choice) | 0.0 (exact tie) | 0.015625 |
| vLLM's choice, rank under custom | **2** | **2** |
| custom's choice, rank under vLLM | **2** | **2** |
| top-k overlap | 0.95 | 1.0 |

In both cases, each engine's disagreement partner is the *other engine's own
second-ranked token* -- not an outlier, not missing from its top-k, not a
low-overlap distribution. The only reason these are "hard failures" under the
committed epsilon (`0.012599945068359375`) is that one engine's margin (0.0
or 0.015625) sits fractionally on the wrong side of a threshold that was
itself the maximum of only 10 calibration candidates.

## 3. New finding: vLLM's own logits are not batch-invariant for a fixed request

`batch_vs_solo_drift` (new, requirement-5 check that no prior run performed)
compares each engine's own top-k for the *identical* request run alone vs.
co-batched with other content. Result:

- **Case 1** (`c8`): custom is drift-free across all 8 steps (`top_k_overlap
  = 1.0`, `intersection_max_abs_diff` between 0.0044-0.0082 at every
  position, no token-level change). **vLLM shows drift**: `intersection_max_abs_diff
  = 0.0148` at position 0 (already above the descriptive 0.01 flag used
  here), and by position 1 **vLLM's own output token itself differs between
  the solo and batched runs of the same request** (`prefix_diverged`).
- **Case 2** (`c32`): both engines show batch-vs-solo *token* drift at
  position 1 -- but the underlying margins here are razor-thin in every
  condition (0.0 to 0.016 across all four observed engine x condition
  combinations), consistent with ordinary floating-point non-associativity
  at a genuine numerical tie rather than a defect specific to either engine.

This is a real, previously-unchecked property: for case 1, **vLLM's greedy
decode for a fixed request is not invariant to what else is in its batch**,
while the custom engine's is (at least in this instance). This directly
explains why case 1 surfaced as a "hard failure" against a fixed, single-shot
holdout evaluation: it isn't that the custom engine computed something wrong,
it's that vLLM's own answer for this exact request changes depending on
batch composition, and the holdout evaluation only ever observed one
composition.

## 4. Verdict is unstable under reordering -- a Protocol V2 design gap, not a new bug

Rerunning each target in its original batch position vs. the same batch
content reordered so the target is first:

| variant | case 1 verdict | case 2 verdict |
|---|---|---|
| alone (c1) | exact_match | **hard_failure** (`disagreement_at_concurrency_one` -- requirement 1 has zero tolerance at c1, regardless of margin) |
| original position | **hard_failure** | **hard_failure** |
| reordered, target first | exact_match | exact_match |

Same weights, same engines, same request content, same batch *membership* --
only the target's position within the batch changed -- and the classification
flips from "hard failure" to "exact match" in both cases. This is the clearest
evidence in the whole investigation that Protocol V2's per-request
evaluation, which observes each holdout request exactly once at whatever
batch position it happens to land in, cannot distinguish "this request
reveals a real problem" from "this request landed in a numerically
unlucky batch slot." **That is a design property of the protocol's single-shot
evaluation, not evidence the sealed verdict was computed incorrectly** --
epsilon was applied correctly, exactly once, exactly as preregistered. It is
evidence that a single observation per request is not enough to certify
absence of hard failures at the confidence this project needs, which is what
Protocol V3 (see `CORRECTNESS_PROTOCOL_V3.md`) is designed to address.

Case 2's `alone` result is independently notable: even with **no batch
composition to vary at all**, custom and vLLM pick different tokens (margins
0.0029 and 0.0 respectively) -- confirming this pair of tokens sits at a
genuine, engine-independent near-tie for this specific prefix, not something
introduced by batching.

## 5. No cross-request identity/KV contamination in either case

`check_cross_request_identity` (new, requirement-5 check) compared the
target's output token sequence against every other request in its batch, for
both the `original` and `reordered_target_first` variants, both engines, both
targets: **zero contamination flagged in every check** (8 checks total, all
`contamination_suspected: false`). This rules out cross-request KV/identity
corruption as an explanation for either hard failure.

## What this document does not do

It does not change `CORRECTNESS_PROTOCOL_V2.md`'s recorded sealed-holdout
result (2 hard failures, gate does not pass) or re-run the sealed holdout
set. It does not resume the 10-pair performance protocol. See
`CORRECTNESS_PROTOCOL_V3.md` for the next preregistered step.
