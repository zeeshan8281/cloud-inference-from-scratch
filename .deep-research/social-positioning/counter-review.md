# P6 Counter-Review

AS_OF: 2026-08-25
Scope: `report.md`, `registry.md`, and research notes `task-a.md` through `task-e.md`

## Verification Summary

- **Citation registry:** Pass. Report citations `[1]` through `[22]` all exist in Approved Sources; no dropped source is cited.
- **Numerical consistency:** Pass. `1.68×`, `0.80×`, `4.45% worse`, `1.10×`, `~9.7%`, `567.7 MiB`, and `2.63 MiB` match the repository/notes and use the correct local baselines.
- **Platform differentiation:** Pass. LinkedIn, X, r/LocalLLaMA, r/MachineLearning, r/programming, and r/learnmachinelearning receive materially different advice.
- **Issues found:** 5 total — 1 high, 3 medium, 1 low.

## Issues

### 1. High — The repeated technical CTA presents the wrong implementation choice

**Report sections / quotes:**

- §3 LinkedIn draft: `“would you build packed decode first or mixed ragged prefill first?”`
- §4 X draft: `“What would you prove first: packed decode or mixed ragged prefill?”`

**Evidence:** The project source of truth says the missing core is one packed multi-request model forward plus ragged metadata; the research notes distinguish functional packed execution from later kernel optimization. “Packed decode” and “mixed ragged prefill” are not peer alternatives: packed transformer execution and a correct segmented/Torch attention reference should establish the runtime contract before choosing which custom attention path to optimize. The report's r/LocalLLaMA question is more technically sound because it asks whether the *first Triton kernel* should cover mixed prefill or whether batched decode should precede fusion.

**Risk:** An inference engineer may read the CTA as evidence that the architecture plan is confused about the boundary between model batching and attention-kernel scope.

**Recommended fix:** Use one technically ordered question across LinkedIn/X: `“After the packed Torch reference is correct, would you optimize batched paged decode first, or fuse mixed ragged prefill and decode?”` Alternatively ask about token-budget profiling or recompute-preemption policy.

### 2. Medium — The central hook can imply that continuous batching and paging themselves regressed

**Report sections / quotes:**

- Executive summary: `“Two optimizations made it worse—and the traces explain why.”`
- LinkedIn/X drafts: `“Two ‘optimizations’ made it worse.”`

**Evidence:** The report correctly explains later that the “batched” mode still ran per-request `B=1` forwards, paging reserved worst-case capacity, batching sharply improved TTFT, and the results do not refute Orca or PagedAttention. The task-d counter-claim explicitly says these are incomplete implementations on a 0.5B model and fixed synthetic workloads.

**Risk:** The standalone hook, thumbnail, or first X post can circulate without the qualification and be interpreted as “continuous batching and PagedAttention are slower,” which the evidence does not establish.

**Recommended fix:** Keep the tension but bind it to implementation scope in the hook itself: `“Two first-pass optimizations missed their gates—and the traces show the missing engine contracts.”` If retaining the current hook, put `my B=1 scheduler / my eager allocator` in the first visible frame rather than waiting for later explanation.

### 3. Medium — Exact LinkedIn timing and document-limit advice is more confident than its evidence

**Report section / quotes:**

- §3: `“Follow 2–4 days later with a six-to-eight-page architecture PDF”`
- §3: LinkedIn permits a document `“up to 100 MB and 300 pages”`
- §7 says the sequence is `“not an algorithmic timing claim.”`

**Evidence:** No task note supports a 2–4-day organic timing optimum; task-a explicitly lists posting time as an evidence gap. Registry source `[5]` is labeled As Of `2023`, making exact product limits stale for an AS_OF 2026 report about a fast-changing platform. The six-to-eight-page editorial recommendation is reasonable, but it is not established by the upload ceiling.

**Risk:** This creates an internal contradiction and can leave the user following an arbitrary calendar or obsolete upload specification.

**Recommended fix:** Replace `2–4 days` with `after the initial discussion slows or when the PDF adds new value`. Re-verify the live LinkedIn help page before retaining exact MB/page limits; otherwise omit the ceilings and keep the durable advice that PDF documents support multi-page knowledge sharing.

### 4. Medium — Executive confidence is too aggregated for volatile platform advice

**Report section / quote:** Executive summary: `“Overall confidence: High.”`

**Evidence:** The report itself assigns medium confidence to distribution, low confidence to r/learnmachinelearning, acknowledges an unreadable/dropped rule source for that subreddit, relies on a current monthly r/MachineLearning thread, uses a four-month-old moderator comment for LocalLLaMA interpretation, and labels master-video timing an editorial hypothesis. Technical/project claims are high-confidence, but organic format effectiveness and community permission are not.

**Risk:** Readers may treat suggested lengths, sequence, format, and subreddit readiness as equally proven as repository benchmark facts.

**Recommended fix:** Split the marker: `High confidence on technical positioning and claim boundaries; Medium on platform format fit; Low-to-medium on organic distribution and unverified subreddit rules.` Keep the mandatory posting-day rule check prominent.

### 5. Low — The Reddit clip duration is unsupported precision

**Report section / quote:** §6 Social cuts: `“Reddit, 90–120 seconds”`

**Evidence:** Task-c says no controlled evidence establishes an optimal video length, and its core recommendation is a substantive native text post with video secondary. Task-e proposes 90–120 seconds as an editorial format but does not supply community-specific retention or discussion evidence. The Reddit video-limit source was dropped from the registry as nonessential.

**Risk:** Editing effort may optimize an arbitrary duration instead of the failure analysis that subreddit moderators explicitly require.

**Recommended fix:** Describe it as `an optional short proof clip, cut to the minimum duration needed to show the trace and two causal failures`; retain 90–120 seconds only as a testable production hypothesis, not a recommendation backed by Reddit behavior.

## Opposing Interpretation Check

The report appropriately includes the strongest opposing view: the released project is a small-model educational reimplementation of known techniques, not competitive inference evidence, and Reddit may be premature until Ragged L4 ships. It also correctly separates this implementation's failed gates from the validity of Orca/PagedAttention and warns that video may not be the best X asset.

One additional interpretation should be made explicit during final revision: a failure-first campaign may maximize credibility with specialists while reducing clarity for recruiters unfamiliar with inference terminology. The existing LinkedIn section mentions this tradeoff, but the opening hook should still identify the artifact as an educational engine lab within the first sentence/frame.
