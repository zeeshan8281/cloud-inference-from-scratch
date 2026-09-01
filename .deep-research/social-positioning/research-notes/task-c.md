---
task_id: c
role: Reddit Community Strategist
status: complete
sources_found: 9
as_of: 2026-08-25
---

## Sources

[1] Spam — Reddit Help | https://support.reddithelp.com/hc/en-us/articles/360043504051-Spam | Source-Type: official | Accessibility: public | As Of: 2026-05-19 | Authority: 10/10
[2] Reddiquette — Reddit Help | https://support.reddithelp.com/hc/en-us/articles/205926439-Reddiquette | Source-Type: official | Accessibility: public | As Of: 2025-08-18 | Authority: 9/10
[3] Rules for r/LocalLLaMA | https://old.reddit.com/r/LocalLLaMA/about/rules/ | Source-Type: community | Accessibility: public | As Of: 2026-08-25 | Authority: 10/10
[4] r/LocalLLaMA moderator explanation on benchmark quality and self-promotion | https://www.reddit.com/r/LocalLLaMA/comments/1sg3vhp/removed_by_moderator/ | Source-Type: community | Accessibility: public | As Of: 2026-04 | Authority: 9/10
[5] r/LocalLLaMA moderator explanation of Rule Four participation | https://www.reddit.com/r/LocalLLaMA/comments/1tt75k0/removed/ | Source-Type: community | Accessibility: public | As Of: 2026-06 | Authority: 8/10
[6] r/MachineLearning Self-Promotion Thread | https://www.reddit.com/r/MachineLearning/comments/1vd5kqk/d_selfpromotion_thread/ | Source-Type: community | Accessibility: public | As Of: 2026-08 | Authority: 10/10
[7] Rules for r/programming | https://old.reddit.com/r/programming/about/rules/ | Source-Type: community | Accessibility: public | As Of: 2026-08-25 | Authority: 10/10
[8] Rules endpoint for r/learnmachinelearning | https://www.reddit.com/r/learnmachinelearning/about/rules | Source-Type: community | Accessibility: public-limited-rendering | As Of: 2026-08-25 | Authority: 7/10
[9] Cloud Inference Engine Lab repository and README | https://github.com/zeeshan8281/cloud-inference-from-scratch | Source-Type: official | Accessibility: public | As Of: 2026-08-25 | Authority: 9/10

## Findings

- Reddit-wide policy forbids repeated or unsolicited mass engagement, advises authentic participation in communities of genuine interest, and leaves each community's moderators final discretion over unwanted self-promotion. [1]
- Reddiquette permits linking one's own work within reason, gives a non-binding 9:1 participation rule of thumb, requires factual non-sensational titles, and forbids asking for votes, flooding submissions, link shorteners and coordinated voting. [2]
- r/LocalLLaMA is the strongest primary audience because the project directly concerns LLM inference, but Rule Four requires disclosed affiliation and recommends that self-promotion stay below 10% of the account's activity. [3][5][9]
- r/LocalLLaMA moderators currently require benchmark posts to contain on-topic analysis that brings new understanding rather than a screenshot or link, and they explicitly treat an obviously human-written summary as evidence against link-dumping. [4]
- r/MachineLearning currently directs personal projects, blogs and product placements into its recurring `[D] Self-Promotion Thread`, where direct links are allowed but pricing/payment requirements must be disclosed and shorteners or auto-subscribe links are prohibited. [6]
- r/programming forbids project demos and generic LLM content but permits deeply technical write-ups whose main value is how the system was built; its rules also prohibit LLM-written posts, including translation or summarization. [7]
- r/learnmachinelearning is a plausible secondary audience only if the video is a structured educational walkthrough with prerequisites and learning outcomes, but no launch post should be made until its live rules are manually rechecked or moderators confirm the format. [8][9]
- The most discussion-worthy framing is the project's negative result: its “batched” mode still executes per-request forwards and eager paged allocation lost its memory gate, while direct-block Triton removed 567.7 MiB of decode gather traffic and improved torch-paged throughput by about 9.7%. [9]
- The safest post format is a substantive Reddit text post or technical article containing the architecture, reproducible protocol, failed gates, raw artifact links and one precise systems question, with the walkthrough video and repository as supporting links rather than the entire submission. [2][4][7][9]
- The shipped 0.5B five-stage baseline and the proposed Qwen2.5-3B Ragged L4 Engine must be clearly separated, because presenting the planned packed runtime as completed would undermine the candor that makes the existing project worth discussing. [9]

## Deep Read Notes

### Sources [1] and [2]: Reddit-wide spam policy and reddiquette
Key data: repeated mass posting and unsolicited messaging are prohibited; the 9:1 ratio is a rule of thumb rather than a site-wide hard threshold; vote requests and coordinated voting can trigger bans.
Key insight: posting adapted versions to several relevant communities is not inherently spam, but rapid copy-paste distribution from an account dominated by its own links looks like exposure-seeking rather than participation.
Useful for: launch cadence, disclosure, account preparation, titles, links and voting safeguards.

### Sources [3], [4], and [5]: r/LocalLLaMA rules plus moderator interpretation
Key data: posts must concern LLMs, low-effort/primarily LLM-generated copy is barred, self-promotion should stay under 10%, affiliation must be disclosed, and benchmark links need analysis that creates new understanding.
Key insight: the repo's honest failed optimization gates are the strongest community contribution; “I built an inference server” is promotional, while “why request rotation was not tensor batching, with traces and raw results” is an inference-systems discussion.
Useful for: primary subreddit selection, participation prerequisite, post body, flair and discussion prompt.

### Sources [6] and [7]: r/MachineLearning routing and r/programming's technical-write-up test
Key data: r/MachineLearning has a current dedicated self-promotion thread; r/programming rejects project demos, requires deeply technical AI content and says the write-up—not a feature list—must be the focus.
Key insight: one launch asset cannot be pasted everywhere: r/MachineLearning gets a concise thread comment, whereas r/programming requires a separate human-authored engineering article and may reject even a technically strong video-first submission.
Useful for: subreddit-specific formats, effort allocation and avoiding predictable removals.

## Recommended Audience and Format

| Priority | Audience | Defensible angle | Format | Readiness |
|---:|---|---|---|---|
| 1 | r/LocalLLaMA | Reproducible inference-engine teardown: two failed gates, one Triton memory-traffic win, and the exact packed-runtime seam still missing | Native text post under `Tutorial / Guide` or `Discussion`; one benchmark table, architecture image, raw JSON/repo links, video secondary | Post only after checking the account has substantial non-promotional participation and rereading live rules |
| 2 | r/MachineLearning | Educational systems project and negative-results methodology, not a new ML research result | Concise comment in the current `[D] Self-Promotion Thread`; disclose free/open-source and cloud cost; repo plus optional video | Ready once the current monthly thread is confirmed |
| 3 | r/programming | How a Python scheduler that looked batched still issued B=1 transformer forwards, and how the code boundary must change | Separate human-authored technical article focused on code/data flow; article link as submission, repo/video references inside | Not ready from a walkthrough video alone |
| 4 | r/learnmachinelearning | Learn KV caching, paged allocation, streaming lifecycle and benchmark interpretation by following a working small engine | Tutorial self-post with prerequisites, chapter timestamps and exercises; avoid production claims | Conditional: manually verify live rules or message moderators first |

## r/LocalLLaMA Post Blueprint

**Recommended factual title:** `I rebuilt five layers of an LLM inference server on one L4 — two optimizations failed, and the traces show why`

Body order:

1. Disclose immediately: `I built this open-source educational engine; there is no paid product.`
2. State the tested boundary: pinned Qwen2.5-0.5B, FP16, one L4, greedy decoding, measured commit and raw artifacts.
3. Show the five-stage architecture in one compact diagram.
4. Lead with the failed gates: “batched” was scheduler interleaving but still B=1 forwards; paging reserved worst-case blocks and slightly worsened unused KV.
5. Show the verified win: direct-block Triton removed the full decode gather and improved the torch-paged comparison by about 9.7% under the fixed protocol.
6. Separate shipped reality from the Ragged L4 target: one packed forward, token-budget chunked prefill, transactional dynamic KV and pressure preemption are planned, not implemented.
7. End with one expert question, such as: `For a single L4, would you implement mixed ragged prefill in the first Triton kernel, or prove batched decode plus a segmented Torch prefill before fusing both?`
8. Put direct GitHub, artifact and optional video links at the bottom; do not use a landing page, newsletter, waitlist or URL shortener.

Use the video to make the technical text easier to inspect, not as the value proposition. A defensible walkthrough sequence is: execution trace showing B=1 forwards; cache layouts; benchmark protocol and failures; Triton block-table read; raw result reproduction; then the not-yet-shipped packed architecture. Avoid thumbnail/title language such as “I beat vLLM,” “production-grade,” “insane speed,” or “continuous batching from scratch” without the current B=1 qualification.

## Posting Safeguards

- Do not publish the same copy to four subreddits on the same day; adapt each submission to the community and engage with responses before considering another post. [1][2]
- Do not ask friends, followers or video viewers to upvote, and do not announce the Reddit link elsewhere with a voting request. [2]
- Disclose authorship and Modal/cloud costs; do not write “I found this repo.” [3][6]
- Do not let an LLM author or rewrite the r/programming article; the user must write it personally under that community's explicit rule. [7]
- Recheck every live rule page immediately before posting because moderator policy and recurring thread routing can change without notice. [1][3][6][7][8]
- Remain available for technical replies, including skeptical questions about the failed gates, small model, single GPU, synthetic workload and lack of steady-state arrival-rate testing. [4][9]

## Gaps

- r/learnmachinelearning's public rule endpoint existed but its rule text did not render through the available logged-out research path, so self-promotion permission and required flair remain unverified.
- r/MachineLearning's current self-promotion thread is clear, but the full logged-out rule page did not render; a future main-feed `[P]` or `[D]` post therefore needs a fresh rules check or moderator confirmation.
- No controlled evidence establishes an optimal video length, posting time, title formula or subreddit conversion rate; these should not be represented as researched facts.
- Reddit votes and comments cannot distinguish technical merit from account history, timing, title and moderator discretion, so engagement is not a clean measure of project quality.
- Community rules are current only as of 2026-08-25 and can change before the Ragged L4 milestone ships.

## Counter-Claim

Even perfect rule compliance and candid technical framing may not earn discussion because the released engine uses a 0.5B model, runs on rented cloud hardware rather than a local machine, and does not yet implement true tensor batching or dynamic paging. [9] r/LocalLLaMA readers may reasonably see the current release as an educational reimplementation rather than a novel local-inference contribution, while r/MachineLearning may see it as engineering rather than research. The strategy should not conceal that weakness: post the current work only as a failure-driven teardown, or wait until the Ragged L4 engine supplies multi-request forward traces, online load curves and same-protocol vLLM comparison.
