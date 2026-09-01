# Positioning the Cloud Inference Engine Lab on LinkedIn, X, and Reddit

> Research date: 2026-08-25 | Approved sources: 22 | Mode: Standard | AS_OF: 2026-08-25 | Official-source share: 59.1%

## Executive summary

Do not position this as another “I built an LLM API” project, a production inference engine, or a vLLM competitor. Position it as an **executable, falsifiable inference-systems lab**: a compact serving stack where every claim can be followed from code to correctness oracle to GPU benchmark—and where two expected optimizations failed in public.

The strongest hook is:

> **I built five stages of an LLM inference engine on one NVIDIA L4. My first-pass batched and paged modes missed their gates—and the traces exposed the missing engine contracts.**

That sentence works because it contains tension, proof, and humility. The repository really does implement the serving path rather than delegating generation; publishes raw artifacts and explicit pass/fail gates; reports contiguous KV at 1.68× the naïve baseline; reports “batched” throughput at only 0.80× contiguous despite much lower TTFT; reports paged allocation making unused KV 4.45% worse; and reports direct-block Triton at 1.10× torch-paged while eliminating the full-cache decode gather.[1] The negative results are the differentiation.

Use one shared technical thesis but package it differently:

- **LinkedIn:** a professional engineering case study. Emphasize judgment, experimental honesty, and end-to-end ownership. Use a 100–120 second native proof video and a readable text post.
- **X/Twitter:** a compact technical argument. Use a 50–60 second native clip plus a three-post thread containing hook, causal proof, and repository/critique CTA.
- **Reddit:** a contribution to a community discussion, not a launch announcement. Lead with the failure analysis in a substantive native text post; make the video and repository supporting evidence. Start with r/LocalLLaMA only if the account has real non-promotional participation and the live rules still allow it.[11][12][13][14]

Create one 6–10 minute master walkthrough, recommended at about 8:05, then cut it into platform-specific clips. The master should not begin with setup, personal biography, or an architecture tour. Open with the failed gates, state the non-production boundary, then prove the request path, model implementation, correctness, benchmarks, Triton memory path, and unshipped Ragged L4 target.

**Confidence:** High on technical positioning and claim boundaries; Medium on platform-format fit; Low-to-medium on organic distribution and any subreddit rule not verified again on posting day. No evidence supports guaranteed reach, a magic posting time, or a universally favored media format.[4][9]

## 1. The product is not the server; the product is the investigation

### Recommended category

Call it one of these:

1. **An executable LLM inference-systems lab** — best general label.
2. **A from-scratch inference-engine teardown with reproducible failures** — best social hook.
3. **A five-stage GPU serving lab from naïve decoding to direct-block Triton attention** — best technical subtitle.

Avoid “mini-vLLM.” It invites a performance comparison that the released baseline cannot support. There is no same-model, same-revision, same-L4 vLLM benchmark yet, the current scheduler still performs per-request `B=1` model forwards, paged admission reserves worst-case capacity, prefill remains a torch path, and the tested model is a small base model that Qwen does not recommend as a conversational assistant.[1][22]

### The differentiated thesis

The project’s value is a chain of falsifiable claims:

1. A request enters an authenticated FastAPI endpoint and exits as ordered JSON or SSE.
2. The serving path uses a custom Qwen2 forward, scheduler, KV implementations, greedy decode, and direct-block Triton decode kernel; Hugging Face execution is only a correctness oracle.[1]
3. Five stages are tested against identical token output and a fixed protocol.
4. Each optimization has an explicit gate rather than a celebratory label.
5. Two gates fail, and the implementation explains why.
6. The next architecture is stated as an unshipped hypothesis with seven acceptance tests, not shown as completed work.[1]

That makes the project useful to three audiences:

| Audience | What they should conclude |
|---|---|
| ML-systems learners | “I can trace the actual data and ownership path without reading a production-scale codebase first.” |
| Inference/GPU engineers | “The author knows the difference between scheduler interleaving and tensor batching, and between a block pool and demand-paged serving.” |
| Technical hiring managers | “This shows model internals, backend lifecycle, GPU work, testing discipline, measurement, and honest technical judgment in one artifact.” |

The foundational literature reinforces the distinction. Orca’s iteration-level scheduling works because model operations are selectively batched; rotating requests while still launching separate model forwards does not capture that benefit.[20] PagedAttention’s serving advantage combines dynamic block allocation, scheduling/preemption, and a block-aware kernel; a shared pool with eager maximum reservation is a useful implementation stage but does not establish the production memory-efficiency claim.[19]

**Confidence: High.** Repository behavior and primary systems sources agree.

**Counter-interpretation:** Experts may still see the implementation as a re-creation of known techniques rather than a novel engine. Accept that. Claim pedagogical and experimental value, not algorithmic novelty.

## 2. The core story and message hierarchy

### One-sentence message

> I rebuilt the path behind an LLM API—from naïve decoding through KV caching, scheduling, paged allocation, and a Triton kernel—and the most useful result was learning exactly why “batched” and “paged” were not yet the real thing.

### Three proof lines

Use these repeatedly across platforms:

- **The cache worked:** contiguous KV delivered 1.68× the naïve decode throughput under the fixed profile.[1]
- **The labels lied until the measurements corrected them:** scheduler-level “batching” improved TTFT but reached only 0.80× contiguous throughput, while eager paged reservation made unused KV 4.45% worse.[1]
- **The kernel removed real work:** direct-block Triton eliminated the full-cache decode gather and improved the torch-paged comparison by about 9.7%; the measured paths reported 567.7 MiB of temporary K/V gathering for paged decode versus 2.63 MiB of prefill-only gathering in the Triton run.[1]

### The boundary sentence

Say this in the video and every serious post:

> This is tested educational systems code, not a production service. The shipped release is the five-stage 0.5B baseline; packed multi-request execution, dynamic KV growth, preemption, chunked prefill, the 3B model, and the vLLM comparison are the next milestone—not completed results.

### Why failure should lead

A conventional portfolio post lists components. This one can demonstrate judgment. An ICSE 2024 study of software-engineering discussion on LinkedIn found that useful posts explained results and their relevance rather than merely announcing an achievement; much of the resulting discussion came from industry participants.[6] GitHub’s maintainer guidance similarly recommends one simple value message, supporting tutorials/screencasts, and responsive follow-up rather than a diffuse feature list.[10]

The failed gates are not an apology. They create the narrative:

```text
Expectation                 Measurement                  Root cause                     Next falsifiable step
Batching raises throughput  0.80× contiguous            Request rotation, still B=1    One packed multi-request forward
Paging reduces KV waste     4.45% worse                  Eager worst-case reservation   Transactional demand growth
Block-aware decode helps    ~9.7% over torch-paged       Removed decode gather          Ragged mixed-work kernel
```

**Confidence: High.** The story is directly supported by committed artifacts.[1]

**Counter-interpretation:** The failures partly reflect a 0.5B model, short contexts, one L4, and synthetic fixed workloads; they do not refute continuous batching or PagedAttention generally.[19][20] Phrase the lesson as “my implementation did not yet realize the technique,” not “the technique does not work.”

## 3. LinkedIn positioning

### Objective and audience

LinkedIn is the portfolio and engineering-judgment channel. Optimize for qualified attention from ML infrastructure engineers, backend/GPU engineers, and technical hiring managers—not maximum generic AI reach.

The right archetype is **engineering case study**, not demo reel:

> Problem → implementation staircase → surprising measurements → causal diagnosis → next architecture → invitation to critique.

LinkedIn supports native videos up to 15 minutes, reviewed captions, thumbnails, and broad aspect-ratio flexibility.[2] Its own sharing guidance recommends an immediate hook, visible context, vertical-friendly production, and a question that starts conversation.[3] However, LinkedIn’s feed engineering documentation describes contextual, multi-objective ranking rather than a simple “video always wins” rule.[4]

### Recommended LinkedIn asset

- 100–120 second native video, 4:5 or square for phone readability.
- First frame contains the failed result, not a logo.
- Brief face-to-camera opening is optional; the majority should be architecture, code, terminal, and benchmark evidence.
- Upload reviewed captions.
- Put the GitHub link in the text where it reads naturally; there is no official evidence that hiding the link is an organic-ranking requirement.
- Follow with a six-to-eight-page architecture PDF after the initial discussion slows or when the PDF adds new value; LinkedIn document posts support multi-page knowledge sharing.[5]

### Draft LinkedIn post

> I built five stages of an educational LLM inference-engine lab on one NVIDIA L4.
>
> My first-pass batched and paged modes missed their performance gates.
>
> I implemented the serving path from scratch using PyTorch primitives and the official Qwen tokenizer/weights: model forward, greedy decoding, KV caching, request scheduling, paged KV allocation, JSON/SSE streaming, and a direct-block Triton decode-attention kernel.
>
> Then I put every stage behind the same correctness and benchmark gates.
>
> What happened:
>
> • Contiguous KV caching: **1.68×** naïve decode throughput
> • “Batched” scheduler: TTFT fell sharply, but throughput was only **0.80×** contiguous
> • Paged KV: unused capacity became **4.45% worse**
> • Direct-block Triton: removed the full-cache decode gather and improved torch-paged throughput by **~9.7%**
>
> The failures were the useful part.
>
> My scheduler rotated requests but still launched one model forward per request. My allocator used blocks but reserved the worst case up front. The names sounded right; the execution path was not there yet.
>
> The next milestone is a real packed runtime: one multi-request forward, token-budget scheduling, demand-paged KV growth, pressure preemption, and a same-L4 vLLM comparison. That architecture is a target—not shipped code.
>
> Walkthrough + raw artifacts + reproducible commands: [GitHub link]
>
> After the packed Torch reference is correct, would you optimize batched paged decode first, or fuse mixed ragged prefill and decode?

### Why this version works

It leads with a result, establishes ownership without dumping every feature, names exact failures, explains causality, separates the future design, and asks a question an experienced engineer can answer. That matches the result-discussion pattern observed in software-engineering posts better than a “proud to announce” post.[6]

### LinkedIn follow-up PDF

1. The thesis: two optimizations failed.
2. The five-stage architecture.
3. Correctness gates and fixed protocol.
4. Batching: low TTFT, poor throughput, why `B=1` matters.
5. Paging: block pool without demand growth.
6. Triton: logical gather versus direct physical-block reads.
7. Ragged L4 target and seven acceptance gates.
8. Reproduce it / technical question.

**Confidence: High on message fit; Medium on distribution.** Platform mechanics and content structure are documented, but organic reach cannot be forecast from public evidence.[2][3][4]

**Counter-interpretation:** A narrower technical post may receive fewer reactions than a broad AI demo. That can still be a better result if the viewers are engineers and hiring managers who understand the work.

## 4. X/Twitter positioning

### Objective and audience

X should deliver one compact systems argument that an inference engineer can understand without opening the link. Use the video as proof, not as the post’s only meaning.

X allows a non-Premium web upload up to 140 seconds/512 MB; videos at or below 60 seconds loop.[7] Threads of four or more posts are truncated behind a “Show this thread” treatment, so the hook, proof, and CTA should fit within the first three posts.[8] Third-party 2025 data from Buffer found text and image posts in its sample had higher median engagement than video and link posts, which is useful mainly as a warning against assuming video automatically wins.[9]

### Recommended X asset

- 50–60 second native clip, 16:9 or square.
- Burned-in captions and very large metric labels.
- Make the last frame transition cleanly into the first because it may loop.
- Three-post launch thread; deeper implementation material can continue as replies after the core bundle.
- Be present for technical responses. Treat reply engagement as conversation, not an algorithm hack.

### Draft three-post thread

**Post 1**

> I built 5 stages of an educational LLM inference-engine lab on one L4.
>
> My first-pass batched and paged modes missed their gates:
> - “batched” throughput: 0.80× contiguous
> - paged KV waste: 4.45% worse
>
> The profiler showed exactly why. [native 50–60s clip]

**Post 2**

> The scheduler rotated across requests—but still ran one `B=1` transformer forward per request.
>
> The KV cache used blocks—but reserved `prompt + max_output` up front.
>
> Correct concepts, incomplete execution paths.
>
> Direct-block Triton did remove the decode gather and recovered ~9.7% vs torch-paged.

**Post 3**

> The repo includes the custom Qwen forward, scheduler, KV layouts, Triton kernel, correctness oracles, raw benchmark JSON, and the failures.
>
> Next target: one packed multi-request forward + demand-paged growth + preemption on the same L4.
>
> Code/results: [GitHub link]
>
> After the packed Torch reference is correct, what would you optimize first: batched paged decode, or fused mixed prefill/decode?

### Alternate single-post version

> “Continuous batching” is not request rotation.
>
> I learned that by building five stages of an LLM server and watching my batched version run at 0.80× the contiguous baseline—because every scheduled request still triggered its own model forward.
>
> The failure, trace, and next packed-runtime design: [link]

This alternate is useful if the clip is not ready. Buffer’s sample provides no basis for treating that as a downgrade; text can be a strong native format on X.[9]

### What not to do on X

- Do not start with “Finally launching my new project.”
- Do not use a full-screen terminal recording with unreadable text.
- Do not say “PagedAttention failed”; say your eager allocator did not implement demand-paged growth.
- Do not say “from scratch” without immediately defining the boundary: PyTorch tensor primitives plus official tokenizer/weights, no vLLM/SGLang/HF generation in the serving path.[1]
- Do not ask only for stars.
- Do not bury the technical CTA after a long thread.

**Confidence: High on format limits and core message; Medium on engagement outcome.** X does not publish enough organic ranking detail to support reach promises.[7][8][9]

**Counter-interpretation:** A benchmark image plus precise text may outperform the video. If editing time is limited, publish the three-post technical argument and use the walkthrough as a reusable proof asset later.

## 5. Reddit positioning

### Reddit is not a distribution channel

Treat Reddit as several communities with different rules, not one social platform. Reddit-wide guidance discourages repeated unsolicited promotion and emphasizes authentic participation; Reddiquette allows one’s own links within reason, uses a non-binding 9:1 participation heuristic, discourages sensational titles, and forbids vote solicitation or coordinated voting.[11][12]

The safest order is:

| Priority | Community | Positioning | Submission |
|---:|---|---|---|
| 1 | r/LocalLLaMA | Failure-driven inference teardown | Substantive native text post; repo/video secondary |
| 2 | r/MachineLearning | Educational systems project, not research novelty | Current self-promotion thread only unless rules change |
| 3 | r/programming | Deep implementation article about why scheduling was still `B=1` | Human-authored technical article; not a project demo |
| 4 | r/learnmachinelearning | Structured tutorial with prerequisites and chapters | Only after manually confirming current rules |

r/LocalLLaMA’s rules require disclosed affiliation and recommend keeping self-promotion below 10% of account activity.[13] Current moderator guidance expects benchmark submissions to provide analysis that creates new understanding, not a screenshot or link drop.[14] r/MachineLearning currently routes personal projects to a dedicated self-promotion thread.[15] r/programming forbids project demos and generic LLM content while allowing deeply technical write-ups; it also prohibits LLM-written submissions.[16]

### Before posting to r/LocalLLaMA

Do not post if the account is new or mostly promotes your own work. Participate genuinely first. Re-read the live rules on posting day. Disclose that you built the repository and that it is free/open-source. Never ask another network to upvote the Reddit submission.[11][12][13]

### Recommended r/LocalLLaMA title

> I rebuilt five layers of an LLM inference server on one L4—two optimizations failed, and the traces show why

This is factual and specific. Avoid “I beat vLLM,” “production-grade,” “insane speed,” or “PagedAttention from scratch” without qualification.

### Recommended post body structure

1. **Disclosure:** “I built this open-source educational engine; there is no paid product.”
2. **Tested boundary:** Qwen2.5-0.5B base, FP16, one L4, greedy decode, pinned revision and measured commit.
3. **Architecture:** one compact diagram of the five stages.
4. **Negative result one:** scheduler interleaving still issued per-request forwards; show TTFT and throughput together.
5. **Negative result two:** paging reserved worst-case blocks; show why fragmentation did not improve.
6. **Positive result:** direct-block Triton removed the decode gather; show bytes gathered and correctness tolerance.
7. **Limitations:** small model, fixed synthetic workloads, no arrival-rate curve, no vLLM comparison, one narrow kernel shape.
8. **Unshipped design:** clearly label Ragged L4 as the next experiment.
9. **Expert question:** one design decision, not “thoughts?”
10. **Links at bottom:** direct GitHub, raw artifact, optional walkthrough.

### Draft opening paragraphs

> I built this open-source educational engine to understand the serving path below an LLM API. It is not a production service and it does not claim to beat vLLM.
>
> The interesting part is that two stages with the right names had the wrong execution behavior. “Batched” rotated across requests but still launched one model forward per request. “Paged” used a shared block pool but reserved prompt-plus-maximum-output capacity up front.
>
> Under the fixed eight-request decode profile, batching sharply reduced TTFT but delivered only 0.80× the contiguous-cache throughput. Under the fragmentation profile, paged allocation made unused KV 4.45% worse. The direct-block Triton decode path did remove the full-cache gather and improved the torch-paged result by about 9.7%.
>
> Below is the architecture, protocol, failure analysis, raw data, and the next design I intend to test.

### The r/programming constraint

Do not submit the walkthrough video or repository as a project demo. If you target r/programming, personally write a standalone article such as:

> **Why my “continuous batching” scheduler was still executing an LLM at batch size one**

The article’s main value must be code/data-flow analysis, with the repository merely supporting it. Because the community explicitly bars LLM-written posts, this research can guide your thinking, but you must personally author that article.[16]

**Confidence: High for currently verified communities; Low for r/learnmachinelearning.** Community rules can change, and the latter’s rules were not reliably available during research.

**Counter-interpretation:** Even a compliant post may receive little interest because the released model is small and the core techniques are known. Reddit should get the post only if the failure analysis itself teaches something; otherwise wait for the Ragged L4 multi-request traces and vLLM comparison.

## 6. Walkthrough video strategy

### The master video

Recommended master length: **about 8 minutes**, with a hard editorial rule that every chapter answers a distinct engineering question. Wistia’s 2025 analysis found shorter videos generally retain a higher percentage of viewers, but how-to content sustains attention better than many other long formats; this supports a short-social/long-tutorial split, not a guarantee that eight minutes is optimal.[18]

Record at 1920×1080. Keep one dominant visual surface. Crop code and terminal text aggressively for phone readability. Use architecture insets only when they explain the current frame. Never show the API secret, bearer header, notifications, or an unedited desktop.

### 8:05 storyboard

| Time | Screen | Narration job |
|---|---|---|
| 0:00–0:15 | Benchmark failures, then Triton recovery | “My first-pass batched and paged modes missed their gates; the traces show why.” |
| 0:15–0:45 | README scope box | State educational/non-production boundary and what “from scratch” means. |
| 0:45–1:25 | Architecture animation | Trace one request: auth → scheduler → model → KV → JSON/SSE. |
| 1:25–2:15 | Model code + parity output | Prove custom Qwen execution and HF-oracle boundary. |
| 2:15–3:05 | Naïve vs contiguous chart | Explain why retaining KV produced the clean 1.68× win. |
| 3:05–4:10 | Scheduler trace showing separate forwards | Pair low TTFT with lower throughput; explain request rotation vs tensor batching.[20] |
| 4:10–5:15 | Block table/allocation visualization | Explain why eager maximum reservation did not create demand paging.[19] |
| 5:15–6:20 | Torch gather vs Triton block reads | Show 567.7 MiB vs 2.63 MiB gather traffic, correctness, and ~9.7% gain; avoid calling it fully fused.[1][21] |
| 6:20–7:10 | Tests, cancellation, SSE, metrics | Show that systems work includes ownership and failure paths. |
| 7:10–7:50 | Ragged L4 architecture with “TARGET” label | Explain packed forward, token budget, dynamic KV, preemption, and vLLM comparison as future gates. |
| 7:50–8:05 | Reproduction commands | Ask viewers to reproduce and challenge one design choice. |

### Social cuts

**LinkedIn, 100–120 seconds**

- 0:00–0:04 — “My first-pass batched and paged modes missed their gates.”
- 0:04–0:18 — architecture.
- 0:18–0:48 — batching failure.
- 0:48–1:12 — paging failure.
- 1:12–1:38 — Triton recovery.
- 1:38–1:50 — reproduce/critique CTA.

**X, 50–60 seconds**

- Frame one: failed gate.
- Eight seconds: request path.
- Fifteen seconds: batching cause.
- Fifteen seconds: paging cause.
- Ten seconds: Triton proof.
- Final frame: repo and technical question.

**Reddit, optional short proof clip**

- Start on the benchmark table, not personal branding.
- Cut it to the minimum duration needed to show the trace and two causal failures; 90–120 seconds is only a production hypothesis, not a researched optimum.
- Spend most of the clip on the profiler and root cause.
- Put protocol, caveats, and reproduction details in the post body.
- End with a precise Ragged L4 design question.

### Accessibility and production

Prepare a reviewed SRT/VTT transcript for the master and LinkedIn, and burn concise captions into every social cut. YouTube supports timed caption files,[17] LinkedIn supports caption upload/review,[2] and X video often autoplays in contexts where sound is absent or disabled.[7]

Use visual language consistently:

- Green: passed gate.
- Red: failed gate.
- Blue: current execution path.
- Dashed/purple: unshipped target.

Do not animate boxes merely for polish. Animate the request or memory path only when motion explains causality.

**Confidence: Medium-high.** Platform specifications and project proof points are strong; exact editorial timing is a hypothesis to test with retention and repository-referral data.

**Counter-interpretation:** A 3–5 minute master could earn higher completion. If a chapter does not introduce a new proof or causal explanation, cut it and let the README carry the detail.

## 7. Launch sequence and measurement

### Recommended sequence

1. Publish the full walkthrough on YouTube and add it near the top of the README.
2. Publish LinkedIn with the 100–120 second native cut and the engineering-case-study text.
3. Publish X with the 50–60 second native cut and three-post thread.
4. Spend time replying to technical questions and correcting misunderstandings.
5. Publish the LinkedIn architecture PDF follow-up after the initial discussion.
6. Post to r/LocalLLaMA only after verifying current rules and account participation; make it a self-contained failure analysis.
7. Use r/MachineLearning’s current self-promotion thread if still active.
8. Write a separate human-authored article before considering r/programming.

This is sequencing for reuse and community fit, not an algorithmic timing claim.

### What to measure

Do not optimize for impressions alone. Track:

- Qualified comments from inference/backend/GPU practitioners.
- GitHub visitors and stars, but also README depth and artifact clicks if available.
- Walkthrough first-30-second retention and exits by chapter.
- Clicks to raw benchmark artifacts.
- Reproduction attempts, issues, and benchmark corrections.
- Questions about `B=1`, demand paging, kernel scope, or the Ragged L4 design.
- Hiring or collaboration conversations that reference a specific technical detail.

The strongest success event is not “nice project.” It is someone reproducing a result, finding a flaw, proposing a better scheduler/KV contract, or discussing the work in a technical interview.

**Confidence: Medium.** These metrics align with the project’s technical objective, but referral visibility and analytics availability vary by platform.

**Counter-interpretation:** If the immediate objective is job discovery, qualified recruiter conversations may matter more than public reproductions; decide the primary outcome before launch.

## 8. Claims guardrail

| Safe claim | Unsafe version |
|---|---|
| Educational inference-systems lab | Production inference engine |
| Custom Qwen serving path built from tensor primitives and official weights/tokenizer | Entire model ecosystem built from zero dependencies |
| Iteration-level request scheduling, currently per-request model forwards | True tensor/continuous batching |
| Shared block allocator with eager reservation | Memory-efficient PagedAttention equivalent to vLLM |
| Direct-block Triton decode-attention kernel for the pinned shape | Fully fused, general-purpose FlashAttention replacement |
| 9.7% improvement over this repository’s torch-paged mode under its fixed protocol | 9.7% faster than vLLM or production serving |
| Ragged L4 target architecture | Implemented Ragged L4 engine |
| Tested on one pinned L4 and model revision | General performance result across GPUs/models |

**Confidence: High.** These boundaries are directly traceable to the repository and the primary systems/model sources.[1][19][20][21][22]

**Counter-interpretation:** Short social copy cannot carry every caveat, but the model, hardware, baseline, and shipped-versus-target boundary should never be omitted.

## 9. Counter-review: where this strategy could be wrong

1. **Failure-first framing may overfit technical audiences.** It is excellent for specialists and hiring managers but may suppress broad engagement. That is acceptable if the objective is credible technical positioning, not vanity reach.
2. **The current release may still be too early for Reddit.** r/LocalLLaMA readers can reasonably say the 0.5B model, fixed synthetic workloads, and absence of true packed execution make it a tutorial rather than a new contribution. Waiting for the Ragged L4 milestone is a valid choice.[1]
3. **The benchmark failures do not generalize.** PagedAttention and continuous batching are validated techniques in larger serving systems; the result diagnoses this implementation’s missing contracts, not those techniques.[19][20]
4. **Video may not be the best X asset.** Platform documentation does not promise preferential organic distribution, and third-party data does not show video universally winning.[7][9]
5. **The eight-minute cut may be too long.** Keep it only if every segment contains visible proof. Otherwise collapse it to 3–5 minutes and let the documentation carry the rest.[18]
6. **Platform rules and product limits change.** Recheck native upload limits and all subreddit rules immediately before posting.

**Confidence: High that these are material risks; Low on their eventual impact.** The report can identify failure modes but cannot predict audience response.

## Key findings

- **Best position:** executable, falsifiable inference-systems lab—not a mini-vLLM.[1][19][20]
- **Best hook:** two optimizations made the server worse, and the traces explain why.[1]
- **LinkedIn:** engineering judgment and end-to-end ownership, delivered as a 100–120 second proof-led native video plus a result-discussion post.[2][3][6]
- **X:** three-post technical argument with a 50–60 second proof clip; video is evidence, not the whole message.[7][8][9]
- **Reddit:** self-contained failure analysis, rules and participation first, links second.[11][12][13][14]
- **Video:** one roughly eight-minute master with platform cuts; open on results and keep the Ragged L4 architecture visibly labeled as unshipped.[1][18]

## Limitations

No public source can predict organic reach, optimal posting time, or conversion for this account. Platform documentation covers formats and broad publishing practices but not a deterministic ranking recipe. The LinkedIn academic study concerns posts about software-engineering research rather than open-source launch posts. Buffer’s dataset covers posts published through Buffer and is correlational. Wistia measures videos hosted on its own platform, not social feeds. Reddit rules and moderator interpretations can change quickly. The project’s own performance evidence is limited to a small model, one GPU type, fixed workloads, and the measured release commit.

## References

[1] Cloud Inference Engine Lab. “Repository and README.” Source-Type: official. As Of: 2026-08-25. https://github.com/zeeshan8281/cloud-inference-from-scratch

[2] LinkedIn. “Share videos on LinkedIn.” Source-Type: official. As Of: 2026-08-25. https://www.linkedin.com/help/linkedin/answer/a7174587

[3] LinkedIn. “LinkedIn Sharing Guide.” Source-Type: official. As Of: undated. https://content.linkedin.com/content/dam/help/linkedin/en-us/LinkedIn-Sharing-Guide.pdf

[4] LinkedIn Engineering. “Leveraging Dwell Time to Improve Member Experiences on the LinkedIn Feed.” Source-Type: official. As Of: 2024-10. https://www.linkedin.com/blog/engineering/feed/leveraging-dwell-time-to-improve-member-experiences-on-the-linkedin-feed

[5] LinkedIn. “Upload and share documents on LinkedIn.” Source-Type: official. As Of: 2023. https://www.linkedin.com/help/linkedin/answer/a518909

[6] Rainer et al. “Beyond Self-Promotion: How Software Engineering Research Is Discussed on LinkedIn.” Source-Type: academic. As Of: 2024-01. https://arxiv.org/abs/2401.02268

[7] X Help. “How to share and watch videos on X.” Source-Type: official. As Of: 2026-08-25. https://help.x.com/en/using-x/x-videos

[8] X Help. “How to create a thread on X.” Source-Type: official. As Of: 2026-08-25. https://help.x.com/en/using-x/create-a-thread

[9] Buffer. “The State of Social Media Engagement in 2026.” Source-Type: secondary-industry. As Of: 2026-03-05. https://buffer.com/resources/state-of-social-media-engagement-2026/

[10] GitHub. “Five tips for promoting your open source project.” Source-Type: official. As Of: 2025-02-07. https://github.blog/open-source/maintainers/5-tips-for-promoting-your-open-source-project/

[11] Reddit Help. “Spam.” Source-Type: official. As Of: 2026-05-19. https://support.reddithelp.com/hc/en-us/articles/360043504051-Spam

[12] Reddit Help. “Reddiquette.” Source-Type: official. As Of: 2025-08-18. https://support.reddithelp.com/hc/en-us/articles/205926439-Reddiquette

[13] r/LocalLLaMA. “Community rules.” Source-Type: community. As Of: 2026-08-25. https://old.reddit.com/r/LocalLLaMA/about/rules/

[14] r/LocalLLaMA moderator. “Benchmark quality and self-promotion guidance.” Source-Type: community. As Of: 2026-04. https://www.reddit.com/r/LocalLLaMA/comments/1sg3vhp/removed_by_moderator/

[15] r/MachineLearning. “Self-Promotion Thread.” Source-Type: community. As Of: 2026-08. https://www.reddit.com/r/MachineLearning/comments/1vd5kqk/d_selfpromotion_thread/

[16] r/programming. “Community rules.” Source-Type: community. As Of: 2026-08-25. https://old.reddit.com/r/programming/about/rules/

[17] YouTube Help. “Add subtitles and captions.” Source-Type: official. As Of: 2026-08-25. https://support.google.com/youtube/answer/2734796

[18] Wistia. “2025 State of Video Report.” Source-Type: secondary-industry. As Of: 2025-03-26. https://downloads.ctfassets.net/j7pfe8y48ry3/64ag3cizBbgAeE0awGnbZ4/b4bf00170ad1453dfe180b89aa7c76f0/Wistia-2025-State-Of-Video-Report.pdf

[19] Kwon et al. “Efficient Memory Management for Large Language Model Serving with PagedAttention.” Source-Type: academic. As Of: 2023-09. https://arxiv.org/abs/2309.06180

[20] Yu et al. “Orca: A Distributed Serving System for Transformer-Based Generative Models.” Source-Type: academic. As Of: 2022-07. https://www.usenix.org/conference/osdi22/presentation/yu

[21] Triton. “Fused Attention tutorial.” Source-Type: official. As Of: 2026-08. https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

[22] Qwen. “Qwen2.5-0.5B model card.” Source-Type: official. As Of: 2024-09. https://huggingface.co/Qwen/Qwen2.5-0.5B
