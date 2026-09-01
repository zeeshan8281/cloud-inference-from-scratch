---
task_id: e
role: Walkthrough Video Strategist
status: complete
sources_found: 8
as_of: 2026-08-25
---

## Sources

[1] Cloud Inference Engine Lab | https://github.com/zeeshan8281/cloud-inference-from-scratch | Source-Type: official | Accessibility: public | As Of: 2026-08-25 | Authority: 10/10
[2] Share videos on LinkedIn | https://www.linkedin.com/help/linkedin/answer/a7174587 | Source-Type: official | Accessibility: public | As Of: 2026-08-25 | Authority: 10/10
[3] How to share and watch videos on X | https://help.x.com/en/using-x/x-videos | Source-Type: official | Accessibility: public | As Of: 2026-08-25 | Authority: 10/10
[4] How do I post and comment on Reddit? | https://support.reddithelp.com/hc/en-us/articles/360060422572-How-do-I-post-and-comment-on-Reddit | Source-Type: official | Accessibility: public | As Of: 2026-08-06 | Authority: 10/10
[5] Changelog — November 4, 2025 | https://support.reddithelp.com/hc/en-us/articles/42961314311700-Changelog-November-4-2025 | Source-Type: official | Accessibility: public | As Of: 2025-11-04 | Authority: 10/10
[6] Add subtitles & captions | https://support.google.com/youtube/answer/2734796 | Source-Type: official | Accessibility: public | As Of: 2026-08-25 | Authority: 10/10
[7] Wistia 2025 State of Video Report | https://downloads.ctfassets.net/j7pfe8y48ry3/64ag3cizBbgAeE0awGnbZ4/b4bf00170ad1453dfe180b89aa7c76f0/Wistia-2025-State-Of-Video-Report.pdf | Source-Type: secondary-industry | Accessibility: public | As Of: 2025-03-26 | Authority: 8/10
[8] 5 tips for promoting your open source project | https://github.blog/open-source/maintainers/5-tips-for-promoting-your-open-source-project/ | Source-Type: official | Accessibility: public | As Of: 2025-02-07 | Authority: 9/10

## Findings

- The walkthrough should be framed as “I built five LLM-serving stages on one L4, and the benchmark showed why two intuitive optimizations failed,” because the repository’s most credible differentiator is measured causal learning—not a claim that it replaces vLLM or SGLang. [1][8]
- Recommended 8:05 master timeline is 0:00–0:15 cold-open the failed gates and 9.7% Triton recovery; 0:15–0:45 state the educational/non-production boundary; 0:45–1:25 trace one request through API, scheduler, model, KV cache, and GPU; 1:25–2:15 prove custom Qwen execution and parity; 2:15–3:05 show contiguous KV’s 1.68× result; 3:05–4:10 show batching cutting TTFT while throughput falls; 4:10–5:15 show eager paged allocation worsening memory; 5:15–6:20 show direct-block Triton eliminating the decode gather and gaining 9.7%; 6:20–7:10 show test/failure-path evidence; 7:10–7:50 mark Ragged L4 as the next unshipped design; and 7:50–8:05 give one reproducibility CTA. [1]
- The proof moments should be captured live or from committed artifacts: exact token parity, 45/45 local and Modal CPU/API tests, 34/34 L4 checks, the 27.5-to-21.9 tok/s batching regression beside the 9276-to-230 ms TTFT improvement, the paged cache’s -4.45% memory gate, 567.7 MiB versus 2.63 MiB gather traffic, and Triton/reference absolute difference below 0.003 through context 2048. [1]
- A 6–10 minute master is defensible for an intentional tutorial audience, but it should reveal the thesis in the first 15 seconds and make every section independently navigable because Wistia’s 2025 dataset found shorter videos have higher percentage engagement while how-to content retains attention better than most long-form formats. [7]
- Record the master at 1920×1080 with a single dominant surface, crop code or terminal output until it is readable on a phone, use a small architecture inset only when it explains the current frame, show metric callouts beside their baselines, keep all edges clear for platform UI, and never display the API secret, auth header, notifications, or an unedited desktop. [1][2]
- Produce a reviewed SRT/VTT transcript for YouTube and LinkedIn, burn concise captions into every social cut, describe important visual-only changes in narration, and keep caption placement inside the safe area; YouTube supports timed caption files, LinkedIn supports uploaded or reviewed automatic captions, and X timelines may autoplay with captions while audio is unavailable or off. [2][3][6]
- LinkedIn cut: 100–120 seconds in 4:5 or 1:1, with 0:00–0:04 “two optimizations made my LLM server worse,” 0:04–0:18 request-path diagram, 0:18–0:48 batching failure, 0:48–1:12 paging failure, 1:12–1:38 Triton recovery, and 1:38–1:50 repo/reproduce CTA; this sits far below LinkedIn’s 15-minute/5 GB ceiling and should use its caption and thumbnail controls plus explicit edge-safe composition. [1][2]
- X cut: 50–60 seconds in 16:9 or 1:1, with the failed gate in frame one, 8 seconds of architecture, 15 seconds each for the batching and paging causal result, 10 seconds for Triton proof, and a final repo/technical-question card; keeping it at or below 60 seconds makes it loop and keeps it universally below the non-Premium 140-second/512 MB limit. [1][3]
- Reddit cut: 90–120 seconds in 16:9 or 1:1, lead with the benchmark table rather than personal branding, spend most of the clip on how the profiler falsified the batching and paging assumptions, put reproduction details and limitations in the post body, and end by asking for critique of one Ragged L4 design choice; verify the target community permits video because Reddit says post types depend on community rules, while official limits allow 15-minute non-Premium and 30-minute Premium video uploads. [1][4][5]
- The closing CTA should be “run the pinned smoke/benchmark commands, inspect the raw artifacts, and challenge the next scheduler/KV design,” while the edit should avoid logo intros, typing in real time, unsupported production claims, victory-lap language over failed gates, unexplained benchmark numbers, a generic request for stars, or presenting the Ragged L4 target as shipped. [1][7][8]

## Deep Read Notes

### Source [1]: Cloud Inference Engine Lab
Key data: the public project contains five measured stages, 45 local/CPU tests, 34 GPU checks, exact token parity, a 1.68× contiguous-over-naive gain, two failed batching/paging gates, a 9.7% Triton-over-paged gain, raw benchmark artifacts, and an explicitly unshipped Ragged L4 target.
Key insight: the video can sustain a serious engineering narrative because every turning point has a number, a causal explanation, and a reproducible artifact; the failed gates are the story rather than material to hide.
Useful for: opening hook, chapter order, screen captures, exact claims, limitation language, CTA, and the distinction between shipped evidence and future architecture.

### Source [3]: X native video help
Key data: non-Premium accounts can upload up to 140 seconds/512 MB; Premium supports video under four hours at up to 1080p/16 GB; native video autoplays, videos of 60 seconds or less loop, captions/subtitles are available, and a single-video post can include timestamps.
Key insight: a self-contained 50–60 second proof cut is the lowest-friction X asset, while the full walkthrough should remain a linked destination or a separate Premium upload instead of bloating the launch post.
Useful for: X duration, export ceiling, caption QA, loop endpoint, and keeping the final frame visually compatible with a replay into the opening claim.

### Source [7]: Wistia 2025 State of Video Report
Key data: Wistia analyzed more than 100 million hosted videos and 2.7 million hours of viewing; shorter videos generally earned higher percentage engagement, but how-to content retained about 82% below one minute and remained above 50% across one-to-30-minute lengths, with lower retention for 30–60-minute how-to content.
Key insight: length should follow viewer intent—the long master can teach, while short platform cuts should deliver one proof immediately—and percentage engagement should not be confused with total useful watch time or comprehension.
Useful for: the 8-minute master/short-cut split, eliminating a slow introduction, proof-first ordering, and the counter-claim that a shorter master might perform better.

## Gaps

- No public platform source proves that 4:5, 1:1, or 16:9 will receive more organic distribution for this developer audience; aspect-ratio recommendations here are production choices for readability and reuse, not ranking claims.
- Reddit’s public help documents community-dependent video posting and account-level duration limits but does not publish a complete current organic-post codec, caption-file, bitrate, or safe-zone specification, so the final Reddit export must be test-uploaded and burned-in captions are the dependable fallback.
- Wistia measures videos hosted on its own platform, not LinkedIn, X, Reddit, or YouTube developer walkthroughs, and engagement is viewing percentage rather than a direct measure of technical comprehension or repository adoption.
- The proposed timings are an editorial hypothesis anchored to the project’s evidence hierarchy; after publishing, inspect first-30-second retention, chapter exits, social completion, repository referral clicks, and reproduction attempts before making a second cut.
- Counter-claim: a 3–5 minute master may outperform the proposed 8:05 version in completion rate, and Wistia’s broad data does not prove that this project needs eight minutes; keep the long cut only if each proof moment answers a distinct engineering question, otherwise publish a compact proof reel and let the README carry implementation depth. [1][7]
