---
task_id: a
role: LinkedIn Distribution Strategist
status: complete
sources_found: 8
---

## Sources

[1] Cloud Inference Engine Lab | https://github.com/zeeshan8281/cloud-inference-from-scratch | Source-Type: official | Accessibility: public | As Of: 2026-08-25 | Authority: 10/10
[2] Share videos on LinkedIn | https://www.linkedin.com/help/linkedin/answer/a7174587 | Source-Type: official | Accessibility: public | As Of: 2025 | Authority: 10/10
[3] LinkedIn Sharing Guide | https://content.linkedin.com/content/dam/help/linkedin/en-us/LinkedIn-Sharing-Guide.pdf | Source-Type: official | Accessibility: public | As Of: undated | Authority: 8/10
[4] Leveraging Dwell Time to Improve Member Experiences on the LinkedIn Feed | https://www.linkedin.com/blog/engineering/feed/leveraging-dwell-time-to-improve-member-experiences-on-the-linkedin-feed | Source-Type: official | Accessibility: public | As Of: 2024-10 | Authority: 10/10
[5] Upload and share documents on LinkedIn | https://www.linkedin.com/help/linkedin/answer/a518909 | Source-Type: official | Accessibility: public | As Of: 2023 | Authority: 9/10
[6] The Art & Science of Video Storytelling | https://business.linkedin.com/content/dam/business/marketing-solutions/global/en_US/site/pdf/wp/2025/the-art-and-science-of-video.pdf | Source-Type: official | Accessibility: public | As Of: 2025 | Authority: 8/10
[7] Beyond Self-Promotion: How Software Engineering Research Is Discussed on LinkedIn | https://arxiv.org/abs/2401.02268 | Source-Type: academic | Accessibility: public | As Of: 2024-01 | Authority: 9/10
[8] Post and share updates | https://www.linkedin.com/help/linkedin/answer/a528176 | Source-Type: official | Accessibility: public | As Of: 2026-08 | Authority: 10/10

## Findings

- The credible target audience is inference/GPU engineers, ML-systems learners, and technical hiring managers, and the defensible positioning is “a tested educational inference-engine lab that exposes where simplified batching and paging fail,” not “a production vLLM replacement.” [1]
- The strongest launch hook is the measured surprise—“I built five stages of a Qwen inference engine on one L4; two optimizations made it worse”—because the repository publishes both failed gates, explains their root causes, and distinguishes shipped results from the future Ragged L4 target. [1]
- The primary launch asset should be a native 60-120 second vertical walkthrough with a custom thumbnail, reviewed captions, and UI-safe margins; LinkedIn supports native videos up to 15 minutes, while its sharing guide recommends vertical filming, an immediate visual/opening-line hook, on-screen context, and roughly 30 seconds to two minutes. [2][3]
- A high-credibility video structure is `0-5s failed-results hook -> 5-20s architecture -> 20-45s live request/SSE demo -> 45-75s benchmark table and why batching/paging failed -> 75-100s Triton win -> final technical question`, keeping the claim-to-proof distance short and showing the actual repository rather than only talking to camera. [1][3]
- The accompanying text should use a result-discussion structure—hook, what was built, three measured results, two failures and their causes, what changes next, GitHub link, then one specific engineering question—because strong software-engineering posts in a 98-post academic sample explained results in enough detail to establish relevance and added the poster's own interpretation rather than merely announcing an achievement. [7]
- A concise launch post can stay well below LinkedIn's 3,000-character limit, but should still use short paragraphs or bullets: the study's positive examples were often longer than the sample median and remained readable through headings, paragraphs, bullet points, and a supporting result figure. [7][8]
- A native PDF document is best used as a separate follow-up asset—approximately 6-8 slides covering architecture, the five stages, benchmark gates, failure analysis, Triton kernel, and next milestone—because LinkedIn positions document posts for knowledge sharing and allows one downloadable PDF up to 100 MB and 300 pages, but does not support document animation. [1][5]
- The call to action should invite experience rather than applause, for example “If this were your engine, would you fix packed execution or demand-paged KV first?”, because LinkedIn's guide recommends asking a question to start conversation and the software-engineering study treats questions, criticism, and experience sharing as higher-value discussion than congratulations. [3][7]
- Human, conversational delivery is a reasonable creative choice—brief face-to-camera opening followed by terminal, architecture, and chart footage—because LinkedIn's analysis of more than 13,000 B2B video ads associated expert speakers with 31% higher dwell and conversational treatment with 13% higher dwell, although those ad results must not be presented as guaranteed organic-post lift. [6]
- Counter-claim: video is not automatically favored or superior, since LinkedIn says feed ranking balances multiple passive and active objectives and normalizes long-dwell expectations by content type and other attributes, so relevance, technical proof, and audience fit matter more than stretching watch time or using “algorithm hacks.” [4]

## Deep Read Notes

### Source [1]: Cloud Inference Engine Lab
Key data: The public README reports 45 local/CPU checks, 34 L4 checks, exact greedy-token parity across five modes, a 1.68x cache win, a 0.80x batching failure, a -4.45% paging failure, and a 1.10x Triton-over-paged win.
Key insight: The unusual portfolio value is the honest experimental narrative—implementation, reproducible evidence, failed hypotheses, root-cause analysis, and a sharply specified next milestone—not a generic “built an LLM API” announcement.
Useful for: Audience selection, hook, proof points, video storyboard, precise disclaimers, and the final technical question.

### Source [4]: Leveraging Dwell Time to Improve Member Experiences on the LinkedIn Feed
Key data: LinkedIn's feed uses a multi-objective system that incorporates passive and active behavior; the 2024 engineering post says video dwell is useful but noisy and that fixed universal thresholds create content-type bias.
Key insight: There is no official basis for simplistic claims such as “make the post 31 seconds” or “video always wins”; the system evaluates relevance and quality relative to context and evolving behavior distributions.
Useful for: Countering algorithm folklore and keeping the launch optimized for the right technical audience rather than broad low-intent reach.

### Source [7]: Beyond Self-Promotion: How Software Engineering Research Is Discussed on LinkedIn
Key data: The ICSE 2024 study examined 98 posts; only 34% elaborated on or discussed results, positive examples explained significance and industrial relevance, 71% of comments came from industry, and 54% of comments were low-information congratulations.
Key insight: A technically useful interpretation of results produces a better portfolio signal than an achievement announcement, especially when the post is structured for scanning and asks for practitioner experience or criticism.
Useful for: Post structure, credibility framing, comment prompt, and examples of result-led engineering communication.

## Gaps

- LinkedIn publishes upload limits and broad creative guidance but no public rule promising higher organic reach for native video, documents, a specific posting time, link placement, hashtag count, or a fixed watch-time threshold.
- The LinkedIn video-effect percentages come from paid B2B ads, not organic individual engineering posts, so they support a human/expert creative hypothesis only and should not be converted into reach forecasts.
- The software-engineering study covers posts about academic papers from 2018-2023 rather than open-source project launches in 2026, so its result-discussion pattern is transferable but its engagement levels are not a benchmark for this account.
- Alternative interpretation: a narrowly technical post may attract fewer total reactions than a broad AI demo, but fewer high-relevance views from inference engineers and hiring managers can still be the better portfolio outcome.
