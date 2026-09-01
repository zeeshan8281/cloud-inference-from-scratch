# P7 Verification

AS_OF: 2026-08-25

## Registry checks

- 22/22 approved references are cited in the report.
- Every report citation resolves to an approved registry entry.
- No dropped source appears in the report.
- Citation density: one citation marker per 54.9 words.
- Official-source share: 59.1%; 15 unique domains; maximum domain share: 18.2%.

## Spot checks

1. Repository results (`1.68×`, `0.80×`, `4.45% worse`, `1.10×`, `567.7 MiB`, `2.63 MiB`) trace to task-a/task-d findings and the public README/artifact source. Pass.
2. LinkedIn native-video/caption mechanics and lack of a simple dwell-time ranking rule trace to task-a sources [2]-[4]. Pass.
3. X upload/thread mechanics and the warning that video is not automatically superior trace to task-b sources [2], [3], and [6]. Pass.
4. Reddit spam, disclosure, r/LocalLLaMA benchmark-quality, r/MachineLearning routing, and r/programming restrictions trace to task-c sources [1]-[7]. Pass.
5. The distinction between request rotation and operation batching, and between eager block reservation and demand paging, traces to task-d and the Orca/PagedAttention primary sources. Pass.
6. Caption workflow and short-cut/long-master split trace to task-e sources [2], [3], [6], and [7]. Pass.

## Corrections made after counter-review

- Replaced the technically confused packed-decode-versus-ragged-prefill CTA with an ordered question that starts after the packed Torch reference is correct.
- Qualified the central hook so it refers to this project's first-pass modes, not continuous batching or PagedAttention generally.
- Removed unsupported 2–4 day LinkedIn timing advice.
- Split confidence between technical positioning, platform fit, distribution, and volatile community rules.
- Made Reddit clip duration an optional testable hypothesis rather than an evidence-backed optimum.
- Dropped an unused Modal source so every approved reference is actually cited.

Result: pass. No remaining citation or traceability violations found.
