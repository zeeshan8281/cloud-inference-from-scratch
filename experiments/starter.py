"""Starter experiment: shortest remaining prefill first, decode still wins."""

NAME = "short-prefill-first"
HYPOTHESIS = "Short prefills reduce queueing and TTFT without changing generated tokens."


def priority(candidate):
    return (
        candidate.phase != "decode",
        candidate.remaining_tokens if candidate.phase == "prefill" else 0,
        candidate.arrival_ns,
    )
