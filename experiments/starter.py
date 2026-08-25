"""Starter experiment: shortest remaining prefill first, decode still wins."""

NAME = "short-prefill-first"
HYPOTHESIS = "Short prefills reduce queueing and TTFT without changing generated tokens."

# Optional safe knobs: max_active_sequences, max_batched_tokens, prefill_chunk_size.
CONFIG_OVERRIDES = {"prefill_chunk_size": 128}


def priority(candidate):
    return (
        candidate.phase != "decode",
        candidate.remaining_tokens if candidate.phase == "prefill" else 0,
        candidate.arrival_ns,
    )


# Optional under KV pressure; the candidate is read-only.
def preemption_priority(candidate):
    return (candidate.allocated_tokens, candidate.tokens_fed, candidate.arrival_ns)
