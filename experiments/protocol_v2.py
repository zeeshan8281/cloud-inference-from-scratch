"""Correctness Protocol V2 calibration + sealed-holdout harness.

Implements CORRECTNESS_PROTOCOL_V2.md (repo root) requirements 3-10. Does
NOT resume the 10-pair performance protocol -- see main.sentinel_pilot for
that; this module only measures whether the concurrency > 1 correctness
gate, revised to a preregistered near-tie tolerance, would pass.

Separate from experiments/sentinel_diagnostics.py, which investigated one
specific request and is complete; this module runs a broader, disjoint
calibration/holdout sample to propose and then test a general tolerance.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from experiments.controlled import Cell
from experiments.sentinel_diagnostics import TOP_K, compare_top_k, step_margin, top_k_from_logits
from experiments.sentinel_pilot import prompt_seed, sentinel_token_ids

PROTOCOL_VERSION = "correctness-protocol-v2"
# Concurrency 1 is included as a zero-tolerance sanity check (requirement 1:
# no batch composition exists to vary at c1, so any disagreement there is a
# correctness bug, never a near-tie). c8/c32 are where the bounded diagnostic
# found divergence and are what this protocol's tolerance actually governs.
V2_CELLS: tuple[Cell, ...] = (Cell(512, 128, 1), Cell(512, 128, 8), Cell(512, 128, 32))
V2_STEPS = 8
V2_BATCHES_PER_CELL = {1: 3, 8: 5, 32: 2}  # bounded: 3*1 + 5*8 + 2*32 = 107 requests per split


# --------------------------------------------------------------------------
# Deterministic, disjoint calibration/holdout batch construction
# --------------------------------------------------------------------------


def build_split_batches(tokenizer: Any, split: str) -> dict[str, list[list[list[int]]]]:
    """`split` is "calibration" or "holdout" -- distinct seed namespaces make
    the two sets disjoint by construction, not by post-hoc deduplication."""
    if split not in ("calibration", "holdout"):
        raise ValueError(f"unknown split: {split}")
    batches: dict[str, list[list[list[int]]]] = {}
    for cell in V2_CELLS:
        cell_batches = []
        for batch_index in range(V2_BATCHES_PER_CELL[cell.concurrency]):
            batch = [
                sentinel_token_ids(
                    prompt_seed(f"protocol-v2-{split}", batch_index + 1, cell.name, request_index, "v2"),
                    cell.input_tokens,
                    tokenizer,
                )
                for request_index in range(cell.concurrency)
            ]
            cell_batches.append(batch)
        batches[cell.name] = cell_batches
    return batches


# --------------------------------------------------------------------------
# Per-request first-disagreement diagnosis and classification (pure)
# --------------------------------------------------------------------------


def first_disagreement(custom_result: dict[str, Any], vllm_result: dict[str, Any]) -> dict[str, Any] | None:
    """None if the two engines agree over the whole observed window;
    otherwise the full diagnostic bundle at the first differing position."""
    custom_tokens = custom_result["output_tokens"]
    vllm_tokens = vllm_result["output_tokens"]
    window = min(len(custom_tokens), len(vllm_tokens))
    for position in range(window):
        if custom_tokens[position] == vllm_tokens[position]:
            continue
        custom_top_k = custom_result["top_k"][position]
        vllm_top_k = vllm_result["top_k"][position]
        comparison = compare_top_k(custom_top_k, vllm_top_k)
        return {
            "position": position,
            "custom_token": custom_tokens[position],
            "vllm_token": vllm_tokens[position],
            "custom_margin": step_margin(custom_top_k),
            "vllm_margin": step_margin(vllm_top_k),
            "max_abs_diff": comparison["max_abs_diff"],
            "mean_abs_diff": comparison["mean_abs_diff"],
            "cosine_similarity": comparison["cosine_similarity"],
            "top_k_overlap": comparison["top_k_overlap"],
            "custom_token_in_vllm_top_k": custom_tokens[position] in dict(vllm_top_k),
            "vllm_token_in_custom_top_k": vllm_tokens[position] in dict(custom_top_k),
        }
    return None


def _is_finite(top_k: list[tuple[int, float]]) -> bool:
    return all(math.isfinite(value) for _, value in top_k)


def classify_request(
    custom_result: dict[str, Any],
    vllm_result: dict[str, Any],
    epsilon: float | None,
    concurrency: int = 2,
) -> dict[str, Any]:
    """Per CORRECTNESS_PROTOCOL_V2.md requirements 1, 3 and 5. `epsilon` is
    None during calibration (classification is deferred until it's
    proposed); a committed float during holdout evaluation. `concurrency`
    defaults to 2 (i.e. "tolerance may apply") for standalone/test callers
    that don't track it; real callers must pass the request's actual
    concurrency so a concurrency-1 disagreement is never tolerated."""
    reasons: list[str] = []
    if not custom_result["verified"]:
        reasons.append("custom_disagrees_with_own_top_k")
    if not vllm_result["verified"]:
        reasons.append("vllm_disagrees_with_own_top_k")
    for step_top_k in custom_result["top_k"]:
        if not _is_finite(step_top_k):
            reasons.append("custom_non_finite_logit")
            break
    for step_top_k in vllm_result["top_k"]:
        if not _is_finite(step_top_k):
            reasons.append("vllm_non_finite_logit")
            break

    disagreement = first_disagreement(custom_result, vllm_result)
    if disagreement is None:
        verdict = "hard_failure" if reasons else "exact_match"
        return {"verdict": verdict, "disagreement": None, "reasons": reasons}

    if concurrency <= 1:
        # Requirement 1: no batch composition exists to vary at concurrency
        # 1, so a disagreement here is never near-tie-qualified regardless
        # of margin -- it is a correctness bug in one of the two engines.
        reasons.append("disagreement_at_concurrency_one")

    if disagreement["top_k_overlap"] < 0.3:
        reasons.append("low_top_k_overlap")
    if not disagreement["custom_token_in_vllm_top_k"]:
        reasons.append("custom_token_missing_from_vllm_top_k")
    if not disagreement["vllm_token_in_custom_top_k"]:
        reasons.append("vllm_token_missing_from_custom_top_k")

    if epsilon is None:
        verdict = "hard_failure" if reasons else "disagreement_unclassified_pending_epsilon"
        return {"verdict": verdict, "disagreement": disagreement, "reasons": reasons}

    margins_within_epsilon = (
        disagreement["custom_margin"] is not None
        and disagreement["custom_margin"] <= epsilon
        and disagreement["vllm_margin"] is not None
        and disagreement["vllm_margin"] <= epsilon
    )
    if not margins_within_epsilon:
        reasons.append("confident_disagreement_no_near_tie")

    verdict = "hard_failure" if reasons else "near_tie_qualified"
    return {"verdict": verdict, "disagreement": disagreement, "reasons": reasons}


def propose_epsilon(calibration_classifications: list[dict[str, Any]]) -> dict[str, Any]:
    """Per requirement 6/7: epsilon comes only from the calibration set's
    observed margins among disagreements that are otherwise clean (each
    engine's own top-k self-consistent, no non-finite logits, no low top-k
    overlap, each selected token present in the other engine's top-k) --
    i.e. candidate near-ties, not disagreements already disqualified for
    other reasons. epsilon is the maximum such margin observed, so every
    calibration-set near-tie candidate would in fact qualify."""
    candidate_margins = []
    for entry in calibration_classifications:
        if entry["disagreement"] is None:
            continue
        disallowed = {
            "custom_disagrees_with_own_top_k",
            "vllm_disagrees_with_own_top_k",
            "custom_non_finite_logit",
            "vllm_non_finite_logit",
            "low_top_k_overlap",
            "custom_token_missing_from_vllm_top_k",
            "vllm_token_missing_from_custom_top_k",
            "disagreement_at_concurrency_one",
            # Cannot occur during calibration (epsilon is None, so
            # classify_request never adds this reason there), but excluded
            # defensively in case this is ever called on already-epsilon-
            # classified data.
            "confident_disagreement_no_near_tie",
        }
        if disallowed & set(entry["reasons"]):
            continue
        margins = entry["disagreement"]["custom_margin"], entry["disagreement"]["vllm_margin"]
        if None not in margins:
            candidate_margins.append(max(margins))
    if not candidate_margins:
        return {"epsilon": None, "candidate_count": 0, "candidate_margins": []}
    return {
        "epsilon": max(candidate_margins),
        "candidate_count": len(candidate_margins),
        "candidate_margins": sorted(candidate_margins),
    }


def summarize(classifications: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(classifications)
    exact = sum(1 for c in classifications if c["verdict"] == "exact_match")
    near_tie = sum(1 for c in classifications if c["verdict"] == "near_tie_qualified")
    hard_failures = [c for c in classifications if c["verdict"] == "hard_failure"]
    pending = sum(1 for c in classifications if c["verdict"] == "disagreement_unclassified_pending_epsilon")
    return {
        "total": total,
        "exact_match": exact,
        "exact_match_pct": round(100 * exact / total, 2) if total else None,
        "near_tie_qualified": near_tie,
        "near_tie_qualified_pct": round(100 * near_tie / total, 2) if total else None,
        "hard_failures": len(hard_failures),
        "hard_failure_pct": round(100 * len(hard_failures) / total, 2) if total else None,
        "pending_epsilon": pending,
        "hard_failure_detail": hard_failures,
        "near_tie_margins": [
            c["disagreement"]["custom_margin"] for c in classifications if c["verdict"] == "near_tie_qualified"
        ],
    }


# --------------------------------------------------------------------------
# Engine runners: capture per-request top-k for every request in a batch
# (GPU required; imports deferred)
# --------------------------------------------------------------------------


async def run_custom_full_batch(
    model_dir: str, engine_options: dict[str, Any], batch_ids: list[list[int]], steps: int = V2_STEPS
) -> list[dict[str, Any]]:
    """Same non-invasive, self-verifying hook as
    experiments.sentinel_diagnostics.run_custom_diagnostic, generalized to
    every request in the batch instead of one target."""
    import asyncio

    import torch

    from cloud_engine.config import build_config
    from cloud_engine.engine import InferenceEngine, RequestHandle
    from cloud_engine.scheduler import GenerationConfig

    config = build_config(
        "ragged",
        cuda_graph_decode=engine_options.get("cuda_graph_decode", False),
        use_triton_attention=engine_options.get("use_triton_attention", True),
        prefix_cache_max_blocks=engine_options.get("prefix_cache_max_blocks", 0),
        max_active_sequences=engine_options.get("max_active_sequences", 32),
        max_batched_tokens=engine_options.get("max_batched_tokens", 2048),
        prefill_chunk_size=engine_options.get("prefill_chunk_size", 256),
    )
    engine = InferenceEngine(config, model_dir=model_dir)
    await engine.start()

    runner = engine.runner
    captured_packs: list[Any] = []
    captured_logits: list[Any] = []
    original_pack = runner._pack
    original_forward = runner.model.forward
    original_graph_forward = runner._graph_forward

    def wrapped_pack(plan):
        packed = original_pack(plan)
        captured_packs.append(packed)
        return packed

    def record(logits):
        captured_logits.append(logits.detach().to(torch.float32).cpu().clone())
        return logits

    def wrapped_forward(*args, **kwargs):
        return record(original_forward(*args, **kwargs))

    def wrapped_graph_forward(*args, **kwargs):
        return record(original_graph_forward(*args, **kwargs))

    runner._pack = wrapped_pack
    runner.model.forward = wrapped_forward
    runner._graph_forward = wrapped_graph_forward

    async def submit_one(index: int, ids: list[int]):
        return await engine.scheduler.submit(
            f"v2-{index}", ids, GenerationConfig(max_output_tokens=steps, temperature=0, eos_token_id=None)
        )

    try:
        requests = await asyncio.gather(*(submit_one(index, ids) for index, ids in enumerate(batch_ids)))
        handles = [RequestHandle(request, engine) for request in requests]
        await asyncio.gather(*(handle.wait() for handle in handles))
    finally:
        runner._pack = original_pack
        runner.model.forward = original_forward
        runner._graph_forward = original_graph_forward
        await engine.close()

    per_request_top_k: dict[str, list[Any]] = {request.request_id: [] for request in requests}
    for pack, logits in zip(captured_packs, captured_logits, strict=True):
        for row, request_id in enumerate(pack.sampled_request_ids):
            if request_id in per_request_top_k:
                per_request_top_k[request_id].append(top_k_from_logits(logits[row].tolist()))

    results = []
    for index, request in enumerate(requests):
        tokens = list(request.generated_token_ids)[:steps]
        top_k = per_request_top_k[request.request_id][: len(tokens)]
        verified = len(top_k) >= len(tokens) and all(top_k[i][0][0] == tokens[i] for i in range(len(tokens)))
        results.append({"index": index, "output_tokens": tokens, "top_k": top_k, "verified": verified})
    return results


def run_vllm_full_batch(
    model_dir: str,
    model: dict[str, Any],
    engine_options: dict[str, Any],
    batch_ids: list[list[int]],
    steps: int = V2_STEPS,
) -> list[dict[str, Any]]:
    from vllm import EngineArgs, LLMEngine, SamplingParams

    args = EngineArgs(
        model=model_dir,
        tokenizer=model_dir,
        dtype=engine_options.get("dtype", "half"),
        max_model_len=model["max_model_len"],
        block_size=16,
        disable_log_stats=True,
        trust_remote_code=False,
        seed=0,
        max_num_seqs=max(len(batch_ids), 1),
        max_num_batched_tokens=2048 * 4,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=engine_options.get("enable_prefix_caching", False),
        enforce_eager=engine_options.get("enforce_eager", True),
    )
    engine = LLMEngine.from_engine_args(args)
    params = SamplingParams(
        temperature=0, top_p=1, top_k=-1, max_tokens=steps, ignore_eos=True, logprobs=TOP_K
    )
    try:
        for index, ids in enumerate(batch_ids):
            engine.add_request(f"v2-{index}", {"prompt_token_ids": ids}, params)
        finished: dict[str, Any] = {}
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    finished[output.request_id] = output.outputs[0]
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            shutdown()

    results = []
    for index in range(len(batch_ids)):
        output = finished[f"v2-{index}"]
        tokens = list(output.token_ids)[:steps]
        top_k = []
        for step_logprobs in (output.logprobs or [])[:steps]:
            ranked = sorted(step_logprobs.items(), key=lambda item: item[1].logprob, reverse=True)
            top_k.append([(token_id, entry.logprob) for token_id, entry in ranked[:TOP_K]])
        results.append({"index": index, "output_tokens": tokens, "top_k": top_k, "verified": True})
    return results


def main() -> None:
    import asyncio

    if len(sys.argv) != 3:
        raise SystemExit("usage: protocol_v2.py <config.json> <output.json>")
    config = json.loads(Path(sys.argv[1]).read_text())
    output_path = Path(sys.argv[2])

    pinned = json.loads(Path("/root/engine_config.json").read_text())
    model = pinned["ragged_model"]
    model_dir = config["model_dir"]
    implementation = config["implementation"]
    engine_options = config.get("engine_options", {})
    batches = config["batches"]  # list of batch_ids (list[list[int]])

    if implementation == "custom":
        batch_results = [
            asyncio.run(run_custom_full_batch(model_dir, engine_options, batch)) for batch in batches
        ]
    elif implementation == "vllm":
        batch_results = [run_vllm_full_batch(model_dir, model, engine_options, batch) for batch in batches]
    else:
        raise SystemExit(f"unknown implementation: {implementation}")

    output_path.write_text(json.dumps({"config": config, "batch_results": batch_results}, indent=2))


if __name__ == "__main__":
    main()
