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
from experiments.sentinel_diagnostics import (
    TOP_K,
    check_cross_request_identity,
    compare_top_k,
    cross_choice_diagnostics,
    step_margin,
    top_k_from_logits,
)
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
# Bounded audit of the two sealed-holdout hard failures (diagnostic only --
# per CORRECTNESS_PROTOCOL_V2.md audit instructions, this does NOT change
# the sealed verdict: Protocol V2 still failed as originally preregistered)
# --------------------------------------------------------------------------

# The two sealed-holdout hard failures, located by manually recovering each
# request's position within its batch from protocol-v2-holdout.json (that
# file does not itself store index_in_batch).
AUDIT_TARGETS: tuple[dict[str, Any], ...] = (
    {"cell": "in512-out128-c8", "batch_index": 1, "index_in_batch": 6},
    {"cell": "in512-out128-c32", "batch_index": 1, "index_in_batch": 22},
)


def build_audit_variants(tokenizer: Any, target: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Three batch-construction variants for one flagged request, per the
    audit instructions: alone, in its original batch and position, and in
    the same batch reordered so the target is first. Rebuilt from the same
    deterministic holdout seeds `build_split_batches` already uses -- this
    is reconstruction, not new prompt content, so it stays inside the
    sealed holdout namespace (these two requests are no longer being used
    as validation data, only re-examined as diagnostics)."""
    batches_by_cell = build_split_batches(tokenizer, "holdout")
    original_batch = batches_by_cell[target["cell"]][target["batch_index"]]
    index_in_batch = target["index_in_batch"]
    target_ids = original_batch[index_in_batch]
    reordered = [target_ids, *(ids for i, ids in enumerate(original_batch) if i != index_in_batch)]
    return {
        "alone": {"batch_ids": [target_ids], "target_index": 0},
        "original": {"batch_ids": original_batch, "target_index": index_in_batch},
        "reordered_target_first": {"batch_ids": reordered, "target_index": 0},
    }


def batch_vs_solo_drift(alone_result: dict[str, Any], batched_result: dict[str, Any]) -> dict[str, Any]:
    """Same-engine comparison of one request's own top-k between running it
    alone and running it co-batched, at every step where both conditions
    still share the same generated prefix (a later step is only comparable
    if the two conditions agree on every token up to it). `intersection_*`
    metrics come from compare_top_k, so this reports no invented values
    either. `high_drift` uses a fixed, stated 0.01 log-prob bound purely to
    flag steps worth a human look in this diagnostic report -- it is not a
    preregistered protocol threshold and must not be used as one."""
    alone_tokens = alone_result["output_tokens"]
    batched_tokens = batched_result["output_tokens"]
    window = min(len(alone_tokens), len(batched_tokens))
    steps = []
    drift_detected = False
    for position in range(window):
        if alone_tokens[position] != batched_tokens[position]:
            # The request's own output token changed depending on whether it
            # ran alone or co-batched -- a stronger signal than any log-prob
            # magnitude threshold below, so this always counts as drift.
            drift_detected = True
            steps.append({"position": position, "prefix_diverged": True})
            break
        comparison = compare_top_k(alone_result["top_k"][position], batched_result["top_k"][position])
        high_drift = (
            comparison["intersection_max_abs_diff"] is not None and comparison["intersection_max_abs_diff"] > 0.01
        )
        drift_detected = drift_detected or high_drift
        steps.append({"position": position, "prefix_diverged": False, "high_drift": high_drift, **comparison})
    return {"drift_detected": drift_detected, "steps": steps}


def analyze_audit_entry(variants: dict[str, dict[str, Any]], runs: dict[str, dict[str, Any]], epsilon: float) -> dict:
    """Full diagnostic bundle for one audited request: per-variant
    classification (under the corrected metrics and the already-committed
    epsilon, for comparison against the original sealed verdict only -- not
    to replace it), raw top-k, cross-request identity checks for every
    batched variant, and batch-vs-solo drift for both engines."""
    crashed = {name: r for name, r in runs.items() if r["crashed"]}
    if crashed:
        return {"crashed": crashed}

    per_variant = {}
    for variant_name, variant in variants.items():
        target_index = variant["target_index"]
        concurrency = 1 if variant_name == "alone" else len(variant["batch_ids"])
        per_variant[variant_name] = {}
        engine_targets: dict[str, dict[str, Any]] = {}
        for implementation in ("custom", "vllm"):
            batch = runs[f"{variant_name}_{implementation}"]["result"]["result"]
            target_result = batch[target_index]
            engine_targets[implementation] = target_result
            identity_check = None
            if variant_name != "alone":
                others = [{"index": r["index"], "output_tokens": r["output_tokens"]} for r in batch]
                identity_check = check_cross_request_identity(
                    {"index": target_index, "output_tokens": target_result["output_tokens"]}, others
                )
            per_variant[variant_name][f"{implementation}_identity_check"] = identity_check
            per_variant[variant_name][f"{implementation}_target"] = target_result
        per_variant[variant_name]["classification"] = classify_request(
            engine_targets["custom"], engine_targets["vllm"], epsilon=epsilon, concurrency=concurrency
        )

    return {
        "per_variant": per_variant,
        "batch_vs_solo_drift": {
            implementation: batch_vs_solo_drift(
                per_variant["alone"][f"{implementation}_target"],
                per_variant["original"][f"{implementation}_target"],
            )
            for implementation in ("custom", "vllm")
        },
    }


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
        cross_choice = cross_choice_diagnostics(
            custom_top_k, vllm_top_k, custom_tokens[position], vllm_tokens[position]
        )
        return {
            "position": position,
            "custom_token": custom_tokens[position],
            "vllm_token": vllm_tokens[position],
            "custom_margin": step_margin(custom_top_k),
            "vllm_margin": step_margin(vllm_top_k),
            "top_k_overlap": comparison["top_k_overlap"],
            "intersection_size": comparison["intersection_size"],
            "intersection_max_abs_diff": comparison["intersection_max_abs_diff"],
            "intersection_mean_abs_diff": comparison["intersection_mean_abs_diff"],
            "intersection_cosine_similarity": comparison["intersection_cosine_similarity"],
            "custom_token_in_vllm_top_k": custom_tokens[position] in dict(vllm_top_k),
            "vllm_token_in_custom_top_k": vllm_tokens[position] in dict(custom_top_k),
            "cross_choice": cross_choice,
            "custom_top_k": custom_top_k,
            "vllm_top_k": vllm_top_k,
        }
    return None


def _is_finite(top_k: list[tuple[int, float]]) -> bool:
    return all(math.isfinite(value) for _, value in top_k)


def classify_request(
    custom_result: dict[str, Any],
    vllm_result: dict[str, Any],
    epsilon: float | None,
    concurrency: int = 2,
    batch_vs_solo_drift: dict[str, Any] | None = None,
    identity_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per CORRECTNESS_PROTOCOL_V2.md requirements 1, 3 and 5. `epsilon` is
    None during calibration (classification is deferred until it's
    proposed); a committed float during holdout evaluation. `concurrency`
    defaults to 2 (i.e. "tolerance may apply") for standalone/test callers
    that don't track it; real callers must pass the request's actual
    concurrency so a concurrency-1 disagreement is never tolerated.

    `batch_vs_solo_drift` and `identity_check` are optional: requirement 5
    lists batch-vs-solo logit drift and cross-request identity/KV corruption
    as hard-failure conditions, but neither was previously computed by any
    caller, so the already-executed sealed holdout never checked for them.
    They default to None (not checked) so this function's behavior is
    unchanged for any already-collected result that doesn't have this data;
    passing them in is how a new (V3) run actually enforces requirement 5's
    full list instead of just the subset checked so far."""
    reasons: list[str] = []
    if batch_vs_solo_drift is not None and batch_vs_solo_drift.get("drift_detected"):
        reasons.append("batch_vs_solo_drift")
    if identity_check is not None and identity_check.get("contamination_suspected"):
        reasons.append("cross_request_identity_or_kv_corruption")
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
    batch_ids = config["batch_ids"]  # exactly one batch: list[list[int]]

    # One batch per process, deliberately: an earlier version ran multiple
    # engine constructions in a loop inside one process (like
    # sentinel_diagnostics.py's original self-consistency-check bug) and hit
    # a real CUDA OOM after a couple of iterations -- PyTorch's allocator did
    # not return the prior engine's KV cache and weights before the next
    # engine tried to allocate. Fresh subprocess per batch sidesteps that
    # entirely, matching the pattern already used elsewhere in this repo.
    if implementation == "custom":
        result = asyncio.run(run_custom_full_batch(model_dir, engine_options, batch_ids))
    elif implementation == "vllm":
        result = run_vllm_full_batch(model_dir, model, engine_options, batch_ids)
    else:
        raise SystemExit(f"unknown implementation: {implementation}")

    output_path.write_text(json.dumps({"config": config, "result": result}, indent=2))


if __name__ == "__main__":
    main()
