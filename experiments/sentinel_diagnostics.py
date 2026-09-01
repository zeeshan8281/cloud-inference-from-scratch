"""Bounded root-cause diagnostic for the sentinel pilot's concurrency > 1
token-ID divergence between the custom engine and vLLM.

NOT part of the 10-pair protocol: does not resume performance measurement and
does not relax the correctness gate. Reproduces the exact failing request
(experiments/sentinel-pilot/raw/resource-normalized/pair-01.json,
in512-out128-c8, request_index 1) at concurrency 1/2/8/32 for the custom
engine, vLLM, and a plain Hugging Face Transformers reference, captures
per-step logit detail, and classifies the result per a fixed decision rule.

See experiments/sentinel-pilot/summaries/divergence-analysis.md for the
narrative writeup and reproduction command.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from experiments.controlled import Cell
from experiments.sentinel_pilot import materialize_cell_workload, prompt_seed, sentinel_token_ids

TOP_K = 20
STEPS = 3
TARGET_CELL = Cell(512, 128, 8)
TARGET_REQUEST_INDEX = 1  # the request that actually mismatched in pair 1
NUM_EXTRA_FILLERS = 24  # + the 7 non-target real c8 requests = 31 fillers for c32


# --------------------------------------------------------------------------
# Deterministic batch construction (host-side, no GPU needed)
# --------------------------------------------------------------------------


def build_diagnostic_prompts(tokenizer: Any) -> dict[str, Any]:
    """Reconstructs the exact real c8 cell (same call the pilot itself made)
    so the target request's content is byte-identical to what actually
    diverged, plus extra same-shape fillers to reach concurrency 32."""
    real_cell = materialize_cell_workload(
        "resource_normalized", 1, TARGET_CELL, "unique", tokenizer
    )
    target = real_cell[TARGET_REQUEST_INDEX]
    real_fillers = [r for r in real_cell if r["request_index"] != TARGET_REQUEST_INDEX]
    extra_fillers = []
    for index in range(NUM_EXTRA_FILLERS):
        seed = prompt_seed("diagnostic", 1, "in512-out128-extra-filler", index, "reproduce")
        extra_fillers.append(
            {
                "request_index": f"extra-{index}",
                "input_token_ids": sentinel_token_ids(seed, TARGET_CELL.input_tokens, tokenizer),
            }
        )
    return {"target": target, "fillers": real_fillers + extra_fillers}


def batch_for_concurrency(prompts: dict[str, Any], concurrency: int) -> list[list[int]]:
    """Target always at position 0, for a stable, simple row identity across
    every concurrency level. Order-of-submission is itself a candidate cause
    of batch-composition-dependent divergence (see natural_order_c8_batch),
    so this alone is not sufficient to rule out the original failing case."""
    fillers = prompts["fillers"][: concurrency - 1]
    return [prompts["target"]["input_token_ids"]] + [f["input_token_ids"] for f in fillers]


def natural_order_c8_batch(prompts: dict[str, Any]) -> tuple[list[list[int]], int]:
    """The exact 8-request c8 cell in its original request_index submission
    order (target at its natural position, TARGET_REQUEST_INDEX), not
    reordered to put the target first. Returns (batch_ids, target_index)."""
    real_fillers_by_index = {
        f["request_index"]: f["input_token_ids"] for f in prompts["fillers"] if isinstance(f["request_index"], int)
    }
    batch = []
    for index in range(TARGET_CELL.concurrency):
        if index == TARGET_REQUEST_INDEX:
            batch.append(prompts["target"]["input_token_ids"])
        else:
            batch.append(real_fillers_by_index[index])
    return batch, TARGET_REQUEST_INDEX


# --------------------------------------------------------------------------
# Metrics and classification (pure; unit tested without a GPU)
# --------------------------------------------------------------------------


def logprobs_from_logits(logits: list[float]) -> list[float]:
    maximum = max(logits)
    shifted = [value - maximum for value in logits]
    denom = math.log(sum(math.exp(value) for value in shifted))
    return [value - denom for value in shifted]


def top_k_from_logits(logits: list[float], k: int = TOP_K) -> list[tuple[int, float]]:
    logprobs = logprobs_from_logits(logits)
    ranked = sorted(range(len(logprobs)), key=lambda i: logprobs[i], reverse=True)[:k]
    return [(index, logprobs[index]) for index in ranked]


def step_margin(top_k: list[tuple[int, float]]) -> float | None:
    if len(top_k) < 2:
        return None
    return top_k[0][1] - top_k[1][1]


def compare_top_k(a: list[tuple[int, float]], b: list[tuple[int, float]]) -> dict[str, Any]:
    """Metrics over the union of two top-k lists' token IDs. Neither engine's
    public API exposes the full ~150k-wide vocabulary logit vector in all
    three cases (vLLM's SamplingParams only returns its own top-k), so every
    cross-engine comparison in this diagnostic uses this restricted support,
    applied uniformly for comparability."""
    a_map = dict(a)
    b_map = dict(b)
    union = sorted(set(a_map) | set(b_map))
    floor_a = (min(a_map.values()) - 10) if a_map else -100.0
    floor_b = (min(b_map.values()) - 10) if b_map else -100.0
    va = [a_map.get(t, floor_a) for t in union]
    vb = [b_map.get(t, floor_b) for t in union]
    diffs = [abs(x - y) for x, y in zip(va, vb, strict=True)]
    dot = sum(x * y for x, y in zip(va, vb, strict=True))
    norm_a = math.sqrt(sum(x * x for x in va))
    norm_b = math.sqrt(sum(y * y for y in vb))
    cosine = dot / (norm_a * norm_b) if norm_a and norm_b else None
    overlap = len(set(a_map) & set(b_map)) / TOP_K
    return {
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
        "cosine_similarity": cosine,
        "top_k_overlap": overlap,
    }


def classify_divergence(
    custom_by_concurrency: dict[int, list[int]],
    vllm_by_concurrency: dict[int, list[int]],
    hf_c1: list[int] | None,
) -> dict[str, Any]:
    """Applies the preregistered decision rule to the target request's
    generated tokens (first STEPS positions) at each concurrency level."""
    custom_c1 = custom_by_concurrency.get(1)
    vllm_c1 = vllm_by_concurrency.get(1)
    custom_matches_hf_at_c1 = hf_c1 is not None and custom_c1 == hf_c1
    vllm_matches_hf_at_c1 = hf_c1 is not None and vllm_c1 == hf_c1
    custom_batch_invariant = all(
        value == custom_c1 for value in custom_by_concurrency.values() if value is not None
    )
    vllm_batch_invariant = all(
        value == vllm_c1 for value in vllm_by_concurrency.values() if value is not None
    )

    if hf_c1 is not None and not custom_matches_hf_at_c1 and vllm_matches_hf_at_c1:
        verdict = "custom_engine_correctness_bug"
    elif hf_c1 is not None and custom_matches_hf_at_c1 and not vllm_matches_hf_at_c1:
        verdict = "vllm_backend_or_configuration_effect"
    elif not custom_batch_invariant or not vllm_batch_invariant:
        verdict = "numerical_equivalence_issue_batch_dependent"
    else:
        verdict = "no_divergence_observed_in_this_run"

    return {
        "verdict": verdict,
        "custom_matches_hf_at_c1": custom_matches_hf_at_c1,
        "vllm_matches_hf_at_c1": vllm_matches_hf_at_c1,
        "custom_batch_invariant": custom_batch_invariant,
        "vllm_batch_invariant": vllm_batch_invariant,
    }


# --------------------------------------------------------------------------
# Engine runners (GPU required; imports deferred)
# --------------------------------------------------------------------------


async def run_custom_diagnostic(
    model_dir: str, engine_options: dict[str, Any], batch_ids: list[list[int]], target_index: int = 0
) -> dict:
    """Runs `batch_ids` through the custom engine and captures the request at
    `target_index`'s per-step top-k via a non-invasive, self-verifying
    observer hook -- no reimplementation of RaggedRunner's internals.

    Submission uses asyncio.gather over one coroutine per request, matching
    experiments/controlled.py's _run_custom_phase exactly (not a sequential
    awaited loop): submission concurrency/ordering is itself a candidate
    variable, not just batch content.

    Hook: wrap RaggedRunner._pack (records which request IDs are in each
    batched forward call, in order) and BOTH of the two logit-producing call
    sites in execute_batch -- the model's own forward (eager path) and
    _graph_forward (CUDA-graph path, called on both capture and replay,
    unlike model.forward which is skipped once a graph is captured). Exactly
    one of the two fires per execute_batch call, so the captured-logits list
    stays index-aligned with the captured-packs list regardless of whether
    graphs are enabled. Correctness is self-verified: the captured logits'
    argmax must equal the token the engine actually emitted for every step,
    or the run is marked unverified rather than trusted.
    """
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
        # Match the resource_normalized sentinel-pilot mode exactly, so this
        # diagnostic's "base" condition is the same configuration that
        # actually showed the mismatch, not the engine's separate defaults.
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
            f"diag-{index}", ids,
            GenerationConfig(max_output_tokens=STEPS, temperature=0, eos_token_id=None),
        )

    try:
        requests = await asyncio.gather(
            *(submit_one(index, ids) for index, ids in enumerate(batch_ids))
        )
        handles = [RequestHandle(request, engine) for request in requests]
        await asyncio.gather(*(handle.wait() for handle in handles))
    finally:
        runner._pack = original_pack
        runner.model.forward = original_forward
        runner._graph_forward = original_graph_forward
        await engine.close()

    target_id = f"diag-{target_index}"
    target_tokens = list(requests[target_index].generated_token_ids)[:STEPS]
    per_step_top_k = []
    for pack, logits in zip(captured_packs, captured_logits, strict=True):
        if target_id in pack.sampled_request_ids:
            row = pack.sampled_request_ids.index(target_id)
            per_step_top_k.append(top_k_from_logits(logits[row].tolist()))

    verified = len(per_step_top_k) >= len(target_tokens) and all(
        per_step_top_k[i][0][0] == target_tokens[i] for i in range(len(target_tokens))
    )
    return {
        "output_tokens": target_tokens,
        "top_k": per_step_top_k[: len(target_tokens)],
        "verified": verified,
        "debug": {
            "pack_calls": len(captured_packs),
            "logit_calls": len(captured_logits),
            "sampled_request_ids_seen": [list(pack.sampled_request_ids) for pack in captured_packs],
        },
    }


def run_vllm_diagnostic(
    model_dir: str,
    model: dict[str, Any],
    engine_options: dict[str, Any],
    batch_ids: list[list[int]],
    target_index: int = 0,
) -> dict:
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
        temperature=0, top_p=1, top_k=-1, max_tokens=STEPS, ignore_eos=True, logprobs=TOP_K
    )
    try:
        for index, ids in enumerate(batch_ids):
            engine.add_request(f"diag-{index}", {"prompt_token_ids": ids}, params)
        finished: dict[str, Any] = {}
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    finished[output.request_id] = output.outputs[0]
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            shutdown()

    target = finished[f"diag-{target_index}"]
    token_ids = list(target.token_ids)[:STEPS]
    per_step_top_k = []
    for step_logprobs in (target.logprobs or [])[:STEPS]:
        ranked = sorted(step_logprobs.items(), key=lambda item: item[1].logprob, reverse=True)
        per_step_top_k.append([(token_id, entry.logprob) for token_id, entry in ranked[:TOP_K]])
    return {"output_tokens": token_ids, "top_k": per_step_top_k, "verified": True}


def run_hf_diagnostic(model_dir: str, dtype: str, batch_ids: list[list[int]]) -> dict:
    """HF Transformers reference. Only ever run at concurrency 1 in this
    diagnostic (no padding needed, so nothing about HF's own batching can
    confound the result) -- it exists purely as a third, independently
    implemented oracle for what "the model" computes, not a fourth engine
    under the same batching pressure as custom/vLLM."""
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype).to("cuda").eval()
    if len(batch_ids) != 1:
        raise ValueError("run_hf_diagnostic only supports concurrency 1 by design")
    input_ids = torch.tensor([batch_ids[0]], device="cuda")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=STEPS,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
    token_ids = output.sequences[0, input_ids.shape[1]:].tolist()
    per_step_top_k = [top_k_from_logits(step_logits[0].float().tolist()) for step_logits in output.logits]
    return {"output_tokens": token_ids, "top_k": per_step_top_k, "verified": True}


def main() -> None:
    import asyncio

    if len(sys.argv) != 3:
        raise SystemExit("usage: sentinel_diagnostics.py <config.json> <output.json>")
    config = json.loads(Path(sys.argv[1]).read_text())
    output_path = Path(sys.argv[2])

    pinned = json.loads(Path("/root/engine_config.json").read_text())
    model = pinned["ragged_model"]
    model_dir = config["model_dir"]
    batch_ids = config["batch_ids"]
    implementation = config["implementation"]
    target_index = config.get("target_index", 0)

    if implementation == "custom":
        result = asyncio.run(
            run_custom_diagnostic(model_dir, config.get("engine_options", {}), batch_ids, target_index)
        )
    elif implementation == "vllm":
        result = run_vllm_diagnostic(model_dir, model, config.get("engine_options", {}), batch_ids, target_index)
    elif implementation == "hf":
        result = run_hf_diagnostic(model_dir, config.get("dtype", "float16"), batch_ids)
    else:
        raise SystemExit(f"unknown implementation: {implementation}")

    output_path.write_text(json.dumps({"config": config, "result": result}, indent=2))


if __name__ == "__main__":
    main()
