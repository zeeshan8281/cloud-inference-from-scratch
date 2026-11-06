"""Deterministic admission-wave replay; diagnostic only, no gate relaxation.

Run in a fresh process: admission_probe.py CONFIG_JSON OUTPUT_JSON.
Atomic admits every request before the first engine step. Staged executes
exactly one step with request diag-0 before admitting the rest.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def drive(engine, params, batch_ids, admission):
    """Preserve request IDs/order and sampling; vary only the admission boundary."""
    if admission not in ("atomic", "staged"):
        raise ValueError(f"unknown admission: {admission}")
    if not batch_ids:
        raise ValueError("empty batch")
    finished = {}
    events = []

    def step():
        events.append({"event": "step"})
        for output in engine.step():
            if output.finished:
                finished[output.request_id] = output.outputs[0]

    for index, ids in enumerate(batch_ids):
        engine.add_request(f"diag-{index}", {"prompt_token_ids": ids}, params)
        events.append({"event": "admit", "request_id": f"diag-{index}"})
        if admission == "staged" and index == 0:
            step()
    while engine.has_unfinished_requests():
        step()
    assert set(finished) == {f"diag-{i}" for i in range(len(batch_ids))}
    return finished, events


def main():
    # Background execution would confound explicit admission-wave control.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from experiments.sentinel_diagnostics import STEPS, TOP_K

    config = json.loads(Path(sys.argv[1]).read_text())
    import torch

    reduction = config.get("fp16_reduced_precision_reduction", True)
    if not isinstance(reduction, bool):
        raise ValueError("FP16 reduction policy must be boolean")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduction
    output_path = Path(sys.argv[2])
    trace = output_path.with_suffix(".jsonl")
    if output_path.exists() or trace.exists():
        raise ValueError("refusing to overwrite retained probe evidence")
    os.environ["PROJECTION_TRACE"] = str(trace)
    operator_trace = output_path.with_suffix(".operators.jsonl")
    if operator_trace.exists():
        raise ValueError("refusing to overwrite retained operator evidence")
    os.environ["OPERATOR_TRACE"] = str(operator_trace)
    os.environ["PROJECTION_TARGET_ID"] = f"diag-{config['target_index']}"
    linear_trace = output_path.with_suffix(".linear.jsonl")
    if linear_trace.exists():
        raise ValueError("refusing to overwrite retained linear evidence")
    if config.get("linear_trace"):
        os.environ["LINEAR_TRACE"] = str(linear_trace)
    batch_ids = config["batch_ids"]
    if config["admission"] not in ("atomic", "staged") or not batch_ids:
        raise ValueError("nonempty batch and atomic/staged admission required")
    if not 0 <= config["target_index"] < len(batch_ids):
        raise ValueError("invalid target index")
    args = EngineArgs(
        model=config["model_dir"], tokenizer=config["model_dir"], dtype="half",
        max_model_len=config["model"]["max_model_len"], block_size=16,
        disable_log_stats=True, trust_remote_code=False, seed=0,
        max_num_seqs=max(len(batch_ids), 1), max_num_batched_tokens=2048 * 4,
        gpu_memory_utilization=0.85, enable_prefix_caching=False, enforce_eager=True,
        worker_extension_cls=("experiments.linear_probe_hook.Observer" if config.get("linear_trace")
                              else "experiments.operator_probe_hook.Observer" if config.get("operator_trace")
                              else "experiments.projection_hook.Observer"),
    )
    engine = LLMEngine.from_engine_args(args)
    params = SamplingParams(
        temperature=0, top_p=1, top_k=-1, max_tokens=STEPS, ignore_eos=True, logprobs=TOP_K
    )
    schedules = []
    execute = GPUModelRunner.execute_model

    def record_schedule(self, scheduler_output, *args, **kwargs):
        schedules.append({
            "num_scheduled_tokens": dict(scheduler_output.num_scheduled_tokens),
            "total_num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
        })
        return execute(self, scheduler_output, *args, **kwargs)

    GPUModelRunner.execute_model = record_schedule
    try:
        finished, events = drive(engine, params, batch_ids, config["admission"])
    finally:
        GPUModelRunner.execute_model = execute
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            shutdown()

    captures = [json.loads(line) for line in trace.read_text().splitlines()]
    assert captures and len(captures) == len(schedules), "Unaligned forward evidence"
    results = []
    for index in range(len(batch_ids)):
        request_id = f"diag-{index}"
        output = finished[request_id]
        tokens = list(output.token_ids)
        observed = [row for call in captures for row in call["rows"] if row["request_id"] == request_id]
        assert len(tokens) == STEPS == len(observed), "Expected full-prefill three-token window"
        assert [row["native_argmax"] for row in observed] == tokens, "Projection/output mismatch"
        top_k = [
            [(token, entry.logprob) for token, entry in sorted(step.items(), key=lambda item: item[1].logprob, reverse=True)[:TOP_K]]
            for step in output.logprobs
        ]
        results.append({"index": index, "request_id": request_id, "output_tokens": tokens,
                        "top_k": top_k, "verified": True})
    expected_first = {"diag-0"} if config["admission"] == "staged" else {f"diag-{i}" for i in range(len(batch_ids))}
    assert set(schedules[0]["num_scheduled_tokens"]) == expected_first, "Admission boundary not realized"
    operator_calls = [json.loads(line) for line in operator_trace.read_text().splitlines()] if operator_trace.exists() else []
    if config.get("operator_trace"):
        assert operator_calls, "Operator observer did not execute"
    linear_calls = [json.loads(line) for line in linear_trace.read_text().splitlines()] if linear_trace.exists() else []
    if config.get("linear_trace"):
        assert linear_calls, "Linear observer did not execute"
    output_path.write_text(json.dumps({"config": config, "result": results[config["target_index"]],
                                      "all_results": results, "projection_calls": captures,
                                      "schedules": schedules, "admission_events": events,
                                      "operator_calls": operator_calls, "linear_calls": linear_calls,
                                      "execution": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                                                    "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}}, indent=2))


if __name__ == "__main__":
    main()
