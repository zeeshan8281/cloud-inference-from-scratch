"""Strict fixed-policy calibration/holdout captures; no performance claims."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from experiments.protocol_v2 import V2_BATCHES_PER_CELL, V2_CELLS, build_split_batches
from experiments.sentinel_diagnostics import TOP_K, compare_top_k
from experiments.sentinel_pilot import SENTINEL_CELLS, prompt_seed, sentinel_token_ids

NAMESPACE = "numerical-fix-20261106-holdout-v1"
MODES = ("resource_normalized", "complete_system")
MIN_OVERLAP = 0.3


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def computational_source(root: Path, *, deployed: bool = False) -> dict:
    """Hash computation/config bytes; documentation and frozen evidence are excluded."""
    engine = root / ("cloud_engine" if deployed else "src/cloud_engine")
    paths = {"src/cloud_engine/" + path.relative_to(engine).as_posix(): path
             for path in engine.rglob("*.py")}
    for name in ("engine_config.json", "modal_app.py", "pyproject.toml", "uv.lock", "experiments/controlled.py", "experiments/sentinel_pilot.py",
                 "experiments/sentinel_diagnostics.py", "experiments/sentinel_runtime.py", "experiments/protocol_v2.py",
                 "experiments/fix_holdout.py", "experiments/fix_holdout_worker.py", "experiments/parallel_validation.py"):
        paths[name] = root / name
    paths["fix_holdout_app.py"] = root / ("fix_holdout_app_source.py" if deployed else "fix_holdout_app.py")
    if not any(name.startswith("src/cloud_engine/") for name in paths):
        raise ValueError("missing computational engine sources")
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(paths.items())}
    return {"files": hashes, "sha256": digest(hashes)}


def graph_snapshot(engine, implementation: str):
    try:
        if implementation == "custom":
            return {"cuda_graph_captures": engine.runner.cuda_graph_captures,
                    "cuda_graph_replays": engine.runner.cuda_graph_replays}
        return engine.collective_rpc("graph_status")
    except Exception as exc:
        return {"snapshot_error": f"{type(exc).__name__}: {exc}"}


def graph_evidence(engine, implementation: str, before) -> dict:
    from experiments.sentinel_pilot import _graph_counters
    engine._cie_phase_graph_observation = (before, graph_snapshot(engine, implementation))
    return _graph_counters(implementation, engine, "")


def validate_graph_evidence(evidence: dict, mode: str) -> None:
    if mode not in MODES or evidence.get("unresolved") is not False:
        raise ValueError("missing actual phase graph evidence")
    if "worker_deltas" in evidence:
        before, after, deltas = (evidence.get(key, []) for key in
                                 ("workers_before", "workers_after", "worker_deltas"))
        if len(before) != 1 or len(after) != 1 or len(deltas) != 1:
            raise ValueError("missing TP1 graph observer")
        if any(row.get("observer_active") is not True for row in before + after):
            raise ValueError("inactive graph observer")
        names = ("capture_count", "replay_count")
    else:
        before, after, deltas = [evidence.get("before", {})], [evidence.get("after", {})], [evidence]
        names = ("cuda_graph_captures", "cuda_graph_replays")
    for first, last, delta in zip(before, after, deltas, strict=True):
        for name in names:
            values = (first.get(name), last.get(name), delta.get(name))
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError("invalid graph counter")
            if last[name] - first[name] != delta[name]:
                raise ValueError("inconsistent phase graph delta")
        if mode == "resource_normalized" and any(delta[name] != 0 for name in names):
            raise ValueError("resource-normalized phase used CUDA graphs")
        if mode == "complete_system" and delta[names[1]] <= 0:
            raise ValueError("complete-system phase did not replay CUDA graphs")


def validate_capture(capture: dict, ids: list[list[int]], steps: int, *, cold: bool, mode: str | None = None, implementation: str | None = None) -> None:
    # Legacy diagnostic audits omit mode; full-workload and gate paths require it.
    if mode is not None:
        from cloud_engine.precision import require_numerical_policy

        policy = capture.get("numerical_policy", {})
        for timing in ("before", "after"):
            status = policy.get(timing, {})
            require_numerical_policy(status, implementation or status.get("implementation"))
        if policy["before"]["implementation"] != policy["after"]["implementation"]:
            raise ValueError("numerical policy implementation changed within phase")
        validate_graph_evidence(capture.get("graph_counters", {}), mode)
    states = capture.get("precision", [])
    states = states if isinstance(states, list) else [states]
    if not states or any(state.get("allow_fp16_reduced_precision_reduction") is not False for state in states):
        raise ValueError("actual inference precision policy missing")
    if mode is not None and any(state != {"allow_fp16_reduced_precision_reduction": False, "allow_tf32": False,
                                          "float32_matmul_precision": "highest"} for state in states):
        raise ValueError("actual full arithmetic policy missing")
    identity = capture.get("identity", {})
    if identity.get("failures") != [] or identity.get("pack_checks", 0) <= 0:
        raise ValueError("missing or failed actual identity checks")
    records = capture.get("records", [])
    if [record.get("index") for record in records] != list(range(len(ids))):
        raise ValueError("incorrect request identity/count/order")
    if len({record.get("request_id") for record in records}) != len(ids):
        raise ValueError("duplicate request identity")
    for record, prompt in zip(records, ids, strict=True):
        validate_record(record, prompt, steps, cold=cold)


def validate_manifest(manifest: dict) -> None:
    if manifest.get("split") not in ("calibration", "holdout") or set(manifest["workloads"]) != set(MODES):
        raise ValueError("invalid manifest split/modes")
    cells = V2_CELLS if manifest["split"] == "calibration" else SENTINEL_CELLS
    expected = [(cell, index) for cell in cells for index in range(V2_BATCHES_PER_CELL[cell.concurrency])]
    for mode in MODES:
        batches = manifest["workloads"][mode]
        if len(batches) != len(expected):
            raise ValueError("incomplete workload cells")
        for batch, (cell, index) in zip(batches, expected, strict=True):
            if any(batch.get(key) != value for key, value in {
                "cell": cell.name, "batch_index": index, "input_tokens": cell.input_tokens,
                "output_tokens": cell.output_tokens, "concurrency": cell.concurrency}.items()):
                raise ValueError("incorrect workload shape")
            ids = batch["input_token_ids"]
            if len(ids) != cell.concurrency or any(len(prompt) != cell.input_tokens for prompt in ids):
                raise ValueError("incomplete input window")
            if batch.get("prompt_hashes") != [digest(prompt) for prompt in ids]:
                raise ValueError("prompt checksum mismatch")
    if manifest["split"] == "holdout":
        hashes = [value for batches in manifest["workloads"].values() for batch in batches for value in batch["prompt_hashes"]]
        if len(hashes) != 214 or len(set(hashes)) != 214 or set(hashes) & set(manifest["excluded_prompt_hashes"]):
            raise ValueError("fresh prompt manifest is duplicated or overlaps exposed prompts")


def materialize(tokenizer, split: str, mode: str, *, frozen: bool = False) -> list[dict]:
    """Fresh token IDs may only be produced after the caller verifies a frozen protocol."""
    if mode not in MODES or split not in ("calibration", "holdout"):
        raise ValueError("unknown split or mode")
    if split == "holdout" and not frozen:
        raise ValueError("fresh prompt materialization requires the frozen protocol")
    cells = V2_CELLS if split == "calibration" else SENTINEL_CELLS
    exposed = build_split_batches(tokenizer, "calibration") if split == "calibration" else None
    workloads = []
    for cell in cells:
        for batch_index in range(V2_BATCHES_PER_CELL[cell.concurrency]):
            seeds = [prompt_seed(NAMESPACE + "/" + mode, batch_index + 1, cell.name, i, "holdout")
                     for i in range(cell.concurrency)]
            batch = exposed[cell.name][batch_index] if exposed is not None else [
                sentinel_token_ids(seed, cell.input_tokens, tokenizer) for seed in seeds]
            workloads.append({"cell": cell.name, "input_tokens": cell.input_tokens,
                              "output_tokens": cell.output_tokens, "concurrency": cell.concurrency,
                              "batch_index": batch_index, "input_token_ids": batch,
                              "prompt_hashes": [digest(ids) for ids in batch],
                              "seeds": None if exposed is not None else seeds})
    return workloads


def validate_pack(plan, packed, cache, expected_prompts: dict[str, list[int]]) -> dict:
    """Check actual custom packed inputs/physical ownership, before graph padding."""
    owners, free = cache.allocator._owner, cache.allocator._free
    if (len(set(free)) != len(free) or set(free) & set(owners)
            or set(free) | set(owners) != set(range(cache.allocator.total_blocks))):
        raise ValueError("allocator free/owned partition is invalid")
    referenced = []
    for request_id, table in cache._tables.items():
        if any(owners.get(block) != request_id for block in table):
            raise ValueError("request block table references a foreign block")
        referenced.extend(table)
    for owner, table, _ in cache._prefixes.values():
        if any(owners.get(block) != owner for block in table):
            raise ValueError("retained prefix references a foreign block")
        referenced.extend(table)
    if len(referenced) != len(set(referenced)) or set(referenced) != set(owners):
        raise ValueError("aliased or unaccounted owned blocks")
    ids, positions, slots, sampled, logit_indices, ranges = [], [], [], [], [], []
    tables = packed.context.block_tables.tolist()
    offset = 0
    request_ids = [item.request.request_id for item in plan.items]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("duplicate request in actual plan")
    for row, item in enumerate(plan.items):
        request = item.request
        if request.prompt_token_ids != expected_prompts.get(request.prompt):
            raise ValueError("request prompt does not match submitted identity")
        known = request.prompt_token_ids + request.generated_token_ids
        expected = known[item.start_pos:item.end_pos]
        if list(item.token_ids) != expected:
            raise ValueError("plan reads another prompt/generated prefix")
        table = cache.block_table(request.request_id)
        if tables[row][:len(table)] != table:
            raise ValueError("device block table differs from actual request allocation")
        ids.extend(expected)
        positions.extend(range(item.start_pos, item.end_pos))
        slots.extend(table[position // cache.block_size] * cache.block_size + position % cache.block_size
                     for position in range(item.start_pos, item.end_pos))
        ranges.append((offset, offset + len(expected)))
        offset += len(expected)
        if item.sample:
            sampled.append(request.request_id)
            logit_indices.append(offset - 1)
    if (packed.input_ids.tolist() != ids or packed.context.positions.tolist() != positions
            or packed.context.slot_mapping.tolist() != slots or len(slots) != len(set(slots))
            or tuple(request_ids) != packed.context.request_ids
            or tuple(ranges) != packed.context.query_ranges
            or tuple(sampled) != packed.sampled_request_ids
            or logit_indices != packed.logit_indices.tolist()):
        raise ValueError("packed request/input/position/slot/logit attribution mismatch")
    return {"request_ids": request_ids, "sampled_request_ids": sampled,
            "query_ranges": ranges, "input_sha256": digest(ids), "slots_sha256": digest(slots),
            "block_tables_sha256": digest(tables), "owner_table_sha256": digest(owners),
            "checked_slots": len(slots), "checks_passed": True}


def validate_record(record: dict, expected_input: list[int], steps: int, *, cold: bool) -> None:
    if record.get("input_token_ids") != expected_input or record.get("input_verified") is not True:
        raise ValueError("input identity was not verified")
    tokens, observations = record.get("output_tokens", []), record.get("steps", [])
    if len(tokens) != steps or len(observations) != steps:
        raise ValueError("incomplete output or logit observation window")
    if cold and record.get("initial_computed_tokens") != 0:
        raise ValueError("cold phase reused a prefix")
    for position, (token, observation) in enumerate(zip(tokens, observations, strict=True)):
        if "output_tokens_before" in observation and observation["output_tokens_before"] != tokens[:position]:
            raise ValueError("observed worker generated prefix does not match request output history")
        top = observation["top_k"]
        if observation.get("all_finite") is not True or observation.get("argmax") not in dict(top):
            raise ValueError("nonfinite full logits or emitted token differs from observed argmax")
        if (len(top) != TOP_K or len(dict(top)) != TOP_K or token not in dict(top)
                or any(not math.isfinite(value) for _, value in top)
                or any(a[1] < b[1] for a, b in zip(top, top[1:], strict=False))
                or dict(top)[token] != top[0][1] or dict(top)[observation["argmax"]] != top[0][1]):
            raise ValueError("invalid observed top-k or selected maximum")


async def capture_custom(engine, batch: list[list[int]], steps: int, run_id: str) -> dict:
    import torch

    from cloud_engine.engine import RequestHandle
    from cloud_engine.precision import numerical_policy_snapshot, precision_status
    from cloud_engine.scheduler import GenerationConfig

    runner = engine.runner
    before_policy = numerical_policy_snapshot(engine, "custom")
    before_graphs = graph_snapshot(engine, "custom")
    expected = {f"{run_id}-{index}": ids for index, ids in enumerate(batch)}
    captured: dict[str, list] = {}
    initial = {}
    traces = []
    active_pack = None
    in_graph = False
    original_pack, original_forward, original_graph = runner._pack, runner.model.forward, runner._graph_forward

    def pack(plan):
        nonlocal active_pack
        active_pack = original_pack(plan)
        traces.append(validate_pack(plan, active_pack, runner.cache, expected))
        for item in plan.items:
            initial.setdefault(item.request.request_id, item.start_pos)
        return active_pack

    def record(logits):
        if active_pack is None:
            raise ValueError("logits without an attributed pack")
        count = len(active_pack.sampled_request_ids)
        values = logits[:count].float()
        finite = torch.isfinite(values).all(dim=-1).tolist()
        argmax = values.argmax(dim=-1).tolist() if count else []
        probabilities = torch.log_softmax(values, dim=-1)
        top_values, top_ids = probabilities.topk(TOP_K, dim=-1)
        for request_id, maximum, valid, ids, vals in zip(active_pack.sampled_request_ids, argmax, finite,
                                                        top_ids.tolist(), top_values.tolist(), strict=True):
            captured.setdefault(request_id, []).append({"argmax": maximum, "all_finite": valid,
                                                        "top_k": list(zip(ids, vals, strict=True)) if valid else []})
        return logits

    def forward(*args, **kwargs):
        logits = original_forward(*args, **kwargs)
        return logits if in_graph else record(logits)

    def graph(*args, **kwargs):
        nonlocal in_graph
        in_graph = True
        try:
            logits = original_graph(*args, **kwargs)
        finally:
            in_graph = False
        return record(logits)

    runner._pack, runner.model.forward, runner._graph_forward = pack, forward, graph
    requests, failures = [], []
    try:
        requests = await asyncio.gather(*(engine.scheduler.submit(label, ids,
            GenerationConfig(max_output_tokens=steps, temperature=0, eos_token_id=None)) for label, ids in expected.items()))
        await asyncio.wait_for(asyncio.gather(*(RequestHandle(request, engine).wait() for request in requests)), 900)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        runner._pack, runner.model.forward, runner._graph_forward = original_pack, original_forward, original_graph
    try:
        after_policy = numerical_policy_snapshot(engine, "custom")
    except Exception as exc:
        after_policy = {"error": f"{type(exc).__name__}: {exc}"}
        failures.append(after_policy["error"])
    return {"records": [{"index": index, "request_id": request.request_id,
                          "input_token_ids": list(request.prompt_token_ids), "input_verified": True,
                          "initial_computed_tokens": initial.get(request.request_id),
                          "output_tokens": list(request.generated_token_ids),
                          "steps": captured.get(request.request_id, [])} for index, request in enumerate(requests)],
            "identity": {"failures": failures, "pack_checks": len(traces), "traces": traces},
            "precision": precision_status(), "numerical_policy": {"before": before_policy, "after": after_policy},
            "graph_counters": graph_evidence(engine, "custom", before_graphs)}


def capture_vllm(engine, batch: list[list[int]], steps: int, run_id: str) -> dict:
    import time

    from vllm import SamplingParams

    from cloud_engine.precision import numerical_policy_snapshot, require_vllm_precision

    before_policy = numerical_policy_snapshot(engine, "vllm")
    expected = {f"{run_id}-{index}": ids for index, ids in enumerate(batch)}
    engine.collective_rpc("holdout_reset", args=(expected,))
    before_graphs = graph_snapshot(engine, "vllm")
    params = SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=steps, ignore_eos=True, logprobs=TOP_K)
    for request_id, ids in expected.items():
        engine.add_request(request_id, {"prompt_token_ids": ids}, params)
    completed, failures = {}, []
    started = time.monotonic()
    try:
        while engine.has_unfinished_requests():
            if time.monotonic() - started > 900:
                raise TimeoutError("vLLM correctness phase timed out")
            for output in engine.step():
                if output.finished:
                    if output.request_id in completed or len(output.outputs) != 1:
                        raise ValueError("duplicate or ambiguous vLLM request output")
                    completed[output.request_id] = output.outputs[0]
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    try:
        evidence = engine.collective_rpc("holdout_evidence")
        if len(evidence) != 1:
            raise ValueError("this protocol requires exactly one TP=1 worker")
        worker = evidence[0]
    except Exception as exc:
        worker = {"requests": {}, "failures": [f"{type(exc).__name__}: {exc}"], "pack_checks": 0}
    if set(completed) != set(expected) or set(worker["requests"]) != set(expected):
        failures.append("missing or extra worker/request evidence")
    identity = {key: value for key, value in worker.items() if key != "requests"}
    identity["failures"] = identity.get("failures", []) + failures
    try:
        precision = require_vllm_precision(engine)
    except Exception as exc:
        precision = []
        identity["failures"].append(f"{type(exc).__name__}: {exc}")
    try:
        after_policy = numerical_policy_snapshot(engine, "vllm")
    except Exception as exc:
        after_policy = {"error": f"{type(exc).__name__}: {exc}"}
        identity["failures"].append(after_policy["error"])
    return {"records": [{"index": index, "request_id": request_id, "input_token_ids": ids,
                          "output_tokens": list(completed[request_id].token_ids) if request_id in completed else [],
                          **worker["requests"].get(request_id, {})}
                         for index, (request_id, ids) in enumerate(expected.items())],
            "identity": identity, "precision": precision, "raw_worker": worker,
            "numerical_policy": {"before": before_policy, "after": after_policy},
            "graph_counters": graph_evidence(engine, "vllm", before_graphs)}


def reset_prefixes(engine, implementation: str) -> None:
    if implementation == "vllm":
        if engine.has_unfinished_requests():
            raise ValueError("cannot reset prefixes while requests are active")
        engine.reset_prefix_cache()
    else:
        cache = engine.runner.cache
        if cache._tables:
            raise ValueError("custom request blocks remain live before reset")
        for owner, blocks, _ in cache._prefixes.values():
            cache.allocator.free(blocks, owner)
        cache._prefixes.clear()
        cache.assert_invariants()


def observed_kv_capacity(engine, implementation: str) -> dict:
    """Inspect allocated custom tensors or actual vLLM worker configuration."""
    if implementation == "vllm":
        from experiments.sentinel_pilot import _vllm_kv_capacity
        return {"implementation": implementation, "worker": _vllm_kv_capacity(engine)}
    if implementation != "custom":
        raise ValueError("unknown implementation")
    cache = engine.runner.cache
    pools = (cache.key_pool, cache.value_pool)
    return {"implementation": implementation, "num_blocks": cache.num_blocks,
            "allocator_blocks": cache.allocator.total_blocks, "block_size": cache.block_size,
            "pool_shapes": [list(pool.shape) for pool in pools],
            "pool_element_sizes": [pool.element_size() for pool in pools],
            "reserved_bytes": sum(pool.numel() * pool.element_size() for pool in pools)}


def validate_kv_capacity(evidence: dict, mode: str, plan: dict | None) -> None:
    if mode not in MODES:
        raise ValueError("unknown capacity mode")
    if evidence.get("implementation") == "vllm":
        actual = evidence["worker"]
        blocks, block_size = actual.get("num_gpu_blocks"), 16
        if actual.get("tensor_parallel_size") != 1 or actual.get("block_sizes") != [16]:
            raise ValueError("unsupported actual worker KV layout")
    elif evidence.get("implementation") == "custom":
        blocks, block_size = evidence.get("num_blocks"), evidence.get("block_size")
        shapes = evidence.get("pool_shapes", [])
        if (len(shapes) != 2 or shapes[0] != shapes[1] or len(shapes[0]) != 5
                or any(type(dimension) is not int or dimension <= 0 for dimension in shapes[0])
                or shapes[0][1:3] != [blocks, block_size]
                or evidence.get("allocator_blocks") != blocks
                or evidence.get("pool_element_sizes") != [2, 2]
                or evidence.get("reserved_bytes") != 4 * math.prod(shapes[0])):
            raise ValueError("actual custom KV tensor/allocator layout mismatch")
    else:
        raise ValueError("missing actual KV implementation")
    if type(blocks) is not int or blocks <= 0 or block_size != 16:
        raise ValueError("invalid actual KV block capacity")
    if mode == "resource_normalized":
        if not plan or blocks != plan["num_blocks"] or block_size != plan["block_size"]:
            raise ValueError("requested KV capacity was not applied")
        if evidence["implementation"] == "custom" and evidence["reserved_bytes"] != plan["resolved_bytes"]:
            raise ValueError("actual custom KV bytes differ from matched plan")


async def run_workload(config: dict, checkpoint: Path) -> dict:
    from cloud_engine.precision import precision_status
    from experiments.sentinel_pilot import _build_custom_engine, _build_vllm_engine, resolve_kv_plan

    implementation, mode = config["implementation"], config["mode"]
    workload, model_dir = config["workload"], config["model_dir"]
    batch, steps = workload["input_token_ids"], workload["output_tokens"]
    kv = resolve_kv_plan(model_dir, 16, config["kv_bytes"]) if mode == "resource_normalized" else None
    engine = None
    result = {"config": config, "batch": {}, "solo": [], "complete": False, "error": None,
              "kv_plan": kv.as_dict() if kv is not None else None}

    def save():
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, allow_nan=False))
        temporary.replace(checkpoint)

    async def phases(ids, label):
        reset_prefixes(engine, implementation)
        captures = {}
        for phase in (("unique",) if mode == "resource_normalized" else ("cold", "warm")):
            capture = await capture_custom(engine, ids, steps, label + "-" + phase) if implementation == "custom" else capture_vllm(engine, ids, steps, label + "-" + phase)
            result.setdefault("partial_phases", {}).setdefault(label, {})[phase] = capture
            save()
            validate_capture(capture, ids, steps, cold=phase != "warm", mode=mode, implementation=implementation)
            captures[phase] = capture
        return captures

    try:
        if implementation == "custom":
            _, engine = _build_custom_engine(model_dir, mode, kv)
            await engine.start()
            if precision_status()["allow_fp16_reduced_precision_reduction"] is not False:
                raise ValueError("custom precision policy missing")
        elif implementation == "vllm":
            import vllm
            original_args = vllm.EngineArgs
            def holdout_args(**kwargs):
                kwargs["worker_extension_cls"] = "experiments.fix_holdout_worker.HoldoutWorker"
                return original_args(**kwargs)
            vllm.EngineArgs = holdout_args
            try:
                engine = _build_vllm_engine(model_dir, config["model"], mode, kv)
            finally:
                vllm.EngineArgs = original_args
        else:
            raise ValueError("unknown implementation")
        result["kv_capacity"] = observed_kv_capacity(engine, implementation)
        save()
        validate_kv_capacity(result["kv_capacity"], mode, result["kv_plan"])
        result["batch"] = await phases(batch, "batch")
        save()
        for index, ids in enumerate(batch):
            result["solo"].append(await phases([ids], f"solo-{index}"))
            save()
        result["complete"] = True
    except Exception as exc:  # noqa: BLE001 - incomplete evidence is retained and cannot pass
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        save()
        if engine is not None:
            if implementation == "custom":
                await engine.close()
            elif getattr(engine, "shutdown", None) is not None:
                engine.shutdown()
    return result


def compare_records(left: dict, right: dict, *, bound: float | None) -> dict:
    a, b = left["output_tokens"], right["output_tokens"]
    if not a or len(a) != len(b) or len(left["steps"]) != len(a) or len(right["steps"]) != len(b):
        raise ValueError("incomplete comparison window")
    failures, max_drift, min_overlap = [], 0.0, 1.0
    for position, (x, y) in enumerate(zip(a, b, strict=True)):
        metric = compare_top_k(left["steps"][position]["top_k"], right["steps"][position]["top_k"])
        drift = metric["intersection_max_abs_diff"]
        overlap = metric["top_k_overlap"]
        if drift is None:
            failures.append({"position": position, "reason": "missing_shared_support"})
        else:
            max_drift = max(max_drift, drift)
            if bound is not None and drift > bound:
                failures.append({"position": position, "reason": "batch_solo_logprob_drift", "drift": drift, "bound": bound})
        min_overlap = min(min_overlap, overlap)
        if overlap < MIN_OVERLAP:
            failures.append({"position": position, "reason": "low_top_k_overlap", "overlap": overlap})
        if x != y:
            failures.append({"position": position, "reason": "token_difference", "left": x, "right": y})
            break
    return {"failures": failures, "max_intersection_logprob_drift": max_drift, "min_top_k_overlap": min_overlap,
            "exact_tokens": a == b, "expected_tokens": len(a)}


def summarize(manifest: dict, captures: dict, bounds: dict | None = None, *, partial: bool = False) -> dict:
    if not partial:
        validate_manifest(manifest)
    failures, comparisons = [], []
    maxima = {mode: {engine: 0.0 for engine in ("custom", "vllm")} for mode in MODES}
    expected_keys = {f"{mode}/{engine}/{index}" for mode in MODES for engine in ("custom", "vllm")
                     for index in range(len(manifest["workloads"][mode]))}
    if set(captures) != expected_keys:
        return {"complete": False, "passes": False, "failures": ["missing_or_extra_child_captures"], "comparisons": []}
    for mode in MODES:
        for index, workload in enumerate(manifest["workloads"][mode]):
            engines = {engine: captures[f"{mode}/{engine}/{index}"] for engine in ("custom", "vllm")}
            if any(not capture.get("complete") for capture in engines.values()):
                failures.append({"mode": mode, "batch": index, "reason": "incomplete_child"})
                continue
            phases = ("unique",) if mode == "resource_normalized" else ("cold", "warm")
            try:
                for engine, capture in engines.items():
                    config = capture.get("config", {})
                    if config.get("implementation") != engine or config.get("mode") != mode or config.get("workload") != workload:
                        raise ValueError("capture configuration does not match exact workload")
                    validate_kv_capacity(capture["kv_capacity"], mode, capture["kv_plan"])
                    if mode == "resource_normalized" and capture["kv_plan"]["requested_bytes"] != config["kv_bytes"]:
                        raise ValueError("KV plan does not match requested bytes")
                    if len(capture["solo"]) != workload["concurrency"] or set(capture["batch"]) != set(phases):
                        raise ValueError("missing batch/solo phase")
                    if any(set(solo) != set(phases) for solo in capture["solo"]):
                        raise ValueError("missing or extra solo phase")
                    for phase in phases:
                        validate_capture(capture["batch"][phase], workload["input_token_ids"], workload["output_tokens"], cold=phase != "warm", mode=mode, implementation=engine)
                        for solo, ids in zip(capture["solo"], workload["input_token_ids"], strict=True):
                            validate_capture(solo[phase], [ids], workload["output_tokens"], cold=phase != "warm", mode=mode, implementation=engine)
                        if len(capture["batch"][phase]["records"]) != workload["concurrency"]:
                            raise ValueError("incomplete batched request count")
                        for request_index, ids in enumerate(workload["input_token_ids"]):
                            batch_record = capture["batch"][phase]["records"][request_index]
                            solo_record = capture["solo"][request_index][phase]["records"][0]
                            for record in (batch_record, solo_record):
                                validate_record(record, ids, workload["output_tokens"], cold=phase != "warm")
                            comparison = compare_records(batch_record, solo_record, bound=bounds[mode][engine] if bounds is not None else None)
                            maxima[mode][engine] = max(maxima[mode][engine], comparison["max_intersection_logprob_drift"])
                            comparisons.append({"kind": "batch_solo", "mode": mode, "engine": engine, "batch": index,
                                                "phase": phase, "request_index": request_index, **comparison})
                if mode == "resource_normalized" and engines["custom"]["kv_plan"] != engines["vllm"]["kv_plan"]:
                    raise ValueError("cross-engine KV plans differ")
                for phase in phases:
                    for request_index in range(workload["concurrency"]):
                        for kind in ("batch", "solo"):
                            records = [capture["batch"][phase]["records"][request_index] if kind == "batch" else
                                       capture["solo"][request_index][phase]["records"][0] for capture in engines.values()]
                            comparison = compare_records(*records, bound=None)
                            comparisons.append({"kind": "cross_engine_" + kind, "mode": mode, "batch": index,
                                                "phase": phase, "request_index": request_index, **comparison})
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                failures.append({"mode": mode, "batch": index, "reason": f"invalid_evidence: {exc}"})
    failures.extend(comparison for comparison in comparisons if comparison["failures"])
    return {"complete": not any(isinstance(entry, dict) and entry.get("reason", "").startswith(("incomplete", "invalid")) for entry in failures),
            "passes": not failures, "failures": failures, "comparisons": comparisons,
            "calibration_maxima": maxima if bounds is None else None, "gate_cleared": False}


if __name__ == "__main__":
    config_path, output_path = map(Path, sys.argv[1:])
    asyncio.run(run_workload(json.loads(config_path.read_text()), output_path))
