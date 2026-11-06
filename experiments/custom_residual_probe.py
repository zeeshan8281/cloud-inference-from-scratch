"""Exact exposed-input custom batch/solo trace. Diagnostic, never gate evidence.

Usage: custom_residual_probe.py CONFIG_JSON OUTPUT_JSON. Config requires
model_dir, model, kv_bytes, batch_ids, target_index; optional cases list defaults
to batch/solo/batch_repeat/solo_repeat. Only resource_normalized and two output
tokens. Tensor rows use exact contiguous dtype bytes, base64 encoded. Sibling
CASE.operators.jsonl files must be retained with OUTPUT_JSON.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def phase_plan(config):
    """Keep legacy resource/two-token defaults; warm diagnostic is explicit."""
    warm = config.get("warm_prefix_protocol", False)
    if warm:
        if config.get("candidate") not in ("fp32_qkv_and_head", "fixed_tile_qkv_fp32_head") or config.get("output_tokens", 128) != 128:
            raise ValueError("warm diagnostic requires explicit candidate and full128 windows")
        if type(config.get("graph_enabled")) is not bool or type(config.get("observe")) is not bool:
            raise ValueError("explicit graph/observation flags required")
        if config["graph_enabled"] and config["observe"]:
            raise ValueError("graph replay does not execute operator hooks; observe eager only")
        cases = config.get("cases", ["batch", "solo"])
        if cases != ["batch", "solo"]:
            raise ValueError("warm diagnostic requires batch then target solo")
        return [(case, f"{case}-{phase}", phase) for case in cases for phase in ("cold", "warm")]
    cases = config.get("cases", ["batch_control", "solo_control", "batch", "solo", "batch_repeat", "solo_repeat"])
    if not cases or len(set(cases)) != len(cases) or set(cases) - {"batch_control", "solo_control", "batch", "solo", "batch_repeat", "solo_repeat"}:
        raise ValueError("invalid diagnostic cases")
    return [(case,case,None) for case in cases]


async def run(config, output):
    import torch

    import cloud_engine.model as model_module
    from cloud_engine.precision import precision_status
    from experiments.fix_holdout import (
        capture_custom,
        reset_prefixes,
        validate_capture,
        validate_graph_evidence,
    )
    from experiments.sentinel_pilot import _build_custom_engine, resolve_kv_plan

    warm_protocol = config.get("warm_prefix_protocol", False)
    mode = "complete_system" if warm_protocol else "resource_normalized"
    steps = 128 if warm_protocol else 2
    plan = phase_plan(config)
    batch = config["batch_ids"]
    target = config["target_index"]
    if not batch or not 0 <= target < len(batch) or config.get("mode", mode) != mode:
        raise ValueError("nonempty saved batch, valid target and resource_normalized mode required")
    if any(not ids or any(type(token) is not int or token < 0 for token in ids) for ids in batch):
        raise ValueError("invalid saved input token IDs")
    paths = {case: output.with_name(output.stem + "." + case + ".operators.jsonl") for _,case,_ in plan}
    if output.exists() or any(path.exists() for path in paths.values()):
        raise ValueError("refusing to overwrite diagnostic evidence")
    kv = None if warm_protocol else resolve_kv_plan(config["model_dir"], 16, config["kv_bytes"])
    _, engine = _build_custom_engine(config["model_dir"], mode, kv)
    candidate_status = None
    if warm_protocol:
        from cloud_engine import engine as engine_module
        from cloud_engine.precision import configure_inference_precision
        from experiments.qkv_policy_probe import install_custom, policy_state
        configure_inference_precision()
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        policy_state()
        engine.config = replace(engine.config, cuda_graph_decode=config["graph_enabled"])
        if engine.config.prefix_cache_max_blocks <= 0:
            raise ValueError("warm-prefix diagnostic requires prefix cache enabled")
        original_loader = engine_module.load_model
        def candidate_loader(*args, **kwargs):
            nonlocal candidate_status
            loaded = original_loader(*args, **kwargs)
            candidate_status = install_custom(loaded[0], config["candidate"])
            return loaded
        engine_module.load_model = candidate_loader
        try:
            await engine.start()
        finally:
            engine_module.load_model = original_loader
    else:
        await engine.start()
    runner = engine.runner
    result = {"schema": "custom-residual-probe-v1", "config": config, "batch_sha256": digest(batch),
              "cases": {}, "complete": False, "error": None,
              "scope": "target last packed row at every forward, not globally earliest divergent token; hooks synchronize and may alter scheduling"}
    if warm_protocol:
        result["schema"] = "warm-prefix-probe-v1"
        result["policy"] = {"mode": mode, "graph_enabled": engine.config.cuda_graph_decode,
                            "prefix_cache_max_blocks": engine.config.prefix_cache_max_blocks,
                            "output_tokens": steps, "observed_generated_positions": 2}
    source_root = Path(__file__).resolve().parent.parent
    engine_root = Path(model_module.__file__).parent
    sources = [Path(__file__), source_root / "experiments/fix_holdout.py", source_root / "experiments/sentinel_pilot.py", *engine_root.rglob("*.py")]
    if warm_protocol:
        sources += [source_root / "experiments" / name for name in ("warm_prefix_probe.py", "qkv_policy_probe.py")]
    result["source_hashes"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    active = None
    current_layer = None
    trace = None
    calls = 0
    schedules = []
    logits_calls = []
    selected_offsets = []
    selected_hidden = None
    original_pack, original_rope = runner._pack, model_module.apply_rope
    backend = runner.model.attn_backend
    original_attend = backend.attend
    handles = []
    current_label = None

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, allow_nan=False))
        temporary.replace(output)

    def tensors(value, path="output"):
        if isinstance(value, torch.Tensor):
            yield path, value
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                yield from tensors(item, f"{path}[{index}]")

    def observe(name, value):
        nonlocal calls, selected_hidden
        if active is None or trace is None:
            return
        if name == "norm":
            selected_hidden = value.index_select(0, torch.tensor(selected_offsets, dtype=torch.long, device=value.device)).detach()
        vectors = []
        for path, tensor in tensors(value):
            if not tensor.ndim or tensor.shape[0] != active["packed_tokens"]:
                continue
            row = tensor[active["target_offset"]].detach().contiguous().cpu()
            raw = row.view(torch.uint8).numpy().tobytes()
            vectors.append({"path": path, "shape": list(tensor.shape), "row_shape": list(row.shape),
                            "dtype": str(row.dtype), "finite": bool(torch.isfinite(row).all()),
                            "sha256": hashlib.sha256(raw).hexdigest(), "bytes_base64": base64.b64encode(raw).decode()})
        if vectors:
            calls += 1
            trace.write(json.dumps({"call": calls, "operator": name, **active, "vectors": vectors}, allow_nan=False)+"\n")
            trace.flush()

    def pack(plan):
        nonlocal active, selected_offsets, selected_hidden
        packed = original_pack(plan)
        active = None
        selected_hidden = None
        selected_offsets = packed.logit_indices.tolist()
        row = {"forward": len(schedules), "request_ids": list(packed.context.request_ids),
               "sampled_request_ids": list(packed.sampled_request_ids), "query_ranges": list(packed.context.query_ranges),
               "items": [{"request_id": item.request.request_id, "label": item.request.prompt,
                          "start": item.start_pos, "end": item.end_pos, "sample": item.sample,
                          "token_ids": list(item.token_ids), "output_tokens_before": list(item.request.generated_token_ids)} for item in plan.items]}
        schedules.append(row)
        for item, (begin, end) in zip(plan.items, packed.context.query_ranges, strict=True):
            if item.request.prompt != current_label:
                continue
            if item.request.prompt_token_ids != batch[target] or end <= begin:
                raise ValueError("target identity or packed range mismatch")
            expected = (item.request.prompt_token_ids + item.request.generated_token_ids)[item.end_pos-1]
            actual = int(packed.input_ids[end-1])
            if actual != expected:
                raise ValueError("target packed token mismatch")
            if warm_protocol and len(item.request.generated_token_ids) >= 2:
                continue
            active = {"forward": row["forward"], "packed_tokens": int(packed.input_ids.numel()),
                      "request_id": item.request.request_id, "target_offset": end-1, "target_position": item.end_pos-1, "target_input_token": actual,
                      "query_start": item.start_pos, "query_end": item.end_pos,
                      "sample": item.sample, "output_tokens_before": list(item.request.generated_token_ids),
                      "sampled_index": list(packed.sampled_request_ids).index(item.request.request_id) if item.sample else None}
        return packed

    def logits_hook(module, args, value):
        if active is None or active["sampled_index"] is None or trace is None:
            return
        row = value[active["sampled_index"]].detach().float()
        def stats(logits):
            return {"argmax": int(logits.argmax()), "all_finite": bool(torch.isfinite(logits).all()),
                    "candidate_logits": {str(token): float(logits[token]) for token in (2182, 3015, 279)}}
        entry = {**active, "logits_shape": list(value.shape), **stats(row)}
        if selected_hidden is None or len(selected_hidden) != len(value):
            raise ValueError("cannot align frozen final hidden rows")
        if warm_protocol:
            entry["hidden_sha256"] = hashlib.sha256(selected_hidden[active["sampled_index"]].contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
            logits_calls.append(entry)
            return
        weight = runner.model.embed_tokens.weight if runner.model.lm_head is None else runner.model.lm_head.weight
        weight_fp32 = weight.detach().float().t()
        frozen = selected_hidden[active["sampled_index"]].float()
        entry["hidden_sha256"] = hashlib.sha256(frozen.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        native_replay = torch.mm(selected_hidden.float(), weight_fp32)[active["sampled_index"]]
        entry["native_layout_replay"] = {**stats(native_replay), "exact_full_vocab": bool(torch.equal(row, native_replay)),
                                         "max_abs_difference": float((row-native_replay).abs().max())}
        entry["frozen_hidden_replays"] = []
        for rows, offset in ((1, 0), (8, 0), (8, 2)):
            repeated = frozen.unsqueeze(0).repeat(rows, 1)
            replay = torch.mm(repeated, weight_fp32)[offset]
            entry["frozen_hidden_replays"].append({"rows": rows, "target_offset": offset, **stats(replay),
                "exact_native_full_vocab": bool(torch.equal(row, replay)), "max_abs_native_difference": float((row-replay).abs().max())})
        entry["replay_scope"] = "fixed target hidden and FP32 weights; all other rows repeat target; shape and row offset varied separately"
        logits_calls.append(entry)


    def enter(name, inputs):
        nonlocal current_layer
        if name.startswith("layers."):
            current_layer = int(name.split(".")[1])
        observe(name + ".input", inputs)

    def rope(*args, **kwargs):
        value = original_rope(*args, **kwargs)
        observe(f"layers.{current_layer}.rope", value)
        return value

    def attend(layer_index, ctx, q, k, v):
        nonlocal calls
        value = original_attend(layer_index, ctx, q, k, v)
        observe(f"layers.{layer_index}.attention", value)
        if active is not None and trace is not None:
            view = backend.cache.view(active["request_id"], layer_index, length=active["query_end"])
            hashes = {}
            for name, tensor in (("keys", view.keys), ("values", view.values)):
                raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                hashes[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": hashlib.sha256(raw).hexdigest()}
            calls += 1
            trace.write(json.dumps({"call": calls, "operator": f"layers.{layer_index}.logical_kv", **active,
                                    "logical_kv": hashes, "scope": "all target logical positions through current query end after append"})+"\n")
            trace.flush()
        return value

    try:
        if precision_status()["allow_fp16_reduced_precision_reduction"] is not False:
            raise ValueError("actual custom precision policy missing")
        result["precision"] = {**precision_status(), "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                               "float32_matmul_precision": torch.get_float32_matmul_precision()}
        observing = config["observe"] if warm_protocol else True
        runner._pack = pack
        if observing:
            model_module.apply_rope, backend.attend = rope, attend
            handles.append(runner.model.register_forward_hook(logits_hook))
            for name, module in runner.model.named_modules():
                if name == "embed_tokens" or name == "norm" or name.startswith("layers.") and (
                    name.endswith(("layernorm", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ".mlp"))):
                    handles.append(module.register_forward_hook(lambda m, args, value, name=name: observe(name, value)))
                    if name == "norm" or name.endswith(("input_layernorm", "post_attention_layernorm", "down_proj")):
                        handles.append(module.register_forward_pre_hook(lambda m, args, name=name: enter(name, args)))
        for selection, case, phase in plan:
            selected = batch if selection.startswith("batch") else [batch[target]]
            local_index = target if selection.startswith("batch") else 0
            current_label = f"{case}-{local_index}"
            schedules, calls, logits_calls = [], 0, []
            if phase != "warm":
                reset_prefixes(engine, "custom")
            prefix_before = runner.cache.stats().as_metrics() if warm_protocol else None
            with paths[case].open("x") as trace_handle:
                enabled = observing and not selection.endswith("_control")
                trace = trace_handle if enabled else None
                capture = await capture_custom(engine, selected, steps, case)
            trace = None
            record = {"capture": capture, "schedules": schedules, "target_local_index": local_index,
                      "trace_path": str(paths[case]), "trace_calls": calls, "logits_calls": logits_calls,
                      "operator_observation_enabled": enabled,
                      "trace_sha256": hashlib.sha256(paths[case].read_bytes()).hexdigest()}
            result["cases"][case] = record
            if warm_protocol:
                record.update(phase=phase, selection=selection, prefix_counters_before=prefix_before,
                              prefix_counters_after=runner.cache.stats().as_metrics(),
                              candidate_policy=candidate_status())
                record["target_initial_computed_tokens"] = capture["records"][local_index]["initial_computed_tokens"]
            save()
            validate_capture(capture, selected, steps, cold=phase != "warm")
            if warm_protocol:
                status = record["candidate_policy"]
                if not status["wrapper_active"] or status["module_count"] != status["expected_module_count"] or not all(status["calls"].values()):
                    raise ValueError("candidate did not execute expected QKV modules")
                if config["candidate"] == "fixed_tile_qkv_fp32_head":
                    if status["candidate"] != config["candidate"] or status["fixed_tile_policy"].get("operator_active") is not True:
                        raise ValueError("fixed-tile operator was not installed")
                if phase == "warm" and record["target_initial_computed_tokens"] <= 0:
                    raise ValueError("warm target did not reuse cached prefix")
                validate_graph_evidence(capture["graph_counters"], "complete_system" if config["graph_enabled"] else "resource_normalized")
        result["target_outputs"] = {case: value["capture"]["records"][value["target_local_index"]]["output_tokens"] for case, value in result["cases"].items()}
        if "batch" in result["target_outputs"] and "solo" in result["target_outputs"]:
            left, right = result["target_outputs"]["batch"], result["target_outputs"]["solo"]
            result["first_token_difference"] = next(({"position": i, "batch": a, "solo": b, "shared_prefix": left[:i]} for i, (a,b) in enumerate(zip(left,right,strict=True)) if a != b), None)
        if warm_protocol:
            result["target_comparisons"] = {}
            for phase in ("cold", "warm"):
                left, right = result["target_outputs"][f"batch-{phase}"], result["target_outputs"][f"solo-{phase}"]
                result["target_comparisons"][phase] = {"exact_tokens": left == right,
                    "first_difference": next(({"position": i, "batch": a, "solo": b, "shared_prefix": left[:i]}
                        for i,(a,b) in enumerate(zip(left,right,strict=True)) if a != b), None)}
        result["complete"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        runner._pack, model_module.apply_rope, backend.attend = original_pack, original_rope, original_attend
        for handle in handles:
            handle.remove()
        save()
        await engine.close()
    return result


if __name__ == "__main__":
    config_path, output_path = map(Path, sys.argv[1:])
    result = asyncio.run(run(json.loads(config_path.read_text()), output_path))
    if not result["complete"]:
        raise SystemExit("diagnostic incomplete; retained evidence")
