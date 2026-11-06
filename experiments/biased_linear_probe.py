"""Frozen biased-linear ablation on exposed custom-model inputs; no model fix.

CONFIG_JSON OUTPUT_JSON uses model_dir/model/kv_bytes/batch_ids/target_index.
Default cases batch,solo (batch first supplies actual packed layout). Two output
tokens, resource-normalized policy. Replays never replace model outputs.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path


def layout_variants(batch_rows, batch_offset, local_rows, local_offset):
    if not 0 <= batch_offset < batch_rows or not 0 <= local_offset < local_rows:
        raise ValueError("invalid observed layout")
    return list(dict.fromkeys(((batch_rows, batch_offset), (batch_rows, local_offset), (local_rows, local_offset))))


def vector_metrics(actual, reference):
    if not actual or len(actual) != len(reference):
        raise ValueError("incompatible target vectors")
    differences = [abs(a-b) for a,b in zip(actual, reference, strict=True)]
    return {"changed_elements": sum(a != b for a,b in zip(actual, reference, strict=True)),
            "max_abs": max(differences), "mean_abs": sum(differences)/len(differences)}


async def run(config, output):
    import torch
    import torch.nn.functional as functional

    from cloud_engine.precision import precision_status
    from experiments.fix_holdout import capture_custom, digest, reset_prefixes, validate_capture
    from experiments.sentinel_pilot import _build_custom_engine, resolve_kv_plan

    if output.exists():
        raise ValueError("refusing to overwrite retained evidence")
    batch, target = config["batch_ids"], config["target_index"]
    if not batch or not 0 <= target < len(batch):
        raise ValueError("invalid target batch")
    cases = config.get("cases", ["batch", "solo"])
    if not cases or cases[0] != "batch" or len(set(cases)) != len(cases) or set(cases) - {"batch", "solo", "batch_repeat", "solo_repeat"}:
        raise ValueError("batch must precede solo layout replay")
    result = {"schema": "biased-linear-probe-v1", "config": config, "batch_sha256": digest(batch),
              "cases": {}, "calls": [], "complete": False, "error": None,
              "scope": "frozen target input/weights/bias; other rows repeat target; native forward outputs are never replaced"}

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, allow_nan=False))
        temporary.replace(output)

    def checksum(tensor):
        raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    engine = None
    handles = []
    original_pack = None
    active = None
    current_label = None
    current_case = None
    schedules = []
    layouts = {}
    weights = {}
    frozen_inputs = {}
    try:
        kv = resolve_kv_plan(config["model_dir"], 16, config["kv_bytes"])
        _, engine = _build_custom_engine(config["model_dir"], "resource_normalized", kv)
        await engine.start()
        runner = engine.runner
        original_pack = runner._pack
        result["precision"] = {**precision_status(), "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                               "float32_matmul_precision": torch.get_float32_matmul_precision()}
        if result["precision"]["allow_fp16_reduced_precision_reduction"] is not False:
            raise ValueError("precision policy missing")

        def pack(plan):
            nonlocal active
            packed = original_pack(plan)
            active = None
            schedules.append({"request_ids": list(packed.context.request_ids), "query_ranges": list(packed.context.query_ranges),
                              "items": [{"request_id": item.request.request_id, "label": item.request.prompt,
                                         "start": item.start_pos, "end": item.end_pos, "token_ids": list(item.token_ids),
                                         "output_tokens_before": list(item.request.generated_token_ids)} for item in plan.items]})
            for item, (begin, end) in zip(plan.items, packed.context.query_ranges, strict=True):
                if item.request.prompt != current_label:
                    continue
                if item.request.prompt_token_ids != batch[target]:
                    raise ValueError("target prompt identity mismatch")
                expected = (batch[target]+item.request.generated_token_ids)[item.end_pos-1]
                if int(packed.input_ids[end-1]) != expected:
                    raise ValueError("target packed identity mismatch")
                position = item.end_pos-1
                active = {"case": current_case, "target_position": position, "target_input_token": expected,
                          "output_tokens_before": list(item.request.generated_token_ids), "forward": len(schedules)-1,
                          "rows": int(packed.input_ids.numel()), "offset": end-1,
                          "local_rows": end-begin, "local_offset": end-begin-1}
                if current_case == "batch":
                    layouts[position] = layout_variants(active["rows"], active["offset"], active["local_rows"], active["local_offset"])
                if position not in layouts:
                    raise ValueError("solo position has no observed batch layout")
            return packed

        def before(name, args):
            if active is not None:
                frozen_inputs[name] = args[0].detach().clone()

        def after(name, module, actual):
            if active is None:
                return
            x = frozen_inputs.pop(name)
            weight, bias = module.weight.detach(), module.bias.detach()
            if x.shape[0] != active["rows"] or actual.shape[0] != active["rows"]:
                raise ValueError("projection row count mismatch")
            target_x = x[active["offset"]].clone()
            native = actual[active["offset"]].detach().float()
            native_values = native.cpu().tolist()
            replay = functional.linear(x, weight, bias)[active["offset"]].float()
            reference = torch.mv(weight.double(), target_x.double()) + bias.double()
            rounded = reference.to(actual.dtype).float().cpu().tolist()
            reference_values = reference.cpu().tolist()
            entry = {**active, "projection": name, "input_dtype": str(x.dtype), "output_dtype": str(actual.dtype),
                     "target_input_sha256": checksum(target_x), "actual_input_matrix_sha256": checksum(x),
                     "weight_sha256": weights[name]["weight"], "bias_sha256": weights[name]["bias"],
                     "native": native_values, "native_replay": replay.cpu().tolist(),
                     "native_replay_exact": bool(torch.equal(native, replay)),
                     "fp64_reference": reference_values, "fp64_rounded_fp16": rounded,
                     "native_error_vs_fp64": vector_metrics(native_values, reference_values),
                     "native_error_vs_fp64_rounded": vector_metrics(native_values, rounded), "replays": []}
            weight32, bias32 = weight.float(), bias.float()
            for rows, offset in layouts[active["target_position"]]:
                repeated = target_x.unsqueeze(0).repeat(rows, 1)
                variants = {
                    "fp16_biased_linear": lambda repeated=repeated: functional.linear(repeated, weight, bias),
                    "fp16_linear_then_bias": lambda repeated=repeated: functional.linear(repeated, weight, None) + bias,
                    "fp32_biased_linear_cast_fp16": lambda repeated=repeated: functional.linear(repeated.float(), weight32, bias32).to(actual.dtype),
                }
                for method, operation in variants.items():
                    values = operation()[offset].float().cpu().tolist()
                    entry["replays"].append({"rows": rows, "offset": offset, "method": method, "values": values,
                                             "error_vs_native": vector_metrics(values, native_values),
                                             "error_vs_fp64": vector_metrics(values, reference_values),
                                             "error_vs_fp64_rounded": vector_metrics(values, rounded)})
            if checksum(module.weight) != weights[name]["weight"] or checksum(module.bias) != weights[name]["bias"]:
                raise ValueError("projection parameter mutated during replay")
            entry["parameters_unchanged"] = True
            result["calls"].append(entry)
            save()

        runner._pack = pack
        for name in ("q_proj", "k_proj", "v_proj"):
            module = getattr(runner.model.layers[0].self_attn, name)
            if module.bias is None:
                raise ValueError("expected biased layer0 Q/K/V")
            weights[name] = {"weight": checksum(module.weight), "bias": checksum(module.bias)}
            handles.append(module.register_forward_pre_hook(lambda m, args, name=name: before(name, args)))
            handles.append(module.register_forward_hook(lambda m, args, value, name=name: after(name, m, value)))
        for case in cases:
            current_case = case
            ids = batch if case.startswith("batch") else [batch[target]]
            local_index = target if case.startswith("batch") else 0
            current_label = f"{case}-{local_index}"
            schedules = []
            reset_prefixes(engine, "custom")
            capture = await capture_custom(engine, ids, 2, case)
            result["cases"][case] = {"capture": capture, "schedules": schedules, "target_local_index": local_index}
            save()
            validate_capture(capture, ids, 2, cold=True)
        result["target_outputs"] = {case: data["capture"]["records"][data["target_local_index"]]["output_tokens"] for case,data in result["cases"].items()}
        result["complete"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for handle in handles:
            handle.remove()
        if engine is not None:
            if original_pack is not None:
                engine.runner._pack = original_pack
            await engine.close()
        save()
    return result


if __name__ == "__main__":
    config_path, output_path = map(Path, sys.argv[1:])
    result = asyncio.run(run(json.loads(config_path.read_text()), output_path))
    if not result["complete"]:
        raise SystemExit("biased-linear diagnostic incomplete; raw checkpoint retained")
