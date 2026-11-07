"""Validate and compare admission probes; partial studies stay explicitly partial."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

VARIANTS = ("atomic", "staged", "atomic_repeat", "staged_repeat", "target_first_atomic", "target_first_staged")
SOURCES = {"projection_probe.py", "projection_hook.py", "admission_probe.py", "operator_probe_hook.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_schedules(batch, admission):
    full = {f"diag-{i}": len(ids) for i, ids in enumerate(batch)}
    ones = dict.fromkeys(full, 1)
    if admission == "atomic":
        return [full, ones, ones]
    require(admission == "staged", "Unknown admission policy")
    return [{"diag-0": full["diag-0"]}, {**full, "diag-0": 1}, ones,
            {key: 1 for key in full if key != "diag-0"}]


def validate(payload):
    require(payload.get("exit_code") == 0 and bool(payload.get("capture")), "Crashed/missing capture")
    require(bool(payload.get("gpu", {}).get("uuid")), "Missing physical GPU identity")
    hashes = payload.get("source_hashes", {})
    require(SOURCES <= set(hashes) and all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v) for v in hashes.values()), "Missing/invalid source hashes")
    require(payload.get("probe_sha256") == hashes["admission_probe.py"] and payload.get("hook_sha256") == hashes["projection_hook.py"], "Inconsistent source hashes")
    capture = payload["capture"]
    config = capture["config"]
    require(bool(config.get("model", {}).get("revision")) and bool(config["model"].get("id")), "Unpinned model")
    batch = config["batch_ids"]
    require(len(batch) > 1 and all(ids and all(type(t) is int and t >= 0 for t in ids) for ids in batch), "Invalid input IDs")
    require(len({tuple(ids) for ids in batch}) == len(batch), "Ambiguous duplicate prompt identity")
    index = config["target_index"]
    require(type(index) is int and 0 <= index < len(batch), "Invalid target index")
    require(capture.get("execution", {}).get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "Admission was not explicitly controlled")
    plans = expected_schedules(batch, config["admission"])
    schedules = capture["schedules"]
    require(len(schedules) == len(plans), "Unexpected number of forward schedules")
    for actual, expected in zip(schedules, plans, strict=True):
        require(actual["num_scheduled_tokens"] == expected and actual["total_num_scheduled_tokens"] == sum(expected.values()), "Admission/query schedule differs from intervention")
    expected_events = []
    for i in range(len(batch)):
        expected_events.append({"event": "admit", "request_id": f"diag-{i}"})
        if config["admission"] == "staged" and i == 0:
            expected_events.append({"event": "step"})
    expected_events.extend([{"event": "step"}] * 3)
    require(capture["admission_events"] == expected_events, "Host admission sequence differs")
    results = capture["all_results"]
    require(len(results) == len(batch), "Missing request outputs")
    for i, result in enumerate(results):
        require(result["index"] == i and result["request_id"] == f"diag-{i}" and result.get("verified") is True, "Misidentified/unverified request")
        require(len(result["output_tokens"]) == 3 and all(type(t) is int and t >= 0 for t in result["output_tokens"]), "Incomplete token window")
    require(capture["result"] == results[index], "Target result differs from its request record")
    calls = capture["projection_calls"]
    require(len(calls) == len(plans), "Unaligned projection calls")
    seen = dict.fromkeys(plans[1] if config["admission"] == "staged" else plans[0], 0)
    for call, plan in zip(calls, plans, strict=True):
        rows = call["rows"]
        require(len(rows) == len(plan) and {r["request_id"] for r in rows} == set(plan), "Projection membership differs from schedule")
        require(call["shape"][0] == len(rows), "Projection shape mismatch")
        for row in rows:
            rid = row["request_id"]
            i = int(rid.removeprefix("diag-"))
            position = seen[rid]
            tokens = results[i]["output_tokens"]
            require(position < 3 and row["output_tokens_before"] == tokens[:position], "Generated prefix mismatch")
            require(row["computed_tokens"] == (0 if position == 0 else len(batch[i]) + position - 1), "Computed token position mismatch")
            require(row["native_argmax"] == tokens[position], "Projection does not explain emitted token")
            ids = row["candidate_ids"]
            require(ids and len(ids) == len(set(ids)) and row["native_argmax"] in ids, "Invalid candidate support")
            for key in ("native_logits", "fp32_projection", "permuted_logits"):
                require(len(row[key]) == len(ids) and all(math.isfinite(v) for v in row[key]), "Invalid/nonfinite candidate vector")
            require(len(row["hidden"]) == call["shape"][1] and all(math.isfinite(v) for v in row["hidden"]), "Invalid/nonfinite hidden vector")
            require(row["native_logits"][ids.index(row["native_argmax"])] == max(row["native_logits"]), "Selected token is not maximal")
            seen[rid] += 1
    require(set(seen.values()) == {3}, "Incomplete per-request projections")
    if config.get("operator_trace"):
        require(bool(capture.get("operator_calls")), "Missing operator trace")
    target = batch[index]
    target_plan = next(plan for plan in plans if f"diag-{index}" in plan)
    for record in capture.get("operator_calls", []):
        require(record["request_id"] == f"diag-{index}" and record["computed_tokens"] == 0 and record["output_tokens_before"] == [], "Operator trace is not target first prefill")
        require(record["prompt_length"] == len(target) and record["target_token_position"] == len(target) - 1 and record["target_input_token"] == target[-1], "Operator token identity mismatch")
        req_ids, lengths = record["req_ids"], record["query_lengths"]
        require(len(req_ids) == len(set(req_ids)) and dict(zip(req_ids, lengths, strict=True)) == target_plan, "Operator query identity mismatch")
        offset = sum(lengths[:req_ids.index(f"diag-{index}") + 1]) - 1
        require(record["target_packed_offset"] == offset and record["total_num_scheduled_tokens"] == sum(lengths), "Operator packed offset mismatch")
        require(record["vectors"], "Empty operator record")
        for vector in record["vectors"]:
            require(vector["values"] and all(math.isfinite(v) for v in vector["values"]), "Nonfinite operator values")
            require(vector["shape"][0] == sum(lengths) and math.prod(vector["shape"][1:]) == len(vector["values"]), "Operator output shape mismatch")
    return {tuple(prompt): result for prompt, result in zip(batch, results, strict=True)}


def layer_zero_records(records):
    require(records and records[0]["module"] == "model.embed_tokens", "Missing embedding trace start")
    end = next((i + 1 for i, record in enumerate(records) if record["module"] == "model.layers.0"), None)
    require(end is not None, "Missing layer-zero completion")
    trailing = records[end:]
    require(all(record["module"] == "model.layers.0.self_attn.rotary_emb" for record in trailing), "Unexplained trailing operator records")
    return records[:end], len(trailing)


def operator_comparison(left, right):
    a, b = left["capture"].get("operator_calls", []), right["capture"].get("operator_calls", [])
    if not a or not b:
        return {"status": "unavailable", "reason": "both operator traces required"}
    a, excluded_left = layer_zero_records(a)
    b, excluded_right = layer_zero_records(b)
    require(len(a) == len(b), "Operator execution sequence differs")
    comparisons = []
    for x, y in zip(a, b, strict=True):
        require(x["module"] == y["module"] and x["precision"] == y["precision"], "Operator sequence/precision mismatch")
        require([v["path"] for v in x["vectors"]] == [v["path"] for v in y["vectors"]], "Operator output structure differs")
        for vx, vy in zip(x["vectors"], y["vectors"], strict=True):
            require(vx["shape"][1:] == vy["shape"][1:] and vx["dtype"] == vy["dtype"], "Operator vector layout differs")
            diffs = [abs(u-v) for u, v in zip(vx["values"], vy["values"], strict=True)]
            comparisons.append({"module": x["module"], "path": vx["path"],
                                "exact_equal": vx["values"] == vy["values"], "changed_elements": sum(d != 0 for d in diffs),
                                "max_abs_difference": max(diffs), "mean_abs_difference": math.fsum(diffs)/len(diffs)})
    first_index = next((i for i, c in enumerate(comparisons) if not c["exact_equal"]), None)
    return {"status": "compared", "first_observed_unequal": comparisons[first_index] if first_index is not None else None,
            "preceding_observed_vector": comparisons[first_index - 1] if first_index else None,
            "excluded_trailing_alias_records": {"left": excluded_left, "right": excluded_right},
            "alias_limitation": "Older captures reuse the rotary module across later layers under its layer-zero name. Only the first contiguous sequence through layer-zero completion is analyzed; raw trailing records are retained.",
            "vectors": comparisons, "scope": "Identical target prompt/final input token; matched observed module/output sequence. Only embedding and layer-zero last-token outputs, not global earliest divergence or equal operator inputs."}


def analyze(cases, baseline=None):
    require(bool(cases) and set(cases) <= set(VARIANTS), "No known admission cases")
    validated = {name: validate(payload) for name, payload in cases.items()}
    first = next(iter(cases.values()))
    first_config = first["capture"]["config"]
    target = tuple(first_config["batch_ids"][first_config["target_index"]])
    original = next((p["capture"]["config"]["batch_ids"] for n, p in cases.items() if not n.startswith("target_first")), first_config["batch_ids"])
    reordered = [list(target), *(ids for ids in original if tuple(ids) != target)]
    for name, payload in cases.items():
        require(payload["gpu"]["uuid"] == first["gpu"]["uuid"], "Physical GPU changed")
        require(payload["source_hashes"] == first["source_hashes"] and payload.get("packages") == first.get("packages"), "Source/packages changed")
        config = payload["capture"]["config"]
        require(config["admission"] == ("staged" if "staged" in name else "atomic"), "Case label/admission mismatch")
        require(config["batch_ids"] == (reordered if name.startswith("target_first") else original), "Case order differs from intervention")
        require(config["model"] == first_config["model"] and tuple(config["batch_ids"][config["target_index"]]) == target, "Model/target identity changed")
        require(set(validated[name]) == set(next(iter(validated.values()))), "Peer prompt identities changed")
    comparisons = []
    for left, right in (("atomic", "staged"), ("atomic", "atomic_repeat"), ("staged", "staged_repeat"),
                        ("atomic", "target_first_atomic"), ("staged", "target_first_staged")):
        if left not in cases or right not in cases:
            continue
        requests = []
        for prompt, a in validated[left].items():
            b = validated[right][prompt]
            requests.append({"prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                             "left_request_id": a["request_id"], "right_request_id": b["request_id"],
                             "is_target": prompt == target, "left_tokens": a["output_tokens"], "right_tokens": b["output_tokens"],
                             "exact_equal": a["output_tokens"] == b["output_tokens"]})
        entry = {"left": left, "right": right, "all_requests_equal": all(r["exact_equal"] for r in requests), "requests": requests}
        if (left, right) == ("atomic", "staged"):
            entry["operators"] = operator_comparison(cases[left], cases[right])
        comparisons.append(entry)
    baseline_comparisons = []
    for name, payload in cases.items():
        for label, control in (baseline or {}).items():
            require(control.get("exit_code") == 0 and control.get("capture"), "Crashed baseline")
            config = control["capture"]["config"]
            require(config["model"] == first_config["model"], "Baseline model differs")
            prompt = tuple(config["batch_ids"][config["target_index"]])
            require(prompt == target and control["capture"]["result"].get("verified") is True, "Baseline target unverified/different")
            tokens = control["capture"]["result"]["output_tokens"]
            require(len(tokens) == 3, "Incomplete baseline window")
            baseline_comparisons.append({"case": name, "baseline": label, "tokens_equal": validated[name][target]["output_tokens"] == tokens,
                                         "baseline_gpu_uuid": control["gpu"]["uuid"], "same_gpu": control["gpu"]["uuid"] == payload["gpu"]["uuid"],
                                         "scope": "token reproduction only across separate runs; no cross-run numerical equivalence claim"})
    return {"complete_six_case_study": set(cases) == set(VARIANTS), "present": list(cases), "missing": [v for v in VARIANTS if v not in cases],
            "gpu_uuid": first["gpu"]["uuid"], "comparisons": comparisons, "baseline_token_comparisons": baseline_comparisons,
            "gate": "unchanged; diagnostic evidence only"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    cases = {name: json.loads((args.directory/f"{name}.json").read_text()) for name in VARIANTS if (args.directory/f"{name}.json").exists()}
    baseline = {name: json.loads((args.baseline/f"{name}.json").read_text()) for name in ("original_control", "original", "target_first", "target_first_control") if (args.baseline/f"{name}.json").exists()} if args.baseline else None
    result = analyze(cases, baseline)
    (args.directory/"admission-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))
