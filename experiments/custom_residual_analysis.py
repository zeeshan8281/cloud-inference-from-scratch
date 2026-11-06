"""Pure integrity checks and bounded observations for custom residual captures.

Accepts parent Modal payload JSON (with files text bundle) or standalone child
output JSON plus its sibling trace files. Never interprets this as gate clearance.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import struct
from pathlib import Path

from experiments.fix_holdout import digest, validate_capture


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode_vector(vector):
    raw = base64.b64decode(vector["bytes_base64"], validate=True)
    require(hashlib.sha256(raw).hexdigest() == vector["sha256"], "vector checksum mismatch")
    dtype = vector["dtype"]
    formats = {"torch.float16": "e", "torch.float32": "f", "torch.float64": "d"}
    if dtype == "torch.bfloat16":
        require(len(raw) % 2 == 0, "invalid bfloat16 byte count")
        values = [struct.unpack("<f", struct.pack("<I", bits[0] << 16))[0] for bits in struct.iter_unpack("<H", raw)]
    else:
        require(dtype in formats, f"unsupported tensor dtype {dtype}")
        fmt = "<" + formats[dtype]
        require(len(raw) % struct.calcsize(fmt) == 0, "invalid vector byte count")
        values = [entry[0] for entry in struct.iter_unpack(fmt, raw)]
    require(len(values) == math.prod(vector["row_shape"]) > 0, "vector shape mismatch")
    require(vector.get("finite") is True and all(math.isfinite(value) for value in values), "nonfinite operator vector")
    return values


def row_index(records):
    indexed = {}
    for record in records:
        require(record["query_end"] - 1 == record["target_position"], "target row is not final query position")
        require(0 <= record["target_offset"] < record["packed_tokens"], "invalid target packed offset")
        for vector in record.get("vectors", []):
            key = (record["target_position"], record["operator"], vector["path"])
            require(key not in indexed, f"duplicate observed row {key}")
            indexed[key] = (record, vector, decode_vector(vector))
    return indexed


def compare_traces(left, right):
    a, b = row_index(left), row_index(right)
    require(a.keys() == b.keys(), "operator coverage differs")
    rows, first_by_position = [], {}
    for key, (record_a, vector_a, values_a) in a.items():
        record_b, vector_b, values_b = b[key]
        for field in ("target_input_token", "output_tokens_before", "query_start", "query_end"):
            require(record_a[field] == record_b[field], f"unaligned input/history {key}/{field}")
        require(vector_a["row_shape"] == vector_b["row_shape"] and vector_a["dtype"] == vector_b["dtype"], "incompatible row vectors")
        exact = vector_a["sha256"] == vector_b["sha256"]
        differences = [abs(x-y) for x, y in zip(values_a, values_b, strict=True)]
        row = {"target_position": key[0], "operator": key[1], "path": key[2], "exact_bytes": exact,
               "changed_elements": sum(x != y for x, y in zip(values_a, values_b, strict=True)),
               "max_abs": max(differences), "mean_abs": sum(differences)/len(differences),
               "left_shape": vector_a["shape"], "right_shape": vector_b["shape"],
               "shared_generated_prefix": record_a["output_tokens_before"]}
        rows.append(row)
        if not exact:
            first_by_position.setdefault(str(key[0]), row)
    kv_a = {(r["target_position"], r["operator"]): r for r in left if "logical_kv" in r}
    kv_b = {(r["target_position"], r["operator"]): r for r in right if "logical_kv" in r}
    require(kv_a.keys() == kv_b.keys(), "logical KV coverage differs")
    kv_differences, first_kv = [], {}
    for key, first in kv_a.items():
        second = kv_b[key]
        require(first["target_input_token"] == second["target_input_token"] and first["output_tokens_before"] == second["output_tokens_before"], "KV history differs")
        for kind in ("keys", "values"):
            x, y = first["logical_kv"][kind], second["logical_kv"][kind]
            require(x["shape"] == y["shape"] and x["dtype"] == y["dtype"], "incompatible logical KV shapes")
            if x["sha256"] != y["sha256"]:
                item = {"target_position": key[0], "operator": key[1], "kind": kind, "shape": x["shape"]}
                kv_differences.append(item)
                first_kv.setdefault(str(key[0]), item)
    return {"observed_row_count": len(rows), "all_observed_rows_equal": all(row["exact_bytes"] for row in rows),
            "first_observed_unequal_row": next((row for row in rows if not row["exact_bytes"]), None),
            "first_observed_unequal_row_per_position": first_by_position,
            "first_logical_kv_difference_per_position": first_kv, "logical_kv_differences": kv_differences,
            "rows": rows, "scope": "first unequal observed target last row in trace order; not globally earliest divergent token or causal proof"}



def validate_schedules(case, record, ids):
    requests = record["capture"]["records"]
    known = {request["request_id"]: prompt + request["output_tokens"]
             for request, prompt in zip(requests, ids, strict=True)}
    cursor = {key: 0 for key in known}
    for index, schedule in enumerate(record["schedules"]):
        require(schedule["forward"] == index, "schedule forward sequence mismatch")
        require(schedule["request_ids"] == [item["request_id"] for item in schedule["items"]], "schedule membership mismatch")
        require(len(set(schedule["request_ids"])) == len(schedule["request_ids"]), "duplicate scheduled request")
        for item, (begin, end) in zip(schedule["items"], schedule["query_ranges"], strict=True):
            key = item["request_id"]
            require(key in known and item["start"] == cursor[key] and item["end"] > item["start"], "noncontiguous/unknown request schedule")
            require(end-begin == item["end"]-item["start"], "packed query length mismatch")
            require(item["token_ids"] == known[key][item["start"]:item["end"]], "actual scheduled tokens differ from capture history")
            request_index = next(i for i, request in enumerate(requests) if request["request_id"] == key)
            require(item["label"] == f"{case}-{request_index}", "scheduled label mismatch")
            require(item["output_tokens_before"] == requests[request_index]["output_tokens"][:max(0, item["start"]-len(ids[request_index])+1)], "scheduled generated history mismatch")
            cursor[key] = item["end"]
    require(all(cursor[request["request_id"]] == len(prompt)+1 for request, prompt in zip(requests, ids, strict=True)), "incomplete two-token schedule")


def analyze(path):
    path = Path(path)
    document = json.loads(path.read_text())
    environment = None
    if "files" in document:
        require(document["process"]["returncode"] == 0, "child process failed")
        environment = document["environment"]
        require(environment["gpu_before"]["uuid"] and environment["gpu_before"]["uuid"] == environment["gpu_after"]["uuid"], "GPU changed or missing")
        files = document["files"]
        child = json.loads(files["output.json"])
        def read_trace(name):
            return files[name]
        for source, checksum in child["source_hashes"].items():
            relative = source.removeprefix("/root/")
            require(environment["source_hashes"].get(relative) == checksum, f"executed source mismatch: {relative}")
    else:
        child = document
        def read_trace(name):
            return (path.parent / name).read_text()
    require(child.get("complete") is True and child.get("error") is None, "incomplete diagnostic")
    require(child["precision"]["allow_fp16_reduced_precision_reduction"] is False, "wrong actual precision policy")
    batch, target = child["config"]["batch_ids"], child["config"]["target_index"]
    require(child["batch_sha256"] == digest(batch), "saved input checksum mismatch")
    expected_cases = child["config"].get("cases", ["batch_control", "solo_control", "batch", "solo", "batch_repeat", "solo_repeat"])
    require(set(child["cases"]) == set(expected_cases), "missing or extra cases")
    traces, outputs, projections = {}, {}, {}
    for case, record in child["cases"].items():
        ids = batch if case.startswith("batch") else [batch[target]]
        require(record["target_local_index"] == (target if case.startswith("batch") else 0), "target index mismatch")
        validate_capture(record["capture"], ids, 2, cold=True)
        validate_schedules(case, record, ids)
        outputs[case] = [request["output_tokens"] for request in record["capture"]["records"]]
        text = read_trace(Path(record["trace_path"]).name)
        require(hashlib.sha256(text.encode()).hexdigest() == record["trace_sha256"], "operator trace checksum mismatch")
        trace = [json.loads(line) for line in text.splitlines() if line]
        require(len(trace) == record["trace_calls"], "trace call count mismatch")
        require([row["call"] for row in trace] == list(range(1, len(trace)+1)), "trace call sequence mismatch")
        if case.endswith("_control"):
            require(not trace and record["operator_observation_enabled"] is False, "observed control")
        else:
            require(trace and record["operator_observation_enabled"] is True, "missing operator trace")
            row_index(trace)
            traces[case] = trace
        projections[case] = record["logits_calls"]
    comparisons = {}
    for a, b in (("batch", "solo"), ("batch", "batch_repeat"), ("solo", "solo_repeat")):
        if a in traces and b in traces:
            comparisons[a+"_vs_"+b] = compare_traces(traces[a], traces[b])
    tokens = {}
    for a,b in (("batch_control","batch"),("solo_control","solo"),("batch","batch_repeat"),("solo","solo_repeat")):
        if a in outputs and b in outputs:
            tokens[a+"_vs_"+b] = {"all_request_outputs_equal": outputs[a] == outputs[b]}
    target_outputs = {case: output[child["cases"][case]["target_local_index"]] for case, output in outputs.items()}
    require(target_outputs == child["target_outputs"], "reported target outputs mismatch actual captures")
    return {"complete_capture": True, "target_outputs": target_outputs, "token_reproducibility": tokens,
            "trace_comparisons": comparisons, "frozen_projection_replays": projections,
            "precision": child["precision"], "environment": environment, "gate_cleared": False,
            "scope": "bounded diagnostic on exposed prompts; row observations and KV hashes localize differences but do not establish a root cause"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("output")
    args = parser.parse_args()
    target = Path(args.output)
    require(not target.exists(), "refusing to overwrite analysis")
    target.write_text(json.dumps(analyze(args.capture), indent=2, allow_nan=False)+"\n")
