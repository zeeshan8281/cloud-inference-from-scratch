"""Integrity checks and observations for frozen-GEMM and model precision probes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from experiments.admission_analysis import analyze, layer_zero_records, require, validate

VARIANTS = ("staged_linear", "atomic_precise", "staged_precise", "atomic_precise_repeat", "staged_precise_repeat", "target_first_staged_precise")


def differences(left, right):
    require(len(left) == len(right) and bool(left), "Vector dimensions differ")
    require(all(math.isfinite(v) for v in [*left, *right]), "Nonfinite comparison vector")
    errors = [abs(a-b) for a, b in zip(left, right, strict=True)]
    return {"exact_equal": left == right, "changed_elements": sum(v != 0 for v in errors),
            "max_abs_difference": max(errors), "mean_abs_difference": math.fsum(errors)/len(errors)}


def linear_comparison(record):
    require(record["module"] == "model.layers.0.mlp.down_proj", "Unexpected linear operator")
    require(record["weight_dtype"] == "torch.float16" and record["weight_shape"][0] == 2048, "Wrong operator precision/width")
    values = record["input_values"]
    require(len(values) == record["weight_shape"][1] and all(math.isfinite(v) for v in values), "Invalid frozen input")
    packed = struct.pack(f"<{len(values)}e", *values)
    require(hashlib.sha256(packed).hexdigest() == record["input_sha256"], "Frozen input hash mismatch")
    require(len(record["weight_sha256"]) == 64 and all(c in "0123456789abcdef" for c in record["weight_sha256"]), "Invalid weight hash")
    native_flag = record["native_allow_fp16_reduced_precision_reduction"]
    require(type(native_flag) is bool and record["restored_allow_fp16_reduced_precision_reduction"] is native_flag, "Reduction flag was not restored")
    for key in ("native_output", "fp64_reference", "fp64_reference_rounded_fp16"):
        require(len(record[key]) == 2048 and all(math.isfinite(v) for v in record[key]), "Incomplete/nonfinite full output vector")
    rounded = [struct.unpack("<e", struct.pack("<e", v))[0] for v in record["fp64_reference"]]
    require(rounded == record["fp64_reference_rounded_fp16"], "FP64 reference rounding mismatch")
    require(len(record["replays"]) == 4, "Expected four isolated replays")
    replays = {}
    for replay in record["replays"]:
        flag = replay["allow_fp16_reduced_precision_reduction"]
        require(type(flag) is bool, "Invalid replay reduction flag")
        key = (flag, replay["input_shape"][0])
        require(key not in replays, "Duplicate replay condition")
        require(replay["input_shape"][1] == len(values), "Replay input width mismatch")
        require(replay["input_sha256"] == record["input_sha256"] and replay["weight_sha256"] == record["weight_sha256"], "Frozen input/weight changed")
        require(replay["target_offset"] == {4096: 1023, 3585: 512}.get(key[1]), "Unexpected replay row position")
        require(len(replay["output"]) == 2048 and all(math.isfinite(v) for v in replay["output"]), "Incomplete/nonfinite replay vector")
        replays[key] = replay["output"]
    require(set(replays) == {(flag, rows) for flag in (True, False) for rows in (4096, 3585)}, "Missing replay condition")
    require(record["native_input_shape"] == [3585, len(values)] and record["native_target_offset"] == 512, "Native staged layout differs")
    per_condition = [{"allow_fp16_reduced_precision_reduction": flag, "rows": rows,
                      "versus_fp64": differences(vector, record["fp64_reference"]),
                      "versus_fp64_rounded_fp16": differences(vector, rounded)}
                     for (flag, rows), vector in replays.items()]
    return {"input_sha256": record["input_sha256"], "weight_sha256": record["weight_sha256"],
            "full_output_width": 2048, "reduction_flag_restored": True,
            "shape_drift_reduced_true": differences(replays[True, 4096], replays[True, 3585]),
            "shape_drift_reduced_false": differences(replays[False, 4096], replays[False, 3585]),
            "native_versus_same_shape_replay": differences(record["native_output"], replays[native_flag, 3585]),
            "per_condition": per_condition,
            "layout_conditions": [{"rows": 4096, "target_offset": 1023}, {"rows": 3585, "target_offset": 512}],
            "scope": "Same recorded FP16 target input and weight hashes; zero-peer operator probes. Row count and target offset vary together, so this does not isolate row-count-only effects. FP64 is a dot-product reference for these FP16 operands, not a full-model oracle. Scientific agreement is reported, never required."}


def analyze_precision(cases):
    require(cases and set(cases) <= set(VARIANTS), "No known precision cases")
    validated = {name: validate(payload) for name, payload in cases.items()}
    first = next(iter(cases.values()))
    cfg = first["capture"]["config"]
    target = tuple(cfg["batch_ids"][cfg["target_index"]])
    first_prompts = set(next(iter(validated.values())))
    for name, payload in cases.items():
        config = payload["capture"]["config"]
        require(payload["gpu"]["uuid"] == first["gpu"]["uuid"], "Physical GPU changed")
        require(payload["source_hashes"] == first["source_hashes"] and payload.get("packages") == first.get("packages"), "Probe source/packages changed")
        require("linear_probe_hook.py" in payload["source_hashes"], "Missing linear observer fingerprint")
        require(config["model"] == cfg["model"] and tuple(config["batch_ids"][config["target_index"]]) == target and set(validated[name]) == first_prompts, "Model or prompt identities changed")
        flag = name == "staged_linear"
        require(config.get("fp16_reduced_precision_reduction") is flag and payload["capture"]["execution"].get("fp16_reduced_precision_reduction") is flag, "Requested precision policy not realized")
        require(config["admission"] == ("staged" if "staged" in name else "atomic"), "Admission label mismatch")
        for record in payload["capture"].get("operator_calls", []):
            require(record["precision"]["allow_fp16_reduced_precision_reduction"] is flag, "Operator precision differs from policy")
    linear = None
    if "staged_linear" in cases:
        capture = cases["staged_linear"]["capture"]
        records = capture.get("linear_calls", [])
        require(len(records) == 1, "Missing/multiple frozen linear observations")
        record = records[0]
        require(record["request_id"] == f"diag-{capture['config']['target_index']}" and record["computed_tokens"] == 0 and record["output_tokens_before"] == [], "Linear target prefix mismatch")
        require(record["target_token_position"] == len(target)-1, "Linear target position mismatch")
        operators, _ = layer_zero_records(capture["operator_calls"])
        activation = next(r for r in operators if r["module"] == "model.layers.0.mlp.act_fn")
        projection = next(r for r in operators if r["module"] == "model.layers.0.mlp.down_proj")
        require(record["input_values"] == activation["vectors"][0]["values"] and record["native_output"] == projection["vectors"][0]["values"], "Frozen/native vectors differ from observed model operator")
        require(record["req_ids"] == activation["req_ids"] and record["query_lengths"] == activation["query_lengths"], "Frozen operator request layout mismatch")
        linear = linear_comparison(record)
    precise = {name.replace("_precise", ""): payload for name, payload in cases.items() if name != "staged_linear"}
    model = analyze(precise) if precise else None
    ablation = None
    if "staged_linear" in cases and "staged_precise" in cases:
        ablation = [{"prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(), "is_target": prompt == target,
                     "reduced_true_tokens": result["output_tokens"], "reduced_false_tokens": validated["staged_precise"][prompt]["output_tokens"],
                     "exact_equal": result["output_tokens"] == validated["staged_precise"][prompt]["output_tokens"]}
                    for prompt, result in validated["staged_linear"].items()]
    return {"complete_precision_study": set(cases) == set(VARIANTS), "present": list(cases), "missing": [name for name in VARIANTS if name not in cases],
            "gpu_uuid": first["gpu"]["uuid"], "frozen_linear": linear, "reduced_false_model_comparisons": model,
            "staged_policy_ablation": ablation, "gate": "unchanged; no pilot or holdout clearance from this diagnostic"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    cases = {name: json.loads((args.directory/f"{name}.json").read_text()) for name in VARIANTS if (args.directory/f"{name}.json").exists()}
    result = analyze_precision(cases)
    (args.directory/"precision-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))
