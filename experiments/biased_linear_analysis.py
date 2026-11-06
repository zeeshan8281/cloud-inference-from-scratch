"""Pure integrity and separated layout comparisons for the biased-linear probe."""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

from experiments.biased_linear_probe import layout_variants, vector_metrics
from experiments.fix_holdout import digest, validate_capture

METHODS = ("fp16_biased_linear", "fp16_linear_then_bias", "fp32_biased_linear_cast_fp16")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def layout_effects(replays, batch_rows, batch_offset, local_rows, local_offset):
    indexed = {(r["rows"], r["offset"], r["method"]): r["values"] for r in replays}
    require(len(indexed) == len(replays), "duplicate replay layout")
    expected = {(rows, offset, method) for rows, offset in layout_variants(batch_rows, batch_offset, local_rows, local_offset) for method in METHODS}
    require(set(indexed) == expected, "missing or extra layout/method replay")
    result = {}
    for method in METHODS:
        original = indexed[(batch_rows, batch_offset, method)]
        same_rows = indexed[(batch_rows, local_offset, method)]
        solo = indexed[(local_rows, local_offset, method)]
        result[method] = {
            "offset_only": {"rows": batch_rows, "offsets": [batch_offset, local_offset], **vector_metrics(original, same_rows)},
            "row_count_only": {"rows": [batch_rows, local_rows], "offset": local_offset, **vector_metrics(same_rows, solo)},
            "observed_layout_pair": vector_metrics(original, solo),
        }
    return result


def analyze(path):
    document = json.loads(Path(path).read_text())
    environment = None
    if "files" in document:
        require(document["process"]["returncode"] == 0, "failed child process")
        environment = document["environment"]
        require(environment["gpu_before"]["uuid"] and environment["gpu_before"]["uuid"] == environment["gpu_after"]["uuid"], "GPU changed or missing")
        require(len(environment["source_hashes"].get("experiments/biased_linear_probe.py", "")) == 64, "missing executed probe hash")
        child = json.loads(document["files"]["output.json"])
    else:
        child = document
    require(child.get("complete") is True and child.get("error") is None, "incomplete biased-linear capture")
    require(child["precision"]["allow_fp16_reduced_precision_reduction"] is False, "wrong actual precision policy")
    config = child["config"]
    batch, target = config["batch_ids"], config["target_index"]
    require(child["batch_sha256"] == digest(batch), "input batch checksum mismatch")
    expected_cases = config.get("cases", ["batch", "solo"])
    require(set(child["cases"]) == set(expected_cases) and {"batch", "solo"} <= set(expected_cases), "missing batch/solo cases")
    expected_calls, outputs, schedules = set(), {}, {}
    for case, data in child["cases"].items():
        ids = batch if case.startswith("batch") else [batch[target]]
        local_target = target if case.startswith("batch") else 0
        require(data["target_local_index"] == local_target, "target attribution mismatch")
        validate_capture(data["capture"], ids, 2, cold=True)
        outputs[case] = data["capture"]["records"][local_target]["output_tokens"]
        cursor = 0
        for forward, schedule in enumerate(data["schedules"]):
            total = sum(end-begin for begin,end in schedule["query_ranges"])
            for item,(begin,end) in zip(schedule["items"], schedule["query_ranges"], strict=True):
                if item["label"] != f"{case}-{local_target}":
                    continue
                require(item["start"] == cursor and item["end"] > cursor, "noncontiguous target schedule")
                cursor = item["end"]
                require(item["token_ids"] == (batch[target]+outputs[case])[item["start"]:item["end"]], "target scheduled history mismatch")
                position = item["end"]-1
                require((case,position) not in schedules, "duplicate target position")
                schedules[(case,position)] = {"rows": total, "offset": end-1, "local_rows": end-begin,
                                              "local_offset": end-begin-1, "forward": forward,
                                              "output_tokens_before": item["output_tokens_before"]}
                expected_calls.update((case,position,projection) for projection in ("q_proj", "k_proj", "v_proj"))
        require(cursor == len(batch[target])+1, "incomplete target schedule")
    require(child["target_outputs"] == outputs, "reported target output mismatch")
    calls = {(c["case"],c["target_position"],c["projection"]): c for c in child["calls"]}
    require(len(calls) == len(child["calls"]) and set(calls) == expected_calls, "missing or duplicate projection observations")
    summaries = []
    for key, call in calls.items():
        case, position, projection = key
        require(all(call[field] == value for field,value in schedules[(case,position)].items()), "projection schedule mismatch")
        require(call["parameters_unchanged"] is True, "mutated replay parameters")
        require(call["native_replay_exact"] is True and call["native"] == call["native_replay"], "actual-layout replay failed to reproduce native output")
        require(call["input_dtype"] == call["output_dtype"] == "torch.float16", "unexpected operator precision")
        for field in ("target_input_sha256", "actual_input_matrix_sha256", "weight_sha256", "bias_sha256"):
            require(isinstance(call[field], str) and len(call[field]) == 64 and all(c in "0123456789abcdef" for c in call[field]), "invalid tensor checksum")
        vectors = [call[name] for name in ("native", "native_replay", "fp64_reference", "fp64_rounded_fp16")]
        vectors += [replay["values"] for replay in call["replays"]]
        require(all(len(vector) == len(vectors[0]) > 0 and all(math.isfinite(x) for x in vector) for vector in vectors), "nonfinite or incomplete target vector")
        require(call["fp64_rounded_fp16"] == [struct.unpack("<e", struct.pack("<e", value))[0] for value in call["fp64_reference"]], "incorrect FP64-to-FP16 reference rounding")
        base = calls[("batch",position,projection)]
        require(all(call[field] == base[field] for field in ("target_input_sha256", "weight_sha256", "bias_sha256")), "batch/solo frozen inputs or parameters differ")
        require(call["fp64_reference"] == base["fp64_reference"], "fixed-input FP64 references differ")
        for replay in call["replays"]:
            for metric, reference in (("error_vs_native",call["native"]), ("error_vs_fp64",call["fp64_reference"]), ("error_vs_fp64_rounded",call["fp64_rounded_fp16"])):
                require(replay[metric] == vector_metrics(replay["values"],reference), "recorded error metric mismatch")
        effects = layout_effects(call["replays"],base["rows"],base["offset"],base["local_rows"],base["local_offset"])
        summaries.append({"case": case, "position": position, "projection": projection,
                          "native_replay_exact": True, "frozen_inputs_match_batch": True,
                          "native_vs_batch": vector_metrics(call["native"],base["native"]),
                          "native_error_vs_fp64": vector_metrics(call["native"],call["fp64_reference"]),
                          "native_error_vs_fp64_rounded": vector_metrics(call["native"],call["fp64_rounded_fp16"]),
                          "layout_effects": effects,
                          "reference_errors": [{key: value for key,value in replay.items() if key != "values"} for replay in call["replays"]]})
    return {"complete_capture": True, "native_replays_exact": True, "target_outputs": outputs,
            "precision": child["precision"], "environment": environment, "comparisons": summaries,
            "gate_cleared": False, "scope": "observed target Q/K/V outputs under frozen-input layout/method intervention; no full-model policy fix validated"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("output")
    args = parser.parse_args()
    target = Path(args.output)
    require(not target.exists(), "refusing to overwrite analysis")
    target.write_text(json.dumps(analyze(args.capture), indent=2, allow_nan=False)+"\n")
