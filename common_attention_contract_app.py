"""Durable A5 installed-attention contract followed by the exposed failed pair."""

import base64
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path

import modal

FILES = ("common_attention_contract_probe.py",)
if modal.is_local():
    from fix_holdout_app import capture_image

    image = capture_image.add_local_file(
        "experiments/common_attention_contract_probe.py",
        "/root/experiments/common_attention_contract_probe.py",
        copy=True,
    )
    image = image.add_local_file(
        "common_attention_contract_app.py", "/root/common_attention_contract_app.py", copy=True
    )
else:
    image = None
app = modal.App("cie-common-attention-A5-contract")
volume = modal.Volume.from_name("cloud-inference-model-cache", create_if_missing=False)
evidence_volume = modal.Volume.from_name("cie-retained-diagnostics", create_if_missing=True)


def source(root, deployed=False):
    from experiments.fix_holdout import computational_source

    files = ["common_attention_contract_app.py", "experiments/common_attention_contract_probe.py"]
    return {
        "core": computational_source(root, deployed=deployed),
        "diagnostic": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
        },
    }


def tensor_bytes(record):
    raw = base64.b64decode(record["bytes_base64"], validate=True)
    formats = {"torch.float16": "e", "torch.float32": "f", "torch.float64": "d"}
    fmt = formats[record["dtype"]]
    count = math.prod(record["shape"])
    if (
        len(raw) != count * struct.calcsize(fmt)
        or hashlib.sha256(raw).hexdigest() != record["sha256"]
    ):
        raise ValueError("tensor shape/byte/checksum mismatch")
    if not all(math.isfinite(v[0]) for v in struct.iter_unpack("<" + fmt, raw)):
        raise ValueError("nonfinite captured tensor")
    return raw


def check_fixture(fixture, expected):
    wanted = {
        impl + "-" + selection for impl in ("custom", "vllm") for selection in ("batch", "solo")
    }
    if (
        fixture.get("schema") != "captured-attention-A4-input-v1"
        or set(fixture["states"]) != wanted
        or fixture.get("target_index") != 12
        or fixture.get("layer_index") != 0
        or fixture.get("logical_position") != 511
    ):
        raise ValueError("incorrect actual attention fixture")
    reference = fixture["states"]["custom-batch"]
    for state in fixture["states"].values():
        if (
            state["environment"]["source_manifest"]["core"]["sha256"]
            != fixture["historical_computational_source_sha256"]
        ):
            raise ValueError("historical computational source mismatch")
        for key, shape in (("keys", [512, 2, 128]), ("values", [512, 2, 128])):
            if state[key]["shape"] != shape or state[key]["dtype"] != "torch.float16":
                raise ValueError("requires complete actual512 causal KV")
        for key in ("query", "keys", "values", "output"):
            raw = tensor_bytes(state[key])
            if state[key]["dtype"] != "torch.float16":
                raise ValueError("actual native operand dtype mismatch")
            if key in ("query", "output") and len(raw) != 4096:
                raise ValueError("target native vector size mismatch")
            if key != "output" and raw != tensor_bytes(reference[key]):
                raise ValueError("actual target operands differ across engines/layouts")


def verify_historical(fixture):
    cached = {}
    for state in fixture["states"].values():
        path = state["raw_path"]
        if path not in cached:
            raw = Path(path).read_bytes()
            cached[path] = (hashlib.sha256(raw).hexdigest(), json.loads(raw))
        checksum, payload = cached[path]
        if checksum != state["raw_sha256"] or payload["environment"] != state["environment"]:
            raise ValueError("actual historical raw/environment changed")
        implementation, selection = state["implementation"], state["selection"]
        phase = payload["capture"]["cases"][selection]
        rows = [
            json.loads(line)
            for line in payload["raw_traces"][f"capture.{selection}.operators.jsonl"].splitlines()
        ]
        rows = [row for row in rows if row.get("target_position") == 511]
        operator = (
            "layers.0.attention" if implementation == "custom" else "model.layers.0.self_attn.attn"
        )

        def get(op, direction, rows=rows):
            matches = [
                row
                for row in rows
                if row.get("operator") == op and row.get("direction") == direction
            ]
            if len(matches) != 1:
                raise ValueError("missing/ambiguous actual operator")
            return matches[0]

        inp, out = get(operator, "input"), get(operator, "output")
        kv = (
            get("layers.0.logical_kv", "after_append")
            if implementation == "custom"
            else get(operator, "logical_kv_after")
        )
        if (
            state["query"] != inp["vectors"][0]
            or state["output"] != out["vectors"][0]
            or any(state[key] != kv["logical_kv"][key] for key in ("keys", "values"))
        ):
            raise ValueError("fixture tensor differs from actual trace")
        if state["query_context"] != {k: v for k, v in inp.items() if k != "vectors"} or state[
            "kv_context"
        ] != {k: v for k, v in kv.items() if k != "logical_kv"}:
            raise ValueError("actual operator context changed")
        matches = [row for row in phase["schedules"] if row["forward"] == inp["forward"]]
        if (
            matches != [state["schedule"]]
            or state["source_policy"] != phase["capture"]["numerical_policy"]
        ):
            raise ValueError("actual schedule/policy changed")
    pair = fixture["failed_pair"]
    raw = Path(pair["raw_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pair["raw_sha256"]:
        raise ValueError("historical failed pair checksum changed")
    historical = json.loads(raw)
    if pair["index"] != 9 or pair["mode"] != "resource_normalized":
        raise ValueError("wrong failed-pair selection")
    for child in historical["children"].values():
        if child["capture"]["config"]["workload"] != pair["workload"]:
            raise ValueError("workload differs from actual failed pair09")


def validate_payload(payload, config, expected, fixture, fixture_sha):
    from cloud_engine.common_attention import ATTENTION_ID
    from cloud_engine.numerical_policy import POLICY_ID

    result, env = payload["capture"], payload["environment"]
    if json.loads(payload["raw_checkpoint"]) != result:
        raise ValueError("raw checkpoint differs from parsed primitive result")
    process = payload["process"]
    if process.get("returncode") != 0 or process.get("timeout") or process.get("cancelled"):
        raise ValueError("primitive process failed")
    if (
        result.get("schema") != "common-attention-contract-v1"
        or result.get("complete") is not True
        or result.get("error") is not None
        or result.get("config") != config
        or result.get("fixture_sha256") != fixture_sha
    ):
        raise ValueError("primitive evidence incomplete/config mismatch")
    if (
        result.get("policy_id") != POLICY_ID
        or result.get("attention_id") != ATTENTION_ID
        or result.get("installed_callable_active") is not True
    ):
        raise ValueError("actual candidate binding missing")
    if (
        result.get("native_builder_class") != "FlashAttentionMetadataBuilder"
        or result.get("full_attention_model_graph_supported") is not False
    ):
        raise ValueError("native FA2 builder/supported scope mismatch")
    if env.get("metadata_error") or env["source_manifest"] != expected:
        raise ValueError("actual executed source mismatch")
    if not env["gpu_before"].get("uuid") or env["gpu_before"]["uuid"] != env["gpu_after"].get(
        "uuid"
    ):
        raise ValueError("actual physical GPU identity missing/changed")
    for state in fixture["states"].values():
        for key in ("model", "dependencies", "model_files_sha256"):
            if env[key] != state["environment"][key]:
                raise ValueError("actual model/runtime mismatch")
    for name, path in (
        ("common_attention", "src/cloud_engine/common_attention.py"),
        ("kernel", "src/cloud_engine/kernel.py"),
    ):
        if result["source_sha256"][name] != expected["core"]["files"][path]:
            raise ValueError("actual imported candidate source mismatch")
    precision = result["precision"]
    if (
        precision.get("allow_fp16_reduced_precision_reduction") is not False
        or precision.get("allow_tf32") is not False
        or precision.get("float32_matmul_precision") != "highest"
    ):
        raise ValueError("actual precision mismatch")
    if result.get("profiling_none_preserves_alias_cache_output") is not True:
        raise ValueError("native profiling contract failed")
    expected_cases = {
        "prefill": (256, 1),
        "mixed": (258, 3),
        "shared_decode": (2, 2),
        "changed_metadata_replay": (2, 1),
    }
    if len(result["cases"]) != 4 or {row["name"] for row in result["cases"]} != set(expected_cases):
        raise ValueError("missing primitive cases")
    for row in result["cases"]:
        if (row["tokens"], row["requests"]) != expected_cases[row["name"]]:
            raise ValueError("primitive scenario shape mismatch")
        if row["name"] == "prefill" and row.get("historical_custom_target_exact") is not True:
            raise ValueError("historical custom target not reproduced")
        if row["name"] == "changed_metadata_replay" and row.get("stable_graph_storage") is not True:
            raise ValueError("metadata replay changed graph storage")
        for phase in ("eager", "compiled", "graph"):
            observed = row[phase]
            if any(
                observed.get(key) is not True
                for key in (
                    "output_alias",
                    "output_exact",
                    "cache_append_and_untouched_exact",
                    "padding_unchanged",
                )
            ):
                raise ValueError("actual installed primitive contract failed")
            for kind in ("output", "cache"):
                value = observed[kind + "_sha256"]
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)
                    or value != row["expected_" + kind + "_sha256"]
                ):
                    raise ValueError("primitive digest mismatch")
    changed = next(row for row in result["cases"] if row["name"] == "changed_metadata_replay")
    initial = next(row for row in result["cases"] if row["name"] == "shared_decode")
    if (
        initial["metadata_ids"] != [0, 1]
        or changed["metadata_ids"] != [0, 0]
        or initial["metadata_positions"] != [511, 511]
        or changed["metadata_positions"] != [510, 511]
    ):
        raise ValueError("graph logical metadata did not change as declared")
    if (
        changed.get("graph_storage_before") != initial.get("graph_storage")
        or changed.get("graph_storage_after") != initial.get("graph_storage")
        or len(initial.get("graph_storage", [])) != 9
        or any(type(pointer) is not int or pointer <= 0 for pointer in initial["graph_storage"])
    ):
        raise ValueError("actual graph buffer addresses missing/changed")
    if initial["query_sha256"] == changed["query_sha256"]:
        raise ValueError("graph query values did not change")
    if initial["write_slots"] == changed["write_slots"]:
        raise ValueError("graph write slots did not change")


def child(config, fixture_bytes, model):
    import sys

    import torch

    from experiments.parallel_validation import run_cancellable_process
    from experiments.sentinel_pilot import gpu_state_snapshot, resolved_dependency_versions

    root = Path(tempfile.mkdtemp(prefix="attention-A5-contract-"))
    (root / "input.json").write_bytes(fixture_bytes)
    config = dict(config, fixture_path=str(root / "input.json"))
    (root / "config.json").write_text(json.dumps(config))
    env = {
        "source_manifest": source(Path("/root"), True),
        "source": os.environ.get("SOURCE_COMMIT"),
        "model": model,
        "dependencies": resolved_dependency_versions(),
        "gpu_before": gpu_state_snapshot(torch.version.cuda),
        "model_files_sha256": {},
    }
    for path in sorted(Path(config["model_dir"]).rglob("*")):
        if path.is_file() and path.suffix in (".json", ".safetensors", ".model", ".txt"):
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    checksum.update(chunk)
            env["model_files_sha256"][path.relative_to(config["model_dir"]).as_posix()] = (
                checksum.hexdigest()
            )
    process = run_cancellable_process(
        [
            sys.executable,
            "/root/experiments/common_attention_contract_probe.py",
            str(root / "config.json"),
            str(root / "capture.json"),
        ],
        env=dict(os.environ, PYTHONPATH="/root"),
        stdout_path=root / "stdout.log",
        stderr_path=root / "stderr.log",
        cancelled=lambda: False,
        timeout_seconds=1800,
    )
    checkpoint = (root / "capture.json").read_text() if (root / "capture.json").exists() else None
    try:
        capture = (
            json.loads(checkpoint)
            if checkpoint
            else {"complete": False, "error": "missing checkpoint"}
        )
    except ValueError as exc:
        capture = {"complete": False, "error": str(exc)}
    payload = {
        "environment": env,
        "process": process,
        "capture": capture,
        "raw_checkpoint": checkpoint,
        "config": config,
    }
    try:
        env["gpu_after"] = gpu_state_snapshot(torch.version.cuda)
    except Exception as exc:
        env["metadata_error"] = str(exc)
    return payload


@app.function(
    image=image,
    gpu="L4",
    volumes={"/cache": volume, "/evidence": evidence_volume},
    timeout=7200,
    startup_timeout=180,
    max_containers=1,
    retries=0,
)
def campaign(campaign_id, fixture_bytes, expected):
    import re

    from cloud_engine.weights import ensure_weights_downloaded

    if re.fullmatch("[0-9a-f]{32}", campaign_id) is None:
        raise ValueError("invalid campaign ID")
    root = Path("/evidence") / campaign_id
    root.mkdir(exist_ok=False)
    index = {
        "campaign_id": campaign_id,
        "complete": False,
        "error": None,
        "artifacts": {},
        "gate_cleared": False,
        "scope": "installed attention primitive plus exposed failed pair09; no full calibration acceptance",
    }

    def commit():
        (root / "index.tmp").write_text(json.dumps(index, allow_nan=False))
        (root / "index.tmp").replace(root / "index.json")
        evidence_volume.commit()

    def retain(name, raw):
        path = root / name
        path.write_bytes(raw)
        index["artifacts"][name] = hashlib.sha256(raw).hexdigest()
        commit()

    retain("input.json", fixture_bytes)
    try:
        fixture = json.loads(fixture_bytes)
        check_fixture(fixture, expected)
        pinned = json.loads(Path("/root/engine_config.json").read_text())
        model = pinned["ragged_model"]
        if model != fixture["states"]["custom-batch"]["environment"]["model"]:
            raise ValueError("pinned model differs from historical model")
        model_dir = str(ensure_weights_downloaded("/cache", model["revision"], model["id"]))
        payload = child({"model_dir": model_dir}, fixture_bytes, model)
        retain("micro.json", json.dumps(payload, allow_nan=False).encode())
        validate_payload(
            payload, payload["config"], expected, fixture, hashlib.sha256(fixture_bytes).hexdigest()
        )
        from experiments.parallel_validation import CANCEL_STORE
        from fix_holdout_app import capture_pair

        store = modal.Dict.from_name(CANCEL_STORE, create_if_missing=True)
        if store.put(campaign_id, False, skip_if_exists=True) is not True:
            raise ValueError("campaign cancellation key already exists")
        index["active_stage"] = "failed-pair09-full128"
        commit()
        pair = capture_pair.local(
            "resource_normalized",
            9,
            fixture["failed_pair"]["workload"],
            campaign_id,
            expected["core"]["sha256"],
            None,
        )
        retain("pair.json", json.dumps(pair, allow_nan=False).encode())
        if (
            pair.get("stop") is not None
            or pair.get("error")
            or set(pair["children"]) != {"custom", "vllm"}
            or pair.get("summary", {}).get("passes") is not True
        ):
            raise ValueError("exact historical full128 pair failed; raw retained")
        for child_payload in pair["children"].values():
            environment = child_payload["environment"]
            if environment["computational_source"] != expected["core"]:
                raise ValueError("pair source differs from micro source")
            for key in ("model", "dependencies", "model_files_sha256"):
                if environment[key] != payload["environment"][key]:
                    raise ValueError("pair actual runtime/model differs from micro")
            for point in ("gpu_before", "gpu_after"):
                if environment[point]["uuid"] != payload["environment"]["gpu_before"]["uuid"]:
                    raise ValueError("pair physical GPU differs from micro")
        index["complete"] = True
    except BaseException as exc:
        index["error"] = f"{type(exc).__name__}: {exc}"
    commit()
    return index


def recover_artifacts(destination, campaign_id):
    from qkv_candidate_app import recover_artifacts as recover

    return recover(destination, campaign_id)


def main(output, fixture_path, recover=False):
    import uuid

    destination = Path(output)
    if recover:
        dispatch = json.loads((destination / "dispatch.json").read_text())
        call = modal.FunctionCall.from_id(dispatch["call_id"])
    else:
        if destination.exists():
            raise ValueError("refusing existing output or resubmission")
        fixture_bytes = Path(fixture_path).read_bytes()
        fixture = json.loads(fixture_bytes)
        expected = source(Path(__file__).resolve().parent)
        check_fixture(fixture, expected)
        verify_historical(fixture)
        destination.mkdir(parents=True)
        campaign_id = uuid.uuid4().hex
        with (destination / "intent.json").open("x") as stream:
            json.dump(
                {
                    "campaign_id": campaign_id,
                    "source": expected,
                    "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        with app.run(detach=True):
            if source(Path(__file__).resolve().parent) != expected:
                raise ValueError("source changed during image build")
            call = campaign.spawn(campaign_id, fixture_bytes, expected)
            dispatch = {
                "campaign_id": campaign_id,
                "call_id": call.object_id,
                "app_id": app.app_id,
                "source": expected,
            }
            with (destination / "dispatch.json").open("x") as stream:
                json.dump(dispatch, stream)
                stream.flush()
                os.fsync(stream.fileno())
    try:
        call.get()
    except BaseException:
        try:
            recover_artifacts(destination, dispatch["campaign_id"])
        except Exception as recovery_error:
            print(
                f"Recovery pending: {recovery_error}; recover saved call without resubmission",
                flush=True,
            )
        raise
    index = recover_artifacts(destination, dispatch["campaign_id"])
    if not index.get("complete") or index.get("error"):
        raise SystemExit("native replay incomplete; raw retained, no retry or gate clearance")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixture", default="attention-contract-input.json")
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    main(args.output, args.fixture, args.recover)
