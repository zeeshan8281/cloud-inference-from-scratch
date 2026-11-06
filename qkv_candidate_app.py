"""Full-window exposed trial of FP32 QKV computation; never fits holdout bounds."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import modal

CANDIDATE_FILES = ("qkv_policy_probe.py", "qkv_policy_worker.py", "qkv_head_policy.py", "fixed_qkv_microprobe.py",
                   "residual_vllm_probe.py", "residual_vllm_full_probe.py", "residual_vllm_full_hook.py",
                   "attention_replay_probe.py", "attention_replay_hook.py", "attention_replay.py", "unbiased_linear_microprobe.py", "fixed_head_microprobe.py")

if modal.is_local():
    from fix_holdout_app import capture_image
    candidate_image = capture_image
    for name in CANDIDATE_FILES:
        candidate_image = candidate_image.add_local_file(f"experiments/{name}", f"/root/experiments/{name}", copy=True)
    candidate_image = candidate_image.add_local_file("qkv_candidate_app.py", "/root/qkv_candidate_app_source.py", copy=True).add_local_file(
        "qkv_candidate_app.py", "/root/qkv_candidate_app.py", copy=True)
else:
    candidate_image = None

app = modal.App("cie-qkv-candidate")
volume = modal.Volume.from_name("cloud-inference-model-cache", create_if_missing=False)
evidence_volume = modal.Volume.from_name("cie-retained-diagnostics", create_if_missing=True)


def candidate_sources(root: Path, *, deployed: bool = False) -> dict:
    paths = {"experiments/" + name: root / "experiments" / name
             for name in CANDIDATE_FILES}
    paths["qkv_candidate_app_source.py"] = root / ("qkv_candidate_app_source.py" if deployed else "qkv_candidate_app.py")
    paths["qkv_candidate_app.py"] = root / "qkv_candidate_app.py"
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def validate_candidate_capture(capture: dict, candidate: str = "fp32_biased_qkv") -> None:
    if capture.get("config", {}).get("candidate") != candidate:
        raise ValueError("captured candidate differs from requested candidate")
    phases = list(capture.get("batch", {}).values())
    phases += [phase for solo in capture.get("solo", []) for phase in solo.values()]
    if not phases:
        raise ValueError("missing candidate phase evidence")
    for phase in phases:
        status = phase.get("qkv_policy", {})
        all_linears = candidate == "fixed_tile_linears_fp32_head"
        fixed = candidate in ("fixed_tile_qkv_fp32_head", "fixed_tile_linears_fp32_head")
        names, calls = status.get("modules", []), status.get("calls", {})
        count = status.get("expected_module_count")
        metadata = status.get("module_metadata", [])
        if (status.get("wrapper_active") is not True or status.get("candidate") != candidate or type(count) is not int or count <= 0
                or status.get("module_count") != count or len(names) != count
                or len(set(names)) != count or (not fixed and set(calls) != set(names))
                or (fixed and calls != {})
                or len(metadata) != count or {item.get("name") for item in metadata} != set(names)
                or any(item.get("weight_dtype") != "torch.float16" or item.get("bias_dtype") != (None if all_linears and not any(item.get("name", "").endswith(suffix) for suffix in ("q_proj", "k_proj", "v_proj", "qkv_proj")) else "torch.float16") for item in metadata)
                or any(type(value) is not int or value <= 0 for value in calls.values())
                or status.get("precision") != {"allow_fp16_reduced_precision_reduction": False,
                                               "allow_tf32": False, "float32_matmul_precision": "highest"}):
            raise ValueError("candidate actual module/arithmetic evidence missing or inconsistent")
        if fixed:
            expected_fixed = {"implementation": "fixed_tile_triton", "operator": "cloud_engine::diagnostic_fixed_tile_qkv",
                              "block_m": 16, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 2,
                              "input_dtype": "torch.float16", "weight_dtype": "torch.float16", "output_dtype": "torch.float16",
                              "accumulation_dtype": "torch.float32", "bias_add_dtype": "torch.float32",
                              "split_k": False, "autotune": False, "atomics": False, "operator_active": True}
            if all_linears:
                expected_fixed.update(operator="cloud_engine::diagnostic_fixed_tile_linear", bias_optional=True)
                expected_count = {"custom": 252, "vllm": 144}.get(capture["config"].get("implementation"))
                if count != expected_count:
                    raise ValueError("all-linear candidate module count mismatch")
            actual_fixed = status.get("fixed_tile_policy", {})
            if any(type(actual_fixed.get(key)) is not type(value) or actual_fixed[key] != value
                   for key, value in expected_fixed.items()):
                raise ValueError("fixed-tile operator installation/policy missing")
        if candidate in ("fp32_qkv_and_head", "fixed_tile_qkv_fp32_head", "fixed_tile_linears_fp32_head"):
            head = status.get("head", {})
            implementation = capture["config"].get("implementation")
            expected_head = {"custom": "custom_existing_fp32", "vllm": "vllm_fp32_output_projection"}.get(implementation)
            if (expected_head is None or head.get("enabled") is not True
                    or head.get("implementation") != expected_head or head.get("wrapper_active") is not True
                    or head.get("module_count") != 1 or head.get("expected_module_count") != 1
                    or head.get("weight_dtype") != "torch.float16"
                    or set(head.get("input_dtypes", [])) != {"torch.float16"}
                    or set(head.get("output_dtypes", [])) != {"torch.float32"}
                    or type(head.get("calls")) is not int or head["calls"] <= 0
                    or type(head.get("emitted_rows")) is not int or head["emitted_rows"] <= 0
                    or head.get("all_finite") is not True or head.get("autocast_enabled") is not False):
                raise ValueError("FP32 output-head evidence missing or inconsistent")
            if implementation == "vllm" and (head.get("head_class") not in {"VocabParallelEmbedding", "ParallelLMHead"}
                                               or type(head.get("tied_input_embedding")) is not bool):
                raise ValueError("missing actual output-projection class/tied identity evidence")
            if implementation == "vllm" and (type(head.get("native_counterfactual_calls")) is not int
                                               or head["native_counterfactual_calls"] <= 0):
                raise ValueError("missing same-hidden native-head counterfactual evidence")



def finish_gpu_metadata(environment, snapshot, cuda_version, finished_utc):
    """A failed post-run GPU query must not discard an already completed child."""
    environment['finished_utc'] = finished_utc
    try:
        environment['gpu_after'] = snapshot(cuda_version)
    except Exception as exc:
        environment['gpu_after'] = {'uuid': None}
        environment['environment_error'] = f'{type(exc).__name__}: {exc}'


@app.function(image=candidate_image, gpu="L4", volumes={"/cache": volume}, timeout=1800,
              startup_timeout=180, max_containers=1, scaledown_window=300)
def capture(implementation: str, mode: str, workload: dict, candidate: str = "fp32_biased_qkv",
            trace_case: str | None = None, attention_replay: bool = False) -> dict:
    import sys
    from datetime import datetime, timezone

    import torch

    from cloud_engine.weights import ensure_weights_downloaded
    from experiments.fix_holdout import computational_source
    from experiments.sentinel_pilot import gpu_state_snapshot, resolved_dependency_versions

    if candidate not in ("fp32_biased_qkv", "fp32_qkv_and_head", "fixed_tile_qkv_fp32_head", "fixed_tile_linears_fp32_head"):
        raise ValueError("unknown diagnostic candidate")
    pinned = json.loads(Path("/root/engine_config.json").read_text())
    model = pinned["ragged_model"]
    model_dir = str(ensure_weights_downloaded("/cache", model["revision"], model["id"]))
    root = Path(tempfile.mkdtemp(prefix="qkv-candidate-"))
    config = {"implementation": implementation, "mode": mode, "model": model, "model_dir": model_dir,
              "kv_bytes": pinned["kv_cache"]["bytes"], "workload": workload, "candidate": candidate}
    script = "qkv_policy_probe.py"
    if attention_replay and trace_case is None:
        raise ValueError("attention replay requires explicit full-window trace/control case")
    if trace_case is not None:
        if (trace_case not in ("control", "trace", "repeat") or implementation != "vllm"
                or mode != "resource_normalized" or candidate != "fixed_tile_qkv_fp32_head"):
            raise ValueError("unsupported full-window trace case")
        config.update(trace_case=trace_case, target_index=2, output_tokens=128)
        if attention_replay:
            config["attention_probe"] = True
        script = "attention_replay_probe.py" if attention_replay else "residual_vllm_probe.py"
    (root / "config.json").write_text(json.dumps(config))
    model_files = {}
    for model_path in sorted(Path(model_dir).rglob("*")):
        if model_path.is_file() and model_path.suffix in (".json", ".safetensors", ".model", ".txt"):
            checksum = hashlib.sha256()
            with model_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    checksum.update(chunk)
            model_files[model_path.relative_to(model_dir).as_posix()] = checksum.hexdigest()
    environment = {"candidate": candidate, "attention_replay": attention_replay, "source": os.environ.get("SOURCE_COMMIT"), "started_utc": datetime.now(timezone.utc).isoformat(),
                   "gpu_before": gpu_state_snapshot(torch.version.cuda), "dependencies": resolved_dependency_versions(),
                   "model": model, "model_files_sha256": model_files,
                   "computational_source": computational_source(Path("/root"), deployed=True),
                   "candidate_sources": candidate_sources(Path("/root"), deployed=True)}
    try:
        completed = subprocess.run([sys.executable, "/root/experiments/" + script, str(root / "config.json"),
                                    str(root / "capture.json")], capture_output=True, text=True, timeout=1500,
                                   env=dict(os.environ, PYTHONPATH="/root"))
        process = {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    except subprocess.TimeoutExpired as exc:
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value
        process = {"returncode": None, "timeout": True, "stdout": decode(exc.stdout), "stderr": decode(exc.stderr)}
    finish_gpu_metadata(environment, gpu_state_snapshot, torch.version.cuda, datetime.now(timezone.utc).isoformat())
    checkpoint = (root / "capture.json").read_text() if (root / "capture.json").exists() else None
    try:
        result = json.loads(checkpoint) if checkpoint is not None else {"complete": False, "error": "missing checkpoint"}
        if not isinstance(result, dict):
            raise ValueError("checkpoint is not an object")
    except (json.JSONDecodeError, ValueError) as exc:
        result = {"complete": False, "error": f"invalid checkpoint: {exc}"}
    traces = {path.name: path.read_text() for path in root.glob("*.jsonl")} if trace_case is not None else {}
    return {"environment": environment, "process": process, "capture": result, "raw_checkpoint": checkpoint,
            "raw_traces": traces}


@app.local_entrypoint()
def run(output: str, manifest: str = "experiments/fix-validation-smoke-01/manifest.json", candidate: str = "fp32_biased_qkv") -> None:
    from experiments.fix_holdout import (
        MODES,
        computational_source,
        digest,
        summarize,
        validate_manifest,
    )

    if candidate not in ("fp32_biased_qkv", "fp32_qkv_and_head", "fixed_tile_qkv_fp32_head", "fixed_tile_linears_fp32_head"):
        raise SystemExit("unknown diagnostic candidate")
    destination = Path(output)
    if destination.exists():
        raise SystemExit("refusing to overwrite candidate evidence")
    source = json.loads(Path(manifest).read_text())
    if (source.get("split") != "calibration" or source.get("namespace") != "protocol-v2-calibration"
            or source.get("workloads_sha256") != digest(source.get("workloads"))):
        raise SystemExit("candidate requires intact exposed calibration inputs")
    validate_manifest(source)
    destination.mkdir(parents=True)
    (destination / "manifest.json").write_text(json.dumps(source))
    repo = Path(__file__).resolve().parent
    expected_source = digest({"core": computational_source(repo), "candidate": candidate_sources(repo)})
    gpu_ids, executed_sources, runtimes = set(), set(), set()
    for mode in MODES:
        workload = next(item for item in source["workloads"][mode] if item["concurrency"] == 8 and item["batch_index"] == 0)
        captures = {}
        for implementation in ("custom", "vllm"):
            payload = capture.remote(implementation, mode, workload, candidate)
            (destination / f"{mode}-{implementation}.json").write_text(json.dumps(payload, allow_nan=False))
            env = payload["environment"]
            gpu_ids.update((env["gpu_before"]["uuid"], env["gpu_after"]["uuid"]))
            executed_sources.add(digest({"core": env["computational_source"], "candidate": env["candidate_sources"]}))
            runtimes.add(digest({key: env[key] for key in ("model", "model_files_sha256", "dependencies")}))
            if (payload["process"]["returncode"] != 0 or not payload["capture"].get("complete")
                    or payload["capture"].get("error") is not None or env.get("candidate") != candidate
                    or env.get("environment_error") is not None
                    or len(gpu_ids) != 1 or executed_sources != {expected_source} or len(runtimes) != 1):
                raise SystemExit("candidate capture failed; raw evidence retained")
            try:
                validate_candidate_capture(payload["capture"], candidate)
            except (ValueError, TypeError) as exc:
                raise SystemExit(f"candidate policy unresolved; raw evidence retained: {exc}") from exc
            captures[f"{mode}/{implementation}/0"] = payload["capture"]
        subset = {"workloads": {item: [workload] if item == mode else [] for item in MODES}}
        summary = summarize(subset, captures, partial=True)
        summary.update(evaluation_kind="exposed_fp32_qkv_candidate", candidate=candidate, calibration_maxima=None, gate_cleared=False)
        (destination / f"{mode}-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        if not summary["passes"]:
            raise SystemExit("candidate failed exact full-window comparison; no bounds fitted")
    print("Candidate passes exposed subset only; full calibration/holdout/pilot remain required.")


@app.function(image=candidate_image, gpu="L4", volumes={"/cache": volume}, timeout=1800,
              startup_timeout=180, max_containers=1, scaledown_window=300)
def capture_micro(fixture_text: str, candidate: str = "fixed_tile_qkv_fp32_head") -> dict:
    import sys
    from datetime import datetime, timezone

    import torch

    from cloud_engine.weights import ensure_weights_downloaded
    from experiments.fix_holdout import computational_source
    from experiments.sentinel_pilot import gpu_state_snapshot, resolved_dependency_versions

    if candidate not in ("fp32_biased_qkv", "fp32_qkv_and_head", "fixed_tile_qkv_fp32_head", "fixed_tile_linears_fp32_head", "fixed_tile_fp32_head_microprobe"):
        raise ValueError("unknown diagnostic candidate")
    pinned = json.loads(Path("/root/engine_config.json").read_text())
    model = pinned["ragged_model"]
    model_dir = str(ensure_weights_downloaded("/cache", model["revision"], model["id"]))
    root = Path(tempfile.mkdtemp(prefix="qkv-candidate-"))
    (root / "fixture.json").write_text(fixture_text)
    config = {"model_dir": model_dir, "fixture_path": str(root / "fixture.json")}
    (root / "config.json").write_text(json.dumps(config))
    model_files = {}
    for model_path in sorted(Path(model_dir).rglob("*")):
        if model_path.is_file() and model_path.suffix in (".json", ".safetensors", ".model", ".txt"):
            checksum = hashlib.sha256()
            with model_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    checksum.update(chunk)
            model_files[model_path.relative_to(model_dir).as_posix()] = checksum.hexdigest()
    environment = {"candidate": candidate, "source": os.environ.get("SOURCE_COMMIT"), "started_utc": datetime.now(timezone.utc).isoformat(),
                   "gpu_before": gpu_state_snapshot(torch.version.cuda), "dependencies": resolved_dependency_versions(),
                   "model": model, "model_files_sha256": model_files,
                   "computational_source": computational_source(Path("/root"), deployed=True),
                   "candidate_sources": candidate_sources(Path("/root"), deployed=True)}
    try:
        script = ("fixed_head_microprobe.py" if candidate == "fixed_tile_fp32_head_microprobe" else
                  "unbiased_linear_microprobe.py" if candidate == "fixed_tile_linears_fp32_head" else "fixed_qkv_microprobe.py")
        completed = subprocess.run([sys.executable, "/root/experiments/" + script, str(root / "config.json"),
                                    str(root / "capture.json")], capture_output=True, text=True, timeout=1500,
                                   env=dict(os.environ, PYTHONPATH="/root"))
        process = {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    except subprocess.TimeoutExpired as exc:
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value
        process = {"returncode": None, "timeout": True, "stdout": decode(exc.stdout), "stderr": decode(exc.stderr)}
    finish_gpu_metadata(environment, gpu_state_snapshot, torch.version.cuda, datetime.now(timezone.utc).isoformat())
    checkpoint = (root / "capture.json").read_text() if (root / "capture.json").exists() else None
    try:
        result = json.loads(checkpoint) if checkpoint is not None else {"complete": False, "error": "missing checkpoint"}
        if not isinstance(result, dict):
            raise ValueError("checkpoint is not an object")
    except (json.JSONDecodeError, ValueError) as exc:
        result = {"complete": False, "error": f"invalid checkpoint: {exc}"}
    return {"environment": environment, "process": process, "capture": result, "raw_checkpoint": checkpoint}



def validate_fixed_payload(payload, expected_source, gpu_ids, runtimes, candidate="fixed_tile_qkv_fp32_head"):
    from experiments.fix_holdout import digest
    env = payload["environment"]
    gpu_ids.update((env["gpu_before"].get("uuid"), env["gpu_after"].get("uuid")))
    actual = digest({"core": env["computational_source"], "candidate": env["candidate_sources"]})
    runtimes.add(digest({key: env[key] for key in ("model", "model_files_sha256", "dependencies")}))
    if (payload["process"].get("returncode") != 0 or payload["capture"].get("complete") is not True
            or payload["capture"].get("error") is not None or env.get("environment_error") is not None
            or env.get("candidate") != candidate or actual != expected_source
            or None in gpu_ids or len(gpu_ids) != 1 or len(runtimes) != 1):
        raise ValueError("fixed candidate process/provenance failure")


def assess_micro_accuracy(result):
    """Development microprobe accuracy check, not a holdout logprob epsilon."""
    import base64
    import math
    import struct
    projections = result.get("projections", {})
    if set(projections) != {"q", "k", "v"}:
        raise ValueError("microprobe requires all Q/K/V projections")
    def values(encoded, width, dtype):
        if encoded.get("shape") != [width] or encoded.get("dtype") != dtype:
            raise ValueError("invalid microprobe vector shape/dtype")
        raw = base64.b64decode(encoded["bytes_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != encoded.get("sha256"):
            raise ValueError("microprobe vector checksum mismatch")
        decoded = struct.unpack("<" + ("e" if dtype == "torch.float16" else "d") * width, raw)
        if not all(math.isfinite(value) for value in decoded):
            raise ValueError("nonfinite microprobe vector")
        return decoded
    audit = {"rule": "candidate worst max_abs FP64 error <= baseline FP32-to-FP16 worst max_abs FP64 error on identical layouts",
             "added_tolerance": 0, "scope": "exposed development microprobe; not holdout epsilon", "projections": {}}
    layout_keys = None
    for name, width in (("q", 2048), ("k", 256), ("v", 256)):
        projection = projections[name]
        reference = values(projection["fp64_reference"], width, "torch.float64")
        layouts = projection.get("layouts", [])
        keys = [(row.get("m"), row.get("offset")) for row in layouts]
        required = {(m, offset) for m in (1, 8, 16, 128, 256, 2048) for offset in (0, m - 1)}
        if (not keys or len(set(keys)) != len(keys) or not required <= set(keys)
                or any(type(m) is not int or type(offset) is not int or not 0 <= offset < m for m, offset in keys)
                or (layout_keys is not None and keys != layout_keys)):
            raise ValueError("incomplete or inconsistent microprobe layouts")
        layout_keys = keys
        errors = {"baseline": [], "candidate": []}
        for layout in layouts:
            for kind in errors:
                actual = values(layout[kind], width, "torch.float16")
                error = max(abs(a - b) for a, b in zip(actual, reference, strict=True))
                recorded = layout.get(kind + "_fp64_error", {}).get("max_abs")
                if type(recorded) not in (int, float) or not math.isfinite(recorded) or recorded != error:
                    raise ValueError("invalid or inconsistent microprobe FP64 error")
                errors[kind].append(error)
        baseline, candidate = max(errors["baseline"]), max(errors["candidate"])
        audit["projections"][name] = {"baseline_worst_max_abs": baseline, "candidate_worst_max_abs": candidate,
                                      "layout_count": len(keys), "passes": candidate <= baseline}
    audit["passes"] = all(row["passes"] for row in audit["projections"].values())
    return audit


def validate_micro(result, fixture_sha256):
    if result.get("precision") != {"allow_fp16_reduced_precision_reduction": False, "allow_tf32": False,
                                   "float32_matmul_precision": "highest"}:
        raise ValueError("microprobe actual precision policy missing")
    if (result.get("schema") != "fixed-qkv-microprobe-v1" or result.get("fixture_sha256") != fixture_sha256
            or any(result.get(key) is not True for key in
                   ("baseline_reproduces_capture", "candidate_layouts_exact", "fused_separate_exact"))):
        raise ValueError("fixed QKV microprobe invariance/reproduction failure")
    if not assess_micro_accuracy(result)["passes"]:
        raise ValueError("fixed QKV FP64 accuracy worse than baseline")
    execution = result.get("execution", [])
    if (len(execution) != 2 or {row.get("m") for row in execution} != {16, 256}
            or any(row.get("compiled_fullgraph_exact") is not True or row.get("cuda_graph_replay_exact") is not True
                   for row in execution)):
        raise ValueError("fixed QKV compiled/graph execution failure")


def assess_unbiased_micro(result, fixture):
    """Recompute finite FP64 errors from retained vectors, with no added tolerance."""
    import base64
    import math
    import struct
    cases = {row['name']: row for row in fixture['cases']}
    operators = result.get('operators', {})
    if set(cases) != {'down_proj', 'o_proj'} or set(operators) != set(cases):
        raise ValueError('missing captured unbiased operators')
    def values(encoded, dtype):
        if encoded.get('shape') != [2048] or encoded.get('dtype') != dtype:
            raise ValueError('invalid unbiased micro vector shape/dtype')
        raw = base64.b64decode(encoded['bytes_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != encoded.get('sha256'):
            raise ValueError('unbiased micro vector checksum mismatch')
        vector = struct.unpack('<'+('e' if dtype == 'torch.float16' else 'd')*2048, raw)
        if not all(math.isfinite(value) for value in vector):
            raise ValueError('nonfinite unbiased micro vector')
        return vector
    audit = {'added_tolerance': 0, 'operators': {}, 'scope': 'exposed development accuracy only; not holdout epsilon'}
    for name, case in cases.items():
        operator = operators[name]
        if operator.get('weight_name') != case['weight_name']:
            raise ValueError('unbiased micro weight identity mismatch')
        reference = values(operator['fp64_reference'], 'torch.float64')
        layouts = operator.get('layouts', [])
        keys = [(row.get('m'), row.get('offset')) for row in layouts]
        required = {(m, offset) for m in (1, 8, 16, 128, 256, 2048) for offset in (0, m-1)}
        required.update((row['M'], row['target_offset']) for row in case['states'])
        if (not required <= set(keys) or len(keys) != len(set(keys))
                or any(type(m) is not int or type(offset) is not int or not 0 <= offset < m for m, offset in keys)):
            raise ValueError('unbiased micro missing captured/required layouts')
        if operator.get('all_layouts_exact') is not True or len({row['candidate']['sha256'] for row in layouts}) != 1 or any(row.get('repeat_exact') is not True for row in layouts):
            raise ValueError('unbiased micro layout/repeat invariance failure')
        matches = operator.get('baseline_native_matches', {})
        if set(matches) != {state['phase'] for state in case['states']} or any(row.get('exact') is not True for row in matches.values()):
            raise ValueError('unbiased micro captured baseline reproduction missing')
        errors = {'baseline': [], 'candidate': []}
        for row in layouts:
            for kind in errors:
                vector = values(row[kind], 'torch.float16')
                error = max(abs(a-b) for a,b in zip(vector, reference, strict=True))
                if row.get(kind+'_fp64_error', {}).get('max_abs') != error:
                    raise ValueError('unbiased micro recorded error mismatch')
                errors[kind].append(error)
        baseline, candidate = max(errors['baseline']), max(errors['candidate'])
        audit['operators'][name] = {'baseline_worst_max_abs': baseline, 'candidate_worst_max_abs': candidate,
                                    'passes': candidate <= baseline}
    audit['passes'] = all(row['passes'] for row in audit['operators'].values())
    return audit


def validate_unbiased_micro(result, fixture_sha256, audit):
    if (result.get('schema') != 'fixed-unbiased-linear-microprobe-v1'
            or result.get('fixture_sha256') != fixture_sha256
            or result.get('precision') != {'allow_fp16_reduced_precision_reduction': False,
                                          'allow_tf32': False, 'float32_matmul_precision': 'highest'}
            or any(result.get(key) is not True for key in ('baseline_reproduces_capture', 'candidate_layouts_exact',
                                                          'accuracy_no_worse', 'execution_exact'))
            or audit.get('passes') is not True):
        raise ValueError('unbiased micro reproduction/invariance/accuracy failure')
    for operator in result['operators'].values():
        execution = operator.get('execution', [])
        if (len(execution) != 2 or {row.get('m') for row in execution} != {16, 256} or any(row.get('compiled_fullgraph_exact') is not True
                                 or row.get('cuda_graph_replay_exact') is not True for row in execution)):
            raise ValueError('unbiased micro compiled/graph execution failure')


def assess_head_micro(result, fixture):
    """Recheck historical score subset and new full-vocabulary FP64 accuracy."""
    import base64
    import math
    import struct
    width = fixture['pinned_model_contract']['original_vocab_size']
    if width != 151936 or fixture['pinned_model_contract']['hidden_size'] != 2048:
        raise ValueError('unsupported pinned head dimensions')
    def decode(encoded, size, dtype):
        if encoded.get('shape') != [size] or encoded.get('dtype') != dtype:
            raise ValueError('head vector shape/dtype mismatch')
        raw = base64.b64decode(encoded['bytes_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != encoded.get('sha256'):
            raise ValueError('head vector checksum mismatch')
        values = struct.unpack('<'+('f' if dtype == 'torch.float32' else 'd')*size, raw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('nonfinite head vector')
        return values
    states = {state['selection']: state for state in fixture['states']}
    history = result.get('historical_reproduction', [])
    if len(states) != 2 or set(states) != {'batch', 'solo'} or len(history) != 2 or {row.get('selection') for row in history} != set(states):
        raise ValueError('missing historical head score reproduction')
    for row in history:
        state = states[row['selection']]
        ids = state['observed_token_ids']
        if (len(ids) != 20 or len(set(ids)) != 20 or any(type(token) is not int or not 0 <= token < width for token in ids)
                or row.get('token_ids') != ids or row.get('m') != state['M'] or row.get('offset') != state['projection_row']
                or row.get('exact') is not True):
            raise ValueError('historical head layout/score identity mismatch')
        actual = decode(state['observed_fp32_logits'], 20, 'torch.float32')
        if decode(row['captured'], 20, 'torch.float32') != actual or decode(row['replay'], 20, 'torch.float32') != actual:
            raise ValueError('native replay does not reproduce historical twenty scores')
    reference = decode(result['full_fp64_reference'], width, 'torch.float64')
    layouts = result.get('layouts', [])
    keys = [(row.get('m'), row.get('offset')) for row in layouts]
    required = {(m, offset) for m in (1, 5, 8, 16, 32) for offset in (0, m-1)} | {(5, 2)}
    if len(keys) != len(set(keys)) or set(keys) != required:
        raise ValueError('missing head layouts')
    errors = {'baseline': [], 'candidate': []}
    hashes = set()
    for row in layouts:
        if row.get('repeat_exact') is not True:
            raise ValueError('head repeat differs')
        hashes.add(row['candidate']['sha256'])
        for kind in errors:
            values = decode(row[kind], width, 'torch.float32')
            if kind == 'baseline':
                for state in states.values():
                    if (row['m'], row['offset']) == (state['M'], state['projection_row']):
                        observed = decode(state['observed_fp32_logits'], 20, 'torch.float32')
                        if tuple(values[token] for token in state['observed_token_ids']) != observed:
                            raise ValueError('full native baseline differs from historical twenty-score subset')
            error = max(abs(a-b) for a,b in zip(values, reference, strict=True))
            if row.get(kind+'_fp64_error', {}).get('max_abs') != error:
                raise ValueError('head FP64 error does not match retained vector')
            errors[kind].append(error)
    if len(hashes) != 1:
        raise ValueError('head full FP32 output differs across layouts')
    baseline, candidate = max(errors['baseline']), max(errors['candidate'])
    return {'passes': candidate <= baseline, 'baseline_worst_max_abs': baseline, 'candidate_worst_max_abs': candidate,
            'added_tolerance': 0, 'historical_score_count_per_layout': 20, 'new_reference_vocabulary_size': width,
            'scope': 'historical twenty-score reproduction; newly computed full-vocabulary accuracy, not historical full-vocabulary reproduction'}


def validate_head_micro(result, fixture_sha256, audit):
    if (result.get('schema') != 'fixed-fp32-head-microprobe-v1' or result.get('fixture_sha256') != fixture_sha256
            or result.get('precision') != {'allow_fp16_reduced_precision_reduction': False,
                                          'allow_tf32': False, 'float32_matmul_precision': 'highest'}
            or any(result.get(key) is not True for key in ('baseline_reproduces_capture', 'candidate_layouts_exact',
                                                          'accuracy_no_worse', 'execution_exact'))
            or audit.get('passes') is not True):
        raise ValueError('head reproduction/invariance/accuracy failure')
    execution = result.get('execution', [])
    if (len(execution) != 3 or {row.get('m') for row in execution} != {1, 5, 32}
            or any(row.get('compiled_fullgraph_exact') is not True or row.get('cuda_graph_replay_exact') is not True
                   for row in execution)):
        raise ValueError('head compiled/graph evidence incomplete')


@app.function(image=candidate_image, gpu="L4", volumes={"/cache": volume, "/evidence": evidence_volume},
              timeout=10800, startup_timeout=180, max_containers=1, retries=0)
def fixed_campaign(campaign_id: str, workloads: dict, fixture_text: str, expected_source: str, linears_only: bool = False, head_only: bool = False) -> dict:
    import re

    from fix_holdout_app_source import _child_hard_failures

    from experiments.fix_holdout import MODES, summarize
    if re.fullmatch(r"[0-9a-f]{32}", campaign_id) is None:
        raise ValueError("invalid campaign ID")
    root = Path("/evidence") / campaign_id
    root.mkdir(exist_ok=False)
    if linears_only and head_only:
        raise ValueError("choose one diagnostic scope")
    candidate = ("fixed_tile_fp32_head_microprobe" if head_only else
                 "fixed_tile_linears_fp32_head" if linears_only else "fixed_tile_qkv_fp32_head")
    index = {"campaign_id": campaign_id, "complete": False, "error": None, "artifacts": {},
             "candidate": candidate, "gate_cleared": False, "linears_only": linears_only, "head_only": head_only,
             "scope": "actual16 head microprobe only; no fullmodel or gate claim" if head_only else
                      "unbiased microprobe then custom-only full128 both modes" if linears_only else "QKV microprobe then paired full128 both modes"}
    def commit():
        temporary = root / "index.tmp"
        temporary.write_text(json.dumps(index, allow_nan=False))
        temporary.replace(root / "index.json")
        evidence_volume.commit()
    def retain(name, value, raw=False):
        path = root / name
        path.write_text(value if raw else json.dumps(value, allow_nan=False))
        index["artifacts"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        commit()
    retain("fixture.json", fixture_text, raw=True)
    retain("workloads.json", workloads)
    gpu_ids, runtimes = set(), set()
    index["active_stage"] = "microprobe"
    commit()
    try:
        payload = capture_micro.local(fixture_text, candidate) if (linears_only or head_only) else capture_micro.local(fixture_text)
        retain("microprobe.json", payload)
        validate_fixed_payload(payload, expected_source, gpu_ids, runtimes, **({"candidate": candidate} if (linears_only or head_only) else {}))
        if head_only:
            audit = assess_head_micro(payload["capture"], json.loads(fixture_text))
            retain("micro-accuracy.json", audit)
            validate_head_micro(payload["capture"], hashlib.sha256(fixture_text.encode()).hexdigest(), audit)
            index["complete"] = True
            commit()
            return index
        if linears_only:
            audit = assess_unbiased_micro(payload["capture"], json.loads(fixture_text))
            retain("micro-accuracy.json", audit)
            validate_unbiased_micro(payload["capture"], hashlib.sha256(fixture_text.encode()).hexdigest(), audit)
        else:
            retain("micro-accuracy.json", assess_micro_accuracy(payload["capture"]))
            validate_micro(payload["capture"], hashlib.sha256(fixture_text.encode()).hexdigest())
        for mode in MODES:
            workload = workloads[mode]
            captures = {}
            for implementation in (("custom",) if linears_only else ("custom", "vllm")):
                index["active_stage"] = f"{mode}-{implementation}"
                commit()
                payload = capture.local(implementation, mode, workload, candidate)
                retain(f"{mode}-{implementation}.json", payload)
                validate_fixed_payload(payload, expected_source, gpu_ids, runtimes, **({"candidate": candidate} if (linears_only or head_only) else {}))
                validate_candidate_capture(payload["capture"], candidate)
                failures = _child_hard_failures(mode, implementation, workload, payload["capture"], None)
                if failures:
                    retain(f"{mode}-{implementation}-failures.json", failures)
                    raise ValueError("fixed candidate intra-engine failure")
                captures[f"{mode}/{implementation}/0"] = payload["capture"]
            if linears_only:
                retain(f"{mode}-summary.json", {"passes": True, "scope": "custom full128 batch/solo only; no cross-engine comparison", "gate_cleared": False})
                continue
            subset = {"workloads": {item: [workload] if item == mode else [] for item in MODES}}
            summary = summarize(subset, captures, partial=True)
            summary.update(evaluation_kind="exposed_fixed_tile_candidate", calibration_maxima=None, gate_cleared=False)
            retain(f"{mode}-summary.json", summary)
            if not summary["passes"]:
                raise ValueError("fixed candidate paired numerical failure")
        index["complete"] = True
    except BaseException as exc:
        index["error"] = f"{type(exc).__name__}: {exc}"
    commit()
    return index



def validate_trace_capture(result, workload):
    from experiments.fix_holdout import validate_capture, validate_kv_capacity
    if set(result.get("cases", {})) != {"batch", "solo"}:
        raise ValueError("missing full128 trace/control cases")
    batch, solo = (result["cases"][key]["capture"] for key in ("batch", "solo"))
    validate_candidate_capture({"config": result["config"], "batch": {"unique": batch},
                                "solo": [{"unique": solo}]}, "fixed_tile_qkv_fp32_head")
    validate_kv_capacity(result["kv_capacity"], "resource_normalized", result["kv_plan"])
    for phase, inputs in ((batch, workload["input_token_ids"]), (solo, [workload["input_token_ids"][2]])):
        validate_capture(phase, inputs, 128, cold=True, mode="resource_normalized")
        if not phase["identity"].get("forward_traces"):
            raise ValueError("missing full realized schedule")


def validate_attention_capture(payload, case):
    """Bind bounded replay evidence to retained raw bytes before interpretation."""
    result = payload["capture"]
    if case not in {"control", "trace", "repeat"} or result.get("config", {}).get("trace_case") != case:
        raise ValueError("attention trace/control identity mismatch")
    for selection in ("batch", "solo"):
        row = result["cases"][selection]
        calls = row.get("attention_calls")
        if not isinstance(calls, list) or len(calls) != int(case != "control"):
            raise ValueError("wrong bounded attention capture count")
        name = Path(row.get("attention_path", "")).name
        raw = payload.get("raw_traces", {}).get(name, "")
        if (not name or hashlib.sha256(raw.encode()).hexdigest() != row.get("attention_sha256")
                or [json.loads(line) for line in raw.splitlines() if line] != calls):
            raise ValueError("attention records do not match retained raw trace")
        if not calls:
            continue
        record = calls[0]
        required = ("past_kv_unchanged", "new_kv_matches_input", "native_replay_exact",
                    "full_replay_repeat_exact", "counterfactual_executed", "target_inputs_fixed",
                    "singleton_repeat_exact", "singleton_preserve_global_max_repeat_exact", "singleton_max2_repeat_exact")
        if any(record.get(key) is not True for key in required):
            raise ValueError("attention native replay/integrity prerequisite failed")
        target = row["target_local_index"]
        target_record = row["capture"]["records"][target]
        tokens = target_record["output_tokens"]
        if (record.get("layer_index") != 0 or record.get("target_position") != 512
                or record.get("computed_tokens") != 512 or record.get("output_tokens_before") != tokens[:1]
                or not target_record.get("request_id") or record.get("request_id") != target_record["request_id"]
                or record.get("actual_fa_version") != 2 or record.get("unused_page_contents_serialized") is not False):
            raise ValueError("attention evidence attributed to wrong layer or token")


def trace_control_signature(result):
    # Physical KV allocation is retained separately; this compares logical work.
    keys = ("req_ids", "query_lengths", "query_starts", "computed_tokens", "packed_input_sha256")
    return {selection: {
        "outputs": [record["output_tokens"] for record in row["capture"]["records"]],
        "schedule": [{key: step[key] for key in keys} for step in row["capture"]["identity"]["forward_traces"]]}
        for selection, row in result["cases"].items()}


@app.function(image=candidate_image, gpu="L4", volumes={"/cache": volume, "/evidence": evidence_volume},
              timeout=10800, startup_timeout=180, max_containers=1, retries=0)
def fixed_followup_campaign(campaign_id: str, workloads: dict, expected_source: str, attention_replay: bool = False) -> dict:
    """Exposed diagnostic only: retain numerical failures and localize the residual."""
    import re

    from fix_holdout_app_source import _child_hard_failures

    if re.fullmatch(r"[0-9a-f]{32}", campaign_id) is None:
        raise ValueError("invalid campaign ID")
    root = Path("/evidence") / campaign_id
    root.mkdir(exist_ok=False)
    index = {"campaign_id": campaign_id, "complete": False, "error": None, "artifacts": {},
             "candidate": "fixed_tile_qkv_fp32_head", "gate_cleared": False,
             "attention_replay": attention_replay,
             "scope": ("exposed full128 vLLM control/trace/repeat with bounded actual KV and attention replay" if attention_replay else
                       "exposed complete-custom and full128 vLLM observer-control diagnostic")}
    def commit():
        temporary = root / "index.tmp"
        temporary.write_text(json.dumps(index, allow_nan=False))
        temporary.replace(root / "index.json")
        evidence_volume.commit()
    def retain(name, value):
        path = root / name
        path.write_text(json.dumps(value, allow_nan=False))
        index["artifacts"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        commit()
    retain("workloads.json", workloads)
    gpu_ids, runtimes = set(), set()
    try:
        if not attention_replay:
            index["active_stage"] = "complete_system-custom"
            commit()
            payload = capture.local("custom", "complete_system", workloads["complete_system"], "fixed_tile_qkv_fp32_head")
            retain("complete_system-custom.json", payload)
            validate_fixed_payload(payload, expected_source, gpu_ids, runtimes)
            validate_candidate_capture(payload["capture"], "fixed_tile_qkv_fp32_head")
            retain("complete_system-custom-failures.json", _child_hard_failures(
                "complete_system", "custom", workloads["complete_system"], payload["capture"], None))
        control = None
        for case in ("control", "trace", "repeat"):
            index["active_stage"] = "vllm-" + case
            commit()
            payload = capture.local("vllm", "resource_normalized", workloads["resource_normalized"],
                                    "fixed_tile_qkv_fp32_head", trace_case=case, **({"attention_replay": True} if attention_replay else {}))
            retain("vllm-" + case + ".json", payload)
            validate_fixed_payload(payload, expected_source, gpu_ids, runtimes)
            validate_trace_capture(payload["capture"], workloads["resource_normalized"])
            if attention_replay:
                if payload["environment"].get("attention_replay") is not True or payload["capture"].get("config", {}).get("attention_probe") is not True:
                    raise ValueError("missing requested attention replay identity")
                validate_attention_capture(payload, case)
            signature = trace_control_signature(payload["capture"])
            if control is None:
                control = signature
            elif signature != control:
                raise ValueError("observer changed full128 outputs or realized logical schedule")
        retain("control-summary.json", {"full128_outputs_and_schedules_equal": True,
                                        "gate_cleared": False, "scope": index["scope"]})
        index["complete"] = True
    except BaseException as exc:
        index["error"] = f"{type(exc).__name__}: {exc}"
    commit()
    return index


def recover_artifacts(destination: Path, campaign_id: str) -> dict:
    """Read the committed index; never starts or retries scientific work."""
    raw = b"".join(evidence_volume.read_file(f"{campaign_id}/index.json"))
    index = json.loads(raw)
    for name, expected in index["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("invalid retained artifact path")
        data = b"".join(evidence_volume.read_file(f"{campaign_id}/{name}"))
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("retained artifact checksum mismatch")
        target = destination / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError("refusing to overwrite different local evidence")
        target.write_bytes(data)
    (destination / "remote-index.json").write_bytes(raw)
    return index



def durable_fixed(output: str, manifest: str, fixture: str, *, recover: bool = False, followup: bool = False, attention_followup: bool = False, linears_only: bool = False, head_only: bool = False) -> None:
    import uuid

    from experiments.fix_holdout import computational_source, digest, validate_manifest
    destination = Path(output)
    dispatch_file = destination / "dispatch.json"
    if recover:
        dispatch = json.loads(dispatch_file.read_text())
        call = modal.FunctionCall.from_id(dispatch["call_id"])
    else:
        if sum((followup, attention_followup, linears_only, head_only)) > 1:
            raise ValueError("choose one followup selector")
        if destination.exists():
            raise SystemExit("refusing to overwrite or resubmit campaign")
        source = json.loads(Path(manifest).read_text())
        if (source.get("split") != "calibration" or source.get("namespace") != "protocol-v2-calibration"
                or source.get("workloads_sha256") != digest(source.get("workloads"))):
            raise SystemExit("requires intact exposed calibration manifest")
        validate_manifest(source)
        workloads = {mode: next(row for row in rows if row["concurrency"] == 8 and row["batch_index"] == 0)
                     for mode, rows in source["workloads"].items()}
        fixture_text = None
        if not attention_followup:
            fixture_text = Path(fixture).read_text()
            expected_schema = ("captured-fp32-head-replay-input-v1" if head_only else
                               "captured-unbiased-linear-replay-input-v1" if linears_only else "captured-qkv-replay-input-v1")
            if json.loads(fixture_text).get("schema") != expected_schema:
                raise SystemExit("requires actual captured QKV fixture")
        repo = Path(__file__).resolve().parent
        expected = digest({"core": computational_source(repo), "candidate": candidate_sources(repo)})
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_text(json.dumps(source))
        if fixture_text is not None:
            (destination / "fixture.json").write_text(fixture_text)
        campaign_id = uuid.uuid4().hex
        # This intent also identifies volume evidence if the client dies before call-ID persistence.
        (destination / "intent.json").write_text(json.dumps({"campaign_id": campaign_id, "source_sha256": expected, "attention_followup": attention_followup, "followup": followup, "linears_only": linears_only, "head_only": head_only}))
        with app.run(detach=True):
            call = (fixed_followup_campaign.spawn(campaign_id, workloads, expected, True) if attention_followup else
                    fixed_followup_campaign.spawn(campaign_id, workloads, expected) if followup else
                    fixed_campaign.spawn(campaign_id, workloads, fixture_text, expected, False, True) if head_only else
                    fixed_campaign.spawn(campaign_id, workloads, fixture_text, expected, True) if linears_only else
                    fixed_campaign.spawn(campaign_id, workloads, fixture_text, expected))
            dispatch = {"campaign_id": campaign_id, "call_id": call.object_id, "app_id": app.app_id, "source_sha256": expected, "attention_followup": attention_followup, "followup": followup, "linears_only": linears_only, "head_only": head_only}
            with dispatch_file.open("x") as handle:
                json.dump(dispatch, handle)
                handle.flush()
                import os
                os.fsync(handle.fileno())
    try:
        call.get()
    except BaseException:
        try:
            recover_artifacts(destination, dispatch["campaign_id"])
        except Exception as recovery_error:
            print(f"Evidence download pending or unavailable: {type(recovery_error).__name__}: {recovery_error}. "
                  f"Saved handle remains at {dispatch_file}; recover it without resubmitting.", flush=True)
        raise
    index = recover_artifacts(destination, dispatch["campaign_id"])
    if not index.get("complete") or index.get("error"):
        raise SystemExit("campaign failed or incomplete; durable evidence retained; no retry")
    print("Diagnostic campaign retained; no gate clearance.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="experiments/fix-validation-smoke-01/manifest.json")
    parser.add_argument("--fixture", default="experiments/numerical-debugger/warm-prefix-probe-13/frozen-qkv-input.json")
    parser.add_argument("--recover", action="store_true")
    selectors = parser.add_mutually_exclusive_group()
    selectors.add_argument("--followup", action="store_true", help="Full128 residual trace and complete-custom diagnostic")
    selectors.add_argument("--attention-followup", action="store_true", help="Full128 vLLM-only actual-KV and attention-replay diagnostic")
    selectors.add_argument("--linears-only", action="store_true", help="Unbiased microprobe then custom-only all-linear full128 diagnostic")
    selectors.add_argument("--head-only", action="store_true", help="Actual16 FP32 output-head microprobe only; no fullmodel stages")
    args = parser.parse_args()
    durable_fixed(args.output, args.manifest, args.fixture, recover=args.recover, followup=args.followup, attention_followup=args.attention_followup, linears_only=args.linears_only, head_only=args.head_only)
