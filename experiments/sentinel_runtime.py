"""Shared sentinel-pair orchestration; imports safely without GPU packages."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

from experiments.sentinel_pilot import (
    COMPARISON_MODES,
    PAIRS,
    PROTOCOL_VERSION,
    SENTINEL_CELLS,
    WARMUP_MAXIMUM,
    StopPilot,
    assert_gpu_identity_stable,
    build_pair_workload,
    check_token_parity,
    gpu_state_snapshot,
    warmup_converged,
    workload_hash,
)

PILOT_SCRIPT = Path("/root/experiments/sentinel_runtime.py")
CHILD_TIMEOUT_SECONDS = 1500


def torch_cuda_version():
    try:
        import torch
        return torch.version.cuda
    except Exception:
        return None


def phase_outputs(child_result):
    return {cell: {phase: [record["output_token_ids"] for record in sorted(data["records"], key=lambda r: r["request_index"])]
                   for phase, data in value["phases"].items()}
            for cell, value in child_result["cells"].items()}


def require_warmup_converged(samples, detail=None):
    if len(samples) > WARMUP_MAXIMUM or not warmup_converged(samples):
        raise StopPilot("warmup_not_converged", {"throughputs": samples, **(detail or {})})


def run_child(implementation, mode, pair, model_dir, workload_path, workdir, cancelled=None):
    from experiments.parallel_validation import run_cancellable_process

    workdir.mkdir(parents=True, exist_ok=False)
    output_path = workdir / "result.json"
    stdout_path, stderr_path = workdir / "stdout.log", workdir / "stderr.log"
    process = run_cancellable_process(
        [sys.executable, str(PILOT_SCRIPT), implementation, mode, str(pair), model_dir,
         str(workload_path), str(output_path)],
        env=dict(os.environ, PYTHONPATH="/root"), stdout_path=stdout_path, stderr_path=stderr_path,
        cancelled=cancelled or (lambda: False), timeout_seconds=CHILD_TIMEOUT_SECONDS, poll_seconds=2,
    )
    stop = None
    reason = process.get("cancellation_reason")
    if reason:
        stop = {"kind": "child_timeout" if reason == "timeout" else reason,
                "detail": {"timeout_seconds": CHILD_TIMEOUT_SECONDS,
                           "cancel_check_error": process.get("cancel_check_error")}}
    child = {"implementation": implementation, **process,
             "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
             "stderr_tail": process.get("stderr", "")[-4000:],
             "crashed": process.get("returncode") != 0, "stop": stop}
    if "CUDA out of memory" in process.get("stderr", "") or "OutOfMemoryError" in process.get("stderr", ""):
        child["oom"] = True
    # Retain final and partial checkpoints even after SIGKILL/nonzero exit.
    for path, key in ((output_path, "raw_result"), (workdir / "partial-phases.json", "partial_checkpoint")):
        if path.exists():
            raw = path.read_text()
            child[key + "_text"] = raw
            try:
                child[key] = json.loads(raw)
            except ValueError as exc:
                child[key + "_parse_error"] = f"{type(exc).__name__}: {exc}"
    if stop is None and process.get("returncode") == 0:
        try:
            payload = child.get("raw_result")
            if not isinstance(payload, dict):
                raise ValueError("result must be a JSON object")
            if "stop" in payload:
                if not isinstance(payload["stop"], dict) or not {"kind", "detail"} <= payload["stop"].keys():
                    raise ValueError("malformed child stop record")
                child["stop"] = payload["stop"]
            else:
                child["result"] = payload
        except (OSError, ValueError) as exc:
            child["stop"] = {"kind": "invalid_child_result", "detail": {"error": f"{type(exc).__name__}: {exc}"}}
    return child


def journaled_child_main():
    """Checkpoint completed phases outside their measured spans, then run original CLI."""
    from experiments import sentinel_pilot

    output = Path(sys.argv[-1])
    checkpoint = output.with_name("partial-phases.json")
    journal = {"complete": False, "scope": "completed phases only; active phase may be interrupted",
               "arguments": sys.argv[1:], "phases": [], "stop": None}
    original = sentinel_pilot._run_phase

    def save():
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(journal, allow_nan=False))
        temporary.replace(checkpoint)

    async def phase(*args, **kwargs):
        try:
            result = await original(*args, **kwargs)
        except StopPilot as exc:
            journal["stop"] = exc.as_dict()
            save()
            raise
        journal["phases"].append({"implementation": args[0], "run_id": args[3], "result": result})
        save()
        return result

    save()
    sentinel_pilot._run_phase = phase
    try:
        sentinel_pilot.main()
        journal["child_main_returned"] = True
        save()
    finally:
        sentinel_pilot._run_phase = original


def validate_child_result(result, workload, model, source, implementation):
    environment = result.get("environment", {})
    expected = {"protocol_version": PROTOCOL_VERSION, "mode": workload["mode"], "workload_hash": workload["workload_hash"]}
    if any(result.get(key) != value for key, value in expected.items()):
        raise StopPilot("workload_or_protocol_mismatch", {"implementation": implementation, "expected": expected})
    provenance = {"model": model["id"], "model_revision": model["revision"], "dtype": model["dtype"],
                  "repository_commit": source, "implementation": "custom-server" if implementation == "custom" else "vllm"}
    if not source or source == "unknown" or any(environment.get(key) != value for key, value in provenance.items()):
        raise StopPilot("provenance_mismatch", {"implementation": implementation, "expected": provenance, "actual": environment})
    try:
        for cell, data in result["cells"].items():
            require_warmup_converged(data["warmup_throughputs"], {"cell": cell})
            for phase, phase_data in data["phases"].items():
                graph = phase_data.get("graph_counters", {})
                allowed = {"eager_fallback"} if workload["mode"] == "resource_normalized" else {"eager_fallback", "capture_or_replay"}
                if graph.get("unresolved") is not False or graph.get("path") not in allowed:
                    raise StopPilot("unresolved_graph_path", {"implementation": implementation, "cell": cell, "phase": phase, "graph_counters": graph})
        # Validate workload completeness independently for each engine before comparison.
        outputs = phase_outputs(result)
        check_token_parity(outputs, outputs, mode=workload["mode"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StopPilot("invalid_child_result", {"implementation": implementation, "error": f"{type(exc).__name__}: {exc}"}) from exc


def executed_source_hashes(root=Path("/root")):
    paths = [root / "engine_config.json", *(root / "experiments").glob("*.py"), *(root / "cloud_engine").rglob("*.py")]
    return {("src/" if path.is_relative_to(root / "cloud_engine") else "") + path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def run_pair(model_dir, mode, pair, cancelled=None):
    from transformers import AutoTokenizer

    if mode not in COMPARISON_MODES or not 1 <= pair <= PAIRS:
        raise ValueError("invalid fixed-protocol pair")
    model = json.loads(Path("/root/engine_config.json").read_text())["ragged_model"]
    source = os.environ.get("SOURCE_COMMIT", "unknown")
    workload = build_pair_workload(mode, pair, SENTINEL_CELLS, AutoTokenizer.from_pretrained(model_dir))
    workdir = Path(tempfile.mkdtemp(prefix=f"sentinel-{mode}-pair{pair:02d}-"))
    workload_path = workdir / "workload.json"
    workload_path.write_text(json.dumps(workload))
    order = "odd" if pair % 2 else "even"
    implementations = ("custom", "vllm") if pair % 2 else ("vllm", "custom")
    result = {"mode": mode, "pair": pair, "order": order, "workload_hash": workload["workload_hash"],
              "children": [], "stop": None, "source_fingerprint": source,
              "executed_source_hashes": executed_source_hashes(),
              "modal_task_id": os.environ.get("MODAL_TASK_ID"), "modal_app_id": os.environ.get("MODAL_APP_ID")}
    cuda_version = torch_cuda_version()
    gpu_states = [gpu_state_snapshot(cuda_version)]
    try:
        for position, implementation in enumerate(implementations, start=1):
            if cancelled is not None and cancelled():
                raise StopPilot("campaign_cancelled", {"before_child": implementation})
            current = json.loads(workload_path.read_text())
            unhashed = {key: value for key, value in current.items() if key != "workload_hash"}
            if current != workload or workload_hash([unhashed]) != workload["workload_hash"]:
                raise StopPilot("workload_hash_mismatch", {"implementation": implementation})
            child = run_child(implementation, mode, pair, model_dir, workload_path, workdir / f"child-{position}", cancelled=cancelled)
            child["position"] = position
            result["children"].append(child)
            gpu_states.append(gpu_state_snapshot(cuda_version))
            assert_gpu_identity_stable(gpu_states)
            if child["stop"] is not None:
                raise StopPilot(child["stop"]["kind"], {"implementation": implementation, "position": position, **child["stop"]["detail"]})
            if child["crashed"]:
                raise StopPilot("oom" if child.get("oom") else "child_crash", {"implementation": implementation, "position": position, "returncode": child["returncode"], "stderr_tail": child["stderr_tail"]})
            if json.loads(workload_path.read_text()) != workload:
                raise StopPilot("workload_hash_mismatch", {"implementation": implementation, "after_child": True})
            validate_child_result(child["result"], workload, model, source, implementation)
        by_implementation = {child["implementation"]: child["result"] for child in result["children"]}
        check_token_parity(phase_outputs(by_implementation["custom"]), phase_outputs(by_implementation["vllm"]), mode=mode)
    except StopPilot as exc:
        result["stop"] = exc.as_dict()
    except Exception as exc:
        result["stop"] = {"kind": "pair_orchestration_error", "detail": {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}}
    result["gpu_states"] = gpu_states
    result["workload"] = workload
    return result


if __name__ == "__main__":
    journaled_child_main()
