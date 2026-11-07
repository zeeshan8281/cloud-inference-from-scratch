"""Exposed-input warm-prefix diagnostics; never calibration or gate evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

FILES = ("warm_prefix_probe.py", "custom_residual_probe.py", "qkv_policy_probe.py", "qkv_policy_worker.py", "qkv_head_policy.py")
if modal.is_local():
    from fix_holdout_app import capture_image
    probe_image = capture_image.add_local_file("warm_prefix_probe_app.py", "/root/warm_prefix_probe_app_source.py", copy=True).add_local_file(
        "warm_prefix_probe_app.py", "/root/warm_prefix_probe_app.py", copy=True)
    for name in FILES:
        probe_image = probe_image.add_local_file("experiments/" + name, "/root/experiments/" + name, copy=True)
else:
    probe_image = None
app = modal.App("cie-warm-prefix-probe")
volume = modal.Volume.from_name("cloud-inference-model-cache", create_if_missing=False)
evidence_volume = modal.Volume.from_name("cie-retained-diagnostics", create_if_missing=True)


def diagnostic_sources(root: Path, *, deployed: bool = False) -> dict:
    paths = {"experiments/" + name: root / "experiments" / name for name in FILES}
    paths["warm_prefix_probe_app.py"] = root / "warm_prefix_probe_app.py"
    paths["warm_prefix_probe_app_source.py"] = root / ("warm_prefix_probe_app_source.py" if deployed else "warm_prefix_probe_app.py")
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


@app.function(image=probe_image, gpu="L4", volumes={"/cache": volume}, timeout=3600,
              startup_timeout=180, max_containers=1, scaledown_window=300)
def capture(workload: dict, case: str, candidate: str = "fp32_qkv_and_head") -> dict:
    import os
    import sys
    import tempfile
    from datetime import datetime, timezone

    import torch

    from cloud_engine.weights import ensure_weights_downloaded
    from experiments.fix_holdout import computational_source
    from experiments.parallel_validation import run_cancellable_process
    from experiments.sentinel_pilot import gpu_state_snapshot, resolved_dependency_versions

    if candidate not in ("fp32_qkv_and_head", "fixed_tile_qkv_fp32_head"):
        raise ValueError("unknown warm-prefix candidate")
    pinned = json.loads(Path("/root/engine_config.json").read_text())
    model = pinned["ragged_model"]
    model_dir = str(ensure_weights_downloaded("/cache", model["revision"], model["id"]))
    root = Path(tempfile.mkdtemp(prefix="warm-prefix-"))
    config = {"model_dir": model_dir, "model": model, "kv_bytes": pinned["kv_cache"]["bytes"],
              "batch_ids": workload["input_token_ids"], "target_index": 2,
              "case": case, "candidate": candidate, "warm_prefix_protocol": True,
              "graph_enabled": case == "graph_control", "observe": case in ("eager_trace", "eager_repeat"),
              "cases": ["batch", "solo"], "output_tokens": 128}
    (root / "config.json").write_text(json.dumps(config))
    model_files = {}
    for path in sorted(Path(model_dir).rglob("*")):
        if path.is_file() and path.suffix in (".json", ".safetensors", ".model", ".txt"):
            checksum = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    checksum.update(chunk)
            model_files[path.relative_to(model_dir).as_posix()] = checksum.hexdigest()
    environment = {"candidate": candidate, "source": os.environ.get("SOURCE_COMMIT"), "started_utc": datetime.now(timezone.utc).isoformat(),
                   "gpu_before": gpu_state_snapshot(torch.version.cuda), "model": model,
                   "model_files_sha256": model_files, "dependencies": resolved_dependency_versions(),
                   "computational_source": computational_source(Path("/root"), deployed=True),
                   "diagnostic_sources": diagnostic_sources(Path("/root"), deployed=True)}
    process = run_cancellable_process(
        [sys.executable, "/root/experiments/warm_prefix_probe.py", str(root / "config.json"), str(root / "output.json")],
        env=dict(os.environ, PYTHONPATH="/root"), stdout_path=root / "stdout.log", stderr_path=root / "stderr.log",
        cancelled=lambda: False, timeout_seconds=3300)
    environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        environment["gpu_after"] = gpu_state_snapshot(torch.version.cuda)
    except Exception as exc:
        environment["gpu_after"] = {}
        environment["environment_error"] = f"{type(exc).__name__}: {exc}"
    return {"config": config, "environment": environment, "process": process,
            "files": {path.name: path.read_text() for path in root.iterdir() if path.suffix in (".json", ".jsonl")}}


def validate_payload(payload, expected_source, gpu_ids, runtimes, candidate="fp32_qkv_and_head"):
    from experiments.fix_holdout import digest
    if payload["process"].get("returncode") != 0:
        raise SystemExit("warm-prefix child crashed; raw retained")
    try:
        child = json.loads(payload["files"]["output.json"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit("missing child checkpoint; raw retained") from exc
    if not isinstance(child, dict) or child.get("complete") is not True or child.get("error") is not None:
        raise SystemExit("incomplete warm-prefix child; raw retained")
    if (payload.get("config", {}).get("candidate") != candidate
            or child.get("config", {}).get("candidate") != candidate
            or payload.get("environment", {}).get("candidate") != candidate):
        raise SystemExit("warm-prefix candidate identity mismatch; raw retained")
    expected_cases = {f"{selection}-{phase}" for selection in ("batch", "solo") for phase in ("cold", "warm")}
    if child.get("schema") != "warm-prefix-probe-v1" or set(child.get("cases", {})) != expected_cases:
        raise SystemExit("missing warm-prefix phases; raw retained")
    for phase_name, phase in child["cases"].items():
        if phase.get("candidate_policy", {}).get("candidate") != candidate:
            raise SystemExit("warm-prefix phase candidate identity mismatch; raw retained")
        records = phase.get("capture", {}).get("records", [])
        if len(records) != (8 if phase_name.startswith("batch-") else 1) or any(
                len(record.get("output_tokens", [])) != 128 for record in records):
            raise SystemExit("incomplete full128 control; raw retained")
    env = payload["environment"]
    if env.get("environment_error") or not env.get("gpu_after", {}).get("uuid"):
        raise SystemExit("post-child environment observation failed; raw retained")
    gpu_ids.update((env["gpu_before"]["uuid"], env["gpu_after"]["uuid"]))
    actual_source = digest({"core": env["computational_source"], "diagnostic": env["diagnostic_sources"]})
    runtimes.add(digest({key: env[key] for key in ("model", "model_files_sha256", "dependencies")}))
    if len(gpu_ids) != 1 or len(runtimes) != 1 or actual_source != expected_source:
        raise SystemExit("warm-prefix provenance mismatch; raw retained")


@app.local_entrypoint()
def run(output: str, manifest: str = "experiments/fix-validation-smoke-01/manifest.json", candidate: str = "fp32_qkv_and_head") -> None:
    from experiments.fix_holdout import computational_source, digest, validate_manifest
    destination = Path(output)
    if candidate not in ("fp32_qkv_and_head", "fixed_tile_qkv_fp32_head"):
        raise SystemExit("unknown warm-prefix candidate")
    if destination.exists():
        raise SystemExit("refusing to overwrite warm-prefix evidence")
    source = json.loads(Path(manifest).read_text())
    if (source.get("split") != "calibration" or source.get("namespace") != "protocol-v2-calibration"
            or source.get("workloads_sha256") != digest(source.get("workloads"))):
        raise SystemExit("warm-prefix diagnostic requires intact exposed calibration inputs")
    validate_manifest(source)
    workload = next(item for item in source["workloads"]["complete_system"]
                    if item["concurrency"] == 8 and item["batch_index"] == 0)
    repo = Path(__file__).resolve().parent
    expected_source = digest({"core": computational_source(repo), "diagnostic": diagnostic_sources(repo)})
    destination.mkdir(parents=True)
    (destination / "manifest.json").write_text(json.dumps(source))
    gpu_ids, runtimes = set(), set()
    for case in ("graph_control", "eager_control", "eager_trace", "eager_repeat"):
        payload = capture.remote(workload, case, candidate)
        (destination / f"{case}.json").write_text(json.dumps(payload, allow_nan=False))
        validate_payload(payload, expected_source, gpu_ids, runtimes, candidate)
        print(f"{case} retained; numerical mismatches are diagnostic evidence, not a gate verdict", flush=True)


@app.function(image=probe_image, gpu="L4", volumes={"/cache": volume, "/evidence": evidence_volume},
              timeout=14400, startup_timeout=180, max_containers=1, retries=0)
def campaign(campaign_id: str, workload: dict, expected_source: str, candidate: str = "fp32_qkv_and_head") -> dict:
    """One allocation, no retries; commit every result before interpreting it."""
    import re
    if candidate not in ("fp32_qkv_and_head", "fixed_tile_qkv_fp32_head"):
        raise ValueError("unknown warm-prefix candidate")
    if re.fullmatch(r"[0-9a-f]{32}", campaign_id) is None:
        raise ValueError("invalid campaign ID")
    root = Path("/evidence") / campaign_id
    root.mkdir(exist_ok=False)
    index = {"campaign_id": campaign_id, "complete": False, "error": None, "artifacts": {}, "cases": [], "candidate": candidate}
    def commit_index():
        (root / "index.json").write_text(json.dumps(index, allow_nan=False))
        evidence_volume.commit()
    commit_index()
    gpu_ids, runtimes = set(), set()
    for case in ("graph_control", "eager_control", "eager_trace", "eager_repeat"):
        try:
            payload = capture.local(workload, case, candidate)
        except BaseException as exc:
            payload = {"orchestration_error": f"{type(exc).__name__}: {exc}"}
        path = root / f"{case}.json"
        path.write_text(json.dumps(payload, allow_nan=False))
        index["artifacts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        index["cases"].append(case)
        commit_index()
        try:
            validate_payload(payload, expected_source, gpu_ids, runtimes, candidate)
        except BaseException as exc:
            index["error"] = f"{type(exc).__name__}: {exc}"
            commit_index()
            return index
    index["complete"] = True
    commit_index()
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


def durable(output: str, manifest: str, *, recover: bool = False, candidate: str = "fp32_qkv_and_head") -> None:
    import uuid

    from experiments.fix_holdout import computational_source, digest, validate_manifest
    destination = Path(output)
    dispatch_file = destination / "dispatch.json"
    if recover:
        dispatch = json.loads(dispatch_file.read_text())
        call = modal.FunctionCall.from_id(dispatch["call_id"])
    else:
        if candidate not in ("fp32_qkv_and_head", "fixed_tile_qkv_fp32_head"):
            raise SystemExit("unknown warm-prefix candidate")
        if destination.exists():
            raise SystemExit("refusing to overwrite or resubmit campaign")
        source = json.loads(Path(manifest).read_text())
        if (source.get("split") != "calibration" or source.get("namespace") != "protocol-v2-calibration"
                or source.get("workloads_sha256") != digest(source.get("workloads"))):
            raise SystemExit("requires intact exposed calibration manifest")
        validate_manifest(source)
        workload = next(row for row in source["workloads"]["complete_system"]
                        if row["concurrency"] == 8 and row["batch_index"] == 0)
        repo = Path(__file__).resolve().parent
        expected = digest({"core": computational_source(repo), "diagnostic": diagnostic_sources(repo)})
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_text(json.dumps(source))
        campaign_id = uuid.uuid4().hex
        # This intent also identifies volume evidence if the client dies before call-ID persistence.
        (destination / "intent.json").write_text(json.dumps({"campaign_id": campaign_id, "source_sha256": expected, "candidate": candidate}))
        with app.run(detach=True):
            call = campaign.spawn(campaign_id, workload, expected, candidate)
            dispatch = {"campaign_id": campaign_id, "call_id": call.object_id, "app_id": app.app_id, "source_sha256": expected, "candidate": candidate}
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
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--candidate", choices=("fp32_qkv_and_head", "fixed_tile_qkv_fp32_head"), default="fp32_qkv_and_head")
    args = parser.parse_args()
    durable(args.output, args.manifest, recover=args.recover, candidate=args.candidate)
