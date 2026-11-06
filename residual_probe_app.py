"""Retained real-GPU traces for residual batch/solo divergence on exposed inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

FILES = ("custom_residual_probe.py", "biased_linear_probe.py", "residual_vllm_probe.py", "residual_vllm_hook.py",
         "fix_holdout.py", "fix_holdout_worker.py", "operator_probe_hook.py", "projection_hook.py")
if modal.is_local():
    from modal_app import SOURCE_COMMIT, sentinel_image
    probe_image = sentinel_image
    probe_image = probe_image.add_local_file("residual_probe_app.py", "/root/residual_probe_app_source.py", copy=True)
    for name in FILES:
        probe_image = probe_image.add_local_file(f"experiments/{name}", f"/root/experiments/{name}", copy=True)
else:
    SOURCE_COMMIT, probe_image = None, None

app = modal.App("cie-residual-probe")
volume = modal.Volume.from_name("cloud-inference-model-cache", create_if_missing=False)


@app.function(image=probe_image, gpu="L4", volumes={"/cache": volume}, timeout=1800,
              startup_timeout=180, max_containers=1, scaledown_window=300)
def capture(implementation: str, batch_ids: list[list[int]], selection: str = "batch", observe: bool = True) -> dict:
    import os
    import subprocess
    import sys
    import tempfile
    from datetime import datetime, timezone

    import torch

    from cloud_engine.weights import ensure_weights_downloaded
    from experiments.sentinel_pilot import gpu_state_snapshot, resolved_dependency_versions

    if implementation not in ("custom", "vllm", "biased"):
        raise ValueError("unknown implementation")
    pinned = json.loads(Path("/root/engine_config.json").read_text())
    model = pinned["ragged_model"]
    model_dir = str(ensure_weights_downloaded("/cache", model["revision"], model["id"]))
    root = Path(tempfile.mkdtemp(prefix="residual-"))
    config = {"model_dir": model_dir, "model": model, "kv_bytes": pinned["kv_cache"]["bytes"],
              "batch_ids": batch_ids, "target_index": 2, "target_indices": [2, 6],
              "selection": selection, "observe": observe}
    (root / "config.json").write_text(json.dumps(config))
    source_paths = list(Path("/root/cloud_engine").rglob("*.py")) + [Path("/root/experiments") / name for name in FILES]
    source_paths += [Path("/root/experiments") / name for name in ("sentinel_pilot.py", "sentinel_runtime.py", "sentinel_diagnostics.py", "protocol_v2.py", "controlled.py")]
    source_paths += [Path("/root/engine_config.json"), Path("/root/residual_probe_app_source.py")]
    metadata = {"source": os.environ.get("SOURCE_COMMIT"), "started_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_before": gpu_state_snapshot(torch.version.cuda), "dependencies": resolved_dependency_versions(),
                "source_hashes": {str(path.relative_to('/root')): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}}
    script = "/root/experiments/" + {"custom": "custom_residual_probe.py", "vllm": "residual_vllm_probe.py",
                                     "biased": "biased_linear_probe.py"}[implementation]
    try:
        result = subprocess.run([sys.executable, script, str(root / "config.json"), str(root / "output.json")],
                                capture_output=True, text=True, timeout=1500, env=dict(os.environ, PYTHONPATH="/root"))
        process = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired as exc:
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value
        process = {"returncode": None, "timeout": True, "stdout": decode(exc.stdout), "stderr": decode(exc.stderr)}
    metadata.update(finished_utc=datetime.now(timezone.utc).isoformat(), gpu_after=gpu_state_snapshot(torch.version.cuda))
    return {"config": config, "environment": metadata, "process": process,
            "files": {path.name: path.read_text() for path in root.iterdir() if path.suffix in (".json", ".jsonl")}}


@app.local_entrypoint()
def run(output: str, manifest: str = "experiments/fix-validation-smoke-01/manifest.json", experiment: str = "residual") -> None:
    from experiments.fix_holdout import digest, validate_manifest

    destination = Path(output)
    if destination.exists():
        raise SystemExit("refusing to overwrite retained residual evidence")
    if experiment not in ("residual", "biased"):
        raise SystemExit("unknown experiment")
    source = json.loads(Path(manifest).read_text())
    if (source.get("split") != "calibration" or source.get("namespace") != "protocol-v2-calibration"
            or source.get("workloads_sha256") != digest(source.get("workloads"))):
        raise SystemExit("residual probes require the exposed calibration manifest with an intact checksum")
    validate_manifest(source)
    workload = next(item for item in source["workloads"]["resource_normalized"]
                    if item["concurrency"] == 8 and item["batch_index"] == 0)
    destination.mkdir(parents=True)
    (destination / "workload.json").write_text(json.dumps(workload, indent=2))
    cases = [("custom", "batch", True, "all")]
    cases += [("vllm", selection, observe, repeat) for observe, repeat in
              ((False, "control"), (True, "trace"), (True, "repeat"))
              for selection in ("batch", "solo-2", "solo-6")]
    if experiment == "biased":
        cases = [("biased", "batch", True, "all")]
    gpu_ids = set()
    for implementation, selection, observe, repeat in cases:
        payload = capture.remote(implementation, workload["input_token_ids"], selection, observe)
        name = f"{implementation}-{selection}-{repeat}"
        (destination / f"{name}.json").write_text(json.dumps(payload))
        if payload["process"]["returncode"] != 0:
            raise SystemExit(f"{implementation} probe failed; raw evidence retained")
        try:
            child = json.loads(payload["files"]["output.json"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"{name} missing or invalid child checkpoint; raw evidence retained") from exc
        if not isinstance(child, dict) or child.get("complete") is not True or child.get("error") is not None:
            raise SystemExit(f"{name} child incomplete or failed; raw evidence retained")
        gpu_ids.update((payload["environment"]["gpu_before"]["uuid"], payload["environment"]["gpu_after"]["uuid"]))
        if len(gpu_ids) != 1:
            raise SystemExit("GPU identity changed; evidence unresolved")
        print(f"{name} probe retained; no gate clearance", flush=True)
