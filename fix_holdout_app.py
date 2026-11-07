"""Checkpointed fixed-policy calibration and separately frozen fresh holdout."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import modal

if modal.is_local():
    from modal_app import SOURCE_COMMIT, sentinel_image
    capture_image = sentinel_image.add_local_file("experiments/fix_holdout.py", "/root/experiments/fix_holdout.py", copy=True).add_local_file(
        "experiments/fix_holdout_worker.py", "/root/experiments/fix_holdout_worker.py", copy=True).add_local_file(
        "fix_holdout_app.py", "/root/fix_holdout_app_source.py", copy=True).add_local_file(
        "fix_holdout_app.py", "/root/fix_holdout_app.py", copy=True)
    for source_file in ("modal_app.py", "pyproject.toml", "uv.lock", "experiments/parallel_validation.py"):
        capture_image = capture_image.add_local_file(source_file, "/root/" + source_file, copy=True)
else:
    SOURCE_COMMIT, capture_image = None, None

app = modal.App("cie-fix-validation")
volume = modal.Volume.from_name("cloud-inference-model-cache", create_if_missing=False)
evidence_volume = modal.Volume.from_name("cie-retained-validation", create_if_missing=True)
REPO = Path(__file__).parent


def evidence_reference(path: Path, repo: Path = REPO) -> str:
    """Keep external successful-run references usable until the explicit freeze copy."""
    resolved = path.resolve()
    return resolved.relative_to(repo.resolve()).as_posix() if resolved.is_relative_to(repo.resolve()) else str(resolved)


def committed(path: str) -> dict:
    target = (REPO / path).resolve()
    relative = target.relative_to(REPO).as_posix()
    frozen = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=REPO)
    if frozen != target.read_bytes():
        raise ValueError(f"not identical to committed evidence: {path}")
    return json.loads(frozen)


def frozen_input_lineage(manifest, bounds):
    return {"execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "manifest_repo_path": (REPO / manifest).resolve().relative_to(REPO.resolve()).as_posix(),
            "bounds_repo_path": (REPO / bounds).resolve().relative_to(REPO.resolve()).as_posix()}


def frozen_bounds(path: str) -> dict:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise ValueError("fresh validation requires a clean committed source tree")
    protocol = subprocess.check_output(["git", "show", "HEAD:CORRECTNESS_FIX_VALIDATION.md"], cwd=REPO, text=True)
    if "**Status: FROZEN" not in protocol:
        raise ValueError("fresh protocol is not frozen")
    record = committed(path)
    from experiments.fix_holdout import MODES, NAMESPACE, computational_source, digest
    if record.get("namespace") != NAMESPACE or record.get("calibration_passes") is not True:
        raise ValueError("missing successful frozen calibration")
    for mode in MODES:
        for engine in ("custom", "vllm"):
            value = record["bounds"][mode][engine]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid frozen drift bound")
    if record.get("computational_source_sha256") != computational_source(REPO)["sha256"]:
        raise ValueError("computational source changed after calibration")
    summary = committed(record["calibration_summary_path"])
    summary_bytes = (REPO / record["calibration_summary_path"]).read_bytes()
    if (hashlib.sha256(summary_bytes).hexdigest() != record["calibration_summary_sha256"]
            or summary.get("complete") is not True or summary.get("passes") is not True
            or summary.get("calibration_maxima") != record["bounds"]):
        raise ValueError("frozen bounds are not derived from the committed complete calibration summary")
    environments = summary.get("provenance", {}).values()
    hashes = {digest({field: env[field] for field in ("model", "model_files_sha256", "dependencies")}) for env in environments}
    if len(summary.get("provenance", {})) != 40 or hashes != {record.get("runtime_environment_sha256")}:
        raise ValueError("calibration runtime environment evidence incomplete or changed")
    return record


@app.function(image=capture_image, volumes={"/cache": volume}, timeout=600)
def prepare(split: str, excluded_hashes: list[str] | None = None) -> dict:
    from transformers import AutoTokenizer

    from cloud_engine.weights import ensure_weights_downloaded
    from experiments.fix_holdout import MODES, NAMESPACE, computational_source, digest, materialize
    from experiments.protocol_v2 import build_split_batches

    pinned = json.loads(Path("/root/engine_config.json").read_text())
    model = pinned["ragged_model"]
    model_dir = ensure_weights_downloaded("/cache", model["revision"], model["id"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    excluded = set(excluded_hashes or [])
    if split == "holdout":
        for old_split in ("calibration", "holdout"):
            for batches in build_split_batches(tokenizer, old_split).values():
                excluded.update(digest(ids) for batch in batches for ids in batch)
    workloads = {mode: materialize(tokenizer, split, mode, frozen=split == "holdout") for mode in MODES}
    hashes = [value for batches in workloads.values() for batch in batches for value in batch["prompt_hashes"]]
    if split == "holdout" and (len(hashes) != 214 or len(set(hashes)) != 214 or set(hashes) & excluded):
        raise ValueError("fresh prompt manifest is incomplete, duplicated or overlaps exposed prompts")
    volume.commit()
    return {"schema": "numerical-fix-validation-v1", "split": split,
            "namespace": NAMESPACE if split == "holdout" else "protocol-v2-calibration",
            "model": model, "workloads": workloads, "excluded_prompt_hashes": sorted(excluded),
            "workloads_sha256": digest(workloads), "computational_source": computational_source(Path("/root"), deployed=True)}


def _capture_child(mode: str, implementation: str, workload: dict, cancelled=lambda: False) -> dict:
    import sys
    from datetime import datetime, timezone

    import torch

    from cloud_engine.weights import ensure_weights_downloaded
    from experiments.fix_holdout import computational_source
    from experiments.sentinel_pilot import gpu_state_snapshot, resolved_dependency_versions

    pinned = json.loads(Path("/root/engine_config.json").read_text())
    model = pinned["ragged_model"]
    model_dir = str(ensure_weights_downloaded("/cache", model["revision"], model["id"]))
    root = Path(tempfile.mkdtemp(prefix="fix-validation-"))
    config = {"implementation": implementation, "mode": mode, "model": model, "model_dir": model_dir,
              "kv_bytes": pinned["kv_cache"]["bytes"], "workload": workload}
    (root / "config.json").write_text(json.dumps(config))
    model_files = {}
    for model_path in sorted(Path(model_dir).rglob("*")):
        if model_path.is_file() and model_path.suffix in (".json", ".safetensors", ".model", ".txt"):
            checksum = hashlib.sha256()
            with model_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    checksum.update(chunk)
            model_files[model_path.relative_to(model_dir).as_posix()] = checksum.hexdigest()
    environment = {"source": os.environ.get("SOURCE_COMMIT", "unknown"),
                   "started_utc": datetime.now(timezone.utc).isoformat(), "model": model, "model_files_sha256": model_files,
                   "modal_environment_ids": {key: value for key, value in os.environ.items() if key in ("MODAL_TASK_ID", "MODAL_FUNCTION_ID", "MODAL_IMAGE_ID", "MODAL_APP_ID")},
                   "gpu_before": gpu_state_snapshot(torch.version.cuda),
                   "dependencies": resolved_dependency_versions(),
                   "computational_source": computational_source(Path("/root"), deployed=True),
                   "capture_source_sha256": hashlib.sha256(Path("/root/experiments/fix_holdout.py").read_bytes()).hexdigest(),
                   "worker_source_sha256": hashlib.sha256(Path("/root/experiments/fix_holdout_worker.py").read_bytes()).hexdigest()}
    from experiments.parallel_validation import run_cancellable_process
    process = run_cancellable_process(
        [sys.executable, "/root/experiments/fix_holdout.py", str(root / "config.json"), str(root / "capture.json")],
        env=dict(os.environ, PYTHONPATH="/root"), stdout_path=root / "stdout.log", stderr_path=root / "stderr.log",
        cancelled=cancelled, timeout_seconds=1500)
    environment.update(finished_utc=datetime.now(timezone.utc).isoformat())
    try:
        evidence = json.loads((root / "capture.json").read_text())
    except (OSError, ValueError) as exc:
        evidence = {"complete": False, "error": f"no valid child checkpoint: {exc}"}
    try:
        environment["gpu_after"] = gpu_state_snapshot(torch.version.cuda)
    except Exception as exc:
        environment["gpu_after"] = {"uuid": None, "metadata_error": f"{type(exc).__name__}: {exc}"}
        evidence["complete"] = False
        evidence["post_gpu_metadata_error"] = environment["gpu_after"]["metadata_error"]
    if process["cancelled"] or process["timeout"]:
        evidence["complete"] = False
        evidence["cancellation_reason"] = process["cancellation_reason"]
    if process["returncode"] != 0 or environment["gpu_before"]["uuid"] != environment["gpu_after"]["uuid"]:
        evidence["complete"] = False
    return {"environment": environment, "process": process, "capture": evidence}


@app.function(image=capture_image, gpu="L4", volumes={"/cache": volume}, timeout=1800,
              startup_timeout=180, max_containers=1, scaledown_window=300)
def capture(mode: str, implementation: str, workload: dict) -> dict:
    """Single-engine exposed smoke only; full validation uses capture_pair."""
    return _capture_child(mode, implementation, workload)


def _child_hard_failures(mode, implementation, workload, capture, bounds):
    """Only intra-engine checks; cross-engine agreement requires both real children."""
    from experiments.fix_holdout import compare_records, validate_capture

    phases = {"unique"} if mode == "resource_normalized" else {"cold", "warm"}
    failures = []
    try:
        if (capture.get("complete") is not True or set(capture["batch"]) != phases
                or len(capture["solo"]) != workload["concurrency"]
                or any(set(solo) != phases for solo in capture["solo"])):
            raise ValueError("incomplete batch/solo phases")
        for phase in sorted(phases):
            validate_capture(capture["batch"][phase], workload["input_token_ids"], workload["output_tokens"], cold=phase != "warm", mode=mode)
            for index, (solo, prompt) in enumerate(zip(capture["solo"], workload["input_token_ids"], strict=True)):
                validate_capture(solo[phase], [prompt], workload["output_tokens"], cold=phase != "warm", mode=mode)
                comparison = compare_records(capture["batch"][phase]["records"][index], solo[phase]["records"][0],
                    bound=bounds[mode][implementation] if bounds is not None else None)
                if comparison["failures"]:
                    failures.append({"phase": phase, "request_index": index, **comparison})
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        failures.append({"reason": f"invalid_child_evidence: {type(exc).__name__}: {exc}"})
    return failures


@app.function(image=capture_image, gpu="L4", volumes={"/cache": volume}, timeout=3600,
              startup_timeout=180, max_containers=4, scaledown_window=300, retries=0)
def capture_pair(mode: str, index: int, workload: dict, campaign_id: str, expected_source: str,
                 bounds: dict | None = None) -> dict:
    from experiments.fix_holdout import MODES, summarize
    from experiments.parallel_validation import CANCEL_STORE, campaign_cancelled

    store = modal.Dict.from_name(CANCEL_STORE, create_if_missing=True)
    def cancelled():
        return campaign_cancelled(store, campaign_id)
    result = {"mode": mode, "index": index, "campaign_id": campaign_id, "children": {}, "stop": None}
    try:
        for implementation in ("custom", "vllm"):
            if cancelled():
                result["stop"] = "campaign_cancelled"
                break
            payload = _capture_child(mode, implementation, workload, cancelled)
            result["children"][implementation] = payload
            env = payload["environment"]
            if (payload["process"]["returncode"] != 0 or payload["capture"].get("complete") is not True
                    or env["computational_source"]["sha256"] != expected_source):
                result["stop"] = "child_failed_or_source_mismatch"
                break
            failures = _child_hard_failures(mode, implementation, workload, payload["capture"], bounds)
            if failures:
                result["child_failures"] = {"implementation": implementation, "failures": failures}
                result["stop"] = "intraengine_hard_validation_failure"
                break
        if result["stop"] is None:
            uuids = {payload["environment"][point]["uuid"] for payload in result["children"].values() for point in ("gpu_before", "gpu_after")}
            if len(uuids) != 1 or not next(iter(uuids)):
                result["stop"] = "paired_gpu_identity_mismatch"
            else:
                manifest = {"workloads": {item: [workload] if item == mode else [] for item in MODES}}
                captures = {f"{mode}/{engine}/0": payload["capture"] for engine,payload in result["children"].items()}
                result["summary"] = summarize(manifest, captures, bounds, partial=True)
                if not result["summary"]["passes"]:
                    result["stop"] = "hard_validation_failure"
    except Exception as exc:
        result["stop"] = "paired_orchestration_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    if result["stop"] is not None:
        try:
            store.put(campaign_id, True)
        except Exception as exc:
            result["cancel_signal_error"] = f"{type(exc).__name__}: {exc}"
    return result


@app.local_entrypoint()
def materialize_holdout(output: str, bounds: str = "fix-validation-bounds.json") -> None:
    """Materialize CPU-only fresh prompts after protocol/bounds freeze, before their commit."""
    from experiments.fix_holdout import computational_source, digest
    if Path(output).exists():
        raise SystemExit("refusing to overwrite a prompt manifest")
    bound_record = frozen_bounds(bounds)
    excluded = set()
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("input_token_ids", "batch_ids") and isinstance(item, list) and item:
                    for ids in ([item] if isinstance(item[0], int) else item):
                        if isinstance(ids, list) and all(isinstance(token, int) for token in ids):
                            excluded.add(digest(ids))
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    for path in (REPO / "experiments").rglob("*.json"):
        visit(json.loads(path.read_text()))
    for path in (REPO / "experiments").rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            visit(json.loads(line))
    manifest = prepare.remote("holdout", sorted(excluded))
    if manifest["computational_source"]["sha256"] != computational_source(REPO)["sha256"]:
        raise SystemExit("materialization source mismatch")
    manifest["bounds_sha256"] = digest(bound_record)
    manifest["freeze_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    with Path(output).open("x") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Fresh prompts materialized at {output}; commit this manifest before GPU execution.")


@app.local_entrypoint()
def run(output: str, split: str = "calibration", manifest: str = "", bounds: str = "fix-validation-bounds.json") -> None:
    """Default runs exposed calibration only; fresh generation requires committed inputs/bounds."""
    from experiments.fix_holdout import (
        NAMESPACE,
        computational_source,
        digest,
        validate_manifest,
    )

    destination = Path(output)
    if destination.exists():
        raise SystemExit("refusing to overwrite or silently resume retained validation")
    if split == "calibration":
        from modal_app import _source_commit
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        if (subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
                or SOURCE_COMMIT != _source_commit() or not SOURCE_COMMIT.startswith(head + "+tree-")):
            raise SystemExit("full calibration requires clean committed source matching the built image")
        frozen = None
        workload_manifest = prepare.remote("calibration")
    elif split == "holdout":
        frozen = frozen_bounds(bounds)
        workload_manifest = committed(manifest)
        if (workload_manifest.get("split") != "holdout" or workload_manifest.get("namespace") != NAMESPACE
                or workload_manifest.get("bounds_sha256") != digest(frozen)):
            raise SystemExit("fresh manifest does not bind the frozen bounds")
    else:
        raise SystemExit("unknown validation split")
    if workload_manifest["workloads_sha256"] != digest(workload_manifest["workloads"]):
        raise SystemExit("workload manifest checksum mismatch")
    validate_manifest(workload_manifest)
    source = computational_source(REPO)
    if workload_manifest["computational_source"]["sha256"] != source["sha256"]:
        raise SystemExit("manifest computational source mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "manifest.json").write_text(json.dumps(workload_manifest, indent=2))
    if split == "holdout":
        marker = Path(subprocess.check_output(["git", "rev-parse", "--git-path", f"fix-holdout-{NAMESPACE}.json"], cwd=REPO, text=True).strip())
        if not marker.is_absolute():
            marker = REPO / marker
        with marker.open("x") as handle:
            json.dump({"namespace": NAMESPACE, "manifest_sha256": digest(workload_manifest), "output": str(destination.resolve())}, handle)
    from modal_app import _source_commit
    def verify_source():
        return computational_source(REPO)["sha256"] == source["sha256"] and SOURCE_COMMIT == _source_commit()
    execute_validation(destination, split, workload_manifest, frozen, source, SOURCE_COMMIT,
                       subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                       evidence_reference(destination / "summary.json"), str(uuid.uuid4()), verify_source,
                       frozen_lineage=frozen_input_lineage(manifest, bounds) if split == "holdout" else None)


def execute_validation(destination, split, workload_manifest, frozen, source, source_commit,
                       calibration_commit, summary_reference, campaign_id, verify_source, checkpoint=lambda paths: None, frozen_lineage=None):
    from experiments.fix_holdout import MODES, NAMESPACE, digest, summarize
    from experiments.parallel_validation import CANCEL_STORE, campaign_cancelled, run_bounded
    if split == "holdout" and not frozen_lineage:
        raise ValueError("holdout missing original committed input lineage")
    store = modal.Dict.from_name(CANCEL_STORE, create_if_missing=True)
    if not store.put(campaign_id, False, skip_if_exists=True):
        raise SystemExit("campaign cancellation key already exists")
    (destination / "campaign.json").write_text(json.dumps({"campaign_id": campaign_id, "workers": 4,
        "unit": "sequential custom/vllm pair on one L4", "retries": 0, "cancel_poll_seconds": 2}, indent=2))
    if frozen is not None:
        (destination / "bounds.json").write_text(json.dumps(frozen, indent=2))
    checkpoint([path for path in (destination / "campaign.json", destination / "manifest.json", destination / "bounds.json") if path.exists()])
    captures, provenance, environment_hashes, hardware_hashes = {}, {}, set(), set()
    artifact_index = {}

    def cancel_campaign():
        store.put(campaign_id, True)

    def run_job(job):
        mode, index, workload = job
        if campaign_cancelled(store, campaign_id):
            return {"mode": mode, "index": index, "children": {}, "stop": "campaign_cancelled_before_dispatch"}
        if not verify_source():
            cancel_campaign()
            return {"mode": mode, "index": index, "children": {}, "stop": "local_source_changed"}
        return capture_pair.remote(mode, index, workload, campaign_id, source["sha256"], frozen["bounds"] if frozen is not None else None)

    def retain_result(job, pair):
        mode, index, workload = job
        pair_path = destination / "pairs" / f"{mode}-{index:02d}.json"
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        with pair_path.open("x") as handle:
            json.dump(pair, handle, allow_nan=False)
        checkpoint([pair_path])
        accepted = pair.get("stop") is None and "orchestration_error" not in pair and set(pair.get("children", {})) == {"custom", "vllm"}
        pair_uuids = set()
        for engine, payload in pair.get("children", {}).items():
            if engine not in ("custom", "vllm"):
                accepted = False
                continue
            path = destination / mode / engine / f"batch-{index:02d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as handle:
                json.dump(payload, handle, allow_nan=False)
            checkpoint([path])
            artifact_index[path.relative_to(destination).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            key = f"{mode}/{engine}/{index}"
            captures[key], provenance[key] = payload["capture"], payload["environment"]
            env = payload["environment"]
            pair_uuids.update(env[point]["uuid"] for point in ("gpu_before", "gpu_after"))
            environment_hash = digest({field: env[field] for field in ("model", "model_files_sha256", "dependencies")})
            environment_hashes.add(environment_hash)
            hardware_hashes.update(digest({field: env[point].get(field) for field in ("name", "driver_version", "cuda_version", "memory.total")}) for point in ("gpu_before", "gpu_after"))
            if (env["source"] != source_commit or env["computational_source"]["sha256"] != source["sha256"]
                    or len(environment_hashes) != 1 or len(hardware_hashes) != 1
                    or (frozen is not None and environment_hash != frozen["runtime_environment_sha256"])
                    or payload["process"]["returncode"] != 0 or payload["capture"].get("complete") is not True):
                accepted = False
        if len(pair_uuids) != 1 or not next(iter(pair_uuids), None):
            accepted = False
        if accepted:
            pair_manifest = {"workloads": {item: [workload] if item == mode else [] for item in MODES}}
            children = {f"{mode}/{engine}/0": pair["children"][engine]["capture"] for engine in ("custom", "vllm")}
            accepted = summarize(pair_manifest, children, frozen["bounds"] if frozen is not None else None, partial=True)["passes"]
        if not accepted:
            with (destination / "stops.jsonl").open("a") as handle:
                handle.write(json.dumps({"mode": mode, "index": index, "reason": pair.get("stop") or "local_integrity_failure"})+"\n")
            checkpoint([destination / "stops.jsonl"])
        return accepted

    for mode in MODES:
        jobs = [(mode,index,workload) for index,workload in enumerate(workload_manifest["workloads"][mode])]
        dispatch = run_bounded(jobs, run_job, retain_result, cancel_campaign, max_workers=4)
        (destination / f"{mode}-dispatch.json").write_text(json.dumps(dispatch, indent=2))
        checkpoint([destination / f"{mode}-dispatch.json"])
        if dispatch["stopped"] or campaign_cancelled(store, campaign_id):
            raise SystemExit("validation stopped; all started pair results retained; no retry or gate clearance")
    if not verify_source():
        cancel_campaign()
        raise SystemExit("source changed before campaign completion")
    summary = summarize(workload_manifest, captures, frozen["bounds"] if frozen is not None else None)
    summary["provenance"] = provenance
    summary.update(split=split, namespace=workload_manifest["namespace"])
    summary["validation_evidence"] = {"split": split, "namespace": workload_manifest["namespace"],
        "manifest_path": "manifest.json", "bounds_path": "bounds.json" if frozen is not None else None,
        "manifest_sha256": digest(workload_manifest), "bounds_sha256": digest(frozen) if frozen is not None else None,
        "computational_source_sha256": source["sha256"], "artifact_index": artifact_index,
        "frozen_input_lineage": frozen_lineage}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    checkpoint([destination / "summary.json"])
    if split == "calibration" and summary["passes"]:
        (destination / "proposed-bounds.json").write_text(json.dumps({
            "namespace": NAMESPACE, "calibration_passes": True, "bounds": summary["calibration_maxima"],
            "calibration_summary_path": summary_reference,
            "runtime_environment_sha256": next(iter(environment_hashes)),
            "calibration_summary_sha256": hashlib.sha256((destination / "summary.json").read_bytes()).hexdigest(),
            "source": source_commit, "calibration_commit": calibration_commit,
            "computational_source_sha256": source["sha256"], "metric": "shared-top20 max absolute log-probability drift"}, indent=2))
        checkpoint([destination / "proposed-bounds.json"])
    print(f"{split}: {'PASS' if summary['passes'] else 'FAIL'}; full sentinel pilot and final audit still required.")
    if not summary["passes"]:
        raise SystemExit("validation failed; gate remains stopped")



@app.local_entrypoint()
def smoke(output: str) -> None:
    """Exposed calibration c8/batch0, full windows, both modes; never fits bounds."""
    from experiments.fix_holdout import MODES, computational_source, summarize, validate_manifest
    destination = Path(output)
    if destination.exists():
        raise SystemExit("refusing to overwrite smoke evidence")
    manifest = prepare.remote("calibration")
    validate_manifest(manifest)
    source = computational_source(REPO)
    if manifest["computational_source"]["sha256"] != source["sha256"]:
        raise SystemExit("smoke source mismatch")
    destination.mkdir(parents=True)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    gpu_ids = set()
    for mode in MODES:
        workload = next(item for item in manifest["workloads"][mode] if item["concurrency"] == 8 and item["batch_index"] == 0)
        captures = {}
        for engine in ("custom", "vllm"):
            payload = capture.remote(mode, engine, workload)
            (destination / f"{mode}-{engine}.json").write_text(json.dumps(payload, allow_nan=False))
            captures[f"{mode}/{engine}/0"] = payload["capture"]
            gpu_ids.add(payload["environment"]["gpu_before"]["uuid"])
            if (not payload["capture"].get("complete") or len(gpu_ids) != 1
                    or payload["environment"]["source"] != SOURCE_COMMIT
                    or payload["environment"]["computational_source"]["sha256"] != source["sha256"]):
                raise SystemExit("incomplete smoke; raw evidence retained")
        pair_manifest = {"workloads": {item: [workload] if item == mode else [] for item in MODES}}
        summary = summarize(pair_manifest, captures, partial=True)
        summary.update(evaluation_kind="incomplete_exposed_calibration_smoke", calibration_maxima=None, gate_cleared=False)
        (destination / f"{mode}-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        if not summary["passes"]:
            raise SystemExit("smoke failed; no bounds fitted")
    print("Smoke passed its exposed subset; full calibration remains required. No bounds fitted.")


def validation_preflight(split, manifest, bounds):
    """Local Git/freeze checks precede detached coordinator dispatch."""
    from experiments.fix_holdout import NAMESPACE, computational_source, digest
    from modal_app import _source_commit
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if (subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()
            or SOURCE_COMMIT != _source_commit() or not SOURCE_COMMIT.startswith(head + '+tree-')):
        raise ValueError('detached validation requires clean committed source matching its image')
    if split == 'calibration':
        frozen, workload = None, None
    elif split == 'holdout':
        frozen, workload = frozen_bounds(bounds), committed(manifest)
        if (workload.get('split') != 'holdout' or workload.get('namespace') != NAMESPACE
                or workload.get('bounds_sha256') != digest(frozen)):
            raise ValueError('fresh manifest does not bind frozen bounds')
    else:
        raise ValueError('unknown validation split')
    return {'split': split, 'frozen': frozen, 'workload_manifest': workload,
            'source': computational_source(REPO), 'source_commit': SOURCE_COMMIT, 'calibration_commit': head,
            'frozen_input_lineage': frozen_input_lineage(manifest, bounds) if split == 'holdout' else None}


@app.function(image=capture_image, volumes={'/evidence': evidence_volume}, memory=8192,
              timeout=86400, max_containers=1, retries=0)
def validation_coordinator(campaign_id, inputs):
    from experiments.fix_holdout import NAMESPACE, computational_source, digest, validate_manifest
    from experiments.parallel_validation import CANCEL_STORE, open_durable_campaign
    root, state, checkpoint = open_durable_campaign(campaign_id, evidence_volume)
    store = modal.Dict.from_name(CANCEL_STORE, create_if_missing=True)
    try:
        workload = inputs['workload_manifest']
        source, frozen = inputs['source'], inputs['frozen']
        def verify_source():
            return (computational_source(Path('/root'), deployed=True)['sha256'] == source['sha256']
                    and os.environ.get('SOURCE_COMMIT') == inputs['source_commit']
                    and hashlib.sha256(Path('/root/fix_holdout_app.py').read_bytes()).hexdigest() == source['files']['fix_holdout_app.py'])
        if not verify_source():
            raise ValueError('coordinator deployed source mismatch')
        validate_manifest(workload)
        if workload['workloads_sha256'] != digest(workload['workloads']) or workload['computational_source']['sha256'] != source['sha256']:
            raise ValueError('coordinator workload/source mismatch')
        (root/'manifest.json').write_text(json.dumps(workload, indent=2))
        (root/'stage-inputs.json').write_text(json.dumps(inputs, indent=2))
        checkpoint([root/'manifest.json', root/'stage-inputs.json'])
        if inputs['split'] == 'holdout':
            lineage = inputs.get('frozen_input_lineage', {})
            if (lineage.get('execution_commit') != inputs['calibration_commit']
                    or any(not isinstance(lineage.get(key), str) or not lineage[key]
                           or Path(lineage[key]).is_absolute() or '..' in Path(lineage[key]).parts
                           for key in ('manifest_repo_path', 'bounds_repo_path'))):
                raise ValueError('holdout missing original committed input lineage')
            if (workload.get('namespace') != NAMESPACE or workload.get('split') != 'holdout'
                    or frozen is None or workload.get('bounds_sha256') != digest(frozen)
                    or frozen.get('computational_source_sha256') != source['sha256']):
                raise ValueError('invalid frozen fresh-stage binding')
            if not store.put(f'fresh-attempt:{NAMESPACE}', {'campaign_id': campaign_id, 'manifest_sha256': digest(workload)}, skip_if_exists=True):
                raise ValueError('fresh namespace already attempted; no replacement run')
        execute_validation(root, inputs['split'], workload, frozen, source, inputs['source_commit'],
                           inputs['calibration_commit'], inputs['summary_reference'], campaign_id, verify_source, checkpoint,
                           frozen_lineage=inputs.get('frozen_input_lineage'))
        checkpoint(complete=True)
    except BaseException as exc:
        checkpoint(error=f'{type(exc).__name__}: {exc}')
    finally:
        store.put(campaign_id, True)
    return state


def durable_validation(output, split='calibration', manifest='', bounds='fix-validation-bounds.json', *, recover=False):
    from experiments.fix_holdout import NAMESPACE, digest, validate_manifest
    from experiments.parallel_validation import save_dispatch, wait_durable
    destination = Path(output).resolve()
    if recover:
        dispatch = json.loads((destination/'dispatch.json').read_text())
        return wait_durable(modal.FunctionCall.from_id(dispatch['call_id']), evidence_volume, destination, dispatch)
    if destination.is_relative_to(REPO.resolve()):
        raise ValueError('durable validation output must be outside the frozen worktree')
    if destination.exists():
        raise ValueError('refusing overwrite or resubmission; recover saved call only')
    inputs = validation_preflight(split, manifest, bounds)
    campaign_id = str(uuid.uuid4())
    destination.mkdir(parents=True)
    (destination/'intent.json').write_text(json.dumps({'campaign_id': campaign_id, 'split': split}))
    if split == 'holdout':
        marker = Path(subprocess.check_output(['git', 'rev-parse', '--git-path', f'fix-holdout-{NAMESPACE}.json'], cwd=REPO, text=True).strip())
        if not marker.is_absolute():
            marker = REPO / marker
        with marker.open('x') as handle:
            json.dump({'namespace': NAMESPACE, 'manifest_sha256': digest(inputs['workload_manifest']),
                       'output': str(destination), 'campaign_id': campaign_id}, handle)
    with app.run(detach=True):
        if split == 'calibration':
            inputs['workload_manifest'] = prepare.remote('calibration')
        workload = inputs['workload_manifest']
        validate_manifest(workload)
        if workload['workloads_sha256'] != digest(workload['workloads']) or workload['computational_source']['sha256'] != inputs['source']['sha256']:
            raise ValueError('manifest source/checksum mismatch before dispatch')
        # A final local check catches edits made while the image/input preparation ran.
        check = validation_preflight(split, manifest, bounds)
        if check['source'] != inputs['source'] or check['source_commit'] != inputs['source_commit']:
            raise ValueError('source changed before detached dispatch')
        inputs['summary_reference'] = str(destination/'summary.json')
        (destination/'manifest.json').write_text(json.dumps(workload, indent=2))
        (destination/'stage-inputs.json').write_text(json.dumps(inputs, indent=2))
        call = validation_coordinator.spawn(campaign_id, inputs)
        dispatch = save_dispatch(destination, campaign_id, call, app.app_id)
    return wait_durable(call, evidence_volume, destination, dispatch)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('calibration', 'holdout'), default='calibration')
    parser.add_argument('--manifest', default='')
    parser.add_argument('--bounds', default='fix-validation-bounds.json')
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()
    durable_validation(args.output, args.split, args.manifest, args.bounds, recover=args.recover)
