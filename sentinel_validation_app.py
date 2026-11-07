"""Isolated full sentinel pilot: both modes, ten paired rounds per mode."""
from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path

import modal

if modal.is_local():
    from modal_app import SOURCE_COMMIT, _source_commit, sentinel_image
    sentinel_image = sentinel_image.add_local_file("sentinel_validation_app.py", "/root/sentinel_validation_app_source.py", copy=True)
    for source_file in [Path(name) for name in ("sentinel_validation_app.py", "modal_app.py", "engine_config.json", "pyproject.toml", "uv.lock")]:
        sentinel_image = sentinel_image.add_local_file(str(source_file), "/root/" + source_file.as_posix(), copy=True)
    for source_file in sorted(Path("experiments").rglob("*.py")):
        sentinel_image = sentinel_image.add_local_file(str(source_file), "/root/" + source_file.as_posix(), copy=True)
else:
    sentinel_image = None

PINNED = json.loads(Path(__file__).with_name("engine_config.json").read_text())
MODAL_CFG = PINNED["modal"]
volume = modal.Volume.from_name(MODAL_CFG["volume_name"], create_if_missing=False)
app = modal.App("cie-sentinel-validation")
evidence_volume = modal.Volume.from_name("cie-retained-validation", create_if_missing=True)


@app.function(image=sentinel_image, gpu=MODAL_CFG["gpu"], volumes={"/cache": volume},
              timeout=3600, startup_timeout=180, max_containers=4, scaledown_window=300, retries=0)
def pair(mode: str, round_number: int, campaign_id: str):
    from cloud_engine.weights import ensure_weights_downloaded
    from experiments.parallel_validation import campaign_cancelled
    from experiments.sentinel_pilot import hash_file_tree, resolved_dependency_versions
    from experiments.sentinel_runtime import run_pair

    store = modal.Dict.from_name("cie-validation-cancellation")
    def cancelled():
        return campaign_cancelled(store, campaign_id)
    if cancelled():
        return {"mode": mode, "pair": round_number, "stop": {"kind": "campaign_cancelled", "detail": {"before_model": True}}}

    model = PINNED["ragged_model"]
    model_dir = str(ensure_weights_downloaded("/cache", model["revision"], model["id"]))
    volume.commit()
    # File hashing and package inspection stay outside child measurement spans.
    before = hash_file_tree(Path(model_dir))
    dependencies = resolved_dependency_versions()
    result = run_pair(model_dir, mode, round_number, cancelled=cancelled)
    def signal_stop():
        try:
            store.put(campaign_id, True)
        except Exception as exc:
            result["cancel_signal_error"] = f"{type(exc).__name__}: {exc}"
    try:
        result["executed_source_hashes"]["sentinel_validation_app.py"] = hashlib.sha256(Path("/root/sentinel_validation_app_source.py").read_bytes()).hexdigest()
    except Exception as exc:
        result["source_metadata_error"] = f"{type(exc).__name__}: {exc}"
        if result.get("stop") is None:
            result["stop"] = {"kind": "source_metadata_unresolved", "detail": {"error": result["source_metadata_error"]}}
    result.update(model_files_before=before, dependency_versions=dependencies)
    if result.get("stop") is not None:
        signal_stop()
        result["model_files_after"] = None
        result["model_hash_after_skipped"] = "failed pair; cancellation signal attempted immediately"
        return result
    try:
        stopped = cancelled()
    except Exception as exc:
        result.update(stop={"kind": "cancel_state_unresolved", "detail": {"error": f"{type(exc).__name__}: {exc}"}}, model_files_after=None)
        signal_stop()
        return result
    if stopped:
        result.update(stop={"kind": "campaign_cancelled", "detail": {"after_children": True}}, model_files_after=None)
        return result
    try:
        result["model_files_after"] = hash_file_tree(Path(model_dir))
    except Exception as exc:
        result["model_files_after"] = None
        result["model_hash_error"] = f"{type(exc).__name__}: {exc}"
        if result.get("stop") is None:
            result["stop"] = {"kind": "model_files_unresolved", "detail": {"error": result["model_hash_error"]}}
    if result.get("stop") is not None:
        signal_stop()
    return result


def preflight(output):
    from experiments.sentinel_pilot import build_source_manifest

    root = Path(__file__).parent
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"refusing to overwrite retained evidence: {destination}")
    paths = [root / name for name in ("sentinel_validation_app.py", "modal_app.py", "engine_config.json", "pyproject.toml", "uv.lock")]
    paths.extend(sorted((root / "src").rglob("*.py")))
    paths.extend(sorted((root / "experiments").rglob("*.py")))
    manifest = build_source_manifest(root, paths)
    if manifest["dirty"] or not SOURCE_COMMIT.startswith(manifest["git_commit"] + "+tree-") or SOURCE_COMMIT != _source_commit():
        raise SystemExit("sentinel pilot requires a clean frozen source tree matching its built image")
    manifest.update(source_fingerprint=SOURCE_COMMIT, model=PINNED["ragged_model"])
    return manifest


def verify_frozen_lineage(evidence, manifest, bounds):
    """Bind external result copies to the inputs committed before holdout dispatch."""
    from experiments.fix_holdout import digest
    from fix_holdout_app import frozen_bounds
    repo = Path(__file__).resolve().parent
    lineage = evidence.get('frozen_input_lineage', {})
    commit = lineage.get('execution_commit', '')
    if len(commit) != 40 or any(char not in '0123456789abcdef' for char in commit):
        raise SystemExit('holdout missing original committed input lineage')
    try:
        subprocess.check_call(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=repo,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        protocol = subprocess.check_output(['git', 'show', f'{commit}:CORRECTNESS_FIX_VALIDATION.md'], cwd=repo, text=True)
        if '**Status: FROZEN' not in protocol:
            raise ValueError('holdout execution commit did not freeze protocol')
        for key, value in (('manifest', manifest), ('bounds', bounds)):
            relative = lineage[key+'_repo_path']
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts or path.as_posix() != relative:
                raise ValueError('invalid repo-relative frozen input path')
            target = (repo/path).resolve()
            target.relative_to(repo)
            original = json.loads(subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=repo))
            current = subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=repo)
            if current != target.read_bytes() or digest(original) != digest(value) or digest(json.loads(current)) != digest(value):
                raise ValueError('copied/committed frozen inputs differ')
        if digest(frozen_bounds(lineage['bounds_repo_path'])) != digest(bounds):
            raise ValueError('current frozen calibration lineage differs')
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'holdout frozen input lineage invalid: {exc}') from exc
    return lineage


def verify_holdout_prerequisite(summary_path):
    """Recompute fresh verdict from hash-bound raw children before launching any pilot."""
    import math

    from experiments.fix_holdout import (
        MODES,
        NAMESPACE,
        computational_source,
        digest,
        summarize,
        validate_manifest,
    )

    if not summary_path:
        raise SystemExit("full pilot requires --holdout-summary from the successful fresh strict holdout")
    path = Path(summary_path).resolve()
    summary = json.loads(path.read_text())
    evidence = summary["validation_evidence"]
    if (summary.get("complete") is not True or summary.get("passes") is not True
            or summary.get("split") != "holdout" or summary.get("namespace") != NAMESPACE
            or evidence.get("split") != "holdout" or evidence.get("namespace") != NAMESPACE):
        raise SystemExit("pilot prerequisite is not a complete successful fresh holdout")
    def resolve(relative):
        target = (path.parent / relative).resolve()
        target.relative_to(path.parent)
        return target
    manifest = json.loads(resolve(evidence["manifest_path"]).read_text())
    bounds = json.loads(resolve(evidence["bounds_path"]).read_text())
    lineage = verify_frozen_lineage(evidence, manifest, bounds)
    code = computational_source(Path(__file__).parent)["sha256"]
    if (manifest.get("split") != "holdout" or manifest.get("namespace") != NAMESPACE
            or bounds.get("namespace") != NAMESPACE or bounds.get("calibration_passes") is not True
            or evidence.get("manifest_sha256") != digest(manifest)
            or evidence.get("bounds_sha256") != digest(bounds)
            or manifest.get("bounds_sha256") != digest(bounds)
            or manifest.get("workloads_sha256") != digest(manifest.get("workloads"))
            or evidence.get("computational_source_sha256") != code
            or bounds.get("computational_source_sha256") != code
            or manifest.get("computational_source", {}).get("sha256") != code):
        raise SystemExit("holdout prerequisite source/manifest/bounds binding mismatch")
    validate_manifest(manifest)
    for mode in MODES:
        for engine in ("custom", "vllm"):
            value = bounds["bounds"][mode][engine]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise SystemExit("invalid frozen holdout bound")
    index = evidence["artifact_index"]
    expected_paths = {f"{mode}/{engine}/batch-{number:02d}.json" for mode in MODES
                      for engine in ("custom", "vllm") for number in range(10)}
    if set(index) != expected_paths:
        raise SystemExit("fresh holdout prerequisite requires exactly forty raw engine captures")
    captures, runtimes, pair_gpus = {}, set(), {}
    for mode in MODES:
        for engine in ("custom", "vllm"):
            for number in range(10):
                relative = f"{mode}/{engine}/batch-{number:02d}.json"
                raw = resolve(relative).read_bytes()
                if hashlib.sha256(raw).hexdigest() != index[relative]:
                    raise SystemExit("fresh holdout raw evidence checksum mismatch")
                payload = json.loads(raw)
                env = payload["environment"]
                if (payload["process"].get("returncode") != 0
                        or env["computational_source"]["sha256"] != code
                        or env["gpu_before"]["uuid"] != env["gpu_after"]["uuid"]):
                    raise SystemExit("fresh holdout execution provenance mismatch")
                gpu = env["gpu_before"]["uuid"]
                if not gpu or pair_gpus.setdefault((mode, number), gpu) != gpu:
                    raise SystemExit("fresh holdout paired GPU identity mismatch")
                runtimes.add(digest({field: env[field] for field in ("model", "model_files_sha256", "dependencies")}))
                captures[f"{mode}/{engine}/{number}"] = payload["capture"]
    if runtimes != {bounds["runtime_environment_sha256"]}:
        raise SystemExit("fresh holdout model/dependencies differ from frozen calibration")
    result = summarize(manifest, captures, bounds["bounds"])
    if result.get("complete") is not True or result.get("passes") is not True:
        raise SystemExit("fresh holdout does not pass independent raw revalidation")
    return {"summary_path": str(path), "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "manifest_sha256": digest(manifest), "bounds_sha256": digest(bounds),
            "computational_source_sha256": code, "raw_capture_count": len(captures),
            "fresh_unique_requests": 214, "independently_revalidated": True, "frozen_input_lineage": lineage}


def verify_local_source(manifest):
    root = Path(__file__).parent
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != manifest["git_commit"] or any(hashlib.sha256((root/path).read_bytes()).hexdigest() != digest for path, digest in manifest["sources"].items()):
        raise SystemExit("source changed during the pilot; retained completed pairs, no further launch")



def verify_artifact_provenance(result):
    from experiments.sentinel_pilot import StopPilot

    before, after = result.get("model_files_before"), result.get("model_files_after")
    dependencies = result.get("dependency_versions")
    if (not isinstance(before, dict) or not before or before != after
            or any(not isinstance(value, str) or len(value) != 64
                   or any(character not in "0123456789abcdef" for character in value) for value in before.values())):
        raise StopPilot("model_files_mismatch", {"reason": "missing, malformed or changed model/tokenizer bytes"})
    if (not isinstance(dependencies, list) or not dependencies
            or any(not isinstance(value, str) or not value for value in dependencies)
            or dependencies != sorted(set(dependencies))):
        raise StopPilot("dependency_provenance_missing", {})
    return {"model_files": before, "dependency_versions": dependencies}


def verify_child_records(result, workload, implementation):
    from cloud_engine.precision import require_numerical_policy
    from experiments.sentinel_pilot import StopPilot

    precision = result["engine_config"]["inference_precision"]
    states = precision if isinstance(precision, list) else [precision]
    if not states or any(state.get("allow_fp16_reduced_precision_reduction") is not False for state in states):
        raise StopPilot("precision_policy_mismatch", {})
    for name, cell in result["cells"].items():
        for phase, data in cell["phases"].items():
            for field in (("numerical_policy", "prime_numerical_policy") if phase == "warm" else ("numerical_policy",)):
                policies = data.get(field, {})
                for timing in ("before", "after"):
                    policy = policies.get(timing, {})
                    require_numerical_policy(policy, implementation)
            expected = {record["request_index"]: record["output_tokens"] for record in workload["cells"][name][phase]}
            records = data["records"]
            if (len(records) != len(expected) or {record["request_index"] for record in records} != set(expected)
                    or any(record.get("error") is not None or record.get("timeout") is not False
                           or len(record["output_token_ids"]) != expected[record["request_index"]] for record in records)):
                raise StopPilot("invalid_request_evidence", {"cell": name, "phase": phase})


def verify_pair(result, mode, round_number, manifest):
    from experiments.sentinel_pilot import (
        StopPilot,
        assert_gpu_identity_stable,
        check_token_parity,
        workload_hash,
    )
    from experiments.sentinel_runtime import phase_outputs, validate_child_result

    if result.get("stop") is not None:
        return
    try:
        verify_artifact_provenance(result)
        source = result["executed_source_hashes"]
        required = {name for name in manifest["sources"] if name.startswith("src/")} | {
            "engine_config.json", "experiments/sentinel_runtime.py", "experiments/sentinel_pilot.py", "experiments/controlled.py", "experiments/parallel_validation.py", "sentinel_validation_app.py"}
        if (result["source_fingerprint"] != manifest["source_fingerprint"] or not required <= source.keys()
                or any(manifest["sources"].get(name) != digest for name, digest in source.items())):
            raise StopPilot("source_mismatch", {"actual_source": result.get("source_fingerprint")})
        workload = result["workload"]
        unhashed = {key: value for key, value in workload.items() if key != "workload_hash"}
        if (result["mode"] != mode or result["pair"] != round_number or workload["mode"] != mode or workload["pair"] != round_number
                or result["workload_hash"] != workload["workload_hash"] or workload_hash([unhashed]) != workload["workload_hash"]):
            raise StopPilot("workload_hash_mismatch", {})
        order = ("custom", "vllm") if round_number % 2 else ("vllm", "custom")
        if (result["order"] != ("odd" if round_number % 2 else "even")
                or tuple(child["implementation"] for child in result["children"]) != order
                or [child["position"] for child in result["children"]] != [1, 2]):
            raise StopPilot("incomplete_evidence", {"reason": "child order/count mismatch"})
        if len(result["gpu_states"]) != 3 or any(not state.get("uuid") for state in result["gpu_states"]):
            raise StopPilot("incomplete_evidence", {"reason": "missing GPU identity samples"})
        assert_gpu_identity_stable(result["gpu_states"])
        for child in result["children"]:
            if child["crashed"] or child["stop"] is not None:
                raise StopPilot("incomplete_evidence", {"reason": "child failure omitted from pair stop"})
            validate_child_result(child["result"], workload, manifest["model"], manifest["source_fingerprint"], child["implementation"])
            verify_child_records(child["result"], workload, child["implementation"])
        children = {child["implementation"]: child["result"] for child in result["children"]}
        check_token_parity(phase_outputs(children["custom"]), phase_outputs(children["vllm"]), mode=mode)
    except (KeyError, TypeError, ValueError) as exc:
        raise StopPilot("incomplete_evidence", {"error": f"{type(exc).__name__}: {exc}"}) from exc


@app.local_entrypoint()
def run(output: str = "experiments/sentinel-validation", holdout_summary: str = ""):
    manifest = preflight(output)
    prerequisite = verify_holdout_prerequisite(holdout_summary)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    execute_pilot(root, manifest, prerequisite, str(uuid.uuid4()), lambda: verify_local_source(manifest))


def execute_pilot(root, manifest, prerequisite, campaign_id, verify_source, checkpoint=lambda paths: None):
    from experiments.parallel_validation import campaign_cancelled, run_bounded
    from experiments.sentinel_pilot import (
        COMPARISON_MODES,
        PAIRS,
        PROTOCOL_VERSION,
        SENTINEL_CELLS,
        WARMUP_MAXIMUM,
        WARMUP_MINIMUM,
        WARMUP_TOLERANCE,
        StopPilot,
    )
    from experiments.sentinel_runtime import CHILD_TIMEOUT_SECONDS

    (root/"source-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (root/"holdout-prerequisite.json").write_text(json.dumps(prerequisite, indent=2)+"\n")
    protocol = {"protocol_version": PROTOCOL_VERSION, "modes": list(COMPARISON_MODES), "pairs_per_mode": PAIRS,
                "cells": [asdict(cell) | {"name": cell.name} for cell in SENTINEL_CELLS],
                "warmup": {"minimum": WARMUP_MINIMUM, "maximum": WARMUP_MAXIMUM, "tolerance": WARMUP_TOLERANCE},
                "child_timeout_seconds": CHILD_TIMEOUT_SECONDS, "pair_timeout_seconds": 3600,
                "max_concurrent_pairs": 4, "gpu_type": "L4", "gpu_identity_scope": "stable within each sequential engine pair",
                "cancellation_poll_seconds": 2, "retry_replacement": False,
                "gate": "exact tokens; full fixed workloads; original stop rules preserved"}
    (root/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    checkpoint([root/"source-manifest.json", root/"holdout-prerequisite.json", root/"protocol.json"])
    mode_dirs = {"resource_normalized": "resource-normalized", "complete_system": "complete-policy"}
    completed, artifact_baseline, identities = 0, None, set()
    store = modal.Dict.from_name("cie-validation-cancellation", create_if_missing=True)
    if not store.put(campaign_id, False, skip_if_exists=True):
        raise SystemExit("could not create unique cancellation campaign")
    (root/"campaign.json").write_text(json.dumps({"id": campaign_id, "store": "cie-validation-cancellation"}))

    checkpoint([root/"campaign.json"])

    def cancel():
        store.put(campaign_id, True)

    def execute(job):
        if campaign_cancelled(store, campaign_id):
            return {"stop": {"kind": "campaign_cancelled", "detail": {"before_dispatch": True}}}
        try:
            verify_source()
        except SystemExit as exc:
            return {"orchestration_error": str(exc)}
        mode, number = job
        return pair.remote(mode, number, campaign_id)

    def retain(job, result):
        nonlocal completed, artifact_baseline
        mode, number = job
        path = root/"raw"/mode_dirs[mode]/f"pair-{number:02d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)
        checkpoint([path])
        if "orchestration_error" in result:
            result["stop"] = {"kind": "pair_orchestration_error", "detail": {"error": result["orchestration_error"]}}
        try:
            verify_pair(result, mode, number, manifest)
            if result.get("stop") is None:
                actual = verify_artifact_provenance(result)
                if artifact_baseline is None:
                    artifact_baseline = actual
                elif actual != artifact_baseline:
                    raise StopPilot("artifact_provenance_changed", {"reason": "model/tokenizer bytes or dependencies changed between pairs"})
            identities.update(state["uuid"] for state in result.get("gpu_states", []) if state.get("uuid"))
        except (StopPilot, KeyError, TypeError, ValueError) as exc:
            result["stop"] = exc.as_dict() if isinstance(exc, StopPilot) else {"kind": "invalid_pair_evidence", "detail": {"error": str(exc)}}
        # Preserve raw bytes; validation annotations have a separate immutable artifact.
        path = root/"validation"/mode_dirs[mode]/f"pair-{number:02d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)
        checkpoint([path])
        if result.get("workload") is not None:
            with (root/"workloads.jsonl").open("a") as handle:
                handle.write(json.dumps(result["workload"], separators=(",", ":"))+"\n")
                handle.flush()
            checkpoint([root/"workloads.jsonl"])
        print(f"{mode} pair {number:02d}: stop={result.get('stop')}", flush=True)
        accepted = result.get("stop") is None
        if accepted and campaign_cancelled(store, campaign_id):
            accepted = False
        completed += int(accepted)
        return accepted

    try:
        for mode in COMPARISON_MODES:
            campaign = run_bounded([(mode, number) for number in range(1, PAIRS + 1)], execute, retain, cancel, max_workers=4)
            (root/f"{mode}-dispatch.json").write_text(json.dumps(campaign, indent=2)+"\n")
            checkpoint([root/f"{mode}-dispatch.json"])
            if campaign["stopped"]:
                raise SystemExit(f"sentinel pilot STOPPED; all started jobs drained and evidence retained in {root}; no performance claim")
        verify_source()
    finally:
        cancel()
    assert completed == len(COMPARISON_MODES) * PAIRS == 20
    (root/"completion.json").write_text(json.dumps({"complete": True, "paired_rounds": completed,
        "source_fingerprint": manifest["source_fingerprint"], "gpu_uuids": sorted(identities),
        "gpu_identity_scope": "within each pair", "max_concurrent_pairs": 4,
        "fresh_holdout_revalidated": True, "gate_cleared": False}, indent=2)+"\n")
    checkpoint([root/"completion.json"])
    print(f"All 20 paired rounds complete: {root}. Final raw-evidence audit remains required.")



def verify_deployed_pilot_source(manifest):
    import os
    if os.environ.get('SOURCE_COMMIT') != manifest['source_fingerprint']:
        raise ValueError('deployed pilot source fingerprint mismatch')
    for name, expected in manifest['sources'].items():
        deployed = name.removeprefix('src/') if name.startswith('src/cloud_engine/') else name
        path = Path('/root') / deployed
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'deployed pilot source differs: {name}')
    alias = Path('/root/sentinel_validation_app_source.py')
    if hashlib.sha256(alias.read_bytes()).hexdigest() != manifest['sources']['sentinel_validation_app.py']:
        raise ValueError('pilot source alias differs')


@app.function(image=sentinel_image, volumes={'/evidence': evidence_volume}, memory=8192,
              timeout=86400, max_containers=1, retries=0)
def pilot_coordinator(campaign_id, manifest, prerequisite):
    from experiments.parallel_validation import CANCEL_STORE, open_durable_campaign
    root, state, checkpoint = open_durable_campaign(campaign_id, evidence_volume)
    store = modal.Dict.from_name(CANCEL_STORE, create_if_missing=True)
    try:
        verify_deployed_pilot_source(manifest)
        if (prerequisite.get('independently_revalidated') is not True
                or prerequisite.get('raw_capture_count') != 40 or prerequisite.get('fresh_unique_requests') != 214):
            raise ValueError('pilot missing independently revalidated fresh holdout prerequisite')
        execute_pilot(root, manifest, prerequisite, campaign_id,
                      lambda: verify_deployed_pilot_source(manifest), checkpoint)
        checkpoint(complete=True)
    except BaseException as exc:
        checkpoint(error=f'{type(exc).__name__}: {exc}')
    finally:
        store.put(campaign_id, True)
    return state


def durable_pilot(output, holdout_summary='', *, recover=False):
    from experiments.parallel_validation import save_dispatch, wait_durable
    destination = Path(output).resolve()
    if recover:
        dispatch = json.loads((destination/'dispatch.json').read_text())
        return wait_durable(modal.FunctionCall.from_id(dispatch['call_id']), evidence_volume, destination, dispatch)
    if destination.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError('durable pilot output must be outside the frozen worktree')
    manifest = preflight(str(destination))
    prerequisite = verify_holdout_prerequisite(holdout_summary)
    destination.mkdir(parents=True, exist_ok=False)
    campaign_id = str(uuid.uuid4())
    (destination/'intent.json').write_text(json.dumps({'campaign_id': campaign_id, 'stage': 'pilot'}))
    (destination/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (destination/'holdout-prerequisite.json').write_text(json.dumps(prerequisite, indent=2)+'\n')
    with app.run(detach=True):
        verify_local_source(manifest)
        if verify_holdout_prerequisite(holdout_summary) != prerequisite:
            raise ValueError('fresh holdout prerequisite changed before pilot dispatch')
        call = pilot_coordinator.spawn(campaign_id, manifest, prerequisite)
        dispatch = save_dispatch(destination, campaign_id, call, app.app_id)
    return wait_durable(call, evidence_volume, destination, dispatch)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--holdout-summary', default='')
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()
    durable_pilot(args.output, args.holdout_summary, recover=args.recover)
